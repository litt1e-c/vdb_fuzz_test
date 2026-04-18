#!/usr/bin/env python3
"""
Qdrant coverage-oriented fuzz scheduler.

Builds and launches a coverage-enabled Qdrant once, runs a scalar-focused
matrix plus optional targeted/vector smoke passes, then merges `.profraw`
data into LLVM text/HTML reports so the current fuzzer's server-side coverage
is visible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LAUNCHER = SCRIPT_DIR / "start_qdrant_cov.sh"
DEFAULT_RUNNER = SCRIPT_DIR / "run_qdrant_fuzz_suite.py"
DEFAULT_ORACLE_PATH = SCRIPT_DIR / "qdrant_fuzz_oracle.py"
DEFAULT_VECTOR_RUNNER = SCRIPT_DIR / "qdrant_vector_fuzz_test.py"
DEFAULT_FACET_OPERATOR = SCRIPT_DIR / "operator_test" / "qdrant" / "facet_operator.py"
DEFAULT_TEXT_INDEXED_OPERATOR = SCRIPT_DIR / "operator_test" / "qdrant" / "text_indexed_operator.py"
DEFAULT_TEXT_PROFILE_OPERATOR = SCRIPT_DIR / "operator_test" / "qdrant" / "text_profile_operator.py"
DEFAULT_FORMULA_OPERATOR = SCRIPT_DIR / "operator_test" / "qdrant" / "formula" / "formula_suite.py"
DEFAULT_SCALAR_OPERATOR = SCRIPT_DIR / "operator_test" / "qdrant" / "scalar_operator_suite.py"
DEFAULT_PERSISTENT_INDEX_OPERATOR = SCRIPT_DIR / "operator_test" / "qdrant" / "persistent_index_operator.py"
DEFAULT_IGNORE_REGEX = r"(/\.cargo/|/rustc/|/registry/src/|/cargo/registry/|/usr/share/cargo/registry/|/git/checkouts/|/\.cargo-local/)"
DEFAULT_FIGURE_SCRIPT = SCRIPT_DIR / "gen_qdrant_timeline_figure.py"
DEFAULT_EXPERIMENT_ROOT = Path(
    os.environ.get("QDRANT_EXPERIMENT_ROOT", str(Path.home() / "qdrant_artifacts"))
).expanduser().resolve()

SCALAR_COVERAGE_GROUPS: dict[str, list[str]] = {
    "scalar_target": [
        "lib/segment/src/index/field_index/",
        "lib/segment/src/payload_storage/",
        "lib/segment/src/index/query_optimization/",
        "lib/segment/src/index/struct_payload_index.rs",
        "lib/segment/src/index/plain_payload_index.rs",
        "lib/collection/src/collection/payload_index_schema.rs",
        "src/actix/api/query_api.rs",
        "src/tonic/api/query_common.rs",
        "lib/collection/src/collection/query.rs",
        "lib/collection/src/collection/state_management.rs",
        "lib/shard/src/query/",
        "src/actix/api/facet_api.rs",
        "lib/collection/src/collection/facet.rs",
        "lib/collection/src/shards/local_shard/facet.rs",
        "lib/segment/src/segment/facet.rs",
    ],
    "field_index": ["lib/segment/src/index/field_index/"],
    "payload_storage": ["lib/segment/src/payload_storage/"],
    "payload_mutation_update": [
        "src/actix/api/update_api.rs",
        "src/common/update.rs",
        "src/tonic/api/update_common.rs",
        "src/tonic/api/points_api.rs",
        "lib/shard/src/update.rs",
        "lib/shard/src/operations/payload_ops.rs",
    ],
    "query_optimization": ["lib/segment/src/index/query_optimization/"],
    "payload_index_core": [
        "lib/segment/src/index/struct_payload_index.rs",
        "lib/segment/src/index/plain_payload_index.rs",
        "lib/collection/src/collection/payload_index_schema.rs",
    ],
    "query_api_stack": [
        "src/actix/api/query_api.rs",
        "src/tonic/api/query_common.rs",
        "lib/collection/src/collection/query.rs",
        "lib/shard/src/query/",
    ],
    "facet_related": [
        "facet.rs",
        "facet_api.rs",
        "facet_index.rs",
    ],
    "full_text_index": ["lib/segment/src/index/field_index/full_text_index/"],
    "formula_rescore": [
        "lib/segment/src/index/query_optimization/rescore_formula/",
        "lib/shard/src/query/formula.rs",
    ],
    "state_recovery": [
        "lib/collection/src/collection/state_management.rs",
        "lib/segment/src/index/field_index/index_selector.rs",
        "lib/segment/src/payload_storage/mmap_payload_storage.rs",
        "lib/segment/src/payload_storage/on_disk_payload_storage.rs",
    ],
}


def resolve_experiment_layout(experiment_root: Path) -> tuple[Path, Path, Path]:
    """
    Accept either:
    1. a parent experiment root, which stores logs under `<root>/qdrant_log`, or
    2. an already-moved `qdrant_log` bundle root that directly contains
       `coverage_suite_*`, `.cov/`, and `data/`.
    """
    has_nested_qdrant_log = (experiment_root / "qdrant_log").exists()
    has_bundle_markers = experiment_root.name == "qdrant_log" or (
        not has_nested_qdrant_log
        and (
            any(experiment_root.glob("coverage_suite_*"))
            or ((experiment_root / ".cov").exists() and (experiment_root / "data").exists())
        )
    )
    if has_bundle_markers:
        log_root = experiment_root
        cov_root = experiment_root / ".cov" / "qdrant"
        data_root = experiment_root / "data" / "qdrant"
        return log_root, cov_root, data_root

    log_root = experiment_root / "qdrant_log"
    cov_root = experiment_root / ".cov" / "qdrant"
    data_root = experiment_root / "data" / "qdrant"
    return log_root, cov_root, data_root


def resolve_llvm_tool(tool_name: str, requested: str, extra_roots: list[Path] | None = None) -> str:
    if os.path.sep in requested:
        return requested
    if requested != tool_name:
        return requested

    def candidate_toolchain_roots(root: Path) -> list[Path]:
        roots: list[Path] = []
        resolved = root.expanduser().resolve()
        rustup_local = resolved / ".rustup-local" / "toolchains"
        if rustup_local.exists():
            roots.append(rustup_local)
        if resolved.name == ".rustup-local" and (resolved / "toolchains").exists():
            roots.append(resolved / "toolchains")
        return roots

    def existing_local_tools() -> list[Path]:
        candidates: list[Path] = []
        search_roots = [SCRIPT_DIR]
        if extra_roots:
            search_roots.extend(extra_roots)
        env_exp_root = os.environ.get("QDRANT_EXPERIMENT_ROOT")
        if env_exp_root:
            search_roots.append(Path(env_exp_root))
        home = os.environ.get("HOME")
        if home:
            search_roots.append(Path(home).expanduser() / "qdrant_artifacts" / "qdrant_log")

        for root in search_roots:
            for toolchain_root in candidate_toolchain_roots(root):
                candidates.extend(toolchain_root.glob(f"*/lib/rustlib/*/bin/{tool_name}"))

        rustup_home = os.environ.get("RUSTUP_HOME")
        if rustup_home:
            candidates.extend(Path(rustup_home).expanduser().glob(f"toolchains/*/lib/rustlib/*/bin/{tool_name}"))

        if home:
            candidates.extend(Path(home).expanduser().glob(f".rustup/toolchains/*/lib/rustlib/*/bin/{tool_name}"))

        return sorted({path.resolve() for path in candidates if path.exists()})

    rustc = shutil.which("rustc")
    if rustc:
        try:
            sysroot = subprocess.check_output([rustc, "--print", "sysroot"], text=True).strip()
            host = subprocess.check_output([rustc, "-vV"], text=True)
            host_line = next((line for line in host.splitlines() if line.startswith("host: ")), None)
            if host_line:
                host_triple = host_line.split(":", 1)[1].strip()
                candidate = Path(sysroot) / "lib" / "rustlib" / host_triple / "bin" / tool_name
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass

    for candidate in existing_local_tools():
        return str(candidate)

    return requested


def now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m{secs}s"
    return f"{minutes}m{secs}s"


def colorize(ok: bool, text: str) -> str:
    return f"{GREEN if ok else RED}{text}{RESET}"


def slugify(value: object, max_len: int = 96) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def write_summary_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary_json_compat(log_dir: Path, payload: dict) -> Path:
    summary_path = log_dir / "summary.json"
    legacy_summary_path = log_dir / "suite_summary.json"
    write_summary_json(summary_path, payload)
    if legacy_summary_path != summary_path:
        write_summary_json(legacy_summary_path, payload)
    return summary_path


def extract_summary_snippet(path: Path, lines: int = 12) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(content[-lines:])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_coverage_metrics_json(path: Path, coverage: dict[str, object]) -> None:
    write_summary_json(path, coverage)


def parse_percent_value(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if text.endswith("%"):
        text = text[:-1]
    try:
        return round(float(text), 2)
    except ValueError:
        return 0.0


def collect_llvm_coverage_snapshot(
    qdrant_src: Path,
    binary_path: Path,
    cov_dir: Path,
    out_dir: Path,
    ignore_regex: str,
    llvm_profdata_bin: str,
    llvm_cov_bin: str,
) -> dict[str, object]:
    summary: dict[str, object] = {"available": False}

    if shutil.which(llvm_profdata_bin) is None:
        summary["detail"] = f"`{llvm_profdata_bin}` not found; skipped coverage snapshot"
        return summary
    if shutil.which(llvm_cov_bin) is None:
        summary["detail"] = f"`{llvm_cov_bin}` not found; skipped coverage snapshot"
        return summary
    if not binary_path.exists() or not cov_dir.exists():
        summary["detail"] = "binary or coverage dir missing for snapshot"
        return summary

    profraw_files = sorted(cov_dir.rglob("*.profraw"))
    if not profraw_files:
        summary["detail"] = f"no .profraw files found under {cov_dir}"
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_profdata = out_dir / "merged.profdata"
    merge_cmd = [llvm_profdata_bin, "merge", "-sparse", *[str(path) for path in profraw_files], "-o", str(merged_profdata)]
    merge_proc = subprocess.run(merge_cmd, cwd=str(qdrant_src), capture_output=True, text=True)
    merge_output = (merge_proc.stdout or "") + (merge_proc.stderr or "")
    (out_dir / "llvm_profdata_merge.txt").write_text(merge_output, encoding="utf-8")
    if merge_proc.returncode != 0:
        summary["detail"] = merge_output.strip() or "llvm-profdata merge failed"
        summary["merge_cmd"] = " ".join(merge_cmd)
        return summary

    demangler = shutil.which("rustfilt")
    demangler_arg = [f"--Xdemangler={demangler}"] if demangler else []
    report_cmd = [
        llvm_cov_bin,
        "report",
        str(binary_path),
        f"--instr-profile={merged_profdata}",
        f"--ignore-filename-regex={ignore_regex}",
        *demangler_arg,
    ]
    summary_path = out_dir / "coverage_summary.txt"
    report_proc = subprocess.run(report_cmd, cwd=str(qdrant_src), capture_output=True, text=True)
    report_output = (report_proc.stdout or "") + (report_proc.stderr or "")
    summary_path.write_text(report_output, encoding="utf-8")

    total_line = ""
    total_regions = None
    total_missed_regions = None
    total_region_cover = None
    total_functions = None
    total_missed_functions = None
    total_function_cover = None
    total_lines = None
    total_missed_lines = None
    total_line_cover = None
    for line in reversed(report_output.splitlines()):
        if line.strip().startswith("TOTAL"):
            total_line = line.strip()
            parts = total_line.split()
            if len(parts) >= 10:
                total_regions = int(parts[1])
                total_missed_regions = int(parts[2])
                total_region_cover = parts[3]
                total_functions = int(parts[4])
                total_missed_functions = int(parts[5])
                total_function_cover = parts[6]
                total_lines = int(parts[7])
                total_missed_lines = int(parts[8])
                total_line_cover = parts[9]
            break

    summary.update(
        {
            "available": report_proc.returncode == 0,
            "detail": "coverage snapshot complete" if report_proc.returncode == 0 else (report_output.strip() or "llvm-cov report failed"),
            "profraw_count": len(profraw_files),
            "cov_dir": str(cov_dir),
            "binary_path": str(binary_path),
            "merged_profdata": str(merged_profdata),
            "summary_path": str(summary_path),
            "merge_cmd": " ".join(merge_cmd),
            "summary_cmd": " ".join(report_cmd),
            "merge_returncode": merge_proc.returncode,
            "summary_returncode": report_proc.returncode,
            "total_line": total_line or None,
            "total_line_cover": total_line_cover,
            "total_region_cover": total_region_cover,
            "total_function_cover": total_function_cover,
            "total_regions": total_regions,
            "missed_regions": total_missed_regions,
            "covered_regions": (total_regions - total_missed_regions) if total_regions is not None and total_missed_regions is not None else None,
            "total_functions": total_functions,
            "missed_functions": total_missed_functions,
            "covered_functions": (total_functions - total_missed_functions) if total_functions is not None and total_missed_functions is not None else None,
            "total_lines": total_lines,
            "missed_lines": total_missed_lines,
            "covered_lines": (total_lines - total_missed_lines) if total_lines is not None and total_missed_lines is not None else None,
        }
    )
    if report_proc.returncode == 0:
        summary["scalar_groups"] = compute_scalar_coverage_groups(summary_path)
    return summary


def coverage_timeline_fieldnames() -> list[str]:
    fields = [
        "step_index",
        "job_index",
        "event",
        "job_name",
        "job_result",
        "blocking",
        "job_elapsed_seconds",
        "suite_elapsed_seconds",
        "wall_elapsed_seconds",
        "profraw_count",
        "coverage_available",
        "overall_line_percent",
        "overall_covered_lines",
        "overall_total_lines",
        "overall_function_percent",
        "overall_region_percent",
        "summary_path",
        "snapshot_dir",
    ]
    for name in SCALAR_COVERAGE_GROUPS:
        fields.extend(
            [
                f"{name}_line_percent",
                f"{name}_covered_lines",
                f"{name}_total_lines",
            ]
        )
    return fields


def build_timeline_row(
    *,
    step_index: int,
    job_index: int,
    event: str,
    job_name: str,
    job_result: str,
    blocking: bool,
    job_elapsed_seconds: float,
    suite_elapsed_seconds: float,
    wall_elapsed_seconds: float,
    coverage: dict[str, object] | None = None,
    snapshot_dir: Path | None = None,
) -> dict[str, object]:
    coverage = coverage or {}
    row: dict[str, object] = {
        "step_index": step_index,
        "job_index": job_index,
        "event": event,
        "job_name": job_name,
        "job_result": job_result,
        "blocking": "yes" if blocking else "no",
        "job_elapsed_seconds": round(float(job_elapsed_seconds), 6),
        "suite_elapsed_seconds": round(float(suite_elapsed_seconds), 6),
        "wall_elapsed_seconds": round(float(wall_elapsed_seconds), 6),
        "profraw_count": int(coverage.get("profraw_count") or 0),
        "coverage_available": "yes" if bool(coverage.get("available")) else "no",
        "overall_line_percent": parse_percent_value(coverage.get("total_line_cover")),
        "overall_covered_lines": int(coverage.get("covered_lines") or 0),
        "overall_total_lines": int(coverage.get("total_lines") or 0),
        "overall_function_percent": parse_percent_value(coverage.get("total_function_cover")),
        "overall_region_percent": parse_percent_value(coverage.get("total_region_cover")),
        "summary_path": str(coverage.get("summary_path") or ""),
        "snapshot_dir": str(snapshot_dir or ""),
    }
    scalar_groups = coverage.get("scalar_groups", {}) if isinstance(coverage, dict) else {}
    for name in SCALAR_COVERAGE_GROUPS:
        group = scalar_groups.get(name, {}) if isinstance(scalar_groups, dict) else {}
        row[f"{name}_line_percent"] = round(float(group.get("line_percent") or 0.0), 2)
        row[f"{name}_covered_lines"] = int(group.get("covered_lines") or 0)
        row[f"{name}_total_lines"] = int(group.get("lines") or 0)
    return row


def port_in_use(host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_seconds)
        try:
            sock.connect((host, port))
        except OSError:
            return False
        return True


def port_ready(host: str, port: int, timeout_seconds: float) -> bool:
    urls = [
        f"http://{host}:{port}/readyz",
        f"http://{host}:{port}/",
        f"http://{host}:{port}/collections",
    ]
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    if resp.status < 500:
                        return True
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return True
            except urllib.error.URLError:
                pass
            except TimeoutError:
                pass
            except (ConnectionError, OSError, http.client.HTTPException):
                pass
        time.sleep(1.0)
    return False


def build_suite_command(
    python_bin: str,
    runner: Path,
    args: argparse.Namespace,
    oracle_log_dir: Path | None = None,
    run_id: str | None = None,
    log_dir: Path | None = None,
    summary_json: Path | None = None,
    case_results_csv: Path | None = None,
) -> list[str]:
    cmd = [
        python_bin,
        str(runner),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--grpc-port",
        str(args.grpc_port),
    ]
    if args.quick:
        cmd.append("--quick")
    if args.no_stress:
        cmd.append("--no-stress")
    if args.stress_only:
        cmd.append("--stress-only")
    if args.prefer_grpc:
        cmd.append("--prefer-grpc")
    if args.read_consistency != "all":
        cmd += ["--read-consistency", args.read_consistency]
    if args.write_ordering != "strong":
        cmd += ["--write-ordering", args.write_ordering]
    if args.scalar_rows is not None:
        cmd += ["--rows", str(args.scalar_rows)]
    if args.scalar_dim is not None:
        cmd += ["--dim", str(args.scalar_dim)]
    if args.scalar_batch_size is not None:
        cmd += ["--batch-size", str(args.scalar_batch_size)]
    if args.scalar_sleep_interval is not None:
        cmd += ["--sleep-interval", str(args.scalar_sleep_interval)]
    if args.scalar_distance is not None:
        cmd += ["--distance", args.scalar_distance]
    if oracle_log_dir is not None:
        cmd += ["--oracle-log-dir", str(oracle_log_dir)]
    if run_id is not None:
        cmd += ["--run-id", run_id]
    if log_dir is not None:
        cmd += ["--log-dir", str(log_dir)]
    if summary_json is not None:
        cmd += ["--summary-json", str(summary_json)]
    if case_results_csv is not None:
        cmd += ["--case-results-csv", str(case_results_csv)]
    if args.runner_args:
        cmd.extend(shlex.split(args.runner_args))
    return cmd


def run_command(command: list[str], log_path: Path, timeout_seconds: float | None = None) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            command,
            cwd=str(SCRIPT_DIR),
            stdout=fout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_code = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            fout.write(f"\n[TIMEOUT] command exceeded {timeout_seconds:.1f}s; terminating process group\n")
            fout.flush()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
            exit_code = 124
    return exit_code, time.time() - started


def analyze_suite_log(log_path: Path) -> tuple[bool, str]:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False, "suite log missing"

    if "Traceback (most recent call last)" in content:
        return False, "python traceback"
    if "❌ FAIL" in content:
        return False, "suite reported failed cases"
    if "失败详情:" in content:
        return False, "suite reported failures"
    if "🎉 全部" in content and "测试用例通过" in content:
        return True, "success marker found"
    return False, "no success marker found"


def analyze_oracle_log(log_path: Path) -> tuple[bool, str]:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False, "oracle log missing"

    if "Traceback (most recent call last)" in content:
        return False, "python traceback"
    if "CRASHED!" in content or " -> ERROR:" in content:
        return False, "oracle reported runtime errors"
    if "🚫" in content:
        return False, "oracle reported mismatches"

    success_markers = [
        "✅ 所有",
        "✅ 所有等价性测试通过。",
        "✅ PQS 测试完成。",
        "✅ GroupBy 测试完成。",
    ]
    if any(marker in content for marker in success_markers):
        return True, "success marker found"
    if "MISMATCH" in content:
        return False, "oracle reported mismatches"
    return False, "no oracle success marker found"


def analyze_vector_log(log_path: Path) -> tuple[bool, str]:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False, "vector log missing"

    if "Traceback (most recent call last)" in content:
        return False, "python traceback"
    if "failure(s) detected" in content or "❌" in content:
        return False, "vector fuzzer reported failures"
    if "🎉 All tests passed!" in content:
        return True, "success marker found"
    return False, "no vector success marker found"


def analyze_operator_log(log_path: Path) -> tuple[bool, str]:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False, "operator log missing"

    if "Traceback (most recent call last)" in content:
        return False, "python traceback"
    if "Summary: at least one" in content or ": FAIL |" in content:
        return False, "operator validation reported failures"
    if "Summary: all" in content and "checks passed" in content:
        return True, "success marker found"
    return False, "no operator success marker found"


def build_targeted_oracle_command(
    python_bin: str,
    oracle_path: Path,
    args: argparse.Namespace,
    oracle_log_dir: Path,
) -> list[str]:
    cmd = [
        python_bin,
        str(oracle_path),
        "--oracle",
        "--dynamic",
        "--rounds",
        str(args.targeted_rounds),
        "--seed",
        str(args.targeted_seed),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--grpc-port",
        str(args.grpc_port),
        "--prefer-grpc",
        "--read-consistency",
        args.read_consistency,
        "--write-ordering",
        args.write_ordering,
        "--scalar-depth-profile",
        "oracle-scalar",
        "--log-dir",
        str(oracle_log_dir),
    ]
    if args.targeted_rows is not None:
        cmd += ["-N", str(args.targeted_rows)]
    if args.targeted_dim is not None:
        cmd += ["--dim", str(args.targeted_dim)]
    if args.targeted_batch_size is not None:
        cmd += ["--batch-size", str(args.targeted_batch_size)]
    if args.targeted_sleep_interval is not None:
        cmd += ["--sleep-interval", str(args.targeted_sleep_interval)]
    if args.targeted_distance is not None:
        cmd += ["--distance", args.targeted_distance]
    if args.targeted_oracle_args:
        cmd.extend(shlex.split(args.targeted_oracle_args))
    cmd += ["--run-id", f"{args.run_id or 'qdrant-cov'}-targeted"]
    return cmd


def build_vector_command(
    python_bin: str,
    vector_runner: Path,
    args: argparse.Namespace,
    vector_log_dir: Path,
) -> list[str]:
    cmd = [
        python_bin,
        str(vector_runner),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--grpc-port",
        str(args.grpc_port),
        "--rounds",
        str(args.vector_rounds),
        "-N",
        str(args.vector_rows),
        "--dim",
        str(args.vector_dim),
        "--log-dir",
        str(vector_log_dir),
    ]
    if not args.vector_no_dynamic:
        cmd.append("--dynamic")
    if args.vector_prefer_grpc:
        cmd.append("--prefer-grpc")
    if args.vector_seed is not None:
        cmd += ["--seed", str(args.vector_seed)]
    cmd += ["--run-id", f"{args.run_id or 'qdrant-cov'}-vector"]
    if args.vector_args:
        cmd.extend(shlex.split(args.vector_args))
    return cmd


def build_operator_command(
    python_bin: str,
    operator_runner: Path,
    args: argparse.Namespace,
    raw_args: str | None = None,
    run_id: str | None = None,
) -> list[str]:
    cmd = [
        python_bin,
        str(operator_runner),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--grpc-port",
        str(args.grpc_port),
    ]
    if run_id is not None:
        cmd += ["--run-id", run_id]
    if raw_args:
        cmd.extend(shlex.split(raw_args))
    return cmd


def build_persistent_index_command(
    python_bin: str,
    operator_runner: Path,
    args: argparse.Namespace,
    *,
    phase: str,
    state_path: Path,
    raw_args: str | None = None,
    run_id: str | None = None,
) -> list[str]:
    cmd = build_operator_command(
        python_bin=python_bin,
        operator_runner=operator_runner,
        args=args,
        raw_args=raw_args,
        run_id=run_id,
    )
    cmd += [
        "--phase",
        phase,
        "--state-path",
        str(state_path),
    ]
    return cmd


def stable_hash_value(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def stable_seed(*parts: object, modulo: int = 2**31 - 1) -> int:
    value = stable_hash_value(*parts)
    if modulo > 0:
        value %= modulo
    return value or 1


def build_budget_oracle_command(
    python_bin: str,
    oracle_path: Path,
    args: argparse.Namespace,
    *,
    mode: str,
    rounds: int,
    seed: int,
    rows: int,
    dim: int,
    batch_size: int,
    sleep_interval: float,
    log_dir: Path,
    run_id: str,
    prefer_grpc: bool = False,
    read_consistency: str = "random",
    write_ordering: str = "random",
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [python_bin, str(oracle_path)]
    mode_flag = {
        "oracle": "--oracle",
        "equiv": "--equiv",
        "pqs": "--pqs",
        "group": "--group",
    }.get(mode)
    if mode_flag is None:
        raise ValueError(f"unsupported oracle mode: {mode}")
    cmd.append(mode_flag)
    if mode == "pqs":
        cmd += ["--pqs-rounds", str(rounds)]
    else:
        cmd += ["--rounds", str(rounds)]
    cmd += [
        "--seed",
        str(seed),
        "-N",
        str(rows),
        "--dim",
        str(dim),
        "--batch-size",
        str(batch_size),
        "--sleep-interval",
        str(sleep_interval),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--grpc-port",
        str(args.grpc_port),
        "--read-consistency",
        read_consistency,
        "--write-ordering",
        write_ordering,
        "--log-dir",
        str(log_dir),
        "--run-id",
        run_id,
    ]
    if prefer_grpc:
        cmd.append("--prefer-grpc")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_budget_vector_command(
    python_bin: str,
    vector_runner: Path,
    args: argparse.Namespace,
    *,
    rounds: int,
    seed: int,
    rows: int,
    dim: int,
    dynamic: bool,
    log_dir: Path,
    run_id: str,
    prefer_grpc: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        python_bin,
        str(vector_runner),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--grpc-port",
        str(args.grpc_port),
        "--rounds",
        str(rounds),
        "--seed",
        str(seed),
        "-N",
        str(rows),
        "--dim",
        str(dim),
        "--log-dir",
        str(log_dir),
        "--run-id",
        run_id,
    ]
    if dynamic:
        cmd.append("--dynamic")
    if prefer_grpc:
        cmd.append("--prefer-grpc")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def select_budget_template(
    templates: list[dict[str, object]],
    *,
    budget_seed: int,
    index: int,
    schedule: str,
) -> dict[str, object]:
    if schedule == "cycle":
        return templates[(index - 1) % len(templates)]
    if schedule == "random":
        weights: list[int] = [max(1, int(template.get("weight") or 1)) for template in templates]
        total_weight = sum(weights)
        if total_weight <= 0:
            template_index = stable_hash_value("qdrant-budget-template", budget_seed, index, len(templates)) % len(templates)
            return templates[template_index]
        pick = stable_hash_value("qdrant-budget-template", budget_seed, index, len(templates), total_weight) % total_weight
        for template, weight in zip(templates, weights):
            if pick < weight:
                return template
            pick -= weight
        return templates[-1]
    raise ValueError(f"unsupported budget schedule: {schedule}")


def materialize_budget_job_spec(
    template: dict[str, object],
    *,
    global_index: int,
    budget_seed: int,
    args: argparse.Namespace,
    python_bin: str,
    oracle_path: Path,
    vector_runner: Path,
    scalar_operator_runner: Path,
    persistent_index_runner: Path,
    facet_operator_runner: Path,
    text_indexed_operator_runner: Path,
    text_profile_operator_runner: Path,
    formula_operator_runner: Path,
    log_dir: Path,
    persistent_index_state_path: Path,
) -> dict[str, object]:
    template_type = str(template["type"])
    template_name = str(template["name"])
    label = f"{global_index:03d}_{slugify(template_name, max_len=48)}"
    run_id = f"{args.run_id or 'qdrant-cov'}-budget-{label}"
    log_path = log_dir / "budget_jobs" / f"{label}.log"
    estimated_seconds = float(template.get("estimated_seconds") or 0.0)
    timeout_seconds = float(
        template.get("timeout_seconds")
        or (max(120.0, estimated_seconds * 3.0, estimated_seconds + 120.0) if estimated_seconds > 0 else 0.0)
    )
    blocking = bool(template.get("blocking", False))
    restart_server_after = bool(template.get("restart_server_after", False))
    extra_args = list(template.get("extra_args") or [])

    if template_type == "oracle":
        seed = int(template.get("seed") or stable_seed("qdrant-budget-oracle-seed", budget_seed, template_name, global_index))
        command = build_budget_oracle_command(
            python_bin=python_bin,
            oracle_path=oracle_path,
            args=args,
            mode=str(template["mode"]),
            rounds=int(template["rounds"]),
            seed=seed,
            rows=int(template["rows"]),
            dim=int(template["dim"]),
            batch_size=int(template.get("batch_size") or 200),
            sleep_interval=float(template.get("sleep_interval") or 0.0),
            log_dir=log_dir / "oracle_logs" / "budget" / label,
            run_id=run_id,
            prefer_grpc=bool(template.get("prefer_grpc", False)),
            read_consistency=str(template.get("read_consistency") or "random"),
            write_ordering=str(template.get("write_ordering") or "random"),
            extra_args=extra_args,
        )
        analyzer = analyze_oracle_log
    elif template_type == "vector":
        seed = int(template.get("seed") or stable_seed("qdrant-budget-vector-seed", budget_seed, template_name, global_index))
        command = build_budget_vector_command(
            python_bin=python_bin,
            vector_runner=vector_runner,
            args=args,
            rounds=int(template["rounds"]),
            seed=seed,
            rows=int(template["rows"]),
            dim=int(template["dim"]),
            dynamic=bool(template.get("dynamic", False)),
            log_dir=log_dir / "vector_logs" / "budget" / label,
            run_id=run_id,
            prefer_grpc=bool(template.get("prefer_grpc", False)),
            extra_args=extra_args,
        )
        analyzer = analyze_vector_log
    elif template_type == "operator":
        runner_key = str(template["runner_key"])
        operator_runner = {
            "scalar": scalar_operator_runner,
            "facet": facet_operator_runner,
            "text_indexed": text_indexed_operator_runner,
            "text_profile": text_profile_operator_runner,
            "formula": formula_operator_runner,
        }.get(runner_key)
        if operator_runner is None:
            raise ValueError(f"unsupported operator runner_key: {runner_key}")
        command = build_operator_command(
            python_bin=python_bin,
            operator_runner=operator_runner,
            args=args,
            raw_args=shlex.join(extra_args) if extra_args else None,
            run_id=run_id,
        )
        analyzer = analyze_operator_log
    elif template_type == "persistent":
        command = build_persistent_index_command(
            python_bin=python_bin,
            operator_runner=persistent_index_runner,
            args=args,
            phase=str(template["phase"]),
            state_path=persistent_index_state_path,
            raw_args=shlex.join(extra_args) if extra_args else None,
            run_id=run_id,
        )
        analyzer = analyze_operator_log
    else:
        raise ValueError(f"unsupported budget template type: {template_type}")

    return {
        "name": template_name,
        "command": command,
        "log_path": log_path,
        "analyzer": analyzer,
        "blocking": blocking,
        "restart_server_after": restart_server_after,
        "estimated_seconds": estimated_seconds,
        "timeout_seconds": timeout_seconds or None,
    }


def build_budget_job_specs(
    *,
    args: argparse.Namespace,
    python_bin: str,
    oracle_path: Path,
    vector_runner: Path,
    scalar_operator_runner: Path,
    persistent_index_runner: Path,
    facet_operator_runner: Path,
    text_indexed_operator_runner: Path,
    text_profile_operator_runner: Path,
    formula_operator_runner: Path,
    log_dir: Path,
    persistent_index_state_path: Path,
) -> tuple[list[dict[str, object]], int]:
    run_id = args.run_id or "qdrant-cov"
    budget_seed = int(args.budget_seed) if args.budget_seed is not None else stable_seed("qdrant-budget-seed", run_id)
    profile_by_mode = {
        "oracle": "oracle-scalar",
        "pqs": "pqs-scalar",
        "equiv": "equiv-scalar",
        "group": "group-scalar",
    }
    effective_experimental_scalar_operators = (
        "robustness" if args.experimental_scalar_operators is None else str(args.experimental_scalar_operators).strip()
    )
    if not effective_experimental_scalar_operators:
        effective_experimental_scalar_operators = None

    def normalize_budget_oracle_templates(templates: list[dict[str, object]]) -> None:
        for template in templates:
            if template.get("type") != "oracle":
                continue
            if str(template.get("read_consistency") or "random") == "random":
                template["read_consistency"] = args.read_consistency
            if str(template.get("write_ordering") or "random") == "random":
                template["write_ordering"] = args.write_ordering
            extra_args = list(template.get("extra_args") or [])
            template_name = str(template.get("name") or "")
            if "int64" not in template_name:
                extra_args = [item for item in extra_args if item != "--include-known-int64-boundaries"]
            extra_args = [item for item in extra_args if item != "--evo-null-sync"]
            if "--scalar-depth-profile" not in extra_args:
                profile = profile_by_mode.get(str(template.get("mode")), "oracle-scalar")
                extra_args.extend(["--scalar-depth-profile", profile])
            if args.budget_oracle_args:
                extra_args.extend(shlex.split(args.budget_oracle_args))
            template["extra_args"] = extra_args

    preheat_templates: list[dict[str, object]] = [
        {
            "type": "oracle",
            "name": "budget-preheat-oracle-rest-core",
            "mode": "oracle",
            "rounds": 10,
            "rows": 1200,
            "dim": 128,
            "batch_size": 200,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--evo-null-sync", "--include-known-int64-boundaries"],
            "estimated_seconds": 90,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-preheat-oracle-grpc-core",
            "mode": "oracle",
            "rounds": 10,
            "rows": 1200,
            "dim": 128,
            "batch_size": 200,
            "sleep_interval": 0.0,
            "prefer_grpc": True,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--evo-null-sync", "--include-known-int64-boundaries"],
            "estimated_seconds": 90,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-preheat-oracle-rest-payload-mutations",
            "mode": "oracle",
            "rounds": 10,
            "rows": 1000,
            "dim": 128,
            "batch_size": 180,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--payload-mutations"],
            "estimated_seconds": 80,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-preheat-oracle-rest-paged",
            "mode": "oracle",
            "rounds": 8,
            "rows": 1000,
            "dim": 128,
            "batch_size": 180,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--scroll-mode", "paged"],
            "estimated_seconds": 120,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-preheat-pqs-rest-dynamic",
            "mode": "pqs",
            "rounds": 8,
            "rows": 900,
            "dim": 128,
            "batch_size": 180,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic"],
            "estimated_seconds": 110,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-preheat-group-rest-dynamic",
            "mode": "group",
            "rounds": 8,
            "rows": 1000,
            "dim": 128,
            "batch_size": 180,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic"],
            "estimated_seconds": 85,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-preheat-equiv-rest-dynamic",
            "mode": "equiv",
            "rounds": 3,
            "rows": 600,
            "dim": 128,
            "batch_size": 160,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic"],
            "estimated_seconds": 150,
            "blocking": False,
        },
    ]

    if not args.skip_vector:
        preheat_templates.extend(
            [
                {
                    "type": "vector",
                    "name": "budget-preheat-vector-rest-dynamic",
                    "rounds": 40,
                    "rows": 1300,
                    "dim": 96,
                    "dynamic": True,
                    "prefer_grpc": False,
                    "estimated_seconds": 35,
                    "blocking": False,
                },
                {
                    "type": "vector",
                    "name": "budget-preheat-vector-grpc-dynamic",
                    "rounds": 40,
                    "rows": 1300,
                    "dim": 96,
                    "dynamic": True,
                    "prefer_grpc": True,
                    "estimated_seconds": 35,
                    "blocking": False,
                },
            ]
        )

    if not args.skip_scalar_operator:
        preheat_templates.append(
            {
                "type": "operator",
                "name": "budget-preheat-scalar-operator-suite",
                "runner_key": "scalar",
                "estimated_seconds": 25,
                "blocking": True,
            }
        )
        if effective_experimental_scalar_operators:
            experimental_scalar_extra_args = ["--operators", effective_experimental_scalar_operators]
            if args.experimental_scalar_args:
                experimental_scalar_extra_args.extend(shlex.split(args.experimental_scalar_args))
            preheat_templates.append(
                {
                    "type": "operator",
                    "name": "budget-preheat-scalar-robustness",
                    "runner_key": "scalar",
                    "extra_args": experimental_scalar_extra_args,
                    "estimated_seconds": 15,
                    "blocking": bool(args.experimental_scalar_blocking),
                }
            )
    if not args.skip_facet_operator:
        preheat_templates.append(
            {
                "type": "operator",
                "name": "budget-preheat-facet-operator",
                "runner_key": "facet",
                "estimated_seconds": 8,
                "blocking": True,
            }
        )
    if not args.skip_text_indexed_operator:
        preheat_templates.append(
            {
                "type": "operator",
                "name": "budget-preheat-text-indexed-operator",
                "runner_key": "text_indexed",
                "estimated_seconds": 8,
                "blocking": True,
            }
        )
    if not args.skip_text_profile_operator:
        preheat_templates.append(
            {
                "type": "operator",
                "name": "budget-preheat-text-profile-operator",
                "runner_key": "text_profile",
                "estimated_seconds": 8,
                "blocking": True,
            }
        )
    if not args.skip_formula_operator:
        preheat_templates.append(
            {
                "type": "operator",
                "name": "budget-preheat-formula-operator",
                "runner_key": "formula",
                "estimated_seconds": 12,
                "blocking": True,
            }
        )
    if not args.skip_persistent_index:
        preheat_templates.extend(
            [
                {
                    "type": "persistent",
                    "name": "budget-preheat-persistent-index-prepare",
                    "phase": "prepare",
                    "estimated_seconds": 12,
                    "blocking": True,
                    "restart_server_after": True,
                },
                {
                    "type": "persistent",
                    "name": "budget-preheat-persistent-index-verify",
                    "phase": "verify",
                    "estimated_seconds": 12,
                    "blocking": True,
                },
            ]
        )

    fill_templates: list[dict[str, object]] = [
        {
            "type": "oracle",
            "name": "budget-fill-oracle-rest-core",
            "mode": "oracle",
            "rounds": 12,
            "rows": 1500,
            "dim": 128,
            "batch_size": 220,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--evo-null-sync", "--include-known-int64-boundaries"],
            "estimated_seconds": 120,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-oracle-grpc-core",
            "mode": "oracle",
            "rounds": 12,
            "rows": 1500,
            "dim": 128,
            "batch_size": 220,
            "sleep_interval": 0.0,
            "prefer_grpc": True,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--evo-null-sync", "--include-known-int64-boundaries"],
            "estimated_seconds": 120,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-oracle-rest-payload-mutations",
            "mode": "oracle",
            "rounds": 12,
            "rows": 1400,
            "dim": 128,
            "batch_size": 220,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--payload-mutations"],
            "estimated_seconds": 100,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-oracle-grpc-payload-mutations",
            "mode": "oracle",
            "rounds": 12,
            "rows": 1400,
            "dim": 128,
            "batch_size": 220,
            "sleep_interval": 0.0,
            "prefer_grpc": True,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--payload-mutations"],
            "estimated_seconds": 100,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-oracle-rest-paged",
            "mode": "oracle",
            "rounds": 10,
            "rows": 1200,
            "dim": 128,
            "batch_size": 200,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--scroll-mode", "paged"],
            "estimated_seconds": 140,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-pqs-rest-dynamic",
            "mode": "pqs",
            "rounds": 10,
            "rows": 1000,
            "dim": 128,
            "batch_size": 200,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic"],
            "estimated_seconds": 140,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-pqs-grpc-dynamic",
            "mode": "pqs",
            "rounds": 10,
            "rows": 1000,
            "dim": 128,
            "batch_size": 200,
            "sleep_interval": 0.0,
            "prefer_grpc": True,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic"],
            "estimated_seconds": 130,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-group-rest-dynamic",
            "mode": "group",
            "rounds": 10,
            "rows": 1100,
            "dim": 128,
            "batch_size": 200,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic"],
            "estimated_seconds": 95,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-equiv-rest-dynamic",
            "mode": "equiv",
            "rounds": 4,
            "rows": 750,
            "dim": 128,
            "batch_size": 180,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic"],
            "estimated_seconds": 190,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-oracle-rest-chaos",
            "mode": "oracle",
            "rounds": 10,
            "rows": 1000,
            "dim": 128,
            "batch_size": 200,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--dynamic", "--chaos-rate", "0.2"],
            "estimated_seconds": 95,
            "blocking": False,
        },
        {
            "type": "oracle",
            "name": "budget-fill-oracle-rest-int64",
            "mode": "oracle",
            "rounds": 10,
            "rows": 1100,
            "dim": 128,
            "batch_size": 200,
            "sleep_interval": 0.0,
            "prefer_grpc": False,
            "read_consistency": "random",
            "write_ordering": "random",
            "extra_args": ["--include-known-int64-boundaries"],
            "estimated_seconds": 80,
            "blocking": False,
        },
    ]

    if not args.skip_vector:
        fill_templates.extend(
            [
                {
                    "type": "vector",
                    "name": "budget-fill-vector-rest-dynamic",
                    "rounds": 60,
                    "rows": 1400,
                    "dim": 128,
                    "dynamic": True,
                    "prefer_grpc": False,
                    "estimated_seconds": 45,
                    "weight": 1,
                    "blocking": False,
                },
                {
                    "type": "vector",
                    "name": "budget-fill-vector-grpc-dynamic",
                    "rounds": 60,
                    "rows": 1400,
                    "dim": 128,
                    "dynamic": True,
                    "prefer_grpc": True,
                    "estimated_seconds": 45,
                    "weight": 1,
                    "blocking": False,
                },
                {
                    "type": "vector",
                    "name": "budget-fill-vector-rest-static",
                    "rounds": 48,
                    "rows": 1200,
                    "dim": 96,
                    "dynamic": False,
                    "prefer_grpc": False,
                    "estimated_seconds": 30,
                    "weight": 1,
                    "blocking": False,
                },
                {
                    "type": "vector",
                    "name": "budget-fill-vector-rest-dynamic-r150",
                    "rounds": 150,
                    "rows": 1500,
                    "dim": 128,
                    "dynamic": True,
                    "prefer_grpc": False,
                    "estimated_seconds": 95,
                    "weight": 3,
                    "blocking": False,
                },
                {
                    "type": "vector",
                    "name": "budget-fill-vector-grpc-dynamic-r150",
                    "rounds": 150,
                    "rows": 1500,
                    "dim": 128,
                    "dynamic": True,
                    "prefer_grpc": True,
                    "estimated_seconds": 95,
                    "weight": 3,
                    "blocking": False,
                },
                {
                    "type": "vector",
                    "name": "budget-fill-vector-rest-dynamic-r300",
                    "rounds": 300,
                    "rows": 1600,
                    "dim": 128,
                    "dynamic": True,
                    "prefer_grpc": False,
                    "estimated_seconds": 180,
                    "weight": 5,
                    "blocking": False,
                },
                {
                    "type": "vector",
                    "name": "budget-fill-vector-grpc-dynamic-r300",
                    "rounds": 300,
                    "rows": 1600,
                    "dim": 128,
                    "dynamic": True,
                    "prefer_grpc": True,
                    "estimated_seconds": 180,
                    "weight": 5,
                    "blocking": False,
                },
                {
                    "type": "vector",
                    "name": "budget-fill-vector-rest-dynamic-r500",
                    "rounds": 500,
                    "rows": 1600,
                    "dim": 128,
                    "dynamic": True,
                    "prefer_grpc": False,
                    "estimated_seconds": 300,
                    "weight": 6,
                    "blocking": False,
                },
                {
                    "type": "vector",
                    "name": "budget-fill-vector-grpc-dynamic-r500",
                    "rounds": 500,
                    "rows": 1600,
                    "dim": 128,
                    "dynamic": True,
                    "prefer_grpc": True,
                    "estimated_seconds": 300,
                    "weight": 6,
                    "blocking": False,
                },
            ]
        )

    normalize_budget_oracle_templates(preheat_templates)
    normalize_budget_oracle_templates(fill_templates)

    job_specs: list[dict[str, object]] = []
    for index, template in enumerate(preheat_templates, start=1):
        job_specs.append(
            materialize_budget_job_spec(
                template,
                global_index=index,
                budget_seed=budget_seed,
                args=args,
                python_bin=python_bin,
                oracle_path=oracle_path,
                vector_runner=vector_runner,
                scalar_operator_runner=scalar_operator_runner,
                persistent_index_runner=persistent_index_runner,
                facet_operator_runner=facet_operator_runner,
                text_indexed_operator_runner=text_indexed_operator_runner,
                text_profile_operator_runner=text_profile_operator_runner,
                formula_operator_runner=formula_operator_runner,
                log_dir=log_dir,
                persistent_index_state_path=persistent_index_state_path,
            )
        )

    fill_index = 1
    planned_fill_jobs = int(
        math.ceil(
            (float(args.time_budget_seconds) / max(0.5, float(args.budget_min_fill_job_seconds)))
            * float(args.budget_fill_overprovision)
        )
    )
    min_fill_jobs = max(len(fill_templates) * 2, planned_fill_jobs)
    max_planned_jobs = max(len(preheat_templates) + 1, int(args.budget_max_planned_jobs))
    target_total_jobs = min(max_planned_jobs, len(preheat_templates) + max(1, min_fill_jobs))
    while len(job_specs) < target_total_jobs:
        template = select_budget_template(
            fill_templates,
            budget_seed=budget_seed,
            index=fill_index,
            schedule=args.budget_schedule,
        )
        job_specs.append(
            materialize_budget_job_spec(
                template,
                global_index=len(job_specs) + 1,
                budget_seed=budget_seed,
                args=args,
                python_bin=python_bin,
                oracle_path=oracle_path,
                vector_runner=vector_runner,
                scalar_operator_runner=scalar_operator_runner,
                persistent_index_runner=persistent_index_runner,
                facet_operator_runner=facet_operator_runner,
                text_indexed_operator_runner=text_indexed_operator_runner,
                text_profile_operator_runner=text_profile_operator_runner,
                formula_operator_runner=formula_operator_runner,
                log_dir=log_dir,
                persistent_index_state_path=persistent_index_state_path,
            )
        )
        fill_index += 1

    return job_specs, budget_seed


def resolve_binary_path(args: argparse.Namespace, qdrant_src: Path) -> Path:
    if args.binary:
        return Path(args.binary).expanduser().resolve()
    target_dir = (
        Path(args.qdrant_target_dir).expanduser().resolve()
        if args.qdrant_target_dir
        else (SCRIPT_DIR / ".cov" / "qdrant" / "target").resolve()
    )
    return target_dir / args.cargo_profile / "qdrant"


def collect_llvm_coverage(
    qdrant_src: Path,
    binary_path: Path,
    cov_dir: Path,
    out_dir: Path,
    ignore_regex: str,
    low_threshold: int,
    llvm_profdata_bin: str,
    llvm_cov_bin: str,
) -> dict:
    summary: dict[str, object] = {"available": False}

    if shutil.which(llvm_profdata_bin) is None:
        summary["detail"] = f"`{llvm_profdata_bin}` not found; skipped coverage merge"
        return summary
    if shutil.which(llvm_cov_bin) is None:
        summary["detail"] = f"`{llvm_cov_bin}` not found; skipped coverage report"
        return summary
    if not binary_path.exists():
        summary["detail"] = f"instrumented binary not found: {binary_path}"
        return summary
    if not cov_dir.exists():
        summary["detail"] = f"coverage dir not found: {cov_dir}"
        return summary

    profraw_files = sorted(cov_dir.rglob("*.profraw"))
    if not profraw_files:
        summary["detail"] = f"no .profraw files found under {cov_dir}"
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_profdata = out_dir / "merged.profdata"
    merge_cmd = [llvm_profdata_bin, "merge", "-sparse", *[str(path) for path in profraw_files], "-o", str(merged_profdata)]
    merge_proc = subprocess.run(merge_cmd, cwd=str(qdrant_src), capture_output=True, text=True)
    merge_output = (merge_proc.stdout or "") + (merge_proc.stderr or "")
    (out_dir / "llvm_profdata_merge.txt").write_text(merge_output, encoding="utf-8")
    if merge_proc.returncode != 0:
        summary["detail"] = merge_output.strip() or "llvm-profdata merge failed"
        summary["merge_cmd"] = " ".join(merge_cmd)
        return summary

    demangler = shutil.which("rustfilt")
    demangler_arg = [f"--Xdemangler={demangler}"] if demangler else []
    source_files: list[str] = []
    for root_name in ("src", "lib"):
        root_path = qdrant_src / root_name
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.rs"):
            if "target" in path.parts:
                continue
            source_files.append(path.relative_to(qdrant_src).as_posix())
    source_args = [f"--sources={path}" for path in sorted(set(source_files))]
    report_base = [
        llvm_cov_bin,
        "report",
        str(binary_path),
        f"--instr-profile={merged_profdata}",
        f"--ignore-filename-regex={ignore_regex}",
        *demangler_arg,
    ]
    show_base = [
        llvm_cov_bin,
        "show",
        str(binary_path),
        f"--instr-profile={merged_profdata}",
        "--format=html",
        f"--output-dir={out_dir / 'html'}",
        "--project-title=Qdrant fuzz coverage",
        "--show-line-counts-or-regions",
        f"--ignore-filename-regex={ignore_regex}",
        *demangler_arg,
    ]

    summary_path = out_dir / "coverage_summary.txt"
    functions_path = out_dir / "coverage_functions.txt"
    low_functions_path = out_dir / f"coverage_functions.lt{low_threshold}.txt"

    summary_proc = subprocess.run(report_base, cwd=str(qdrant_src), capture_output=True, text=True)
    summary_output = (summary_proc.stdout or "") + (summary_proc.stderr or "")
    summary_path.write_text(summary_output, encoding="utf-8")

    functions_output = ""
    low_functions_output = ""
    functions_cmd: list[str] | None = None
    low_functions_cmd: list[str] | None = None
    functions_returncode: int | None = None
    low_functions_returncode: int | None = None
    if source_args:
        functions_cmd = [*report_base, "--show-functions", *source_args]
        functions_proc = subprocess.run(functions_cmd, cwd=str(qdrant_src), capture_output=True, text=True)
        functions_output = (functions_proc.stdout or "") + (functions_proc.stderr or "")
        functions_returncode = functions_proc.returncode

        low_functions_cmd = [*report_base, "--show-functions", f"--line-coverage-lt={low_threshold}", *source_args]
        low_functions_proc = subprocess.run(low_functions_cmd, cwd=str(qdrant_src), capture_output=True, text=True)
        low_functions_output = (low_functions_proc.stdout or "") + (low_functions_proc.stderr or "")
        low_functions_returncode = low_functions_proc.returncode
    else:
        functions_output = "no Rust source files discovered under src/ or lib/; skipped function coverage report\n"
        low_functions_output = "no Rust source files discovered under src/ or lib/; skipped low-function coverage report\n"
    functions_path.write_text(functions_output, encoding="utf-8")
    low_functions_path.write_text(low_functions_output, encoding="utf-8")

    html_proc = subprocess.run(show_base, cwd=str(qdrant_src), capture_output=True, text=True)
    html_output = (html_proc.stdout or "") + (html_proc.stderr or "")
    (out_dir / "llvm_cov_show.txt").write_text(html_output, encoding="utf-8")

    total_line = ""
    total_region_cover = None
    total_function_cover = None
    total_line_cover = None
    total_regions = None
    total_missed_regions = None
    total_functions = None
    total_missed_functions = None
    total_lines = None
    total_missed_lines = None
    for line in reversed(summary_output.splitlines()):
        if line.strip().startswith("TOTAL"):
            total_line = line.strip()
            parts = total_line.split()
            if len(parts) >= 10:
                total_regions = int(parts[1])
                total_missed_regions = int(parts[2])
                total_region_cover = parts[3]
                total_functions = int(parts[4])
                total_missed_functions = int(parts[5])
                total_function_cover = parts[6]
                total_lines = int(parts[7])
                total_missed_lines = int(parts[8])
                total_line_cover = parts[9]
            break

    available = summary_proc.returncode == 0
    detail = "coverage export complete"
    optional_failures: list[str] = []
    if html_proc.returncode != 0:
        optional_failures.append("html")
    if functions_returncode not in (None, 0):
        optional_failures.append("functions")
    if low_functions_returncode not in (None, 0):
        optional_failures.append("low-functions")
    if optional_failures:
        detail = f"coverage summary complete; optional steps failed: {', '.join(optional_failures)}"
    summary.update(
        {
            "available": available,
            "detail": detail,
            "profraw_count": len(profraw_files),
            "cov_dir": str(cov_dir),
            "binary_path": str(binary_path),
            "merged_profdata": str(merged_profdata),
            "summary_path": str(summary_path),
            "functions_path": str(functions_path),
            "low_functions_path": str(low_functions_path),
            "html_dir": str(out_dir / "html"),
            "source_file_count": len(source_args),
            "ignore_regex": ignore_regex,
            "low_threshold": low_threshold,
            "merge_cmd": " ".join(merge_cmd),
            "summary_cmd": " ".join(report_base),
            "functions_cmd": " ".join(functions_cmd) if functions_cmd else None,
            "low_functions_cmd": " ".join(low_functions_cmd) if low_functions_cmd else None,
            "html_cmd": " ".join(show_base),
            "merge_returncode": merge_proc.returncode,
            "summary_returncode": summary_proc.returncode,
            "functions_returncode": functions_returncode,
            "low_functions_returncode": low_functions_returncode,
            "html_returncode": html_proc.returncode,
            "total_region_cover": total_region_cover,
            "total_function_cover": total_function_cover,
            "total_line": total_line or None,
            "total_line_cover": total_line_cover,
            "total_regions": total_regions,
            "missed_regions": total_missed_regions,
            "covered_regions": (total_regions - total_missed_regions) if total_regions is not None and total_missed_regions is not None else None,
            "total_functions": total_functions,
            "missed_functions": total_missed_functions,
            "covered_functions": (total_functions - total_missed_functions) if total_functions is not None and total_missed_functions is not None else None,
            "total_lines": total_lines,
            "missed_lines": total_missed_lines,
            "covered_lines": (total_lines - total_missed_lines) if total_lines is not None and total_missed_lines is not None else None,
        }
    )
    if not available and summary_proc.returncode != 0:
        summary["detail"] = summary_output.strip() or "llvm-cov report failed"
    return summary


def parse_coverage_summary_rows(summary_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return rows

    for line in lines:
        if not line or line.startswith("Filename") or line.startswith("-") or line.startswith("TOTAL"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            rows.append(
                {
                    "filename": parts[0],
                    "regions": int(parts[1]),
                    "missed_regions": int(parts[2]),
                    "functions": int(parts[4]),
                    "missed_functions": int(parts[5]),
                    "lines": int(parts[7]),
                    "missed_lines": int(parts[8]),
                }
            )
        except ValueError:
            continue
    return rows


def summarize_coverage_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "file_count": 0,
            "regions": 0,
            "missed_regions": 0,
            "covered_regions": 0,
            "functions": 0,
            "missed_functions": 0,
            "covered_functions": 0,
            "lines": 0,
            "missed_lines": 0,
            "covered_lines": 0,
            "line_percent": 0.0,
            "function_percent": 0.0,
            "region_percent": 0.0,
        }

    regions = sum(int(row["regions"]) for row in rows)
    missed_regions = sum(int(row["missed_regions"]) for row in rows)
    functions = sum(int(row["functions"]) for row in rows)
    missed_functions = sum(int(row["missed_functions"]) for row in rows)
    lines = sum(int(row["lines"]) for row in rows)
    missed_lines = sum(int(row["missed_lines"]) for row in rows)

    return {
        "file_count": len(rows),
        "regions": regions,
        "missed_regions": missed_regions,
        "covered_regions": regions - missed_regions,
        "functions": functions,
        "missed_functions": missed_functions,
        "covered_functions": functions - missed_functions,
        "lines": lines,
        "missed_lines": missed_lines,
        "covered_lines": lines - missed_lines,
        "region_percent": round(((regions - missed_regions) / regions * 100.0) if regions else 0.0, 2),
        "function_percent": round(((functions - missed_functions) / functions * 100.0) if functions else 0.0, 2),
        "line_percent": round(((lines - missed_lines) / lines * 100.0) if lines else 0.0, 2),
    }


def compute_scalar_coverage_groups(summary_path: Path) -> dict[str, dict[str, object]]:
    rows = parse_coverage_summary_rows(summary_path)
    groups: dict[str, dict[str, object]] = {}
    for name, patterns in SCALAR_COVERAGE_GROUPS.items():
        matched = [row for row in rows if any(pattern in str(row["filename"]) for pattern in patterns)]
        groups[name] = summarize_coverage_rows(matched)
    return groups


def job_result_label(job: dict[str, object]) -> str:
    if bool(job.get("passed")):
        return "PASS"
    if bool(job.get("blocking", True)):
        return "FAIL"
    return "WARN"


def job_status_text(job: dict[str, object]) -> str:
    label = job_result_label(job)
    if label == "PASS":
        return colorize(True, label)
    if label == "FAIL":
        return colorize(False, label)
    return f"{YELLOW}{label}{RESET}"


def build_case_results_rows(job_results: list[dict[str, object]], scalar_suite_csv: Path | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job in job_results:
        rows.append(
            {
                "row_type": "job",
                "name": job["name"],
                "seed": "",
                "actual_seed": "",
                "result": job_result_label(job),
                "elapsed_seconds": f"{float(job['elapsed_seconds']):.3f}",
                "detail": job["detail"],
                "log_path": job["log_path"],
                "command": shlex.join(list(job["command"])),
                "reproduce_command": shlex.join(list(job["command"])),
                "blocking": "yes" if bool(job.get("blocking", True)) else "no",
            }
        )

    if scalar_suite_csv and scalar_suite_csv.exists():
        with scalar_suite_csv.open("r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                rows.append(
                    {
                        "row_type": "scalar-suite-case",
                        "name": row.get("group", ""),
                        "seed": row.get("seed", ""),
                        "actual_seed": row.get("actual_seed", ""),
                        "result": row.get("result", ""),
                        "elapsed_seconds": row.get("elapsed_seconds", ""),
                        "detail": row.get("detail", ""),
                        "log_path": row.get("log_path", ""),
                        "command": row.get("command", ""),
                        "reproduce_command": row.get("reproduce_command", ""),
                        "blocking": row.get("blocking", ""),
                    }
                )
    return rows


def expected_figure_outputs(output_dir: Path, basename: str) -> list[Path]:
    return [
        output_dir / f"{basename}.tex",
        output_dir / f"{basename}.data.csv",
        output_dir / f"{basename}.pdf",
        output_dir / f"{basename}.svg",
        output_dir / f"{basename}.png",
    ]


def generate_qdrant_figure_artifacts(
    *,
    python_bin: str,
    figure_script: Path,
    summary_json_path: Path,
    timeline_csv_path: Path | None,
    output_dir: Path,
    manifest_path: Path,
    log_path: Path,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "enabled": True,
        "ok": False,
        "script": str(figure_script),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
        "figures": [],
        "failed": [],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not figure_script.exists():
        manifest["detail"] = f"figure script not found: {figure_script}"
        write_summary_json(manifest_path, manifest)
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    figure_specs: list[dict[str, object]] = []
    if timeline_csv_path and timeline_csv_path.exists():
        figure_specs.extend(
            [
                {
                    "name": "scalar-target-timeline",
                    "basename": "fig_qdrant_scalar_timeline",
                    "command": [
                        python_bin,
                        str(figure_script),
                        "timeline",
                        "--timeline-csv",
                        str(timeline_csv_path),
                        "--metric-group",
                        "scalar_target",
                        "--prepend-origin",
                        "--x-field",
                        "auto",
                        "--output-dir",
                        str(output_dir),
                        "--basename",
                        "fig_qdrant_scalar_timeline",
                    ],
                },
                {
                    "name": "overall-timeline",
                    "basename": "fig_qdrant_overall_timeline",
                    "command": [
                        python_bin,
                        str(figure_script),
                        "timeline",
                        "--timeline-csv",
                        str(timeline_csv_path),
                        "--metric-group",
                        "overall",
                        "--prepend-origin",
                        "--x-field",
                        "auto",
                        "--output-dir",
                        str(output_dir),
                        "--basename",
                        "fig_qdrant_overall_timeline",
                    ],
                },
            ]
        )
    if summary_json_path.exists():
        figure_specs.append(
            {
                "name": "scalar-subsystem-summary",
                "basename": "fig_qdrant_scalar_subsystem_summary",
                "command": [
                    python_bin,
                    str(figure_script),
                    "summary-groups",
                    "--summary-json",
                    str(summary_json_path),
                    "--output-dir",
                    str(output_dir),
                    "--basename",
                    "fig_qdrant_scalar_subsystem_summary",
                ],
            }
        )

    if not figure_specs:
        manifest["detail"] = "no timeline CSV or summary JSON available for figure generation"
        write_summary_json(manifest_path, manifest)
        return manifest

    with log_path.open("w", encoding="utf-8") as flog:
        for spec in figure_specs:
            command = list(spec["command"])
            basename = str(spec["basename"])
            expected_outputs = expected_figure_outputs(output_dir, basename)
            flog.write(f"$ {shlex.join(command)}\n")
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.stdout:
                flog.write(proc.stdout)
                if not proc.stdout.endswith("\n"):
                    flog.write("\n")
            if proc.stderr:
                flog.write(proc.stderr)
                if not proc.stderr.endswith("\n"):
                    flog.write("\n")
            flog.write(f"[returncode] {proc.returncode}\n\n")
            entry = {
                "name": spec["name"],
                "basename": basename,
                "command": shlex.join(command),
                "returncode": proc.returncode,
                "expected_outputs": [str(path) for path in expected_outputs],
                "outputs": [str(path) for path in expected_outputs if path.exists()],
            }
            manifest["figures"].append(entry)
            if proc.returncode != 0:
                manifest["failed"].append(entry)

    manifest["ok"] = bool(manifest["figures"]) and not bool(manifest["failed"])
    manifest["detail"] = (
        "figure generation complete"
        if manifest["ok"]
        else "figure generation completed with failures; inspect log_path"
    )
    write_summary_json(manifest_path, manifest)
    return manifest


def write_summary_markdown(path: Path, payload: dict) -> None:
    coverage = payload.get("coverage", {})
    jobs = payload.get("jobs", [])
    scalar_groups = coverage.get("scalar_groups", {})
    blocking_failures = payload.get("blocking_failures", [])
    non_blocking_failures = payload.get("non_blocking_failures", [])
    figure_artifacts = payload.get("figure_artifacts", {})

    lines = [
        "# Qdrant Coverage Experiment Summary",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Suite: `{'PASS' if payload.get('suite_passed') else 'FAIL'}`",
        f"- Host: `{payload.get('host')}:{payload.get('port')}` (grpc: `{payload.get('grpc_port')}`)",
        f"- Qdrant source: `{payload.get('qdrant_src')}`",
        f"- Suite time: `{fmt_duration(float(payload.get('suite_elapsed_seconds', 0.0)))}`",
        f"- Total time: `{fmt_duration(float(payload.get('total_elapsed_seconds', 0.0)))}`",
    ]
    if payload.get("time_budget_seconds") is not None:
        lines.append(f"- Time budget: `{fmt_duration(float(payload.get('time_budget_seconds', 0.0)))}`")
        lines.append(f"- Budget schedule: `{payload.get('budget_schedule', 'random')}`")
        if payload.get("budget_seed") is not None:
            lines.append(f"- Budget seed: `{payload.get('budget_seed')}`")
    if blocking_failures:
        lines.append(f"- Blocking failures: `{', '.join(blocking_failures)}`")
    if non_blocking_failures:
        lines.append(f"- Non-blocking failures: `{', '.join(non_blocking_failures)}`")
    if payload.get("coverage_timeline_csv"):
        lines.append(f"- Coverage timeline: `{payload['coverage_timeline_csv']}`")
    if payload.get("coverage_metrics_json"):
        lines.append(f"- Coverage metrics: `{payload['coverage_metrics_json']}`")
    if isinstance(figure_artifacts, dict) and figure_artifacts.get("manifest_path"):
        lines.append(f"- Figure manifest: `{figure_artifacts['manifest_path']}`")
    if isinstance(figure_artifacts, dict) and figure_artifacts.get("output_dir"):
        lines.append(f"- Figure output dir: `{figure_artifacts['output_dir']}`")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Overall line coverage: `{coverage.get('total_line_cover')}`",
            f"- Overall covered lines: `{coverage.get('covered_lines', 0)}` / `{coverage.get('total_lines', 0)}`",
            f"- Overall function coverage: `{coverage.get('total_function_cover')}`",
            f"- Overall region coverage: `{coverage.get('total_region_cover')}`",
        ]
    )

    scalar_target = scalar_groups.get("scalar_target")
    if scalar_target:
        lines.extend(
            [
                f"- Scalar target line coverage: `{scalar_target.get('line_percent', 0.0):.2f}%`",
                f"- Scalar target covered lines: `{scalar_target.get('covered_lines', 0)}` / `{scalar_target.get('lines', 0)}`",
                f"- Scalar target function coverage: `{scalar_target.get('function_percent', 0.0):.2f}%`",
                f"- Scalar target region coverage: `{scalar_target.get('region_percent', 0.0):.2f}%`",
                "",
                "## Scalar Subsystems",
                "",
                "| Group | Covered Lines | Line | Function | Region | Files |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, group in scalar_groups.items():
            lines.append(
                f"| `{name}` | {group.get('covered_lines', 0)} / {group.get('lines', 0)} | "
                f"{group.get('line_percent', 0.0):.2f}% | "
                f"{group.get('function_percent', 0.0):.2f}% | {group.get('region_percent', 0.0):.2f}% | "
                f"{group.get('file_count', 0)} |"
            )

    if isinstance(figure_artifacts, dict) and figure_artifacts.get("figures"):
        lines.extend(
            [
                "",
                "## Figures",
                "",
                "| Figure | Status | Outputs |",
                "| --- | --- | --- |",
            ]
        )
        for figure in figure_artifacts.get("figures", []):
            outputs = figure.get("outputs", [])
            output_text = "<br>".join(f"`{item}`" for item in outputs) if outputs else ""
            status = "PASS" if int(figure.get("returncode", 1)) == 0 else "FAIL"
            lines.append(f"| `{figure.get('name')}` | `{status}` | {output_text} |")

    lines.extend(
        [
            "",
            "## Jobs",
            "",
            "| Job | Result | Blocking | Time | Log |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for job in jobs:
        lines.append(
            f"| `{job['name']}` | `{job_result_label(job)}` | "
            f"`{'yes' if bool(job.get('blocking', True)) else 'no'}` | "
            f"{fmt_duration(float(job['elapsed_seconds']))} | `{job['log_path']}` |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def start_server(
    launcher: Path,
    run_id: str,
    server_log_path: Path,
    host: str,
    port: int,
    ready_timeout: float,
    env: dict[str, str],
    append_log: bool = False,
) -> tuple[subprocess.Popen, object]:
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    server_log_handle = server_log_path.open("a" if append_log else "w", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", str(launcher), run_id],
        cwd=str(SCRIPT_DIR),
        stdout=server_log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if port_ready(host, port, timeout_seconds=1.5):
            return process, server_log_handle
        if process.poll() is not None:
            server_log_handle.flush()
            server_log_handle.close()
            snippet = extract_summary_snippet(server_log_path, lines=20)
            raise RuntimeError(
                f"Qdrant exited before becoming ready on {host}:{port}. "
                f"Inspect {server_log_path}.\n{snippet}"
            )
        time.sleep(0.5)
    stop_server(process)
    server_log_handle.close()
    raise RuntimeError(
        f"Qdrant did not become ready on {host}:{port} within {ready_timeout:.0f}s. "
        f"Inspect {server_log_path}."
    )


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def can_launch_job(
    job_specs: list[dict[str, object]],
    candidate_job_idx: int,
    *,
    deadline: float | None,
    now: float,
    launch_guard_seconds: float = 8.0,
) -> tuple[bool, str | None]:
    if candidate_job_idx > len(job_specs):
        return False, "all planned jobs completed"
    if deadline is None:
        return True, None

    remaining_seconds = deadline - now
    if remaining_seconds <= 0:
        return False, f"Time budget reached; stopping after completed job {candidate_job_idx - 1}."

    estimated_seconds = float(job_specs[candidate_job_idx - 1].get("estimated_seconds") or 0.0)
    guard_seconds = max(0.0, float(launch_guard_seconds))
    launch_threshold = max(guard_seconds, min(20.0, estimated_seconds * 0.1))
    if estimated_seconds > 0 and remaining_seconds < launch_threshold:
        return (
            False,
            f"Budget remaining {fmt_duration(remaining_seconds)} is below "
            f"launch threshold for {job_specs[candidate_job_idx - 1].get('name')} "
            f"(~{fmt_duration(estimated_seconds)}); stopping.",
        )

    return True, None


def print_report_summary(log_dir: Path, summary_path: Path, payload: dict) -> int:
    suite_ok = payload.get("suite_passed", False)
    cov_summary = payload.get("coverage", {})
    jobs = payload.get("jobs", [])
    blocking_failures = payload.get("blocking_failures", [])
    non_blocking_failures = payload.get("non_blocking_failures", [])
    figure_artifacts = payload.get("figure_artifacts", {})
    print(f"\n{BOLD}{'=' * 76}{RESET}")
    print(f"{BOLD}Qdrant Coverage Summary{RESET}")
    print(f"{'=' * 76}")
    print(f"  Logs:        {log_dir}")
    print(f"  Summary:     {summary_path}")
    if payload.get("summary_md"):
        print(f"  Summary MD:  {payload['summary_md']}")
    if payload.get("case_results_csv"):
        print(f"  Case CSV:    {payload['case_results_csv']}")
    if payload.get("coverage_timeline_csv"):
        print(f"  Timeline:    {payload['coverage_timeline_csv']}")
    if payload.get("coverage_metrics_json"):
        print(f"  Cov JSON:    {payload['coverage_metrics_json']}")
    if isinstance(figure_artifacts, dict) and figure_artifacts.get("manifest_path"):
        print(f"  Figures:     {figure_artifacts['manifest_path']}")
    if isinstance(figure_artifacts, dict) and figure_artifacts.get("output_dir"):
        print(f"  Figure dir:  {figure_artifacts['output_dir']}")
    if "suite_elapsed_seconds" in payload:
        print(f"  Suite time:  {fmt_duration(float(payload['suite_elapsed_seconds']))}")
    if payload.get("time_budget_seconds") is not None:
        print(f"  Budget:      {fmt_duration(float(payload['time_budget_seconds']))}")
        print(f"  Schedule:    {payload.get('budget_schedule', 'random')}")
        if payload.get("budget_seed") is not None:
            print(f"  Budget seed: {payload.get('budget_seed')}")
    print(f"  Suite:       {colorize(bool(suite_ok), 'PASS' if suite_ok else 'FAIL')}")
    if blocking_failures:
        print(f"  Blocking:    {', '.join(blocking_failures)}")
    if non_blocking_failures:
        print(f"  Warnings:    {', '.join(non_blocking_failures)}")
    if payload.get("suite_log"):
        print(f"  Suite log:   {payload['suite_log']}")
    if payload.get("server_log"):
        print(f"  Server log:  {payload['server_log']}")
    if jobs:
        for job in jobs:
            blocking_tag = "blocking" if bool(job.get("blocking", True)) else "non-blocking"
            print(
                f"  Job:         {job.get('name')} -> {job_status_text(job)} "
                f"[{blocking_tag}] ({fmt_duration(float(job.get('elapsed_seconds', 0.0)))})"
            )
            print(f"               {job.get('log_path')}")
    if cov_summary.get("available"):
        print(f"  Cov dir:     {cov_summary.get('cov_dir')}")
        print(f"  Profraw:     {cov_summary.get('profraw_count')}")
        print(f"  Binary:      {cov_summary.get('binary_path')}")
        if cov_summary.get("total_line_cover"):
            print(f"  Line cov:    {cov_summary.get('total_line_cover')}")
        if cov_summary.get("total_function_cover"):
            print(f"  Func cov:    {cov_summary.get('total_function_cover')}")
        if cov_summary.get("total_region_cover"):
            print(f"  Region cov:  {cov_summary.get('total_region_cover')}")
        scalar_groups = cov_summary.get("scalar_groups", {})
        scalar_target = scalar_groups.get("scalar_target")
        if scalar_target:
            print(f"  Scalar line: {scalar_target.get('line_percent', 0.0):.2f}%")
            print(f"  Scalar func: {scalar_target.get('function_percent', 0.0):.2f}%")
            print(f"  Scalar reg:  {scalar_target.get('region_percent', 0.0):.2f}%")
        if cov_summary.get("total_line"):
            print(f"  Total:       {cov_summary.get('total_line')}")
        print(f"  Summary txt: {cov_summary.get('summary_path')}")
        print(f"  Functions:   {cov_summary.get('functions_path')}")
        print(f"  Low funcs:   {cov_summary.get('low_functions_path')}")
        print(f"  HTML dir:    {cov_summary.get('html_dir')}")
    else:
        print(f"  Coverage:    {RED}unavailable{RESET} ({cov_summary.get('detail', 'unknown reason')})")

    if not suite_ok:
        return 1
    if not cov_summary.get("available"):
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the current Qdrant fuzz suite against a coverage-enabled server")
    parser.add_argument("--qdrant-src", default=os.environ.get("QDRANT_SRC"), help="Local qdrant source checkout path")
    parser.add_argument(
        "--experiment-root",
        default=os.environ.get("QDRANT_EXPERIMENT_ROOT", str(DEFAULT_EXPERIMENT_ROOT)),
        help="Artifact root for logs/.cov/data; accepts either a parent experiment dir or an existing qdrant_log bundle root",
    )
    parser.add_argument("--launcher", default=str(DEFAULT_LAUNCHER), help="Coverage launcher script path")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER), help="Scalar fuzz suite runner path")
    parser.add_argument("--oracle-path", default=str(DEFAULT_ORACLE_PATH), help="Direct qdrant_fuzz_oracle.py path for targeted scalar smoke")
    parser.add_argument("--vector-runner", default=str(DEFAULT_VECTOR_RUNNER), help="Vector fuzz runner path")
    parser.add_argument("--scalar-operator-runner", default=str(DEFAULT_SCALAR_OPERATOR), help="Scalar/filter operator suite runner path")
    parser.add_argument("--persistent-index-runner", default=str(DEFAULT_PERSISTENT_INDEX_OPERATOR), help="Persistent payload-index restart validator path")
    parser.add_argument("--text-profile-operator-runner", default=str(DEFAULT_TEXT_PROFILE_OPERATOR), help="Path to text profile validator")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for the fuzz suite")
    parser.add_argument("--host", default="127.0.0.1", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant REST port")
    parser.add_argument("--grpc-port", type=int, default=6334, help="Qdrant gRPC port")
    parser.add_argument("--run-id", default=None, help="Run id used for log/.cov directories")
    parser.add_argument("--cov-dir", default=None, help="Explicit .profraw directory")
    parser.add_argument("--qdrant-target-dir", default=None, help="Instrumented cargo target dir")
    parser.add_argument("--binary", default=None, help="Explicit instrumented qdrant binary path")
    parser.add_argument("--cargo-profile", choices=["debug", "release"], default="debug", help="Cargo profile used by the launcher")
    parser.add_argument("--qdrant-config-path", default=None, help="Optional qdrant config path")
    parser.add_argument("--ready-timeout", type=float, default=1800.0, help="Seconds to wait for the first instrumented build + server readiness")
    parser.add_argument("--quick", action="store_true", help="Pass --quick to run_qdrant_fuzz_suite.py")
    parser.add_argument("--stress-only", action="store_true", help="Pass --stress-only to run_qdrant_fuzz_suite.py")
    parser.add_argument("--no-stress", action="store_true", help="Pass --no-stress to run_qdrant_fuzz_suite.py")
    parser.add_argument("--prefer-grpc", action="store_true", help="Pass --prefer-grpc to the scalar suite runner")
    parser.add_argument(
        "--read-consistency",
        choices=["all", "majority", "quorum", "1", "random"],
        default="all",
        help="Pass-through read consistency for qdrant_fuzz_oracle.py",
    )
    parser.add_argument(
        "--write-ordering",
        choices=["weak", "medium", "strong", "random"],
        default="strong",
        help="Pass-through write ordering for qdrant_fuzz_oracle.py",
    )
    parser.add_argument("--scalar-rows", type=int, default=None, help="Initial rows passed to the scalar suite")
    parser.add_argument("--scalar-dim", type=int, default=None, help="Vector dimension passed to the scalar suite")
    parser.add_argument("--scalar-batch-size", type=int, default=None, help="Batch size passed to the scalar suite")
    parser.add_argument("--scalar-sleep-interval", type=float, default=None, help="Insert sleep interval passed to the scalar suite")
    parser.add_argument(
        "--scalar-distance",
        choices=["random", "euclid", "cosine", "dot", "manhattan"],
        default=None,
        help="Distance mode passed to the scalar suite",
    )
    parser.add_argument("--runner-args", default=None, help="Additional raw args appended to run_qdrant_fuzz_suite.py")
    parser.add_argument("--skip-targeted-oracle", action="store_true", help="Skip the targeted gRPC scalar smoke pass")
    parser.add_argument("--targeted-rounds", type=int, default=36, help="Rounds for the targeted scalar smoke pass")
    parser.add_argument("--targeted-seed", type=int, default=1701, help="Seed for the targeted scalar smoke pass")
    parser.add_argument("--targeted-rows", type=int, default=1600, help="Rows for the targeted scalar smoke pass")
    parser.add_argument("--targeted-dim", type=int, default=128, help="Dim for the targeted scalar smoke pass")
    parser.add_argument("--targeted-batch-size", type=int, default=250, help="Batch size for the targeted scalar smoke pass")
    parser.add_argument("--targeted-sleep-interval", type=float, default=0.0, help="Insert sleep interval for the targeted scalar smoke pass")
    parser.add_argument(
        "--targeted-distance",
        choices=["random", "euclid", "cosine", "dot", "manhattan"],
        default="random",
        help="Distance mode for the targeted scalar smoke pass",
    )
    parser.add_argument("--targeted-oracle-args", default=None, help="Additional raw args appended to the targeted scalar smoke command")
    parser.add_argument(
        "--budget-oracle-args",
        default=None,
        help="Additional raw args appended to every time-budget direct oracle/PQS/equiv/group command",
    )
    parser.add_argument("--skip-vector", action="store_true", help="Skip the vector smoke pass")
    parser.add_argument("--vector-seed", type=int, default=424242, help="Seed for the vector smoke pass")
    parser.add_argument("--vector-rounds", type=int, default=60, help="Rounds for the vector smoke pass")
    parser.add_argument("--vector-rows", type=int, default=1200, help="Rows for the vector smoke pass")
    parser.add_argument("--vector-dim", type=int, default=128, help="Dim for the vector smoke pass")
    parser.add_argument("--vector-no-dynamic", action="store_true", help="Disable dynamic ops in the vector smoke pass")
    parser.add_argument("--vector-prefer-grpc", action="store_true", help="Use gRPC transport in the vector smoke pass")
    parser.add_argument("--vector-args", default=None, help="Additional raw args appended to qdrant_vector_fuzz_test.py")
    parser.add_argument("--facet-operator-runner", default=str(DEFAULT_FACET_OPERATOR), help="Path to scalar facet operator validator")
    parser.add_argument("--text-indexed-operator-runner", default=str(DEFAULT_TEXT_INDEXED_OPERATOR), help="Path to indexed text operator validator")
    parser.add_argument("--formula-operator-runner", default=str(DEFAULT_FORMULA_OPERATOR), help="Path to FormulaQuery operator validator suite")
    parser.add_argument("--skip-scalar-operator", action="store_true", help="Skip the scalar/filter operator suite")
    parser.add_argument("--skip-persistent-index", action="store_true", help="Skip the persistent payload-index restart pass")
    parser.add_argument("--skip-facet-operator", action="store_true", help="Skip the scalar facet operator pass")
    parser.add_argument("--skip-text-indexed-operator", action="store_true", help="Skip the indexed text operator pass")
    parser.add_argument("--skip-text-profile-operator", action="store_true", help="Skip the indexed text profile operator pass")
    parser.add_argument("--skip-formula-operator", action="store_true", help="Skip the scalar FormulaQuery operator pass")
    parser.add_argument("--scalar-operator-args", default=None, help="Additional raw args appended to scalar_operator_suite.py")
    parser.add_argument(
        "--experimental-scalar-operators",
        default=None,
        help=(
            "Comma-separated extra scalar operators, or a scalar_operator_suite.py group "
            "name, to run as a second scalar operator pass. Default auto-runs 'robustness'; "
            "pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--experimental-scalar-args",
        default=None,
        help="Additional raw args appended to the experimental scalar operator pass",
    )
    parser.add_argument(
        "--experimental-scalar-blocking",
        action="store_true",
        help="Make failures from the experimental scalar operator pass fail the overall suite",
    )
    parser.add_argument("--persistent-index-args", default=None, help="Additional raw args appended to persistent_index_operator.py")
    parser.add_argument("--facet-operator-args", default=None, help="Additional raw args appended to facet_operator.py")
    parser.add_argument("--text-indexed-operator-args", default=None, help="Additional raw args appended to text_indexed_operator.py")
    parser.add_argument("--text-profile-operator-args", default=None, help="Additional raw args appended to text_profile_operator.py")
    parser.add_argument("--formula-operator-args", default=None, help="Additional raw args appended to formula_suite.py")
    parser.add_argument("--low-threshold", type=int, default=80, help="Emit a low-coverage function report below this line-coverage threshold")
    parser.add_argument("--llvm-profdata-bin", default="llvm-profdata", help="llvm-profdata binary")
    parser.add_argument("--llvm-cov-bin", default="llvm-cov", help="llvm-cov binary")
    parser.add_argument("--ignore-filename-regex", default=DEFAULT_IGNORE_REGEX, help="Regex for llvm-cov --ignore-filename-regex")
    parser.add_argument(
        "--coverage-timeline",
        action="store_true",
        help="Restart Qdrant after each top-level job, collect cumulative LLVM coverage snapshots, and write coverage_timeline.csv",
    )
    parser.add_argument(
        "--coverage-timeline-interval-jobs",
        type=int,
        default=1,
        help="Collect coverage timeline snapshots every N completed jobs when --coverage-timeline is enabled",
    )
    parser.add_argument(
        "--coverage-timeline-min-interval-seconds",
        type=float,
        default=0.0,
        help="Minimum wall-clock seconds between coverage timeline snapshots",
    )
    parser.add_argument(
        "--coverage-timeline-dense-jobs",
        type=int,
        default=12,
        help="During the early phase, collect timeline snapshots after every completed job for this many jobs",
    )
    parser.add_argument(
        "--coverage-timeline-dense-minutes",
        type=float,
        default=20.0,
        help="During the early phase, collect timeline snapshots after every completed job for this many minutes",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and paths without executing")
    parser.add_argument(
        "--coverage-report-only",
        action="store_true",
        help="Skip launch/fuzz execution and only merge/generate coverage artifacts from an existing cov dir",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="Enable coverage-first budget scheduling and stop after roughly this many seconds",
    )
    parser.add_argument(
        "--budget-schedule",
        choices=["cycle", "random"],
        default="random",
        help="How budget-mode fill jobs are selected after the initial all-mode preheat",
    )
    parser.add_argument(
        "--budget-seed",
        type=int,
        default=None,
        help="Deterministic seed used for budget-mode job selection and per-job derived seeds",
    )
    parser.add_argument(
        "--budget-min-fill-job-seconds",
        type=float,
        default=2.0,
        help="Planner assumption for the shortest expected fill job duration; smaller means more planned random jobs",
    )
    parser.add_argument(
        "--budget-fill-overprovision",
        type=float,
        default=1.25,
        help="Planner multiplier on top of the min fill job estimate when pre-generating random fill jobs",
    )
    parser.add_argument(
        "--budget-max-planned-jobs",
        type=int,
        default=5000,
        help="Hard cap for pre-generated budget jobs (preheat + random fill)",
    )
    parser.add_argument(
        "--budget-launch-guard-seconds",
        type=float,
        default=8.0,
        help="Do not launch a new job when remaining budget is below this guard (or 10%% estimate up to 20s)",
    )
    parser.add_argument("--figure-script", default=str(DEFAULT_FIGURE_SCRIPT), help="Publication-style figure generator path")
    parser.add_argument("--figure-output-dir", default=None, help="Optional explicit output dir for generated figure assets")
    parser.add_argument("--skip-figures", action="store_true", help="Skip coverage figure generation")
    args = parser.parse_args()
    if args.time_budget_seconds is not None and args.time_budget_seconds <= 0:
        print(f"{RED}--time-budget-seconds must be positive.{RESET}")
        return 2
    if args.coverage_timeline_interval_jobs <= 0:
        print(f"{RED}--coverage-timeline-interval-jobs must be positive.{RESET}")
        return 2
    if args.coverage_timeline_min_interval_seconds < 0:
        print(f"{RED}--coverage-timeline-min-interval-seconds must be non-negative.{RESET}")
        return 2
    if args.coverage_timeline_dense_jobs < 0:
        print(f"{RED}--coverage-timeline-dense-jobs must be non-negative.{RESET}")
        return 2
    if args.coverage_timeline_dense_minutes < 0:
        print(f"{RED}--coverage-timeline-dense-minutes must be non-negative.{RESET}")
        return 2
    if args.budget_min_fill_job_seconds <= 0:
        print(f"{RED}--budget-min-fill-job-seconds must be positive.{RESET}")
        return 2
    if args.budget_fill_overprovision <= 0:
        print(f"{RED}--budget-fill-overprovision must be positive.{RESET}")
        return 2
    if args.budget_max_planned_jobs <= 0:
        print(f"{RED}--budget-max-planned-jobs must be positive.{RESET}")
        return 2
    if args.budget_launch_guard_seconds < 0:
        print(f"{RED}--budget-launch-guard-seconds must be non-negative.{RESET}")
        return 2
    extra_tool_roots = [
        Path(args.experiment_root).expanduser(),
        Path(args.experiment_root).expanduser() / "qdrant_log",
    ]
    if args.qdrant_target_dir:
        target_dir = Path(args.qdrant_target_dir).expanduser()
        extra_tool_roots.extend([target_dir, *list(target_dir.parents[:4])])
    if args.binary:
        binary_path = Path(args.binary).expanduser()
        extra_tool_roots.extend([binary_path.parent, *list(binary_path.parents[:4])])
    args.llvm_profdata_bin = resolve_llvm_tool("llvm-profdata", args.llvm_profdata_bin, extra_roots=extra_tool_roots)
    args.llvm_cov_bin = resolve_llvm_tool("llvm-cov", args.llvm_cov_bin, extra_roots=extra_tool_roots)

    launcher = Path(args.launcher).expanduser().resolve()
    runner = Path(args.runner).expanduser().resolve()
    oracle_path = Path(args.oracle_path).expanduser().resolve()
    vector_runner = Path(args.vector_runner).expanduser().resolve()
    scalar_operator_runner = Path(args.scalar_operator_runner).expanduser().resolve()
    persistent_index_runner = Path(args.persistent_index_runner).expanduser().resolve()
    facet_operator_runner = Path(args.facet_operator_runner).expanduser().resolve()
    text_indexed_operator_runner = Path(args.text_indexed_operator_runner).expanduser().resolve()
    text_profile_operator_runner = Path(args.text_profile_operator_runner).expanduser().resolve()
    formula_operator_runner = Path(args.formula_operator_runner).expanduser().resolve()
    figure_script = Path(args.figure_script).expanduser().resolve()

    if not args.qdrant_src:
        print(f"{RED}--qdrant-src is required (or set QDRANT_SRC).{RESET}")
        return 2
    qdrant_src = Path(args.qdrant_src).expanduser().resolve()
    if not qdrant_src.exists():
        print(f"{RED}Qdrant source not found:{RESET} {qdrant_src}")
        return 2

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    log_root, cov_root, data_root = resolve_experiment_layout(experiment_root)

    run_id = args.run_id or f"qdrant-cov-{now_utc_stamp()}"
    args.run_id = run_id
    log_dir = log_root / f"coverage_suite_{run_id}"
    cov_dir = Path(args.cov_dir).expanduser().resolve() if args.cov_dir else (cov_root / run_id)
    storage_dir = data_root / run_id
    artifacts_dir = log_dir / "artifacts"
    figure_output_dir = (
        Path(args.figure_output_dir).expanduser().resolve()
        if args.figure_output_dir
        else (artifacts_dir / "figures")
    )
    figure_manifest_path = artifacts_dir / "plot_manifest.json"
    figure_log_path = artifacts_dir / "figure_generation.log"
    if not args.qdrant_target_dir:
        args.qdrant_target_dir = str(cov_root / "target")
    binary_path = resolve_binary_path(args, qdrant_src)

    if args.coverage_report_only:
        cov_summary = collect_llvm_coverage(
            qdrant_src=qdrant_src,
            binary_path=binary_path,
            cov_dir=cov_dir,
            out_dir=artifacts_dir,
            ignore_regex=args.ignore_filename_regex,
            low_threshold=args.low_threshold,
            llvm_profdata_bin=args.llvm_profdata_bin,
            llvm_cov_bin=args.llvm_cov_bin,
        )
        if cov_summary.get("available") and cov_summary.get("summary_path"):
            cov_summary["scalar_groups"] = compute_scalar_coverage_groups(Path(str(cov_summary["summary_path"])))
        coverage_metrics_path = log_dir / "coverage_metrics.json"
        write_coverage_metrics_json(coverage_metrics_path, cov_summary)
        payload = {
            "run_id": run_id,
            "qdrant_src": str(qdrant_src),
            "experiment_root": str(experiment_root),
            "log_root": str(log_root),
            "cov_root": str(cov_root),
            "data_root": str(data_root),
            "host": args.host,
            "port": args.port,
            "grpc_port": args.grpc_port,
            "suite_passed": True,
            "suite_elapsed_seconds": 0.0,
            "total_elapsed_seconds": 0.0,
            "suite_log": None,
            "server_log": None,
            "blocking_failures": [],
            "non_blocking_failures": [],
            "jobs": [],
            "coverage": cov_summary,
            "coverage_metrics_json": str(coverage_metrics_path),
            "coverage_timeline_csv": None,
        }
        summary_path = write_summary_json_compat(log_dir, payload)
        summary_md_path = log_dir / "summary.md"
        case_results_csv_path = log_dir / "case_results.csv"
        payload["summary_md"] = str(summary_md_path)
        payload["case_results_csv"] = str(case_results_csv_path)
        write_csv(
            case_results_csv_path,
            ["row_type", "name", "seed", "actual_seed", "result", "elapsed_seconds", "detail", "log_path", "command", "reproduce_command", "blocking"],
            [],
        )
        if args.skip_figures:
            payload["figure_artifacts"] = {"enabled": False, "detail": "skipped by --skip-figures"}
        else:
            timeline_csv_candidate = log_dir / "coverage_timeline.csv"
            payload["figure_artifacts"] = generate_qdrant_figure_artifacts(
                python_bin=args.python_bin,
                figure_script=figure_script,
                summary_json_path=summary_path,
                timeline_csv_path=timeline_csv_candidate if timeline_csv_candidate.exists() else None,
                output_dir=figure_output_dir,
                manifest_path=figure_manifest_path,
                log_path=figure_log_path,
            )
        summary_path = write_summary_json_compat(log_dir, payload)
        write_summary_markdown(summary_md_path, payload)
        return print_report_summary(log_dir, summary_path, payload)

    if not launcher.exists():
        print(f"{RED}Launcher not found:{RESET} {launcher}")
        return 2
    if not runner.exists():
        print(f"{RED}Runner not found:{RESET} {runner}")
        return 2
    if not args.skip_targeted_oracle and not oracle_path.exists():
        print(f"{RED}Oracle script not found:{RESET} {oracle_path}")
        return 2
    if not args.skip_vector and not vector_runner.exists():
        print(f"{RED}Vector runner not found:{RESET} {vector_runner}")
        return 2
    if not args.skip_scalar_operator and not scalar_operator_runner.exists():
        print(f"{RED}Scalar operator runner not found:{RESET} {scalar_operator_runner}")
        return 2
    if not args.skip_persistent_index and not persistent_index_runner.exists():
        print(f"{RED}Persistent index runner not found:{RESET} {persistent_index_runner}")
        return 2
    if not args.skip_facet_operator and not facet_operator_runner.exists():
        print(f"{RED}Facet operator runner not found:{RESET} {facet_operator_runner}")
        return 2
    if not args.skip_text_indexed_operator and not text_indexed_operator_runner.exists():
        print(f"{RED}Indexed text operator runner not found:{RESET} {text_indexed_operator_runner}")
        return 2
    if not args.skip_text_profile_operator and not text_profile_operator_runner.exists():
        print(f"{RED}Text profile operator runner not found:{RESET} {text_profile_operator_runner}")
        return 2
    if not args.skip_formula_operator and not formula_operator_runner.exists():
        print(f"{RED}Formula operator runner not found:{RESET} {formula_operator_runner}")
        return 2

    scalar_oracle_log_dir = log_dir / "oracle_logs" / "matrix"
    targeted_oracle_log_dir = log_dir / "oracle_logs" / "targeted"
    vector_inner_log_dir = log_dir / "vector_logs"
    scalar_suite_runner_log_dir = log_dir / "scalar_suite_cases"
    scalar_suite_summary_json = log_dir / "scalar_suite_summary.json"
    scalar_suite_case_results_csv = log_dir / "scalar_suite_case_results.csv"
    persistent_index_state_path = log_dir / "persistent_index_state.json"

    budget_seed: int | None = None
    if args.time_budget_seconds is not None:
        job_specs, budget_seed = build_budget_job_specs(
            args=args,
            python_bin=args.python_bin,
            oracle_path=oracle_path,
            vector_runner=vector_runner,
            scalar_operator_runner=scalar_operator_runner,
            persistent_index_runner=persistent_index_runner,
            facet_operator_runner=facet_operator_runner,
            text_indexed_operator_runner=text_indexed_operator_runner,
            text_profile_operator_runner=text_profile_operator_runner,
            formula_operator_runner=formula_operator_runner,
            log_dir=log_dir,
            persistent_index_state_path=persistent_index_state_path,
        )
    else:
        job_specs = []
        suite_cmd = build_suite_command(
            args.python_bin,
            runner,
            args,
            oracle_log_dir=scalar_oracle_log_dir,
            run_id=f"{run_id}-matrix",
            log_dir=scalar_suite_runner_log_dir,
            summary_json=scalar_suite_summary_json,
            case_results_csv=scalar_suite_case_results_csv,
        )
        job_specs.append(
            {
                "name": "scalar-suite",
                "command": suite_cmd,
                "log_path": log_dir / "scalar_suite.log",
                "analyzer": analyze_suite_log,
                "summary_json": str(scalar_suite_summary_json),
                "case_results_csv": str(scalar_suite_case_results_csv),
                "blocking": True,
            }
        )
        if not args.skip_targeted_oracle:
            job_specs.append(
                {
                    "name": "scalar-targeted",
                    "command": build_targeted_oracle_command(args.python_bin, oracle_path, args, targeted_oracle_log_dir),
                    "log_path": log_dir / "scalar_targeted.log",
                    "analyzer": analyze_oracle_log,
                    "blocking": True,
                }
            )
        if not args.skip_vector:
            job_specs.append(
                {
                    "name": "vector-smoke",
                    "command": build_vector_command(args.python_bin, vector_runner, args, vector_inner_log_dir),
                    "log_path": log_dir / "vector_smoke.log",
                    "analyzer": analyze_vector_log,
                    "blocking": True,
                }
            )
        if not args.skip_scalar_operator:
            job_specs.append(
                {
                    "name": "scalar-operator-suite",
                    "command": build_operator_command(
                        args.python_bin,
                        scalar_operator_runner,
                        args,
                        raw_args=args.scalar_operator_args,
                        run_id=f"{run_id}-scalar-operators",
                    ),
                    "log_path": log_dir / "scalar_operator_suite.log",
                    "analyzer": analyze_operator_log,
                    "blocking": True,
                }
            )
        effective_experimental_scalar_operators = (
            "robustness" if args.experimental_scalar_operators is None else str(args.experimental_scalar_operators).strip()
        )
        if effective_experimental_scalar_operators:
            experimental_scalar_parts = ["--operators", effective_experimental_scalar_operators]
            if args.experimental_scalar_args:
                experimental_scalar_parts.extend(shlex.split(args.experimental_scalar_args))
            job_specs.append(
                {
                    "name": "scalar-operator-experimental",
                    "command": build_operator_command(
                        args.python_bin,
                        scalar_operator_runner,
                        args,
                        raw_args=shlex.join(experimental_scalar_parts),
                        run_id=f"{run_id}-scalar-operators-experimental",
                    ),
                    "log_path": log_dir / "scalar_operator_experimental.log",
                    "analyzer": analyze_operator_log,
                    "blocking": bool(args.experimental_scalar_blocking),
                }
            )
        if not args.skip_persistent_index:
            persistent_run_id = f"{run_id}-persistent-index"
            job_specs.append(
                {
                    "name": "persistent-index-prepare",
                    "command": build_persistent_index_command(
                        args.python_bin,
                        persistent_index_runner,
                        args,
                        phase="prepare",
                        state_path=persistent_index_state_path,
                        raw_args=args.persistent_index_args,
                        run_id=persistent_run_id,
                    ),
                    "log_path": log_dir / "persistent_index_prepare.log",
                    "analyzer": analyze_operator_log,
                    "blocking": True,
                    "restart_server_after": True,
                }
            )
            job_specs.append(
                {
                    "name": "persistent-index-verify",
                    "command": build_persistent_index_command(
                        args.python_bin,
                        persistent_index_runner,
                        args,
                        phase="verify",
                        state_path=persistent_index_state_path,
                        raw_args=args.persistent_index_args,
                        run_id=persistent_run_id,
                    ),
                    "log_path": log_dir / "persistent_index_verify.log",
                    "analyzer": analyze_operator_log,
                    "blocking": True,
                }
            )
        if not args.skip_facet_operator:
            job_specs.append(
                {
                    "name": "facet-operator",
                    "command": build_operator_command(
                        args.python_bin,
                        facet_operator_runner,
                        args,
                        raw_args=args.facet_operator_args,
                        run_id=f"{run_id}-facet",
                    ),
                    "log_path": log_dir / "facet_operator.log",
                    "analyzer": analyze_operator_log,
                    "blocking": True,
                }
            )
        if not args.skip_text_indexed_operator:
            job_specs.append(
                {
                    "name": "text-indexed-operator",
                    "command": build_operator_command(
                        args.python_bin,
                        text_indexed_operator_runner,
                        args,
                        raw_args=args.text_indexed_operator_args,
                        run_id=f"{run_id}-text-indexed",
                    ),
                    "log_path": log_dir / "text_indexed_operator.log",
                    "analyzer": analyze_operator_log,
                    "blocking": True,
                }
            )
        if not args.skip_text_profile_operator:
            job_specs.append(
                {
                    "name": "text-profile-operator",
                    "command": build_operator_command(
                        args.python_bin,
                        text_profile_operator_runner,
                        args,
                        raw_args=args.text_profile_operator_args,
                        run_id=f"{run_id}-text-profile",
                    ),
                    "log_path": log_dir / "text_profile_operator.log",
                    "analyzer": analyze_operator_log,
                    "blocking": True,
                }
            )
        if not args.skip_formula_operator:
            job_specs.append(
                {
                    "name": "formula-operator",
                    "command": build_operator_command(
                        args.python_bin,
                        formula_operator_runner,
                        args,
                        raw_args=args.formula_operator_args,
                        run_id=f"{run_id}-formula",
                    ),
                    "log_path": log_dir / "formula_operator.log",
                    "analyzer": analyze_operator_log,
                    "blocking": True,
                }
            )

    server_log_path = log_dir / "server.log"

    print(f"\n{BOLD}{CYAN}{'=' * 76}{RESET}")
    print(f"{BOLD}{CYAN}  Qdrant Coverage Fuzz Suite{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 76}{RESET}")
    print(f"  Qdrant src:  {qdrant_src}")
    print(f"  Launcher:    {launcher}")
    print(f"  Scalar:      {runner}")
    print(f"  Targeted:    {oracle_path if not args.skip_targeted_oracle else '(skipped)'}")
    print(f"  Vector:      {vector_runner if not args.skip_vector else '(skipped)'}")
    print(f"  ScalarOps:   {scalar_operator_runner if not args.skip_scalar_operator else '(skipped)'}")
    print(f"  Persistent:  {persistent_index_runner if not args.skip_persistent_index else '(skipped)'}")
    print(f"  TextProfile: {text_profile_operator_runner if not args.skip_text_profile_operator else '(skipped)'}")
    print(f"  Formula:     {formula_operator_runner if not args.skip_formula_operator else '(skipped)'}")
    print(f"  Run ID:      {run_id}")
    print(f"  Host:Port:   {args.host}:{args.port} (grpc:{args.grpc_port})")
    print(f"  Exp root:    {experiment_root}")
    print(f"  Log root:    {log_root}")
    print(f"  Storage dir: {storage_dir}")
    print(f"  Cov dir:     {cov_dir}")
    print(f"  Binary:      {binary_path}")
    print(f"  Log dir:     {log_dir}")
    if args.time_budget_seconds is not None:
        print(f"  Budget:      {fmt_duration(float(args.time_budget_seconds))}")
        print(f"  Schedule:    {args.budget_schedule}")
        print(f"  Budget seed: {budget_seed}")
        print(
            f"  Planner:     min-job {args.budget_min_fill_job_seconds:g}s, "
            f"overprovision x{args.budget_fill_overprovision:g}, "
            f"max-jobs {args.budget_max_planned_jobs}"
        )
        print(f"  Guard:       launch guard {args.budget_launch_guard_seconds:g}s")
        print(f"  Budget mode: coverage-first preheat + seeded fill")
    if args.coverage_timeline:
        print(
            f"  Timeline:    dense<=jobs:{args.coverage_timeline_dense_jobs}, "
            f"dense<=mins:{args.coverage_timeline_dense_minutes:g}, "
            f"late every {args.coverage_timeline_interval_jobs} jobs + {args.coverage_timeline_min_interval_seconds:g}s"
        )
    preview_job_specs = job_specs
    if args.time_budget_seconds is not None and len(job_specs) > 24:
        preview_job_specs = job_specs[:24]
    for job in preview_job_specs:
        print(f"  Job cmd:     {job['name']} -> {' '.join(job['command'])}")
    if preview_job_specs is not job_specs:
        print(f"  Job cmd:     ... {len(job_specs) - len(preview_job_specs)} more planned budget jobs")
    print(f"{CYAN}{'-' * 76}{RESET}")

    if args.dry_run:
        print(f"{YELLOW}Dry run only; nothing executed.{RESET}")
        return 0

    if port_in_use(args.host, args.port):
        print(
            f"{RED}Port {args.port} on {args.host} is already in use.{RESET}\n"
            f"Stop the existing service first, or rerun the suite on a different host/port."
        )
        return 2

    env = os.environ.copy()
    env["QDRANT_SRC"] = str(qdrant_src)
    env["QDRANT_EXPERIMENT_ROOT"] = str(experiment_root)
    env["QDRANT_COV_DIR"] = str(cov_dir)
    env["QDRANT_STORAGE_DIR"] = str(storage_dir)
    env["QDRANT_HOST"] = args.host
    env["QDRANT_HTTP_PORT"] = str(args.port)
    env["QDRANT_GRPC_PORT"] = str(args.grpc_port)
    env["QDRANT_CARGO_PROFILE"] = args.cargo_profile
    if args.qdrant_target_dir:
        env["QDRANT_TARGET_DIR"] = str(Path(args.qdrant_target_dir).expanduser().resolve())
    if args.qdrant_config_path:
        env["QDRANT_CONFIG_PATH"] = str(Path(args.qdrant_config_path).expanduser().resolve())

    server_proc: subprocess.Popen | None = None
    server_log_handle = None
    job_results: list[dict[str, object]] = []
    coverage_timeline_rows: list[dict[str, object]] = []
    coverage_timeline_csv_path = log_dir / "coverage_timeline.csv"
    coverage_metrics_path = log_dir / "coverage_metrics.json"
    started = time.time()
    deadline = (started + float(args.time_budget_seconds)) if args.time_budget_seconds is not None else None
    last_timeline_snapshot_time = started
    last_timeline_snapshot_job_idx = 0

    try:
        print(f"\n{CYAN}Starting coverage-enabled Qdrant...{RESET}")
        server_proc, server_log_handle = start_server(
            launcher=launcher,
            run_id=run_id,
            server_log_path=server_log_path,
            host=args.host,
            port=args.port,
            ready_timeout=args.ready_timeout,
            env=env,
        )
        env["QDRANT_SKIP_BUILD"] = "1"
        print(f"{GREEN}Qdrant is ready.{RESET}")
        if args.coverage_timeline:
            coverage_timeline_rows.append(
                build_timeline_row(
                    step_index=0,
                    job_index=0,
                    event="start",
                    job_name="start",
                    job_result="READY",
                    blocking=True,
                    job_elapsed_seconds=0.0,
                    suite_elapsed_seconds=0.0,
                    wall_elapsed_seconds=time.time() - started,
                    coverage=None,
                    snapshot_dir=None,
                )
            )

        for job_idx, job in enumerate(job_specs, start=1):
            if job_idx > 1:
                can_run, stop_reason = can_launch_job(
                    job_specs,
                    job_idx,
                    deadline=deadline,
                    now=time.time(),
                    launch_guard_seconds=args.budget_launch_guard_seconds,
                )
                if not can_run:
                    print(f"{YELLOW}{stop_reason}{RESET}")
                    break
            name = str(job["name"])
            command = list(job["command"])
            log_path = Path(job["log_path"])
            analyzer = job["analyzer"]
            blocking = bool(job.get("blocking", True))
            timeout_seconds = float(job["timeout_seconds"]) if job.get("timeout_seconds") else None
            print(f"{CYAN}Running {name}...{RESET}")
            exit_code, elapsed = run_command(command, log_path, timeout_seconds=timeout_seconds)
            passed, detail = analyzer(log_path)
            result = {
                "name": name,
                "command": command,
                "log_path": str(log_path),
                "exit_code": exit_code,
                "passed": bool(exit_code == 0 and passed),
                "detail": detail if exit_code == 0 else f"exit={exit_code}, {detail}",
                "elapsed_seconds": elapsed,
                "blocking": blocking,
                "timeout_seconds": timeout_seconds,
            }
            job_results.append(result)
            print(f"    -> {job_status_text(result)} ({fmt_duration(elapsed)}) {result['detail']}")
            now_after_job = time.time()
            should_start_next_job, _ = can_launch_job(
                job_specs,
                job_idx + 1,
                deadline=deadline,
                now=now_after_job,
                launch_guard_seconds=args.budget_launch_guard_seconds,
            )
            take_timeline_snapshot = False
            if args.coverage_timeline and should_start_next_job:
                jobs_since_snapshot = job_idx - last_timeline_snapshot_job_idx
                wall_since_snapshot = now_after_job - last_timeline_snapshot_time
                dense_phase_enabled = args.coverage_timeline_dense_jobs > 0 or args.coverage_timeline_dense_minutes > 0
                dense_job_phase = args.coverage_timeline_dense_jobs > 0 and job_idx <= args.coverage_timeline_dense_jobs
                dense_time_phase = (
                    args.coverage_timeline_dense_minutes > 0
                    and (now_after_job - started) <= args.coverage_timeline_dense_minutes * 60.0
                )
                in_dense_phase = dense_phase_enabled and (dense_job_phase or dense_time_phase)
                if in_dense_phase:
                    take_timeline_snapshot = True
                else:
                    take_timeline_snapshot = (
                        jobs_since_snapshot >= args.coverage_timeline_interval_jobs
                        and wall_since_snapshot >= args.coverage_timeline_min_interval_seconds
                    )
            should_restart_after_job = bool(job.get("restart_server_after")) or take_timeline_snapshot
            if should_restart_after_job:
                print(f"{CYAN}Restarting coverage-enabled Qdrant after {name}...{RESET}")
                stop_server(server_proc)
                server_proc = None
                if server_log_handle is not None:
                    server_log_handle.close()
                    server_log_handle = None
                if take_timeline_snapshot:
                    snapshot_dir = artifacts_dir / "timeline" / f"{job_idx:02d}_{slugify(name, max_len=48)}"
                    snapshot_cov = collect_llvm_coverage_snapshot(
                        qdrant_src=qdrant_src,
                        binary_path=binary_path,
                        cov_dir=cov_dir,
                        out_dir=snapshot_dir,
                        ignore_regex=args.ignore_filename_regex,
                        llvm_profdata_bin=args.llvm_profdata_bin,
                        llvm_cov_bin=args.llvm_cov_bin,
                    )
                    coverage_timeline_rows.append(
                        build_timeline_row(
                            step_index=job_idx,
                            job_index=job_idx,
                            event="job_complete",
                            job_name=name,
                            job_result=job_result_label(result),
                            blocking=blocking,
                            job_elapsed_seconds=elapsed,
                            suite_elapsed_seconds=sum(float(item["elapsed_seconds"]) for item in job_results),
                            wall_elapsed_seconds=time.time() - started,
                            coverage=snapshot_cov,
                            snapshot_dir=snapshot_dir,
                        )
                    )
                    last_timeline_snapshot_time = time.time()
                    last_timeline_snapshot_job_idx = job_idx
                if should_start_next_job:
                    server_proc, server_log_handle = start_server(
                        launcher=launcher,
                        run_id=run_id,
                        server_log_path=server_log_path,
                        host=args.host,
                        port=args.port,
                        ready_timeout=max(120.0, min(args.ready_timeout, 600.0)),
                        env=env,
                        append_log=True,
                    )
                    print(f"{GREEN}Qdrant is ready after restart.{RESET}")
    except RuntimeError as exc:
        print(f"{RED}{exc}{RESET}")
        return 1
    finally:
        stop_server(server_proc)
        if server_log_handle is not None:
            server_log_handle.close()

    cov_summary = collect_llvm_coverage(
        qdrant_src=qdrant_src,
        binary_path=binary_path,
        cov_dir=cov_dir,
        out_dir=artifacts_dir,
        ignore_regex=args.ignore_filename_regex,
        low_threshold=args.low_threshold,
        llvm_profdata_bin=args.llvm_profdata_bin,
        llvm_cov_bin=args.llvm_cov_bin,
    )
    if cov_summary.get("available") and cov_summary.get("summary_path"):
        cov_summary["scalar_groups"] = compute_scalar_coverage_groups(Path(str(cov_summary["summary_path"])))
    write_coverage_metrics_json(coverage_metrics_path, cov_summary)

    suite_elapsed = sum(float(job["elapsed_seconds"]) for job in job_results)
    if args.coverage_timeline:
        last_job = job_results[-1] if job_results else None
        last_row = coverage_timeline_rows[-1] if coverage_timeline_rows else None
        scalar_target = (
            cov_summary.get("scalar_groups", {}).get("scalar_target", {})
            if isinstance(cov_summary.get("scalar_groups"), dict)
            else {}
        )
        final_scalar_line_percent = round(float(scalar_target.get("line_percent") or 0.0), 2)
        final_overall_covered_lines = int(cov_summary.get("covered_lines") or 0)
        needs_final_row = last_job is not None or bool(coverage_timeline_rows)
        if isinstance(last_row, dict):
            needs_final_row = (
                str(last_row.get("event")) != "final"
                or int(last_row.get("overall_covered_lines") or 0) != final_overall_covered_lines
                or float(last_row.get("scalar_target_line_percent") or 0.0) != final_scalar_line_percent
            )
        if needs_final_row:
            coverage_timeline_rows.append(
                build_timeline_row(
                    step_index=len(job_results) + 1,
                    job_index=len(job_results),
                    event="final",
                    job_name=str(last_job["name"]) if last_job else "suite_final",
                    job_result=job_result_label(last_job) if last_job else ("PASS" if cov_summary.get("available") else "FAIL"),
                    blocking=True,
                    job_elapsed_seconds=float(last_job["elapsed_seconds"]) if last_job else 0.0,
                    suite_elapsed_seconds=suite_elapsed,
                    wall_elapsed_seconds=time.time() - started,
                    coverage=cov_summary,
                    snapshot_dir=None,
                )
            )
    blocking_failures = [str(job["name"]) for job in job_results if bool(job.get("blocking", True)) and not bool(job["passed"])]
    non_blocking_failures = [str(job["name"]) for job in job_results if not bool(job.get("blocking", True)) and not bool(job["passed"])]
    suite_passed = bool(job_results) and not blocking_failures
    case_results_csv_path = log_dir / "case_results.csv"
    summary_md_path = log_dir / "summary.md"
    case_rows = build_case_results_rows(job_results, scalar_suite_case_results_csv if scalar_suite_case_results_csv.exists() else None)
    payload = {
        "run_id": run_id,
        "qdrant_src": str(qdrant_src),
        "experiment_root": str(experiment_root),
        "log_root": str(log_root),
        "cov_root": str(cov_root),
        "data_root": str(data_root),
        "launcher": str(launcher),
        "runner": str(runner),
        "oracle_path": str(oracle_path),
        "vector_runner": str(vector_runner),
        "scalar_operator_runner": str(scalar_operator_runner),
        "persistent_index_runner": str(persistent_index_runner),
        "text_profile_operator_runner": str(text_profile_operator_runner),
        "host": args.host,
        "port": args.port,
        "grpc_port": args.grpc_port,
        "cov_dir": str(cov_dir),
        "storage_dir": str(storage_dir),
        "binary_path": str(binary_path),
        "time_budget_seconds": float(args.time_budget_seconds) if args.time_budget_seconds is not None else None,
        "budget_schedule": args.budget_schedule if args.time_budget_seconds is not None else None,
        "budget_seed": budget_seed,
        "suite_passed": suite_passed,
        "suite_elapsed_seconds": suite_elapsed,
        "total_elapsed_seconds": time.time() - started,
        "suite_log": str(job_results[0]["log_path"]) if job_results else None,
        "server_log": str(server_log_path),
        "case_results_csv": str(case_results_csv_path),
        "summary_md": str(summary_md_path),
        "coverage_metrics_json": str(coverage_metrics_path),
        "coverage_timeline_csv": str(coverage_timeline_csv_path) if coverage_timeline_rows else None,
        "blocking_failures": blocking_failures,
        "non_blocking_failures": non_blocking_failures,
        "jobs": job_results,
        "coverage": cov_summary,
    }
    summary_path = write_summary_json_compat(log_dir, payload)
    if coverage_timeline_rows:
        write_csv(coverage_timeline_csv_path, coverage_timeline_fieldnames(), coverage_timeline_rows)
    write_csv(
        case_results_csv_path,
        ["row_type", "name", "seed", "actual_seed", "result", "elapsed_seconds", "detail", "log_path", "command", "reproduce_command", "blocking"],
        case_rows,
    )
    if args.skip_figures:
        payload["figure_artifacts"] = {"enabled": False, "detail": "skipped by --skip-figures"}
    else:
        payload["figure_artifacts"] = generate_qdrant_figure_artifacts(
            python_bin=args.python_bin,
            figure_script=figure_script,
            summary_json_path=summary_path,
            timeline_csv_path=coverage_timeline_csv_path if coverage_timeline_rows else None,
            output_dir=figure_output_dir,
            manifest_path=figure_manifest_path,
            log_path=figure_log_path,
        )
    summary_path = write_summary_json_compat(log_dir, payload)
    write_summary_markdown(summary_md_path, payload)

    print(f"\n{BOLD}{'=' * 76}{RESET}")
    print(f"{BOLD}Run Summary{RESET}")
    print(f"{'=' * 76}")
    print(f"  Suite:       {colorize(payload['suite_passed'], 'PASS' if payload['suite_passed'] else 'FAIL')}")
    print(f"  Suite time:  {fmt_duration(suite_elapsed)}")
    print(f"  Total time:  {fmt_duration(payload['total_elapsed_seconds'])}")
    print(f"  Server log:  {server_log_path}")
    print(f"  Summary:     {summary_path}")
    print(f"  Summary MD:  {summary_md_path}")
    print(f"  Case CSV:    {case_results_csv_path}")
    if isinstance(payload.get("figure_artifacts"), dict) and payload["figure_artifacts"].get("manifest_path"):
        print(f"  Figures:     {payload['figure_artifacts']['manifest_path']}")
    if blocking_failures:
        print(f"  Blocking:    {', '.join(blocking_failures)}")
    if non_blocking_failures:
        print(f"  Warnings:    {', '.join(non_blocking_failures)}")
    for job in job_results:
        blocking_tag = "blocking" if bool(job.get("blocking", True)) else "non-blocking"
        print(f"  Job:         {job['name']} -> {job_status_text(job)} [{blocking_tag}] ({fmt_duration(float(job['elapsed_seconds']))})")
        print(f"               {job['detail']}")
        print(f"               {job['log_path']}")
    if cov_summary.get("available"):
        print(f"  Coverage:    {cov_summary.get('summary_path')}")
        print(f"  HTML dir:    {cov_summary.get('html_dir')}")
        scalar_target = cov_summary.get("scalar_groups", {}).get("scalar_target")
        if scalar_target:
            print(f"  Scalar line: {scalar_target.get('line_percent', 0.0):.2f}%")
        if cov_summary.get("total_line"):
            print(f"  Total:       {cov_summary.get('total_line')}")
    else:
        print(f"  Coverage:    skipped ({cov_summary.get('detail', 'unknown reason')})")

    failed_jobs = [job for job in job_results if not job["passed"]]
    if failed_jobs:
        print(f"\n{RED}{BOLD}Failed job tails:{RESET}")
        for job in failed_jobs:
            print(f"- {job['name']} :: {job['log_path']}")
            snippet = extract_summary_snippet(Path(job["log_path"]), lines=18)
            if snippet:
                print(snippet)
                print("-" * 40)
    if not cov_summary.get("available"):
        print(f"\n{YELLOW}Coverage collection did not complete. Inspect logs and rerun with --coverage-report-only if needed.{RESET}")

    return print_report_summary(log_dir, summary_path, payload)


if __name__ == "__main__":
    raise SystemExit(main())
