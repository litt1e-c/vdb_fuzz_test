#!/usr/bin/env python3
"""
Weaviate coverage-oriented fuzz scheduler.

Boots a coverage-enabled Weaviate instance once, runs a matrix of fuzz jobs
against it, then emits a suite summary plus `go tool covdata` reports.
The default matrix emphasizes scalar-filter coverage rather than vector-only paths.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_PATH = SCRIPT_DIR / "weaviate_fuzz_oracle.py"
DEFAULT_LAUNCHER = SCRIPT_DIR / "start_weaviate_cov.sh"
DEFAULT_LOG_ROOT = SCRIPT_DIR / "weaviate_log"
DEFAULT_COV_ROOT = SCRIPT_DIR / ".cov"


@dataclass
class CaseTemplate:
    name: str
    mode: str
    rounds: int
    seeds: list[int | None]
    rows: int
    extra_args: list[str] = field(default_factory=list)


@dataclass
class SuiteCase:
    name: str
    mode: str
    rounds: int
    seed: int | None
    rows: int
    extra_args: list[str]


@dataclass
class CaseResult:
    name: str
    mode: str
    seed: int | None
    rows: int
    rounds: int
    command: list[str]
    log_path: str
    exit_code: int
    passed: bool
    detail: str
    elapsed_seconds: float


SCALAR_SMOKE_TEMPLATES = [
    CaseTemplate("oracle-nodyn-smoke", "oracle", 12, [1238], 1200, ["--no-dynamic"]),
    CaseTemplate("oracle-dyn-smoke", "oracle", 18, [2238], 1200, []),
    CaseTemplate("oracle-inverted-smoke", "oracle", 16, [1838], 1400, ["--no-dynamic", "--profile", "inverted"]),
    CaseTemplate("equiv-smoke", "equiv", 10, [3238], 1000, []),
    CaseTemplate("pqs-smoke", "pqs", 12, [4238], 1000, []),
    CaseTemplate("aggregate-smoke", "aggregate", 12, [5238], 1200, []),
    CaseTemplate("aggregate-inverted-smoke", "aggregate", 14, [5838], 1400, ["--profile", "inverted"]),
    CaseTemplate(
        "rest-filter-smoke",
        "rest-filter",
        12,
        [6238],
        1200,
        ["--rest-filter-min-depth", "3", "--rest-filter-max-depth", "6"],
    ),
    CaseTemplate(
        "rest-filter-inverted-smoke",
        "rest-filter",
        14,
        [6838],
        1400,
        ["--profile", "inverted", "--rest-filter-min-depth", "3", "--rest-filter-max-depth", "6"],
    ),
    CaseTemplate(
        "oracle-inverted-rest-xcheck-smoke",
        "oracle",
        16,
        [7438],
        1400,
        [
            "--no-dynamic",
            "--profile",
            "inverted",
            "--oracle-rest-crosscheck-rate",
            "0.70",
            "--rest-filter-min-depth",
            "3",
            "--rest-filter-max-depth",
            "6",
        ],
    ),
    CaseTemplate(
        "oracle-rest-xcheck-smoke",
        "oracle",
        12,
        [7238],
        1200,
        [
            "--no-dynamic",
            "--oracle-rest-crosscheck-rate",
            "0.70",
            "--rest-filter-min-depth",
            "3",
            "--rest-filter-max-depth",
            "6",
        ],
    ),
]

SCALAR_TEMPLATES = [
    CaseTemplate("oracle-scalar-nodyn-a", "oracle", 35, [1238, 2027], 2500, ["--no-dynamic"]),
    CaseTemplate("oracle-scalar-nodyn-b", "oracle", 35, [3137, 4488], 2500, ["--no-dynamic"]),
    CaseTemplate("oracle-inverted-a", "oracle", 36, [18381, 18382], 2600, ["--no-dynamic", "--profile", "inverted"]),
    CaseTemplate("oracle-scalar-dyn-a", "oracle", 45, [5501, 6602], 2500, []),
    CaseTemplate("oracle-scalar-dyn-b", "oracle", 45, [7703], 4000, []),
    CaseTemplate("oracle-scalar-randcl", "oracle", 35, [8804], 2500, ["--random-consistency"]),
    CaseTemplate("equiv-scalar-a", "equiv", 20, [9101, 9102], 1800, []),
    CaseTemplate("pqs-scalar-a", "pqs", 25, [9201, 9202], 1800, []),
    CaseTemplate("aggregate-scalar-a", "aggregate", 30, [9301, 9302], 2200, []),
    CaseTemplate("aggregate-inverted-a", "aggregate", 28, [9331, 9332], 2400, ["--profile", "inverted"]),
    CaseTemplate(
        "oracle-inverted-rest-xcheck-a",
        "oracle",
        30,
        [9431, 9432],
        2400,
        [
            "--no-dynamic",
            "--profile",
            "inverted",
            "--oracle-rest-crosscheck-rate",
            "0.70",
            "--rest-filter-min-depth",
            "3",
            "--rest-filter-max-depth",
            "6",
        ],
    ),
    CaseTemplate(
        "oracle-rest-xcheck-a",
        "oracle",
        28,
        [9401, 9402],
        2200,
        [
            "--no-dynamic",
            "--oracle-rest-crosscheck-rate",
            "0.70",
            "--rest-filter-min-depth",
            "3",
            "--rest-filter-max-depth",
            "6",
        ],
    ),
    CaseTemplate(
        "rest-filter-scalar-a",
        "rest-filter",
        24,
        [9501, 9502],
        2200,
        ["--rest-filter-min-depth", "3", "--rest-filter-max-depth", "6"],
    ),
    CaseTemplate(
        "rest-filter-inverted-a",
        "rest-filter",
        26,
        [9551, 9552],
        2400,
        ["--profile", "inverted", "--rest-filter-min-depth", "3", "--rest-filter-max-depth", "6"],
    ),
]

FULL_TEMPLATES = SCALAR_TEMPLATES + [
    CaseTemplate("oracle-large-dyn", "oracle", 25, [10001], 12000, []),
    CaseTemplate("groupby-broad", "groupby", 18, [11001], 2000, []),
    CaseTemplate("aggregate-broad", "aggregate", 28, [12001], 5000, []),
    CaseTemplate(
        "rest-filter-broad",
        "rest-filter",
        24,
        [13001],
        4500,
        ["--rest-filter-min-depth", "3", "--rest-filter-max-depth", "6"],
    ),
]

SUITES = {
    "scalar-smoke": SCALAR_SMOKE_TEMPLATES,
    "scalar": SCALAR_TEMPLATES,
    "inverted": [case for case in [*SCALAR_SMOKE_TEMPLATES, *SCALAR_TEMPLATES] if "inverted" in case.name],
    "aggregate": [*SCALAR_SMOKE_TEMPLATES[-1:], *SCALAR_TEMPLATES[-1:], FULL_TEMPLATES[-1]],
    "full": FULL_TEMPLATES,
}


def expand_templates(templates: Iterable[CaseTemplate]) -> list[SuiteCase]:
    cases: list[SuiteCase] = []
    for template in templates:
        for seed in template.seeds:
            cases.append(
                SuiteCase(
                    name=template.name,
                    mode=template.mode,
                    rounds=template.rounds,
                    seed=seed,
                    rows=template.rows,
                    extra_args=list(template.extra_args),
                )
            )
    return cases


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


def build_case_command(python_bin: str, case: SuiteCase, query_page_size: int | None) -> list[str]:
    cmd = [
        python_bin,
        str(ORACLE_PATH),
        "--mode",
        case.mode,
        "--rounds",
        str(case.rounds),
        "-N",
        str(case.rows),
    ]
    if case.seed is not None:
        cmd += ["--seed", str(case.seed)]
    if query_page_size is not None:
        cmd += ["--query-page-size", str(query_page_size)]
    cmd.extend(case.extra_args)
    return cmd


def port_ready(host: str, port: int, timeout_seconds: float) -> bool:
    ready_urls = [
        f"http://{host}:{port}/v1/.well-known/ready",
        f"http://{host}:{port}/v1/meta",
    ]
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        for url in ready_urls:
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    if resp.status < 500:
                        return True
            except urllib.error.URLError:
                pass
            except TimeoutError:
                pass
        time.sleep(1.0)
    return False


def port_in_use(host: str, port: int) -> bool:
    return port_ready(host, port, timeout_seconds=1.5)


def start_server(launcher: Path, run_id: str, host: str, port: int, ready_timeout: float) -> subprocess.Popen:
    process = subprocess.Popen(
        ["bash", str(launcher), run_id],
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not port_ready(host, port, ready_timeout):
        stop_server(process)
        raise RuntimeError(f"Weaviate did not become ready on {host}:{port} within {ready_timeout:.0f}s")
    return process


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def analyze_log(log_path: Path, mode: str) -> tuple[bool, str]:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False, "log file missing"

    if "Traceback (most recent call last)" in content:
        return False, "python traceback"
    if "CRASH:" in content:
        return False, "runtime crash marker"
    if "🚫" in content:
        return False, "suite reported failures"
    if "[Dyn] Insert fail" in content or "[Dyn] Update fail" in content or "[Dyn] Upsert fail" in content:
        return False, "dynamic op failure"

    success_patterns = [
        "✅ All",
        "✅ All passed",
    ]
    if any(pattern in content for pattern in success_patterns):
        return True, "success marker found"

    if mode == "oracle" and "Stats:" in content and "Fail:0" in content:
        return True, "oracle stats indicate pass"

    return False, "no success marker found"


def extract_summary_snippet(log_path: Path, lines: int = 12) -> str:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(content[-lines:])


def run_case(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            command,
            cwd=str(SCRIPT_DIR),
            stdout=fout,
            stderr=subprocess.STDOUT,
        )
        exit_code = proc.wait()
    return exit_code, time.time() - start


def collect_covdata(go_bin: str, cov_dir: Path, out_dir: Path, percent_regex: str | None) -> dict:
    summary: dict[str, str | int | bool] = {"available": False}
    if shutil.which(go_bin) is None:
        summary["detail"] = f"`{go_bin}` not found; skipped covdata export"
        return summary
    if not cov_dir.exists():
        summary["detail"] = f"coverage dir missing: {cov_dir}"
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    percent_path = out_dir / "coverage_percent.txt"
    textfmt_path = out_dir / "coverage_profile.txtfmt"

    percent_cmd = [go_bin, "tool", "covdata", "percent", "-i", str(cov_dir)]
    percent_proc = subprocess.run(percent_cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)
    percent_output = (percent_proc.stdout or "") + (percent_proc.stderr or "")
    percent_path.write_text(percent_output, encoding="utf-8")

    filtered_path = None
    if percent_regex:
        regex = re.compile(percent_regex)
        filtered_lines = [line for line in percent_output.splitlines() if regex.search(line)]
        filtered_path = out_dir / "coverage_percent.filtered.txt"
        filtered_path.write_text("\n".join(filtered_lines) + ("\n" if filtered_lines else ""), encoding="utf-8")

    textfmt_cmd = [go_bin, "tool", "covdata", "textfmt", "-i", str(cov_dir), "-o", str(textfmt_path)]
    textfmt_proc = subprocess.run(textfmt_cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)

    summary.update(
        {
            "available": percent_proc.returncode == 0,
            "percent_cmd": " ".join(percent_cmd),
            "percent_path": str(percent_path),
            "percent_returncode": percent_proc.returncode,
            "textfmt_cmd": " ".join(textfmt_cmd),
            "textfmt_path": str(textfmt_path),
            "textfmt_returncode": textfmt_proc.returncode,
            "filtered_percent_path": str(filtered_path) if filtered_path else None,
        }
    )
    if percent_proc.returncode != 0:
        summary["detail"] = percent_output.strip() or "covdata percent failed"
    elif textfmt_proc.returncode != 0:
        summary["detail"] = (textfmt_proc.stdout or textfmt_proc.stderr or "").strip() or "covdata textfmt failed"
    else:
        summary["detail"] = "covdata export complete"
    return summary


def write_summary_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Weaviate fuzz matrix against a coverage-enabled server")
    parser.add_argument("--suite", choices=sorted(SUITES.keys()), default="scalar", help="Preset coverage matrix")
    parser.add_argument("--launcher", default=str(DEFAULT_LAUNCHER), help="Server launcher script path")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for fuzz cases")
    parser.add_argument("--go-bin", default="go", help="Go toolchain binary for covdata export")
    parser.add_argument("--host", default="127.0.0.1", help="Weaviate host to wait on")
    parser.add_argument("--port", type=int, default=8080, help="Weaviate HTTP port")
    parser.add_argument("--run-id", default=None, help="Run id used for data/.cov directories")
    parser.add_argument("--cov-dir", default=None, help="Explicit coverage dir; useful with --reuse-running-server")
    parser.add_argument("--ready-timeout", type=float, default=90.0, help="Seconds to wait for Weaviate readiness")
    parser.add_argument("--query-page-size", type=int, default=1000, help="Pass-through query page size for fuzz cases")
    parser.add_argument("--stop-on-fail", action="store_true", help="Abort the suite on first failed fuzz case")
    parser.add_argument("--max-cases", type=int, default=None, help="Run only the first N expanded cases")
    parser.add_argument("--dry-run", action="store_true", help="Print the case matrix without starting Weaviate")
    parser.add_argument(
        "--reuse-running-server",
        action="store_true",
        help="Attach to an already-running coverage-enabled Weaviate instead of launching one",
    )
    parser.add_argument(
        "--coverage-percent-regex",
        default=None,
        help="Optional regex to keep only matching `covdata percent` lines in a filtered report",
    )
    args = parser.parse_args()

    launcher = Path(args.launcher).expanduser().resolve()
    if not launcher.exists():
        print(f"{RED}Launcher not found:{RESET} {launcher}")
        return 2
    if not ORACLE_PATH.exists():
        print(f"{RED}Oracle script not found:{RESET} {ORACLE_PATH}")
        return 2

    cases = expand_templates(SUITES[args.suite])
    if args.max_cases is not None:
        cases = cases[: max(0, args.max_cases)]
    if not cases:
        print(f"{YELLOW}No cases selected.{RESET}")
        return 0

    run_id = args.run_id or f"weaviate-cov-{args.suite}-{now_utc_stamp()}"
    log_dir = DEFAULT_LOG_ROOT / f"coverage_suite_{run_id}"
    cov_dir = Path(args.cov_dir).expanduser().resolve() if args.cov_dir else (DEFAULT_COV_ROOT / run_id)
    artifacts_dir = log_dir / "artifacts"

    print(f"\n{BOLD}{CYAN}{'=' * 76}{RESET}")
    print(f"{BOLD}{CYAN}  Weaviate Coverage Fuzz Suite{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 76}{RESET}")
    print(f"  Suite:      {args.suite}")
    print(f"  Launcher:   {launcher}")
    print(f"  Oracle:     {ORACLE_PATH}")
    print(f"  Run ID:     {run_id}")
    print(f"  Log dir:    {log_dir}")
    print(f"  Cov dir:    {cov_dir}")
    print(f"  Cases:      {len(cases)}")
    print(f"  Host:Port:  {args.host}:{args.port}")
    print(f"  Server:     {'reuse existing' if args.reuse_running_server else 'launch new'}")
    print(f"{CYAN}{'-' * 76}{RESET}")

    for index, case in enumerate(cases, start=1):
        seed_label = "random" if case.seed is None else str(case.seed)
        cmd = build_case_command(args.python_bin, case, args.query_page_size)
        print(
            f"  [{index:02d}] {case.name:<22} mode={case.mode:<7} "
            f"seed={seed_label:<6} rounds={case.rounds:<4} N={case.rows:<6} cmd={' '.join(cmd)}"
        )

    if args.dry_run:
        print(f"\n{YELLOW}Dry run only; nothing executed.{RESET}")
        return 0

    if args.reuse_running_server:
        if not port_ready(args.host, args.port, args.ready_timeout):
            print(
                f"{RED}No ready Weaviate found at {args.host}:{args.port}.{RESET}\n"
                f"Start your coverage-enabled server first, then rerun with --reuse-running-server."
            )
            return 2
    else:
        if port_in_use(args.host, args.port):
            print(
                f"{RED}Port {args.port} on {args.host} is already in use.{RESET}\n"
                f"Stop the existing Weaviate instance first, or rerun with --reuse-running-server."
            )
            return 2

    server_proc: subprocess.Popen | None = None
    results: list[CaseResult] = []
    suite_start = time.time()
    launched_server = False

    try:
        if args.reuse_running_server:
            print(f"\n{CYAN}Reusing running Weaviate at {args.host}:{args.port}.{RESET}")
        else:
            print(f"\n{CYAN}Starting coverage-enabled Weaviate...{RESET}")
            server_proc = start_server(launcher, run_id, args.host, args.port, args.ready_timeout)
            launched_server = True
            print(f"{GREEN}Weaviate is ready.{RESET}")

        for index, case in enumerate(cases, start=1):
            cmd = build_case_command(args.python_bin, case, args.query_page_size)
            seed_label = "random" if case.seed is None else str(case.seed)
            log_name = f"{index:02d}_{case.name}_seed_{seed_label}.log"
            log_path = log_dir / log_name

            print(
                f"[{index}/{len(cases)}] {case.name} "
                f"(mode={case.mode}, seed={seed_label}, rounds={case.rounds}, N={case.rows}) ..."
            )
            exit_code, elapsed = run_case(cmd, log_path)
            passed, detail = analyze_log(log_path, case.mode)
            result = CaseResult(
                name=case.name,
                mode=case.mode,
                seed=case.seed,
                rows=case.rows,
                rounds=case.rounds,
                command=cmd,
                log_path=str(log_path),
                exit_code=exit_code,
                passed=(exit_code == 0 and passed),
                detail=detail if exit_code == 0 else f"exit={exit_code}, {detail}",
                elapsed_seconds=elapsed,
            )
            results.append(result)

            status = colorize(result.passed, "PASS" if result.passed else "FAIL")
            print(f"    -> {status} ({fmt_duration(elapsed)}) {result.detail}")
            if not result.passed and args.stop_on_fail:
                print(f"{RED}Stopping on first failure as requested.{RESET}")
                break
    finally:
        if launched_server:
            stop_server(server_proc)

    if args.reuse_running_server and not launched_server:
        cov_summary = {
            "available": False,
            "detail": (
                "reused running server; coverage files may remain incomplete until that process exits. "
                "Run `go tool covdata ...` after stopping the server, or rerun without --reuse-running-server."
            ),
        }
    else:
        cov_summary = collect_covdata(args.go_bin, cov_dir, artifacts_dir, args.coverage_percent_regex)
    suite_elapsed = time.time() - suite_start
    passed_count = sum(1 for item in results if item.passed)
    failed = [item for item in results if not item.passed]

    payload = {
        "suite": args.suite,
        "run_id": run_id,
        "launcher": str(launcher),
        "oracle_path": str(ORACLE_PATH),
        "host": args.host,
        "port": args.port,
        "query_page_size": args.query_page_size,
        "suite_elapsed_seconds": suite_elapsed,
        "results": [asdict(item) for item in results],
        "coverage": cov_summary,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "suite_summary.json"
    write_summary_json(summary_path, payload)

    print(f"\n{BOLD}{'=' * 76}{RESET}")
    print(f"{BOLD}Suite Summary{RESET}")
    print(f"{'=' * 76}")
    print(f"  Total cases: {len(results)}")
    print(f"  Passed:      {GREEN}{passed_count}{RESET}")
    print(f"  Failed:      {RED}{len(failed)}{RESET}")
    print(f"  Duration:    {fmt_duration(suite_elapsed)}")
    print(f"  Logs:        {log_dir}")
    print(f"  Summary:     {summary_path}")
    if cov_summary.get("available"):
        print(f"  Coverage:    {cov_summary.get('percent_path')}")
        if cov_summary.get("filtered_percent_path"):
            print(f"  Cov filter:  {cov_summary.get('filtered_percent_path')}")
        print(f"  Textfmt:     {cov_summary.get('textfmt_path')}")
    else:
        print(f"  Coverage:    skipped ({cov_summary.get('detail', 'unknown reason')})")

    if failed:
        print(f"\n{RED}{BOLD}Failed cases:{RESET}")
        for item in failed:
            print(f"  - {item.name} seed={item.seed} -> {item.detail}")
            snippet = extract_summary_snippet(Path(item.log_path), lines=8)
            if snippet:
                print(snippet)
                print("-" * 40)
        return 1

    print(f"\n{GREEN}{BOLD}🎉 All cases passed.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
