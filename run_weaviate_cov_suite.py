#!/usr/bin/env python3
"""
Weaviate coverage-oriented fuzz scheduler.

Boots a coverage-enabled Weaviate instance once, runs a matrix of fuzz jobs
against it, then emits a suite summary plus `go tool covdata` reports.
The default matrix emphasizes scalar-filter coverage rather than vector-only paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
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
VECTOR_FUZZ_PATH = SCRIPT_DIR / "weaviate_vector_fuzz_test.py"
DEFAULT_LAUNCHER = SCRIPT_DIR / "start_weaviate_cov.sh"
DEFAULT_LOG_ROOT = SCRIPT_DIR / "weaviate_log"
DEFAULT_COV_ROOT = SCRIPT_DIR / ".cov"
DEFAULT_DATA_ROOT = SCRIPT_DIR / "data"
DEFAULT_SCALAR_PERCENT_REGEX = (
    r"adapters/repos/db$|adapters/repos/db/(aggregator|inverted)|"
    r"adapters/handlers/rest/filterext|"
    r"adapters/handlers/graphql/local/(aggregate|common_filters)|"
    r"entities/(filters|inverted)"
)
DEFAULT_SCALAR_FUNC_REGEX = (
    r"adapters/repos/db/(aggregator|inverted)/|"
    r"adapters/handlers/rest/filterext/|"
    r"adapters/handlers/graphql/local/(aggregate|common_filters)/|"
    r"entities/(filters|inverted)/|^total:"
)
DEFAULT_SCALAR_FILE_REGEX = (
    r"adapters/repos/db/(aggregator|inverted)/|"
    r"adapters/repos/db/[^/]*(aggregate|bitmap|filter|inverted|prop|search|shard)[^/]*\.go|"
    r"adapters/handlers/rest/filterext/|"
    r"adapters/handlers/graphql/local/(aggregate|common_filters)/|"
    r"entities/(filters|inverted)/"
)

KEY_MODULE_FILE_REGEXES = [
    ("REST filter parser", r"adapters/handlers/rest/filterext/"),
    ("GraphQL aggregate/common filters", r"adapters/handlers/graphql/local/(aggregate|common_filters)/"),
    ("Filter entities", r"entities/filters/"),
    ("Inverted entities", r"entities/inverted/"),
    ("DB aggregator", r"adapters/repos/db/aggregator/"),
    ("DB inverted index", r"adapters/repos/db/inverted/"),
    ("DB scalar search path", r"adapters/repos/db/[^/]*(aggregate|bitmap|filter|inverted|prop|search|shard)[^/]*\.go"),
]


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


STANDARD_SCALAR_PROFILE_ARGS = ["--scalar-depth-profile", "default"]
SMOKE_SCALAR_PROFILE_ARGS = ["--scalar-depth-profile", "coverage-smoke"]
REST_XCHECK_PROFILE_ARGS = ["--scalar-depth-profile", "rest-xcheck"]
SCALAR_DEEP_PROFILE_ARGS = ["--scalar-depth-profile", "scalar-deep"]
SCALAR_DEEP_RELAXED_PROFILE_ARGS = ["--scalar-depth-profile", "scalar-deep-relaxed"]


SCALAR_SMOKE_TEMPLATES = [
    CaseTemplate("oracle-nodyn-smoke", "oracle", 12, [1238], 1200, ["--no-dynamic", *SMOKE_SCALAR_PROFILE_ARGS]),
    CaseTemplate("oracle-dyn-smoke", "oracle", 18, [2238], 1200, [*SMOKE_SCALAR_PROFILE_ARGS]),
    CaseTemplate("oracle-inverted-smoke", "oracle", 16, [1838], 1400, ["--no-dynamic", "--profile", "inverted", *SMOKE_SCALAR_PROFILE_ARGS]),
    CaseTemplate("graphql-probe-smoke", "graphql-probe", 8, [2638], 400, []),
    CaseTemplate("equiv-smoke", "equiv", 10, [3238], 1000, []),
    CaseTemplate("pqs-smoke", "pqs", 12, [4238], 1000, []),
    CaseTemplate("aggregate-smoke", "aggregate", 12, [5238], 1200, [*SMOKE_SCALAR_PROFILE_ARGS]),
    CaseTemplate("aggregate-inverted-smoke", "aggregate", 14, [5838], 1400, ["--profile", "inverted", *SMOKE_SCALAR_PROFILE_ARGS]),
    CaseTemplate(
        "rest-filter-smoke",
        "rest-filter",
        12,
        [6238],
        1200,
        [*SMOKE_SCALAR_PROFILE_ARGS],
    ),
    CaseTemplate(
        "rest-filter-inverted-smoke",
        "rest-filter",
        14,
        [6838],
        1400,
        ["--profile", "inverted", *SMOKE_SCALAR_PROFILE_ARGS],
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
            *REST_XCHECK_PROFILE_ARGS,
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
            *REST_XCHECK_PROFILE_ARGS,
        ],
    ),
]

SCALAR_TEMPLATES = [
    CaseTemplate("oracle-scalar-nodyn-a", "oracle", 35, [1238, 2027], 2500, ["--no-dynamic", *STANDARD_SCALAR_PROFILE_ARGS]),
    CaseTemplate("oracle-scalar-nodyn-b", "oracle", 35, [3137, 4488], 2500, ["--no-dynamic", *STANDARD_SCALAR_PROFILE_ARGS]),
    CaseTemplate("oracle-inverted-a", "oracle", 36, [18381, 18382], 2600, ["--no-dynamic", "--profile", "inverted", *STANDARD_SCALAR_PROFILE_ARGS]),
    CaseTemplate("oracle-scalar-dyn-a", "oracle", 45, [5501, 6602], 2500, [*STANDARD_SCALAR_PROFILE_ARGS]),
    CaseTemplate("oracle-scalar-dyn-b", "oracle", 45, [7703], 4000, [*STANDARD_SCALAR_PROFILE_ARGS]),
    CaseTemplate("oracle-scalar-randcl", "oracle", 35, [8804], 2500, ["--random-consistency", *STANDARD_SCALAR_PROFILE_ARGS]),
    CaseTemplate("graphql-probe-a", "graphql-probe", 12, [9031, 9032], 600, []),
    CaseTemplate("equiv-scalar-a", "equiv", 20, [9101, 9102], 1800, []),
    CaseTemplate("pqs-scalar-a", "pqs", 25, [9201, 9202], 1800, []),
    CaseTemplate("aggregate-scalar-a", "aggregate", 30, [9301, 9302], 2200, [*STANDARD_SCALAR_PROFILE_ARGS]),
    CaseTemplate("aggregate-inverted-a", "aggregate", 28, [9331, 9332], 2400, ["--profile", "inverted", *STANDARD_SCALAR_PROFILE_ARGS]),
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
            *REST_XCHECK_PROFILE_ARGS,
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
            *REST_XCHECK_PROFILE_ARGS,
        ],
    ),
    CaseTemplate(
        "rest-filter-scalar-a",
        "rest-filter",
        24,
        [9501, 9502],
        2200,
        [*STANDARD_SCALAR_PROFILE_ARGS],
    ),
    CaseTemplate(
        "rest-filter-inverted-a",
        "rest-filter",
        26,
        [9551, 9552],
        2400,
        ["--profile", "inverted", *STANDARD_SCALAR_PROFILE_ARGS],
    ),
]

FULL_TEMPLATES = SCALAR_TEMPLATES + [
    CaseTemplate("oracle-large-dyn", "oracle", 25, [10001], 12000, []),
    CaseTemplate("groupby-broad", "groupby", 18, [11001], 2000, []),
    CaseTemplate("aggregate-broad", "aggregate", 28, [12001], 5000, [*STANDARD_SCALAR_PROFILE_ARGS]),
    CaseTemplate(
        "rest-filter-broad",
        "rest-filter",
        24,
        [13001],
        4500,
        [*STANDARD_SCALAR_PROFILE_ARGS],
    ),
]

VECTOR_SMOKE_TEMPLATES = [
    CaseTemplate("vector-smoke-random", "vector", 36, [7838], 1200, ["--dim", "64", "--vector-index", "random"]),
    CaseTemplate("vector-smoke-dynamic", "vector", 40, [7938], 1200, ["--dim", "64", "--vector-index", "dynamic", "--dynamic"]),
]

VECTOR_TEMPLATES = [
    CaseTemplate("vector-hnsw-a", "vector", 72, [9811], 1800, ["--dim", "96", "--vector-index", "hnsw"]),
    CaseTemplate("vector-flat-a", "vector", 72, [9812], 1800, ["--dim", "96", "--vector-index", "flat"]),
    CaseTemplate("vector-hfresh-a", "vector", 64, [9813], 1800, ["--dim", "96", "--vector-index", "hfresh"]),
    CaseTemplate("vector-dynamic-a", "vector", 84, [9814], 2000, ["--dim", "128", "--vector-index", "dynamic", "--dynamic"]),
    CaseTemplate("vector-random-a", "vector", 84, [9815], 2000, ["--dim", "128", "--vector-index", "random", "--dynamic"]),
]

DEEP_TEMPLATES = [
    CaseTemplate(
        "oracle-scalar-dyn-deep-150",
        "oracle",
        150,
        [15001],
        3000,
        [*SCALAR_DEEP_PROFILE_ARGS],
    ),
    CaseTemplate(
        "oracle-inverted-dyn-deep-120",
        "oracle",
        120,
        [15002],
        3000,
        [
            "--profile",
            "inverted",
            *SCALAR_DEEP_PROFILE_ARGS,
        ],
    ),
    CaseTemplate(
        "oracle-randcl-dyn-deep-120",
        "oracle",
        120,
        [15003],
        3000,
        [
            "--random-consistency",
            *SCALAR_DEEP_RELAXED_PROFILE_ARGS,
        ],
    ),
    CaseTemplate(
        "rest-filter-scalar-deep-96",
        "rest-filter",
        96,
        [15101],
        3200,
        [*SCALAR_DEEP_PROFILE_ARGS],
    ),
    CaseTemplate(
        "rest-filter-inverted-deep-96",
        "rest-filter",
        96,
        [15102],
        3200,
        [
            "--profile",
            "inverted",
            *SCALAR_DEEP_PROFILE_ARGS,
        ],
    ),
    CaseTemplate(
        "aggregate-scalar-deep-72",
        "aggregate",
        72,
        [15201],
        4200,
        [*SCALAR_DEEP_PROFILE_ARGS],
    ),
    CaseTemplate(
        "vector-dynamic-deep-150",
        "vector",
        150,
        [15301],
        2200,
        ["--dim", "128", "--vector-index", "dynamic", "--dynamic"],
    ),
    CaseTemplate(
        "vector-hfresh-deep-120",
        "vector",
        120,
        [15302],
        2000,
        ["--dim", "128", "--vector-index", "hfresh"],
    ),
]

COMPREHENSIVE_TEMPLATES = [
    *SCALAR_SMOKE_TEMPLATES,
    *FULL_TEMPLATES,
    *VECTOR_SMOKE_TEMPLATES,
    *VECTOR_TEMPLATES,
]

COMPREHENSIVE_DEEP_TEMPLATES = [
    *COMPREHENSIVE_TEMPLATES,
    *DEEP_TEMPLATES,
]

AGGREGATE_SUITE_TEMPLATES = [
    *[case for case in SCALAR_SMOKE_TEMPLATES if case.mode in {"aggregate", "graphql-probe"}],
    *[case for case in SCALAR_TEMPLATES if case.mode in {"aggregate", "graphql-probe"}],
    *[case for case in FULL_TEMPLATES[len(SCALAR_TEMPLATES):] if case.mode in {"aggregate", "groupby"}],
]

SUITES = {
    "scalar-smoke": SCALAR_SMOKE_TEMPLATES,
    "scalar": SCALAR_TEMPLATES,
    "inverted": [case for case in [*SCALAR_SMOKE_TEMPLATES, *SCALAR_TEMPLATES] if "inverted" in case.name],
    "aggregate": AGGREGATE_SUITE_TEMPLATES,
    "full": FULL_TEMPLATES,
    "vector-smoke": VECTOR_SMOKE_TEMPLATES,
    "vector": VECTOR_TEMPLATES,
    "deep": DEEP_TEMPLATES,
    "comprehensive": COMPREHENSIVE_TEMPLATES,
    "comprehensive-deep": COMPREHENSIVE_DEEP_TEMPLATES,
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


def derive_case_seed(suite_seed: int, case: SuiteCase, case_index: int) -> int:
    payload = (
        f"{suite_seed}|{case_index}|{case.name}|{case.mode}|"
        f"{case.rounds}|{case.rows}|{case.seed}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)
    return seed or 1


def apply_seed_strategy(cases: list[SuiteCase], seed_strategy: str, suite_seed: int | None) -> list[SuiteCase]:
    if seed_strategy == "fixed":
        return cases
    if suite_seed is None:
        raise ValueError("--suite-seed is required when --seed-strategy=derived")

    derived_cases: list[SuiteCase] = []
    for index, case in enumerate(cases, start=1):
        derived_cases.append(
            SuiteCase(
                name=case.name,
                mode=case.mode,
                rounds=case.rounds,
                seed=derive_case_seed(suite_seed, case, index),
                rows=case.rows,
                extra_args=list(case.extra_args),
            )
        )
    return derived_cases


def select_budget_base_case(
    cases: list[SuiteCase],
    suite_seed: int,
    global_index: int,
    budget_schedule: str,
) -> SuiteCase:
    if budget_schedule == "cycle":
        return cases[(global_index - 1) % len(cases)]
    if budget_schedule == "random":
        payload = f"{suite_seed}|budget-random-case|{global_index}|{len(cases)}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        case_index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(cases)
        return cases[case_index]
    raise ValueError(f"unsupported budget schedule: {budget_schedule}")


def materialize_budget_case(case: SuiteCase, suite_seed: int, global_index: int) -> SuiteCase:
    return SuiteCase(
        name=case.name,
        mode=case.mode,
        rounds=case.rounds,
        seed=derive_case_seed(suite_seed, case, global_index),
        rows=case.rows,
        extra_args=list(case.extra_args),
    )


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


def shell_join(command: list[str]) -> str:
    return shlex.join([str(part) for part in command])


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return label or "case"


def build_case_command(
    python_bin: str,
    case: SuiteCase,
    query_page_size: int | None,
    host: str,
    port: int,
    grpc_port: int,
    case_log_dir: Path | None = None,
    log_suffix: str | None = None,
    robustness_rate: float | None = None,
    robustness_start_fraction: float | None = None,
    robustness_frontdoors: str | None = None,
    robustness_category_mode: str | None = None,
) -> list[str]:
    if case.mode == "vector":
        cmd = [
            python_bin,
            str(VECTOR_FUZZ_PATH),
            "--rounds",
            str(case.rounds),
            "--host",
            host,
            "--port",
            str(port),
            "--grpc-port",
            str(grpc_port),
            "-N",
            str(case.rows),
        ]
    else:
        cmd = [
            python_bin,
            str(ORACLE_PATH),
            "--mode",
            case.mode,
            "--rounds",
            str(case.rounds),
            "--host",
            host,
            "--port",
            str(port),
            "--grpc-port",
            str(grpc_port),
            "-N",
            str(case.rows),
        ]
    if case.seed is not None:
        cmd += ["--seed", str(case.seed)]
    if case.mode != "vector" and query_page_size is not None:
        cmd += ["--query-page-size", str(query_page_size)]
    if case_log_dir is not None:
        cmd += ["--log-dir", str(case_log_dir)]
    if log_suffix:
        cmd += ["--log-suffix", str(log_suffix)]
    if case.mode in {"oracle", "rest-filter"} and robustness_rate is not None and robustness_rate > 0.0:
        cmd += ["--robustness-rate", str(robustness_rate)]
        if robustness_start_fraction is not None:
            cmd += ["--robustness-start-fraction", str(robustness_start_fraction)]
        if robustness_frontdoors:
            cmd += ["--robustness-frontdoors", str(robustness_frontdoors)]
        if robustness_category_mode:
            cmd += ["--robustness-category-mode", str(robustness_category_mode)]
    cmd.extend(case.extra_args)
    return cmd


def port_ready(host: str, port: int, timeout_seconds: float) -> bool:
    ready_url = f"http://{host}:{port}/v1/.well-known/ready"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(ready_url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        except TimeoutError:
            pass
        time.sleep(1.0)
    return False


def port_in_use(host: str, port: int) -> bool:
    return port_ready(host, port, timeout_seconds=1.5)


def start_server(
    launcher: Path,
    run_id: str,
    host: str,
    port: int,
    grpc_port: int,
    ready_timeout: float,
    cov_dir: Path,
    data_dir: Path,
    server_log_path: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["WEAVIATE_COV_DIR"] = str(cov_dir)
    env["WEAVIATE_DATA_DIR"] = str(data_dir)
    env["WEAVIATE_HOST"] = "0.0.0.0"
    env["WEAVIATE_PORT"] = str(port)
    env["WEAVIATE_GRPC_PORT"] = str(grpc_port)
    env["WEAVIATE_RAFT_PORT"] = str(port + 220)
    env["WEAVIATE_RAFT_INTERNAL_RPC_PORT"] = str(port + 221)
    env["CLUSTER_HOSTNAME"] = f"node-{safe_label(run_id)}"
    env["CLUSTER_GOSSIP_BIND_PORT"] = str(port + 222)
    env["CLUSTER_DATA_BIND_PORT"] = str(port + 223)
    env["CLUSTER_ADVERTISE_PORT"] = str(port + 222)
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    server_log = server_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", str(launcher), run_id],
        cwd=str(SCRIPT_DIR),
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
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
    vector_failures = re.search(r"Total failures:\s*(\d+)", content)
    if vector_failures is not None:
        failures = int(vector_failures.group(1))
        if failures > 0:
            return False, f"vector failures={failures}"
        if mode == "vector":
            return True, "vector summary indicates pass"
    if "failure(s) detected" in content:
        return False, "vector failure marker"

    success_patterns = [
        "✅ All",
        "✅ All passed",
        "🎉 All tests passed!",
    ]
    if any(pattern in content for pattern in success_patterns):
        return True, "success marker found"

    if mode == "oracle" and "Stats:" in content and "Fail:0" in content:
        return True, "oracle stats indicate pass"
    if mode == "vector" and re.search(r"Total:\s+PASS=\d+\s+FAIL=0\b", content):
        return True, "vector totals indicate pass"

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


def resolve_go_cover_cwd() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("WEAVIATE_GO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            SCRIPT_DIR,
            SCRIPT_DIR.parent / "weaviate",
            Path.home() / "weaviate",
        ]
    )

    for candidate in candidates:
        go_mod = candidate / "go.mod"
        if not go_mod.is_file():
            continue
        try:
            content = go_mod.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "module github.com/weaviate/weaviate" in content:
            return candidate
    return SCRIPT_DIR


def parse_textfmt_coverage(path: Path, file_regex: str | None = None) -> dict:
    regex = re.compile(file_regex) if file_regex else None
    total_statements = 0
    covered_statements = 0
    total_blocks = 0
    covered_blocks = 0
    files: set[str] = set()
    block_re = re.compile(
        r"^(?P<file>.+?):\d+\.\d+,\d+\.\d+\s+(?P<statements>\d+)\s+(?P<count>\d+)$"
    )

    if not path.is_file():
        return {
            "available": False,
            "file_regex": file_regex,
            "detail": f"profile not found: {path}",
            "total_statements": 0,
            "covered_statements": 0,
            "coverage_percent": 0.0,
            "total_blocks": 0,
            "covered_blocks": 0,
            "block_coverage_percent": 0.0,
            "file_count": 0,
        }

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("mode:"):
            continue
        match = block_re.match(line)
        if match is None:
            continue
        filename = match.group("file")
        if regex is not None and not regex.search(filename):
            continue

        statements = int(match.group("statements"))
        count = int(match.group("count"))
        total_statements += statements
        total_blocks += 1
        files.add(filename)
        if count > 0:
            covered_statements += statements
            covered_blocks += 1

    statement_pct = (covered_statements / total_statements * 100.0) if total_statements else 0.0
    block_pct = (covered_blocks / total_blocks * 100.0) if total_blocks else 0.0
    return {
        "available": total_statements > 0,
        "file_regex": file_regex,
        "total_statements": total_statements,
        "covered_statements": covered_statements,
        "coverage_percent": round(statement_pct, 2),
        "total_blocks": total_blocks,
        "covered_blocks": covered_blocks,
        "block_coverage_percent": round(block_pct, 2),
        "file_count": len(files),
    }


def write_coverage_metrics_artifacts(out_dir: Path, metrics: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "coverage_metrics.json"
    csv_path = out_dir / "coverage_metrics.csv"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "scope",
                "coverage_percent",
                "covered_statements",
                "total_statements",
                "block_coverage_percent",
                "covered_blocks",
                "total_blocks",
                "file_count",
                "file_regex",
            ]
        )
        rows = [("overall", metrics.get("overall", {})), ("scalar_target", metrics.get("scalar_target", {}))]
        rows.extend((f"module:{name}", item) for name, item in metrics.get("key_modules", {}).items())
        for scope, item in rows:
            writer.writerow(
                [
                    scope,
                    item.get("coverage_percent", 0.0),
                    item.get("covered_statements", 0),
                    item.get("total_statements", 0),
                    item.get("block_coverage_percent", 0.0),
                    item.get("covered_blocks", 0),
                    item.get("total_blocks", 0),
                    item.get("file_count", 0),
                    item.get("file_regex"),
                ]
            )
    return json_path, csv_path


def collect_coverage_metrics(textfmt_path: Path, out_dir: Path, scalar_file_regex: str | None) -> dict:
    metrics = {
        "definition": {
            "overall": "All Go coverage blocks in coverage_profile.txtfmt.",
            "scalar_target": "Statement-weighted coverage over files matching scalar_file_regex.",
            "key_modules": "Statement-weighted coverage over fixed filter/inverted/aggregate module groups.",
        },
        "overall": parse_textfmt_coverage(textfmt_path),
        "scalar_target": parse_textfmt_coverage(textfmt_path, scalar_file_regex),
        "key_modules": {
            name: parse_textfmt_coverage(textfmt_path, regex)
            for name, regex in KEY_MODULE_FILE_REGEXES
        },
    }
    json_path, csv_path = write_coverage_metrics_artifacts(out_dir, metrics)
    metrics["metrics_json_path"] = str(json_path)
    metrics["metrics_csv_path"] = str(csv_path)
    return metrics


def write_coverage_timeline_artifacts(log_dir: Path, rows: list[dict]) -> dict:
    artifacts_dir = log_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / "coverage_timeline.csv"
    jsonl_path = artifacts_dir / "coverage_timeline.jsonl"
    fieldnames = [
        "case_index",
        "cycle",
        "case_name",
        "mode",
        "seed",
        "rows",
        "rounds",
        "passed",
        "case_elapsed_seconds",
        "suite_elapsed_seconds",
        "overall_coverage_percent",
        "overall_covered_statements",
        "overall_total_statements",
        "scalar_coverage_percent",
        "scalar_covered_statements",
        "scalar_total_statements",
        "snapshot_dir",
        "cov_inputs",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "enabled": True,
        "snapshots": len(rows),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }


def build_timeline_row(
    *,
    case_index: int,
    cycle: int,
    case: SuiteCase,
    result: CaseResult,
    suite_start: float,
    snapshot_dir: Path,
    cov_inputs: str,
    cov_summary: dict,
) -> dict:
    metrics = cov_summary.get("metrics") if isinstance(cov_summary.get("metrics"), dict) else {}
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    scalar = metrics.get("scalar_target", {}) if isinstance(metrics, dict) else {}
    return {
        "case_index": case_index,
        "cycle": cycle,
        "case_name": case.name,
        "mode": case.mode,
        "seed": case.seed,
        "rows": case.rows,
        "rounds": case.rounds,
        "passed": result.passed,
        "case_elapsed_seconds": round(result.elapsed_seconds, 3),
        "suite_elapsed_seconds": round(time.time() - suite_start, 3),
        "overall_coverage_percent": overall.get("coverage_percent", 0.0),
        "overall_covered_statements": overall.get("covered_statements", 0),
        "overall_total_statements": overall.get("total_statements", 0),
        "scalar_coverage_percent": scalar.get("coverage_percent", 0.0),
        "scalar_covered_statements": scalar.get("covered_statements", 0),
        "scalar_total_statements": scalar.get("total_statements", 0),
        "snapshot_dir": str(snapshot_dir),
        "cov_inputs": cov_inputs,
        "coverage_available": bool(cov_summary.get("available")),
        "coverage_detail": cov_summary.get("detail", ""),
    }


def build_timeline_baseline_row(*, suite_seed: int | None) -> dict:
    return {
        "case_index": 0,
        "cycle": 0,
        "case_name": "baseline",
        "mode": "baseline",
        "seed": suite_seed if suite_seed is not None else "",
        "rows": 0,
        "rounds": 0,
        "passed": True,
        "case_elapsed_seconds": 0.0,
        "suite_elapsed_seconds": 0.0,
        "overall_coverage_percent": 0.0,
        "overall_covered_statements": 0,
        "overall_total_statements": 0,
        "scalar_coverage_percent": 0.0,
        "scalar_covered_statements": 0,
        "scalar_total_statements": 0,
        "snapshot_dir": "",
        "cov_inputs": "",
        "coverage_available": False,
        "coverage_detail": "synthetic zero baseline",
    }


def collect_covdata(
    go_bin: str,
    cov_inputs: str,
    out_dir: Path,
    percent_regex: str | None,
    func_regex: str | None,
    scalar_file_regex: str | None,
) -> dict:
    summary: dict[str, str | int | bool] = {"available": False}
    if shutil.which(go_bin) is None:
        summary["detail"] = f"`{go_bin}` not found; skipped covdata export"
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    percent_path = out_dir / "coverage_percent.txt"
    textfmt_path = out_dir / "coverage_profile.txtfmt"
    func_path = out_dir / "coverage_func.txt"
    cover_cwd = resolve_go_cover_cwd()

    percent_cmd = [go_bin, "tool", "covdata", "percent", "-i", cov_inputs]
    percent_proc = subprocess.run(percent_cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)
    percent_output = (percent_proc.stdout or "") + (percent_proc.stderr or "")
    percent_path.write_text(percent_output, encoding="utf-8")

    filtered_path = None
    if percent_regex:
        regex = re.compile(percent_regex)
        filtered_lines = [line for line in percent_output.splitlines() if regex.search(line)]
        filtered_path = out_dir / "coverage_percent.filtered.txt"
        filtered_path.write_text("\n".join(filtered_lines) + ("\n" if filtered_lines else ""), encoding="utf-8")

    textfmt_cmd = [go_bin, "tool", "covdata", "textfmt", "-i", cov_inputs, "-o", str(textfmt_path)]
    textfmt_proc = subprocess.run(textfmt_cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)
    func_cmd = [go_bin, "tool", "cover", "-func", str(textfmt_path)]
    func_proc = subprocess.run(func_cmd, cwd=str(cover_cwd), capture_output=True, text=True) if textfmt_proc.returncode == 0 else None
    func_output = ""
    filtered_func_path = None
    if func_proc is not None:
        func_output = (func_proc.stdout or "") + (func_proc.stderr or "")
        func_path.write_text(func_output, encoding="utf-8")
        if func_regex:
            regex = re.compile(func_regex)
            filtered_lines = [line for line in func_output.splitlines() if regex.search(line)]
            filtered_func_path = out_dir / "coverage_func.filtered.txt"
            filtered_func_path.write_text("\n".join(filtered_lines) + ("\n" if filtered_lines else ""), encoding="utf-8")

    summary.update(
        {
            "available": percent_proc.returncode == 0,
            "percent_cmd": " ".join(percent_cmd),
            "percent_path": str(percent_path),
            "percent_returncode": percent_proc.returncode,
            "textfmt_cmd": " ".join(textfmt_cmd),
            "textfmt_path": str(textfmt_path),
            "textfmt_returncode": textfmt_proc.returncode,
            "func_cmd": " ".join(func_cmd),
            "func_cwd": str(cover_cwd),
            "func_path": str(func_path) if func_proc is not None else None,
            "func_returncode": func_proc.returncode if func_proc is not None else None,
            "filtered_percent_path": str(filtered_path) if filtered_path else None,
            "filtered_func_path": str(filtered_func_path) if filtered_func_path else None,
        }
    )
    if percent_proc.returncode != 0:
        summary["detail"] = percent_output.strip() or "covdata percent failed"
    elif textfmt_proc.returncode != 0:
        summary["detail"] = (textfmt_proc.stdout or textfmt_proc.stderr or "").strip() or "covdata textfmt failed"
    elif func_proc is not None and func_proc.returncode != 0:
        summary["detail"] = func_output.strip() or "go tool cover -func failed"
    else:
        summary["detail"] = "covdata export complete"
        summary["metrics"] = collect_coverage_metrics(textfmt_path, out_dir, scalar_file_regex)
    return summary


def write_summary_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_coverage_rows(filtered_percent_path: str | None) -> list[tuple[str, str]]:
    if not filtered_percent_path:
        return []
    path = Path(filtered_percent_path)
    if not path.is_file():
        return []
    rows: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or "coverage:" not in line:
            continue
        package, _, tail = line.partition("coverage:")
        rows.append((package.strip(), tail.strip()))
    return rows


def write_case_results_csv(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["name", "mode", "seed", "rows", "rounds", "passed", "elapsed_seconds", "detail", "log_path", "command"])
        for item in results:
            writer.writerow(
                [
                    item.name,
                    item.mode,
                    item.seed,
                    item.rows,
                    item.rounds,
                    item.passed,
                    f"{item.elapsed_seconds:.3f}",
                    item.detail,
                    item.log_path,
                    shell_join(item.command),
                ]
            )


def write_failure_artifacts(log_dir: Path, results: list[CaseResult]) -> dict:
    failed = [item for item in results if not item.passed]
    jsonl_path = log_dir / "failure_repros.jsonl"
    md_path = log_dir / "failure_repros.md"

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for item in failed:
            payload = asdict(item)
            payload["reproduce"] = shell_join(item.command)
            fout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    lines = ["# Failed Weaviate Fuzz Cases", ""]
    if not failed:
        lines.append("No failed cases.")
    else:
        for index, item in enumerate(failed, start=1):
            lines.append(f"## {index}. {item.name}")
            lines.append("")
            lines.append(f"- Mode: `{item.mode}`")
            lines.append(f"- Seed: `{item.seed}`")
            lines.append(f"- Detail: `{item.detail}`")
            lines.append(f"- Log: `{item.log_path}`")
            lines.append("")
            lines.append("```bash")
            lines.append(shell_join(item.command))
            lines.append("```")
            lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "failed_count": len(failed),
        "jsonl_path": str(jsonl_path),
        "markdown_path": str(md_path),
    }


def write_summary_markdown(
    path: Path,
    *,
    payload: dict,
    results: list[CaseResult],
    coverage_rows: list[tuple[str, str]],
    coverage_metrics: dict | None,
    summary_json_path: Path,
    cases_csv_path: Path | None,
) -> None:
    lines: list[str] = []
    lines.append("# Weaviate Coverage Fuzz Summary")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- Run ID: `{payload.get('run_id')}`")
    if payload.get("suite"):
        lines.append(f"- Suite: `{payload.get('suite')}`")
    if payload.get("seed_strategy"):
        lines.append(f"- Seed strategy: `{payload.get('seed_strategy')}`")
    if payload.get("suite_seed") is not None:
        lines.append(f"- Suite seed: `{payload.get('suite_seed')}`")
    if payload.get("time_budget_seconds") is not None:
        lines.append(f"- Time budget: `{fmt_duration(float(payload.get('time_budget_seconds')))}`")
        lines.append(f"- Budget schedule: `{payload.get('budget_schedule', 'cycle')}`")
    lines.append(f"- Summary JSON: `{summary_json_path}`")
    if cases_csv_path is not None:
        lines.append(f"- Case CSV: `{cases_csv_path}`")
    failure_artifacts = payload.get("failure_artifacts") or {}
    if failure_artifacts.get("jsonl_path"):
        lines.append(f"- Failure repros: `{failure_artifacts.get('jsonl_path')}`")
    coverage_timeline = payload.get("coverage_timeline") or {}
    if coverage_timeline.get("csv_path"):
        lines.append(f"- Coverage timeline: `{coverage_timeline.get('csv_path')}`")
    lines.append("")

    if results:
        passed = sum(1 for item in results if item.passed)
        failed = len(results) - passed
        lines.append("## Cases")
        lines.append("")
        lines.append(f"- Total: `{len(results)}`")
        lines.append(f"- Passed: `{passed}`")
        lines.append(f"- Failed: `{failed}`")
        lines.append("")
        lines.append("| # | Name | Mode | Seed | Rows | Rounds | Pass | Time(s) |")
        lines.append("|---:|---|---|---:|---:|---:|:---:|---:|")
        for index, item in enumerate(results, start=1):
            lines.append(
                f"| {index} | {item.name} | {item.mode} | {item.seed} | "
                f"{item.rows} | {item.rounds} | {'Y' if item.passed else 'N'} | {item.elapsed_seconds:.1f} |"
            )
        lines.append("")

    if coverage_rows:
        lines.append("## Coverage")
        lines.append("")
        lines.append("| Package | Coverage |")
        lines.append("|---|---:|")
        for package, percent in coverage_rows:
            lines.append(f"| `{package}` | {percent} |")
        lines.append("")

    if coverage_metrics:
        overall = coverage_metrics.get("overall", {})
        scalar = coverage_metrics.get("scalar_target", {})
        lines.append("## Statement Coverage Metrics")
        lines.append("")
        lines.append("| Scope | Coverage | Covered / Total Statements | Files |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| Overall | {overall.get('coverage_percent', 0.0):.2f}% | "
            f"{overall.get('covered_statements', 0)} / {overall.get('total_statements', 0)} | "
            f"{overall.get('file_count', 0)} |"
        )
        lines.append(
            f"| Scalar target | {scalar.get('coverage_percent', 0.0):.2f}% | "
            f"{scalar.get('covered_statements', 0)} / {scalar.get('total_statements', 0)} | "
            f"{scalar.get('file_count', 0)} |"
        )
        for name, item in coverage_metrics.get("key_modules", {}).items():
            lines.append(
                f"| {name} | {item.get('coverage_percent', 0.0):.2f}% | "
                f"{item.get('covered_statements', 0)} / {item.get('total_statements', 0)} | "
                f"{item.get('file_count', 0)} |"
            )
        lines.append("")

    if coverage_timeline.get("csv_path"):
        lines.append("## Coverage Timeline")
        lines.append("")
        lines.append(f"- Snapshots: `{coverage_timeline.get('snapshots', 0)}`")
        lines.append(f"- CSV: `{coverage_timeline.get('csv_path')}`")
        lines.append(f"- JSONL: `{coverage_timeline.get('jsonl_path')}`")
        lines.append("- Timeline includes a synthetic zero-coverage baseline row before the first completed case.")
        lines.append("- The first non-baseline point is cumulative coverage after case #1, not an initial empty-server measurement.")
        lines.append(
            "- Note: timeline mode restarts the coverage-enabled server per case so Go coverage counters are flushed "
            "before each cumulative snapshot."
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_cov_inputs(raw_inputs: str | None, fallback_cov_dir: Path | None) -> str | None:
    parts: list[str] = []
    if raw_inputs:
        for item in raw_inputs.split(","):
            item = item.strip()
            if item:
                parts.append(str(Path(item).expanduser().resolve()))
    elif fallback_cov_dir is not None:
        parts.append(str(fallback_cov_dir.expanduser().resolve()))

    if not parts:
        return None
    return ",".join(parts)


def print_report_summary(
    log_dir: Path,
    summary_path: Path,
    cov_summary: dict,
    elapsed_seconds: float,
    title: str = "Coverage Report",
) -> int:
    print(f"\n{BOLD}{'=' * 76}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{'=' * 76}")
    print(f"  Duration:    {fmt_duration(elapsed_seconds)}")
    print(f"  Logs:        {log_dir}")
    print(f"  Summary:     {summary_path}")
    if cov_summary.get("available"):
        print(f"  Coverage:    {cov_summary.get('percent_path')}")
        if cov_summary.get("filtered_percent_path"):
            print(f"  Cov filter:  {cov_summary.get('filtered_percent_path')}")
        if cov_summary.get("func_path"):
            print(f"  Cover func:  {cov_summary.get('func_path')}")
        if cov_summary.get("filtered_func_path"):
            print(f"  Func filter: {cov_summary.get('filtered_func_path')}")
        print(f"  Textfmt:     {cov_summary.get('textfmt_path')}")
        metrics = cov_summary.get("metrics") if isinstance(cov_summary.get("metrics"), dict) else None
        if metrics:
            overall = metrics.get("overall", {})
            scalar = metrics.get("scalar_target", {})
            print(f"  Overall cov: {overall.get('coverage_percent', 0.0):.2f}%")
            print(f"  Scalar cov:  {scalar.get('coverage_percent', 0.0):.2f}%")
            print(f"  Metrics:     {metrics.get('metrics_json_path')}")
        print(f"\n{GREEN}{BOLD}Coverage report generated.{RESET}")
        return 0

    print(f"  Coverage:    failed ({cov_summary.get('detail', 'unknown reason')})")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Weaviate fuzz matrix against a coverage-enabled server")
    parser.add_argument("--suite", choices=sorted(SUITES.keys()), default="scalar", help="Preset coverage matrix")
    parser.add_argument(
        "--seed-strategy",
        choices=["fixed", "derived"],
        default="fixed",
        help="How case seeds are chosen: keep template seeds, or derive fresh reproducible seeds from --suite-seed",
    )
    parser.add_argument(
        "--suite-seed",
        type=int,
        default=None,
        help="Master seed used when --seed-strategy=derived",
    )
    parser.add_argument("--launcher", default=str(DEFAULT_LAUNCHER), help="Server launcher script path")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for fuzz cases")
    parser.add_argument("--go-bin", default="go", help="Go toolchain binary for covdata export")
    parser.add_argument("--host", default="127.0.0.1", help="Weaviate host to wait on")
    parser.add_argument("--port", type=int, default=8080, help="Weaviate HTTP port")
    parser.add_argument("--grpc-port", type=int, default=50051, help="Weaviate gRPC port for vector fuzz cases")
    parser.add_argument("--run-id", default=None, help="Run id used for data/.cov directories")
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT), help="Directory that receives suite logs and artifacts")
    parser.add_argument("--cov-root", default=str(DEFAULT_COV_ROOT), help="Directory that receives GOCOVERDIR runs by default")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Directory that receives Weaviate persistence data by default")
    parser.add_argument("--data-dir", default=None, help="Explicit Weaviate persistence data directory for this run")
    parser.add_argument("--cov-dir", default=None, help="Explicit coverage dir; useful with --reuse-running-server")
    parser.add_argument(
        "--coverage-inputs",
        default=None,
        help="Comma-separated GOCOVERDIR directories to analyze/merge for the coverage report",
    )
    parser.add_argument("--ready-timeout", type=float, default=90.0, help="Seconds to wait for Weaviate readiness")
    parser.add_argument("--query-page-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--robustness-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--robustness-start-fraction", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--robustness-frontdoors", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--robustness-category-mode",
        choices=["ignore", "warn", "strict"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--stop-on-fail", action="store_true", help="Abort the suite on first failed fuzz case")
    parser.add_argument("--start-case", type=int, default=1, help="1-based case index to start from after suite expansion")
    parser.add_argument("--max-cases", type=int, default=None, help="Run only the first N expanded cases")
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="Keep cycling the selected case matrix until this wall-clock budget is reached",
    )
    parser.add_argument(
        "--budget-schedule",
        choices=["cycle", "random"],
        default="cycle",
        help=(
            "Case selection policy used with --time-budget-seconds. `cycle` repeats the matrix in order; "
            "`random` chooses a reproducible pseudo-random case per completed slot from --suite-seed."
        ),
    )
    parser.add_argument(
        "--coverage-timeline",
        action="store_true",
        help=(
            "Collect cumulative coverage after completed cases. This restarts the coverage-enabled server per case "
            "so Go coverage counters are flushed for each snapshot."
        ),
    )
    parser.add_argument(
        "--coverage-timeline-interval-cases",
        type=int,
        default=1,
        help="Snapshot cumulative coverage every N completed cases when --coverage-timeline is enabled",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the case matrix without starting Weaviate")
    parser.add_argument(
        "--coverage-report-only",
        action="store_true",
        help="Skip fuzz execution and only generate a coverage report for --coverage-inputs/--cov-dir",
    )
    parser.add_argument(
        "--reuse-running-server",
        action="store_true",
        help="Attach to an already-running coverage-enabled Weaviate instead of launching one",
    )
    parser.add_argument(
        "--coverage-percent-regex",
        default=DEFAULT_SCALAR_PERCENT_REGEX,
        help="Optional regex to keep only matching `covdata percent` lines in a filtered report",
    )
    parser.add_argument(
        "--coverage-func-regex",
        default=DEFAULT_SCALAR_FUNC_REGEX,
        help="Optional regex to keep only matching `go tool cover -func` lines in a filtered report",
    )
    parser.add_argument(
        "--scalar-file-regex",
        default=DEFAULT_SCALAR_FILE_REGEX,
        help="Regex over Go file paths used to compute statement-weighted scalar target coverage",
    )
    args = parser.parse_args()

    launcher = Path(args.launcher).expanduser().resolve()
    if not args.coverage_report_only and not launcher.exists():
        print(f"{RED}Launcher not found:{RESET} {launcher}")
        return 2
    if not args.coverage_report_only and not ORACLE_PATH.exists():
        print(f"{RED}Oracle script not found:{RESET} {ORACLE_PATH}")
        return 2
    if not args.coverage_report_only and not VECTOR_FUZZ_PATH.exists():
        print(f"{RED}Vector fuzz script not found:{RESET} {VECTOR_FUZZ_PATH}")
        return 2

    log_root = Path(args.log_root).expanduser().resolve()
    cov_root = Path(args.cov_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    fallback_cov_dir = Path(args.cov_dir).expanduser().resolve() if args.cov_dir else None
    cov_inputs = normalize_cov_inputs(args.coverage_inputs, fallback_cov_dir)

    if args.coverage_report_only:
        report_run_id = args.run_id or f"coverage-report-{now_utc_stamp()}"
        log_dir = log_root / f"coverage_suite_{report_run_id}"
        artifacts_dir = log_dir / "artifacts"
        started = time.time()
        if not cov_inputs:
            print(f"{RED}No coverage input provided.{RESET} Use --coverage-inputs or --cov-dir.")
            return 2
        cov_summary = collect_covdata(
            args.go_bin,
            cov_inputs,
            artifacts_dir,
            args.coverage_percent_regex,
            args.coverage_func_regex,
            args.scalar_file_regex,
        )
        payload = {
            "run_id": report_run_id,
            "coverage_inputs": cov_inputs.split(","),
            "seed_strategy": args.seed_strategy,
            "suite_seed": args.suite_seed,
            "log_root": str(log_root),
            "scalar_file_regex": args.scalar_file_regex,
            "coverage": cov_summary,
        }
        log_dir.mkdir(parents=True, exist_ok=True)
        summary_path = log_dir / "suite_summary.json"
        summary_md_path = log_dir / "suite_summary.md"
        write_summary_json(summary_path, payload)
        write_summary_markdown(
            summary_md_path,
            payload=payload,
            results=[],
            coverage_rows=extract_coverage_rows(cov_summary.get("filtered_percent_path")),
            coverage_metrics=(cov_summary.get("metrics") if isinstance(cov_summary.get("metrics"), dict) else None),
            summary_json_path=summary_path,
            cases_csv_path=None,
        )
        return print_report_summary(log_dir, summary_path, cov_summary, time.time() - started)

    cases = expand_templates(SUITES[args.suite])
    try:
        cases = apply_seed_strategy(cases, args.seed_strategy, args.suite_seed)
    except ValueError as exc:
        print(f"{RED}{exc}{RESET}")
        return 2
    if args.time_budget_seconds is not None:
        if args.time_budget_seconds <= 0:
            print(f"{RED}--time-budget-seconds must be positive when provided.{RESET}")
            return 2
        if args.seed_strategy != "derived" or args.suite_seed is None:
            print(
                f"{RED}--time-budget-seconds requires --seed-strategy derived --suite-seed <int>{RESET} "
                "so repeated cycles are reproducible but not identical."
            )
            return 2
    elif args.budget_schedule != "cycle":
        print(f"{RED}--budget-schedule is only meaningful with --time-budget-seconds.{RESET}")
        return 2
    if args.coverage_timeline:
        if args.reuse_running_server:
            print(f"{RED}--coverage-timeline cannot be combined with --reuse-running-server.{RESET}")
            return 2
        if args.coverage_inputs:
            print(f"{RED}--coverage-timeline manages per-case coverage inputs; omit --coverage-inputs.{RESET}")
            return 2
        if args.coverage_timeline_interval_cases <= 0:
            print(f"{RED}--coverage-timeline-interval-cases must be positive.{RESET}")
            return 2
    if args.start_case > 1:
        cases = cases[max(0, args.start_case - 1):]
    if args.max_cases is not None and args.time_budget_seconds is None:
        cases = cases[: max(0, args.max_cases)]
    if not cases:
        print(f"{YELLOW}No cases selected.{RESET}")
        return 0

    run_id = args.run_id or f"weaviate-cov-{args.suite}-{now_utc_stamp()}"
    log_dir = log_root / f"coverage_suite_{run_id}"
    cov_dir = Path(args.cov_dir).expanduser().resolve() if args.cov_dir else (cov_root / run_id)
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else (data_root / run_id)
    cov_inputs = normalize_cov_inputs(args.coverage_inputs, cov_dir)
    artifacts_dir = log_dir / "artifacts"

    print(f"\n{BOLD}{CYAN}{'=' * 76}{RESET}")
    print(f"{BOLD}{CYAN}  Weaviate Coverage Fuzz Suite{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 76}{RESET}")
    print(f"  Suite:      {args.suite}")
    print(f"  Seed mode:  {args.seed_strategy}")
    if args.suite_seed is not None:
        print(f"  Suite seed: {args.suite_seed}")
    print(f"  Launcher:   {launcher}")
    print(f"  Oracle:     {ORACLE_PATH}")
    print(f"  Vector:     {VECTOR_FUZZ_PATH}")
    print(f"  Run ID:     {run_id}")
    print(f"  Log dir:    {log_dir}")
    print(f"  Cov dir:    {cov_dir}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Cases:      {len(cases)}")
    if args.time_budget_seconds is not None:
        print(f"  Budget:     {fmt_duration(args.time_budget_seconds)}")
        print(f"  Schedule:   {args.budget_schedule}")
    if args.max_cases is not None:
        cap_label = args.max_cases if args.time_budget_seconds is not None else len(cases)
        print(f"  Case cap:   {cap_label}")
    print(f"  Host:Port:  {args.host}:{args.port}")
    print(f"  gRPC Port:  {args.grpc_port}")
    print(f"  Server:     {'reuse existing' if args.reuse_running_server else 'launch new'}")
    if args.robustness_rate is not None and args.robustness_rate > 0.0:
        print(
            f"  Robustness: rate={args.robustness_rate:.2f} "
            f"start={float(args.robustness_start_fraction or 0.80):.2f} "
            f"frontdoors={args.robustness_frontdoors or 'python,rest,graphql'} "
            f"category={args.robustness_category_mode or 'warn'}"
        )
    if args.coverage_timeline:
        print(f"  Timeline:   every {args.coverage_timeline_interval_cases} completed case(s)")
    print(f"{CYAN}{'-' * 76}{RESET}")

    preview_cases = cases
    if args.time_budget_seconds is not None:
        preview_count = min(len(cases), len(cases) if args.max_cases is None else max(0, args.max_cases))
        preview_cases = [
            materialize_budget_case(
                select_budget_base_case(cases, int(args.suite_seed), index, args.budget_schedule),
                int(args.suite_seed),
                index,
            )
            for index in range(1, preview_count + 1)
        ]
    for index, case in enumerate(preview_cases, start=1):
        seed_label = "random" if case.seed is None else str(case.seed)
        log_suffix = f"{index:03d}_{case.name}_seed_{seed_label}"
        cmd = build_case_command(
            args.python_bin,
            case,
            args.query_page_size,
            args.host,
            args.port,
            args.grpc_port,
            case_log_dir=log_dir / "case_logs",
            log_suffix=log_suffix,
            robustness_rate=args.robustness_rate,
            robustness_start_fraction=args.robustness_start_fraction,
            robustness_frontdoors=args.robustness_frontdoors,
            robustness_category_mode=args.robustness_category_mode,
        )
        print(
            f"  [{index:02d}] {case.name:<22} mode={case.mode:<7} "
            f"seed={seed_label:<6} rounds={case.rounds:<4} N={case.rows:<6} cmd={shell_join(cmd)}"
        )
    if args.time_budget_seconds is not None and len(preview_cases) < len(cases):
        print(f"  ... {len(cases) - len(preview_cases)} more base cases in each cycle")
    if args.time_budget_seconds is not None:
        if args.budget_schedule == "random":
            print("  (Budget mode will draw reproducible pseudo-random cases with derived per-case seeds.)")
        else:
            print("  (Budget mode will continue cycling this matrix with derived per-case seeds.)")
    if args.coverage_timeline:
        print("  (Coverage timeline mode restarts Weaviate per case to flush Go counters.)")

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
    coverage_timeline_rows: list[dict] = []
    completed_cov_dirs: list[Path] = []
    suite_start = time.time()
    launched_server = False

    if args.coverage_timeline:
        coverage_timeline_rows.append(build_timeline_baseline_row(suite_seed=args.suite_seed))

    try:
        if args.reuse_running_server:
            print(f"\n{CYAN}Reusing running Weaviate at {args.host}:{args.port}.{RESET}")
        elif args.coverage_timeline:
            print(f"\n{CYAN}Coverage timeline mode: launching Weaviate separately for each case.{RESET}")
        else:
            print(f"\n{CYAN}Starting coverage-enabled Weaviate...{RESET}")
            server_proc = start_server(
                launcher,
                run_id,
                args.host,
                args.port,
                args.grpc_port,
                args.ready_timeout,
                cov_dir,
                data_dir,
                log_dir / "weaviate_server.log",
            )
            launched_server = True
            print(f"{GREEN}Weaviate is ready.{RESET}")

        deadline = (suite_start + args.time_budget_seconds) if args.time_budget_seconds is not None else None
        index = 1
        while True:
            if deadline is None:
                if index > len(cases):
                    break
                case = cases[index - 1]
                progress_total = str(len(cases))
                cycle = 1
            else:
                if index > 1 and time.time() >= deadline:
                    print(f"{YELLOW}Time budget reached; stopping after completed case {index - 1}.{RESET}")
                    break
                if args.max_cases is not None and index > args.max_cases:
                    print(f"{YELLOW}Case cap reached; stopping after completed case {index - 1}.{RESET}")
                    break
                base_case = select_budget_base_case(cases, int(args.suite_seed), index, args.budget_schedule)
                case = materialize_budget_case(base_case, int(args.suite_seed), index)
                progress_total = f"budget {fmt_duration(args.time_budget_seconds)}"
                cycle = ((index - 1) // len(cases)) + 1
            seed_label = "random" if case.seed is None else str(case.seed)
            log_suffix = f"{index:03d}_cycle{cycle:02d}_{case.name}_seed_{seed_label}"
            cmd = build_case_command(
                args.python_bin,
                case,
                args.query_page_size,
                args.host,
                args.port,
                args.grpc_port,
                case_log_dir=log_dir / "case_logs",
                log_suffix=log_suffix,
                robustness_rate=args.robustness_rate,
                robustness_start_fraction=args.robustness_start_fraction,
                robustness_frontdoors=args.robustness_frontdoors,
                robustness_category_mode=args.robustness_category_mode,
            )
            log_name = f"{index:03d}_cycle{cycle:02d}_{case.name}_seed_{seed_label}.log"
            log_path = log_dir / log_name
            case_label = safe_label(f"{index:03d}_cycle{cycle:02d}_{case.name}_seed_{seed_label}")
            case_cov_dir = cov_dir
            case_data_dir = data_dir
            case_server_proc: subprocess.Popen | None = None
            if args.coverage_timeline:
                case_cov_dir = cov_dir / case_label
                case_data_dir = data_dir / case_label

            print(
                f"[{index}/{progress_total}] {case.name} "
                f"(mode={case.mode}, seed={seed_label}, rounds={case.rounds}, N={case.rows}) ..."
            )
            if args.coverage_timeline:
                print(f"    starting isolated coverage server for {case_label} ...")
                case_server_proc = start_server(
                    launcher,
                    f"{run_id}-{case_label}",
                    args.host,
                    args.port,
                    args.grpc_port,
                    args.ready_timeout,
                    case_cov_dir,
                    case_data_dir,
                    log_dir / "server_logs" / f"{case_label}.log",
                )
            try:
                exit_code, elapsed = run_case(cmd, log_path)
            finally:
                if args.coverage_timeline:
                    stop_server(case_server_proc)

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

            if args.coverage_timeline:
                completed_cov_dirs.append(case_cov_dir)
                should_snapshot = (
                    index % args.coverage_timeline_interval_cases == 0
                    or not result.passed
                    or (deadline is None and index == len(cases))
                )
                if should_snapshot:
                    snapshot_dir = artifacts_dir / "coverage_timeline" / f"step_{index:03d}_{case_label}"
                    snapshot_inputs = ",".join(str(path) for path in completed_cov_dirs)
                    snapshot_summary = collect_covdata(
                        args.go_bin,
                        snapshot_inputs,
                        snapshot_dir,
                        args.coverage_percent_regex,
                        args.coverage_func_regex,
                        args.scalar_file_regex,
                    )
                    timeline_row = build_timeline_row(
                        case_index=index,
                        cycle=cycle,
                        case=case,
                        result=result,
                        suite_start=suite_start,
                        snapshot_dir=snapshot_dir,
                        cov_inputs=snapshot_inputs,
                        cov_summary=snapshot_summary,
                    )
                    coverage_timeline_rows.append(timeline_row)
                    timeline_artifacts = write_coverage_timeline_artifacts(log_dir, coverage_timeline_rows)
                    partial_payload = {
                        "suite": args.suite,
                        "seed_strategy": args.seed_strategy,
                        "suite_seed": args.suite_seed,
                        "run_id": run_id,
                        "launcher": str(launcher),
                        "oracle_path": str(ORACLE_PATH),
                        "vector_fuzz_path": str(VECTOR_FUZZ_PATH),
                        "log_root": str(log_root),
                        "cov_dir": str(cov_dir),
                        "data_dir": str(data_dir),
                        "host": args.host,
                        "port": args.port,
                        "grpc_port": args.grpc_port,
                        "query_page_size": args.query_page_size,
                        "time_budget_seconds": args.time_budget_seconds,
                        "budget_schedule": args.budget_schedule,
                        "suite_elapsed_seconds": time.time() - suite_start,
                        "results": [asdict(item) for item in results],
                        "coverage_timeline": timeline_artifacts,
                        "coverage": snapshot_summary,
                        "partial": True,
                    }
                    write_summary_json(log_dir / "suite_summary.partial.json", partial_payload)
                    print(
                        "    coverage snapshot: "
                        f"overall={float(timeline_row['overall_coverage_percent']):.2f}% "
                        f"scalar={float(timeline_row['scalar_coverage_percent']):.2f}%"
                    )
            if not result.passed and args.stop_on_fail:
                print(f"{RED}Stopping on first failure as requested.{RESET}")
                break
            index += 1
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
    elif args.coverage_timeline:
        final_inputs = ",".join(str(path) for path in completed_cov_dirs)
        if final_inputs:
            cov_summary = collect_covdata(
                args.go_bin,
                final_inputs,
                artifacts_dir,
                args.coverage_percent_regex,
                args.coverage_func_regex,
                args.scalar_file_regex,
            )
        else:
            cov_summary = {"available": False, "detail": "no completed per-case coverage directories"}
    else:
        cov_summary = collect_covdata(
            args.go_bin,
            cov_inputs or str(cov_dir),
            artifacts_dir,
            args.coverage_percent_regex,
            args.coverage_func_regex,
            args.scalar_file_regex,
        )
    suite_elapsed = time.time() - suite_start
    passed_count = sum(1 for item in results if item.passed)
    failed = [item for item in results if not item.passed]
    log_dir.mkdir(parents=True, exist_ok=True)
    failure_artifacts = write_failure_artifacts(log_dir, results)
    timeline_artifacts = (
        write_coverage_timeline_artifacts(log_dir, coverage_timeline_rows)
        if args.coverage_timeline
        else {"enabled": False}
    )

    payload = {
        "suite": args.suite,
        "seed_strategy": args.seed_strategy,
        "suite_seed": args.suite_seed,
        "run_id": run_id,
        "launcher": str(launcher),
        "oracle_path": str(ORACLE_PATH),
        "vector_fuzz_path": str(VECTOR_FUZZ_PATH),
        "log_root": str(log_root),
        "cov_dir": str(cov_dir),
        "data_dir": str(data_dir),
        "host": args.host,
        "port": args.port,
        "grpc_port": args.grpc_port,
        "query_page_size": args.query_page_size,
        "time_budget_seconds": args.time_budget_seconds,
        "budget_schedule": args.budget_schedule,
        "suite_elapsed_seconds": suite_elapsed,
        "results": [asdict(item) for item in results],
        "failure_artifacts": failure_artifacts,
        "coverage_timeline": timeline_artifacts,
        "coverage": cov_summary,
    }
    summary_path = log_dir / "suite_summary.json"
    summary_md_path = log_dir / "suite_summary.md"
    cases_csv_path = log_dir / "case_results.csv"
    write_summary_json(summary_path, payload)
    write_case_results_csv(cases_csv_path, results)
    write_summary_markdown(
        summary_md_path,
        payload=payload,
        results=results,
        coverage_rows=extract_coverage_rows(cov_summary.get("filtered_percent_path")),
        coverage_metrics=(cov_summary.get("metrics") if isinstance(cov_summary.get("metrics"), dict) else None),
        summary_json_path=summary_path,
        cases_csv_path=cases_csv_path,
    )

    print(f"\n{BOLD}{'=' * 76}{RESET}")
    print(f"{BOLD}Suite Summary{RESET}")
    print(f"{'=' * 76}")
    print(f"  Total cases: {len(results)}")
    print(f"  Passed:      {GREEN}{passed_count}{RESET}")
    print(f"  Failed:      {RED}{len(failed)}{RESET}")
    print(f"  Duration:    {fmt_duration(suite_elapsed)}")
    print(f"  Logs:        {log_dir}")
    print(f"  Summary:     {summary_path}")
    print(f"  Summary MD:  {summary_md_path}")
    print(f"  Case CSV:    {cases_csv_path}")
    print(f"  Repros:      {failure_artifacts.get('jsonl_path')}")
    if timeline_artifacts.get("csv_path"):
        print(f"  Timeline:    {timeline_artifacts.get('csv_path')}")
    if cov_summary.get("available"):
        print(f"  Coverage:    {cov_summary.get('percent_path')}")
        if cov_summary.get("filtered_percent_path"):
            print(f"  Cov filter:  {cov_summary.get('filtered_percent_path')}")
        if cov_summary.get("func_path"):
            print(f"  Cover func:  {cov_summary.get('func_path')}")
        if cov_summary.get("filtered_func_path"):
            print(f"  Func filter: {cov_summary.get('filtered_func_path')}")
        print(f"  Textfmt:     {cov_summary.get('textfmt_path')}")
        metrics = cov_summary.get("metrics") if isinstance(cov_summary.get("metrics"), dict) else None
        if metrics:
            overall = metrics.get("overall", {})
            scalar = metrics.get("scalar_target", {})
            print(f"  Overall cov: {overall.get('coverage_percent', 0.0):.2f}%")
            print(f"  Scalar cov:  {scalar.get('coverage_percent', 0.0):.2f}%")
            print(f"  Metrics:     {metrics.get('metrics_json_path')}")
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
