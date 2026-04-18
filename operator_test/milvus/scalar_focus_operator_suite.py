#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


OPERATOR_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "milvus_log"

OPERATORS = {
    "bool_domain_rewrite": OPERATOR_DIR / "bool_domain_rewrite_operator.py",
    "column_compare": OPERATOR_DIR / "column_compare_operator.py",
    "deep_expression": OPERATOR_DIR / "deep_expression_operator.py",
    "in": OPERATOR_DIR / "in_operator.py",
    "json_array_contains_family": OPERATOR_DIR / "json_array_contains_family_operator.py",
    "not_in": OPERATOR_DIR / "not_in_operator.py",
    "like": OPERATOR_DIR / "like_operator.py",
    "like_ngram_equivalence": OPERATOR_DIR / "like_ngram_equivalence_operator.py",
    "not": OPERATOR_DIR / "not_operator.py",
    "logical_metamorphic": OPERATOR_DIR / "logical_metamorphic_operator.py",
    "range_rewrite_equivalence": OPERATOR_DIR / "range_rewrite_equivalence_operator.py",
    "array_length": OPERATOR_DIR / "array_length_operator.py",
    "dynamic_field": OPERATOR_DIR / "dynamic_field_operator.py",
    "scalar_index_equivalence": OPERATOR_DIR / "scalar_index_equivalence_operator.py",
    "string_function_call": OPERATOR_DIR / "string_function_call_operator.py",
    "three_valued_logic": OPERATOR_DIR / "three_valued_logic_operator.py",
    "templated_unary_not": OPERATOR_DIR / "templated_unary_not_operator.py",
}

DEFAULT_SET = [
    "bool_domain_rewrite",
    "column_compare",
    "deep_expression",
    "in",
    "json_array_contains_family",
    "not_in",
    "like",
    "like_ngram_equivalence",
    "not",
    "dynamic_field",
    "logical_metamorphic",
    "range_rewrite_equivalence",
    "scalar_index_equivalence",
    "string_function_call",
    "three_valued_logic",
    "templated_unary_not",
]

COLLECTION_ARG_OPERATORS = {
    "bool_domain_rewrite",
    "column_compare",
    "deep_expression",
    "dynamic_field",
    "json_array_contains_family",
    "like_ngram_equivalence",
    "range_rewrite_equivalence",
    "scalar_index_equivalence",
    "string_function_call",
    "three_valued_logic",
}


def parse_operator_list(raw: str) -> list[str]:
    token = raw.strip().lower()
    if token == "all":
        return list(DEFAULT_SET)

    selected = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(OPERATORS))
    if unknown:
        raise ValueError(f"unknown operators: {', '.join(unknown)}")
    if not selected:
        raise ValueError("empty operator list")
    return selected


def slugify(text: str) -> str:
    out = []
    for ch in text.lower().strip():
        out.append(ch if ch.isalnum() else "_")
    value = "".join(out).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or "x"


def run_one(
    *,
    name: str,
    script: Path,
    python_bin: str,
    host: str,
    port: str,
    run_id: str,
    case_logs_dir: Path,
    expected_error_category_mode: str,
) -> dict[str, object]:
    started = time.time()
    case_slug = slugify(name)
    log_path = case_logs_dir / f"{case_slug}.log"

    command = [python_bin, str(script)]
    if name in COLLECTION_ARG_OPERATORS:
        command.extend(["--host", host, "--port", str(port), "--collection", f"{case_slug}_{slugify(run_id)}"])

    env = os.environ.copy()
    env["MILVUS_HOST"] = host
    env["MILVUS_PORT"] = str(port)
    env["MILVUS_EXPECTED_ERROR_CATEGORY_MODE"] = expected_error_category_mode

    with log_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.run(command, stdout=fout, stderr=subprocess.STDOUT, env=env)

    elapsed = time.time() - started
    passed = proc.returncode == 0
    return {
        "name": name,
        "script": str(script),
        "command": shlex.join(command),
        "log_path": str(log_path),
        "exit_code": proc.returncode,
        "passed": passed,
        "elapsed_seconds": round(elapsed, 3),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["name", "script", "command", "log_path", "exit_code", "passed", "elapsed_seconds"]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Milvus scalar-focused operator suite")
    parser.add_argument("--host", default=os.getenv("MILVUS_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.getenv("MILVUS_PORT", "19630"))
    parser.add_argument("--operators", default="all", help="Comma-separated operator keys or 'all'")
    parser.add_argument("--run-id", default=f"scalar-focus-{int(time.time())}")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument(
        "--expected-error-category-mode",
        choices=["ignore", "warn", "strict"],
        default=os.getenv("MILVUS_EXPECTED_ERROR_CATEGORY_MODE", "warn"),
        help="How to handle expected_error category mismatches inside operator validators",
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)

    try:
        selected = parse_operator_list(args.operators)
    except ValueError as exc:
        print(f"suite_config_error: {exc}")
        return 2

    suite_dir = Path(args.log_dir).expanduser().resolve() / args.run_id
    case_logs_dir = suite_dir / "case_logs"
    suite_dir.mkdir(parents=True, exist_ok=True)
    case_logs_dir.mkdir(parents=True, exist_ok=True)

    print("Milvus scalar-focused operator suite")
    print(f"Target: {args.host}:{args.port}")
    print(f"Run id: {args.run_id}")
    print(f"Operators: {', '.join(selected)}")
    print(f"Expected-error category mode: {args.expected_error_category_mode}")

    results: list[dict[str, object]] = []
    for name in selected:
        script = OPERATORS[name]
        if not script.exists():
            result = {
                "name": name,
                "script": str(script),
                "command": "",
                "log_path": "",
                "exit_code": 127,
                "passed": False,
                "elapsed_seconds": 0.0,
            }
            results.append(result)
            print(f"{name}: FAIL | missing script {script}")
            if args.stop_on_failure:
                break
            continue

        result = run_one(
            name=name,
            script=script,
            python_bin=args.python_bin,
            host=args.host,
            port=str(args.port),
            run_id=args.run_id,
            case_logs_dir=case_logs_dir,
            expected_error_category_mode=args.expected_error_category_mode,
        )
        results.append(result)
        print(
            f"{name}: {'PASS' if result['passed'] else 'FAIL'} | "
            f"exit={result['exit_code']} | log={result['log_path']}"
        )
        if args.stop_on_failure and not result["passed"]:
            break

    summary = {
        "run_id": args.run_id,
        "host": args.host,
        "port": str(args.port),
        "operators": selected,
        "expected_error_category_mode": args.expected_error_category_mode,
        "results": results,
        "passed": all(bool(item.get("passed")) for item in results),
        "result_count": len(results),
        "failed_count": sum(1 for item in results if not bool(item.get("passed"))),
    }

    summary_json = suite_dir / "suite_summary.json"
    summary_md = suite_dir / "suite_summary.md"
    cases_csv = suite_dir / "case_results.csv"

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(cases_csv, results)

    md_lines = [
        "# Milvus Scalar Focus Operator Suite",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Target: `{args.host}:{args.port}`",
        f"- Expected-error category mode: `{args.expected_error_category_mode}`",
        f"- Passed: `{summary['passed']}`",
        f"- Total operators: `{summary['result_count']}`",
        f"- Failed operators: `{summary['failed_count']}`",
        f"- Summary JSON: `{summary_json}`",
        f"- Case CSV: `{cases_csv}`",
        "",
        "## Results",
    ]
    for item in results:
        md_lines.append(
            f"- {item['name']}: {'PASS' if item['passed'] else 'FAIL'} "
            f"(exit={item['exit_code']}, log=`{item['log_path']}`)"
        )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD:   {summary_md}")
    print(f"Case CSV:     {cases_csv}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
