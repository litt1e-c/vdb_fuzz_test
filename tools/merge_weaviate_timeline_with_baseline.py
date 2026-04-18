#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_weaviate_cov_suite import (  # noqa: E402
    DEFAULT_SCALAR_FILE_REGEX,
    DEFAULT_SCALAR_FUNC_REGEX,
    DEFAULT_SCALAR_PERCENT_REGEX,
    collect_covdata,
)


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_timeline_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"timeline csv is empty: {path}")
    return rows


def float_value(value: str | int | float | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def int_value(value: str | int | float | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def join_cov_inputs(parts: list[str]) -> str:
    normalized: list[str] = []
    for part in parts:
        if not part:
            continue
        for item in str(part).split(","):
            item = item.strip()
            if not item:
                continue
            normalized.append(str(Path(item).expanduser().resolve()))
    seen: set[str] = set()
    ordered: list[str] = []
    for item in normalized:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ",".join(ordered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a stage-B Weaviate coverage timeline with a fixed baseline coverage run"
    )
    parser.add_argument("--timeline-csv", required=True, help="Stage-B coverage_timeline.csv")
    parser.add_argument(
        "--baseline-cov-inputs",
        required=True,
        help="Comma-separated baseline GOCOVERDIR inputs, typically the stage-A cov dir",
    )
    parser.add_argument(
        "--baseline-summary-json",
        required=True,
        help="Stage-A suite_summary.json used to seed baseline elapsed time and metrics",
    )
    parser.add_argument(
        "--baseline-timeline-csv",
        default=None,
        help="Optional stage-A coverage_timeline.csv; when provided, baseline is merged as a full timeline instead of a single summary point",
    )
    parser.add_argument(
        "--baseline-elapsed-seconds",
        type=float,
        default=None,
        help="Optional explicit baseline elapsed time; defaults to suite_summary.json",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for merged timeline artifacts")
    parser.add_argument("--go-bin", default="go", help="Go binary used for covdata export")
    parser.add_argument(
        "--coverage-percent-regex",
        default=DEFAULT_SCALAR_PERCENT_REGEX,
        help="Regex for filtered covdata percent output",
    )
    parser.add_argument(
        "--coverage-func-regex",
        default=DEFAULT_SCALAR_FUNC_REGEX,
        help="Regex for filtered go tool cover -func output",
    )
    parser.add_argument(
        "--scalar-file-regex",
        default=DEFAULT_SCALAR_FILE_REGEX,
        help="Regex for statement-weighted scalar-target coverage",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    timeline_csv = Path(args.timeline_csv).expanduser().resolve()
    baseline_summary_json = Path(args.baseline_summary_json).expanduser().resolve()
    baseline_timeline_csv = (
        Path(args.baseline_timeline_csv).expanduser().resolve()
        if args.baseline_timeline_csv
        else None
    )
    rows = load_timeline_rows(timeline_csv)
    baseline_summary = read_summary(baseline_summary_json)

    baseline_cov_inputs = join_cov_inputs([args.baseline_cov_inputs])
    if not baseline_cov_inputs:
        raise SystemExit("baseline cov inputs are empty")

    baseline_elapsed = (
        args.baseline_elapsed_seconds
        if args.baseline_elapsed_seconds is not None
        else float_value(baseline_summary.get("suite_elapsed_seconds"))
    )
    baseline_coverage = baseline_summary.get("coverage", {})
    baseline_metrics = baseline_coverage.get("metrics", {}) if isinstance(baseline_coverage, dict) else {}
    overall_baseline = baseline_metrics.get("overall", {}) if isinstance(baseline_metrics, dict) else {}
    scalar_baseline = baseline_metrics.get("scalar_target", {}) if isinstance(baseline_metrics, dict) else {}

    merged_rows: list[dict[str, object]] = []
    if baseline_timeline_csv and baseline_timeline_csv.exists():
        baseline_timeline_rows = load_timeline_rows(baseline_timeline_csv)
        for row in baseline_timeline_rows:
            suite_elapsed_seconds = float_value(row.get("suite_elapsed_seconds"))
            baseline_row_inputs = join_cov_inputs([row.get("cov_inputs", "")])
            merged_rows.append(
                {
                    "case_index": len(merged_rows),
                    "cycle": int_value(row.get("cycle")),
                    "case_name": f"stage-a-{row.get('case_name', 'baseline')}",
                    "mode": row.get("mode", "baseline"),
                    "seed": row.get("seed", baseline_summary.get("suite_seed") or ""),
                    "rows": int_value(row.get("rows")),
                    "rounds": int_value(row.get("rounds")),
                    "passed": row.get("passed", True),
                    "case_elapsed_seconds": float_value(row.get("case_elapsed_seconds")),
                    "suite_elapsed_seconds": round(suite_elapsed_seconds, 3),
                    "elapsed_minutes": round(suite_elapsed_seconds / 60.0, 3),
                    "overall_coverage_percent": float_value(row.get("overall_coverage_percent")),
                    "overall_covered_statements": int_value(row.get("overall_covered_statements")),
                    "overall_total_statements": int_value(row.get("overall_total_statements")),
                    "scalar_coverage_percent": float_value(row.get("scalar_coverage_percent")),
                    "scalar_covered_statements": int_value(row.get("scalar_covered_statements")),
                    "scalar_total_statements": int_value(row.get("scalar_total_statements")),
                    "merged_cov_inputs": baseline_row_inputs,
                    "snapshot_dir": row.get("snapshot_dir", ""),
                    "timeline_snapshot_dir": row.get("snapshot_dir", ""),
                    "coverage_available": row.get("coverage_available", True),
                    "coverage_detail": row.get("coverage_detail", "stage-a baseline timeline"),
                }
            )
        if merged_rows:
            baseline_elapsed = float_value(merged_rows[-1].get("suite_elapsed_seconds"))
    else:
        merged_rows.append(
            {
                "case_index": 0,
                "cycle": 0,
                "case_name": "stage-a-baseline",
                "mode": "baseline",
                "seed": baseline_summary.get("suite_seed") or "",
                "rows": 0,
                "rounds": 0,
                "passed": True,
                "case_elapsed_seconds": round(baseline_elapsed, 3),
                "suite_elapsed_seconds": round(baseline_elapsed, 3),
                "elapsed_minutes": round(baseline_elapsed / 60.0, 3),
                "overall_coverage_percent": float_value(overall_baseline.get("coverage_percent")),
                "overall_covered_statements": int_value(overall_baseline.get("covered_statements")),
                "overall_total_statements": int_value(overall_baseline.get("total_statements")),
                "scalar_coverage_percent": float_value(scalar_baseline.get("coverage_percent")),
                "scalar_covered_statements": int_value(scalar_baseline.get("covered_statements")),
                "scalar_total_statements": int_value(scalar_baseline.get("total_statements")),
                "merged_cov_inputs": baseline_cov_inputs,
                "snapshot_dir": "",
                "timeline_snapshot_dir": "",
                "coverage_available": True,
                "coverage_detail": "stage-a baseline summary",
            }
        )

    for row in rows:
        case_index = int_value(row.get("case_index"))
        if case_index <= 0:
            continue

        stage_b_cov_inputs = join_cov_inputs([row.get("cov_inputs", "")])
        merged_cov_inputs = join_cov_inputs([baseline_cov_inputs, stage_b_cov_inputs])
        snapshot_dir = snapshots_dir / f"step_{case_index:03d}"
        cov_summary = collect_covdata(
            args.go_bin,
            merged_cov_inputs,
            snapshot_dir,
            args.coverage_percent_regex,
            args.coverage_func_regex,
            args.scalar_file_regex,
        )
        metrics = cov_summary.get("metrics", {}) if isinstance(cov_summary.get("metrics"), dict) else {}
        overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
        scalar = metrics.get("scalar_target", {}) if isinstance(metrics, dict) else {}
        merged_elapsed = baseline_elapsed + float_value(row.get("suite_elapsed_seconds"))
        merged_rows.append(
            {
                "case_index": len(merged_rows),
                "cycle": int_value(row.get("cycle")),
                "case_name": row.get("case_name", ""),
                "mode": row.get("mode", ""),
                "seed": row.get("seed", ""),
                "rows": int_value(row.get("rows")),
                "rounds": int_value(row.get("rounds")),
                "passed": row.get("passed", ""),
                "case_elapsed_seconds": float_value(row.get("case_elapsed_seconds")),
                "suite_elapsed_seconds": round(merged_elapsed, 3),
                "elapsed_minutes": round(merged_elapsed / 60.0, 3),
                "overall_coverage_percent": float_value(overall.get("coverage_percent")),
                "overall_covered_statements": int_value(overall.get("covered_statements")),
                "overall_total_statements": int_value(overall.get("total_statements")),
                "scalar_coverage_percent": float_value(scalar.get("coverage_percent")),
                "scalar_covered_statements": int_value(scalar.get("covered_statements")),
                "scalar_total_statements": int_value(scalar.get("total_statements")),
                "merged_cov_inputs": merged_cov_inputs,
                "snapshot_dir": str(snapshot_dir),
                "timeline_snapshot_dir": row.get("snapshot_dir", ""),
                "coverage_available": bool(cov_summary.get("available")),
                "coverage_detail": cov_summary.get("detail", ""),
            }
        )

    csv_path = output_dir / "coverage_timeline_merged.csv"
    jsonl_path = output_dir / "coverage_timeline_merged.jsonl"
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
        "elapsed_minutes",
        "overall_coverage_percent",
        "overall_covered_statements",
        "overall_total_statements",
        "scalar_coverage_percent",
        "scalar_covered_statements",
        "scalar_total_statements",
        "merged_cov_inputs",
        "snapshot_dir",
        "timeline_snapshot_dir",
        "coverage_available",
        "coverage_detail",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for row in merged_rows:
            fout.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    final_row = merged_rows[-1]
    summary = {
        "baseline_summary_json": str(baseline_summary_json),
        "baseline_timeline_csv": str(baseline_timeline_csv) if baseline_timeline_csv and baseline_timeline_csv.exists() else None,
        "baseline_cov_inputs": baseline_cov_inputs.split(","),
        "timeline_csv": str(timeline_csv),
        "merged_csv": str(csv_path),
        "merged_jsonl": str(jsonl_path),
        "snapshots": len(merged_rows),
        "final_overall_coverage_percent": final_row["overall_coverage_percent"],
        "final_scalar_coverage_percent": final_row["scalar_coverage_percent"],
        "final_overall_covered_statements": final_row["overall_covered_statements"],
        "final_scalar_covered_statements": final_row["scalar_covered_statements"],
    }
    summary_path = output_dir / "merged_timeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"merged csv:     {csv_path}")
    print(f"merged jsonl:   {jsonl_path}")
    print(f"summary json:   {summary_path}")
    print(
        "final coverage: "
        f"overall={float_value(final_row['overall_coverage_percent']):.2f}% "
        f"scalar={float_value(final_row['scalar_coverage_percent']):.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
