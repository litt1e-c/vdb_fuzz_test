#!/usr/bin/env python3
"""
Regression probe for Milvus constant-expression filters.

Coverage:
1. Pure constant boolean expressions.
2. Constant expressions mixed with scalar predicates.
3. Constant expressions combined with nullable fields.

Expected behavior follows normal boolean logic. If any covered case still
raises an exception or produces the wrong row set, the old bug is not fully
fixed for the probed surface.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "19531"
DEFAULT_COLLECTION_NAME = "probe_constant_expr_regression"
CONNECTION_ALIAS = "constant_expr_probe"
VECTOR_DIM = 4


@dataclass(frozen=True)
class TestCase:
    group: str
    name: str
    expression: str
    expected_ids: list[int]
    description: str


@dataclass
class TestResult:
    case: TestCase
    actual_ids: list[int] | None = None
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "ERROR"
        if self.actual_ids == self.case.expected_ids:
            return "PASS"
        return "FAIL"


TEST_CASES = [
    TestCase(
        group="pure_constants",
        name="const_true_returns_all_rows",
        expression="1 < 2",
        expected_ids=[1, 2, 3, 4],
        description="Pure constant true expression should match every row.",
    ),
    TestCase(
        group="pure_constants",
        name="const_false_returns_no_rows",
        expression="1 > 2",
        expected_ids=[],
        description="Pure constant false expression should match no rows.",
    ),
    TestCase(
        group="pure_constants",
        name="const_arithmetic_true_returns_all_rows",
        expression="1 + 1 == 2",
        expected_ids=[1, 2, 3, 4],
        description="Arithmetic constant expression should be evaluated normally.",
    ),
    TestCase(
        group="pure_constants",
        name="negated_const_true_returns_no_rows",
        expression="not (1 < 2)",
        expected_ids=[],
        description="Negated constant expression should be evaluated normally.",
    ),
    TestCase(
        group="mixed_with_scalar_predicates",
        name="const_true_and_scalar_predicate",
        expression="1 < 2 and score >= 20",
        expected_ids=[2, 3, 4],
        description="Constant true should not block a normal scalar predicate.",
    ),
    TestCase(
        group="mixed_with_scalar_predicates",
        name="const_false_or_scalar_predicate",
        expression="1 > 2 or id == 1",
        expected_ids=[1],
        description="Constant false OR predicate should behave like the predicate.",
    ),
    TestCase(
        group="mixed_with_scalar_predicates",
        name="scalar_predicate_or_const_false",
        expression="id == 1 or 2 == 3",
        expected_ids=[1],
        description="Predicate OR constant false should be accepted and evaluated.",
    ),
    TestCase(
        group="mixed_with_nullable_fields",
        name="nullable_bool_and_const_true",
        expression="flag is null and 1 < 2",
        expected_ids=[3],
        description="Nullable scalar field should compose with constant true.",
    ),
    TestCase(
        group="mixed_with_nullable_fields",
        name="const_true_and_nullable_bool",
        expression="1 < 2 and flag is null",
        expected_ids=[3],
        description="Constant true should compose regardless of operand order.",
    ),
    TestCase(
        group="mixed_with_nullable_fields",
        name="nullable_json_and_const_true",
        expression="payload is null and 1 < 2",
        expected_ids=[3],
        description="Nullable JSON field should compose with constant true.",
    ),
    TestCase(
        group="sanity_checks",
        name="baseline_scalar_query",
        expression="id < 2",
        expected_ids=[1],
        description="Baseline scalar filter verifies the collection is queryable.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether Milvus constant-expression query bugs are fixed."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Milvus host")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Milvus port")
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Temporary collection name used by this probe",
    )
    parser.add_argument(
        "--report-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory for the text report",
    )
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Keep the temporary collection instead of dropping it at the end",
    )
    return parser.parse_args()


def build_rows() -> list[dict]:
    return [
        {
            "id": 1,
            "vector": [0.10, 0.00, 0.00, 0.00],
            "score": 10,
            "flag": True,
            "payload": {"k": "v"},
        },
        {
            "id": 2,
            "vector": [0.00, 0.10, 0.00, 0.00],
            "score": 20,
            "flag": False,
            "payload": {"k": "x"},
        },
        {
            "id": 3,
            "vector": [0.00, 0.00, 0.10, 0.00],
            "score": 30,
            "flag": None,
            "payload": None,
        },
        {
            "id": 4,
            "vector": [0.00, 0.00, 0.00, 0.10],
            "score": 40,
            "flag": True,
            "payload": {},
        },
    ]


def ensure_collection_absent(collection_name: str) -> None:
    if utility.has_collection(collection_name, using=CONNECTION_ALIAS):
        utility.drop_collection(collection_name, using=CONNECTION_ALIAS)


def create_probe_collection(collection_name: str) -> Collection:
    ensure_collection_absent(collection_name)

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="score", dtype=DataType.INT64),
            FieldSchema(name="flag", dtype=DataType.BOOL, nullable=True),
            FieldSchema(name="payload", dtype=DataType.JSON, nullable=True),
        ],
        description="Temporary collection for constant-expression regression probing",
    )

    collection = Collection(
        name=collection_name,
        schema=schema,
        using=CONNECTION_ALIAS,
    )
    collection.insert(build_rows())
    collection.flush()
    collection.create_index(
        field_name="vector",
        index_params={"metric_type": "L2", "index_type": "FLAT", "params": {}},
    )
    collection.load()
    return collection


def run_case(collection: Collection, case: TestCase) -> TestResult:
    try:
        rows = collection.query(expr=case.expression, output_fields=["id"])
        actual_ids = sorted(row["id"] for row in rows)
        return TestResult(case=case, actual_ids=actual_ids)
    except Exception as exc:  # pragma: no cover - exercised when bug reproduces
        return TestResult(case=case, error=str(exc))


def count_by_status(results: Iterable[TestResult]) -> dict[str, int]:
    summary = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    for result in results:
        summary[result.status] += 1
    return summary


def overall_assessment(results: list[TestResult]) -> str:
    statuses = {result.status for result in results}
    if statuses == {"PASS"}:
        return "FIXED_FOR_COVERED_CASES"
    if "PASS" in statuses:
        return "PARTIALLY_FIXED_OR_STILL_REGRESSED"
    return "BUG_REPRODUCED"


def format_result_block(result: TestResult) -> str:
    lines = [
        f"[{result.status}] {result.case.name}",
        f"Group: {result.case.group}",
        f"Expr: {result.case.expression}",
        f"Expected IDs: {result.case.expected_ids}",
        f"Description: {result.case.description}",
    ]
    if result.error is not None:
        lines.append(f"Error: {result.error}")
    else:
        lines.append(f"Actual IDs: {result.actual_ids}")
    return "\n".join(lines)


def write_report(
    report_dir: Path,
    host: str,
    port: str,
    collection_name: str,
    server_version: str,
    results: list[TestResult],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"milvus_constant_expression_regression_report_{timestamp}.txt"

    summary = count_by_status(results)
    assessment = overall_assessment(results)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    lines = [
        "Milvus Constant Expression Regression Report",
        f"Generated At: {generated_at}",
        f"Target: {host}:{port}",
        f"Server Version: {server_version}",
        f"Collection: {collection_name}",
        f"Assessment: {assessment}",
        (
            "Summary: "
            f"total={len(results)}, pass={summary['PASS']}, "
            f"fail={summary['FAIL']}, error={summary['ERROR']}"
        ),
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.append(f"Case {index}")
        lines.append(format_result_block(result))
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_runtime_summary(results: list[TestResult], report_path: Path) -> None:
    summary = count_by_status(results)
    assessment = overall_assessment(results)

    print("=" * 72)
    print("Milvus constant-expression regression probe")
    print("=" * 72)
    for result in results:
        print(f"[{result.status}] {result.case.group} :: {result.case.expression}")
        print(f"  expected: {result.case.expected_ids}")
        if result.error is not None:
            print(f"  error:    {result.error}")
        else:
            print(f"  actual:   {result.actual_ids}")
    print("-" * 72)
    print(
        "Summary: "
        f"total={len(results)}, pass={summary['PASS']}, "
        f"fail={summary['FAIL']}, error={summary['ERROR']}"
    )
    print(f"Assessment: {assessment}")
    print(f"Report: {report_path}")


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report_dir).resolve()
    collection = None
    server_version = "unknown"

    try:
        connections.connect(
            alias=CONNECTION_ALIAS,
            host=args.host,
            port=args.port,
        )
        try:
            server_version = utility.get_server_version(using=CONNECTION_ALIAS)
        except Exception:
            server_version = "unknown"

        collection = create_probe_collection(args.collection)
        results = [run_case(collection, case) for case in TEST_CASES]
        report_path = write_report(
            report_dir=report_dir,
            host=args.host,
            port=args.port,
            collection_name=args.collection,
            server_version=server_version,
            results=results,
        )
        print_runtime_summary(results, report_path)
        return 0 if overall_assessment(results) == "FIXED_FOR_COVERED_CASES" else 1
    finally:
        if collection is not None and not args.keep_collection:
            try:
                ensure_collection_absent(args.collection)
            except Exception:
                pass
        try:
            connections.disconnect(CONNECTION_ALIAS)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
