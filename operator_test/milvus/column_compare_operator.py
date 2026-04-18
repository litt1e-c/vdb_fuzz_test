#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


DEFAULT_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
DEFAULT_PORT = os.getenv("MILVUS_PORT", "19630")
DEFAULT_COLLECTION = "column_compare_validation"


def query_ids(collection: Collection, expr: str) -> list[int]:
    rows = collection.query(
        expr=expr,
        output_fields=["id"],
        consistency_level="Strong",
        timeout=20,
    )
    return sorted(int(row["id"]) for row in rows)


def build_rows() -> list[dict[str, Any]]:
    return [
        {"id": 1, "a": 1, "b": 2, "x": 1.0, "y": 2.0, "s1": "a", "s2": "b", "p": True, "q": True, "vec": [0.0, 0.0]},
        {"id": 2, "a": 2, "b": 1, "x": 2.0, "y": 1.0, "s1": "b", "s2": "a", "p": True, "q": False, "vec": [0.1, 0.1]},
        {"id": 3, "a": 2, "b": 2, "x": 2.0, "y": 2.0, "s1": "a", "s2": "a", "p": False, "q": False, "vec": [0.2, 0.2]},
        {"id": 4, "a": None, "b": 2, "x": None, "y": 2.0, "s1": None, "s2": "a", "p": None, "q": False, "vec": [0.3, 0.3]},
        {"id": 5, "a": 3, "b": None, "x": 3.0, "y": None, "s1": "a", "s2": None, "p": True, "q": None, "vec": [0.4, 0.4]},
        {"id": 6, "a": -5, "b": -2, "x": -0.0, "y": 0.0, "s1": "北京", "s2": "北京大学", "p": False, "q": True, "vec": [0.5, 0.5]},
    ]


def build_tests() -> list[tuple[str, str, list[int]]]:
    return [
        ("int_lt", "a < b", [1, 6]),
        ("int_gt", "a > b", [2]),
        ("int_eq", "a == b", [3]),
        ("int_ne", "a != b", [1, 2, 6]),
        ("double_lt", "x < y", [1]),
        ("double_gt", "x > y", [2]),
        ("double_eq_signed_zero", "x == y", [3, 6]),
        ("varchar_lt", "s1 < s2", [1, 6]),
        ("varchar_gt", "s1 > s2", [2]),
        ("varchar_eq", "s1 == s2", [3]),
        ("varchar_ne", "s1 != s2", [1, 2, 6]),
        ("bool_eq", "p == q", [1, 3]),
        ("bool_ne", "p != q", [2, 6]),
        ("bool_order", "p > q", [2]),
        ("logical_and", "a < b and s1 < s2", [1, 6]),
        ("logical_or", "a < b or s1 == s2", [1, 3, 6]),
        ("not_lt_3vl", "not (a < b)", [2, 3]),
        ("null_guarded_ne", "a is not null and b is not null and a != b", [1, 2, 6]),
    ]


def create_collection(name: str) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="a", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="b", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="x", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="y", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="s1", dtype=DataType.VARCHAR, max_length=128, nullable=True),
        FieldSchema(name="s2", dtype=DataType.VARCHAR, max_length=128, nullable=True),
        FieldSchema(name="p", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="q", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    return Collection(name, CollectionSchema(fields))


def run_phase(
    collection: Collection,
    tests: list[tuple[str, str, list[int]]],
    phase: str,
    baseline: dict[str, list[int]] | None = None,
) -> tuple[int, dict[str, list[int]]]:
    observed: dict[str, list[int]] = {}
    failures = 0
    for name, expr, expected in tests:
        actual = query_ids(collection, expr)
        observed[name] = actual
        reference = expected if baseline is None else baseline[name]
        if actual != reference:
            failures += 1
            print(f"mismatch phase={phase} case={name} expr={expr}")
            print(f"  expected={reference}")
            print(f"  actual={actual}")
    return failures, observed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus column-to-column comparison validation")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args(argv)

    try:
        connections.connect("default", host=args.host, port=args.port, timeout=20)
    except Exception as exc:
        print(f"connection_failed: {exc}")
        return 2

    failures = 0
    try:
        if utility.has_collection(args.collection):
            utility.drop_collection(args.collection, timeout=20)

        collection = create_collection(args.collection)
        collection.insert(build_rows())
        collection.flush(timeout=20)
        collection.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=30)
        collection.load(timeout=30)
        time.sleep(0.4)

        tests = build_tests()
        phase_failures, baseline = run_phase(collection, tests, "baseline")
        failures += phase_failures

        for field in ("a", "b", "x", "y", "s1", "s2", "p", "q"):
            try:
                collection.create_index(field, {"index_type": "INVERTED"}, index_name=f"idx_{field}", timeout=30)
            except Exception as exc:
                failures += 1
                print(f"index_create_failed field={field}: {exc}")

        phase_failures, _ = run_phase(collection, tests, "post_inverted_index", baseline)
        failures += phase_failures

        collection.release(timeout=20)
        collection.load(timeout=30)
        time.sleep(0.4)
        phase_failures, _ = run_phase(collection, tests, "post_reload", baseline)
        failures += phase_failures
    except Exception as exc:
        print(f"operator_failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception as cleanup_exc:
            print(f"cleanup_warning: {cleanup_exc}")
        connections.disconnect("default")

    if failures:
        print(f"column_compare: FAIL mismatches={failures}")
        return 1
    print("column_compare: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
