#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


DEFAULT_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
DEFAULT_PORT = os.getenv("MILVUS_PORT", "19630")
DEFAULT_COLLECTION = "logical_metamorphic_validation"


def query_ids(collection: Collection, expr: str) -> list[int]:
    rows = collection.query(
        expr,
        output_fields=["id"],
        consistency_level="Strong",
        timeout=20,
    )
    return sorted(int(row["id"]) for row in rows)


def build_rows() -> list[dict]:
    return [
        {"id": 1, "x": 5, "flag": True, "vec": [0.0, 0.0]},
        {"id": 2, "x": 10, "flag": False, "vec": [0.1, 0.1]},
        {"id": 3, "x": 15, "flag": True, "vec": [0.2, 0.2]},
        {"id": 4, "x": 20, "flag": False, "vec": [0.3, 0.3]},
        {"id": 5, "x": 30, "flag": True, "vec": [0.4, 0.4]},
        {"id": 6, "x": 40, "flag": False, "vec": [0.5, 0.5]},
        {"id": 7, "x": None, "flag": True, "vec": [0.6, 0.6]},
        {"id": 8, "x": 25, "flag": None, "vec": [0.7, 0.7]},
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus logical/metamorphic scalar validation")
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

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="x", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="flag", dtype=DataType.BOOL, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(args.collection, CollectionSchema(fields))
        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=30)
        col.load(timeout=30)
        time.sleep(0.4)

        A = "(x is not null and (x >= 10 and x <= 30))"
        true_expr = "(id >= 1)"
        false_expr = "(id < 0)"

        equivalent_pairs = [
            ("and_true_equiv", A, f"(({A}) and ({true_expr}))"),
            ("or_false_equiv", A, f"(({A}) or ({false_expr}))"),
            ("double_not_equiv", A, f"(not (not ({A})))"),
            (
                "in_disjunction_equiv",
                "(x is not null and x in [10, 15, 30])",
                "(x is not null and ((x == 10) or (x == 15) or (x == 30)))",
            ),
            (
                "not_in_notin_equiv",
                "(x is not null and x not in [10, 15, 30])",
                "(x is not null and not (x in [10, 15, 30]))",
            ),
            (
                "precedence_or_and_equiv",
                "((flag == true) or ((flag == false) and (x > 20)))",
                "(flag == true or flag == false and x > 20)",
            ),
        ]

        direct_expectations = [
            ("contradiction_empty", f"(({A}) and (not ({A})))", []),
            ("tautology_full_nonnull_guard", f"((x is not null) and (({A}) or (not ({A}))))", [1, 2, 3, 4, 5, 6, 8]),
        ]

        print("--- logical metamorphic equivalence checks ---")
        for name, lhs, rhs in equivalent_pairs:
            lhs_ids = query_ids(col, lhs)
            rhs_ids = query_ids(col, rhs)
            ok = lhs_ids == rhs_ids
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} | "
                f"lhs={lhs_ids} | rhs={rhs_ids} | lhs_expr={lhs} | rhs_expr={rhs}"
            )
            if not ok:
                failures += 1

        print("--- logical direct expectation checks ---")
        for name, expr, expected in direct_expectations:
            actual = query_ids(col, expr)
            ok = actual == expected
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} | "
                f"expected={expected} | actual={actual} | expr={expr}"
            )
            if not ok:
                failures += 1

    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception:
            pass
        connections.disconnect("default")

    if failures:
        print(f"Summary: FAIL ({failures} logical/metamorphic mismatches)")
        return 1
    print("Summary: PASS (all logical/metamorphic checks consistent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
