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
DEFAULT_PORT = os.getenv("MILVUS_PORT", "19531")
DEFAULT_COLLECTION = "in_operator_validation"


def query_ids(collection: Collection, expr: str) -> list[int]:
    rows = collection.query(expr, output_fields=["id"], consistency_level="Strong", timeout=20)
    return sorted(int(row["id"]) for row in rows)


def build_rows() -> list[dict]:
    return [
        {
            "id": 1,
            "age": 20,
            "role": "admin",
            "active": True,
            "meta": {"role": "admin", "num": 10},
            "vec": [0.0, 0.0],
        },
        {
            "id": 2,
            "age": 30,
            "role": "user",
            "active": False,
            "meta": {"role": "user", "num": 9223372036854775800},
            "vec": [1.0, 1.0],
        },
        {
            "id": 3,
            "age": 40,
            "role": "guest",
            "active": True,
            "meta": {"team": "ops"},
            "vec": [2.0, 2.0],
        },
        {
            "id": 4,
            "age": None,
            "role": None,
            "active": None,
            "meta": None,
            "vec": [3.0, 3.0],
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus IN operator validation")
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
            FieldSchema(name="age", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=32, nullable=True),
            FieldSchema(name="active", dtype=DataType.BOOL, nullable=True),
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        schema = CollectionSchema(fields)
        col = Collection(args.collection, schema)

        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=20)
        col.load(timeout=20)
        time.sleep(0.4)

        direct_cases = [
            ("scalar_varchar_in_guarded", '(role is not null and (role in ["admin", "guest"]))', [1, 3]),
            ("scalar_int_in_guarded", "(age is not null and (age in [20, 30]))", [1, 2]),
            ("scalar_bool_in_guarded", "(active is not null and (active in [true]))", [1, 3]),
            ("json_role_in_direct", 'meta["role"] in ["admin"]', [1]),
            ("json_missing_key_filtered", 'meta["role"] in ["guest"]', []),
            ("json_numeric_in_exact", 'meta["num"] in [9223372036854775800]', [2]),
            ("json_numeric_in_nonmatch", 'meta["num"] in [9223372036854775807]', []),
        ]

        equiv_pairs = [
            (
                "age_in_or_equiv",
                "(age is not null and age in [20, 30])",
                "(age is not null and (age == 20 or age == 30))",
            ),
            (
                "role_in_or_equiv",
                '(role is not null and role in ["admin", "guest"])',
                '(role is not null and (role == "admin" or role == "guest"))',
            ),
            (
                "json_role_in_or_equiv",
                'meta["role"] in ["admin", "user"]',
                '(meta["role"] == "admin" or meta["role"] == "user")',
            ),
        ]

        print("--- IN operator direct checks ---")
        for name, expr, expected in direct_cases:
            actual = query_ids(col, expr)
            ok = actual == expected
            print(f"{name}: {'PASS' if ok else 'FAIL'} | expr={expr} | expected={expected} | actual={actual}")
            if not ok:
                failures += 1

        print("--- IN operator equivalence checks ---")
        for name, lhs, rhs in equiv_pairs:
            lhs_ids = query_ids(col, lhs)
            rhs_ids = query_ids(col, rhs)
            ok = lhs_ids == rhs_ids
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} | lhs={lhs_ids} | rhs={rhs_ids} | "
                f"lhs_expr={lhs} | rhs_expr={rhs}"
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
        print(f"Summary: FAIL ({failures} IN checks failed)")
        return 1
    print("Summary: PASS (all IN checks consistent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
