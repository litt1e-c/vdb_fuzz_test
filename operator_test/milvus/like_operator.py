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
DEFAULT_COLLECTION = "like_operator_validation"


def query_ids(collection: Collection, expr: str) -> list[int]:
    rows = collection.query(expr, output_fields=["id"], consistency_level="Strong", timeout=20)
    return sorted(int(row["id"]) for row in rows)


def build_rows() -> list[dict]:
    return [
        {"id": 1, "role": "admin", "meta": {"role": "admin"}, "tags": ["admin", "ops"], "vec": [0.0, 0.0]},
        {"id": 2, "role": "user", "meta": {"role": "user"}, "tags": ["user", "eng"], "vec": [1.0, 1.0]},
        {"id": 3, "role": "guest", "meta": {"role": "guest"}, "tags": ["guest", "sales"], "vec": [2.0, 2.0]},
        {"id": 4, "role": None, "meta": {"team": "ops"}, "tags": ["ops"], "vec": [3.0, 3.0]},
        {"id": 5, "role": None, "meta": None, "tags": None, "vec": [4.0, 4.0]},
        {"id": 6, "role": "alpha", "meta": {"role": None}, "tags": ["alpha"], "vec": [5.0, 5.0]},
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus LIKE operator validation")
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
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=32, nullable=True),
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(
                name="tags",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=4,
                max_length=32,
                nullable=True,
            ),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        schema = CollectionSchema(fields)
        col = Collection(args.collection, schema)

        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=20)
        col.load(timeout=20)
        time.sleep(0.4)

        tests = [
            ("scalar_prefix_like", '(role is not null and (role like "ad%"))', [1]),
            ("scalar_suffix_like", '(role is not null and (role like "%er"))', [2]),
            ("scalar_infix_like", '(role is not null and (role like "%ues%"))', [3]),
            ("scalar_underscore_like", '(role is not null and (role like "u_er"))', [2]),
            ("json_prefix_like_direct", 'meta["role"] like "ad%"', [1]),
            ("json_suffix_like_direct", 'meta["role"] like "%er"', [2]),
            ("json_infix_like_direct", 'meta["role"] like "%ues%"', [3]),
            ("json_underscore_like_direct", 'meta["role"] like "u_er"', [2]),
            ("json_missing_key_filtered", 'meta["role"] like "%ops%"', []),
            ("json_null_or_null_field_filtered", 'meta["role"] like "alph%"', []),
            ("json_guarded_not_like", 'meta is not null and not (meta["role"] like "ad%")', [2, 3, 4, 6]),
            ("json_guarded_or_like", '(meta is not null and (meta["role"] like "ad%")) or id == 2', [1, 2]),
            ("array_prefix_like", 'tags[0] like "ad%"', [1]),
            ("array_suffix_like", 'tags[0] like "%min"', [1]),
            ("array_infix_like", 'tags[0] like "%dmi%"', [1]),
            ("array_underscore_like", 'tags[0] like "ad_in"', [1]),
        ]

        print("--- LIKE operator validation ---")
        for name, expr, expected in tests:
            actual = query_ids(col, expr)
            ok = actual == expected
            print(f"{name}: {'PASS' if ok else 'FAIL'} | expr={expr} | expected={expected} | actual={actual}")
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
        print(f"Summary: FAIL ({failures} LIKE checks failed)")
        return 1
    print("Summary: PASS (all LIKE checks consistent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
