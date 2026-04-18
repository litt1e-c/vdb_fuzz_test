#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


DEFAULT_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
DEFAULT_PORT = os.getenv("MILVUS_PORT", "19532")
DEFAULT_COLLECTION = "poc_template_unary_not_expr_params"


def query_ids(collection: Collection, expr: str, expr_params: dict[str, Any] | None = None) -> list[int]:
    kwargs: dict[str, Any] = {
        "expr": expr,
        "output_fields": ["id"],
        "consistency_level": "Strong",
        "timeout": 20,
    }
    if expr_params is not None:
        kwargs["expr_params"] = expr_params
    rows = collection.query(**kwargs)
    return sorted(int(row["id"]) for row in rows)


def build_collection(name: str) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="int_field", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=32, nullable=True),
        FieldSchema(name="arr_i", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=8, nullable=True),
        FieldSchema(
            name="arr_s",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=8,
            max_length=32,
            nullable=True,
        ),
        FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    col = Collection(name, CollectionSchema(fields))
    col.insert(
        [
            {
                "id": 1,
                "int_field": 10,
                "role": "admin",
                "arr_i": [1, 2, 3],
                "arr_s": ["admin", "ops"],
                "meta": {"num": 10, "nums": [1, 2, 3], "tags": ["admin", "ops"]},
                "vec": [0.0, 0.0],
            },
            {
                "id": 2,
                "int_field": 20,
                "role": "user",
                "arr_i": [4, 5],
                "arr_s": ["user", "eng"],
                "meta": {"num": 20, "nums": [4, 5], "tags": ["user", "eng"]},
                "vec": [1.0, 1.0],
            },
            {
                "id": 3,
                "int_field": 30,
                "role": "guest",
                "arr_i": [],
                "arr_s": ["guest"],
                "meta": {"role": "guest", "nums": [], "tags": ["guest"]},
                "vec": [2.0, 2.0],
            },
            {"id": 4, "int_field": None, "role": None, "arr_i": None, "arr_s": None, "meta": None, "vec": [3.0, 3.0]},
            {
                "id": 5,
                "int_field": 15,
                "role": "alpha",
                "arr_i": [2, 6],
                "arr_s": ["alpha"],
                "meta": {"num": 15, "nums": [2, 6], "tags": ["alpha"]},
                "vec": [4.0, 4.0],
            },
        ]
    )
    col.flush(timeout=20)
    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=20)
    col.load(timeout=20)
    time.sleep(0.5)
    return col


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce Milvus templated unary-NOT expr_params bug")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args(argv)

    try:
        connections.connect("default", host=args.host, port=args.port, timeout=20)
    except Exception as exc:
        print(f"connection_failed: {exc}")
        return 2

    reproduced = 0
    try:
        if utility.has_collection(args.collection):
            utility.drop_collection(args.collection, timeout=20)
        col = build_collection(args.collection)

        cases = [
            (
                "int_in_under_not",
                "not (int_field in [10, 20, 99])",
                "not (int_field in {vals})",
                {"vals": [10, 20, 99]},
            ),
            (
                "int_gt_under_not",
                "not (int_field > 15)",
                "not (int_field > {x})",
                {"x": 15},
            ),
            (
                "compound_under_not",
                'not ((int_field > 15) and (role == "user"))',
                'not ((int_field > {x}) and (role == "user"))',
                {"x": 15},
            ),
            (
                "json_under_not",
                'not (meta["num"] > 15)',
                'not (meta["num"] > {x})',
                {"x": 15},
            ),
            (
                "array_length_under_not",
                "not (array_length(arr_i) > 1)",
                "not (array_length(arr_i) > {n})",
                {"n": 1},
            ),
            (
                "json_contains_any_under_not",
                'not (json_contains_any(meta["nums"], [2, 9]))',
                'not (json_contains_any(meta["nums"], {vals}))',
                {"vals": [2, 9]},
            ),
            (
                "array_contains_any_under_not",
                "not (array_contains_any(arr_i, [2, 9]))",
                "not (array_contains_any(arr_i, {vals}))",
                {"vals": [2, 9]},
            ),
            (
                "json_contains_under_not",
                'not (json_contains(meta["tags"], "admin"))',
                'not (json_contains(meta["tags"], {v}))',
                {"v": "admin"},
            ),
        ]

        print("--- Milvus templated unary-NOT PoC ---")
        print(f"target={args.host}:{args.port} collection={args.collection}")

        for name, literal_expr, templ_expr, params in cases:
            literal_ids = query_ids(col, literal_expr)
            try:
                templ_ids = query_ids(col, templ_expr, params)
                same = templ_ids == literal_ids
                status = "MATCH" if same else "MISMATCH"
                print(
                    f"{name}: {status} | literal={literal_ids} | templated={templ_ids} | "
                    f"literal_expr={literal_expr} | templ_expr={templ_expr} | params={params}"
                )
                if not same:
                    reproduced += 1
            except Exception as exc:
                reproduced += 1
                print(
                    f"{name}: ERROR_REPRODUCED | literal={literal_ids} | "
                    f"templ_expr={templ_expr} | params={params} | error={type(exc).__name__}: {exc}"
                )

    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception:
            pass
        connections.disconnect("default")

    if reproduced:
        print(f"Summary: BUG_REPRODUCED ({reproduced} mismatching/error cases)")
        return 0
    print("Summary: NOT_REPRODUCED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
