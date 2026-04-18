#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


DEFAULT_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
DEFAULT_PORT = os.getenv("MILVUS_PORT", "19630")
DEFAULT_COLLECTION = "templated_unary_not_validation"


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


def build_rows() -> list[dict[str, Any]]:
    return [
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus templated unary-NOT operator validation")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args(argv)

    try:
        connections.connect("default", host=args.host, port=args.port, timeout=20)
    except Exception as exc:
        print(f"connection_failed: {exc}")
        return 2

    hard_failures = 0
    known_bug_hits = 0
    try:
        if utility.has_collection(args.collection):
            utility.drop_collection(args.collection, timeout=20)

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
        col = Collection(args.collection, CollectionSchema(fields))
        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=20)
        col.load(timeout=20)
        time.sleep(0.4)

        direct_controls = [
            ("templated_in_ok", "int_field in [10, 20, 99]", "int_field in {vals}", {"vals": [10, 20, 99]}),
            ("templated_not_in_ok", "int_field not in [10, 20, 99]", "int_field not in {vals}", {"vals": [10, 20, 99]}),
            ("templated_gt_ok", "int_field > 15", "int_field > {x}", {"x": 15}),
            (
                "templated_compound_ok",
                '(int_field > 15) and (role == "user")',
                '(int_field > {x}) and (role == "user")',
                {"x": 15},
            ),
            ("templated_json_ok", 'meta["num"] > 15', 'meta["num"] > {x}', {"x": 15}),
            ("templated_array_length_ok", "array_length(arr_i) > 1", "array_length(arr_i) > {n}", {"n": 1}),
            (
                "templated_json_contains_any_ok",
                'json_contains_any(meta["nums"], [2, 9])',
                'json_contains_any(meta["nums"], {vals})',
                {"vals": [2, 9]},
            ),
            (
                "templated_array_contains_any_ok",
                "array_contains_any(arr_i, [2, 9])",
                "array_contains_any(arr_i, {vals})",
                {"vals": [2, 9]},
            ),
            (
                "templated_json_contains_ok",
                'json_contains(meta["tags"], "admin")',
                'json_contains(meta["tags"], {v})',
                {"v": "admin"},
            ),
        ]

        known_issue_probes = [
            (
                "templated_unary_not_in_bug",
                "not (int_field in [10, 20, 99])",
                "not (int_field in {vals})",
                {"vals": [10, 20, 99]},
            ),
            (
                "templated_unary_not_gt_bug",
                "not (int_field > 15)",
                "not (int_field > {x})",
                {"x": 15},
            ),
            (
                "templated_unary_not_range_bug",
                "not (25 > int_field > 15)",
                "not ({hi} > int_field > {lo})",
                {"lo": 15, "hi": 25},
            ),
            (
                "templated_unary_not_compound_bug",
                'not ((int_field > 15) and (role == "user"))',
                'not ((int_field > {x}) and (role == "user"))',
                {"x": 15},
            ),
            (
                "templated_unary_not_json_bug",
                'not (meta["num"] > 15)',
                'not (meta["num"] > {x})',
                {"x": 15},
            ),
            (
                "templated_double_not_in_bug",
                'not (not (int_field in [10, 20, 99]))',
                'not (not (int_field in {vals}))',
                {"vals": [10, 20, 99]},
            ),
            (
                "templated_unary_not_array_length_bug",
                "not (array_length(arr_i) > 1)",
                "not (array_length(arr_i) > {n})",
                {"n": 1},
            ),
            (
                "templated_unary_not_json_contains_any_bug",
                'not (json_contains_any(meta["nums"], [2, 9]))',
                'not (json_contains_any(meta["nums"], {vals}))',
                {"vals": [2, 9]},
            ),
            (
                "templated_unary_not_array_contains_any_bug",
                "not (array_contains_any(arr_i, [2, 9]))",
                "not (array_contains_any(arr_i, {vals}))",
                {"vals": [2, 9]},
            ),
            (
                "templated_unary_not_json_contains_bug",
                'not (json_contains(meta["tags"], "admin"))',
                'not (json_contains(meta["tags"], {v}))',
                {"v": "admin"},
            ),
        ]

        print("--- templated expression direct controls ---")
        for name, literal_expr, templ_expr, params in direct_controls:
            literal_ids = query_ids(col, literal_expr)
            templated_ids = query_ids(col, templ_expr, params)
            ok = literal_ids == templated_ids
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} | literal={literal_ids} | templated={templated_ids} | "
                f"literal_expr={literal_expr} | templ_expr={templ_expr} | params={params}"
            )
            if not ok:
                hard_failures += 1

        print("--- templated unary-NOT known-issue probes ---")
        for name, literal_expr, templ_expr, params in known_issue_probes:
            literal_ids = query_ids(col, literal_expr)
            try:
                templated_ids = query_ids(col, templ_expr, params)
                if templated_ids == literal_ids:
                    print(
                        f"{name}: FIXED_OR_NOT_REPRO | literal={literal_ids} | templated={templated_ids} | "
                        f"literal_expr={literal_expr} | templ_expr={templ_expr} | params={params}"
                    )
                else:
                    known_bug_hits += 1
                    print(
                        f"{name}: KNOWN_BUG_REPRODUCED | literal={literal_ids} | templated={templated_ids} | "
                        f"literal_expr={literal_expr} | templ_expr={templ_expr} | params={params}"
                    )
            except Exception as exc:
                known_bug_hits += 1
                print(
                    f"{name}: KNOWN_BUG_REPRODUCED | literal={literal_ids} | templ_expr={templ_expr} | "
                    f"params={params} | error={type(exc).__name__}: {exc}"
                )

    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception:
            pass
        connections.disconnect("default")

    if hard_failures:
        print(f"Summary: FAIL ({hard_failures} templated control checks failed)")
        return 1
    print(f"Summary: PASS (normative controls passed, known_bug_reproduced={known_bug_hits})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
