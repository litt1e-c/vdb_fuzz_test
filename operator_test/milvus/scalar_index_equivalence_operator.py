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
DEFAULT_COLLECTION = "scalar_index_equivalence_validation"


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
        {"id": 1, "age": 10, "score": 1.0, "flag": True, "meta": {"role": "admin", "a": {"b": 101}}, "vec": [0.0, 0.0]},
        {"id": 2, "age": 15, "score": 1.5, "flag": False, "meta": {"role": "user", "a": {"b": 150}}, "vec": [0.1, 0.1]},
        {"id": 3, "age": 20, "score": 2.0, "flag": True, "meta": {"role": "ops", "a": {"b": None}}, "vec": [0.2, 0.2]},
        {"id": 4, "age": 25, "score": 2.5, "flag": None, "meta": {"role": "guest", "a": [{"b": 1}, 2, 3]}, "vec": [0.3, 0.3]},
        {"id": 5, "age": 30, "score": None, "flag": True, "meta": {}, "vec": [0.4, 0.4]},
        {"id": 6, "age": None, "score": 3.5, "flag": False, "meta": None, "vec": [0.5, 0.5]},
        {"id": 7, "age": 35, "score": 4.0, "flag": True, "meta": {"role": "ops", "a": {"b": 205}}, "vec": [0.6, 0.6]},
        {"id": 8, "age": 40, "score": 4.5, "flag": False, "meta": {"role": "admin", "a": {"b": 99}}, "vec": [0.7, 0.7]},
    ]


def build_exprs() -> list[tuple[str, str, list[int]]]:
    return [
        ("age_range", "(age is not null and (age >= 10 and age <= 30))", [1, 2, 3, 4, 5]),
        ("flag_true", "(flag is not null and (flag == true))", [1, 3, 5, 7]),
        ("age_membership", "(age is not null and (age in [10, 15, 40]))", [1, 2, 8]),
        ("json_role_membership", 'meta["role"] in ["admin", "ops"]', [1, 3, 7, 8]),
        ("json_nested_range", 'meta["a"]["b"] >= 100', [1, 2, 7]),
        ("json_nested_null", 'meta["a"]["b"] is null', [3, 4, 5, 6]),
        ("json_nested_exists", 'exists(meta["a"]["b"])', [1, 2, 7, 8]),
        ("json_array_elem_eq", 'meta["a"][0]["b"] == 1', [4]),
        ("json_array_elem_exists", 'exists(meta["a"][0]["b"])', [4]),
        ("json_array_elem_null", 'meta["a"][0]["b"] is null', [1, 2, 3, 5, 6, 7, 8]),
    ]


def create_collection(name: str) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="age", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="score", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="flag", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    return Collection(name, CollectionSchema(fields))


def run_phase(
    collection: Collection,
    tests: list[tuple[str, str, list[int]]],
    phase: str,
    baseline: dict[str, list[int]] | None = None,
) -> int:
    mismatches = 0
    for name, expr, expected in tests:
        actual = query_ids(collection, expr)
        ok = actual == expected
        if baseline is not None:
            ok = ok and actual == baseline.get(name)
        print(
            f"{'PASS' if ok else 'FAIL'} | phase={phase} | name={name} | "
            f"expr={expr} | expected={expected} | actual={actual}"
        )
        if not ok:
            mismatches += 1
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus scalar index/no-index equivalence probe")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args(argv)

    try:
        connections.connect("default", host=args.host, port=args.port, timeout=20)
    except Exception as exc:
        print(f"connection_failed: {exc}")
        return 2

    mismatches = 0
    try:
        if utility.has_collection(args.collection):
            utility.drop_collection(args.collection, timeout=20)

        col = create_collection(args.collection)
        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=30)
        col.load(timeout=30)
        time.sleep(0.5)

        tests = build_exprs()
        baseline: dict[str, list[int]] = {}

        print("--- Scalar index equivalence validation: baseline ---")
        mismatches += run_phase(col, tests, "baseline")
        for name, expr, _ in tests:
            baseline[name] = query_ids(col, expr)

        index_plan = [
            ("age", {"index_type": "INVERTED"}, None),
            ("flag", {"index_type": "BITMAP"}, None),
            (
                "meta",
                {"index_type": "INVERTED", "params": {"json_cast_type": "Varchar", "json_path": "meta['role']"}},
                "meta_role_varchar_idx",
            ),
            (
                "meta",
                {"index_type": "INVERTED", "params": {"json_cast_type": "Double", "json_path": "meta['a']['b']"}},
                "meta_a_b_double_idx",
            ),
            (
                "meta",
                {"index_type": "INVERTED", "params": {"json_cast_type": "Double", "json_path": "meta['a'][0]['b']"}},
                "meta_a_0_b_double_idx",
            ),
        ]
        for field_name, params, index_name in index_plan:
            try:
                create_kwargs = {"timeout": 60}
                if index_name is not None:
                    create_kwargs["index_name"] = index_name
                col.create_index(field_name, params, **create_kwargs)
            except Exception as exc:
                print(f"index_build_warn: field={field_name} error={type(exc).__name__}: {exc}")

        time.sleep(0.5)
        print("--- Scalar index equivalence validation: after index build ---")
        mismatches += run_phase(col, tests, "indexed_loaded", baseline)

        col.release(timeout=30)
        col.load(timeout=30)
        time.sleep(0.5)
        print("--- Scalar index equivalence validation: after reload ---")
        mismatches += run_phase(col, tests, "reloaded", baseline)
    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception:
            pass
        connections.disconnect("default")

    if mismatches:
        print(f"Summary: FAIL ({mismatches} scalar index mismatch cases)")
        return 1
    print("Summary: PASS (scalar, JSON-path, and reload equivalence checks all matched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
