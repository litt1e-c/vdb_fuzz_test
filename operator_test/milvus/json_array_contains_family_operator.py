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
DEFAULT_COLLECTION = "json_array_contains_family_validation"


def query_ids(collection: Collection, expr: str) -> list[int]:
    rows = collection.query(
        expr=expr,
        output_fields=["id"],
        consistency_level="Strong",
        timeout=20,
    )
    return sorted(int(row["id"]) for row in rows)


def build_rows() -> list[dict]:
    return [
        {
            "id": 1,
            "meta": {
                "tags": ["sale", "new"],
                "nums": [1, 2, 3],
                "pairs": [[1, 2], [3, 4]],
                "scalar_tag": "sale",
            },
            "arr_i": [1, 2, 3],
            "arr_s": ["red", "blue"],
            "vec": [0.0, 0.0],
        },
        {
            "id": 2,
            "meta": {
                "tags": ["clearance"],
                "nums": [2],
                "pairs": [[1, 2]],
                "scalar_tag": "clearance",
            },
            "arr_i": [2],
            "arr_s": ["blue"],
            "vec": [1.0, 1.0],
        },
        {
            "id": 3,
            "meta": {"tags": [], "nums": [], "pairs": [], "scalar_tag": ""},
            "arr_i": [],
            "arr_s": [],
            "vec": [2.0, 2.0],
        },
        {
            "id": 4,
            "meta": {"other": "x"},
            "arr_i": None,
            "arr_s": None,
            "vec": [3.0, 3.0],
        },
        {
            "id": 5,
            "meta": {"tags": None, "nums": None, "pairs": None, "scalar_tag": None},
            "arr_i": [1, 2],
            "arr_s": ["red", "green"],
            "vec": [4.0, 4.0],
        },
        {"id": 6, "meta": None, "arr_i": None, "arr_s": None, "vec": [5.0, 5.0]},
        {
            "id": 7,
            "meta": {"tags": "sale", "nums": 1, "pairs": "[1,2]", "scalar_tag": "sale"},
            "arr_i": [9],
            "arr_s": ["yellow"],
            "vec": [6.0, 6.0],
        },
        {
            "id": 8,
            "meta": {
                "mixed": [1, "1", True],
                "pairs": [[1, 2], [9, 9]],
                "tags": ["vip"],
                "nums": [8, 9],
                "scalar_tag": "vip",
            },
            "arr_i": [1, 9],
            "arr_s": ["red", "yellow"],
            "vec": [7.0, 7.0],
        },
    ]


def build_cases() -> list[tuple[str, str, list[int]]]:
    return [
        ("json_contains_string", 'json_contains(meta["tags"], "sale")', [1]),
        ("json_contains_numeric", 'json_contains(meta["nums"], 2)', [1, 2]),
        ("json_contains_subarray", 'json_contains(meta["pairs"], [1,2])', [1, 2, 8]),
        ("json_contains_any_string", 'json_contains_any(meta["tags"], ["sale", "clearance"])', [1, 2]),
        ("json_contains_any_numeric", 'json_contains_any(meta["nums"], [2, 9])', [1, 2, 8]),
        ("json_contains_any_mixed_query", 'json_contains_any(meta["mixed"], [1, "ghost"])', [8]),
        ("json_contains_all_string", 'json_contains_all(meta["tags"], ["sale", "new"])', [1]),
        ("json_contains_all_numeric", 'json_contains_all(meta["nums"], [1, 2])', [1]),
        ("json_contains_all_mixed_query", 'json_contains_all(meta["mixed"], [1, "1"])', [8]),
        ("json_missing_key_filtered", 'json_contains(meta["missing"], 1)', []),
        ("json_non_array_filtered", 'json_contains(meta["scalar_tag"], "sale")', []),
        ("array_contains_int", "array_contains(arr_i, 2)", [1, 2, 5]),
        ("array_contains_any_int", "array_contains_any(arr_i, [1, 9])", [1, 5, 7, 8]),
        ("array_contains_all_int", "array_contains_all(arr_i, [1, 2])", [1, 5]),
        ("array_contains_string", 'array_contains(arr_s, "red")', [1, 5, 8]),
        ("array_contains_any_string", 'array_contains_any(arr_s, ["green", "yellow"])', [5, 7, 8]),
        ("array_contains_all_string", 'array_contains_all(arr_s, ["red", "yellow"])', [8]),
        ("array_empty_filtered", "array_contains(arr_i, 1)", [1, 5, 8]),
    ]


def create_collection(name: str) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="arr_i", dtype=DataType.ARRAY, element_type=DataType.INT32, max_capacity=8, nullable=True),
        FieldSchema(name="arr_s", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=8, max_length=32, nullable=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    return Collection(name, CollectionSchema(fields))


def run_phase(collection: Collection, phase: str, baseline: dict[str, list[int]] | None = None) -> tuple[int, dict[str, list[int]]]:
    failures = 0
    phase_results: dict[str, list[int]] = {}
    for name, expr, expected_ids in build_cases():
        try:
            actual_ids = query_ids(collection, expr)
            phase_results[name] = actual_ids
            ok = actual_ids == expected_ids
            if baseline is not None:
                ok = ok and actual_ids == baseline.get(name)
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} | phase={phase} | "
                f"expr={expr} | expected={expected_ids} | actual={actual_ids}"
            )
            if not ok:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"{name}: ERROR | phase={phase} | expr={expr} | error={type(exc).__name__}: {exc}")
    return failures, phase_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus JSON/ARRAY contains-family validation")
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

        col = create_collection(args.collection)
        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=30)
        col.load(timeout=30)
        time.sleep(0.4)

        print("--- JSON/ARRAY contains-family validation: baseline ---")
        phase_failures, baseline = run_phase(col, "baseline")
        failures += phase_failures

        for field_name, params, index_name in [
            ("arr_i", {"index_type": "BITMAP"}, "arr_i_bitmap_idx"),
            ("arr_s", {"index_type": "INVERTED"}, "arr_s_inverted_idx"),
            (
                "meta",
                {"index_type": "INVERTED", "params": {"json_cast_type": "Json", "json_path": "meta['pairs']"}},
                "meta_pairs_json_idx",
            ),
            (
                "meta",
                {"index_type": "INVERTED", "params": {"json_cast_type": "Json", "json_path": "meta['tags']"}},
                "meta_tags_json_idx",
            ),
        ]:
            try:
                col.create_index(field_name, params, index_name=index_name, timeout=60)
            except Exception as exc:
                print(f"index_build_warn: field={field_name} index={index_name} error={type(exc).__name__}: {exc}")

        time.sleep(0.4)
        print("--- JSON/ARRAY contains-family validation: after array/json indexes ---")
        phase_failures, _ = run_phase(col, "indexed_loaded", baseline)
        failures += phase_failures

        col.release(timeout=30)
        col.load(timeout=30)
        time.sleep(0.4)
        print("--- JSON/ARRAY contains-family validation: after reload ---")
        phase_failures, _ = run_phase(col, "reloaded", baseline)
        failures += phase_failures
    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception:
            pass
        connections.disconnect("default")

    if failures:
        print(f"Summary: FAIL ({failures} JSON/ARRAY contains-family mismatches/errors)")
        return 1
    print("Summary: PASS (JSON/ARRAY contains-family checks stable across baseline/index/reload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
