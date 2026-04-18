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
DEFAULT_COLLECTION = "dynamic_field_operator_validation"


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
            "vec": [0.0, 0.0],
            "words": 150,
            "overview": "Great product",
            "dynamic_json": {"nested": {"value": 42}},
        },
        {
            "id": 2,
            "vec": [1.0, 1.0],
            "words": 80,
            "overview": "Bad product",
            "dynamic_json": {"nested": {"value": 99}},
        },
        {
            "id": 3,
            "vec": [2.0, 2.0],
            "overview": "Missing words",
            "dynamic_json": {"nested": {"value": None}},
        },
        {
            "id": 4,
            "vec": [3.0, 3.0],
            "words": None,
            "dynamic_json": {},
        },
        {
            "id": 5,
            "vec": [4.0, 4.0],
            "words": 100,
            "dynamic_json": {"nested": {}},
        },
        {
            "id": 6,
            "vec": [5.0, 5.0],
            "words": 101,
            "dynamic_json": None,
        },
    ]


def build_tests() -> list[tuple[str, str, str, list[int]]]:
    return [
        (
            "words_direct_vs_meta",
            'words >= 100',
            '$meta["words"] >= 100',
            [1, 5, 6],
        ),
        (
            "overview_direct_vs_meta",
            'overview == "Great product"',
            '$meta["overview"] == "Great product"',
            [1],
        ),
        (
            "nested_compare_direct_vs_meta",
            'dynamic_json["nested"]["value"] < 50',
            '$meta["dynamic_json"]["nested"]["value"] < 50',
            [1],
        ),
        (
            "nested_is_null_direct_vs_meta",
            'dynamic_json["nested"]["value"] is null',
            '$meta["dynamic_json"]["nested"]["value"] is null',
            [3, 4, 5, 6],
        ),
        (
            "nested_exists_direct_vs_meta",
            'exists(dynamic_json["nested"]["value"])',
            'exists($meta["dynamic_json"]["nested"]["value"])',
            [1, 2],
        ),
    ]


def run_phase(
    collection: Collection,
    tests: list[tuple[str, str, str, list[int]]],
    phase: str,
    baseline: dict[str, tuple[list[int], list[int]]] | None = None,
) -> tuple[int, dict[str, tuple[list[int], list[int]]]]:
    failures = 0
    phase_results: dict[str, tuple[list[int], list[int]]] = {}
    for name, expr_direct, expr_meta, expected_ids in tests:
        try:
            direct_ids = query_ids(collection, expr_direct)
            meta_ids = query_ids(collection, expr_meta)
            phase_results[name] = (direct_ids, meta_ids)
            ok = direct_ids == meta_ids == expected_ids
            if baseline is not None:
                ok = ok and phase_results[name] == baseline.get(name)
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} | phase={phase} | "
                f"direct={expr_direct} -> {direct_ids} | "
                f"meta={expr_meta} -> {meta_ids} | expected={expected_ids}"
            )
            if not ok:
                failures += 1
        except Exception as exc:
            failures += 1
            print(
                f"{name}: ERROR | phase={phase} | direct={expr_direct} | meta={expr_meta} | "
                f"error={type(exc).__name__}: {exc}"
            )
    return failures, phase_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus dynamic-field semantic validation")
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
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(args.collection, CollectionSchema(fields, enable_dynamic_field=True))
        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=20)
        col.load(timeout=20)
        time.sleep(0.4)

        tests = build_tests()
        print("--- Dynamic field validation: baseline ---")
        phase_failures, baseline = run_phase(col, tests, "baseline")
        failures += phase_failures

        index_plan = [
            (
                "$meta",
                {
                    "index_type": "INVERTED",
                    "params": {"json_cast_type": "Double", "json_path": "words"},
                },
                "dyn_words_idx",
            ),
            (
                "$meta",
                {
                    "index_type": "INVERTED",
                    "params": {"json_cast_type": "Varchar", "json_path": "overview"},
                },
                "dyn_overview_idx",
            ),
            (
                "$meta",
                {
                    "index_type": "INVERTED",
                    "params": {"json_cast_type": "Double", "json_path": "dynamic_json['nested']['value']"},
                },
                "dyn_nested_value_idx",
            ),
        ]
        for field_name, params, index_name in index_plan:
            try:
                col.create_index(field_name, params, index_name=index_name, timeout=60)
            except Exception as exc:
                failures += 1
                print(
                    f"index_build_error: phase=index_create | field={field_name} | "
                    f"index_name={index_name} | error={type(exc).__name__}: {exc}"
                )

        time.sleep(0.4)
        print("--- Dynamic field validation: after index build ---")
        phase_failures, _ = run_phase(col, tests, "indexed_loaded", baseline)
        failures += phase_failures

        col.release(timeout=20)
        col.load(timeout=20)
        time.sleep(0.4)
        print("--- Dynamic field validation: after reload ---")
        phase_failures, _ = run_phase(col, tests, "reloaded", baseline)
        failures += phase_failures
    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception:
            pass
        connections.disconnect("default")

    if failures:
        print(f"Summary: FAIL ({failures} dynamic-field mismatches)")
        return 1
    print("Summary: PASS (dynamic-field direct/meta and index/reload checks consistent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
