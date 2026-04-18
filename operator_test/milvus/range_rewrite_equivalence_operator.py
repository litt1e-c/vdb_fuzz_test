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
DEFAULT_COLLECTION = "range_rewrite_equivalence_validation"


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
        {"id": 1, "x": 5, "y": 1.0, "meta": {"price": 5, "score": 1.5}, "vec": [0.0, 0.0]},
        {"id": 2, "x": 10, "y": 2.0, "meta": {"price": 10, "score": 2.0}, "vec": [0.1, 0.1]},
        {"id": 3, "x": 15, "y": None, "meta": {"price": 15, "score": 2.5}, "vec": [0.2, 0.2]},
        {"id": 4, "x": 20, "y": 4.0, "meta": {"price": 20, "score": 4.0}, "vec": [0.3, 0.3]},
        {"id": 5, "x": 25, "y": 5.0, "meta": {}, "vec": [0.4, 0.4]},
        {"id": 6, "x": None, "y": 6.0, "meta": None, "vec": [0.5, 0.5]},
        {"id": 7, "x": 30, "y": None, "meta": {"price": None, "score": 7.0}, "vec": [0.6, 0.6]},
        {"id": 8, "x": 35, "y": 8.0, "meta": {"price": 35, "score": None}, "vec": [0.7, 0.7]},
    ]


def build_equivalence_cases() -> list[tuple[str, str, str, list[int]]]:
    return [
        (
            "and_range_intersection",
            "x > 10 and x < 25",
            "(x > 5 and x < 30) and (x >= 15 and x <= 20)",
            [3, 4],
        ),
        (
            "adjacent_or_range_union",
            "x >= 10 and x <= 20",
            "(x >= 10 and x < 15) or (x >= 15 and x <= 20)",
            [2, 3, 4],
        ),
        (
            "in_as_or_equals",
            "x in [10, 15, 20]",
            "x == 10 or x == 15 or x == 20",
            [2, 3, 4],
        ),
        (
            "not_in_as_and_not_equals",
            "x not in [10, 15, 20]",
            "x != 10 and x != 15 and x != 20",
            [1, 5, 7, 8],
        ),
        (
            "float_range_intersection",
            "y is not null and y >= 2.0 and y <= 5.0",
            "(y is not null and y >= 1.0 and y <= 8.0) and (y >= 2.0 and y <= 5.0)",
            [2, 4, 5],
        ),
        (
            "json_path_range_intersection",
            'meta["price"] > 10 and meta["price"] < 25',
            '(meta["price"] > 5 and meta["price"] < 30) and (meta["price"] >= 15 and meta["price"] <= 20)',
            [3, 4],
        ),
        (
            "json_path_adjacent_or_range",
            'meta["price"] >= 10 and meta["price"] <= 20',
            '(meta["price"] >= 10 and meta["price"] < 15) or (meta["price"] >= 15 and meta["price"] <= 20)',
            [2, 3, 4],
        ),
    ]


def create_collection(name: str) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="x", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="y", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    return Collection(name, CollectionSchema(fields))


def run_phase(
    collection: Collection,
    phase: str,
    baseline: dict[str, tuple[list[int], list[int]]] | None = None,
) -> tuple[int, dict[str, tuple[list[int], list[int]]]]:
    mismatches = 0
    phase_results: dict[str, tuple[list[int], list[int]]] = {}
    for name, base_expr, equivalent_expr, expected in build_equivalence_cases():
        try:
            base_ids = query_ids(collection, base_expr)
            equiv_ids = query_ids(collection, equivalent_expr)
            phase_results[name] = (base_ids, equiv_ids)
            ok = base_ids == equiv_ids == expected
            if baseline is not None:
                ok = ok and phase_results[name] == baseline.get(name)
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} | phase={phase} | "
                f"base={base_expr} -> {base_ids} | equiv={equivalent_expr} -> {equiv_ids} | "
                f"expected={expected}"
            )
            if not ok:
                mismatches += 1
        except Exception as exc:
            mismatches += 1
            print(
                f"{name}: ERROR | phase={phase} | base={base_expr} | equiv={equivalent_expr} | "
                f"error={type(exc).__name__}: {exc}"
            )
    return mismatches, phase_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus range-rewrite equivalence validation")
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
        time.sleep(0.4)

        print("--- Range rewrite equivalence validation: baseline ---")
        phase_mismatches, baseline = run_phase(col, "baseline")
        mismatches += phase_mismatches

        index_plan = [
            ("x", {"index_type": "STL_SORT"}, "x_sort_idx"),
            ("y", {"index_type": "INVERTED"}, "y_inverted_idx"),
            (
                "meta",
                {"index_type": "INVERTED", "params": {"json_cast_type": "Double", "json_path": "meta['price']"}},
                "meta_price_double_idx",
            ),
        ]
        for field_name, params, index_name in index_plan:
            try:
                col.create_index(field_name, params, index_name=index_name, timeout=60)
            except Exception as exc:
                print(f"index_build_warn: field={field_name} index={index_name} error={type(exc).__name__}: {exc}")

        time.sleep(0.4)
        print("--- Range rewrite equivalence validation: after scalar/json-path indexes ---")
        phase_mismatches, _ = run_phase(col, "indexed_loaded", baseline)
        mismatches += phase_mismatches

        col.release(timeout=30)
        col.load(timeout=30)
        time.sleep(0.4)
        print("--- Range rewrite equivalence validation: after reload ---")
        phase_mismatches, _ = run_phase(col, "reloaded", baseline)
        mismatches += phase_mismatches
    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception:
            pass
        connections.disconnect("default")

    if mismatches:
        print(f"Summary: FAIL ({mismatches} range rewrite equivalence mismatches)")
        return 1
    print("Summary: PASS (range rewrites stable across baseline/index/reload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
