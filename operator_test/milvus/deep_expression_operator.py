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
DEFAULT_COLLECTION = "deep_expression_operator_validation"


def query_ids(collection: Collection, expr: str) -> list[int]:
    rows = collection.query(
        expr=expr,
        output_fields=["id"],
        consistency_level="Strong",
        timeout=30,
    )
    return sorted(int(row["id"]) for row in rows)


def build_rows() -> list[dict]:
    return [
        {"id": 1, "x": 5, "flag": True, "label": "low", "vec": [0.0, 0.0]},
        {"id": 2, "x": 10, "flag": True, "label": "mid", "vec": [0.1, 0.1]},
        {"id": 3, "x": 20, "flag": False, "label": "mid", "vec": [0.2, 0.2]},
        {"id": 4, "x": 30, "flag": True, "label": "mid", "vec": [0.3, 0.3]},
        {"id": 5, "x": 40, "flag": False, "label": "high", "vec": [0.4, 0.4]},
        {"id": 6, "x": None, "flag": None, "label": None, "vec": [0.5, 0.5]},
    ]


def repeat_and(expr: str, true_expr: str, depth: int) -> str:
    out = f"({expr})"
    for _ in range(depth):
        out = f"({out} and ({true_expr}))"
    return out


def repeat_or_false(expr: str, false_expr: str, depth: int) -> str:
    out = f"({expr})"
    for _ in range(depth):
        out = f"({out} or ({false_expr}))"
    return out


def repeat_double_not(expr: str, depth: int) -> str:
    out = f"({expr})"
    for _ in range(depth):
        out = f"not (not ({out}))"
    return out


def build_cases() -> list[tuple[str, str, list[int]]]:
    guarded_range = "x is not null and x >= 10 and x <= 30"
    true_expr = "id >= 0"
    false_expr = "id < 0"
    return [
        ("and_true_depth_16", repeat_and(guarded_range, true_expr, 16), [2, 3, 4]),
        ("or_false_depth_16", repeat_or_false(guarded_range, false_expr, 16), [2, 3, 4]),
        ("double_not_depth_8", repeat_double_not(guarded_range, 8), [2, 3, 4]),
        (
            "nested_in_or_chain",
            "((((x == 10 or x == 20) or x == 30) or (label == \"never\")) and x is not null)",
            [2, 3, 4],
        ),
        (
            "mixed_guarded_bool_string",
            "((x is not null and x >= 10 and x <= 30) and ((flag == true or flag == false) and label == \"mid\"))",
            [2, 3, 4],
        ),
    ]


def run_phase(
    collection: Collection,
    phase: str,
    baseline: dict[str, list[int]] | None = None,
) -> tuple[int, dict[str, list[int]]]:
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
                f"expected={expected_ids} | actual={actual_ids} | expr_len={len(expr)}"
            )
            if not ok:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"{name}: ERROR | phase={phase} | error={type(exc).__name__}: {exc} | expr_len={len(expr)}")
    return failures, phase_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus controlled deep-expression validation")
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
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=32, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(args.collection, CollectionSchema(fields))
        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=30)
        col.load(timeout=30)
        time.sleep(0.4)

        print("--- Deep expression validation: baseline ---")
        phase_failures, baseline = run_phase(col, "baseline")
        failures += phase_failures

        for field_name, params, index_name in [
            ("x", {"index_type": "STL_SORT"}, "x_sort_idx"),
            ("flag", {"index_type": "BITMAP"}, "flag_bitmap_idx"),
            ("label", {"index_type": "INVERTED"}, "label_inverted_idx"),
        ]:
            try:
                col.create_index(field_name, params, index_name=index_name, timeout=60)
            except Exception as exc:
                print(f"index_build_warn: field={field_name} index={index_name} error={type(exc).__name__}: {exc}")

        time.sleep(0.4)
        print("--- Deep expression validation: after scalar indexes ---")
        phase_failures, _ = run_phase(col, "indexed_loaded", baseline)
        failures += phase_failures

        col.release(timeout=30)
        col.load(timeout=30)
        time.sleep(0.4)
        print("--- Deep expression validation: after reload ---")
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
        print(f"Summary: FAIL ({failures} deep-expression mismatches/errors)")
        return 1
    print("Summary: PASS (controlled deep expressions stable across baseline/index/reload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
