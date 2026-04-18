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
DEFAULT_COLLECTION = "bool_domain_rewrite_operator_validation"


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
        {"id": 1, "flag": True, "vec": [0.0, 0.0]},
        {"id": 2, "flag": False, "vec": [1.0, 1.0]},
        {"id": 3, "flag": None, "vec": [2.0, 2.0]},
        {"id": 4, "vec": [3.0, 3.0]},
    ]


def build_tests() -> list[tuple[str, str, list[int]]]:
    return [
        ("domain_in_matches_not_null", "flag in [true, false]", [1, 2]),
        ("domain_not_in_matches_null", "flag not in [true, false]", [3, 4]),
        ("not_domain_in_matches_null", "not (flag in [true, false])", [3, 4]),
        ("not_domain_not_in_matches_not_null", "not (flag not in [true, false])", [1, 2]),
        ("empty_in_is_false", "flag in []", []),
        ("empty_not_in_matches_not_null", "flag not in []", [1, 2]),
        ("baseline_is_null", "flag is null", [3, 4]),
        ("baseline_is_not_null", "flag is not null", [1, 2]),
    ]


def run_phase(collection: Collection, phase: str, baseline: dict[str, list[int]] | None = None) -> tuple[int, dict[str, list[int]]]:
    failures = 0
    phase_results: dict[str, list[int]] = {}
    for name, expr, expected_ids in build_tests():
        try:
            actual_ids = query_ids(collection, expr)
            phase_results[name] = actual_ids
            ok = actual_ids == expected_ids
            if baseline is not None:
                ok = ok and baseline.get(name) == actual_ids
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
    parser = argparse.ArgumentParser(description="Milvus nullable-bool in/not-in domain rewrite validation")
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
            FieldSchema(name="flag", dtype=DataType.BOOL, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(args.collection, CollectionSchema(fields))
        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=20)
        col.load(timeout=20)
        time.sleep(0.3)

        print("--- Nullable bool domain rewrite validation: baseline ---")
        phase_failures, baseline = run_phase(col, "baseline")
        failures += phase_failures

        col.create_index("flag", {"index_type": "BITMAP"}, index_name="flag_bitmap_idx", timeout=60)
        time.sleep(0.3)
        print("--- Nullable bool domain rewrite validation: after scalar index ---")
        phase_failures, _ = run_phase(col, "indexed_loaded", baseline)
        failures += phase_failures

        col.release(timeout=20)
        col.load(timeout=20)
        time.sleep(0.3)
        print("--- Nullable bool domain rewrite validation: after reload ---")
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
        print(f"Summary: FAIL ({failures} nullable-bool domain rewrite mismatches)")
        return 1
    print("Summary: PASS (nullable-bool domain rewrites stable across baseline/index/reload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
