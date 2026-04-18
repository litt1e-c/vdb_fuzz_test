#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


DEFAULT_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
DEFAULT_PORT = os.getenv("MILVUS_PORT", "19630")
DEFAULT_COLLECTION = "string_function_call_validation"


def query_ids(collection: Collection, expr: str) -> list[int]:
    rows = collection.query(
        expr=expr,
        output_fields=["id"],
        consistency_level="Strong",
        timeout=20,
    )
    return sorted(int(row["id"]) for row in rows)


def build_rows() -> list[dict[str, Any]]:
    return [
        {"id": 1, "s": "", "prefix": "", "vec": [0.0, 0.0]},
        {"id": 2, "s": "admin", "prefix": "ad", "vec": [0.1, 0.1]},
        {"id": 3, "s": "admiral", "prefix": "adm", "vec": [0.2, 0.2]},
        {"id": 4, "s": "user", "prefix": "ad", "vec": [0.3, 0.3]},
        {"id": 5, "s": None, "prefix": "ad", "vec": [0.4, 0.4]},
        {"id": 6, "s": "alpha", "prefix": None, "vec": [0.5, 0.5]},
        {"id": 7, "s": "北京大学", "prefix": "北京", "vec": [0.6, 0.6]},
        {"id": 8, "s": "emoji🙂", "prefix": "emoji", "vec": [0.7, 0.7]},
        {"id": 9, "s": "space value", "prefix": "space ", "vec": [0.8, 0.8]},
    ]


def build_tests() -> list[tuple[str, str, list[int]]]:
    return [
        ("empty_literal_field", "empty(s)", [1]),
        ("not_empty_literal_field", "not empty(s)", [2, 3, 4, 6, 7, 8, 9]),
        ("starts_with_literal", 'starts_with(s, "ad")', [2, 3]),
        ("starts_with_column_prefix", "starts_with(s, prefix)", [1, 2, 3, 7, 8, 9]),
        ("starts_with_empty_literal", 'starts_with(s, "")', [1, 2, 3, 4, 6, 7, 8, 9]),
        ("starts_with_unicode", 'starts_with(s, "北京")', [7]),
        ("starts_with_emoji_prefix", 'starts_with(s, "emoji")', [8]),
        ("starts_with_space_prefix", 'starts_with(s, "space ")', [9]),
        ("empty_prefix_field", "empty(prefix)", [1]),
        ("starts_with_prefix_empty_literal", 'starts_with(prefix, "")', [1, 2, 3, 4, 5, 7, 8, 9]),
        ("function_or_like_equivalence", 'empty(s) or starts_with(s, "ad")', [1, 2, 3]),
        ("function_and_like_equivalence", 'empty(s) and starts_with(s, "")', [1]),
        ("not_function_with_null_guard", 's is not null and not starts_with(s, "ad")', [1, 4, 6, 7, 8, 9]),
    ]


def create_collection(name: str) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="s", dtype=DataType.VARCHAR, max_length=128, nullable=True),
        FieldSchema(name="prefix", dtype=DataType.VARCHAR, max_length=128, nullable=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    return Collection(name, CollectionSchema(fields))


def run_phase(
    collection: Collection,
    tests: list[tuple[str, str, list[int]]],
    phase: str,
    baseline: dict[str, list[int]] | None = None,
) -> tuple[int, dict[str, list[int]]]:
    observed: dict[str, list[int]] = {}
    failures = 0
    for name, expr, expected in tests:
        actual = query_ids(collection, expr)
        observed[name] = actual
        reference = expected if baseline is None else baseline[name]
        if actual != reference:
            failures += 1
            print(f"mismatch phase={phase} case={name} expr={expr}")
            print(f"  expected={reference}")
            print(f"  actual={actual}")
    return failures, observed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus string filter function-call validation")
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

        collection = create_collection(args.collection)
        collection.insert(build_rows())
        collection.flush(timeout=20)
        collection.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=30)
        collection.load(timeout=30)
        time.sleep(0.4)

        tests = build_tests()
        phase_failures, baseline = run_phase(collection, tests, "baseline")
        failures += phase_failures

        for field in ("s", "prefix"):
            try:
                collection.create_index(field, {"index_type": "INVERTED"}, index_name=f"idx_{field}", timeout=30)
            except Exception as exc:
                failures += 1
                print(f"index_create_failed field={field}: {exc}")

        phase_failures, _ = run_phase(collection, tests, "post_inverted_index", baseline)
        failures += phase_failures

        collection.release(timeout=20)
        collection.load(timeout=30)
        time.sleep(0.4)
        phase_failures, _ = run_phase(collection, tests, "post_reload", baseline)
        failures += phase_failures
    except Exception as exc:
        print(f"operator_failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception as cleanup_exc:
            print(f"cleanup_warning: {cleanup_exc}")
        connections.disconnect("default")

    if failures:
        print(f"string_function_call: FAIL mismatches={failures}")
        return 1
    print("string_function_call: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
