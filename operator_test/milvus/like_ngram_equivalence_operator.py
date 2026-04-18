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
DEFAULT_COLLECTION = "like_ngram_equivalence_validation"
NGRAM_PARAMS = {"min_gram": 2, "max_gram": 4}
JSON_NGRAM_PARAMS = {
    "min_gram": 2,
    "max_gram": 4,
    "json_path": "meta['body']",
    "json_cast_type": "varchar",
}


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
        {"id": 1, "content": "admin", "meta": {"body": "admin"}, "vec": [0.0, 0.0]},
        {"id": 2, "content": "admiral", "meta": {"body": "admiral"}, "vec": [1.0, 1.0]},
        {"id": 3, "content": "stadium", "meta": {"body": "stadium"}, "vec": [2.0, 2.0]},
        {"id": 4, "content": "user", "meta": {"body": "user"}, "vec": [3.0, 3.0]},
        {"id": 5, "content": "guest", "meta": {"body": "guest"}, "vec": [4.0, 4.0]},
        {"id": 6, "content": "alpha", "meta": {"body": "alpha"}, "vec": [5.0, 5.0]},
        {"id": 7, "content": "atlas", "meta": {"body": "atlas"}, "vec": [6.0, 6.0]},
        {"id": 8, "content": "zebra", "meta": {"title": "missing-body"}, "vec": [7.0, 7.0]},
        {"id": 9, "content": "adxin", "meta": {"body": "adxin"}, "vec": [8.0, 8.0]},
        {"id": 10, "content": "北京大学", "meta": {"body": "北京大学"}, "vec": [9.0, 9.0]},
        {"id": 11, "content": "void", "meta": {"body": None}, "vec": [10.0, 10.0]},
        {"id": 12, "content": "cadmium", "meta": {"body": "cadmium"}, "vec": [11.0, 11.0]},
    ]


def build_tests() -> list[tuple[str, str, list[int]]]:
    return [
        ("content_prefix_lt_min_gram", 'content like "a%"', [1, 2, 6, 7, 9]),
        ("content_prefix_eq_min_gram", 'content like "ad%"', [1, 2, 9]),
        ("content_prefix_mid_window", 'content like "adm%"', [1, 2]),
        ("content_prefix_gt_window", 'content like "admi%"', [1, 2]),
        ("content_suffix_match", 'content like "%ral"', [2]),
        ("content_infix_match", 'content like "%dmi%"', [1, 2, 12]),
        ("content_underscore_match", 'content like "ad_in"', [1, 9]),
        ("content_unicode_prefix", 'content like "北京%"', [10]),
        ("content_unicode_suffix", 'content like "%大学"', [10]),
        ('json_prefix_lt_min_gram', 'meta["body"] like "a%"', [1, 2, 6, 7, 9]),
        ('json_prefix_eq_min_gram', 'meta["body"] like "ad%"', [1, 2, 9]),
        ('json_suffix_match', 'meta["body"] like "%ral"', [2]),
        ('json_infix_match', 'meta["body"] like "%dmi%"', [1, 2, 12]),
        ('json_underscore_match', 'meta["body"] like "ad_in"', [1, 9]),
        ('json_unicode_prefix', 'meta["body"] like "北京%"', [10]),
        ('json_unicode_suffix', 'meta["body"] like "%大学"', [10]),
    ]


def ensure_collection(name: str) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    return Collection(name=name, schema=CollectionSchema(fields))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus LIKE/NGRAM equivalence validation")
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

        col = ensure_collection(args.collection)
        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=30)
        col.load(timeout=30)
        time.sleep(0.4)

        tests = build_tests()
        baseline: dict[str, list[int]] = {}
        print("--- LIKE/NGRAM baseline (no NGRAM index) ---")
        for name, expr, expected in tests:
            actual = query_ids(col, expr)
            baseline[name] = actual
            ok = actual == expected
            print(f"{name}: {'PASS' if ok else 'FAIL'} | phase=baseline | expr={expr} | expected={expected} | actual={actual}")
            if not ok:
                failures += 1

        col.release(timeout=30)
        col.create_index("content", {"index_type": "NGRAM", "params": dict(NGRAM_PARAMS)}, timeout=30, index_name="content_ngram")
        col.create_index("meta", {"index_type": "NGRAM", "params": dict(JSON_NGRAM_PARAMS)}, timeout=30, index_name="meta_body_ngram")
        col.load(timeout=30)
        time.sleep(0.5)

        print("--- LIKE/NGRAM after NGRAM index build ---")
        for name, expr, expected in tests:
            actual = query_ids(col, expr)
            before = baseline[name]
            ok = actual == expected == before
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} | phase=after_index | expr={expr} | "
                f"expected={expected} | baseline={before} | actual={actual}"
            )
            if not ok:
                failures += 1

        col.release(timeout=30)
        col.load(timeout=30)
        time.sleep(0.5)

        print("--- LIKE/NGRAM after release/load reload ---")
        for name, expr, expected in tests:
            actual = query_ids(col, expr)
            before = baseline[name]
            ok = actual == expected == before
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} | phase=after_reload | expr={expr} | "
                f"expected={expected} | baseline={before} | actual={actual}"
            )
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
        print(f"Summary: FAIL ({failures} LIKE/NGRAM checks failed)")
        return 1
    print("Summary: PASS (LIKE semantics stayed stable across no-index, NGRAM index, and reload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
