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
DEFAULT_COLLECTION = "three_valued_logic_validation"


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
        {
            "id": 1,
            "c_true": True,
            "c_false": False,
            "c_null": None,
            "c_num": 10,
            "meta": {"color": "blue"},
            "vec": [0.0, 0.0],
        },
        {
            "id": 2,
            "c_true": True,
            "c_false": False,
            "c_null": None,
            "c_num": None,
            "meta": {"shape": "circle"},
            "vec": [0.1, 0.1],
        },
        {
            "id": 3,
            "c_true": True,
            "c_false": False,
            "c_null": False,
            "c_num": 5,
            "meta": None,
            "vec": [0.2, 0.2],
        },
        {
            "id": 4,
            "c_true": True,
            "c_false": False,
            "c_null": True,
            "c_num": 0,
            "meta": {"color": "red"},
            "vec": [0.3, 0.3],
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus 3VL semantic probe")
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
            FieldSchema(name="c_true", dtype=DataType.BOOL),
            FieldSchema(name="c_false", dtype=DataType.BOOL),
            FieldSchema(name="c_null", dtype=DataType.BOOL, nullable=True),
            FieldSchema(name="c_num", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(args.collection, CollectionSchema(fields))
        col.insert(build_rows())
        col.flush(timeout=20)
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=30)
        col.load(timeout=30)
        time.sleep(0.5)

        normative_cases = [
            ("is_null_baseline", "c_null is null", [1, 2]),
            ("not_is_null", "not (c_null is null)", [3, 4]),
            ("not_num_is_null", "not (c_num is null)", [1, 3, 4]),
            ("json_field_is_null", "meta is null", [3]),
            ("json_field_not_null", "not (meta is null)", [1, 2, 4]),
        ]

        known_issue_probes = [
            (
                "probe_not_null_and_false",
                "not ((c_null == true) and (c_false == true))",
                [1, 2, 3, 4],
                "SQL 3VL expects UNKNOWN AND FALSE => FALSE, outer NOT => TRUE",
            ),
            (
                "probe_true_or_null",
                "(c_true == true) or (c_null == true)",
                [1, 2, 3, 4],
                "SQL 3VL expects TRUE OR UNKNOWN => TRUE",
            ),
            (
                "probe_json_not_equal_with_null",
                'meta["color"] != "blue"',
                [4],
                "SQL 3VL expects missing-key/null-field => UNKNOWN (filtered)",
            ),
        ]

        print("--- 3VL normative checks ---")
        for name, expr, expected in normative_cases:
            actual = query_ids(col, expr)
            ok = actual == expected
            print(f"{name}: {'PASS' if ok else 'FAIL'} | expr={expr} | expected={expected} | actual={actual}")
            if not ok:
                hard_failures += 1

        print("--- 3VL known-issue probes ---")
        for name, expr, sql_expected, rationale in known_issue_probes:
            actual = query_ids(col, expr)
            if actual == sql_expected:
                print(
                    f"{name}: FIXED_OR_NOT_REPRO | expr={expr} | "
                    f"sql_expected={sql_expected} | actual={actual}"
                )
            else:
                known_bug_hits += 1
                print(
                    f"{name}: KNOWN_BUG_REPRODUCED | expr={expr} | "
                    f"sql_expected={sql_expected} | actual={actual} | note={rationale}"
                )

    finally:
        try:
            if utility.has_collection(args.collection):
                utility.drop_collection(args.collection, timeout=20)
        except Exception:
            pass
        connections.disconnect("default")

    if hard_failures:
        print(f"Summary: FAIL ({hard_failures} normative checks failed)")
        return 1
    print(f"Summary: PASS (normative checks passed, known_bug_reproduced={known_bug_hits})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
