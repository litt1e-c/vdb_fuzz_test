"""
Minimal validation for the Qdrant `range` operator.

This script validates a conservative, non-extreme subset:
1. Integer and float scalar fields support gt/gte/lt/lte.
2. Multiple range bounds are conjunctive.
3. Integer/float arrays match if any element satisfies the range.
4. Missing/null rows do not satisfy ordinary range predicates.
5. Tested type-mismatched string rows do not match numeric range predicates.

Known unstable numeric extremes are intentionally not tested here; historical
bug scripts under history_find_bug/qdrant cover those separately.
"""

from __future__ import annotations

import random
import time
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"range_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


def fetch_server_info() -> dict[str, object]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{HOST}:{PORT}/", timeout=5) as resp:
        import json

        return json.loads(resp.read().decode("utf-8"))


def build_client(prefer_grpc: bool) -> QdrantClient:
    return QdrantClient(
        host=HOST,
        port=PORT,
        grpc_port=GRPC_PORT,
        prefer_grpc=prefer_grpc,
        timeout=30,
    )


def build_points() -> list[PointStruct]:
    return [
        PointStruct(
            id=1,
            vector=[1.0, 0.0],
            payload={
                "int_val": 5,
                "float_val": 1.5,
                "arr_int": [1, 10],
                "arr_float": [0.5, 2.5],
                "str_val": "10",
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "int_val": 10,
                "float_val": 3.0,
                "arr_int": [20],
                "arr_float": [5.0],
                "str_val": "20",
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "int_val": -5,
                "float_val": -1.0,
                "arr_int": [],
                "arr_float": [],
                "str_val": "abc",
            },
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={
                "int_val": None,
                "float_val": None,
                "arr_int": None,
                "arr_float": None,
                "str_val": None,
            },
        ),
        PointStruct(id=5, vector=[0.2, 0.8], payload={}),
        PointStruct(
            id=6,
            vector=[0.8, 0.2],
            payload={
                "int_val": 0,
                "float_val": 0.0,
                "arr_int": [-5, 5],
                "arr_float": [-1.5, 1.0],
                "str_val": "7",
            },
        ),
    ]


def scroll_ids(client: QdrantClient, collection_name: str, scroll_filter: Filter) -> list[int]:
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=64,
        with_payload=False,
        with_vectors=False,
    )
    return sorted(int(point.id) for point in points)


def run_transport(prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(prefer_grpc)
    client = build_client(prefer_grpc)
    all_passed = True

    tests = [
        (
            "int_gt_zero",
            Filter(must=[FieldCondition(key="int_val", range=Range(gt=0))]),
            [1, 2],
            "Strict integer lower bound excludes zero and missing/null.",
        ),
        (
            "int_closed_exact",
            Filter(must=[FieldCondition(key="int_val", range=Range(gte=5, lte=5))]),
            [1],
            "Closed bounds are inclusive and conjunctive.",
        ),
        (
            "int_open_interval",
            Filter(must=[FieldCondition(key="int_val", range=Range(gt=0, lt=10))]),
            [1],
            "Open bounds are strict and conjunctive.",
        ),
        (
            "float_open_interval",
            Filter(must=[FieldCondition(key="float_val", range=Range(gt=0.0, lt=2.0))]),
            [1],
            "Float open interval matches only present values inside the interval.",
        ),
        (
            "float_lte_zero",
            Filter(must=[FieldCondition(key="float_val", range=Range(lte=0.0))]),
            [3, 6],
            "Float lte is inclusive and excludes missing/null.",
        ),
        (
            "array_int_any_element",
            Filter(must=[FieldCondition(key="arr_int", range=Range(gte=9, lte=11))]),
            [1],
            "Integer arrays match when any element satisfies the range.",
        ),
        (
            "array_float_any_element",
            Filter(must=[FieldCondition(key="arr_float", range=Range(gte=2.0, lte=3.0))]),
            [1],
            "Float arrays match when any element satisfies the range.",
        ),
        (
            "missing_null_excluded",
            Filter(must=[FieldCondition(key="int_val", range=Range(gte=-100, lte=100))]),
            [1, 2, 3, 6],
            "Missing/null rows do not satisfy ordinary range predicates.",
        ),
        (
            "type_mismatch_string_excluded",
            Filter(must=[FieldCondition(key="str_val", range=Range(gte=1, lte=20))]),
            [],
            "String values do not satisfy numeric range predicates.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        for field_name, schema in [
            ("int_val", PayloadSchemaType.INTEGER),
            ("float_val", PayloadSchemaType.FLOAT),
            ("arr_int", PayloadSchemaType.INTEGER),
            ("arr_float", PayloadSchemaType.FLOAT),
            ("str_val", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- range operator validation ({transport}) ---")
        print(f"collection={collection_name}")
        for name, scroll_filter, expected_ids, note in tests:
            actual_ids = scroll_ids(client, collection_name, scroll_filter)
            passed = actual_ids == expected_ids
            if not passed:
                all_passed = False
            status = "PASS" if passed else "FAIL"
            print(
                f"{name}: {status} | expected={expected_ids} | actual={actual_ids} | note={note}"
            )
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")

    return all_passed


def main() -> int:
    try:
        server_info = fetch_server_info()
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant range operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all range operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one range operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
