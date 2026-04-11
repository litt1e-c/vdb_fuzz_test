"""
Minimal validation for the Qdrant `is_null` operator.

This script validates a conservative subset:
1. `is_null` matches only when the target field exists and is explicit `null`.
2. Missing keys, `[]`, `[None]`, empty strings, and empty objects do not satisfy `is_null`.
3. Top-level array-of-object `null` satisfies `is_null`, while `[]` and non-empty arrays do not.
4. Nested-path `is_null` follows the same tested explicit-null semantics.
5. `must_not is_null` acts as the complement of `is_null` in the tested subset.
6. REST, gRPC, and Python `:memory:` agree on the tested subset.
"""

from __future__ import annotations

import random
import time
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Filter,
    IsNullCondition,
    PayloadField,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
        PointStruct(id=1, vector=[1.0, 0.0], payload={}),
        PointStruct(id=2, vector=[1.0, 0.0], payload={"reports": None}),
        PointStruct(id=3, vector=[1.0, 0.0], payload={"reports": []}),
        PointStruct(id=4, vector=[1.0, 0.0], payload={"reports": [1]}),
        PointStruct(id=5, vector=[1.0, 0.0], payload={"reports": ""}),
        PointStruct(id=6, vector=[1.0, 0.0], payload={"reports": {}}),
        PointStruct(id=7, vector=[1.0, 0.0], payload={"reports": [None]}),
        PointStruct(id=8, vector=[1.0, 0.0], payload={"items": None}),
        PointStruct(id=9, vector=[1.0, 0.0], payload={"items": []}),
        PointStruct(
            id=10,
            vector=[1.0, 0.0],
            payload={"items": [{"score": 1, "label": "alpha", "active": True}]},
        ),
        PointStruct(id=11, vector=[1.0, 0.0], payload={"meta": {}}),
        PointStruct(id=12, vector=[1.0, 0.0], payload={"meta": {"reports": None}}),
        PointStruct(id=13, vector=[1.0, 0.0], payload={"meta": {"reports": []}}),
        PointStruct(id=14, vector=[1.0, 0.0], payload={"meta": {"reports": [1]}}),
        PointStruct(id=15, vector=[1.0, 0.0], payload={"meta": {"reports": ""}}),
        PointStruct(id=16, vector=[1.0, 0.0], payload={"meta": {"reports": {}}}),
        PointStruct(id=17, vector=[1.0, 0.0], payload={"meta": {"reports": [None]}}),
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


def run_collection_checks(client: QdrantClient, collection_name: str, transport: str) -> bool:
    all_passed = True
    tests = [
        (
            "reports_is_null",
            Filter(must=[IsNullCondition(is_null=PayloadField(key="reports"))]),
            [2],
            "Only explicit top-level null matches; missing, [], [None], '', and {} do not.",
        ),
        (
            "reports_not_null_via_must_not",
            Filter(must_not=[IsNullCondition(is_null=PayloadField(key="reports"))]),
            [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            "The complement of top-level `reports is_null` keeps all tested non-null and missing rows.",
        ),
        (
            "items_is_null_array_object",
            Filter(must=[IsNullCondition(is_null=PayloadField(key="items"))]),
            [8],
            "Only explicit null on the array-of-object field matches; [] and non-empty arrays do not.",
        ),
        (
            "meta_reports_is_null_nested_path",
            Filter(must=[IsNullCondition(is_null=PayloadField(key="meta.reports"))]),
            [12],
            "Only explicit nested null matches; missing nested paths, [], [None], '', and {} do not.",
        ),
        (
            "meta_reports_not_null_via_must_not",
            Filter(must_not=[IsNullCondition(is_null=PayloadField(key="meta.reports"))]),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
            "The complement of nested-path `is_null` keeps all tested non-null and missing nested rows.",
        ),
    ]

    print(f"\n--- is_null operator validation ({transport}) ---")
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
    return all_passed


def run_transport(prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(
        "is_null_operator_grpc" if prefer_grpc else "is_null_operator_rest"
    )
    client = build_client(prefer_grpc)

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)
        return run_collection_checks(client, collection_name, transport)
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")


def run_memory_probe() -> bool:
    collection_name = unique_collection_name("is_null_operator_memory")
    client = QdrantClient(":memory:")

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)
        return run_collection_checks(client, collection_name, "memory")
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning (memory): {exc}")


def main() -> int:
    try:
        server_info = fetch_server_info()
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant is_null operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)
    memory_ok = run_memory_probe()

    if rest_ok and grpc_ok and memory_ok:
        print("\nSummary: all is_null checks passed on REST, gRPC, and Python memory mode.")
        return 0

    print("\nSummary: at least one is_null check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
