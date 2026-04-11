"""
Minimal validation for the Qdrant `is_empty` operator.

This script validates a conservative subset:
1. `is_empty` matches when the field is missing, explicit `null`, or `[]`.
2. Empty strings, empty objects, and `[None]` are not empty in the tested subset.
3. Array-of-object `[]` satisfies `is_empty`, while a non-empty array of objects does not.
4. Nested-path `is_empty` follows the same tested subset semantics.
5. `must_not is_empty` acts as the complement of `is_empty` in the tested subset.
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
    IsEmptyCondition,
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
        PointStruct(id=8, vector=[1.0, 0.0], payload={"items": []}),
        PointStruct(id=9, vector=[1.0, 0.0], payload={"items": [{"score": 1, "label": "alpha", "active": True}]}),
        PointStruct(id=10, vector=[1.0, 0.0], payload={"meta": {}}),
        PointStruct(id=11, vector=[1.0, 0.0], payload={"meta": {"reports": None}}),
        PointStruct(id=12, vector=[1.0, 0.0], payload={"meta": {"reports": []}}),
        PointStruct(id=13, vector=[1.0, 0.0], payload={"meta": {"reports": [1]}}),
        PointStruct(id=14, vector=[1.0, 0.0], payload={"meta": {"reports": ""}}),
        PointStruct(id=15, vector=[1.0, 0.0], payload={"meta": {"reports": {}}}),
        PointStruct(id=16, vector=[1.0, 0.0], payload={"meta": {"reports": [None]}}),
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
            "reports_is_empty",
            Filter(must=[IsEmptyCondition(is_empty=PayloadField(key="reports"))]),
            [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "Missing top-level `reports`, explicit null, and [] are empty; '', {}, and [None] are not.",
        ),
        (
            "reports_not_empty_via_must_not",
            Filter(must_not=[IsEmptyCondition(is_empty=PayloadField(key="reports"))]),
            [4, 5, 6, 7],
            "The complement of top-level `reports is_empty` keeps only the tested non-empty present values.",
        ),
        (
            "items_is_empty_array_object",
            Filter(must=[IsEmptyCondition(is_empty=PayloadField(key="items"))]),
            [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16],
            "Missing `items` and the tested [] array-of-object row are empty; the non-empty object array is not.",
        ),
        (
            "meta_reports_is_empty_nested_path",
            Filter(must=[IsEmptyCondition(is_empty=PayloadField(key="meta.reports"))]),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "Nested-path missing, null, and [] are empty in the tested subset; [1], '', {}, and [None] are not.",
        ),
        (
            "meta_reports_not_empty_via_must_not",
            Filter(must_not=[IsEmptyCondition(is_empty=PayloadField(key="meta.reports"))]),
            [13, 14, 15, 16],
            "The complement of nested-path `is_empty` keeps only the tested non-empty present nested values.",
        ),
    ]

    print(f"\n--- is_empty operator validation ({transport}) ---")
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
        "is_empty_operator_grpc" if prefer_grpc else "is_empty_operator_rest"
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
    collection_name = unique_collection_name("is_empty_operator_memory")
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

    print("Qdrant is_empty operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)
    memory_ok = run_memory_probe()

    if rest_ok and grpc_ok and memory_ok:
        print("\nSummary: all is_empty checks passed on REST, gRPC, and Python memory mode.")
        return 0

    print("\nSummary: at least one is_empty check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
