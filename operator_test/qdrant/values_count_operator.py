"""
Minimal validation for the Qdrant `values_count` operator.

This script validates a conservative subset for server-backed filtering:
1. `values_count` compares the number of stored values with `gt/gte/lt/lte`.
2. Explicit `null` and `[]` behave like count 0 in the tested server subset.
3. Missing keys do not satisfy the tested `values_count` predicates.
4. Non-array present values behave like count 1 in the tested server subset.
5. Arrays count by length, including arrays of objects and arrays containing `null`.
6. REST and gRPC agree on the tested server subset.
7. Python `:memory:` differs from the server on the tested scalar-dict case.
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
    PointStruct,
    ValuesCount,
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
        PointStruct(id=2, vector=[1.0, 0.0], payload={"vc": None}),
        PointStruct(id=3, vector=[1.0, 0.0], payload={"vc": []}),
        PointStruct(id=4, vector=[1.0, 0.0], payload={"vc": [1]}),
        PointStruct(id=5, vector=[1.0, 0.0], payload={"vc": [1, 2]}),
        PointStruct(id=6, vector=[1.0, 0.0], payload={"vc": "scalar"}),
        PointStruct(id=7, vector=[1.0, 0.0], payload={"vc": {"a": 1, "b": 2}}),
        PointStruct(id=8, vector=[1.0, 0.0], payload={"vc": [None]}),
        PointStruct(
            id=9,
            vector=[1.0, 0.0],
            payload={"vc": [{"a": 1}, {"b": 2}]},
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
    collection_name = unique_collection_name(
        "values_count_operator_grpc" if prefer_grpc else "values_count_operator_rest"
    )
    client = build_client(prefer_grpc)
    all_passed = True

    tests = [
        (
            "values_count_lte_0",
            Filter(must=[FieldCondition(key="vc", values_count=ValuesCount(lte=0))]),
            [2, 3],
            "Explicit null and [] behave like count 0, while the missing key is excluded.",
        ),
        (
            "values_count_gte_1",
            Filter(must=[FieldCondition(key="vc", values_count=ValuesCount(gte=1))]),
            [4, 5, 6, 7, 8, 9],
            "Present non-array values behave like count 1, [None] behaves like count 1, and arrays count by length.",
        ),
        (
            "values_count_gt_1",
            Filter(must=[FieldCondition(key="vc", values_count=ValuesCount(gt=1))]),
            [5, 9],
            "Only the tested length-two arrays match; the tested scalar dict is not counted by object key count on the server.",
        ),
        (
            "values_count_lte_1",
            Filter(must=[FieldCondition(key="vc", values_count=ValuesCount(lte=1))]),
            [2, 3, 4, 6, 7, 8],
            "Count-0 and count-1 rows match, including the tested scalar dict and [None].",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- values_count operator validation ({transport}) ---")
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


def run_memory_probe() -> bool:
    collection_name = unique_collection_name("values_count_operator_memory")
    client = QdrantClient(":memory:")
    all_passed = True

    tests = [
        (
            "memory_values_count_gt_1_backend_mismatch",
            Filter(must=[FieldCondition(key="vc", values_count=ValuesCount(gt=1))]),
            [5, 7, 9],
            "Python :memory: counts the tested scalar dict as 2 here, unlike the server.",
        ),
        (
            "memory_values_count_lte_1_backend_mismatch",
            Filter(must=[FieldCondition(key="vc", values_count=ValuesCount(lte=1))]),
            [2, 3, 4, 6, 8],
            "Python :memory: excludes the tested scalar dict from count <= 1, unlike the server.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print("\n--- values_count in-memory probe (Python local) ---")
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
            print(f"cleanup_warning (memory): {exc}")

    return all_passed


def main() -> int:
    try:
        server_info = fetch_server_info()
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant values_count operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)
    memory_ok = run_memory_probe()

    if rest_ok and grpc_ok and memory_ok:
        print("\nSummary: values_count server checks passed on REST and gRPC, and the Python local mismatch probe was reproduced.")
        return 0

    print("\nSummary: at least one values_count check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
