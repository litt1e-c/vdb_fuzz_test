"""
Minimal validation for the Qdrant `has_id` operator.

This script validates a conservative subset:
1. `has_id` matches points whose IDs are listed in the query.
2. Unknown IDs are ignored in the tested subset.
3. Duplicate IDs in the query do not duplicate results.
4. `must_not has_id` excludes the listed IDs and preserves the rest.
5. Both integer and UUID point IDs work in the tested subset.
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
    HasIdCondition,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334
UUID_IDS = [
    "550e8400-e29b-41d4-a716-446655440000",
    "550e8400-e29b-41d4-a716-446655440001",
    "550e8400-e29b-41d4-a716-446655440002",
]


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


def build_int_points() -> list[PointStruct]:
    return [
        PointStruct(id=1, vector=[1.0, 0.0], payload={"name": "one"}),
        PointStruct(id=2, vector=[0.0, 1.0], payload={"name": "two"}),
        PointStruct(id=3, vector=[1.0, 1.0], payload={"name": "three"}),
        PointStruct(id=4, vector=[0.5, 0.5], payload={"name": "four"}),
    ]


def build_uuid_points() -> list[PointStruct]:
    return [
        PointStruct(id=UUID_IDS[0], vector=[1.0, 0.0], payload={"name": "u0"}),
        PointStruct(id=UUID_IDS[1], vector=[0.0, 1.0], payload={"name": "u1"}),
        PointStruct(id=UUID_IDS[2], vector=[1.0, 1.0], payload={"name": "u2"}),
    ]


def scroll_ids(client: QdrantClient, collection_name: str, scroll_filter: Filter) -> list[object]:
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=64,
        with_payload=False,
        with_vectors=False,
    )
    return [point.id for point in points]


def run_int_checks(client: QdrantClient, collection_name: str, transport: str) -> bool:
    all_passed = True
    tests = [
        (
            "int_has_id_existing_ids",
            Filter(must=[HasIdCondition(has_id=[1, 3])]),
            [1, 3],
            "Matching integer IDs are returned.",
        ),
        (
            "int_has_id_existing_and_unknown",
            Filter(must=[HasIdCondition(has_id=[1, 99])]),
            [1],
            "Unknown integer IDs are ignored in the tested subset.",
        ),
        (
            "int_has_id_duplicate_ids",
            Filter(must=[HasIdCondition(has_id=[1, 1, 3])]),
            [1, 3],
            "Duplicate integer IDs in the filter do not duplicate results.",
        ),
        (
            "int_must_not_has_id",
            Filter(must_not=[HasIdCondition(has_id=[2, 4])]),
            [1, 3],
            "The tested complement excludes listed integer IDs and preserves the rest.",
        ),
        (
            "int_must_not_unknown_only",
            Filter(must_not=[HasIdCondition(has_id=[99])]),
            [1, 2, 3, 4],
            "Unknown integer IDs alone do not exclude any stored points.",
        ),
    ]

    print(f"\n--- has_id integer validation ({transport}) ---")
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


def run_uuid_checks(client: QdrantClient, collection_name: str, transport: str) -> bool:
    all_passed = True
    tests = [
        (
            "uuid_has_id_existing_ids",
            Filter(must=[HasIdCondition(has_id=[UUID_IDS[0], UUID_IDS[2]])]),
            [UUID_IDS[0], UUID_IDS[2]],
            "Matching UUID IDs are returned.",
        ),
        (
            "uuid_has_id_existing_and_unknown",
            Filter(
                must=[
                    HasIdCondition(
                        has_id=[UUID_IDS[0], "550e8400-e29b-41d4-a716-446655440099"]
                    )
                ]
            ),
            [UUID_IDS[0]],
            "Unknown UUID IDs are ignored in the tested subset.",
        ),
        (
            "uuid_must_not_has_id",
            Filter(must_not=[HasIdCondition(has_id=[UUID_IDS[1]])]),
            [UUID_IDS[0], UUID_IDS[2]],
            "The tested complement excludes the listed UUID ID and preserves the rest.",
        ),
    ]

    print(f"\n--- has_id UUID validation ({transport}) ---")
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
    int_collection = unique_collection_name(
        "has_id_operator_int_grpc" if prefer_grpc else "has_id_operator_int_rest"
    )
    uuid_collection = unique_collection_name(
        "has_id_operator_uuid_grpc" if prefer_grpc else "has_id_operator_uuid_rest"
    )
    client = build_client(prefer_grpc)

    try:
        client.create_collection(
            collection_name=int_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=int_collection, points=build_int_points(), wait=True)

        client.create_collection(
            collection_name=uuid_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=uuid_collection, points=build_uuid_points(), wait=True)

        int_ok = run_int_checks(client, int_collection, transport)
        uuid_ok = run_uuid_checks(client, uuid_collection, transport)
        return int_ok and uuid_ok
    finally:
        for collection_name in [int_collection, uuid_collection]:
            try:
                client.delete_collection(collection_name)
            except Exception as exc:
                print(f"cleanup_warning ({transport}, {collection_name}): {exc}")


def run_memory_probe() -> bool:
    int_collection = unique_collection_name("has_id_operator_int_memory")
    uuid_collection = unique_collection_name("has_id_operator_uuid_memory")
    client = QdrantClient(":memory:")

    try:
        client.create_collection(
            collection_name=int_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=int_collection, points=build_int_points(), wait=True)

        client.create_collection(
            collection_name=uuid_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=uuid_collection, points=build_uuid_points(), wait=True)

        int_ok = run_int_checks(client, int_collection, "memory")
        uuid_ok = run_uuid_checks(client, uuid_collection, "memory")
        return int_ok and uuid_ok
    finally:
        for collection_name in [int_collection, uuid_collection]:
            try:
                client.delete_collection(collection_name)
            except Exception as exc:
                print(f"cleanup_warning (memory, {collection_name}): {exc}")


def main() -> int:
    try:
        server_info = fetch_server_info()
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant has_id operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)
    memory_ok = run_memory_probe()

    if rest_ok and grpc_ok and memory_ok:
        print("\nSummary: all has_id checks passed on REST, gRPC, and Python memory mode.")
        return 0

    print("\nSummary: at least one has_id check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
