"""
Minimal validation for the Qdrant `UUID Match` condition.

This script validates a conservative subset:
1. Exact match on scalar UUID payloads.
2. Array semantics: match succeeds if at least one UUID element matches.
3. Missing/null rows do not satisfy ordinary UUID match predicates.
4. A UUID payload index can be created and queried on the tested server.

It also probes a type-mismatched integer query against a UUID field to record
current local behavior without upgrading it into official semantics.
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
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334

UUID_A = "550e8400-e29b-41d4-a716-446655440000"
UUID_B = "550e8400-e29b-41d4-a716-446655440001"
UUID_C = "550e8400-e29b-41d4-a716-446655440002"
UUID_ABSENT = "550e8400-e29b-41d4-a716-446655449999"


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"uuid_match_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
                "uuid_value": UUID_A,
                "uuid_array": [UUID_A, UUID_B],
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "uuid_value": UUID_B,
                "uuid_array": [UUID_C],
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "uuid_value": None,
                "uuid_array": None,
            },
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={},
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
            "uuid_match_scalar_exact",
            Filter(must=[FieldCondition(key="uuid_value", match=MatchValue(value=UUID_A))]),
            [1],
            "Scalar UUID equality should match the exact stored UUID.",
        ),
        (
            "uuid_match_array_any_element",
            Filter(must=[FieldCondition(key="uuid_array", match=MatchValue(value=UUID_B))]),
            [1],
            "UUID arrays should satisfy match when any element equals the target.",
        ),
        (
            "uuid_match_absent_uuid",
            Filter(must=[FieldCondition(key="uuid_value", match=MatchValue(value=UUID_ABSENT))]),
            [],
            "A valid but absent UUID should not match any row.",
        ),
        (
            "uuid_match_excludes_missing_and_null",
            Filter(must=[FieldCondition(key="uuid_value", match=MatchValue(value=UUID_C))]),
            [],
            "Missing/null rows should not satisfy ordinary UUID match predicates.",
        ),
        (
            "uuid_match_type_mismatch_int_query",
            Filter(must=[FieldCondition(key="uuid_value", match=MatchValue(value=123))]),
            [],
            "Type-mismatched integer match on a UUID field should be treated as not satisfied.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="uuid_value",
            field_schema=PayloadSchemaType.UUID,
            wait=True,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="uuid_array",
            field_schema=PayloadSchemaType.UUID,
            wait=True,
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- uuid_match operator validation ({transport}) ---")
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

    print("Qdrant uuid_match operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all uuid_match operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one uuid_match operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
