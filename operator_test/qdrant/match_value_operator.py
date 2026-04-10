"""
Minimal validation for the Qdrant `match.value` condition.

This script validates a conservative subset:
1. Exact match on supported scalar payloads: keyword, integer, bool.
2. Array semantics: match succeeds if at least one array element matches.
3. Missing/null rows do not satisfy ordinary match.value predicates.
4. Type-mismatched filters are treated as not satisfied.

It also probes a float field to document current local behavior, without
upgrading undocumented support into oracle scope.
"""

from __future__ import annotations

import random
import time
from importlib.metadata import version as pkg_version

import urllib.request

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"match_value_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
                "color": "red",
                "count": 10,
                "flag": True,
                "tags": ["red", "blue"],
                "sizes": [8, 10],
                "responses": [False, True],
                "price": 1.5,
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "color": "blue",
                "count": 20,
                "flag": False,
                "tags": ["green"],
                "sizes": [20],
                "responses": [False],
                "price": 2.5,
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "color": None,
                "count": None,
                "flag": None,
                "tags": None,
                "sizes": None,
                "responses": None,
                "price": None,
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
            "match_string_scalar",
            Filter(must=[FieldCondition(key="color", match=MatchValue(value="red"))]),
            [1],
            "Supported keyword scalar equality.",
        ),
        (
            "match_int_scalar",
            Filter(must=[FieldCondition(key="count", match=MatchValue(value=10))]),
            [1],
            "Supported integer scalar equality.",
        ),
        (
            "match_bool_scalar",
            Filter(must=[FieldCondition(key="flag", match=MatchValue(value=True))]),
            [1],
            "Supported bool scalar equality.",
        ),
        (
            "match_string_array_any_element",
            Filter(must=[FieldCondition(key="tags", match=MatchValue(value="red"))]),
            [1],
            "Array match should succeed if at least one element matches.",
        ),
        (
            "match_int_array_any_element",
            Filter(must=[FieldCondition(key="sizes", match=MatchValue(value=10))]),
            [1],
            "Integer arrays should follow the same any-element rule.",
        ),
        (
            "match_bool_array_any_element",
            Filter(must=[FieldCondition(key="responses", match=MatchValue(value=True))]),
            [1],
            "Bool arrays should follow the same any-element rule.",
        ),
        (
            "match_value_excludes_missing_and_null",
            Filter(must=[FieldCondition(key="color", match=MatchValue(value="yellow"))]),
            [],
            "Missing/null rows should not satisfy ordinary match.value predicates.",
        ),
        (
            "match_value_type_mismatch_string_on_int",
            Filter(must=[FieldCondition(key="count", match=MatchValue(value="10"))]),
            [],
            "Type-mismatched match.value should be treated as not satisfied.",
        ),
        (
            "match_value_type_mismatch_int_on_string",
            Filter(must=[FieldCondition(key="color", match=MatchValue(value=10))]),
            [],
            "Type-mismatched match.value should be treated as not satisfied.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- match_value operator validation ({transport}) ---")
        print(f"collection={collection_name}")
        try:
            MatchValue(value=1.5)
            float_probe_status = "UNEXPECTED_ACCEPT"
        except Exception as exc:
            float_probe_status = f"CLIENT_REJECTED ({type(exc).__name__})"
        print(
            "match_value_float_probe: "
            f"{float_probe_status} | note=Float values are not accepted by the tested client model."
        )
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

    print("Qdrant match_value operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all match_value operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one match_value operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
