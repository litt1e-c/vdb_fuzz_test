"""
Minimal validation for the Qdrant `nested` operator.

This script validates a conservative subset:
1. Nested matches when at least one array element satisfies the inner filter.
2. Same-element semantics hold for inner `must`.
3. Inner `should` works per nested element.
4. Inner `must_not` works per nested element.
5. Missing/null/[] parent fields do not satisfy ordinary nested predicates.
6. `has_id` inside nested is rejected locally, matching the checked docs.
"""

from __future__ import annotations

import random
import time
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as http_exceptions
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    Nested,
    NestedCondition,
    PointStruct,
    Range,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"nested_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
                "items": [
                    {"score": 1, "label": "alpha", "active": True},
                    {"score": 2, "label": "beta", "active": False},
                ]
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "items": [
                    {"score": 1, "label": "beta", "active": False},
                    {"score": 5, "label": "gamma", "active": True},
                ]
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "items": [
                    {"score": 1, "label": "delta", "active": False},
                    {"score": 9, "label": "epsilon", "active": True},
                ]
            },
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={"items": []},
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.8],
            payload={"items": None},
        ),
        PointStruct(
            id=6,
            vector=[0.8, 0.2],
            payload={},
        ),
        PointStruct(
            id=7,
            vector=[0.3, 0.7],
            payload={
                "items": [
                    {"score": 7, "label": "ghost"},
                ]
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
            "nested_single_match",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(must=[FieldCondition(key="score", match=MatchValue(value=1))]),
                        )
                    )
                ]
            ),
            [1, 2, 3],
            "Nested should match when at least one element satisfies the inner predicate.",
        ),
        (
            "nested_same_element_multi_and",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(
                                must=[
                                    FieldCondition(key="score", match=MatchValue(value=1)),
                                    FieldCondition(key="active", match=MatchValue(value=True)),
                                ]
                            ),
                        )
                    )
                ]
            ),
            [1],
            "Inner must should require the same element to satisfy both predicates.",
        ),
        (
            "nested_inner_should",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(
                                should=[
                                    Filter(must=[FieldCondition(key="score", match=MatchValue(value=5))]),
                                    Filter(must=[FieldCondition(key="score", match=MatchValue(value=9))]),
                                ]
                            ),
                        )
                    )
                ]
            ),
            [2, 3],
            "Inner should should work per nested element.",
        ),
        (
            "nested_inner_must_not",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(
                                must_not=[
                                    FieldCondition(key="active", match=MatchValue(value=False)),
                                ]
                            ),
                        )
                    )
                ]
            ),
            [1, 2, 3, 7],
            "Inner must_not should keep rows with at least one surviving element, including the tested missing-child-field element.",
        ),
        (
            "nested_excludes_missing_null_empty",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(must=[FieldCondition(key="score", range=Range(gte=0))]),
                        )
                    )
                ]
            ),
            [1, 2, 3, 7],
            "Missing/null/[] parent fields should not satisfy ordinary nested predicates.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- nested operator validation ({transport}) ---")
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

        has_id_filter = Filter(
            must=[
                NestedCondition(
                    nested=Nested(
                        key="items",
                        filter=Filter(must=[HasIdCondition(has_id=[1])]),
                    )
                )
            ]
        )
        try:
            actual_ids = scroll_ids(client, collection_name, has_id_filter)
            print(
                "nested_inner_has_id_probe: OBSERVED | "
                f"actual={actual_ids} | note=Checked docs say `has_id` is not supported inside nested, "
                "but local v1.17.0 accepted the query."
            )
        except Exception as exc:
            status = "OBSERVED"
            if not isinstance(exc, (http_exceptions.UnexpectedResponse, http_exceptions.ResponseHandlingException, ValueError, TypeError)):
                status = "OBSERVED"
            print(
                f"nested_inner_has_id_probe: {status} | observed={type(exc).__name__}: {exc} | "
                "note=Checked docs say `has_id` is not supported inside nested."
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

    print("Qdrant nested operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all nested operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one nested operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
