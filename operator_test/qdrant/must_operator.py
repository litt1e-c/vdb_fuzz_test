"""
Minimal validation for the Qdrant `must` filter clause.

This script validates four conservative questions:
1. Ordinary `must` behaves like conjunction across listed conditions.
2. A `Filter` can be nested inside `must` and still participates in conjunction.
3. Missing/null rows do not satisfy ordinary predicates inside `must`.
4. Plain `must` over `array[].field` is not the same as `nested`; same-element
   semantics require `nested`.

The script uses unique task-local collection names and cleans up only the
collections that it created.
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
    return f"must_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
            payload={"city": "London", "color": "red", "score": 10},
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={"city": "London", "color": "green", "score": 5},
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={"city": "Berlin", "color": "red", "score": 3},
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={"city": "Berlin", "color": "blue"},
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.2],
            payload={
                "diet": [
                    {"food": "leaves", "likes": False},
                    {"food": "meat", "likes": True},
                ]
            },
        ),
        PointStruct(
            id=6,
            vector=[0.3, 0.3],
            payload={
                "diet": [
                    {"food": "leaves", "likes": True},
                    {"food": "meat", "likes": False},
                ]
            },
        ),
        PointStruct(
            id=7,
            vector=[0.4, 0.4],
            payload={"score": None},
        ),
        PointStruct(
            id=8,
            vector=[0.6, 0.6],
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
            "must_simple_and",
            Filter(
                must=[
                    FieldCondition(key="city", match=MatchValue(value="London")),
                    FieldCondition(key="color", match=MatchValue(value="red")),
                ]
            ),
            [1],
            "Both listed conditions must hold for the point.",
        ),
        (
            "must_recursive_subfilters",
            Filter(
                must=[
                    Filter(
                        should=[
                            FieldCondition(key="city", match=MatchValue(value="London")),
                            FieldCondition(key="city", match=MatchValue(value="Berlin")),
                        ]
                    ),
                    Filter(
                        must_not=[
                            FieldCondition(key="color", match=MatchValue(value="blue")),
                        ]
                    ),
                ]
            ),
            [1, 2, 3],
            "Nested Filter objects inside must should still combine by conjunction.",
        ),
        (
            "must_excludes_missing_and_null_for_ordinary_predicates",
            Filter(
                must=[
                    FieldCondition(key="score", range=Range(gt=0)),
                ]
            ),
            [1, 2, 3],
            "Missing/null rows should not satisfy ordinary range predicates inside must.",
        ),
        (
            "must_array_path_cross_element_match",
            Filter(
                must=[
                    FieldCondition(key="diet[].food", match=MatchValue(value="meat")),
                    FieldCondition(key="diet[].likes", match=MatchValue(value=True)),
                ]
            ),
            [5, 6],
            "Plain must over array paths can be satisfied across different elements.",
        ),
        (
            "must_nested_same_element_match",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="diet",
                            filter=Filter(
                                must=[
                                    FieldCondition(key="food", match=MatchValue(value="meat")),
                                    FieldCondition(key="likes", match=MatchValue(value=True)),
                                ]
                            ),
                        )
                    )
                ]
            ),
            [5],
            "Same-element semantics require nested, not plain must over array paths.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- must operator validation ({transport}) ---")
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

    print("Qdrant must operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all must operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one must operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
