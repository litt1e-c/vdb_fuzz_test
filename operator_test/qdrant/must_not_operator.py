"""
Minimal validation for the Qdrant `must_not` filter clause.

This script validates a conservative subset used by the fuzzer:
1. `must_not` with multiple children behaves like (NOT A) AND (NOT B).
2. A nested `Filter` inside `must_not` participates as recursive negation.
3. Missing/null rows are kept when ordinary predicates inside `must_not` do not match.
4. Double negation collapses back to the original match set.
5. Outer `must_not` over a nested condition keeps missing/null/[] nested fields.
6. Inner nested `must_not` works per array element.
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
    return f"must_not_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
                "city": "London",
                "color": "green",
                "score": 10,
                "items": [{"a": 1}, {"a": 2}],
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "city": "London",
                "color": "red",
                "score": 5,
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "city": "Berlin",
                "color": "red",
                "score": -1,
                "items": None,
            },
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={
                "city": "Moscow",
                "color": "green",
                "score": None,
                "items": [],
            },
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.2],
            payload={
                "city": "Moscow",
                "color": "blue",
            },
        ),
        PointStruct(
            id=6,
            vector=[0.3, 0.3],
            payload={
                "city": "London",
                "color": "blue",
                "score": 0,
                "items": [{"a": 0}],
            },
        ),
        PointStruct(
            id=7,
            vector=[0.4, 0.4],
            payload={
                "city": "Tokyo",
                "color": "yellow",
                "score": 7,
                "items": [{"a": 1}],
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

    atom_gt_zero = FieldCondition(key="score", range=Range(gt=0))
    nested_a_eq_1 = NestedCondition(
        nested=Nested(
            key="items",
            filter=Filter(must=[FieldCondition(key="a", match=MatchValue(value=1))]),
        )
    )

    tests = [
        (
            "must_not_multiple_children",
            Filter(
                must_not=[
                    FieldCondition(key="city", match=MatchValue(value="London")),
                    FieldCondition(key="color", match=MatchValue(value="red")),
                ]
            ),
            [4, 5, 7],
            "must_not should keep only rows satisfying neither child predicate.",
        ),
        (
            "must_not_recursive_inner_must",
            Filter(
                must_not=[
                    Filter(
                        must=[
                            FieldCondition(key="city", match=MatchValue(value="London")),
                            FieldCondition(key="color", match=MatchValue(value="red")),
                        ]
                    )
                ]
            ),
            [1, 3, 4, 5, 6, 7],
            "Recursive inner must should be negated as a whole.",
        ),
        (
            "must_not_ordinary_predicate_keeps_missing_and_null",
            Filter(must_not=[atom_gt_zero]),
            [3, 4, 5, 6],
            "Rows where the ordinary predicate does not match, including missing/null, should be kept.",
        ),
        (
            "must_not_double_negation",
            Filter(must_not=[Filter(must_not=[atom_gt_zero])]),
            [1, 2, 7],
            "Double negation should collapse back to the original match set.",
        ),
        (
            "must_not_outer_nested_keeps_missing_null_empty",
            Filter(must_not=[nested_a_eq_1]),
            [2, 3, 4, 5, 6],
            "Outer must_not over nested should keep missing/null/[] nested fields.",
        ),
        (
            "must_nested_inner_must_not",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(must_not=[FieldCondition(key="a", match=MatchValue(value=1))]),
                        )
                    )
                ]
            ),
            [1, 6],
            "Inner must_not inside nested should work per array element.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- must_not operator validation ({transport}) ---")
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

    print("Qdrant must_not operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all must_not operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one must_not operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
