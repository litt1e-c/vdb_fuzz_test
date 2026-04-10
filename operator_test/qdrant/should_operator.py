"""
Minimal validation for the Qdrant `should` filter clause.

This script validates a conservative subset used by the fuzzer:
1. Ordinary `should` behaves like disjunction across listed conditions.
2. A `Filter` can be nested inside `should` and still participates in disjunction.
3. Missing/null rows do not satisfy ordinary predicates inside `should`.
4. `A OR NOT A` includes missing/null rows under Qdrant's observed set semantics.
5. The behavior of an empty `should` list is observed explicitly because the docs
   do not specify it clearly.
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
    Range,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"should_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
            payload={"city": "Berlin", "color": "red", "score": -3},
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={"city": "Berlin", "color": "blue"},
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.2],
            payload={"score": None},
        ),
        PointStruct(
            id=6,
            vector=[0.3, 0.3],
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

    atom_gt_zero = FieldCondition(key="score", range=Range(gt=0))
    atom_lt_zero = FieldCondition(key="score", range=Range(lt=0))

    tests = [
        (
            "should_simple_or",
            Filter(
                should=[
                    FieldCondition(key="city", match=MatchValue(value="London")),
                    FieldCondition(key="color", match=MatchValue(value="red")),
                ]
            ),
            [1, 2, 3],
            "At least one listed condition should hold.",
        ),
        (
            "should_recursive_subfilters",
            Filter(
                should=[
                    Filter(
                        must=[
                            FieldCondition(key="city", match=MatchValue(value="London")),
                            FieldCondition(key="color", match=MatchValue(value="green")),
                        ]
                    ),
                    Filter(
                        must=[
                            FieldCondition(key="city", match=MatchValue(value="Berlin")),
                            FieldCondition(key="color", match=MatchValue(value="red")),
                        ]
                    ),
                ]
            ),
            [2, 3],
            "Nested Filter objects inside should should still combine by disjunction.",
        ),
        (
            "should_ordinary_predicates_exclude_missing_and_null",
            Filter(
                should=[
                    atom_gt_zero,
                    atom_lt_zero,
                ]
            ),
            [1, 2, 3],
            "Missing/null rows should not satisfy ordinary predicates when all disjuncts are ordinary.",
        ),
        (
            "should_or_not_includes_missing_and_null",
            Filter(
                should=[
                    Filter(must=[atom_gt_zero]),
                    Filter(must_not=[atom_gt_zero]),
                ]
            ),
            [1, 2, 3, 4, 5, 6],
            "A OR NOT A should include missing/null rows under the observed set semantics.",
        ),
        (
            "should_empty_list_behavior",
            Filter(should=[]),
            [1, 2, 3, 4, 5, 6],
            "Observed scroll behavior for empty should list on the tested version.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- should operator validation ({transport}) ---")
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

    print("Qdrant should operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all should operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one should operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
