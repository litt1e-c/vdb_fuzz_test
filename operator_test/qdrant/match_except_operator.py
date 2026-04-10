"""
Minimal validation for the Qdrant `Match Except` condition.

This script validates a conservative subset:
1. Scalar integer/string MatchExcept excludes listed values and keeps other present values.
2. Array semantics: MatchExcept succeeds if at least one stored element is not in the excluded list.
3. Missing/null rows do not satisfy ordinary MatchExcept predicates.
4. Type-mismatched MatchExcept predicates are treated as not satisfied.

It also probes the local behavior of an empty MatchExcept list without upgrading that
undocumented case into oracle scope.
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
    MatchExcept,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"match_except_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
                "tags": ["red", "blue"],
                "sizes": [10, 20],
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "color": "blue",
                "count": 20,
                "tags": ["green"],
                "sizes": [20],
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "color": "yellow",
                "count": 30,
                "tags": ["red", "green"],
                "sizes": [20, 30],
            },
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={
                "color": None,
                "count": None,
                "tags": None,
                "sizes": None,
            },
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.8],
            payload={},
        ),
        PointStruct(
            id=6,
            vector=[0.8, 0.2],
            payload={
                "color": "red",
                "count": 40,
                "tags": [],
                "sizes": [],
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
            "match_except_string_scalar_multi",
            Filter(must=[FieldCondition(key="color", match=MatchExcept(**{"except": ["red", "blue"]}))]),
            [3],
            "Scalar string MatchExcept should keep present values not in the excluded list.",
        ),
        (
            "match_except_int_scalar_multi",
            Filter(must=[FieldCondition(key="count", match=MatchExcept(**{"except": [10, 20]}))]),
            [3, 6],
            "Scalar integer MatchExcept should keep present values not in the excluded list.",
        ),
        (
            "match_except_string_array_partial_excluded",
            Filter(must=[FieldCondition(key="tags", match=MatchExcept(**{"except": ["red"]}))]),
            [1, 2, 3],
            "Array MatchExcept should match when at least one stored element is not excluded.",
        ),
        (
            "match_except_string_array_all_excluded",
            Filter(must=[FieldCondition(key="tags", match=MatchExcept(**{"except": ["red", "green", "blue"]}))]),
            [],
            "Array MatchExcept should fail when every stored element is excluded or the array is empty.",
        ),
        (
            "match_except_int_array_partial_excluded",
            Filter(must=[FieldCondition(key="sizes", match=MatchExcept(**{"except": [20]}))]),
            [1, 3],
            "Integer arrays should match when at least one stored element is not excluded.",
        ),
        (
            "match_except_excludes_missing_and_null",
            Filter(must=[FieldCondition(key="color", match=MatchExcept(**{"except": ["red"]}))]),
            [2, 3],
            "Missing/null rows should not satisfy ordinary MatchExcept predicates.",
        ),
        (
            "match_except_type_mismatch_string_on_int",
            Filter(must=[FieldCondition(key="count", match=MatchExcept(**{"except": ["10", "20"]}))]),
            [1, 2, 3, 6],
            "Observed locally: type-mismatched excluded values do not exclude present integer rows.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="color",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="count",
            field_schema=PayloadSchemaType.INTEGER,
            wait=True,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="tags",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="sizes",
            field_schema=PayloadSchemaType.INTEGER,
            wait=True,
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- match_except operator validation ({transport}) ---")
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

        try:
            empty_except_ids = scroll_ids(
                client,
                collection_name,
                Filter(must=[FieldCondition(key="color", match=MatchExcept(**{"except": []}))]),
            )
            print(
                "match_except_empty_list_probe: "
                f"OBSERVED actual={empty_except_ids} | note=Undocumented case; not used for oracle claims."
            )
        except Exception as exc:
            print(
                "match_except_empty_list_probe: "
                f"ERROR {type(exc).__name__}: {exc} | note=Undocumented case; not used for oracle claims."
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

    print("Qdrant match_except operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all match_except operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one match_except operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
