"""
Minimal validation for the Qdrant `text_any` operator.

This script validates a conservative subset in the current fuzzer environment:
1. STRING fields use a KEYWORD payload index, not a full-text index.
2. MatchTextAny should match when any query term appears as a substring.
3. Missing/null rows do not satisfy ordinary MatchTextAny predicates.
4. Tested type-mismatched non-string rows do not match.
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
    MatchTextAny,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"text_any_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
            payload={"title": "good hardware"},
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={"title": "cheap hardware"},
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={"title": "good cheap hardware"},
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={"title": "expensive hardware"},
        ),
        PointStruct(
            id=5,
            vector=[0.3, 0.7],
            payload={"title": "goodness hardware"},
        ),
        PointStruct(
            id=6,
            vector=[0.7, 0.3],
            payload={"title": None},
        ),
        PointStruct(
            id=7,
            vector=[0.2, 0.8],
            payload={},
        ),
        PointStruct(
            id=8,
            vector=[0.8, 0.2],
            payload={"title": 12345},
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
            "text_any_multi_term_any_token",
            Filter(must=[FieldCondition(key="title", match=MatchTextAny(text_any="good cheap"))]),
            [1, 2, 3, 5],
            "Rows containing either query term as a substring should match.",
        ),
        (
            "text_any_single_term",
            Filter(must=[FieldCondition(key="title", match=MatchTextAny(text_any="cheap"))]),
            [2, 3],
            "A single query term should behave like a text-term match.",
        ),
        (
            "text_any_excludes_subtoken_missing_null_type_mismatch",
            Filter(must=[FieldCondition(key="title", match=MatchTextAny(text_any="good"))]),
            [1, 3, 5],
            "The tested no-full-text-index path uses substring matching, while missing/null and non-string rows do not match.",
        ),
        (
            "text_any_negative_nonmatch",
            Filter(must=[FieldCondition(key="title", match=MatchTextAny(text_any="absent token"))]),
            [],
            "Absent query terms should not match.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        # Mirror the current fuzzer environment: STRING fields get a KEYWORD index, not a full-text index.
        client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- text_any operator validation ({transport}) ---")
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

    print("Qdrant text_any operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all text_any operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one text_any operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
