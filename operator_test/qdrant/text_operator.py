"""
Minimal validation for the Qdrant `text` operator.

This script validates a conservative subset:
1. On a string field without a full-text index, MatchText behaves as exact substring match.
2. The current fuzzer environment still creates a KEYWORD index for STRING fields, and this
   does not change the observed no-full-text-index substring behavior.
3. Missing/null rows do not satisfy ordinary MatchText predicates.
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
    MatchText,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"text_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
            payload={"title": "the quick brown fox"},
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={"title": "slow green turtle"},
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={"title": "prefixbrown suffix"},
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={"title": None},
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.8],
            payload={},
        ),
        PointStruct(
            id=6,
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
            "text_substring_match_single",
            Filter(must=[FieldCondition(key="title", match=MatchText(text="brown"))]),
            [1, 3],
            "Without a full-text index, MatchText should behave like exact substring match on present strings.",
        ),
        (
            "text_substring_match_phrase_like",
            Filter(must=[FieldCondition(key="title", match=MatchText(text="quick brown"))]),
            [1],
            "A longer contiguous substring should still match exactly.",
        ),
        (
            "text_excludes_missing_null_type_mismatch",
            Filter(must=[FieldCondition(key="title", match=MatchText(text="turtle"))]),
            [2],
            "Missing/null and the tested non-string row should not satisfy ordinary MatchText.",
        ),
        (
            "text_negative_nonmatch",
            Filter(must=[FieldCondition(key="title", match=MatchText(text="absent-token"))]),
            [],
            "Absent substrings should not match.",
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

        print(f"\n--- text operator validation ({transport}) ---")
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

    print("Qdrant text operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all text operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one text operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
