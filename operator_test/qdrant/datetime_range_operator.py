"""
Minimal validation for the Qdrant `datetime_range` operator.

This script validates a conservative documented subset:
1. `DatetimeRange` supports `gt`/`gte`/`lt`/`lte` on datetime payloads.
2. Datetime comparisons normalize values to UTC before comparison.
3. Officially documented non-canonical datetime formats are accepted in the
   tested subset (`+01:00` offset, space instead of `T`, `+0000`, and date-only).
4. Datetime arrays match if any element satisfies the range condition.
5. Missing/null and tested integer type-mismatched rows do not satisfy
   ordinary datetime range predicates.
"""

from __future__ import annotations

import random
import time
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"datetime_range_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": ["2024-01-01T00:00:00Z"],
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "created_at": "2024-01-01T01:00:00+01:00",
                "updated_at": ["2024-01-02T00:00:00Z"],
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": ["2023-12-31T23:00:00Z", "2024-01-03T00:00:00Z"],
            },
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={
                "created_at": "2024-01-01 00:00:00+0000",
                "updated_at": ["2024-01-01"],
            },
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.8],
            payload={
                "created_at": None,
                "updated_at": None,
            },
        ),
        PointStruct(id=6, vector=[0.8, 0.2], payload={}),
        PointStruct(
            id=7,
            vector=[0.3, 0.7],
            payload={
                "created_at": 1704067200,
                "updated_at": [1704067200],
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
            "created_at_exact_midnight_utc_norm",
            Filter(
                must=[
                    FieldCondition(
                        key="created_at",
                        range=DatetimeRange(
                            gte="2024-01-01T00:00:00Z",
                            lte="2024-01-01T00:00:00Z",
                        ),
                    )
                ]
            ),
            [1, 2, 4],
            "Closed bounds match equal instants after UTC normalization and accept tested documented storage formats.",
        ),
        (
            "created_at_gt_midnight",
            Filter(
                must=[
                    FieldCondition(
                        key="created_at",
                        range=DatetimeRange(gt="2024-01-01T00:00:00Z"),
                    )
                ]
            ),
            [3],
            "Strict lower bound excludes equal instants, missing/null, and tested integer type mismatch.",
        ),
        (
            "created_at_lte_midnight",
            Filter(
                must=[
                    FieldCondition(
                        key="created_at",
                        range=DatetimeRange(lte="2024-01-01T00:00:00Z"),
                    )
                ]
            ),
            [1, 2, 4],
            "Inclusive upper bound matches equal instants after UTC normalization.",
        ),
        (
            "updated_at_array_any_element_late_match",
            Filter(
                must=[
                    FieldCondition(
                        key="updated_at",
                        range=DatetimeRange(
                            gte="2024-01-03T00:00:00Z",
                            lte="2024-01-03T00:00:00Z",
                        ),
                    )
                ]
            ),
            [3],
            "Datetime arrays match when at least one element satisfies the range.",
        ),
        (
            "updated_at_date_only_query",
            Filter(
                must=[
                    FieldCondition(
                        key="updated_at",
                        range=DatetimeRange(
                            gte="2024-01-01",
                            lte="2024-01-01",
                        ),
                    )
                ]
            ),
            [1, 4],
            "The tested date-only query literal behaves as midnight UTC and matches equal datetime-array elements.",
        ),
        (
            "created_at_missing_null_type_mismatch_excluded",
            Filter(
                must=[
                    FieldCondition(
                        key="created_at",
                        range=DatetimeRange(
                            gte="2023-12-31T00:00:00Z",
                            lte="2025-01-01T00:00:00Z",
                        ),
                    )
                ]
            ),
            [1, 2, 3, 4],
            "Missing/null and the tested integer value do not satisfy the ordinary datetime range predicate.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        for field_name in ["created_at", "updated_at"]:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.DATETIME,
                wait=True,
            )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- datetime_range operator validation ({transport}) ---")
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

    print("Qdrant datetime_range operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all datetime_range operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one datetime_range operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
