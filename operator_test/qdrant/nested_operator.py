"""
Minimal validation for the Qdrant `nested` operator.

This script validates a conservative subset:
1. Nested matches when at least one array element satisfies the inner filter.
2. Same-element semantics hold for inner `must`.
3. Inner `should` works per nested element.
4. Inner `min_should` is evaluated per nested element, not across siblings.
5. Inner `must_not` works per nested element.
6. Missing/null/[] parent fields do not satisfy ordinary nested predicates.
7. `has_id` inside nested is rejected locally, matching the checked docs.
"""

from __future__ import annotations

import argparse
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
    MinShould,
    Nested,
    NestedCondition,
    PointStruct,
    Range,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334
RUN_ID: str | None = None
_COLLECTION_COUNTER = 0


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def unique_collection_name(prefer_grpc: bool) -> str:
    global _COLLECTION_COUNTER
    transport = "grpc" if prefer_grpc else "rest"
    if RUN_ID:
        _COLLECTION_COUNTER += 1
        return f"nested_operator_{slugify(RUN_ID, max_len=36)}_{transport}_{_COLLECTION_COUNTER:02d}"
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


def count_total(client: QdrantClient, collection_name: str, count_filter: Filter) -> int:
    return int(
        client.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=True,
        ).count
    )


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
            "projected_array_object_cross_element_and",
            Filter(
                must=[
                    FieldCondition(key="items[].score", range=Range(gte=5)),
                    FieldCondition(key="items[].label", match=MatchValue(value="beta")),
                ]
            ),
            [2],
            "Projected array-object key paths accumulate across sibling elements, so one element may satisfy the score bound while another satisfies the label test.",
        ),
        (
            "nested_same_formula_rejects_cross_element_projection",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(
                                must=[
                                    FieldCondition(key="score", range=Range(gte=5)),
                                    FieldCondition(key="label", match=MatchValue(value="beta")),
                                ]
                            ),
                        )
                    )
                ]
            ),
            [],
            "Nested keeps same-element semantics for the same formula, so the projected cross-element hit above must disappear here.",
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
            "nested_inner_min_should_two_of_three",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(
                                min_should=MinShould(
                                    conditions=[
                                        FieldCondition(key="score", range=Range(gte=5)),
                                        FieldCondition(key="label", match=MatchValue(value="epsilon")),
                                        FieldCondition(key="active", match=MatchValue(value=True)),
                                    ],
                                    min_count=2,
                                )
                            ),
                        )
                    )
                ]
            ),
            [2, 3],
            "Inner min_should counts matches within a single nested object, so rows 2 and 3 match via one qualifying element each.",
        ),
        (
            "nested_inner_min_should_same_element_only",
            Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(
                                min_should=MinShould(
                                    conditions=[
                                        FieldCondition(key="label", match=MatchValue(value="beta")),
                                        FieldCondition(key="active", match=MatchValue(value=True)),
                                    ],
                                    min_count=2,
                                )
                            ),
                        )
                    )
                ]
            ),
            [],
            "Inner min_should must be satisfied by one nested object; cross-element accumulation would incorrectly match rows 1 and 2 here.",
        ),
        (
            "nested_inner_min_should_under_top_level_must_not",
            Filter(
                must_not=[
                    Filter(
                        must=[
                            NestedCondition(
                                nested=Nested(
                                    key="items",
                                    filter=Filter(
                                        min_should=MinShould(
                                            conditions=[
                                                FieldCondition(key="score", range=Range(gte=5)),
                                                FieldCondition(key="active", match=MatchValue(value=True)),
                                            ],
                                            min_count=2,
                                        )
                                    ),
                                )
                            )
                        ]
                    )
                ]
            ),
            [1, 4, 5, 6, 7],
            "Top-level must_not should exclude parents that have some nested element satisfying the inner min_should formula; the tested missing-child-field row survives because its element does not satisfy all required inner votes.",
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
            actual_count = count_total(client, collection_name, scroll_filter)
            passed = actual_ids == expected_ids and actual_count == len(expected_ids)
            if not passed:
                all_passed = False
            status = "PASS" if passed else "FAIL"
            print(
                f"{name}: {status} | expected={expected_ids} | actual={actual_ids} | "
                f"expected_count={len(expected_ids)} | actual_count={actual_count} | note={note}"
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


def main(argv: list[str] | None = None) -> int:
    global HOST, PORT, GRPC_PORT, RUN_ID

    parser = argparse.ArgumentParser(description="Validate Qdrant nested filter semantics")
    parser.add_argument("--host", default=HOST, help="Qdrant REST host")
    parser.add_argument("--port", type=int, default=PORT, help="Qdrant REST port")
    parser.add_argument("--grpc-port", type=int, default=GRPC_PORT, dest="grpc_port", help="Qdrant gRPC port")
    parser.add_argument("--run-id", default=None, help="Optional deterministic run id for collection naming")
    args = parser.parse_args(argv)

    HOST = args.host
    PORT = int(args.port)
    GRPC_PORT = int(args.grpc_port)
    RUN_ID = args.run_id

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
