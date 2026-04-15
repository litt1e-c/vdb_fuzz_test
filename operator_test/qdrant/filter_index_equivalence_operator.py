"""
Validate scalar filter equivalence before and after explicit payload indexing.

This operator is intentionally deterministic and focuses on a high-value attack
surface for real bugs:
1. The same scalar filter should return the same answers before indexing.
2. Creating explicit payload indexes must not change filter semantics.
3. Deleting and recreating those indexes must not change semantics either.
4. REST and gRPC should both agree with the deterministic oracle.
5. Multiple read paths (`scroll`, `count`, and exact `query_points`) should
   agree on the same fixture.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from importlib.metadata import version as pkg_version
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    BoolIndexParams,
    BoolIndexType,
    DatetimeIndexParams,
    DatetimeIndexType,
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    FloatIndexParams,
    FloatIndexType,
    IntegerIndexParams,
    IntegerIndexType,
    KeywordIndexParams,
    KeywordIndexType,
    MatchAny,
    MatchExcept,
    MatchValue,
    PointStruct,
    Range,
    SearchParams,
    UuidIndexParams,
    UuidIndexType,
    VectorParams,
)


RUN_ID = "filter-index-equivalence"
QUERY_VECTOR = [1.0, 0.0]


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"filter_index_equivalence_{slugify(RUN_ID, max_len=36)}_{transport}"


def fetch_server_info(host: str, port: int) -> dict[str, object]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{host}:{port}/", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def build_client(args: argparse.Namespace, prefer_grpc: bool) -> QdrantClient:
    return QdrantClient(
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port,
        prefer_grpc=prefer_grpc,
        timeout=args.timeout,
    )


def build_points() -> list[PointStruct]:
    return [
        PointStruct(
            id=1,
            vector=[1.0, 0.0],
            payload={
                "tenant_kw": "alpha",
                "num_lookup": 10,
                "num_range": 5,
                "score_float": 1.0,
                "flag": True,
                "event_time": "2024-01-01T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000001",
                "arr_num": [1, 2],
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "tenant_kw": "beta",
                "num_lookup": 20,
                "num_range": 25,
                "score_float": 1.75,
                "flag": False,
                "event_time": "2024-01-02T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000002",
                "arr_num": [7, 9],
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "tenant_kw": "alpha",
                "num_lookup": 30,
                "num_range": 15,
                "score_float": 0.25,
                "flag": True,
                "event_time": "2024-01-03T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000003",
                "arr_num": [],
            },
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={
                "tenant_kw": "gamma",
                "num_lookup": 10,
                "num_range": 30,
                "score_float": 2.5,
                "flag": True,
                "event_time": "2024-01-04T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000004",
                "arr_num": [5, 8],
            },
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.8],
            payload={
                "tenant_kw": "beta",
                "num_lookup": 40,
                "num_range": 12,
                "score_float": 1.2,
                "flag": False,
                "event_time": "2024-01-05T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000005",
                "arr_num": None,
            },
        ),
        PointStruct(
            id=6,
            vector=[0.8, 0.2],
            payload={
                "tenant_kw": "delta",
                "num_lookup": 50,
                "num_range": 18,
                "score_float": 0.9,
                "flag": False,
                "event_time": "2024-01-06T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000006",
                "arr_num": [10],
            },
        ),
        PointStruct(
            id=7,
            vector=[0.6, 0.4],
            payload={
                "tenant_kw": "alpha",
                "num_lookup": 60,
                "num_range": 25,
                "score_float": None,
                "flag": True,
                "event_time": "2024-01-07T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000007",
                "arr_num": [8, 8],
            },
        ),
        PointStruct(
            id=8,
            vector=[0.4, 0.6],
            payload={
                "tenant_kw": "gamma",
                "num_lookup": 70,
                "num_range": 40,
                "score_float": 1.75,
                "flag": True,
                "event_time": None,
                "uuid_tag": "00000000-0000-0000-0000-000000000008",
                "arr_num": [0, 3],
            },
        ),
    ]


def index_specs() -> list[tuple[str, object, str]]:
    return [
        (
            "tenant_kw",
            KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True, on_disk=True),
            "keyword_tenant_on_disk",
        ),
        (
            "num_lookup",
            IntegerIndexParams(type=IntegerIndexType.INTEGER, lookup=True, range=False, on_disk=True),
            "integer_lookup_only_on_disk",
        ),
        (
            "num_range",
            IntegerIndexParams(
                type=IntegerIndexType.INTEGER,
                lookup=False,
                range=True,
                is_principal=True,
                on_disk=True,
            ),
            "integer_range_only_principal_on_disk",
        ),
        (
            "score_float",
            FloatIndexParams(type=FloatIndexType.FLOAT, is_principal=True, on_disk=True),
            "float_principal_on_disk",
        ),
        (
            "flag",
            BoolIndexParams(type=BoolIndexType.BOOL, on_disk=True),
            "bool_on_disk",
        ),
        (
            "event_time",
            DatetimeIndexParams(type=DatetimeIndexType.DATETIME, is_principal=True, on_disk=True),
            "datetime_principal_on_disk",
        ),
        (
            "uuid_tag",
            UuidIndexParams(type=UuidIndexType.UUID, is_tenant=True, on_disk=True),
            "uuid_tenant_on_disk",
        ),
        (
            "arr_num",
            IntegerIndexParams(type=IntegerIndexType.INTEGER, lookup=True, range=True, on_disk=True),
            "integer_array_lookup_range_on_disk",
        ),
    ]


def query_cases() -> list[tuple[str, Filter, list[int], str]]:
    return [
        (
            "keyword_exact_alpha",
            Filter(must=[FieldCondition(key="tenant_kw", match=MatchValue(value="alpha"))]),
            [1, 3, 7],
            "Keyword exact-match should stay stable before and after indexing.",
        ),
        (
            "keyword_any_alpha_gamma",
            Filter(must=[FieldCondition(key="tenant_kw", match=MatchAny(any=["alpha", "gamma"]))]),
            [1, 3, 4, 7, 8],
            "Keyword MatchAny should keep the same union under index optimization.",
        ),
        (
            "keyword_except_alpha",
            Filter(must=[FieldCondition(key="tenant_kw", match=MatchExcept(**{"except": ["alpha"]}))]),
            [2, 4, 5, 6, 8],
            "Keyword MatchExcept should preserve the same non-null complement set.",
        ),
        (
            "integer_lookup_eq_10",
            Filter(must=[FieldCondition(key="num_lookup", match=MatchValue(value=10))]),
            [1, 4],
            "Lookup-only integer indexing should preserve equality semantics.",
        ),
        (
            "integer_range_two_sided",
            Filter(must=[FieldCondition(key="num_range", range=Range(gte=15, lte=30))]),
            [2, 3, 4, 6, 7],
            "Range-only integer indexing should preserve two-sided range semantics.",
        ),
        (
            "float_range_half_open",
            Filter(must=[FieldCondition(key="score_float", range=Range(gte=1.0, lt=2.0))]),
            [1, 2, 5, 8],
            "Float payload indexing should not change half-open numeric interval behavior.",
        ),
        (
            "bool_match_true",
            Filter(must=[FieldCondition(key="flag", match=MatchValue(value=True))]),
            [1, 3, 4, 7, 8],
            "Bool payload indexing should preserve true/false matching.",
        ),
        (
            "datetime_range_gte",
            Filter(must=[FieldCondition(key="event_time", range=DatetimeRange(gte="2024-01-04T00:00:00Z"))]),
            [4, 5, 6, 7],
            "Datetime payload indexing should preserve chronological filtering and exclude null.",
        ),
        (
            "uuid_exact_match",
            Filter(
                must=[
                    FieldCondition(
                        key="uuid_tag",
                        match=MatchValue(value="00000000-0000-0000-0000-000000000002"),
                    )
                ]
            ),
            [2],
            "UUID payload indexing should preserve canonical exact-match filtering.",
        ),
        (
            "integer_array_any_element_range",
            Filter(must=[FieldCondition(key="arr_num", range=Range(gte=8))]),
            [2, 4, 6, 7],
            "Integer-array payload indexing should preserve any-element range matching.",
        ),
        (
            "composite_keyword_and_range",
            Filter(
                must=[
                    FieldCondition(key="tenant_kw", match=MatchValue(value="alpha")),
                    FieldCondition(key="num_range", range=Range(gte=15)),
                ]
            ),
            [3, 7],
            "Conjunctive filters should stay stable when the optimizer combines keyword and range indexes.",
        ),
        (
            "composite_bool_and_not_small_range",
            Filter(
                must=[FieldCondition(key="flag", match=MatchValue(value=True))],
                must_not=[FieldCondition(key="num_range", range=Range(lt=10))],
            ),
            [3, 4, 7, 8],
            "must + must_not should keep the same semantics after boolean and numeric index selection.",
        ),
        (
            "composite_any_and_float",
            Filter(
                must=[
                    FieldCondition(key="tenant_kw", match=MatchAny(any=["alpha", "gamma"])),
                    FieldCondition(key="score_float", range=Range(gte=1.0)),
                ]
            ),
            [1, 4, 8],
            "A multi-index intersection over keyword-any and float-range should remain stable.",
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


def count_hits(client: QdrantClient, collection_name: str, count_filter: Filter) -> int:
    return int(
        client.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=True,
        ).count
    )


def query_ids_exact(client: QdrantClient, collection_name: str, query_filter: Filter) -> list[int]:
    response = client.query_points(
        collection_name=collection_name,
        query=QUERY_VECTOR,
        query_filter=query_filter,
        limit=64,
        with_payload=False,
        with_vectors=False,
        search_params=SearchParams(exact=True),
    )
    return sorted(int(point.id) for point in response.points)


def capture_case_result(client: QdrantClient, collection_name: str, query_filter: Filter) -> dict[str, object]:
    scroll = scroll_ids(client, collection_name, query_filter)
    count = count_hits(client, collection_name, query_filter)
    query = query_ids_exact(client, collection_name, query_filter)
    return {
        "scroll_ids": scroll,
        "count": count,
        "query_ids": query,
    }


def create_indexes(client: QdrantClient, collection_name: str) -> None:
    for field_name, field_schema, profile_name in index_specs():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )
        print(f"create_index: field={field_name} profile={profile_name}")


def rebuild_indexes(client: QdrantClient, collection_name: str) -> None:
    for field_name, _, _ in index_specs():
        client.delete_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=True,
        )
        print(f"delete_index: field={field_name}")
    time.sleep(0.1)
    create_indexes(client, collection_name)


def normalize_case_result(value: dict[str, object]) -> dict[str, object]:
    return {
        "scroll_ids": list(value["scroll_ids"]),
        "count": int(value["count"]),
        "query_ids": list(value["query_ids"]),
    }


def run_phase(
    client: QdrantClient,
    collection_name: str,
    transport: str,
    phase: str,
    baseline_results: dict[str, dict[str, object]] | None = None,
) -> tuple[bool, dict[str, dict[str, object]]]:
    all_passed = True
    captured: dict[str, dict[str, object]] = {}

    print(f"\n--- filter index equivalence ({transport}, {phase}) ---")
    print(f"collection={collection_name}")

    for name, query_filter, expected_ids, note in query_cases():
        actual = normalize_case_result(capture_case_result(client, collection_name, query_filter))
        captured[name] = actual

        expected = {
            "scroll_ids": expected_ids,
            "count": len(expected_ids),
            "query_ids": expected_ids,
        }

        passed = actual == expected
        detail_parts = [f"expected={expected}", f"actual={actual}"]
        if baseline_results is not None:
            baseline = baseline_results[name]
            baseline_match = actual == baseline
            passed = passed and baseline_match
            detail_parts.append(f"baseline={baseline}")

        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status} | {' | '.join(detail_parts)} | note={note}")

    return all_passed, captured


def run_transport(args: argparse.Namespace, prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(prefer_grpc)
    client = build_client(args, prefer_grpc)
    all_passed = True

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        baseline_ok, baseline_results = run_phase(
            client,
            collection_name,
            transport,
            phase="no-index",
        )
        all_passed = all_passed and baseline_ok

        create_indexes(client, collection_name)
        indexed_ok, _ = run_phase(
            client,
            collection_name,
            transport,
            phase="indexed",
            baseline_results=baseline_results,
        )
        all_passed = all_passed and indexed_ok

        rebuild_indexes(client, collection_name)
        rebuilt_ok, _ = run_phase(
            client,
            collection_name,
            transport,
            phase="rebuilt",
            baseline_results=baseline_results,
        )
        all_passed = all_passed and rebuilt_ok
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")

    return all_passed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qdrant filter equivalence before/after payload indexing")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", type=int, default=6334)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--rest-only", action="store_true")
    parser.add_argument("--grpc-only", action="store_true")
    parser.add_argument("--run-id", default="filter-index-equivalence")
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    global RUN_ID
    args = parse_args(argv)
    RUN_ID = args.run_id

    try:
        server_info = fetch_server_info(args.host, args.port)
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant filter index equivalence validation")
    print(f"target={args.host}:{args.port} grpc:{args.grpc_port}")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")
    print(f"run_id={args.run_id}")

    rest_ok = True if args.grpc_only else run_transport(args, prefer_grpc=False)
    grpc_ok = True if args.rest_only else run_transport(args, prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all filter index equivalence checks passed.")
        return 0

    print("\nSummary: at least one filter index equivalence check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
