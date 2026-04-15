#!/usr/bin/env python3
"""
Probe a high-value scalar bug surface: heterogeneous payload types under the same key.

This validator is intentionally deterministic and conservative:
1. A numeric filter on an integer-indexed field should only match numeric rows.
2. A keyword match on a keyword-indexed field should only match string rows.
3. `is_null` should keep explicit-null semantics even when other rows have mixed types.
4. `is_empty` should keep missing/null/[] semantics even when other rows have mixed types.
5. These answers should remain stable before indexing, after indexing, after payload
   mutation, and after index rebuild.

It is kept out of the default scalar suite until the local behavior has been
validated more broadly, because heterogeneous same-key semantics are a prime
bug-finding surface.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    IntegerIndexParams,
    IntegerIndexType,
    IsEmptyCondition,
    IsNullCondition,
    KeywordIndexParams,
    KeywordIndexType,
    MatchValue,
    PayloadField,
    PointStruct,
    Range,
    SearchParams,
    VectorParams,
)


QUERY_VECTOR = [1.0, 0.0]
RUN_ID = "heterogeneous-payload"


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"heterogeneous_payload_{slugify(RUN_ID, max_len=32)}_{transport}"


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
        timeout=30,
    )


def build_points() -> list[PointStruct]:
    return [
        PointStruct(id=1, vector=[1.0, 0.0], payload={"mixed_num": 10, "mixed_kw": "alpha"}),
        PointStruct(id=2, vector=[0.0, 1.0], payload={"mixed_num": "10", "mixed_kw": 10}),
        PointStruct(id=3, vector=[1.0, 1.0], payload={"mixed_num": True, "mixed_kw": True}),
        PointStruct(id=4, vector=[0.5, 0.5], payload={"mixed_num": None, "mixed_kw": None}),
        PointStruct(id=5, vector=[0.2, 0.8], payload={}),
        PointStruct(id=6, vector=[0.8, 0.2], payload={"mixed_num": [], "mixed_kw": []}),
        PointStruct(id=7, vector=[0.7, 0.3], payload={"mixed_num": {"v": 10}, "mixed_kw": {"v": "alpha"}}),
        PointStruct(id=8, vector=[0.3, 0.7], payload={"mixed_num": -5, "mixed_kw": "beta"}),
    ]


def create_indexes(client: QdrantClient, collection_name: str) -> None:
    specs = [
        (
            "mixed_num",
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
            "mixed_kw",
            KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True, on_disk=True),
            "keyword_tenant_on_disk",
        ),
    ]
    for field_name, field_schema, profile in specs:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )
        print(f"create_index: field={field_name} profile={profile}")


def rebuild_indexes(client: QdrantClient, collection_name: str) -> None:
    for field_name in ["mixed_num", "mixed_kw"]:
        client.delete_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            wait=True,
        )
        print(f"delete_index: field={field_name}")
    time.sleep(0.1)
    create_indexes(client, collection_name)


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
    return {
        "scroll_ids": scroll_ids(client, collection_name, query_filter),
        "count": count_hits(client, collection_name, query_filter),
        "query_ids": query_ids_exact(client, collection_name, query_filter),
    }


def normalize_case_result(value: dict[str, object]) -> dict[str, object]:
    return {
        "scroll_ids": list(value["scroll_ids"]),
        "count": int(value["count"]),
        "query_ids": list(value["query_ids"]),
    }


def range_filter(key: str, **kwargs) -> Filter:
    return Filter(must=[FieldCondition(key=key, range=Range(**kwargs))])


def match_filter(key: str, value) -> Filter:
    return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])


def is_empty_filter(key: str) -> Filter:
    return Filter(must=[IsEmptyCondition(is_empty=PayloadField(key=key))])


def is_null_filter(key: str) -> Filter:
    return Filter(must=[IsNullCondition(is_null=PayloadField(key=key))])


def phase_cases(phase: str) -> list[tuple[str, Filter, list[int], str]]:
    common: list[tuple[str, Filter, list[int], str]] = [
        (
            "mixed_num_ge_zero",
            range_filter("mixed_num", gte=0),
            [1] if phase in {"baseline", "indexed"} else [2],
            "Integer range should match only rows whose stored value is actually numeric in the tested subset.",
        ),
        (
            "mixed_num_lt_zero",
            range_filter("mixed_num", lt=0),
            [8] if phase in {"baseline", "indexed"} else [],
            "Negative integer range should exclude strings, bools, nulls, missing values, empty arrays, and objects.",
        ),
        (
            "mixed_num_is_null",
            is_null_filter("mixed_num"),
            [4] if phase in {"baseline", "indexed"} else [],
            "Only explicit null should satisfy `is_null` for the heterogeneous numeric key.",
        ),
        (
            "mixed_num_is_empty",
            is_empty_filter("mixed_num"),
            [4, 5, 6] if phase in {"baseline", "indexed"} else [1, 4, 5, 6, 8],
            "Missing, explicit null, and [] should satisfy `is_empty` for the heterogeneous numeric key.",
        ),
        (
            "mixed_kw_match_alpha",
            match_filter("mixed_kw", "alpha"),
            [1] if phase in {"baseline", "indexed"} else [2],
            "Keyword exact match should include only the tested scalar string rows.",
        ),
        (
            "mixed_kw_is_null",
            is_null_filter("mixed_kw"),
            [4] if phase in {"baseline", "indexed"} else [],
            "Only explicit null should satisfy `is_null` for the heterogeneous keyword key.",
        ),
        (
            "mixed_kw_is_empty",
            is_empty_filter("mixed_kw"),
            [4, 5, 6] if phase in {"baseline", "indexed"} else [4, 5, 6, 8],
            "Missing, explicit null, and [] should satisfy `is_empty` for the heterogeneous keyword key.",
        ),
    ]
    if phase in {"mutated", "rebuilt"}:
        common.append(
            (
                "mixed_kw_match_gamma",
                match_filter("mixed_kw", "gamma"),
                [1],
                "Overwrite should preserve the surviving scalar keyword field after the numeric field is removed.",
            )
        )
    return common


def run_phase(
    client: QdrantClient,
    collection_name: str,
    transport: str,
    phase: str,
) -> bool:
    all_passed = True

    print(f"\n--- heterogeneous payload validation ({transport}, {phase}) ---")
    print(f"collection={collection_name}")

    for name, query_filter, expected_ids, note in phase_cases(phase):
        actual = normalize_case_result(capture_case_result(client, collection_name, query_filter))
        expected = {
            "scroll_ids": expected_ids,
            "count": len(expected_ids),
            "query_ids": expected_ids,
        }
        passed = actual == expected
        print(
            f"{name}: {'PASS' if passed else 'FAIL'} | "
            f"expected={expected} | actual={actual} | note={note}"
        )
        all_passed &= passed

    return all_passed


def apply_mutations(client: QdrantClient, collection_name: str) -> None:
    client.set_payload(
        collection_name=collection_name,
        payload={"mixed_num": 10, "mixed_kw": "alpha"},
        points=[2],
        wait=True,
    )
    client.overwrite_payload(
        collection_name=collection_name,
        payload={"mixed_kw": "gamma"},
        points=[1],
        wait=True,
    )
    client.clear_payload(
        collection_name=collection_name,
        points_selector=[4],
        wait=True,
    )
    client.delete_payload(
        collection_name=collection_name,
        keys=["mixed_num", "mixed_kw"],
        points=match_filter("mixed_kw", "beta"),
        wait=True,
    )


def run_transport(args: argparse.Namespace, prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(prefer_grpc)
    client = build_client(args, prefer_grpc)
    all_passed = True

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        all_passed &= run_phase(client, collection_name, transport, "baseline")

        create_indexes(client, collection_name)
        all_passed &= run_phase(client, collection_name, transport, "indexed")

        apply_mutations(client, collection_name)
        all_passed &= run_phase(client, collection_name, transport, "mutated")

        rebuild_indexes(client, collection_name)
        all_passed &= run_phase(client, collection_name, transport, "rebuilt")
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")

    return all_passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate heterogeneous payload scalar semantics")
    parser.add_argument("--host", default="127.0.0.1", help="Qdrant REST host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant REST port")
    parser.add_argument("--grpc-port", type=int, default=6334, dest="grpc_port", help="Qdrant gRPC port")
    args = parser.parse_args(argv)

    try:
        server_info = fetch_server_info(args.host, args.port)
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant heterogeneous payload validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(args, prefer_grpc=False)
    grpc_ok = run_transport(args, prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all heterogeneous payload checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one heterogeneous payload check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
