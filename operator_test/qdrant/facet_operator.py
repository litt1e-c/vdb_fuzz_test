"""
Validation for the Qdrant `facet` API on scalar payload indexes.

This script keeps the oracle scope conservative:
1. Facets are checked on indexed scalar payload fields only.
2. Keyword, integer, and bool fields are covered with deterministic counts.
3. Facet filtering is covered by restricting counts with an ordinary filter.
4. REST and gRPC transports are both exercised against the same oracle cases.
"""

from __future__ import annotations

import argparse
import urllib.request
from importlib.metadata import version as pkg_version
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

RUN_ID = "facet-operator"


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"facet_operator_{slugify(RUN_ID, max_len=40)}_{transport}"


def fetch_server_info(host: str, port: int) -> dict[str, object]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{host}:{port}/", timeout=5) as response:
        import json

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
        PointStruct(id=1, vector=[1.0, 0.0], payload={"brand": "nike", "size": 42, "color": "red", "in_stock": True}),
        PointStruct(id=2, vector=[0.0, 1.0], payload={"brand": "adidas", "size": 42, "color": "blue", "in_stock": True}),
        PointStruct(id=3, vector=[1.0, 1.0], payload={"brand": "nike", "size": 43, "color": "red", "in_stock": False}),
        PointStruct(id=4, vector=[0.5, 0.5], payload={"brand": "puma", "size": 44, "color": "green", "in_stock": True}),
        PointStruct(id=5, vector=[0.2, 0.8], payload={"brand": "nike", "size": 44, "color": "red", "in_stock": True}),
        PointStruct(id=6, vector=[0.8, 0.2], payload={"brand": "adidas", "size": 45, "color": "red", "in_stock": False}),
    ]


def normalize_facet_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def facet_counts(response: Any) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for hit in response.hits:
        counts[normalize_facet_value(hit.value)] = int(hit.count)
    return counts


def create_indexes(client: QdrantClient, collection_name: str) -> None:
    index_specs = [
        ("brand", PayloadSchemaType.KEYWORD),
        ("color", PayloadSchemaType.KEYWORD),
        ("size", PayloadSchemaType.INTEGER),
        ("in_stock", PayloadSchemaType.BOOL),
    ]
    for field_name, schema in index_specs:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema,
            wait=True,
        )


def run_transport(args: argparse.Namespace, prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(prefer_grpc)
    client = build_client(args, prefer_grpc)
    all_passed = True

    tests = [
        (
            "facet_keyword_all",
            {"key": "brand", "limit": 10, "exact": True},
            {"nike": 3, "adidas": 2, "puma": 1},
            "Keyword facet should count indexed scalar string values.",
        ),
        (
            "facet_integer_filtered",
            {
                "key": "size",
                "limit": 10,
                "exact": True,
                "facet_filter": Filter(must=[FieldCondition(key="in_stock", match=MatchValue(value=True))]),
            },
            {42: 2, 44: 2},
            "Facet filter should restrict the counted population before integer aggregation.",
        ),
        (
            "facet_bool_all",
            {"key": "in_stock", "limit": 10, "exact": True},
            {True: 4, False: 2},
            "Bool facet should count true/false indexed scalar values.",
        ),
        (
            "facet_limit_top_two",
            {"key": "brand", "limit": 2, "exact": True},
            {"nike": 3, "adidas": 2},
            "Facet limit should return the top two deterministic counts in this fixture.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)
        create_indexes(client, collection_name)

        print(f"\n--- facet operator validation ({transport}) ---")
        print(f"collection={collection_name}")
        for name, kwargs, expected_counts, note in tests:
            response = client.facet(collection_name=collection_name, **kwargs)
            actual_counts = facet_counts(response)
            passed = actual_counts == expected_counts
            if not passed:
                all_passed = False
            status = "PASS" if passed else "FAIL"
            print(
                f"{name}: {status} | expected={expected_counts} | actual={actual_counts} | note={note}"
            )
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")

    return all_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qdrant scalar facet operator semantics")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", type=int, default=6334)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--rest-only", action="store_true")
    parser.add_argument("--grpc-only", action="store_true")
    parser.add_argument("--run-id", default="facet-operator")
    return parser.parse_args()


def main() -> int:
    global RUN_ID
    args = parse_args()
    RUN_ID = args.run_id
    try:
        server_info = fetch_server_info(args.host, args.port)
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant facet operator validation")
    print(f"target={args.host}:{args.port} grpc:{args.grpc_port}")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = True if args.grpc_only else run_transport(args, prefer_grpc=False)
    grpc_ok = True if args.rest_only else run_transport(args, prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all facet operator checks passed.")
        return 0

    print("\nSummary: at least one facet operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
