"""
Validation for scalar-filtered Qdrant scroll pagination.

This operator focuses on paths that the random scalar fuzzer mostly exercises
with one large "fetch everything" scroll:
1. `next_page_offset` based pagination with a deep scalar filter.
2. Explicit `offset` reuse on the same filtered result set.
3. `order_by` over an indexed scalar field in both ascending and descending order.
4. Payload projection while scrolling filtered points.
"""

from __future__ import annotations

import random
import time
import urllib.request
from importlib.metadata import version as pkg_version
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Direction,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"scroll_pagination_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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
    payloads = [
        {"group": "red", "price": 10, "score": 1.0, "active": True, "city": "London"},
        {"group": "blue", "price": 20, "score": 2.0, "active": False, "city": "Berlin"},
        {"group": "red", "price": 30, "score": 3.0, "active": True, "city": "Paris"},
        {"group": "red", "price": 40, "score": 4.0, "active": False, "city": "London"},
        {"group": "blue", "price": 50, "score": 5.0, "active": True, "city": "Rome"},
        {"group": "red", "price": 60, "score": 6.0, "active": True, "city": "Berlin"},
        {"group": "green", "price": 70, "score": 7.0, "active": True, "city": "Madrid"},
        {"group": "red", "price": 80, "score": 8.0, "active": False, "city": "Paris"},
        {"group": "blue", "price": 90, "score": 9.0, "active": True, "city": "London"},
        {"group": "red", "price": 100, "score": 10.0, "active": True, "city": "Rome"},
        {"price": 110, "score": 11.0, "active": True, "city": "NoGroup"},
        {"group": None, "price": 120, "score": 12.0, "active": False, "city": "NullGroup"},
    ]
    return [
        PointStruct(id=idx, vector=[float(idx), 1.0], payload=payload)
        for idx, payload in enumerate(payloads, start=1)
    ]


def create_indexes(client: QdrantClient, collection_name: str) -> None:
    for field_name, field_schema in [
        ("group", PayloadSchemaType.KEYWORD),
        ("price", PayloadSchemaType.INTEGER),
        ("score", PayloadSchemaType.FLOAT),
        ("active", PayloadSchemaType.BOOL),
        ("city", PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


def deep_scroll_filter() -> Filter:
    red_group = FieldCondition(key="group", match=MatchValue(value="red"))
    active_expensive = Filter(
        must=[
            FieldCondition(key="active", match=MatchValue(value=True)),
            FieldCondition(key="price", range=Range(gte=90)),
        ]
    )
    not_green = FieldCondition(key="group", match=MatchValue(value="green"))
    price_floor = FieldCondition(key="price", range=Range(gte=30))

    return Filter(
        must=[
            Filter(should=[red_group, active_expensive]),
            Filter(must_not=[not_green]),
            price_floor,
        ]
    )


def red_filter() -> Filter:
    return Filter(must=[FieldCondition(key="group", match=MatchValue(value="red"))])


def ids_from_records(records: list[Any]) -> list[int]:
    return [int(point.id) for point in records]


def collect_pages(client: QdrantClient, collection_name: str, scroll_filter: Filter, limit: int) -> tuple[list[list[int]], list[Any]]:
    pages: list[list[int]] = []
    offsets: list[Any] = []
    offset = None

    for _ in range(16):
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        pages.append(ids_from_records(points))
        offsets.append(next_offset)
        if next_offset is None:
            break
        offset = next_offset
    return pages, offsets


def ordered_ids_by_price(
    client: QdrantClient,
    collection_name: str,
    scroll_filter: Filter,
    direction: Direction,
    limit: int,
) -> tuple[list[int], list[int], Any]:
    points, next_offset = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=limit,
        order_by=OrderBy(key="price", direction=direction),
        with_payload=True,
        with_vectors=False,
    )
    return ids_from_records(points), [int(point.payload["price"]) for point in points], next_offset


def payload_projection_keys(client: QdrantClient, collection_name: str, scroll_filter: Filter) -> tuple[list[int], list[list[str]]]:
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=2,
        with_payload=["group", "price"],
        with_vectors=False,
    )
    return ids_from_records(points), [sorted((point.payload or {}).keys()) for point in points]


def run_transport(prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(prefer_grpc)
    client = build_client(prefer_grpc)
    all_passed = True

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)
        create_indexes(client, collection_name)

        print(f"\n--- scroll pagination operator validation ({transport}) ---")
        print(f"collection={collection_name}")

        pages, offsets = collect_pages(client, collection_name, deep_scroll_filter(), limit=2)
        expected_pages = [[3, 4], [6, 8], [9, 10], [11]]
        page_ok = pages == expected_pages and offsets[-1] is None and len(set(sum(pages, []))) == len(sum(pages, []))
        all_passed &= page_ok
        print(f"deep_filter_page_through: {'PASS' if page_ok else 'FAIL'} | expected={expected_pages} | actual={pages} | offsets={offsets}")

        offset_points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=deep_scroll_filter(),
            limit=4,
            offset=8,
            with_payload=False,
            with_vectors=False,
        )
        offset_ids = ids_from_records(offset_points)
        offset_ok = offset_ids == [8, 9, 10, 11]
        all_passed &= offset_ok
        print(f"deep_filter_explicit_offset: {'PASS' if offset_ok else 'FAIL'} | expected={[8, 9, 10, 11]} | actual={offset_ids}")

        asc_ids, asc_prices, asc_next = ordered_ids_by_price(
            client, collection_name, red_filter(), Direction.ASC, limit=4
        )
        asc_ok = asc_ids == [1, 3, 4, 6] and asc_prices == [10, 30, 40, 60] and asc_next is None
        all_passed &= asc_ok
        print(f"order_by_price_asc_filtered: {'PASS' if asc_ok else 'FAIL'} | expected_ids={[1, 3, 4, 6]} | actual_ids={asc_ids} | prices={asc_prices} | next={asc_next}")

        desc_ids, desc_prices, desc_next = ordered_ids_by_price(
            client, collection_name, red_filter(), Direction.DESC, limit=3
        )
        desc_ok = desc_ids == [10, 8, 6] and desc_prices == [100, 80, 60] and desc_next is None
        all_passed &= desc_ok
        print(f"order_by_price_desc_filtered: {'PASS' if desc_ok else 'FAIL'} | expected_ids={[10, 8, 6]} | actual_ids={desc_ids} | prices={desc_prices} | next={desc_next}")

        projected_ids, projected_keys = payload_projection_keys(client, collection_name, deep_scroll_filter())
        projection_ok = projected_ids == [3, 4] and projected_keys == [["group", "price"], ["group", "price"]]
        all_passed &= projection_ok
        print(f"filtered_payload_projection: {'PASS' if projection_ok else 'FAIL'} | expected_ids={[3, 4]} | actual_ids={projected_ids} | payload_keys={projected_keys}")
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

    print("Qdrant scroll pagination operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all scroll pagination operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one scroll pagination operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
