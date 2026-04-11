"""
Minimal validation for the Qdrant `geo_bounding_box` operator.

This script validates a conservative research subset:
1. Scalar geo values strictly inside the tested bounding box match consistently.
2. Exact bounding-box edge and corner values are backend-sensitive locally.
3. Geo arrays match if at least one element satisfies the non-boundary subset.
4. Missing/null and tested non-geo rows do not satisfy ordinary geo bounding-box predicates.
5. REST and gRPC agree on the tested server subset, while Python `:memory:` differs on exact boundaries.
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
    GeoBoundingBox,
    GeoPoint,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334


ISSUE_BBOX = GeoBoundingBox(
    top_left=GeoPoint(lat=90.0, lon=158.75),
    bottom_right=GeoPoint(lat=69.4, lon=180.0),
)

INNER_BBOX = GeoBoundingBox(
    top_left=GeoPoint(lat=81.0, lon=169.0),
    bottom_right=GeoPoint(lat=79.0, lon=171.0),
)


def unique_collection_name(backend: str) -> str:
    return f"geo_bounding_box_operator_{backend}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


def fetch_server_info() -> dict[str, object]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{HOST}:{PORT}/", timeout=5) as resp:
        import json

        return json.loads(resp.read().decode("utf-8"))


def build_server_client(prefer_grpc: bool) -> QdrantClient:
    return QdrantClient(
        host=HOST,
        port=PORT,
        grpc_port=GRPC_PORT,
        prefer_grpc=prefer_grpc,
        timeout=30,
    )


def build_points() -> list[PointStruct]:
    return [
        PointStruct(id=1, vector=[1.0, 0.0], payload={"location": {"lat": 80.0, "lon": 170.0}}),
        PointStruct(id=2, vector=[1.0, 0.0], payload={"location": {"lat": 90.0, "lon": 170.0}}),
        PointStruct(id=3, vector=[1.0, 0.0], payload={"location": {"lat": 80.0, "lon": 158.75}}),
        PointStruct(id=4, vector=[1.0, 0.0], payload={"location": {"lat": 69.4, "lon": 170.0}}),
        PointStruct(id=5, vector=[1.0, 0.0], payload={"location": {"lat": 80.0, "lon": 180.0}}),
        PointStruct(id=6, vector=[1.0, 0.0], payload={"location": {"lat": 90.0, "lon": 158.75}}),
        PointStruct(id=7, vector=[1.0, 0.0], payload={"location": {"lat": 69.4, "lon": 180.0}}),
        PointStruct(id=8, vector=[1.0, 0.0], payload={"location": {"lat": 90.0, "lon": 180.0}}),
        PointStruct(id=9, vector=[1.0, 0.0], payload={"location": {"lat": 69.399, "lon": 170.0}}),
        PointStruct(id=10, vector=[1.0, 0.0], payload={"location": {"lat": 80.0, "lon": 158.749}}),
        PointStruct(
            id=11,
            vector=[1.0, 0.0],
            payload={"locations": [{"lat": 80.0, "lon": 158.749}, {"lat": 80.0, "lon": 170.0}]},
        ),
        PointStruct(
            id=15,
            vector=[1.0, 0.0],
            payload={"locations": [{"lat": 90.0, "lon": 180.0}]},
        ),
        PointStruct(id=12, vector=[1.0, 0.0], payload={"location": None, "locations": None}),
        PointStruct(id=13, vector=[1.0, 0.0], payload={}),
        PointStruct(id=14, vector=[1.0, 0.0], payload={"location": "not_geo", "locations": ["not_geo"]}),
    ]


def scroll_ids(client: QdrantClient, collection_name: str, key: str, bbox: GeoBoundingBox) -> list[int]:
    scroll_filter = Filter(must=[FieldCondition(key=key, geo_bounding_box=bbox)])
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=64,
        with_payload=False,
        with_vectors=False,
    )
    return sorted(int(point.id) for point in points)


def run_backend(backend: str, client: QdrantClient, create_indexes: bool) -> bool:
    collection_name = unique_collection_name(backend)
    all_passed = True

    expected_by_backend = {
        "REST": {
            "safe_scalar_inner_bbox": [1],
            "issue_boundary_scalar_profile": [1],
            "issue_boundary_array_profile": [11],
        },
        "gRPC": {
            "safe_scalar_inner_bbox": [1],
            "issue_boundary_scalar_profile": [1],
            "issue_boundary_array_profile": [11],
        },
        "memory": {
            "safe_scalar_inner_bbox": [1],
            "issue_boundary_scalar_profile": [1, 2, 3, 4, 5, 6, 7, 8],
            "issue_boundary_array_profile": [11, 15],
        },
    }

    tests = [
        (
            "safe_scalar_inner_bbox",
            "location",
            INNER_BBOX,
            "A scalar point strictly inside the tested box matches consistently; missing/null/type-mismatched rows are excluded.",
        ),
        (
            "issue_boundary_scalar_profile",
            "location",
            ISSUE_BBOX,
            "REST/gRPC exclude all tested exact edge/corner scalar points, while Python `:memory:` includes them.",
        ),
        (
            "issue_boundary_array_profile",
            "locations",
            ISSUE_BBOX,
            "Geo arrays match on a non-boundary inside element across backends; a boundary-only array is backend-sensitive.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        if create_indexes:
            for field_name in ["location", "locations"]:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.GEO,
                    wait=True,
                )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- geo_bounding_box operator validation ({backend}) ---")
        print(f"collection={collection_name}")
        for name, key, bbox, note in tests:
            expected_ids = expected_by_backend[backend][name]
            actual_ids = scroll_ids(client, collection_name, key, bbox)
            passed = actual_ids == expected_ids
            if not passed:
                all_passed = False
            status = "PASS" if passed else "FAIL"
            print(
                f"{name}: {status} | key={key} | expected={expected_ids} | actual={actual_ids} | note={note}"
            )
        return all_passed
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({backend}, {collection_name}): {exc}")


def main() -> int:
    print("Qdrant geo_bounding_box operator validation")
    try:
        server_info = fetch_server_info()
        result = server_info.get("result") or server_info
        print(f"server_version={result.get('version')}")
        print(f"server_commit={result.get('commit')}")
    except Exception as exc:
        print(f"server_info_warning={exc}")
    print(f"client_version={pkg_version('qdrant-client')}")

    all_passed = True
    all_passed &= run_backend("REST", build_server_client(prefer_grpc=False), create_indexes=True)
    all_passed &= run_backend("gRPC", build_server_client(prefer_grpc=True), create_indexes=True)
    all_passed &= run_backend("memory", QdrantClient(":memory:"), create_indexes=False)

    print(f"\nRESULT: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
