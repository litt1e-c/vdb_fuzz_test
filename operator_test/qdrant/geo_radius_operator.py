"""
Minimal validation for the Qdrant `geo_radius` operator.

This script validates a conservative documented subset:
1. `geo_radius` matches geo payload values inside a circle with radius in meters.
2. The tested point constructed exactly at the chosen radius is excluded locally.
3. Geo arrays match if at least one element satisfies the radius condition.
4. Missing/null and tested non-geo rows do not satisfy ordinary geo radius predicates.
5. REST and gRPC return the same results for the validated subset.
"""

from __future__ import annotations

import math
import random
import time
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    GeoPoint,
    GeoRadius,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334
EARTH_RADIUS_M = 6_371_000.0


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"geo_radius_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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


def north_offset_lat(center_lat: float, meters: float) -> float:
    return center_lat + math.degrees(meters / EARTH_RADIUS_M)


def build_points() -> list[PointStruct]:
    center_lat = 0.0
    center_lon = 0.0
    inside_lat = north_offset_lat(center_lat, 999.5)
    boundary_lat = north_offset_lat(center_lat, 1000.0)
    outside_lat = north_offset_lat(center_lat, 1000.5)

    return [
        PointStruct(
            id=1,
            vector=[1.0, 0.0],
            payload={
                "location": {"lat": center_lat, "lon": center_lon},
                "locations": [{"lat": center_lat, "lon": center_lon}],
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "location": {"lat": inside_lat, "lon": center_lon},
                "locations": [{"lat": inside_lat, "lon": center_lon}],
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "location": {"lat": boundary_lat, "lon": center_lon},
                "locations": [{"lat": boundary_lat, "lon": center_lon}],
            },
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={
                "location": {"lat": outside_lat, "lon": center_lon},
                "locations": [{"lat": outside_lat, "lon": center_lon}],
            },
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.8],
            payload={
                "location": {"lat": 2.0, "lon": 2.0},
                "locations": [
                    {"lat": outside_lat, "lon": center_lon},
                    {"lat": center_lat, "lon": center_lon},
                ],
            },
        ),
        PointStruct(
            id=6,
            vector=[0.8, 0.2],
            payload={
                "location": None,
                "locations": None,
            },
        ),
        PointStruct(id=7, vector=[0.3, 0.7], payload={}),
        PointStruct(
            id=8,
            vector=[0.6, 0.4],
            payload={
                "location": "not_geo",
                "locations": ["not_geo"],
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

    center = GeoPoint(lat=0.0, lon=0.0)
    tests = [
        (
            "location_radius_1000m_boundary_excluded",
            Filter(
                must=[
                    FieldCondition(
                        key="location",
                        geo_radius=GeoRadius(center=center, radius=1000.0),
                    )
                ]
            ),
            [1, 2],
            "The tested inside point is included, while the constructed exact-boundary and 1000.5m points are excluded.",
        ),
        (
            "location_zero_radius_exact_center",
            Filter(
                must=[
                    FieldCondition(
                        key="location",
                        geo_radius=GeoRadius(center=center, radius=0.0),
                    )
                ]
            ),
            [],
            "Zero radius matches no rows in the tested subset, including the exact center point.",
        ),
        (
            "locations_array_any_element",
            Filter(
                must=[
                    FieldCondition(
                        key="locations",
                        geo_radius=GeoRadius(center=center, radius=1000.0),
                    )
                ]
            ),
            [1, 2, 5],
            "Geo arrays match when at least one stored point satisfies the radius condition; the constructed boundary-only array is excluded.",
        ),
        (
            "location_missing_null_type_mismatch_excluded",
            Filter(
                must=[
                    FieldCondition(
                        key="location",
                        geo_radius=GeoRadius(center=center, radius=200000.0),
                    )
                ]
            ),
            [1, 2, 3, 4],
            "Missing/null, the tested non-geo string row, and a distant scalar geo point do not satisfy the ordinary geo radius predicate.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        for field_name in ["location", "locations"]:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.GEO,
                wait=True,
            )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- geo_radius operator validation ({transport}) ---")
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

    print("Qdrant geo_radius operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all geo_radius operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one geo_radius operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
