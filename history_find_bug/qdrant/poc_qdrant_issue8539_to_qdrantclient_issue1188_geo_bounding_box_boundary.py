"""
Canonical repro for the already-known Qdrant GEO boundary issue family.

This file is intentionally kept as the single representative POC for the
North-Pole / GEO boundary problem family that also explains the later
`qdrant-client` issue 1188 style behavior. Newer local POCs that turned out to
be downstream manifestations of this same root cause should be removed instead
of counted as independent bugs.
"""

from __future__ import annotations

import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    GeoBoundingBox,
    GeoPoint,
    PointStruct,
    VectorParams,
)


COLLECTION = "geo_boundary_test"


def run_test(client_type: str, client: QdrantClient) -> list[int]:
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=2, distance=Distance.DOT),
    )

    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.1],
                payload={"location_geo": {"lat": 90.0, "lon": 180.0}},
            )
        ],
        wait=True,
    )

    bbox = GeoBoundingBox(
        top_left=GeoPoint(lon=158.75, lat=90.0),
        bottom_right=GeoPoint(lon=180.0, lat=69.4),
    )

    geo_filter = Filter(must=[FieldCondition(key="location_geo", geo_bounding_box=bbox)])
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=geo_filter,
        limit=16,
        with_payload=False,
        with_vectors=False,
    )
    ids = [int(point.id) for point in results]
    print(f"[{client_type}] Actual IDs returned: {ids}")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the canonical Qdrant GEO bounding-box boundary issue"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", type=int, default=6334, dest="grpc_port")
    parser.add_argument("--prefer-grpc", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("--- Qdrant GeoBoundingBox Consistency Test ---")

    client_memory = QdrantClient(":memory:")
    expected_ids = run_test("Memory Emulator", client_memory)

    try:
        client_server = QdrantClient(
            host=args.host,
            port=args.port,
            grpc_port=args.grpc_port,
            prefer_grpc=args.prefer_grpc,
            timeout=30,
        )
        actual_ids = run_test("Rust Server", client_server)
    except Exception as exc:
        print(f"[Rust Server] Connection failed: {exc}")
        return 1

    print(f"Expected from memory baseline: {expected_ids}")
    if actual_ids == expected_ids:
        print("result=PASS")
        return 0

    print("result=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
