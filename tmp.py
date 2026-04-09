"""
Test Qdrant's handling of 0.0 and -0.0 in range filters.
IEEE 754 requires that 0.0 and -0.0 compare equal:
  - 0.0 == -0.0   -> true
  - 0.0 <  -0.0   -> false
  - 0.0 >  -0.0   -> false
  - 0.0 <= -0.0   -> true
  - 0.0 >= -0.0   -> true

Some databases incorrectly treat -0.0 as less than 0.0 due to bitwise comparison.
This script checks whether Qdrant follows the standard.
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    Range,
    VectorParams,
)

COLLECTION_NAME = "zero_boundary_test"


def run_test(prefer_grpc: bool):
    transport = "gRPC" if prefer_grpc else "REST"
    client = QdrantClient(
        host="127.0.0.1",
        port=6333,
        grpc_port=6334,
        prefer_grpc=prefer_grpc,
        timeout=30,
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=2, distance=Distance.DOT),
    )

    # Insert two points: one with 0.0, one with -0.0
    points = [
        PointStruct(id=1, vector=[0.1, 0.1], payload={"value": 0.0}),
        PointStruct(id=2, vector=[0.1, 0.1], payload={"value": -0.0}),
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    def query_ids(range_obj: Range):
        scroll_filter = Filter(must=[FieldCondition(key="value", range=range_obj)])
        points, _ = client.scroll(
            COLLECTION_NAME, scroll_filter=scroll_filter, limit=10
        )
        return sorted([p.id for p in points])

    print(f"\n--- Qdrant 0.0 / -0.0 test ({transport}) ---")
    print("Stored: id=1 -> 0.0, id=2 -> -0.0")

    # Expected behaviour (IEEE 754):
    # Both points should be treated identically because 0.0 == -0.0.
    # Therefore any condition that matches one should match both.

    tests = [
        ("value <= 0.0", Range(lte=0.0), [1, 2]),
        ("value < 0.0", Range(lt=0.0), []),      # neither is < 0.0
        ("value >= 0.0", Range(gte=0.0), [1, 2]),
        ("value > 0.0", Range(gt=0.0), []),
        ("value == 0.0", Range(gte=0.0, lte=0.0), [1, 2]),
        # Also test with explicit -0.0 as boundary
        ("value <= -0.0", Range(lte=-0.0), [1, 2]),
        ("value < -0.0", Range(lt=-0.0), []),
        ("value >= -0.0", Range(gte=-0.0), [1, 2]),
        ("value > -0.0", Range(gt=-0.0), []),
        ("value == -0.0", Range(gte=-0.0, lte=-0.0), [1, 2]),
    ]

    all_passed = True
    for label, rng, expected in tests:
        actual = query_ids(rng)
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{label}: expected {expected}, got {actual} -> {status}")

    if not all_passed:
        print("\n[BUG DETECTED] Qdrant treats 0.0 and -0.0 differently in range filters.")
        print("This violates IEEE 754 and can cause unexpected filtering results.")

    client.delete_collection(COLLECTION_NAME)


if __name__ == "__main__":
    print("Testing Qdrant's handling of 0.0 vs -0.0...")
    run_test(prefer_grpc=True)
    run_test(prefer_grpc=False)