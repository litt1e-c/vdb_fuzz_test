"""
Minimal reproduction of Qdrant float32 boundary filtering bugs.

Observed issues (based on fuzzer seed 1171171189):
1. Equality (==) never matches a point with value == FLT_MIN.
2. Strict greater-than (>) on REST incorrectly includes the point (c11 > FLT_MIN returns extra IDs).
3. Strict less-than (<) on gRPC incorrectly includes the point.
4. Non-strict comparisons (<= / >=) are inconsistent between gRPC and REST.

Expected behavior (IEEE 754):
- FLT_MIN should be included in <= and >=
- FLT_MIN should be excluded from < and >
- FLT_MIN should be included in ==

This script tests all five comparisons against a single point with value = FLT_MIN.
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

COLLECTION_NAME = "float_min_boundary_repro"
FLOAT_MIN = -3.4028234663852886e38  # float32 minimum finite value


def run_test(prefer_grpc: bool):
    transport = "gRPC" if prefer_grpc else "REST"
    client = QdrantClient(
        host="127.0.0.1",
        port=6333,
        grpc_port=6334,
        prefer_grpc=prefer_grpc,
        timeout=30,
    )

    # Cleanup previous run
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Create collection with dummy vectors
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=2, distance=Distance.DOT),
    )

    # Insert a single point with c15 = FLOAT_MIN and a non-empty scores_array
    # (mirroring the fuzzer condition: scores_array not null)
    point = PointStruct(
        id=1,
        vector=[0.1, 0.1],
        payload={
            "c11": FLOAT_MIN,   # same as c15, just naming for clarity
            "c15": FLOAT_MIN,
            "scores_array": [1.0],
        },
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point], wait=True)

    def query_ids(range_obj: Range, key: str = "c15"):
        """Return point IDs matching numeric range condition on given key."""
        scroll_filter = Filter(must=[FieldCondition(key=key, range=range_obj)])
        points, _ = client.scroll(
            COLLECTION_NAME, scroll_filter=scroll_filter, limit=10
        )
        return [p.id for p in points]

    print(f"\n--- Qdrant float32 boundary test ({transport}) ---")
    print(f"Stored value = {FLOAT_MIN} (point id=1)")

    # Test cases: (description, key, range, expected_ids)
    tests = [
        # Original boundary tests (using key="c15")
        ("c15 <= FLOAT_MIN", "c15", Range(lte=FLOAT_MIN), [1]),
        ("c15 < FLOAT_MIN",  "c15", Range(lt=FLOAT_MIN), []),
        ("c15 >= FLOAT_MIN", "c15", Range(gte=FLOAT_MIN), [1]),
        ("c15 > FLOAT_MIN",  "c15", Range(gt=FLOAT_MIN), []),
        ("c15 == FLOAT_MIN", "c15", Range(gte=FLOAT_MIN, lte=FLOAT_MIN), [1]),
        # Explicit tests for the fuzzer's observed extra IDs (c11 > FLOAT_MIN)
        ("c11 > FLOAT_MIN (fuzzer extra IDs)", "c11", Range(gt=FLOAT_MIN), []),
        ("c11 < FLOAT_MIN (symmetry)",          "c11", Range(lt=FLOAT_MIN), []),
    ]

    all_passed = True
    for label, key, rng, expected in tests:
        actual = query_ids(rng, key=key)
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{label}: expected {expected}, got {actual} -> {status}")

    if not all_passed:
        print(f"\n[BUG DETECTED] Incorrect filtering for float32 min value.")
        print("  - Equality never matches.")
        print("  - Strict greater-than (>) fails on REST (extra IDs).")
        print("  - Strict less-than (<) fails on gRPC (extra IDs).")
        print("  - Non-strict (<= / >=) inconsistent across transports.")

    client.delete_collection(COLLECTION_NAME)


if __name__ == "__main__":
    print("Reproducing Qdrant float32 boundary filtering bugs...")
    run_test(prefer_grpc=True)   # gRPC transport
    run_test(prefer_grpc=False)  # REST transport
