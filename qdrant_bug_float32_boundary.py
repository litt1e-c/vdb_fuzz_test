import math

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    Range,
    VectorParams,
)


COLLECTION = "bug_float32_boundary"
FLOAT32_MAX = 3.402823466385289e38


def run():
    client = QdrantClient(host="127.0.0.1", port=6333, grpc_port=6334, prefer_grpc=True)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=2, distance=Distance.DOT),
    )

    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=1, vector=[0.1, 0.1], payload={"x": FLOAT32_MAX}),
            PointStruct(id=2, vector=[0.1, 0.1], payload={"x": -FLOAT32_MAX}),
        ],
    )

    stored_points = client.retrieve(
        collection_name=COLLECTION,
        ids=[1, 2],
        with_payload=True,
        with_vectors=False,
    )
    stored_values = {point.id: point.payload["x"] for point in stored_points}

    pos_low = math.nextafter(FLOAT32_MAX, -math.inf)
    pos_high = math.nextafter(FLOAT32_MAX, math.inf)
    neg_low = math.nextafter(-FLOAT32_MAX, -math.inf)
    neg_high = math.nextafter(-FLOAT32_MAX, math.inf)

    tests = [
        (
            "exact closed range at +FLOAT32_MAX",
            Filter(must=[FieldCondition(key="x", range=Range(gte=FLOAT32_MAX, lte=FLOAT32_MAX))]),
            [1],
        ),
        (
            "nextafter-bracketed range around +FLOAT32_MAX",
            Filter(must=[FieldCondition(key="x", range=Range(gte=pos_low, lte=pos_high))]),
            [1],
        ),
        (
            "exact closed range at -FLOAT32_MAX",
            Filter(must=[FieldCondition(key="x", range=Range(gte=-FLOAT32_MAX, lte=-FLOAT32_MAX))]),
            [2],
        ),
        (
            "nextafter-bracketed range around -FLOAT32_MAX",
            Filter(must=[FieldCondition(key="x", range=Range(gte=neg_low, lte=neg_high))]),
            [2],
        ),
    ]

    print("--- Qdrant float32 extreme boundary repro ---")
    print(f"Inserted values: id=1 -> {FLOAT32_MAX}, id=2 -> {-FLOAT32_MAX}")
    print(f"Stored values:   id=1 -> {stored_values.get(1)!r}, id=2 -> {stored_values.get(2)!r}")
    print(f"+FLOAT32_MAX neighbors: low={pos_low}, high={pos_high}")
    print(f"-FLOAT32_MAX neighbors: low={neg_low}, high={neg_high}")
    for name, scroll_filter, expected in tests:
        actual = [p.id for p in client.scroll(COLLECTION, scroll_filter=scroll_filter, with_payload=False, limit=10)[0]]
        print(f"{name}")
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}")


if __name__ == "__main__":
    run()
