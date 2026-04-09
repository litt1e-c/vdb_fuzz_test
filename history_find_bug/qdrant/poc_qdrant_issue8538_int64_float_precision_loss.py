from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, PointStruct, Range, VectorParams


COLLECTION = "bug_int64_closed_range_aliasing"
POS = 9223372036854775806
NEG = -9223372036854775807
VALUES = [NEG - 1, NEG, NEG + 1, -1, 0, 1, POS - 1, POS]


def run(prefer_grpc: bool):
    transport = "gRPC" if prefer_grpc else "REST"
    client = QdrantClient(
        host="127.0.0.1",
        port=6333,
        grpc_port=6334,
        prefer_grpc=prefer_grpc,
        timeout=30,
    )

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=2, distance=Distance.DOT),
    )

    points = [
        PointStruct(id=i, vector=[0.1, 0.1], payload={"x": value})
        for i, value in enumerate(VALUES, start=1)
    ]
    client.upsert(collection_name=COLLECTION, points=points, wait=True)

    def query_ids(range_obj: Range):
        scroll_filter = Filter(must=[FieldCondition(key="x", range=range_obj)])
        return [p.id for p in client.scroll(COLLECTION, scroll_filter=scroll_filter, limit=20)[0]]

    print(f"--- Qdrant int64 closed-range aliasing repro ({transport}) ---")
    print("values by id:")
    for idx, value in enumerate(VALUES, start=1):
        print(f"  id={idx}: {value}")

    checks = [
        ("x >= NEG", Range(gte=NEG), [2, 3, 4, 5, 6, 7, 8]),
        ("x > NEG", Range(gt=NEG), [3, 4, 5, 6, 7, 8]),
        ("x == NEG", Range(gte=NEG, lte=NEG), [2]),
        ("x >= POS", Range(gte=POS), [8]),
        ("x > POS-1", Range(gt=POS - 1), [8]),
        ("x == POS", Range(gte=POS, lte=POS), [8]),
        ("x <= POS-1", Range(lte=POS - 1), [1, 2, 3, 4, 5, 6, 7]),
    ]

    for label, range_obj, expected in checks:
        actual = query_ids(range_obj)
        print(label)
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}")

    client.delete_collection(COLLECTION)


if __name__ == "__main__":
    run(prefer_grpc=True)
    run(prefer_grpc=False)
