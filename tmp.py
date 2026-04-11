from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    ValuesCount,
    VectorParams,
)

HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334

COLLECTION = "values_count_dict_repro"


def ids_from_scroll(client, flt):
    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=flt,
        limit=20,
        with_payload=True,
        with_vectors=False,
    )
    return sorted([p.id for p in points])


def main():
    remote = QdrantClient(host=HOST, port=PORT, grpc_port=GRPC_PORT, prefer_grpc=False)
    local = QdrantClient(":memory:")

    for client in (remote, local):
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        client.upsert(
            collection_name=COLLECTION,
            wait=True,
            points=[
                # Scalar dict: should be treated as a single element per docs, count == 1
                PointStruct(id=1, vector=[0.1, 0.2], payload={"vc": {"a": 1, "b": 2}}),
                # Actual array: count == 2
                PointStruct(id=2, vector=[0.2, 0.3], payload={"vc": ["x", "y"]}),
                # Normal scalar: count == 1
                PointStruct(id=3, vector=[0.3, 0.4], payload={"vc": 7}),
            ],
        )

    gt_1 = Filter(
        must=[
            FieldCondition(
                key="vc",
                values_count=ValuesCount(gt=1),
            )
        ]
    )
    lte_1 = Filter(
        must=[
            FieldCondition(
                key="vc",
                values_count=ValuesCount(lte=1),
            )
        ]
    )

    remote_gt = ids_from_scroll(remote, gt_1)
    remote_lte = ids_from_scroll(remote, lte_1)
    local_gt = ids_from_scroll(local, gt_1)
    local_lte = ids_from_scroll(local, lte_1)

    print("=== Remote server ===")
    print("values_count(gt=1): ", remote_gt)
    print("values_count(lte=1):", remote_lte)

    print("\n=== Local :memory: ===")
    print("values_count(gt=1): ", local_gt)
    print("values_count(lte=1):", local_lte)

    print("\n=== Expected if docs/server semantics hold ===")
    print("dict should count as 1, so:")
    print("gt=1  should match only id=2")
    print("lte=1 should match ids=1,3")

    # Assertions: remote will pass, local will currently fail
    assert remote_gt == [2], f"Unexpected remote gt=1 result: {remote_gt}"
    assert remote_lte == [1, 3], f"Unexpected remote lte=1 result: {remote_lte}"

    # This will trigger an AssertionError under Local mode due to the bug
    assert local_gt ==[2], f"local gt=1 result is {local_gt}"
    assert local_lte ==[1, 3], f"local lte=1 result is {local_lte}"


if __name__ == "__main__":
    main()