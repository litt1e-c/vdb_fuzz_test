import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "test_3vl_bug"

def test_3vl_bug():
    connections.connect(host=HOST, port=PORT)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME, timeout=60.0)

    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("c5", DataType.BOOL),
        FieldSchema("evo", DataType.INT16, nullable=True),
        FieldSchema("vec", DataType.FLOAT_VECTOR, dim=2)
    ])
    col = Collection(COLLECTION_NAME, schema)

    data = [[1], [False], [None], [[0.0, 0.0]]]
    col.insert(data)
    col.flush()
    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"})
    col.load()
    time.sleep(1)

    expr = 'not ((c5 == true) and (c5 is not null) and (evo == 574 or evo == 1))'
    res = col.query(expr, output_fields=["id"])
    ids = [r["id"] for r in res]

    print(f"Returned IDs: {ids}")
    if 1 in ids:
        print("PASS: Bug not present")
    else:
        print("FAIL: Record missing – 3VL bug present")

    utility.drop_collection(COLLECTION_NAME)

if __name__ == "__main__":
    test_3vl_bug()