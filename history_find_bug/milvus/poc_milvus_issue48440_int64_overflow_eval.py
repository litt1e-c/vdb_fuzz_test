import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

HOST = "127.0.0.1"
PORT = "19531"
COLLECTION = "test_int_overflow"

def test_int_overflow():
    connections.connect(host=HOST, port=PORT)

    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)

    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("c8", DataType.INT64),
        FieldSchema("vec", DataType.FLOAT_VECTOR, dim=2),
    ])
    col = Collection(COLLECTION, schema)

    ids = [1, 2]
    c8_vals = [9223372036854775806, 100]
    vecs = [[0.0, 0.0], [0.0, 0.0]]
    col.insert([ids, c8_vals, vecs])
    col.flush()

    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()
    time.sleep(1)

    expr = "c8 + 33 <= 19974"
    results = col.query(expr, output_fields=["id", "c8"])

    print(results)

if __name__ == "__main__":
    test_int_overflow()