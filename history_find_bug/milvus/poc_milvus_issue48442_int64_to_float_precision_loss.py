import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

HOST = "127.0.0.1"
PORT = "19531"
COLLECTION = "test_json_precision_bug"

def test_json_precision_bug():
    connections.connect(host=HOST, port=PORT)

    # Clean up existing collection
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)

    # Schema with JSON field
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("json_data", DataType.JSON),
        FieldSchema("vec", DataType.FLOAT_VECTOR, dim=2),
    ])
    col = Collection(COLLECTION, schema)

    # Insert a single row with a very large int (close to int64 max)
    large_val = 9223372036854775800
    col.insert([
        [1],
        [{"num": large_val}],
        [[0.0, 0.0]]
    ])
    col.flush()

    # Index and load (required for query)
    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"})
    col.load()
    time.sleep(2)  # brief wait for load completion

    # Control query: pure int comparison – should return nothing
    query_val = 9223372036854775807
    expr_control = f'json_data["num"] in [{query_val}]'
    res_control = col.query(expr_control, output_fields=["id"])
    print(f"Control query (pure int): returned IDs: {[r['id'] for r in res_control]}")
    # Expected: []

    # Bug trigger: mixing int with float in the IN list
    expr_bug = f'json_data["num"] in [{query_val}, 1.5]'
    res_bug = col.query(expr_bug, output_fields=["id"])
    ids_bug = [r["id"] for r in res_bug]
    print(f"Bug trigger query (int + float): returned IDs: {ids_bug}")
    # Expected: [] ; Actual (with bug): [1]

    if 1 in ids_bug:
        print("Bug reproduced: Milvus incorrectly matched the large int due to float conversion.")
    else:
        print("Bug not reproduced or fixed.")

    # Clean up
    utility.drop_collection(COLLECTION)

if __name__ == "__main__":
    test_json_precision_bug()