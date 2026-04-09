import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Milvus Connection Config
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "issue_ast_parser_bug"

def main():
    connections.connect(host=HOST, port=PORT)
    
    # 1. Setup Collection
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("meta_json", DataType.JSON, nullable=True),
        FieldSchema("vec", DataType.FLOAT_VECTOR, dim=2)
    ]
    schema = CollectionSchema(fields)
    col = Collection(COLLECTION_NAME, schema)
    
    # Insert dummy data
    col.insert([{"id": 1, "meta_json": {"value": 100}, "vec": [0.0, 0.0]}])
    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"})
    col.load()
    time.sleep(1)

    print("--- Testing Milvus AST Parser ---")

    # Test 1: JSON comparison alone (WORKS)
    expr_1 = "meta_json['value'] > 50"
    try:
        col.query(expr_1, output_fields=["id"])
        print(f"✅ Query 1 Passed: {expr_1}")
    except Exception as e:
        print(f"❌ Query 1 Failed: {e}")

    # Test 2: Literal boolean AND JSON comparison (FAILS)
    expr_2 = "true or (meta_json['value'] > 50)"
    try:
        col.query(expr_2, output_fields=["id"])
        print(f"✅ Query 2 Passed: {expr_2}")
    except Exception as e:
        print(f"❌ Query 2 Failed (BUG REPRODUCED): {e}")

    utility.drop_collection(COLLECTION_NAME)
    connections.disconnect("default")

if __name__ == "__main__":
    main()