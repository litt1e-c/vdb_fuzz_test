#!/usr/bin/env python3
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import time

HOST = "127.0.0.1"
PORT = 19531

def setup_collection():
    fields = [
        FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("int_field", DataType.INT64),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=128)
    ]
    schema = CollectionSchema(fields)
    # If collection exists with different schema, drop it to recreate a fresh one
    from pymilvus import utility
    if utility.has_collection("div_zero_repro"):
        print("Found existing collection 'div_zero_repro', dropping it to recreate")
        try:
            existing = Collection("div_zero_repro")
            existing.drop()
        except Exception as e:
            print("Warning: failed to drop existing collection:", e)

    coll = Collection("div_zero_repro", schema=schema)

    # Insert sample data
    coll.insert([
        {"pk": 1, 'vector': [0.1] * 128, "int_field": 10},
        {"pk": 2, 'vector': [0.1] * 128, "int_field": 0},
    ])
    coll.flush()
    # Create an index for the vector field before load to satisfy server requirements
    index_params = {"index_type": "FLAT", "params": {"metric_type": "L2"}}
    coll.create_index("vector", index_params=index_params)
    coll.load()
    return coll

def main():
    print("Connecting to Milvus...")
    connections.connect(host=HOST, port=PORT)

    print("Creating collection...")
    coll = setup_collection()

    time.sleep(1)

    print("\nRunning parameterized division-by-zero query...")
    expr = "int_field / {d} == 1"
    params = {"d": 0}

    # This should return an error,
    # but currently crashes Milvus server (SIGFPE)
    print("expr:", expr)
    print("expr_params:", params)

    try:
        coll.query(expr=expr, expr_params=params)
        print("Query finished (unexpected)")
    except Exception as e:
        print("Client exception:", e)

    print("\nIf Milvus crashed, the server container will exit with code 136.")

if __name__ == "__main__":
    main()
