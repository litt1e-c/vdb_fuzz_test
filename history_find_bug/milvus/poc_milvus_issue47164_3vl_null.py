import random
import time
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# ==============================================================================
# Bug Reproduction Script: JSON NULL Logic Violation
# ==============================================================================
# Issue Description:
# When performing a Not Equal (!=) comparison on a JSON field, Milvus incorrectly
# treats NULL JSON fields (or missing keys) as "Valid TRUE".
#
# Standard SQL Three-Valued Logic (3VL):
#   NULL != "Value"  =>  UNKNOWN (Null) -> Should be filtered out in WHERE clause.
#
# Current Milvus Behavior:
#   NULL != "Value"  =>  TRUE           -> Incorrectly returned in results.
# ==============================================================================

# Configuration
HOST = "127.0.0.1"
PORT = "19531"  # Adjust if using a different port
COLLECTION_NAME = "issue_repro_json_null_logic"
DIM = 8

def run_reproduction():
    print(f"[{time.strftime('%H:%M:%S')}] 🔌 Connecting to Milvus at {HOST}:{PORT}...")
    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # 1. Prepare Collection
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    print(f"[{time.strftime('%H:%M:%S')}] 🛠️ Creating collection '{COLLECTION_NAME}'...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        # Crucial: The JSON field is nullable
        FieldSchema(name="meta", dtype=DataType.JSON, nullable=True)
    ]
    schema = CollectionSchema(fields, description="Reproduction for JSON NULL Logic")
    col = Collection(COLLECTION_NAME, schema)

    # 2. Insert Test Data (Covering all logical states)
    # We define 4 distinct cases to isolate the logic.
    vectors = [[random.random() for _ in range(DIM)] for _ in range(4)]
    
    data = [
        # Case A: Key exists, Value is 'blue' (Should NOT match '!= blue')
        {"id": 1, "embedding": vectors[0], "meta": {"color": "blue"}},
        
        # Case B: Key exists, Value is 'red' (Should match '!= blue')
        {"id": 2, "embedding": vectors[1], "meta": {"color": "red"}},
        
        # Case C: Key is missing in valid JSON (Should evaluate to NULL/False)
        {"id": 3, "embedding": vectors[2], "meta": {"shape": "circle"}},
        
        # Case D: JSON Field is explicitly NULL (Should evaluate to NULL/False)
        {"id": 4, "embedding": vectors[3], "meta": None},
    ]

    print(f"[{time.strftime('%H:%M:%S')}] 🌊 Inserting 4 test rows:")
    print("   ID 1: meta = {'color': 'blue'} (Matches target value)")
    print("   ID 2: meta = {'color': 'red'}  (Differs from target)")
    print("   ID 3: meta = {'shape': '...' } (Missing key)")
    print("   ID 4: meta = None              (NULL Field)")
    
    col.insert(data)
    col.flush()
    
    # Index is required for loading
    col.create_index("embedding", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()

    # 3. Execute Query
    # Logic: Find items where color is NOT 'blue'.
    # Expected (Standard): ID 2 only.
    #   - ID 1 is 'blue' -> False
    #   - ID 3 is missing -> Null (False)
    #   - ID 4 is Null    -> Null (False)
    target_expr = 'meta["color"] != "blue"'
    
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔍 Executing Query: {target_expr}")
    # Using Strong consistency to ensure immediate visibility
    res = col.query(target_expr, output_fields=["id"], consistency_level="Strong")
    
    returned_ids = sorted([r["id"] for r in res])
    print(f"👉 Returned IDs: {returned_ids}")

    # 4. Validate Results
    print("\n" + "="*50)
    print("📊 RESULT ANALYSIS")
    print("="*50)

    # Check ID 1 (Equality Control)
    if 1 in returned_ids:
        print("❌ ID 1 (Value == Target): RETURNED (Incorrect, 'blue' == 'blue')")
    else:
        print("✅ ID 1 (Value == Target): Correctly excluded.")

    # Check ID 2 (Inequality Control)
    if 2 in returned_ids:
        print("✅ ID 2 (Value != Target): Correctly returned.")
    else:
        print("❌ ID 2 (Value != Target): MISSING (Incorrect).")

    # Check ID 3 (Missing Key)
    if 3 in returned_ids:
        print("⚠️ ID 3 (Missing Key): RETURNED.")
        print("   Current Behavior: Missing Keys are treated as valid inequality.")
    else:
        print("✅ ID 3 (Missing Key): Correctly excluded.")

    # Check ID 4 (NULL Field) - THE CORE BUG
    if 4 in returned_ids:
        print("🚨 ID 4 (NULL Field): RETURNED -> BUG CONFIRMED!")
        print("   Logic Violation: NULL != 'Value' evaluated to TRUE.")
        print("   Expected: NULL != 'Value' should evaluate to NULL (False in filter).")
    else:
        print("✅ ID 4 (NULL Field): Correctly excluded.")

    # Cleanup
    utility.drop_collection(COLLECTION_NAME)

if __name__ == "__main__":
    run_reproduction()