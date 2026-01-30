import time
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# --- Configuration ---
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "milvus_3vl_bugs_proof"

def run_proof():
    print(f"🔌 Connecting to Milvus at {HOST}:{PORT}...")
    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # 1. Setup Collection
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # Schema design:
    # We need a Nullable field (c_null) and helper fields (c_true, c_false)
    # to construct valid expressions without parser errors on constants like "1==0".
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=8),
        FieldSchema(name="c_null", dtype=DataType.BOOL, nullable=True),  # Target: NULL
        FieldSchema(name="c_true", dtype=DataType.BOOL),                 # Helper: TRUE
        FieldSchema(name="c_false", dtype=DataType.BOOL),                # Helper: FALSE
    ]
    
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    col = Collection(COLLECTION_NAME, schema)
    
    # 2. Insert Data
    # Row 1: c_null is explicitly None
    print("⚡ Inserting 1 row data: {'c_null': None, 'c_true': True, 'c_false': False}...")
    col.insert([{
        "id": 1,
        "vector": [0.1] * 8,
        "c_null": None,
        "c_true": True,
        "c_false": False
    }])
    col.flush()
    col.create_index("vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()

    print("\n🧪 Starting Logic Verification Tests...")
    print("=" * 80)
    print(f"{'Test Case':<35} | {'Expr':<45} | {'Result':<10} | {'Status'}")
    print("-" * 80)

    # ---------------------------------------------------------
    # Case 0: Sanity Check
    # Ensure the data exists and basic NULL check works.
    # ---------------------------------------------------------
    expr_0 = "c_null is null"
    res_0 = col.query(expr_0, output_fields=["id"])
    print(f"{'0. Sanity Check':<35} | {expr_0:<45} | {len(res_0):<10} | {'✅ Pass' if len(res_0)==1 else '❌ FAIL'}")

    # ---------------------------------------------------------
    # Bug 1: NOT (NOT (IS NULL))
    # Logic: 
    #   c_null IS NULL -> True (Valid)
    #   NOT (True) -> False
    #   NOT (False) -> True
    # Expected: 1 Hit
    # Actual: 0 Hits (Likely due to aggressive validity masking in NOT op)
    # ---------------------------------------------------------
    expr_1 = "not (not (c_null is null))"
    res_1 = col.query(expr_1, output_fields=["id"])
    status_1 = "❌ FAIL" if len(res_1) == 0 else "✅ Pass"
    print(f"{'1. Double NOT on IS NULL':<35} | {expr_1:<45} | {len(res_1):<10} | {status_1}")

    # ---------------------------------------------------------
    # Bug 2: NOT (Null AND False)
    # Logic:
    #   (c_null == true)  -> Unknown/Null
    #   (c_false == true) -> False
    #   Unknown AND False -> False (Standard 3VL Short-circuit)
    #   NOT (False)       -> True
    # Expected: 1 Hit
    # Actual: 0 Hits (Milvus computes Unknown AND False = Unknown)
    # ---------------------------------------------------------
    expr_2 = "not ((c_null == true) and (c_false == true))"
    res_2 = col.query(expr_2, output_fields=["id"])
    status_2 = "❌ FAIL" if len(res_2) == 0 else "✅ Pass"
    print(f"{'2. NOT (Null AND False)':<35} | {expr_2:<45} | {len(res_2):<10} | {status_2}")

    # ---------------------------------------------------------
    # Bug 3: True OR Null
    # Logic:
    #   (c_true == true) -> True
    #   (c_null == true) -> Unknown/Null
    #   True OR Unknown  -> True (Standard 3VL Short-circuit)
    # Expected: 1 Hit
    # Actual: 0 Hits (Milvus computes True OR Unknown = Unknown)
    # ---------------------------------------------------------
    expr_3 = "(c_true == true) or (c_null == true)"
    res_3 = col.query(expr_3, output_fields=["id"])
    status_3 = "❌ FAIL" if len(res_3) == 0 else "✅ Pass"
    print(f"{'3. True OR Null':<35} | {expr_3:<45} | {len(res_3):<10} | {status_3}")

    print("-" * 80)
    
    # Final Report
    failures = []
    if len(res_1) == 0: failures.append("1. Incorrect Validity propagation in NOT operator.")
    if len(res_2) == 0: failures.append("2. Missing short-circuit logic for AND (Null AND False != False).")
    if len(res_3) == 0: failures.append("3. Missing short-circuit logic for OR (True OR Null != True).")

    if failures:
        print("\n🚨 BUG CONFIRMED: Milvus 3VL implementation is flawed.")
        print("Reasons:")
        for f in failures:
            print(f" - {f}")
    else:
        print("\n✅ All logic tests passed.")

if __name__ == "__main__":
    run_proof()