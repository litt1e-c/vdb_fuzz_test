"""
EDC "TIME MACHINE" DIAGNOSTIC
Focus: Isolating Time Travel behavior on Growing vs Sealed segments.
"""
from __future__ import annotations
import time
import uuid
import warnings
import numpy as np
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# --- Config ---
DIM = 128
SEED = 2025

warnings.filterwarnings("ignore")

def main():
    print(f"{'='*60}")
    print(f"🕵️ EDC TIME MACHINE: Milvus Diagnostics")
    print(f"{'='*60}\n")
    
    try:
        connections.connect("default", host="127.0.0.1", port="19530")
    except:
        print("❌ Connection Failed")
        return

    col_name = f"diag_{uuid.uuid4().hex[:4]}"
    
    # Setup
    if utility.has_collection(col_name): utility.drop_collection(col_name)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="val", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields)
    # 使用 Strong 保证普通查询能看到数据
    col = Collection(col_name, schema, consistency_level="Strong")
    
    # Index & Load
    col.create_index("vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()
    
    rng = np.random.default_rng(SEED)
    vec = rng.random(DIM).astype(np.float32)
    tid = 100
    
    print("👉 Phase A: Growing Segment (Pre-Flush)")
    res = col.insert([[tid], [vec], [1]])
    ts_insert = res.timestamp
    print(f"   [Insert] TSO: {ts_insert}")
    
    # 稍微等一下，确保 TSO 推进
    time.sleep(0.5)
    
    # 1. 普通 Strong Search
    res1 = col.search([vec], "vector", {"metric_type": "L2", "params": {}}, limit=1)
    print(f"   [Standard Search]: {'✅ Found' if res1 else '❌ Empty'}")
    
    # 2. Time Travel Search (At Insert Time)
    # 我们加一点点 padding 确保覆盖
    ts_query = ts_insert + 100 
    try:
        res2 = col.search([vec], "vector", {"metric_type": "L2", "params": {}}, limit=1, 
                          travel_timestamp=ts_query)
        print(f"   [TimeTravel Search]: {'✅ Found' if res2 else '❌ Empty'} (TS={ts_query})")
    except Exception as e:
        print(f"   [TimeTravel Search]: 💥 Error {e}")

    print("\n👉 Phase B: Sealed Segment (Post-Flush)")
    col.flush() # Force seal
    # Re-load index info might be needed logic-wise, but flush handles persistence
    
    # 3. Time Travel Search again (Same TS)
    try:
        res3 = col.search([vec], "vector", {"metric_type": "L2", "params": {}}, limit=1, 
                          travel_timestamp=ts_query)
        print(f"   [TimeTravel Search]: {'✅ Found' if res3 else '❌ Empty'} (Post-Flush)")
    except Exception as e:
        print(f"   [TimeTravel Search]: 💥 Error {e}")
        
    print("\n👉 Phase C: Deletion Test")
    res_del = col.delete(f"id in [{tid}]")
    ts_delete = res_del.timestamp
    print(f"   [Delete] TSO: {ts_delete}")
    col.flush() # Ensure delete is processed into delta logs
    
    # 4. Search BEFORE Delete
    try:
        ts_pre_del = ts_delete - 100
        res4 = col.search([vec], "vector", {"metric_type": "L2", "params": {}}, limit=1, 
                          travel_timestamp=ts_pre_del)
        print(f"   [Pre-Delete Search]: {'✅ Found' if res4 else '❌ Empty'} (TS={ts_pre_del})")
    except Exception as e:
        print(f"   [Pre-Delete Search]: 💥 Error {e}")

    # Cleanup
    utility.drop_collection(col_name)

if __name__ == "__main__":
    main()