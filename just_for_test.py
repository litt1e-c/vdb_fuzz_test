import time
import random
import json
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# --- 配置 ---
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "test_issue_45670_json_array"
DIM = 8

def reproduce():
    # 1. 连接 Milvus
    print(f"🔌 Connecting to {HOST}:{PORT}...")
    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # 2. 清理环境
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # 3. 定义 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="meta_json", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    col = Collection(COLLECTION_NAME, schema)
    
    # 4. 插入数据
    print("🌊 Inserting data...")
    rows = []
    target_count = 0
    for i in range(1000):
        vec = [random.random() for _ in range(DIM)]
        if i % 2 == 0:
            json_payload = {"tags": [10, 20], "info": "target"}
            target_count += 1
        else:
            json_payload = {"tags": [30, 40], "info": "noise"}
            
        rows.append({
            "id": i,
            "vector": vec,
            "meta_json": json_payload
        })
    
    col.insert(rows)
    print(f"✅ Inserted {len(rows)} rows. Expected hits: {target_count}")

    # --- Prerequisite: Vector Index ---
    print("🔨 Building Vector Index...")
    col.create_index("vector", {"index_type": "FLAT", "metric_type": "L2", "params": {}})

    # 5. 【阶段一：内存查询】
    print("📥 Loading collection for Phase 1...")
    col.load()
    
    expr = 'meta_json["tags"][0] == 10'
    print(f"\n🔍 [Phase 1] Querying Growing Segment (Memory)...")
    print(f"   Expr: {expr}")
    
    res_mem = col.query(expr, output_fields=["id"])
    print(f"   Hits: {len(res_mem)}")
    
    if len(res_mem) == target_count:
        print("   -> ✅ Memory Query Correct.")
    else:
        print(f"   -> ⚠️ Memory Query Incorrect! Expected {target_count}, got {len(res_mem)}")

    # 6. 【制造 Issue 场景：建立通用 JSON 索引】
    print(f"\n🔨 [Trigger] Building Generic Inverted Index on 'meta_json'...")
    col.release() 
    
    try:
        # ========================================================
        # 【关键修复】：不指定 json_path，建立全字段倒排索引
        # 这会覆盖 meta_json 下的所有路径，包括 "tags"
        # 优化器容易因此误判，认为可以用这个索引去查 tags[0]
        # ========================================================
        index_params = {
            "index_type": "INVERTED", 
            "params": {} # 空参数表示索引整个 JSON
        }
        col.create_index("meta_json", index_name="idx_json_generic", index_params=index_params)
        print("   -> JSON Generic Index built successfully.")
    except Exception as e:
        print(f"   -> ❌ Index build failed: {e}")
        return

    # 7. 【阶段二：落盘与重载】
    print(f"\n💾 [Phase 2] Flushing and Reloading (Sealed Segment)...")
    col.flush()
    col.load()
    
    # 8. 【阶段三：磁盘查询】
    print(f"\n🔍 [Phase 3] Querying Sealed Segment (Disk + Index)...")
    try:
        res_disk = col.query(expr, output_fields=["id"])
        print(f"   Hits: {len(res_disk)}")
        
        if len(res_disk) == target_count:
            print("   -> ✅ Sealed Query Correct. (System works fine)")
        elif len(res_disk) == 0:
            print("   -> ❌ BUG REPRODUCED! Hits dropped to 0 after Flush/Index.")
            print("      [Analysis]: Optimizer failed to handle array index access on sealed segment.")
        else:
            print(f"   -> ⚠️ Result Mismatch. Expected {target_count}, got {len(res_disk)}")
            
    except Exception as e:
        print(f"   -> 💥 Query Crashed: {e}")

if __name__ == "__main__":
    reproduce()