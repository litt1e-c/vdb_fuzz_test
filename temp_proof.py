import random
import time
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# === 配置 ===
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "milvus_bug_repro_dynamic" # 改个名
DIM = 128
NUM_ROWS = 3000
TARGET_ID = 1500  # 埋雷点

def run_dirty_repro():
    print(f"🔌 Connecting to {HOST}:{PORT}...")
    connections.connect("default", host=HOST, port=PORT)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # =================================================================
    # 关键点 1: 必须开启 enable_dynamic_field=True
    # Fuzzer 开启了这个，而之前的简化脚本没开。
    # 这会改变 Milvus 底层对数据的存储和检索方式。
    # =================================================================
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="val", dtype=DataType.DOUBLE, nullable=True),
    ]
    schema = CollectionSchema(fields, enable_dynamic_field=True) 
    col = Collection(COLLECTION_NAME, schema)

    print("🌊 Generating and Inserting data (Simulating Fuzzer)...")
    
    # =================================================================
    # 关键点 2: 模拟碎片化插入 (Batch Insert + Flush)
    # 不一次性插完，而是分批插，并在中间 Flush。
    # 这会强制生成多个 Segment，触发更复杂的查询合并逻辑。
    # =================================================================
    batch_size = 500
    for start in range(0, NUM_ROWS, batch_size):
        ids = []
        vectors = []
        vals = []
        
        for i in range(start, start + batch_size):
            ids.append(i)
            vectors.append([random.random() for _ in range(DIM)])
            
            # 埋雷
            if i == TARGET_ID:
                vals.append(None) # <--- NULL
            else:
                vals.append(100.0)
        
        col.insert([ids, vectors, vals])
        # 模拟 Fuzzer 的频繁落盘
        col.flush() 
        print(f"   - Inserted & Flushed batch {start}-{start+batch_size}")

    # 建索引
    print("🔨 Creating Index...")
    index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}}
    col.create_index("vector", index_params)
    col.load()

    # ==========================================
    # 验证环节
    # ==========================================
    print("\n" + "="*50)
    print("⚖️  VERIFICATION")
    print("="*50)
    
    # 1. 简单标量查 (通常全表扫描，可能没问题)
    res_scan = col.query("val <= 10.0", output_fields=["id"])
    print(f"1️⃣  Query 'val <= 10.0': found {len(res_scan)} rows.")

    # 2. 主键 + 标量 (触发点查优化路径)
    expr_pk = f"id == {TARGET_ID} and val <= 10.0"
    res_pk = col.query(expr_pk, output_fields=["id", "val"])
    print(f"2️⃣  Query '{expr_pk}': found {len(res_pk)} rows.")

    if len(res_pk) > 0:
        row = res_pk[0]
        if row['val'] is None:
            print("\n❌ [BUG REPRODUCED]")
            print(f"   Milvus returned ID {row['id']} with val={row['val']}")
            print("   Conditions met:")
            print("   1. enable_dynamic_field=True")
            print("   2. Multiple flushed segments")
            print("   3. PK Lookup + Scalar Filter")
    else:
        print("\n✅ Bug still not reproduced. This is extremely specific.")

if __name__ == "__main__":
    run_dirty_repro()