import numpy as np
import time
import random
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, 
    Collection, utility
)

# --- ⚡️ 压力升级配置 ---
HOST = '127.0.0.1'
PORT = '19530'
COLLECTION_NAME = "stress_topk_consistency"
DIM = 768             # 升级：从 128 提升到 768 (模拟 LLM 向量)
NUM_ENTITIES = 500000 # 升级：从 1万 提升到 50万
SHARD_NUM = 1

# 颜色代码，让输出更显眼
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def init_collection():
    print(f"🔌 Connecting to Milvus {HOST}...")
    connections.connect("default", host=HOST, port=PORT)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, "High-Dim Top-K Consistency Test")
    collection = Collection(COLLECTION_NAME, schema, consistency_level="Strong")
    return collection

def insert_data(collection):
    print(f"📥 Generating and Inserting {NUM_ENTITIES} entities (Dim={DIM})...")
    # 分批插入，避免 Python 内存溢出
    batch_size = 10000
    total_vectors = [] # 只保留部分向量在内存用于测试，防止 OOM
    
    for i in range(0, NUM_ENTITIES, batch_size):
        ids = list(range(i, i + batch_size))
        vectors = np.random.random((batch_size, DIM)).astype(np.float32)
        collection.insert([ids, vectors])
        print(f"   Inserted batch {i} - {i+batch_size}...")
        
        # 随机抽样保存几个向量用于稍后的查询
        if len(total_vectors) < 100:
            total_vectors.extend(vectors[:10])

    print("💾 Flushing to disk...")
    collection.flush()
    print(f"✅ Insert Done. Row count: {collection.num_entities}")
    return total_vectors

def check_topk_consistency(collection, query_vec, index_tag, search_params):
    print(f"\n{'-'*60}")
    print(f"🔍 Testing Index: {YELLOW}{index_tag}{RESET} | Params: {search_params}")
    
    start_t = time.time()
    
    try:
        # Case A: Search Top-10
        res_10 = collection.search(
            data=[query_vec], anns_field="vector", 
            param=search_params, limit=10, output_fields=["id"]
        )
        ids_10 = [hit.id for hit in res_10[0]]
        dist_10 = [hit.distance for hit in res_10[0]]

        # Case B: Search Top-5
        res_5 = collection.search(
            data=[query_vec], anns_field="vector", 
            param=search_params, limit=5, output_fields=["id"]
        )
        ids_5 = [hit.id for hit in res_5[0]]
        dist_5 = [hit.distance for hit in res_5[0]]
        
        cost = (time.time() - start_t) * 1000

        # --- 可视化输出 ---
        print(f"   ⏱️  Cost: {cost:.2f} ms")
        print(f"   📌 Top-10 (First 5): {ids_10[:5]}")
        print(f"   📌 Top-5  (All):     {ids_5}")
        print(f"   📏 Top-3 Distances: {[f'{d:.6f}' for d in dist_10[:3]]}")

        # --- Oracle 判定 ---
        if ids_10[:5] == ids_5:
            print(f"{GREEN}✅ PASSED: Consistency Verified.{RESET}")
        else:
            print(f"{RED}❌ FAILED: Inconsistent!{RESET}")
            if dist_10[:5] == dist_5:
                print(f"{YELLOW}⚠️  Warning: Distances identical. Sort stability issue.{RESET}")
            else:
                print(f"{RED}🚨 Critical Bug: Search path diverged!{RESET}")

    except Exception as e:
        # 捕获参数错误，不中断测试
        if "should be larger than k" in str(e) or "out of range" in str(e):
            print(f"{YELLOW}⚠️  Expected Error Caught: Parameter invalid ({e}). Skipping this case.{RESET}")
        else:
            print(f"{RED}🚨 Unexpected Error: {e}{RESET}")

def run_test():
    collection = init_collection()
    sample_vectors = insert_data(collection)
    
    # 随机选一个向量作为查询目标
    query_vec = random.choice(sample_vectors)

    # 1. FLAT (基准)
    print("\n🏗️  Building FLAT Index (Ground Truth)...")
    collection.release()
    if collection.has_index(): collection.drop_index()
    collection.create_index("vector", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
    collection.load()
    
    check_topk_consistency(collection, query_vec, "FLAT", {"metric_type": "L2", "params": {}})

    # 2. HNSW (压力测试)
    print("\n🏗️  Building HNSW Index (M=32, efC=300)...")
    collection.release()
    if collection.has_index(): collection.drop_index()
    
    # 使用更激进的建图参数
    index_params_hnsw = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 32, "efConstruction": 300}
    }
    collection.create_index("vector", index_params_hnsw)
    print("📦 Loading HNSW into memory...")
    collection.load()

    print(f"\n🔥 Starting HNSW Stress Test on {NUM_ENTITIES} vectors...")
    
    # 这里的 ef 设置很有讲究：
    # ef=5 (小于 limit=10): 必崩？看看 Milvus 怎么处理非法参数
    # ef=9 (小于 limit=10): 边界测试
    # ef=10 (等于 limit): 临界点，最容易出 Bug
    # ef=40: 正常范围
    for search_ef in [9, 10, 15, 40, 100]:
        check_topk_consistency(
            collection, query_vec, f"HNSW (ef={search_ef})", 
            {"metric_type": "L2", "params": {"ef": search_ef}}
        )

if __name__ == "__main__":
    run_test()