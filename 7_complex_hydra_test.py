import numpy as np
import time
import random
import sys
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, 
    Collection, utility, MilvusException
)

# 检查依赖
try:
    from sklearn.datasets import make_blobs
except ImportError:
    print("⚠️  scikit-learn not found. Please run: pip install scikit-learn")
    sys.exit(1)

# ==============================================================================
# 🔥 全局配置 (稳健版)
# ==============================================================================
HOST = '127.0.0.1'
PORT = '19530'
COLLECTION_NAME = "hydra_hybrid_test"
DIM = 768             
NUM_ENTITIES = 1000000 
BATCH_SIZE = 10000    # 保持 10k 以防止 gRPC 超限
DATA_DISTRIBUTION = "REALISTIC" 

# 颜色输出
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

# ==============================================================================
# 🏭 数据工厂
# ==============================================================================
def generate_data_arrays(num, dim, mode="UNIFORM"):
    print(f"\n🏭 Generating Data... [Mode: {CYAN}{mode}{RESET}]")
    start_t = time.time()
    
    all_ids = np.arange(num)
    
    if mode == "UNIFORM":
        all_vectors = np.random.random((num, dim)).astype(np.float32)
        all_ages = np.random.randint(0, 100, size=num)
        all_prices = np.round(np.random.uniform(0, 1000, size=num), 2)
        all_tags = np.random.randint(0, 5, size=num)
    elif mode == "REALISTIC":
        print(f"   Using make_blobs (1000 clusters)...")
        all_vectors, _ = make_blobs(n_samples=num, n_features=dim, centers=1000, cluster_std=2.0, random_state=42)
        all_vectors = all_vectors.astype(np.float32)
        
        print("   Using Zipf distribution for Tags...")
        raw_zipf = np.random.zipf(a=1.5, size=num)
        all_tags = (raw_zipf % 5).astype(np.int8)
        
        print("   Using Exponential distribution for Prices...")
        all_prices = np.random.exponential(scale=100, size=num)
        all_prices = np.round(all_prices, 2)
        
        all_ages = np.random.normal(loc=30, scale=10, size=num).astype(np.int64)
        all_ages = np.clip(all_ages, 0, 100)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"   ✅ Data Generation Cost: {time.time()-start_t:.2f}s")
    return all_ids, all_vectors, all_ages, all_prices, all_tags

# ==============================================================================
# 🔌 Milvus 操作 (增强版)
# ==============================================================================
def connect_milvus():
    print(f"🔌 Connecting to Milvus {HOST}...")
    try:
        connections.connect("default", host=HOST, port=PORT, timeout=30)
    except Exception as e:
        print(f"{RED}❌ Connection Failed: {e}{RESET}")
        sys.exit(1)

def setup_collection():
    if utility.has_collection(COLLECTION_NAME):
        print(f"🗑️  Dropping existing collection {COLLECTION_NAME}...")
        utility.drop_collection(COLLECTION_NAME)

    print("🔨 Creating Schema...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="age", dtype=DataType.INT64),
        FieldSchema(name="price", dtype=DataType.FLOAT),
        FieldSchema(name="tag", dtype=DataType.INT8),
    ]
    schema = CollectionSchema(fields, "Hydra Complex Logic Test")
    col = Collection(COLLECTION_NAME, schema, consistency_level="Strong")
    return col

def robust_create_index(col):
    """
    鲁棒的索引构建函数，包含重试和轮询机制
    """
    print("🏗️  Building HNSW Index (Lightweight Mode: M=8, efC=64)...")
    # ⚡️ 降级参数：大幅降低构建时的内存和CPU消耗，防止 Server Overload
    index_params = {
        "metric_type": "L2", 
        "index_type": "HNSW", 
        "params": {"M": 8, "efConstruction": 64} 
    }
    
    try:
        # 尝试同步构建
        col.create_index("vector", index_params, timeout=None)
        print("   ✅ Index built successfully.")
    except Exception as e:
        print(f"   ⚠️ Initial build request timed out or failed: {e}")
        print("   🔄 Switching to Polling Mode (Waiting for background build)...")
        
        # 轮询等待索引完成
        for i in range(60): # 最多等 10 分钟
            time.sleep(10)
            progress = utility.index_building_progress(COLLECTION_NAME)
            print(f"      Index Progress: {progress}", end="\r")
            if progress.get("total_rows", 0) == progress.get("indexed_rows", 0) and progress.get("indexed_rows", 0) > 0:
                print(f"\n   ✅ Index finished in background.")
                return
        print(f"\n   {RED}❌ Index build timeout! proceeding anyway (might be slow){RESET}")

def insert_data(col):
    all_ids, all_vectors, all_ages, all_prices, all_tags = generate_data_arrays(
        NUM_ENTITIES, DIM, mode=DATA_DISTRIBUTION
    )
    
    probe_vector = None
    print(f"📥 Inserting {NUM_ENTITIES} entities in batches...")
    total_start = time.time()
    
    for i in range(0, NUM_ENTITIES, BATCH_SIZE):
        end = min(i + BATCH_SIZE, NUM_ENTITIES)
        col.insert([
            all_ids[i:end], all_vectors[i:end], all_ages[i:end], 
            all_prices[i:end], all_tags[i:end]
        ])
        if probe_vector is None: probe_vector = all_vectors[0]
        if i % 200000 == 0: print(f"   🚀 Progress: {i}/{NUM_ENTITIES}...")

    print(f"💾 Flushing... (Total Insert Time: {time.time()-total_start:.2f}s)")
    col.flush()
    
    # 调用鲁棒索引构建
    robust_create_index(col)
    
    print("📦 Loading collection...")
    col.load()
    return probe_vector

# ==============================================================================
# 🧪 测试逻辑 (Hydra)
# ==============================================================================
def run_query(col, query_vec, expr, name):
    # Search ef 也稍微降低，减轻负载
    search_params = {"metric_type": "L2", "params": {"ef": 128}}
    start = time.time()
    try:
        res = col.search(
            data=[query_vec], anns_field="vector", param=search_params, 
            limit=100, expr=expr, output_fields=["id"]
        )
        cost = (time.time() - start) * 1000
        ids = set([hit.id for hit in res[0]])
        print(f"   👉 [{name}] Cost: {cost:.2f}ms | Hits: {len(ids)}")
        return ids, cost
    except Exception as e:
        print(f"   ❌ [{name}] CRASHED: {str(e)[:200]}...")
        return set(), 99999.0 

def compare_results(results_map):
    keys = list(results_map.keys())
    base_key = keys[0]
    base_ids, base_cost = results_map[base_key]
    
    print(f"\n   📊 Analysis (Baseline: {base_key}):")
    for key in keys[1:]:
        curr_ids, curr_cost = results_map[key]
        
        if len(base_ids) == 0 and len(curr_ids) == 0: jaccard = 1.0
        elif len(base_ids) == 0 or len(curr_ids) == 0: jaccard = 0.0
        else: jaccard = len(base_ids.intersection(curr_ids)) / len(base_ids.union(curr_ids))
            
        ratio = curr_cost / (base_cost + 0.01)
        status_icon = "✅"
        if jaccard < 0.95: status_icon = f"{RED}❌ LOGIC BUG{RESET}"
        elif ratio > 5.0: status_icon = f"{YELLOW}⚠️ PERF WARN{RESET}"
        
        print(f"      vs {key}: Overlap={jaccard:.4f} | TimeRatio={ratio:.2f}x | {status_icon}")

def run_hydra_test(col, query_vec):
    print(f"\n{'='*20} STARTING HYDRA TEST (Mode: {DATA_DISTRIBUTION}) {'='*20}")

    print(f"\n{CYAN}🧪 Scenario 1: Range Intersection{RESET}")
    compare_results({
        "Standard": run_query(col, query_vec, "age >= 20 && age <= 30 && price > 100", "Standard"),
        "Split_OR": run_query(col, query_vec, "((age >= 20 && age < 25) || (age >= 25 && age <= 30)) && price > 100", "Split_OR")
    })

    print(f"\n{CYAN}🧪 Scenario 2: Categorical Set (Skewed Tags){RESET}")
    compare_results({
        "IN_List": run_query(col, query_vec, "tag in [1, 4]", "IN_List"),
        "OR_Chain": run_query(col, query_vec, "tag == 1 || tag == 4", "OR_Chain")
    })

    print(f"\n{CYAN}🧪 Scenario 3: Complex Branching{RESET}")
    compare_results({
        "Branch": run_query(col, query_vec, "(tag == 1 && price < 50) || (tag == 4 && price > 500)", "Branch"),
        "Nested": run_query(col, query_vec, "(tag == 1 && (price < 25 || price < 50)) || (tag == 4 && price > 500)", "Nested")
    })

    print(f"\n{CYAN}🧪 Scenario 4: Massive OR Chain (1000 Conditions){RESET}")
    target_ids = list(range(1000))
    expr_in = f"id in {target_ids}"
    expr_or = " || ".join([f"id == {i}" for i in target_ids])
    
    results = {}
    try:
        results["Massive_IN"] = run_query(col, query_vec, expr_in, "Massive_IN")
        results["Massive_OR"] = run_query(col, query_vec, expr_or, "Massive_OR")
        compare_results(results)
    except Exception as e:
        print(f"{RED}🚨 Massive Test Aborted!{RESET}")

if __name__ == "__main__":
    connect_milvus()
    col = setup_collection()
    query_vec = insert_data(col)
    run_hydra_test(col, query_vec)