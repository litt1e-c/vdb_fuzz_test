import numpy as np
import time
from db_adapters import MilvusAdapter, QdrantAdapter, ChromaAdapter, WeaviateAdapter

# --- 配置 ---
DIM = 768
NUM_ENTITIES = 500000 # 50万数据
BATCH_SIZE = 10000

# 颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def setup_db(db, name):
    print(f"\n🏗️  Initializing {name}...")
    db.connect()
    db.recreate_collection("hybrid_rank_test")
    return db

def insert_data_all(dbs):
    print(f"\n📥 Generating & Inserting {NUM_ENTITIES} vectors...")
    all_ids = np.arange(NUM_ENTITIES)
    # Age 均匀分布在 0-100
    ages = np.random.randint(0, 101, size=NUM_ENTITIES).tolist() 
    # 占位
    prices = np.zeros(NUM_ENTITIES).tolist()
    tags = np.zeros(NUM_ENTITIES).tolist()
    
    vectors = np.random.random((NUM_ENTITIES, DIM)).astype(np.float32)
    probe_vec = vectors[0] # 取第一个作为查询向量

    # 依次插入每个库
    for db_name, db in dbs.items():
        print(f"   👉 Inserting into {db_name}...")
        start_t = time.time()
        for i in range(0, NUM_ENTITIES, BATCH_SIZE):
            end = min(i + BATCH_SIZE, NUM_ENTITIES)
            db.insert_batch(
                all_ids[i:end], 
                vectors[i:end], 
                {'age': ages[i:end], 'price': prices[i:end], 'tag': tags[i:end]}
            )
        print(f"      Done. Cost: {time.time()-start_t:.2f}s")
        print(f"      Building Index for {db_name}...")
        db.flush_and_index()
        
    return probe_vec

def compare_rankings(results):
    """
    对比 Milvus (基准) 与其他库的结果重合度
    results = {'Milvus': [(id, dist)...], 'Qdrant': ...}
    """
    base_name = 'Milvus'
    if base_name not in results: return
    
    base_ids = set([x[0] for x in results[base_name]])
    print(f"   📊 Baseline ({base_name}) found {len(base_ids)} items.")
    
    for name, res in results.items():
        if name == base_name: continue
        
        curr_ids = set([x[0] for x in res])
        
        # 计算 Jaccard Overlap
        if len(base_ids) == 0 and len(curr_ids) == 0:
            overlap = 1.0
        elif len(base_ids) == 0 or len(curr_ids) == 0:
            overlap = 0.0
        else:
            overlap = len(base_ids.intersection(curr_ids)) / len(base_ids.union(curr_ids))
            
        status = "✅" if overlap > 0.8 else "❌ LOW RECALL"
        if overlap < 0.1: status = "🚨 MISMATCH"
        
        print(f"      vs {name}: Overlap={overlap:.4f} | Hits={len(curr_ids)} | {status}")
        
        # 如果不一致，打印前3个ID看看
        if overlap < 0.9:
            print(f"         Milvus Top-3: {[x[0] for x in results[base_name][:3]]}")
            print(f"         {name} Top-3:   {[x[0] for x in res[:3]]}")

def run_hybrid_test():
    # 1. 初始化所有数据库
    dbs = {
        "Milvus": MilvusAdapter(DIM),
        "Qdrant": QdrantAdapter(DIM),
        # "Chroma": ChromaAdapter(DIM), # 50万数据 Chroma 可能会太慢，建议先注释掉，或者最后跑
        # "Weaviate": WeaviateAdapter(DIM)
    }
    
    for name, db in dbs.items():
        setup_db(db, name)
        
    # 2. 插入数据
    query_vec = insert_data_all(dbs)
    
    # 3. 混合检索测试 (梯度压力)
    scenarios = [
        ("Medium Filtering (Age > 50)", 50),
        ("Heavy Filtering (Age > 90)", 90),
        ("Extreme Filtering (Age > 99)", 99) # 最容易出问题
    ]
    
    print(f"\n{'='*20} STARTING HYBRID SEARCH TEST {'='*20}")
    
    for desc, age_val in scenarios:
        print(f"\n🧪 Test Scenario: {CYAN}{desc}{RESET}")
        
        scenario_results = {}
        
        for name, db in dbs.items():
            try:
                start = time.time()
                # 查 Top 20
                res = db.search(query_vec, 20, expr_type="age_gt", filter_params={"val": age_val})
                cost = (time.time() - start) * 1000
                scenario_results[name] = res
                print(f"   👉 [{name}] Cost: {cost:.2f}ms | Returned: {len(res)}")
            except Exception as e:
                print(f"   ❌ [{name}] Failed: {e}")
                scenario_results[name] = []
        
        compare_rankings(scenario_results)

if __name__ == "__main__":
    run_hybrid_test()