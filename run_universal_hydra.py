import numpy as np
import time
from db_adapters import MilvusAdapter, QdrantAdapter, ChromaAdapter, WeaviateAdapter

# --- 配置 ---
DIM = 768
NUM_ENTITIES = 500000 # 跨库测试，先用 1万条验证逻辑，别太慢
BATCH_SIZE = 2000

# 颜色
GREEN = '\033[92m'
RESET = '\033[0m'

def run_test_on_db(db: object, db_name: str, custom_batch_size=None):
    print(f"\n🚀 Testing Database: {GREEN}{db_name}{RESET}")
    
    # 1. 初始化
    db.connect()
    db.recreate_collection("universal_hydra")
    
    # 2. 数据生成与插入
    print("   📥 Generating & Inserting data...")
    all_ids = np.arange(NUM_ENTITIES)
    ages = np.random.randint(0, 100, size=NUM_ENTITIES).tolist()
    prices = np.round(np.random.uniform(0, 1000, size=NUM_ENTITIES), 2).tolist()
    tags = np.random.randint(0, 5, size=NUM_ENTITIES).tolist()
    payloads = {'age': ages, 'price': prices, 'tag': tags}
    
    vectors = np.random.random((NUM_ENTITIES, DIM)).astype(np.float32)
    probe_vec = vectors[0]
    
    # 使用自定义 Batch Size
    current_batch = custom_batch_size if custom_batch_size else BATCH_SIZE
    print(f"   ℹ️  Using Batch Size: {current_batch}")

    start_t = time.time()
    for i in range(0, NUM_ENTITIES, current_batch):
        end = min(i + current_batch, NUM_ENTITIES)
        db.insert_batch(all_ids[i:end], vectors[i:end], 
                       {'age': ages[i:end], 'price': prices[i:end], 'tag': tags[i:end]})
    print(f"   ⏱️ Insert Time: {time.time()-start_t:.2f}s")
    
    print("   🏗️ Indexing...")
    db.flush_and_index()
    
    # 3. Hydra 逻辑验证
    print("   🧪 Running Logic Tests...")
    
    scenarios = [
        ("range_standard", "range_split"),  # 范围查询 vs 拆分
        ("cat_in", "cat_or")                # IN vs OR 链
    ]
    
    for case_a, case_b in scenarios:
        start = time.time()
        res_a = db.search(probe_vec, 100, case_a)
        time_a = (time.time() - start) * 1000
        
        start = time.time()
        res_b = db.search(probe_vec, 100, case_b)
        time_b = (time.time() - start) * 1000
        
        # 验证
        if len(res_a) == 0: overlap = 1.0
        else: overlap = len(res_a.intersection(res_b)) / len(res_a.union(res_b))
        
        print(f"      Compare [{case_a}] vs [{case_b}]:")
        print(f"         Time: {time_a:.1f}ms vs {time_b:.1f}ms")
        print(f"         Overlap: {overlap:.4f} {'✅' if overlap > 0.99 else '❌'}")

# --- 主程序 ---
if __name__ == "__main__":
    # 1. Milvus (可以使用大 Batch: 10000)
    try: 
        run_test_on_db(MilvusAdapter(DIM), "Milvus (Baseline)", custom_batch_size=10000)
    except Exception as e: print(f"Milvus Error: {e}")

    # 2. Qdrant (🔥 关键修复：使用小 Batch: 2000)
    try: 
        # 2000条 * 768 * 4 = 6MB，HTTP 传输非常轻松，绝对不会超时
        run_test_on_db(QdrantAdapter(DIM), "Qdrant", custom_batch_size=2000)
    except Exception as e: print(f"Qdrant Error: {e}")

    # 3. Chroma (Challenger 2)
    """
    try:
        chroma = ChromaAdapter(DIM)
        run_test_on_db(chroma, "Chroma")
    except Exception as e: print(f"Chroma Error: {e}")
    """

    # 4. Weaviate (也可以用小一点，稳一点)
    try: 
        run_test_on_db(WeaviateAdapter(DIM), "Weaviate", custom_batch_size=5000)
    except Exception as e: print(f"Weaviate Error: {e}")