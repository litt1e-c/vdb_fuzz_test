import time
import random
import numpy as np
from db_adapters_advanced import MilvusAdapter, QdrantAdapter, WeaviateAdapter, ChromaAdapter

# --- ⚙️ 配置 ---
HOST = '127.0.0.1'
DIM = 768
NUM_ENTITIES = 50000 
BATCH_SIZE = 5000

# 颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def prepare_data(db, name):
    print(f"\n🏗️  [Setup] Preparing {name} with {NUM_ENTITIES} vectors...")
    db.connect()
    db.recreate_collection(f"adv_test_{name.lower()}")
    
    all_ids = np.arange(NUM_ENTITIES)
    vectors = np.random.random((NUM_ENTITIES, DIM)).astype(np.float32)
    prices = np.random.randint(0, 200, size=NUM_ENTITIES)
    
    start_t = time.time()
    for i in range(0, NUM_ENTITIES, BATCH_SIZE):
        end = min(i+BATCH_SIZE, NUM_ENTITIES)
        db.insert_batch(all_ids[i:end], vectors[i:end], {'price': prices[i:end]})
    
    print(f"   💾 Flushing & Indexing (Cost: {time.time()-start_t:.2f}s)...")
    db.flush_and_index()
    return vectors[0] # 返回第一个作为查询向量

def test_a1_monotonicity(db, name, query_vec):
    print(f"\n{CYAN}🧪 [Test A1] Parameter Monotonicity (ef Sensitivity){RESET}")
    
    if name in ["ChromaAdapter", "WeaviateAdapter"]:
        print(f"   ⚠️ {name} does not support query-time 'ef' tuning. Skipping.")
        return

    TEST_LIMIT = 20 
    
    # 1. 低配搜索
    res_low = db.search(query_vec, limit=TEST_LIMIT, search_params={"ef": 32})
    ids_low = set([x[0] for x in res_low])
    
    # 2. 高配搜索
    res_high = db.search(query_vec, limit=TEST_LIMIT, search_params={"ef": 256})
    ids_high = set([x[0] for x in res_high])
    
    if len(ids_low) == 0:
        print(f"   {YELLOW}⚠️ Low params returned 0 results.{RESET}")
        return

    overlap = len(ids_low.intersection(ids_high)) / len(ids_low)
    print(f"   ef=32 found {len(res_low)}, ef=256 found {len(res_high)}")
    print(f"   Overlap Ratio: {overlap:.2%}")
    
    if overlap < 0.8: 
        print(f"   {YELLOW}⚠️ Stability Warning: Results changed significantly with 'ef' boost.{RESET}")
    else:
        print(f"   {GREEN}✅ Pass: Search results are stable.{RESET}")

def test_a2_filter_monotonicity(db, name, query_vec):
    print(f"\n{CYAN}🧪 [Test A2] Filter Monotonicity (Subset Constraint){RESET}")
    
    res_loose = db.search(query_vec, limit=500, expr="price > 50")
    res_strict = db.search(query_vec, limit=500, expr="price > 50 && price < 100")
    
    if len(res_loose) == 0: return
    
    loose_ids = set([x[0] for x in res_loose])
    worst_loose_dist = res_loose[-1][1]
    
    violation_count = 0
    for item in res_strict:
        uid, dist, price = item
        # 检查逻辑：严格过滤的结果，要么在宽松结果里，要么距离比宽松结果的门槛远(被截断)。
        # 如果距离很近(dist < worst)，却没在宽松结果里，说明宽松搜索漏掉了它。
        if uid not in loose_ids:
            if dist < worst_loose_dist and abs(dist - worst_loose_dist) > 1e-5:
                 print(f"      {RED}❌ Violation! ID {uid} (dist={dist:.4f}) found in Strict but missed in Loose (cutoff={worst_loose_dist:.4f}){RESET}")
                 violation_count += 1
    
    if violation_count == 0:
        print(f"   {GREEN}✅ Pass: Strict results respect Loose boundaries.{RESET}")
    else:
        print(f"   {RED}❌ Fail: Filter caused Index Blindness (Recall Inconsistency).{RESET}")

def test_b1_linearizability(db, name, query_vec):
    print(f"\n{CYAN}🧪 [Test B1] Linearizability (Read-after-Write){RESET}")
    
    new_id = 99999999
    db.insert_batch([new_id], np.array([query_vec]), {'price': [150], 'age': [1], 'tag': [1]})
    
    found = db.query_by_id(new_id)
    if found:
        print(f"   {GREEN}✅ Pass: Inserted data is immediately visible.{RESET}")
    else:
        print(f"   {RED}❌ Fail: Stale Read! Inserted data not found immediately.{RESET}")

def test_b2_zombie_data(db, name):
    print(f"\n{CYAN}🧪 [Test B2] Zombie Data Check (Delete Persistence){RESET}")
    
    target_id = 99999999 
    db.delete([target_id])
    
    # 对于最终一致性数据库，稍作等待
    time.sleep(1) 
    
    found = db.query_by_id(target_id)
    
    if not found:
        print(f"   {GREEN}✅ Pass: Deleted data is gone forever.{RESET}")
    else:
        print(f"   {RED}❌ Fail: Zombie Data Detected! ID {target_id} resurrected.{RESET}")

def run_all_tests():
    adapters = [
        MilvusAdapter(DIM),
        QdrantAdapter(DIM),
        WeaviateAdapter(DIM),
        # ChromaAdapter(DIM) # Chroma 性能较弱，可视情况开启
    ]
    
    for db in adapters:
        print(f"\n{'#'*60}")
        print(f"🚀 STARTING TESTS FOR: {db.__class__.__name__}")
        print(f"{'#'*60}")
        
        try:
            query_vec = prepare_data(db, db.__class__.__name__)
            
            test_a1_monotonicity(db, db.__class__.__name__, query_vec)
            test_a2_filter_monotonicity(db, db.__class__.__name__, query_vec)
            test_b1_linearizability(db, db.__class__.__name__, query_vec)
            test_b2_zombie_data(db, db.__class__.__name__)
            
        except Exception as e:
            print(f"   {RED}❌ CRITICAL FAILURE: {e}{RESET}")

if __name__ == "__main__":
    run_all_tests()