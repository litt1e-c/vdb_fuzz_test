import time
import numpy as np
from db_adapters_advanced import MilvusAdapter, QdrantAdapter, WeaviateAdapter

# --- 🔥 配置 ---
HOST = '127.0.0.1'
DIM = 768
NUM_ENTITIES = 50000
BATCH_SIZE = 5000 # 5000 * 768 * 4 ≈ 15MB (Safe for gRPC)

# 颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

# --- 辅助函数：计算召回率 ---
def calculate_recall(result_ids, ground_truth_ids):
    if len(ground_truth_ids) == 0: return 0.0
    res_set = set(result_ids)
    gt_set = set(ground_truth_ids)
    hit = len(res_set.intersection(gt_set))
    return hit / len(gt_set)

def prepare_ground_truth(gt_adapter, query_vec, limit, offset):
    """
    使用 Milvus FLAT 索引计算基准真理 (Ground Truth)
    """
    res = gt_adapter.col.search(
        data=[query_vec], anns_field="vector", 
        param={"metric_type": "L2", "params": {}}, 
        limit=limit + offset, 
        output_fields=["id"]
    )
    
    full_list = [hit.id for hit in res[0]]
    
    if len(full_list) < offset + limit:
        return [] 
    
    return full_list[offset : offset+limit]

def run_experiment_a_ef_sensitivity(db, db_name, query_vec, gt_adapter):
    print(f"\n{CYAN}🧪 [Exp A] '勤能补拙' Test: efSearch vs. Recall (Fixed M=8, efC=64){RESET}")
    print(f"   Target: Top-100 Recall")
    
    # 获取真理
    gt_ids = prepare_ground_truth(gt_adapter, query_vec, limit=100, offset=0)
    
    # 测试不同的 ef
    ef_values = [16, 32, 64, 128, 256, 512]
    
    for ef in ef_values:
        try:
            start_t = time.time()
            
            # 特殊处理：Qdrant/Milvus 支持 search_params，Weaviate 不支持
            if "Weaviate" in db_name:
                # Weaviate 不支持动态 ef，只能跑默认，跳过循环
                if ef != 16: continue 
                res = db.search(query_vec, limit=100)
            else:
                res = db.search(query_vec, limit=100, search_params={"ef": ef})
                
            cost = (time.time() - start_t) * 1000
            
            # db.search 返回 [(id, dist, price)...]
            res_ids = [x[0] for x in res]
            recall = calculate_recall(res_ids, gt_ids)
            
            mark = ""
            if recall > 0.99: mark = "🌟"
            elif recall < 0.5: mark = "⚠️ Poor"
            
            print(f"   👉 ef={ef:<4} | Cost={cost:.2f}ms | Recall={recall:.2%} {mark}")
            
        except Exception as e:
            print(f"   ❌ ef={ef:<4} | Failed: {str(e)[:100]}...")

def run_experiment_b_offset_trap(db, db_name, query_vec, gt_adapter):
    print(f"\n{CYAN}🧪 [Exp B] '深分页陷阱' Test: Offset vs. ef (Fixed ef=100){RESET}")
    
    FIXED_EF = 100
    LIMIT = 10
    offsets = [0, 50, 90, 100, 110, 200]
    
    for offset in offsets:
        gt_ids = prepare_ground_truth(gt_adapter, query_vec, limit=LIMIT, offset=offset)
        if not gt_ids: continue
        
        try:
            search_p = {"ef": FIXED_EF}
            req_limit = offset + LIMIT
            
            # 模拟分页：请求 Top-(Offset+Limit)
            res_full = db.search(query_vec, limit=req_limit, search_params=search_p)
            
            if len(res_full) < req_limit:
                # 某些 DB 可能返回不足
                print(f"   ❌ Offset={offset:<3} | Not enough results ({len(res_full)} < {req_limit})")
                continue
                
            page_ids = [x[0] for x in res_full[offset:]]
            recall = calculate_recall(page_ids, gt_ids)
            
            status = "✅"
            # 核心判定：如果 offset+limit > ef，Recall 是否暴跌？
            if offset + LIMIT > FIXED_EF:
                if recall > 0.9: status = f"{GREEN}✅ Auto-Adjusted{RESET}"
                elif recall < 0.5: status = f"{RED}❌ Recall Drop{RESET}"
                else: status = f"{YELLOW}⚠️ Degraded{RESET}"
            
            print(f"   👉 Offset={offset:<4} (Req {req_limit}) | ef={FIXED_EF} | Recall={recall:.2%} | {status}")

        except Exception as e:
             print(f"   ⚠️ Offset={offset:<4} | Error: {str(e)[:100]}")

def main():
    # 1. 准备基准数据库 (Milvus FLAT)
    print(f"{YELLOW}🔨 Initializing Ground Truth (Milvus FLAT)...{RESET}")
    gt_adapter = MilvusAdapter(DIM)
    gt_adapter.connect()
    gt_adapter.recreate_collection("ground_truth_flat")
    
    all_ids = np.arange(NUM_ENTITIES)
    vectors = np.random.random((NUM_ENTITIES, DIM)).astype(np.float32)
    # 构造全 0 的占位 payload (int64)
    zeros_payload = np.zeros(NUM_ENTITIES, dtype=np.int64)

    # 🔥 关键修复：对 Ground Truth 插入也进行分批，防止 gRPC 超限
    print(f"   📥 Inserting Ground Truth Data (Batch Size: {BATCH_SIZE})...")
    for i in range(0, NUM_ENTITIES, BATCH_SIZE):
        end = min(i + BATCH_SIZE, NUM_ENTITIES)
        gt_adapter.col.insert([
            all_ids[i:end], 
            vectors[i:end], 
            zeros_payload[i:end]
        ])
    
    gt_adapter.col.flush()
    gt_adapter.col.create_index("vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    gt_adapter.col.load()
    
    probe_vec = vectors[0]

    # 2. 准备被测数据库 (Low Quality HNSW)
    adapters = [
        MilvusAdapter(DIM),
        QdrantAdapter(DIM),
        # WeaviateAdapter(DIM) 
    ]

    for db in adapters:
        name = db.__class__.__name__
        print(f"\n{'='*60}")
        print(f"🚀 TESTING: {name} (Low Quality Index)")
        print(f"{'='*60}")
        
        db.connect()
        db.recreate_collection(f"exp_hnsw_{name}")
        
        print("   📥 Inserting data...")
        for i in range(0, NUM_ENTITIES, BATCH_SIZE):
             end = min(i+BATCH_SIZE, NUM_ENTITIES)
             db.insert_batch(all_ids[i:end], vectors[i:end], {'price': zeros_payload[i:end]})
        
        print("   🏗️  Building Low-Quality HNSW Index (M=8, efC=64)...")
        
        # --- 强制降级索引参数 ---
        if isinstance(db, MilvusAdapter):
            db.col.flush()
            db.col.create_index("vector", {
                "metric_type": "L2", "index_type": "HNSW", 
                "params": {"M": 8, "efConstruction": 64} 
            })
            db.col.load()
        elif isinstance(db, QdrantAdapter):
            # Qdrant 已经在 adapter 初始化时设为低配，这里只需 flush
            db.flush_and_index()
        
        # 3. 运行实验
        run_experiment_a_ef_sensitivity(db, name, probe_vec, gt_adapter)
        run_experiment_b_offset_trap(db, name, probe_vec, gt_adapter)

if __name__ == "__main__":
    main()