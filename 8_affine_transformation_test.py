import numpy as np
import time
import random
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, 
    Collection, utility
)

# --- 🔥 配置参数 ---
HOST = '127.0.0.1'
PORT = '19530'
DIM = 768
NUM_ENTITIES = 50000 
BATCH_SIZE = 10000 

# 颜色输出
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def connect_milvus():
    print(f"🔌 Connecting to Milvus {HOST}...")
    connections.connect("default", host=HOST, port=PORT)

def create_collection(name):
    if utility.has_collection(name):
        utility.drop_collection(name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, f"Affine Test: {name}")
    return Collection(name, schema, consistency_level="Strong")

def insert_and_index(col, vectors, ids, desc):
    print(f"   📥 [{desc}] Inserting {len(vectors)} vectors (Batched)...")
    
    total = len(vectors)
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        batch_ids = ids[i:end]
        batch_vectors = vectors[i:end]
        col.insert([batch_ids, batch_vectors])
    
    print(f"   💾 [{desc}] Flushing...")
    col.flush()
    
    # --- 🔥 这里改成了 HNSW 🔥 ---
    print(f"   🏗️  [{desc}] Building HNSW Index...")
    index_params = {
        "metric_type": "L2", 
        "index_type": "HNSW",  # 关键修改
        "params": {"M": 16, "efConstruction": 200} # 关键修改
    }
    col.create_index("vector", index_params, timeout=None)
    col.load()

def analyze_results(res_base, res_target, expected_dist_ratio, name):
    print(f"\n📊 Analysis: Baseline vs {name}")
    
    ids_base = [hit.id for hit in res_base[0]]
    ids_target = [hit.id for hit in res_target[0]]
    
    dists_base = [hit.distance for hit in res_base[0]]
    dists_target = [hit.distance for hit in res_target[0]]
    
    # 1. 排序一致性 (Rank Consistency)
    if ids_base == ids_target:
        print(f"      {GREEN}✅ Rank Consistency: PERFECT MATCH.{RESET}")
    else:
        set_base = set(ids_base)
        set_trans = set(ids_target)
        overlap = len(set_base.intersection(set_trans)) / len(set_base.union(set_trans))
        
        # 对于 HNSW，我们允许一点点抖动 (Jitter)，只要重合度高就行
        if overlap >= 0.9:
            print(f"      {YELLOW}⚠️ Rank Jitter (Expected for HNSW): Overlap={overlap:.2%}{RESET}")
            # 打印前3个看看是不是只是顺序换了
            print(f"         Base Top-3:   {ids_base[:3]}")
            print(f"         Target Top-3: {ids_target[:3]}")
        else:
            print(f"      {RED}❌ FAILED: Major Rank Divergence! Overlap={overlap:.4f}{RESET}")

    # 2. 距离缩放精度
    if len(dists_base) > 0 and len(dists_target) > 0:
        d_base = dists_base[0]
        d_target = dists_target[0]
        
        expected_ratio = expected_dist_ratio
        actual_ratio = d_target / (d_base + 1e-9) 
        
        error_rate = abs(actual_ratio - expected_ratio) / (expected_ratio + 1e-9)
        
        print(f"   📏 Distance Check (Top-1):")
        print(f"      Base Dist:   {d_base:.6f}")
        print(f"      Target Dist: {d_target:.6f}")
        
        if error_rate < 0.01:
             print(f"      {GREEN}✅ Metric Scaling Correct (Error < 1%){RESET}")
        elif error_rate < 0.1:
             print(f"      {YELLOW}⚠️ Metric Precision Loss (Error < 10%){RESET}")
        else:
             # HNSW 如果因为近似导致找错了最近邻（比如找到了第2近的），距离比就会对不上
             print(f"      {RED}❌ Metric Mismatch (Error={error_rate:.2%}) - Likely Found Different Neighbor{RESET}")

def run_affine_test():
    connect_milvus()
    
    print(f"\n{CYAN}🧪 Generating Affine Transformation Datasets...{RESET}")
    ids = np.arange(NUM_ENTITIES)
    
    # 使用标准正态分布
    vec_base = np.random.randn(NUM_ENTITIES, DIM).astype(np.float32)
    
    SCALE_LARGE = 100.0
    vectors_large = vec_base * SCALE_LARGE
    
    SCALE_MICRO = 0.01
    vectors_micro = vec_base * SCALE_MICRO
    
    SHIFT_VAL = 1000.0
    vectors_shift = vec_base + SHIFT_VAL
    
    # 建库
    col_base = create_collection("affine_base")
    insert_and_index(col_base, vec_base, ids, "Baseline")
    
    col_large = create_collection("affine_large")
    insert_and_index(col_large, vectors_large, ids, "Macro (x100)")
    
    col_micro = create_collection("affine_micro")
    insert_and_index(col_micro, vectors_micro, ids, "Micro (x0.01)")
    
    col_shift = create_collection("affine_shift")
    insert_and_index(col_shift, vectors_shift, ids, "Shift (+1000)")
    
    # 查询
    print(f"\n{CYAN}🔍 Executing Comparative Search (HNSW Mode)...{RESET}")
    
    # 生成独立的 Query 向量
    q_vec_new = np.random.randn(1, DIM).astype(np.float32)
    q_base = q_vec_new
    
    q_large = q_base * SCALE_LARGE
    q_micro = q_base * SCALE_MICRO
    q_shift = q_base + SHIFT_VAL
    
    # --- 🔥 这里加上了 ef 参数 ---
    search_params = {"metric_type": "L2", "params": {"ef": 2000}}
    LIMIT = 50 
    
    res_base = col_base.search(q_base, "vector", search_params, limit=LIMIT)
    res_large = col_large.search(q_large, "vector", search_params, limit=LIMIT)
    res_micro = col_micro.search(q_micro, "vector", search_params, limit=LIMIT)
    res_shift = col_shift.search(q_shift, "vector", search_params, limit=LIMIT)
    
    print(f"\n{'='*20} TEST RESULTS {'='*20}")
    
    analyze_results(res_base, res_large, SCALE_LARGE**2, "Macro World (x100)")
    analyze_results(res_base, res_micro, SCALE_MICRO**2, "Micro World (x0.01)")
    analyze_results(res_base, res_shift, 1.0, "Shifted World (+1000)")

if __name__ == "__main__":
    run_affine_test()