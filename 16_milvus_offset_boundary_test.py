import time
import numpy as np
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, 
    Collection, utility, MilvusException
)

# --- 🔥 配置 ---
HOST = '127.0.0.1'
PORT = '19530'
DIM = 128
NUM_ENTITIES = 20000 # 2万数据足够测逻辑
TEST_EF = 64         # 🔥 锁死 ef = 64
LIMIT = 10           # 每次查 10 条

# 颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def connect_milvus():
    print(f"🔌 Connecting to Milvus {HOST}...")
    connections.connect("default", host=HOST, port=PORT)

def create_and_insert(name, index_type):
    if utility.has_collection(name):
        utility.drop_collection(name)
        
    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM)
    ]
    schema = CollectionSchema(fields)
    col = Collection(name, schema, consistency_level="Strong")
    
    print(f"   📥 [{name}] Inserting {NUM_ENTITIES} vectors...")
    # 使用 NumPy 生成
    ids = list(range(NUM_ENTITIES))
    vectors = np.random.random((NUM_ENTITIES, DIM)).astype(np.float32)
    col.insert([ids, vectors])
    col.flush()
    
    # 建索引
    if index_type == "FLAT":
        print(f"   🏗️  [{name}] Building FLAT Index (Ground Truth)...")
        col.create_index("vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    else:
        print(f"   🏗️  [{name}] Building HNSW Index (M=16, efC=200)...")
        col.create_index("vector", {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}})
    
    col.load()
    return col, vectors[0] # 返回探针向量

def run_boundary_test():
    connect_milvus()
    
    # 1. 准备环境
    col_flat, query_vec = create_and_insert("bench_offset_flat", "FLAT")
    col_hnsw, _ = create_and_insert("bench_offset_hnsw", "HNSW")
    
    print(f"\n{'='*60}")
    print(f"🔥 STARTING OFFSET BOUNDARY TEST (Fixed ef={TEST_EF})")
    print(f"{'='*60}")
    
    # 测试梯度：Offset 逐渐逼近并超过 ef
    # limit = 10
    # ef = 64
    # Threshold = ef - limit = 54
    offsets = [0, 20, 50, 54, 55, 64, 100, 200]
    
    for offset in offsets:
        req_k = offset + LIMIT
        print(f"\n🧪 Testing Offset={offset} (Total Top-K needed: {req_k})...")
        
        # 1. 获取真理 (FLAT)
        res_gt = col_flat.search(
            [query_vec], "vector", {"metric_type": "L2"}, 
            limit=LIMIT, offset=offset, output_fields=["id"]
        )
        ids_gt = [hit.id for hit in res_gt[0]]
        
        # 2. 测试 HNSW (锁定 ef)
        search_params = {"metric_type": "L2", "params": {"ef": TEST_EF}}
        
        try:
            start_t = time.time()
            res_test = col_hnsw.search(
                [query_vec], "vector", search_params, 
                limit=LIMIT, offset=offset, output_fields=["id"]
            )
            cost = (time.time() - start_t) * 1000
            ids_test = [hit.id for hit in res_test[0]]
            
            # 计算重合度
            hit_cnt = len(set(ids_test).intersection(set(ids_gt)))
            recall = hit_cnt / len(ids_gt) if len(ids_gt) > 0 else 0.0
            
            # 状态判定
            status = ""
            # 场景 A: 安全区 (req_k <= ef)
            if req_k <= TEST_EF:
                if recall > 0.99: status = f"{GREEN}✅ PASS (Safe Zone){RESET}"
                else: status = f"{YELLOW}⚠️ Low Recall in Safe Zone{RESET}"
            # 场景 B: 危险区 (req_k > ef)
            else:
                if recall > 0.99: status = f"{CYAN}ℹ️  Auto-Adjusted (Milvus handled it){RESET}"
                elif recall < 0.5: status = f"{RED}❌ Recall Collapse (Blind Search){RESET}"
                else: status = f"{YELLOW}⚠️ Degraded (Effort limit reached){RESET}"
            
            print(f"   👉 Result: Cost={cost:.2f}ms | Recall={recall:.0%} | {status}")
            
        except MilvusException as e:
            # 场景 C: 报错 (Strict Check)
            err_msg = str(e)
            if "ef" in err_msg and "k" in err_msg:
                print(f"   🛡️  {GREEN}Milvus Protection Triggered:{RESET} {err_msg}")
            else:
                print(f"   ❌ Unexpected Error: {err_msg}")
        except Exception as e:
            print(f"   ❌ CRASH: {e}")

if __name__ == "__main__":
    run_boundary_test()