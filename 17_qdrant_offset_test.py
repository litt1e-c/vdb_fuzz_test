import time
import numpy as np
from qdrant_client import QdrantClient, models

# --- 🔥 配置 ---
HOST = '127.0.0.1'
PORT = 6333
COLLECTION_NAME = "qdrant_offset_test"
DIM = 768
NUM_ENTITIES = 20000 # 2万数据足够
BATCH_SIZE = 2000

# 颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def calculate_recall(result_ids, gt_ids):
    if not gt_ids: return 0.0
    hit = len(set(result_ids).intersection(set(gt_ids)))
    return hit / len(gt_ids)

def run_qdrant_test():
    print(f"{CYAN}🚀 Initializing Qdrant Deep Dive Test...{RESET}")
    
    # 1. 连接 (优先尝试 gRPC, 失败回退 HTTP)
    try:
        client = QdrantClient(host=HOST, port=PORT, grpc_port=6334, prefer_grpc=True, timeout=None)
        print("   🔌 Connected via gRPC")
    except:
        client = QdrantClient(host=HOST, port=PORT, timeout=None)
        print("   🔌 Connected via HTTP")

    # 2. 重建集合 (HNSW 参数 M=16, ef_construct=100)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=DIM, 
            distance=models.Distance.EUCLID,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
        )
    )

    # 3. 插入数据
    print(f"   📥 Inserting {NUM_ENTITIES} vectors...")
    all_ids = list(range(NUM_ENTITIES))
    vectors = np.random.random((NUM_ENTITIES, DIM)).astype(np.float32)
    
    # 构造 PointStruct
    points = [
        models.PointStruct(id=i, vector=v.tolist()) 
        for i, v in zip(all_ids, vectors)
    ]
    
    # 批量写入
    start_t = time.time()
    for i in range(0, NUM_ENTITIES, BATCH_SIZE):
        end = min(i+BATCH_SIZE, NUM_ENTITIES)
        client.upsert(COLLECTION_NAME, points[i:end], wait=False)
    
    # 强制等待索引构建
    print("   🏗️  Waiting for Indexing...", end="", flush=True)
    while True:
        info = client.get_collection(COLLECTION_NAME)
        if info.status == models.CollectionStatus.GREEN:
            print(" Done.")
            break
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"   ⏱️  Setup Time: {time.time()-start_t:.2f}s")

    # 4. 核心实验：Offset 陷阱
    # 我们固定 ef = 16 (极小)
    FIXED_EF = 16
    LIMIT = 10
    
    # 随机生成一个 Query
    query_vec = np.random.random((1, DIM)).astype(np.float32)[0].tolist()
    
    print(f"\n{CYAN}🧪 Starting Offset Stress Test (Fixed Configured ef={FIXED_EF}){RESET}")
    print(f"   Note: We are requesting Limit={LIMIT} + Offset")
    
    # 测试梯度：Offset 逐渐增大，直到远超 ef
    offsets = [0, 10, 20, 50, 100, 200, 500]
    
    for offset in offsets:
        req_k = offset + LIMIT
        
        # A. 获取真理 (Exact Search / Brute Force)
        # Qdrant 支持 exact=True 参数，强制进行精确搜索
        gt_res = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=LIMIT,
            offset=offset,
            search_params=models.SearchParams(exact=True) # <--- 真理模式
        )
        gt_ids = [h.id for h in gt_res]
        
        # B. 获取实验组 (HNSW, 限制 ef=16)
        test_res = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=LIMIT,
            offset=offset,
            search_params=models.SearchParams(
                hnsw_ef=FIXED_EF, # <--- 强制指定极小的 ef
                exact=False
            )
        )
        test_ids = [h.id for h in test_res]
        
        # C. 对比
        recall = calculate_recall(test_ids, gt_ids)
        
        # 判定状态
        status = ""
        is_boundary_crossed = req_k > FIXED_EF
        
        if not is_boundary_crossed:
            # 安全区 (req_k <= 16)
            status = f"{GREEN}✅ Safe Zone{RESET}"
        else:
            # 危险区 (req_k > 16)
            if recall > 0.9: 
                status = f"{GREEN}✅ Auto-Expanded (Silent Fix){RESET}"
            elif recall == 0:
                status = f"{RED}❌ Recall Collapse{RESET}"
            else:
                status = f"{YELLOW}⚠️ Degraded{RESET}"

        print(f"   👉 Offset={offset:<3} | Req_K={req_k:<3} | Recall={recall:.0%} | {status}")
        
        # 额外验证：如果发生了 Auto-Expanded，打印一条解释
        if is_boundary_crossed and recall > 0.9:
            print(f"      👀 Inference: Although configured ef={FIXED_EF}, Qdrant internally used ef >= {req_k}")

if __name__ == "__main__":
    run_qdrant_test()