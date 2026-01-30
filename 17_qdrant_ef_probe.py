import time
import numpy as np
from qdrant_client import QdrantClient, models

# --- 🔥 配置 ---
HOST = '127.0.0.1'
PORT = 6333
COLLECTION_NAME = "qdrant_ef_probe"
DIM = 768
NUM_ENTITIES = 50000
FIXED_EF = 16 
BATCH_SIZE = 5000 # 分批插入，防止超时

# 颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def setup_qdrant():
    # 尝试 gRPC
    try:
        client = QdrantClient(host=HOST, port=PORT, grpc_port=6334, prefer_grpc=True, timeout=None)
        print("   🔌 Connected via gRPC")
    except:
        client = QdrantClient(host=HOST, port=PORT, timeout=None)
        print("   🔌 Connected via HTTP")

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=DIM, distance=models.Distance.EUCLID,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
        ),
        # 写入时禁用索引
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0)
    )
    
    # 插入数据 (分批)
    print(f"   📥 Inserting {NUM_ENTITIES} vectors...")
    all_points = [
        models.PointStruct(id=i, vector=np.random.random(DIM).tolist()) 
        for i in range(NUM_ENTITIES)
    ]
    
    for i in range(0, NUM_ENTITIES, BATCH_SIZE):
        end = min(i+BATCH_SIZE, NUM_ENTITIES)
        client.upsert(COLLECTION_NAME, all_points[i:end], wait=False)
        
    # 触发索引
    print("   🏗️  Triggering Indexing...", end="", flush=True)
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
    )
    
    # 轮询等待
    for _ in range(120):
        info = client.get_collection(COLLECTION_NAME)
        if info.status == models.CollectionStatus.GREEN:
            print(" Done.")
            return client
        time.sleep(1)
        print(".", end="", flush=True)
        
    print(" (Timeout but proceeding)")
    return client

def probe_latency(client):
    print(f"\n{CYAN}🔍 Probing HNSW 'ef' expansion strategy...{RESET}")
    query_vec = np.random.random(DIM).tolist()
    
    # 探测点
    probe_points = [
        100,            
        250, 255, 256, 257, 300,  
        500, 511, 512, 513, 550   
    ]
    
    results = []
    
    for k in probe_points:
        latencies = []
        # 预热
        client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=k,
            search_params=models.SearchParams(hnsw_ef=FIXED_EF)
        )
        
        # 测量 10 次
        for _ in range(10):
            start = time.perf_counter() 
            client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vec,
                limit=k, 
                search_params=models.SearchParams(hnsw_ef=FIXED_EF)
            )
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_lat = sum(latencies) / len(latencies)
        results.append((k, avg_lat))
        print(f"   Req_K={k:<4} | Avg Latency={avg_lat:.4f} ms")

    return results

def analyze_trend(results):
    print(f"\n📊 Trend Analysis:")
    prev_k, prev_lat = results[0]
    
    for k, lat in results[1:]:
        delta_k = k - prev_k
        delta_lat = lat - prev_lat
        
        # 简单的跳变检测
        is_jump = False
        if k in [257, 513] and lat > prev_lat * 1.3: 
            is_jump = True
        
        marker = f"{RED}🚨 JUMP DETECTED{RESET}" if is_jump else "Smooth"
        print(f"   {prev_k} -> {k:<4} | Delta: {delta_lat:+.4f} ms | {marker}")
        prev_k, prev_lat = k, lat

if __name__ == "__main__":
    client = setup_qdrant()
    data = probe_latency(client)
    analyze_trend(data)