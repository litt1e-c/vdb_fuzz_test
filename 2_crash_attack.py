import threading
import time
import random
from pymilvus import (
    connections, Collection, AnnSearchRequest, RRFRanker, 
    WeightedRanker, MilvusException
)

HOST = '127.0.0.1'
PORT = '19530'
COLLECTION_NAME = "crash_test_empty_v2523"

# ！！！必杀技：这个 Filter 在数据库里匹配不到任何数据！！！
EMPTY_FILTER = "flag == 1" 

def get_collection():
    connections.connect("default", host=HOST, port=PORT)
    return Collection(COLLECTION_NAME)

# --- 场景 1: Empty + Boost (模拟 #44041) ---
def test_empty_boost(col):
    # 在 Milvus 中，Boost 通常通过 metric 参数或 function score 体现
    # 这里我们用一个带极小 radius 的 range search 配合 expr
    # 试图让计算层进行空转
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10, "radius": 0.0001, "range_filter": 0.0}
    }
    try:
        col.search(
            data=[[0.1]*128], anns_field="vector", param=search_params,
            limit=10, expr=EMPTY_FILTER  # <--- 强制为空
        )
    except MilvusException:
        pass # 报错是正常的，只要不崩

# --- 场景 2: Empty + GroupBy (重点关注，v2.6.5有相关修复) ---
def test_empty_groupby(col):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
        "group_by_field": "group_id" # <--- 核心
    }
    try:
        col.search(
            data=[[0.1]*128], anns_field="vector", param=search_params,
            limit=10, expr=EMPTY_FILTER
        )
    except MilvusException:
        pass

# --- 场景 3: Empty + Iterator (状态机测试) ---
def test_empty_iterator(col):
    try:
        # 2.5.x 推荐使用 query_iterator 或 search_iterator
        iterator = col.query_iterator(
            batch_size=10,
            expr=EMPTY_FILTER, # <--- 搜不到任何东西
            output_fields=["id"]
        )
        # 危险动作：对着空迭代器要数据
        res = iterator.next()
    except StopIteration:
        pass
    except MilvusException:
        pass

# --- 场景 4: Empty + Range Search (内存边界测试) ---
def test_empty_range(col):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10, "radius": 1.0, "range_filter": 0.0}
    }
    try:
        col.search(
            data=[[0.1]*128], anns_field="vector", param=search_params,
            limit=10, expr=EMPTY_FILTER
        )
    except MilvusException:
        pass

# --- 场景 5: Empty + Hybrid Search (v2.5.21 修复重点) ---
def test_empty_hybrid(col):
    # 构造两路必定为空的请求
    req1 = AnnSearchRequest(
        data=[[0.1]*128], anns_field="vector", 
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10, expr=EMPTY_FILTER
    )
    req2 = AnnSearchRequest(
        data=[[0.1]*128], anns_field="vector", 
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10, expr=EMPTY_FILTER
    )
    
    try:
        # 只有两路都为空，才能测试 Reranker 除零/空指针风险
        col.hybrid_search(
            reqs=[req1, req2], 
            rerank=RRFRanker(), 
            limit=10
        )
    except MilvusException:
        pass

def worker_thread(tid):
    print(f"Worker {tid} started.")
    col = get_collection()
    
    for i in range(5000): # 每个线程跑500轮
        try:
            # 随机执行一种攻击
            scenario = random.choice([
                test_empty_boost, 
                test_empty_groupby, 
                test_empty_iterator,
                test_empty_range,
                test_empty_hybrid
            ])
            scenario(col)
            
            if i % 100 == 0:
                print(f"Worker {tid}: Alive at round {i}")
                
        except Exception as e:
            # 如果连不上服务器，说明可能崩了
            if "connection" in str(e).lower() or "closed" in str(e).lower():
                print(f"!!! CRITICAL: Worker {tid} lost connection. Server might have CRASHED! Error: {e}")
                return
    print(f"Worker {tid} finished.")

if __name__ == "__main__":
    # 启动并发轰炸
    threads = []
    CONCURRENCY = 10 # 10个并发线程
    
    print(f"Starting {CONCURRENCY} workers to attack Milvus v2.5.23...")
    t0 = time.time()
    
    for i in range(CONCURRENCY):
        t = threading.Thread(target=worker_thread, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    print(f"Attack finished in {time.time()-t0:.2f}s. If you see this, Milvus survived.")