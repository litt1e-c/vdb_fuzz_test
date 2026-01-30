import numpy as np
from pymilvus import (
    connections, Collection, AnnSearchRequest, RRFRanker
)

HOST = '127.0.0.1'
PORT = '19530'
COLLECTION_NAME = "crash_test_empty_v2523"

# 幸存者的 ID（我们在 prepare_data 里设置的）
SURVIVOR_ID = 500000 + 1 
TARGET_FILTER = "flag == 1"

def verify():
    print(f"Connecting to {HOST}...")
    connections.connect("default", host=HOST, port=PORT)
    col = Collection(COLLECTION_NAME)
    col.load()
    
    print(f"Loaded. Verifying Logic Correctness on 'Lone Survivor' (ID={SURVIVOR_ID})...")
    
    # 通用查询向量
    query_vec = [[0.5] * 128] # 随便给个向量

    # --- 验证 1: Empty + Boost ---
    print("\n[1] Verifying Boost Logic...")
    res = col.search(
        data=query_vec, anns_field="vector", 
        param={"metric_type": "L2", "params": {"ef": 64}},
        limit=10, expr=TARGET_FILTER,
        output_fields=["id"]
    )
    if len(res[0]) == 1 and res[0][0].id == SURVIVOR_ID:
        print("   ✅ Boost/Search: Found the survivor correctly.")
    else:
        print(f"   ❌ Boost/Search FAILED: Expected 1 result, got {len(res[0])}")

    # --- 验证 2: Empty + GroupBy ---
    print("\n[2] Verifying GroupBy Logic...")
    try:
        res = col.search(
            data=query_vec, anns_field="vector", 
            param={"metric_type": "L2", "params": {"nprobe": 10}, "group_by_field": "group_id"},
            limit=10, expr=TARGET_FILTER,
            output_fields=["group_id"]
        )
        # 幸存者的 group_id 是 999
        if len(res[0]) == 1 and res[0][0].entity.get("group_id") == 999:
            print("   ✅ GroupBy: Grouped correctly.")
        else:
            print(f"   ❌ GroupBy FAILED: Result mismatch. Got {res[0]}")
    except Exception as e:
        print(f"   ❌ GroupBy CRASHED/ERROR: {e}")

    # --- 验证 3: Empty + Iterator ---
    print("\n[3] Verifying Iterator Logic...")
    try:
        iterator = col.query_iterator(
            batch_size=10, expr=TARGET_FILTER, output_fields=["id"]
        )
        page1 = iterator.next()
        if len(page1) == 1 and page1[0]["id"] == SURVIVOR_ID:
            print("   ✅ Iterator: Page 1 correct.")
            # 验证下一页是否为空（不应崩）
            page2 = iterator.next()
            if len(page2) == 0:
                print("   ✅ Iterator: Page 2 correct (Empty).")
            else:
                 print("   ❌ Iterator: Page 2 should be empty!")
        else:
             print(f"   ❌ Iterator: Page 1 mismatch. Got {page1}")
    except StopIteration:
        print("   ✅ Iterator: StopIteration caught correctly.")
    except Exception as e:
        print(f"   ❌ Iterator CRASHED/ERROR: {e}")

    # --- 验证 4: Empty + Hybrid ---
    print("\n[4] Verifying Hybrid Search Logic...")
    try:
        req1 = AnnSearchRequest(
            data=query_vec, anns_field="vector", 
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10, expr=TARGET_FILTER
        )
        # 模拟两路，都指向同一个幸存者
        req2 = AnnSearchRequest(
            data=query_vec, anns_field="vector", 
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10, expr=TARGET_FILTER
        )
        
        res = col.hybrid_search(
            reqs=[req1, req2], rerank=RRFRanker(), limit=10,
            output_fields=["id"]
        )
        if len(res[0]) == 1 and res[0][0].id == SURVIVOR_ID:
             print("   ✅ Hybrid: Reranked correctly.")
        else:
             print(f"   ❌ Hybrid FAILED: Got {len(res[0])} results.")
    except Exception as e:
        print(f"   ❌ Hybrid CRASHED/ERROR: {e}")

if __name__ == "__main__":
    verify()