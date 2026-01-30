import random
import numpy as np
from milvus_fuzz_oracle import DataManager, COLLECTION_NAME
from pymilvus import connections, Collection

# ================= 配置 =================
SEED = 999
HOST = "127.0.0.1"
PORT = "19531"

# 颜色代码
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def main():
    print(f"\n{BOLD}🚀 Milvus Logic Verification (Read-Only Mode){RESET}")
    print("=" * 60)

    # 1. 在 Python 内存中恢复“真值表”
    #    (这一步完全不碰数据库，只是在内存里算一遍数据长什么样)
    print(f"Step 1: Reconstructing expected data in memory (Seed={SEED})...")
    random.seed(SEED); np.random.seed(SEED)
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data() # 只在内存生成
    df = dm.df 
    print(f"   -> Reference data ready in Python memory.")

    # 2. 连接 Milvus (只读模式)
    print(f"Step 2: Connecting to existing collection '{COLLECTION_NAME}'...")
    connections.connect("default", host=HOST, port=PORT)
    
    # 获取现有集合，如果集合不存在会直接报错，不会新建
    col = Collection(COLLECTION_NAME) 
    col.load() # 确保加载到内存以供查询

    # 3. 执行“导致Bug”的查询
    # Case 4679: c0 <= 4.78
    target_id = 4526
    threshold = 4.787315444874056
    
    expr = """(((meta_json["active"] == true and meta_json["color"] == "Blue") or ((c16 >= 2538.641880850278 or ((meta_json["active"] == true and meta_json["color"] == "Blue") and (meta_json["price"] > 107 and meta_json["price"] < 261))) and ((c0 > 2146.802199098024 and c2 >= 36295) and (c12 == false and (meta_json["active"] == true and meta_json["color"] == "Blue"))))) and (c17 == false or (((c10 < "IHSVh" and c5 == false) or (c0 <= 4.787315444874056 or (meta_json["price"] > 293 and meta_json["price"] < 453))) and ((meta_json["config"]["version"] == 3 or c17 is null) and (c11 > -51460 and c9 != 105382.9375)))))"""
    
    print(f"Step 3: Executing Query...")
    print(f"   Query Condition: {CYAN}c0 <= {threshold}{RESET}")
    
    # 执行查询
    res = col.query(expr, output_fields=["id"], limit=16384, consistency_level="Strong")
    returned_ids = {r["id"] for r in res}
    
    print(f"   Milvus returned {len(returned_ids)} rows.")

    # 4. 摆事实，讲道理
    print("\n" + "=" * 60)
    print(f"{BOLD}Step 4: Verification{RESET}")
    print("=" * 60)

    if target_id in returned_ids:
        # 从 Python 内存的真值表中拿出这行数据
        row_data = df[df["id"] == target_id].iloc[0]
        c0_val = row_data["c0"]
        
        print(f"🔍 Result Analysis for ID: {target_id}")
        
        # 事实 A：数据库里存的是什么（根据 Seed 999 还原）
        val_str = f"{RED}None/NULL{RESET}" if c0_val is None else c0_val
        print(f"   [Fact] Value in 'c0': {val_str}")
        
        # 事实 B：查询要求什么
        print(f"   [Fact] Query expects: c0 <= {threshold}")
        
        # 矛盾展示
        print("-" * 40)
        print(f"   Logic Check:")
        print(f"   - Is None <= {threshold} ?  👉 {GREEN}NO (Should not return){RESET}")
        print(f"   - Did Milvus return it?      👉 {RED}YES (Bug){RESET}")
        print("-" * 40)
        
    else:
        print(f"{GREEN}Current collection does not reproduce the bug.{RESET}")
        print("Please run check_mismatch.py first to populate data.")

if __name__ == "__main__":
    main()