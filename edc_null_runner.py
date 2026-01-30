"""
Milvus "Three-Valued Logic" Verification Runner
------------------------------------------------
Target: Verify how NULL behaves in query filtering (==, !=, IS NULL)
Scale: N=50,000 | DIM=768
Fixes: 
  1. Uses Pandas 'Int64' to fix float conversion.
  2. Converts 'Int64' to 'object' and replaces <NA> with None to fix NAType error.
"""
import time
import numpy as np
import pandas as pd
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# --- Config ---
HOST = "127.0.0.1"
PORT = "19531"  # 你的 Docker 端口
COLLECTION_NAME = "milvus_logic_test_v2"
N = 50000
DIM = 768
BATCH_SIZE = 2000

def connect_milvus():
    print(f"🔌 Connecting to Milvus at {HOST}:{PORT}...")
    try:
        connections.connect("default", host=HOST, port=PORT)
        print("✅ Connected.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        exit(1)

def setup_collection():
    if utility.has_collection(COLLECTION_NAME):
        print(f"🗑️ Dropping existing collection: {COLLECTION_NAME}")
        utility.drop_collection(COLLECTION_NAME)

    print("🛠️ Creating Schema with Nullable fields...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        # 测试 VARCHAR 的 Null 逻辑
        FieldSchema(name="tag", dtype=DataType.VARCHAR, max_length=64, nullable=True),
        # 测试 INT64 的 Null 逻辑
        FieldSchema(name="score", dtype=DataType.INT64, nullable=True)
    ]
    
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    col = Collection(COLLECTION_NAME, schema, consistency_level="Strong")
    return col

def generate_and_insert(col):
    print(f"🌊 Generating {N} items with mixed NULLs, Empty Strings, and Values...")
    
    ids = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(2025)
    vecs = rng.random((N, DIM), dtype=np.float32)
    
    rows = []
    print("    Constructing rows in memory...")
    for i in range(N):
        row = {"id": int(i), "vector": vecs[i]}
        
        if i < 10000:
            row["tag"] = "Group_A"
            row["score"] = 100
        elif i < 20000:
            row["tag"] = "Group_B"
            row["score"] = 0
        elif i < 30000:
            row["tag"] = ""  # 空字符串
            row["score"] = 0
        else:
            row["tag"] = None
            row["score"] = None
            
        rows.append(row)

    print(f"⚡ Inserting in batches of {BATCH_SIZE}...")
    
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        batch_rows = rows[start:end]
        
        # --- 核心修复：Pandas 完美清洗逻辑 ---
        df = pd.DataFrame(batch_rows)
        
        # 1. 处理 INT64 字段 (score)
        # 这一步是为了把 0.0 (float) 变成 0 (int)，同时保留缺失值为 <NA>
        if "score" in df.columns:
            # 先转 Int64 (解决 float 报错)
            s = df["score"].astype("Int64")
            # 再转 object (允许 Python None)
            s = s.astype(object)
            # 最后把 <NA> 替换为 None (解决 NAType 报错)
            s = s.where(pd.notnull(s), None)
            df["score"] = s
            
        # 2. 处理全局空值 (针对 tag 等其他字段)
        # 确保所有 NaN / None 都统一为 Python None
        df = df.where(pd.notnull(df), None)
        
        try:
            col.insert(df)
            print(f"    Inserted {end}/{N}...", end="\r")
        except Exception as e:
            print(f"\n❌ Insert Failed at batch {start}: {e}")
            # 调试：打印数据类型
            print("Types in dataframe:")
            print(df.dtypes)
            print("First row with None:")
            print(df.iloc[-1]) 
            exit(1)
            
    print("\n✅ Insert Complete.")
    
    print("🔨 Flushing and Building Index...")
    col.flush()
    col.create_index("vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()

def verify_logic(col):
    print("\n========================================")
    print("🔍 VERIFYING LOGIC (The Truth of NULL)")
    print("========================================")
    
    # 预期统计
    exp_A = 10000
    exp_B = 10000
    exp_C = 10000 # Empty String
    exp_D = 20000 # NULL
    
    def run_count(expr, desc):
        res = col.query(expr, output_fields=["id"])
        count = len(res)
        print(f"Query: {expr:<25} | Result: {count}")
        return count

    # 1. 等于查询
    print("\n[Case 1] Equality (==)")
    c1 = run_count('tag == "Group_A"', "Equal")
    assert c1 == exp_A, f"Failed: Exp {exp_A}, Got {c1}"
    print("   -> ✅ Correct.")

    # 2. 空字符串查询
    print("\n[Case 2] Empty String (== '')")
    c2 = run_count('tag == ""', "Empty String")
    assert c2 == exp_C, f"Failed: Exp {exp_C}, Got {c2}"
    print("   -> ✅ Correct.")

    # 3. 不等于查询
    print("\n[Case 3] Inequality (!= 'Group_A')")
    c3 = run_count('tag != "Group_A"', "Not Equal")
    expected_neq = exp_B + exp_C 
    
    if c3 == expected_neq:
        print(f"   -> ✅ Correct. Result ({c3}) excludes NULLs.")
    else:
        print(f"   -> ❌ Failed. Exp {expected_neq}, Got {c3}.")

    # 4. 范围查询
    print("\n[Case 4] Range (> 0)")
    c4 = run_count('score > 0', "Range")
    assert c4 == 10000, "Should only find Group A" 
    print("   -> ✅ Correct. NULL is not greater than 0.")

    # 5. NULL 查询
    print("\n[Case 5] IS NULL")
    c5 = run_count('tag is null', "Is Null")
    assert c5 == exp_D, f"Failed: Exp {exp_D}, Got {c5}"
    print("   -> ✅ Correct.")

    # 6. 完备性
    print("\n[Case 6] Completeness Formula")
    total_calc = c1 + c3 + c5
    print(f"   {c1} + {c3} + {c5} = {total_calc}")
    
    if total_calc == N:
        print(f"   -> 🏆 PERFECT! Milvus adheres to SQL Three-Valued Logic.")
    else:
        print(f"   -> ❌ MATH FAIL.")

if __name__ == "__main__":
    connect_milvus()
    col = setup_collection()
    generate_and_insert(col)
    verify_logic(col)