"""
╔════════════════════════════════════════════════════════════════════════════╗
║  Milvus NULL Comparison Bug - Minimal Reproduction Test                   ║
║  重新生成数据并复现 NULL 比较 Bug                                           ║
╚════════════════════════════════════════════════════════════════════════════╝

Bug Description:
  NULL values incorrectly satisfy comparison operators when combined with 
  primary key filters. This violates SQL-92 NULL semantics.

Test Case:
  Query: (id == 3588) AND (c4 <= 7.3154750146117635)
  Row 3588 has c4 = NULL
  Expected result: 0 rows (NULL should NOT satisfy <=)
  Actual result: 1 row (BUG!)

Usage:
  python test_null_comparison_bug.py [--host 127.0.0.1] [--port 19530]
"""

import time
import random
import string
import numpy as np
import pandas as pd
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import sys

# ================= 配置 =================
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "repro_full_structure"  # 使用之前的集合
SEED = 999
TARGET_ID = 3588
THRESHOLD = 7.3154750146117635
N = 5000
DIM = 128
USE_EXISTING_COLLECTION = True  # 使用现有集合，不重新创建

def print_env_info():
    """打印环境信息"""
    print("\n" + "=" * 80)
    print("ENVIRONMENT INFORMATION")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]}")
    try:
        import pymilvus
        print(f"PyMilvus: {pymilvus.__version__}")
    except:
        print("PyMilvus: Unknown")
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")
    print("=" * 80 + "\n")

def generate_data():
    """生成测试数据，确保 ID 3588 的 c4 字段为 NULL"""
    print(f"🌊 1. Generating {N} rows of test data...")
    rng = np.random.default_rng(SEED)
    
    # 生成向量
    vectors = rng.random((N, DIM), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    # 生成 ID
    ids = np.arange(N, dtype=np.int64)
    
    # 生成 20 个字段 (c0-c19)
    data = {"id": ids}
    
    # 字段类型（与原代码保持一致）
    field_rng = np.random.default_rng(SEED)
    types_pool = [DataType.INT64, DataType.DOUBLE, DataType.BOOL, DataType.VARCHAR]
    field_types = [field_rng.choice(types_pool) for _ in range(20)]
    
    # 🎯 强制 c4 为 DOUBLE 类型（确保与浮点阈值匹配）
    field_types[4] = DataType.DOUBLE
    
    print("   -> Generating scalar fields...")
    for i in range(20):
        fname = f"c{i}"
        ftype = field_types[i]
        
        if ftype == DataType.INT64:
            data[fname] = rng.integers(-180000, 180000, size=N)
        elif ftype == DataType.DOUBLE:
            data[fname] = rng.random(N) * 10000
        elif ftype == DataType.BOOL:
            data[fname] = rng.choice([True, False], size=N)
        elif ftype == DataType.VARCHAR:
            chars = string.ascii_letters + string.digits
            data[fname] = [''.join(random.choices(chars, k=random.randint(5, 50))) for _ in range(N)]
        
        # 注入 10% 的 NULL 值
        mask = rng.random(N) < 0.1
        temp_arr = np.array(data[fname], dtype=object)
        temp_arr[mask] = None
        data[fname] = temp_arr
    
    # 生成 JSON 字段
    print("   -> Generating JSON field...")
    json_list = []
    for _ in range(N):
        json_obj = {
            "price": int(rng.integers(0, 1000)),
            "color": rng.choice(["Red", "Blue", "Green"]),
            "active": bool(rng.choice([True, False])),
        }
        if rng.random() < 0.1:
            json_obj["price"] = None
        if rng.random() < 0.8:
            json_obj["config"] = {"version": int(rng.integers(1, 10))}
        json_list.append(json_obj)
    
    data["meta_json"] = np.array(json_list, dtype=object)
    
    # 生成 Array 字段
    print("   -> Generating Array field...")
    arr_list = []
    for _ in range(N):
        length = random.randint(0, 5)
        arr_list.append(list(rng.integers(0, 100, size=length)))
    data["tags_array"] = np.array(arr_list, dtype=object)
    
    # 🎯 关键：确保 ID 3588 的 c4 字段为 NULL
    if TARGET_ID < N:
        # 先确认 c4 的类型（应该是 DOUBLE）
        c4_type = field_types[4]
        print(f"\n   🎯 Setting c4[{TARGET_ID}] = NULL")
        print(f"      c4 field type: {c4_type} (DOUBLE expected for float comparison)")
        data["c4"][TARGET_ID] = None
    
    df = pd.DataFrame(data)
    
    print(f"   ✅ Data generation complete.")
    print(f"   -> DataFrame shape: {df.shape}")
    print(f"\n   🔍 Verification - Row {TARGET_ID} data:")
    print(f"      id={df.loc[TARGET_ID, 'id']}")
    print(f"      c4={df.loc[TARGET_ID, 'c4']}")
    print(f"      c4 type in pandas: {type(df.loc[TARGET_ID, 'c4'])}")
    
    return df, vectors, field_types

def create_collection(field_types):
    """创建 Milvus 集合"""
    print(f"\n📦 2. Creating Milvus Collection '{COLLECTION_NAME}'...")
    
    if utility.has_collection(COLLECTION_NAME):
        print(f"   -> Dropping existing collection...")
        utility.drop_collection(COLLECTION_NAME)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    
    # 添加 20 个标量字段
    for i in range(20):
        fname = f"c{i}"
        ftype = field_types[i]
        
        if ftype == DataType.VARCHAR:
            fields.append(FieldSchema(name=fname, dtype=ftype, nullable=True, max_length=512))
        else:
            fields.append(FieldSchema(name=fname, dtype=ftype, nullable=True))
    
    # 添加 JSON 和 Array 字段
    fields.append(FieldSchema(name="meta_json", dtype=DataType.JSON, nullable=True))
    fields.append(FieldSchema(
        name="tags_array", 
        dtype=DataType.ARRAY, 
        element_type=DataType.INT64, 
        max_capacity=50,
        nullable=True
    ))
    
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    col = Collection(COLLECTION_NAME, schema)
    print(f"   ✅ Collection created with {len(fields)} fields")
    
    return col

def insert_data(col, df, vectors):
    """插入数据到 Milvus"""
    print(f"\n⚡ 3. Inserting {len(df)} rows into collection...")
    
    records = df.to_dict(orient="records")
    batch_size = 200
    
    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        batch_data = records[start:end]
        
        insert_rows = []
        for i, row in enumerate(batch_data):
            row_with_vec = row.copy()
            row_with_vec["vector"] = vectors[start + i].tolist()
            
            # 清理 numpy 类型
            for k, v in row_with_vec.items():
                if hasattr(v, "item"):
                    row_with_vec[k] = v.item()
            
            insert_rows.append(row_with_vec)
        
        col.insert(insert_rows)
        print(f"   Inserted {end}/{len(records)}...", end="\r")
    
    col.flush()
    print(f"\n   ✅ Insert complete. Flushing...")
    
    # 建索引
    print(f"\n🔨 4. Building index...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    col.create_index("vector", index_params)
    col.load()
    print(f"   ✅ Index built and collection loaded")

def verify_stored_data(col):
    """验证存储的数据"""
    print(f"\n🕵️ 5. Verifying stored data for ID {TARGET_ID}...")
    
    # 查询完整的行数据（获取所有字段）
    check_res = col.query(
        f"id == {TARGET_ID}", 
        output_fields=["*"],  # 获取所有字段
        consistency_level="Strong"
    )
    
    if check_res:
        row = check_res[0]
        actual_val = row.get("c4")
        
        # 🎯 打印完整的行数据
        print(f"   -> Complete row data for ID {TARGET_ID}:")
        print("   " + "=" * 76)
        for key, value in row.items():
            if key == "vector":
                print(f"      {key:<15}: [vector data, dim={len(value)}]")
            else:
                print(f"      {key:<15}: {value}")
        print("   " + "=" * 76)
        
        print(f"\n   -> Key field verification:")
        print(f"      id={row['id']}")
        print(f"      c4={actual_val}")
        print(f"      Python type: {type(actual_val)}")
        
        if actual_val is None:
            print("   ✅ CONFIRMED: c4 is truly NULL (NoneType)")
            print("      Any inequality query matching this row IS A BUG.")
            return True
        else:
            print(f"   ⚠️ WARNING: c4 was stored as: {actual_val}")
            print("      This is unexpected! NULL was not preserved.")
            return False
    else:
        print("   ❌ ERROR: ID not found in collection")
        return False

def execute_bug_query(col):
    """执行 bug 查询"""
    print(f"\n🔍 6. Executing bug reproduction query...")
    expr = f"(id == {TARGET_ID}) and (c4 <= {THRESHOLD})"
    print(f"   Expression: {expr}")
    print(f"   Expected result: 0 rows (NULL should NOT satisfy <=)")
    
    try:
        res = col.query(
            expr, 
            output_fields=["id", "c4"], 
            consistency_level="Strong"
        )
        
        print(f"\n" + "=" * 80)
        print("QUERY RESULT")
        print("=" * 80)
        print(f"   Returned rows: {len(res)}")
        
        if len(res) > 0:
            row = res[0]
            print(f"\n❌ BUG REPRODUCED!")
            print("=" * 80)
            print(f"\n   Row data:")
            print(f"      id={row['id']}")
            print(f"      c4={row['c4']}")
            print(f"\n   Analysis:")
            print(f"      Expected: 0 rows (NULL should NOT satisfy <=)")
            print(f"      Actual:   {len(res)} row(s) returned")
            print(f"\n   Root Cause:")
            print(f"      c4=NULL incorrectly satisfies (c4 <= {THRESHOLD})")
            print(f"      This violates SQL-92 NULL semantics.")
            print(f"\n   SQL-92 Standard:")
            print(f"      NULL compared with any value should return UNKNOWN (treated as False)")
            print(f"      Therefore, (c4 <= 7.315...) should be False when c4 is NULL")
            print("=" * 80 + "\n")
            return True
        else:
            print(f"\n✅ CORRECT BEHAVIOR")
            print("=" * 80)
            print(f"   No rows returned (as expected)")
            print(f"   NULL comparison handled correctly")
            print("=" * 80 + "\n")
            return False
    except Exception as e:
        print(f"\n❌ Query Error: {e}")
        return False

def run_additional_tests(col):
    """运行额外的测试用例"""
    print(f"\n🧪 7. Running additional NULL comparison tests...")
    
    test_cases = [
        # 测试不同的比较操作符
        (f"id == {TARGET_ID} and c4 < {THRESHOLD}", "less than"),
        (f"id == {TARGET_ID} and c4 > {THRESHOLD}", "greater than"),
        (f"id == {TARGET_ID} and c4 >= {THRESHOLD}", "greater or equal"),
        (f"id == {TARGET_ID} and c4 == {THRESHOLD}", "equal"),
        (f"id == {TARGET_ID} and c4 != {THRESHOLD}", "not equal"),
        # 测试 IS NULL
        (f"id == {TARGET_ID} and c4 is null", "is null"),
        (f"id == {TARGET_ID} and c4 is not null", "is not null"),
    ]
    
    bug_count = 0
    
    for expr, desc in test_cases:
        try:
            res = col.query(expr, output_fields=["id"], consistency_level="Strong")
            count = len(res)
            
            # 预期结果：只有 "is null" 应该返回 1 行，其他都应该 0 行
            expected = 1 if "is null" in desc and "not" not in desc else 0
            status = "✅" if count == expected else "❌"
            
            print(f"   {status} {desc:20s}: {count} rows (expected: {expected})")
            
            if count != expected:
                bug_count += 1
        except Exception as e:
            print(f"   ⚠️ {desc:20s}: Error - {e}")
    
    print(f"\n   Summary: {bug_count} unexpected results found")
    return bug_count

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Milvus NULL Comparison Bug - Query Existing Collection")
    print("=" * 80)
    
    print_env_info()
    
    # 连接 Milvus
    print(f"🔌 Connecting to Milvus at {HOST}:{PORT}...")
    try:
        connections.connect("default", host=HOST, port=PORT, timeout=30)
        print("   ✅ Connected successfully")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("   Please check if Milvus server is running")
        return
    
    # 检查集合是否存在
    print(f"\n📦 Checking collection '{COLLECTION_NAME}'...")
    if not utility.has_collection(COLLECTION_NAME):
        print(f"   ❌ Collection '{COLLECTION_NAME}' does not exist!")
        print("   Please run 'repro_from_file.py' first to create the collection.")
        return
    
    # 加载集合
    print(f"   ✅ Collection exists, loading...")
    col = Collection(COLLECTION_NAME)
    col.load()
    print(f"   ✅ Collection loaded")
    
    # 获取集合信息
    num_entities = col.num_entities
    print(f"   -> Total entities: {num_entities}")
    
    # 验证存储的数据
    is_null = verify_stored_data(col)
    
    if not is_null:
        print("\n⚠️ Warning: c4 is not NULL in Milvus. Bug may not reproduce.")
    
    # 执行 bug 查询
    bug_found = execute_bug_query(col)
    
    # 运行额外测试
    bug_count = run_additional_tests(col)
    
    # 最终总结
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    if bug_found or bug_count > 0:
        print("❌ NULL comparison bug REPRODUCED")
        print(f"   {bug_count} additional NULL comparison issues found")
    else:
        print("✅ NULL comparison handled correctly")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()