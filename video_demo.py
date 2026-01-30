"""
🎥 Milvus NULL 比较 Bug - 视频演示脚本
适合录制视频或提交 GitHub Issue

环境:
- Milvus v2.6.7
- Collection: fuzz_stable_v3 (5000行数据，使用 seed=999 生成)
"""
from pymilvus import connections, Collection
import pymilvus
import pandas as pd
import random
import numpy as np
from milvus_fuzz_oracle import DataManager
import time
import sys

# 配置
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION = "fuzz_stable_v3"
SEED = 999
TEST_ID = 3588
THRESHOLD = 7.3154750146117635

def pause(seconds=1):
    """录视频时的停顿"""
    time.sleep(seconds)

print("\n" + "=" * 80)
print("🎯 Milvus NULL Comparison Bug Demonstration")
print("=" * 80)
print(f"\nEnvironment:")
print(f"  • Milvus Version : v2.6.7 (server)")
print(f"  • PyMilvus      : {pymilvus.__version__}")
print(f"  • Pandas        : {pd.__version__}")
print(f"  • NumPy         : {np.__version__}")
print(f"  • Python        : {sys.version.split()[0]}")
print(f"  • Collection    : {COLLECTION}")
print(f"  • Test ID       : {TEST_ID}")
print("=" * 80)

pause(1)

# ============================================================================
# STEP 1: 生成数据并查看测试行
# ============================================================================
print("\n" + "=" * 80)
print("📊 STEP 1: Show the data of the test row")
print("=" * 80)

random.seed(SEED)
np.random.seed(SEED)
dm = DataManager()
dm.generate_schema()
dm.generate_data()

# 找到测试行
test_row = dm.df[dm.df['id'] == TEST_ID].iloc[0]

print(f"\n✅ Found row with id={TEST_ID}:")
print(f"\n  id  = {test_row['id']}")
print(f"  c4  = {test_row['c4']}  👈 This is NULL")
print(f"  c9  = {test_row['c9']}")
print(f"  c12 = {test_row['c12']}")
print(f"  c17 = {test_row['c17']}")

print(f"\n💡 Key Point: The field 'c4' is NULL")

pause(2)

# ============================================================================
# STEP 2: 连接 Milvus
# ============================================================================
print("\n" + "=" * 80)
print("🔌 STEP 2: Connect to Milvus")
print("=" * 80)

connections.connect("default", host=HOST, port=PORT)
col = Collection(COLLECTION)
col.load()

print(f"\n✅ Connected to Milvus")
print(f"  Collection: {COLLECTION}")
print(f"  Rows: {col.num_entities}")

pause(1)

# ============================================================================
# STEP 3: 执行查询
# ============================================================================
print("\n" + "=" * 80)
print("🔍 STEP 3: Execute Query")
print("=" * 80)

# 构建查询表达式
query_expr = f"(id == {TEST_ID}) and (c4 <= {THRESHOLD})"

print(f"\n📝 Query Expression:")
print(f"  {query_expr}")

print(f"\n🤔 Logical Analysis:")
print(f"  • (id == {TEST_ID})            → True  (we want this specific row)")
print(f"  • (c4 <= {THRESHOLD})          → ?")
print(f"    - c4 value = NULL")
print(f"    - According to SQL-92 standard:")
print(f"      NULL compared with any value = UNKNOWN")
print(f"      UNKNOWN in boolean context  = False")
print(f"  • True AND False              → False")

print(f"\n✅ Expected Result: Should return 0 rows")

pause(2)

# 执行查询
print(f"\n⏳ Executing query...")
result = col.query(
    query_expr,
    output_fields=["id", "c4"],
    limit=10,
    consistency_level="Strong"
)

pause(1)

# ============================================================================
# STEP 4: 显示结果
# ============================================================================
print("\n" + "=" * 80)
print("📊 STEP 4: Query Result")
print("=" * 80)

print(f"\n📦 Milvus returned {len(result)} row(s):")

if len(result) > 0:
    for r in result:
        print(f"\n  Row:")
        print(f"    id = {r['id']}")
        print(f"    c4 = {r['c4']}")
else:
    print("\n  (empty result)")

pause(2)

# ============================================================================
# STEP 5: 验证和结论
# ============================================================================
print("\n" + "=" * 80)
print("✅ STEP 5: Verification & Conclusion")
print("=" * 80)

if len(result) > 0 and TEST_ID in [r['id'] for r in result]:
    print(f"\n❌ BUG CONFIRMED!")
    print(f"\n  Problem:")
    print(f"    • Milvus returned row {TEST_ID}")
    print(f"    • But c4 = NULL")
    print(f"    • NULL should NOT satisfy (c4 <= {THRESHOLD})")
    
    print(f"\n  Impact:")
    print(f"    1. Violates SQL-92 NULL semantics")
    print(f"    2. Returns incorrect data (false positive)")
    print(f"    3. May cause downstream logic errors")
    
    print(f"\n  Trigger Condition:")
    print(f"    • Primary key exact match (id == X)")
    print(f"    • Combined with scalar field comparison")
    print(f"    • The scalar field value is NULL")
    
    print(f"\n" + "=" * 80)
    print("🚨 This is a serious bug that needs to be fixed!")
    print("=" * 80)
    
else:
    print(f"\n✅ Result is correct (no rows returned)")
    print(f"\n⚠️  Note: This test passed, but the bug was observed before.")
    print(f"   You may need to run this multiple times.")

# ============================================================================
# BONUS: 对比测试
# ============================================================================
print("\n" + "=" * 80)
print("🔬 BONUS: Comparison Test")
print("=" * 80)

print(f"\nLet's try a different query WITHOUT primary key filter:")
query_expr_2 = f"c4 <= {THRESHOLD}"
print(f"\n📝 Query: {query_expr_2}")

result_2 = col.query(
    query_expr_2,
    output_fields=["id"],
    limit=10,
    consistency_level="Strong"
)

id_list = [r['id'] for r in result_2]
print(f"\n📦 Returned {len(result_2)} rows")
print(f"  IDs: {id_list}")
print(f"\n❓ Is {TEST_ID} in the result? {TEST_ID in id_list}")

if TEST_ID not in id_list:
    print(f"\n💡 Interesting Finding:")
    print(f"  • Without (id == X): NULL is handled correctly")
    print(f"  • With (id == X):    NULL is handled incorrectly")
    print(f"\n  This suggests the bug is in the query optimizer")
    print(f"  when combining primary key filter with scalar conditions.")

print("\n" + "=" * 80)
print("🎬 End of Demonstration")
print("=" * 80)
print()
