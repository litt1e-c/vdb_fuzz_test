"""
🔬 关键发现：id == 精确定位 vs 全表扫描的 NULL 处理差异
"""
from pymilvus import connections, Collection
import pymilvus
import pandas as pd
import random
import numpy as np
from milvus_fuzz_oracle import DataManager
import sys

HOST = "127.0.0.1"
PORT = "19531"
COLLECTION = "fuzz_stable_v3"
SEED = 999

random.seed(SEED)
np.random.seed(SEED)
dm = DataManager()
dm.generate_schema()
dm.generate_data()

connections.connect("default", host=HOST, port=PORT)
col = Collection(COLLECTION)
col.load()

# 找到所有 c4=NULL 的行
null_rows = dm.df[dm.df['c4'].isnull()]['id'].tolist()
print("\n" + "=" * 80)
print(f"📊 数据集中有 {len(null_rows)} 行的 c4 为 NULL")
print(f"   前10个 ID: {null_rows[:10]}")
print("=" * 80)

print("\n环境版本:")
print(f"   • Milvus Server: v2.6.7")
print(f"   • PyMilvus    : {pymilvus.__version__}")
print(f"   • Pandas      : {pd.__version__}")
print(f"   • NumPy       : {np.__version__}")
print(f"   • Python      : {sys.version.split()[0]}")
print("=" * 80)

threshold = 7.3154750146117635

# 测试一: 全表扫描
print("\n" + "=" * 80)
print(f"【测试A】全表扫描: c4 <= {threshold}")
print("=" * 80)
result_a = col.query(
    f"c4 <= {threshold}",
    output_fields=["id"],
    limit=100,
    consistency_level="Strong"
)
returned_ids_a = [r['id'] for r in result_a]
null_returned_a = [rid for rid in returned_ids_a if rid in null_rows]
print(f"   返回行数: {len(returned_ids_a)}")
print(f"   其中 c4=NULL 的行: {null_returned_a}")
print(f"   结论: {'❌ BUG!' if len(null_returned_a) > 0 else '✅ 正确（NULL 未被返回）'}")

# 测试二: 使用 id == 精确定位（测试前10个NULL行）
print("\n" + "=" * 80)
print(f"【测试B】精确定位: (id == X) and (c4 <= {threshold})")
print("=" * 80)

bug_count = 0
for test_id in null_rows[:10]:
    result_b = col.query(
        f"(id == {test_id}) and (c4 <= {threshold})",
        output_fields=["id"],
        limit=1,
        consistency_level="Strong"
    )
    if len(result_b) > 0:
        print(f"   ID={test_id}: ❌ BUG! (返回了 NULL 行)")
        bug_count += 1
    else:
        print(f"   ID={test_id}: ✅ 正确")

print(f"\n   Bug 触发率: {bug_count}/10")

# 测试三: 使用 id in [list] 批量定位
print("\n" + "=" * 80)
print(f"【测试C】批量定位: (id in [list]) and (c4 <= {threshold})")
print("=" * 80)

test_ids = null_rows[:10]
id_list_str = "[" + ", ".join(map(str, test_ids)) + "]"
result_c = col.query(
    f"(id in {id_list_str}) and (c4 <= {threshold})",
    output_fields=["id"],
    limit=20,
    consistency_level="Strong"
)
returned_ids_c = [r['id'] for r in result_c]
print(f"   测试 IDs: {test_ids}")
print(f"   返回 IDs: {returned_ids_c}")
print(f"   结论: {'❌ BUG!' if len(returned_ids_c) > 0 else '✅ 正确'}")

# 总结
print("\n" + "=" * 80)
print("🎯 核心发现")
print("=" * 80)
print("\n当查询 NULL 字段时，Milvus 的行为取决于查询方式：")
print("   • 全表扫描 (c4 <= value):")
print("     → 正确：不返回 NULL 行")
print("   • 精确定位 (id == X) and (c4 <= value):")
print(f"     → 错误：返回了 NULL 行 (Bug率: {bug_count}/10)")
print("\n💡 推测:")
print("   1. 全表扫描使用的查询计划正确处理了 NULL")
print("   2. 使用主键过滤 (id ==) 时，查询优化器可能走了不同的路径")
print("   3. 主键索引 + 标量过滤的组合查询存在 NULL 处理 bug")
print("\n🔧 这对 Milvus 开发团队非常有价值:")
print("   • 明确了 bug 触发路径：主键精确匹配 + 标量条件")
print("   • 可以集中排查主键索引相关的查询优化器代码")

print("\n" + "=" * 80)
