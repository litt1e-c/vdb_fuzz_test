from pymilvus import connections, Collection

# 配置
HOST, PORT = "127.0.0.1", "19531"
COL_NAME = "fuzz_stable_v3"
TARGET_ID = 3588
FIELD = "c4"
THRESHOLD = 7.3154750146117635

# 连接Milvus
connections.connect("default", host=HOST, port=PORT)
col = Collection(COL_NAME)
col.load()

# 验证原始数据
print(f"📊 原始数据检查 (ID={TARGET_ID}):")
res = col.query(f"id == {TARGET_ID}", output_fields=[FIELD])
val = res[0][FIELD]
print(f"  {FIELD}值: {val} (Python None = DB NULL)")

# 执行查询 - NULL <= 阈值 应返回假
expr = f"id == {TARGET_ID} and {FIELD} <= {THRESHOLD}"
print(f"\n🔍 执行查询: \"{expr}\"")
result = col.query(expr, output_fields=[FIELD])

# 分析结果
print(f"\n📋 查询结果: {len(result)} 条记录")
if len(result) > 0:
    print(f"❌ BUG复现: NULL <= {THRESHOLD} 被错误地判定为真")
    print(f"   返回的数据: {result[0]}")
else:
    print("✅ 行为正确: NULL比较返回假")