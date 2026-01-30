import pandas as pd
from pymilvus import (
    connections, utility, Collection, 
    CollectionSchema, FieldSchema, DataType
)

# 1. 连接 (使用底层 connections)
print("正在连接 Milvus...")
connections.connect("default", host="localhost", port="19531")

COLLECTION_NAME = "pandas_null_test_final"

# 清理
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# 2. 定义 Schema
# 必须使用 FieldSchema 这种严谨的定义方式
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2),
    
    # 【场景 A】既有默认值，又 Nullable
    # 预期：Pandas 的 None/NaN -> 触发默认值 (Milvus 特性: 默认值优先级 > Null)
    FieldSchema(name="col_default", dtype=DataType.VARCHAR, max_length=100, 
                nullable=True, default_value="MyDefault"),
    
    # 【场景 B】无默认值，只 Nullable
    # 预期：Pandas 的 None/NaN -> 存为 NULL
    FieldSchema(name="col_null", dtype=DataType.VARCHAR, max_length=100, 
                nullable=True, default_value="MyDefault")
]

schema = CollectionSchema(fields, enable_dynamic_field=True)
collection = Collection(name=COLLECTION_NAME, schema=schema)

# 创建索引
index_params = {"metric_type": "L2", "index_type": "FLAT", "params": {}}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()

print(f"集合 {COLLECTION_NAME} 准备就绪。\n")

# 3. 构造 Pandas DataFrame
print("正在构建 Pandas DataFrame...")
df = pd.DataFrame([
    # Case 1: 正常值
    {
        "id": 1, 
        "vector": [0.1, 0.1], 
        "col_default": "UserVal", 
        "col_null": "UserVal"
    },
    # Case 2: 全部给 None
    {
        "id": 2, 
        "vector": [0.2, 0.2],
        "col_default": None, 
        "col_null": None
    },
])

print("DataFrame 预览:")
# 用 where 把 NaN 显示为 None 方便查看
df = df.where(pd.notnull(df), None)
print("-" * 50)
print(df)

# 4. 插入 (必须用 collection.insert)
try:
    # Collection.insert 是支持 DataFrame 的
    collection.insert(df)
    print("✅ 插入成功！(Collection 接口成功接收了 DataFrame)")
except Exception as e:
    print(f"❌ 插入失败: {e}")
    exit()

# 刷盘确保数据可见
collection.flush()

# 5. 验证结果
print("\n=== 验证结果 (Oracle) ===")
res = collection.query(expr="id > 0", output_fields=["id", "col_default", "col_null"])
res.sort(key=lambda x: x['id'])

for row in res:
    # 辅助显示逻辑
    def format_val(val, fname):
        if val is None: return "NULL"
        if fname == "col_default" and val == "MyDefault": return f"'{val}' (Default)"
        return f"'{val}'"

    v_def = row.get("col_default")
    v_null = row.get("col_null")
    
    print(f"ID: {row['id']} | DefaultCol: {format_val(v_def, 'col_default'):<20} | NullCol: {format_val(v_null, 'col_null')}")