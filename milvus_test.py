import random
import numpy as np
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

# ==========================================
# 1. 连接 Milvus (修改点)
# ==========================================
# 你的 Docker Compose 配置中将容器的 19530 映射到了宿主机的 19531
print("正在连接 Docker 部署的 Milvus Standalone...")
try:
    connections.connect("default", host="127.0.0.1", port="19531")
    print("连接成功！")
except Exception as e:
    print(f"连接失败: {e}")
    print("请检查 Docker 容器是否已启动 (docker ps)，以及端口 19531 是否开放。")
    exit(1)

# ==========================================
# 2. 定义包含“丰富标量”的 Schema
# ==========================================
dim = 8  # 向量维度

fields = [
    # 主键
    FieldSchema(name="pk_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    # 向量
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),
    
    # --- 基础标量 ---
    FieldSchema(name="age", dtype=DataType.INT64),             # 整数
    FieldSchema(name="price", dtype=DataType.FLOAT),           # 浮点
    FieldSchema(name="is_active", dtype=DataType.BOOL),        # 布尔
    FieldSchema(name="city_name", dtype=DataType.VARCHAR, max_length=200), # 字符串
    
    # --- 高级标量 (Milvus 2.6+ 支持) ---
    # JSON 字段：非常适合测试复杂嵌套逻辑
    FieldSchema(name="user_meta", dtype=DataType.JSON), 
    # Array 字段：字符串列表，适合测试 IN, CONTAINS 等逻辑
    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=50) 
]

schema = CollectionSchema(fields, "用于自动化测试的丰富标量集合")
col_name = "scalar_fuzz_test_docker"

# 清理旧数据 (方便反复测试)
if utility.has_collection(col_name):
    print(f"检测到旧集合 {col_name}，正在删除...")
    utility.drop_collection(col_name)

collection = Collection(col_name, schema)
print(f"集合 {col_name} 创建成功。")

# ==========================================
# 3. 插入模拟数据
# ==========================================
print("正在生成并插入数据...")
num_entities = 20  # 数据量不用多，够测试就行

data_rows = []
cities = ["Beijing", "Shanghai", "New York", "London", "Tokyo"]
tags_pool = ["tech", "food", "travel", "music", "sport"]

for i in range(num_entities):
    row = {
        "pk_id": i,
        "embeddings": np.random.random(dim).tolist(),
        "age": random.randint(18, 60),
        "price": round(random.uniform(10.0, 100.0), 2),
        "is_active": random.choice([True, False]),
        "city_name": random.choice(cities),
        # JSON 数据
        "user_meta": {"level": random.randint(1, 5), "login_cnt": random.randint(0, 100)},
        # Array 数据 (随机取 1-3 个标签)
        "tags": random.sample(tags_pool, k=random.randint(1, 3))
    }
    data_rows.append(row)

# 插入
collection.insert(data_rows)
print("数据插入完成。")

# 建索引 (Milvus 2.x 必须建索引并 load 才能 query)
# 注意：数据量很小时，FLAT 索引最快且无需训练
index_params = {"metric_type": "L2", "index_type": "FLAT", "params": {}}
collection.create_index("embeddings", index_params)
print("索引构建完成。")

# 加载集合
collection.load()
print("集合已加载到内存，准备进行 API 内省测试。")

# ==========================================
# 4. 【核心】API 获取标量名称、类型和值
# ==========================================

# 辅助：将 DataType 数字转换为可读字符串的映射
TYPE_MAP = {
    DataType.INT64: "INT64",
    DataType.FLOAT: "FLOAT",
    DataType.BOOL: "BOOL",
    DataType.VARCHAR: "VARCHAR",
    DataType.JSON: "JSON",
    DataType.ARRAY: "ARRAY",
    DataType.FLOAT_VECTOR: "VECTOR"
}

def analyze_collection_scalars(target_collection):
    print("\n" + "="*50)
    print(f"开始分析集合: {target_collection.name}")
    print("="*50)
    
    # 步骤 A: 获取 Schema 信息 (名称和类型)
    # ------------------------------------------------
    scalar_field_names = [] 
    field_info_map = {}     
    
    print(f"{'字段名':<15} | {'类型':<10} | {'属性'}")
    print("-" * 50)
    
    for field in target_collection.schema.fields:
        type_str = TYPE_MAP.get(field.dtype, "UNKNOWN")
        props = []
        if field.is_primary: props.append("PK")
        if field.dtype == DataType.FLOAT_VECTOR: props.append("VECTOR")
        
        print(f"{field.name:<15} | {type_str:<10} | {', '.join(props)}")
        
        # 收集非向量标量用于测试
        if not field.is_primary and field.dtype != DataType.FLOAT_VECTOR:
            scalar_field_names.append(field.name)
            field_info_map[field.name] = field.dtype

    print("-" * 50)
    
    # 步骤 B: 获取数据值 (采样)
    # ------------------------------------------------
    print("\n[API] 正在通过 query 接口采样数据...")
    
    # 构造查询：expr="" 或 "id > -1" 视版本而定，Milvus 2.3+ 支持空 expr
    # 如果报错，尝试将 expr 改为 "pk_id > -1"
    try:
        results = target_collection.query(
            expr="", 
            output_fields=scalar_field_names,
            limit=5
        )
    except Exception:
        # Fallback for some versions
        results = target_collection.query(
            expr="pk_id > -1", 
            output_fields=scalar_field_names,
            limit=5
        )
    
    print(f"采样成功，获取到 {len(results)} 条记录。")
    print(f"示例数据 (第一条): {results[0]}")
        
    # 步骤 C: 提取每一列的值 (模拟 SQLancer 的 Value Pool)
    # ------------------------------------------------
    print("\n[Value Pool] 提取的有效标量值:")
    for fname in scalar_field_names:
        # 从结果中提取
        values = [row[fname] for row in results]
        # 去重打印
        print(f" -> Field: {fname:<12} | Values: {str(values)[:80]}...")

# 执行分析
analyze_collection_scalars(collection)