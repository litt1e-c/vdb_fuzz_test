import random
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# --- 1. 配置与连接 ---
HOST, PORT = '127.0.0.1', '19530'
COLLECTION_NAME = "demo_filter_blindness"
DIM = 128

# 🔥 升级点 1: 加大数据量，淹没目标
NUM_ENTITIES = 100000 

# 🔥 升级点 2: 保持 ef=limit (临界状态)，这是 HNSW 最脆弱的时候
LIMIT = 100
SEARCH_PARAMS = {"metric_type": "L2", "params": {"ef": 100}} 

print(f"🔌 Connecting to Milvus...")
connections.connect("default", host=HOST, port=PORT)
if utility.has_collection(COLLECTION_NAME): utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema("id", DataType.INT64, is_primary=True),
    FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM),
    FieldSchema("tag", DataType.INT64) 
]
schema = CollectionSchema(fields)
col = Collection(COLLECTION_NAME, schema)

print(f"📥 Inserting {NUM_ENTITIES} vectors (Batch Insert)...")

# 使用 NumPy 生成，防止慢
# A. 背景噪音: tag 随机在 1-20 之间
vectors = np.random.random((NUM_ENTITIES, DIM)).astype(np.float32)
tags = np.random.randint(1, 20, size=NUM_ENTITIES).tolist()
ids = list(range(NUM_ENTITIES))

# B. 目标陷阱: tag = 100
target_id = NUM_ENTITIES + 1
target_vec = np.random.random((1, DIM)).astype(np.float32) # 随机生成一个目标
target_tag = 100

# 插入
col.insert([ids, vectors, tags])
col.insert([[target_id], target_vec, [target_tag]]) # 单独插入目标
col.flush()

print("🏗️ Building HNSW Index (Hard Mode)...")
# 🔥 升级点 3: 降低 M 值，让图更稀疏，更难走
index_params = {
    "metric_type": "L2", 
    "index_type": "HNSW", 
    "params": {"M": 16, "efConstruction": 100} # M=4 极低，efC=50 建图草率
}
col.create_index("vector", index_params)
col.load()

# --- 3. 触发漏洞 ---
print(f"\n🔍 Executing Differential Filter Test...")

query_vec = target_vec.tolist()

# 场景 A：宽松查询 (tag > 0) -> 包含 10万条噪音
# 预期：因为 M=4，图很稀疏，ef=100 不足以覆盖所有路径，容易漏掉 Target
print(f"   👉 Query A [Loose]: expr='tag > 0'")
res_loose = col.search(query_vec, "vector", SEARCH_PARAMS, limit=LIMIT, expr="tag > 0")[0]
ids_loose = [hit.id for hit in res_loose]

# 场景 B：严格查询 (tag > 50) -> 噪音全被过滤，只剩 Target
# 预期：被迫全图扫描或走很少的路径，必中
print(f"   👉 Query B [Strict]: expr='tag > 50'")
res_strict = col.search(query_vec, "vector", SEARCH_PARAMS, limit=LIMIT, expr="tag > 50")[0]
ids_strict = [hit.id for hit in res_strict]

# --- 4. 验证 ---
print(f"\n📊 Analysis Result:")
found_in_strict = target_id in ids_strict
found_in_loose = target_id in ids_loose

print(f"   Target ID ({target_id}) in Strict Results? {found_in_strict}")
print(f"   Target ID ({target_id}) in Loose Results?  {found_in_loose}")

if found_in_strict and not found_in_loose:
    print(f"\n❌ BUG REPRODUCED: Index Blindness Detected!")
    print(f"   Milvus 在 5万数据+稀疏图(M=16) 环境下，宽松过滤漏掉了最近邻。")
else:
    print(f"\n✅ Case Passed. (HNSW is surprisingly robust today!)")