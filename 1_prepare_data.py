import time
import numpy as np
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, 
    Collection, utility
)

HOST = '127.0.0.1'
PORT = '19530'
COLLECTION_NAME = "crash_test_empty_v2523"
DIM = 128
TOTAL_COUNT = 500_000
BATCH_SIZE = 50_000

def prepare_data():
    print(f"Connecting to Milvus {HOST}...")
    connections.connect("default", host=HOST, port=PORT)

    # 1. 自动清理旧数据（这就是不需要重启数据库的原因）
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"Dropped old collection {COLLECTION_NAME}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="group_id", dtype=DataType.INT64),
        FieldSchema(name="flag", dtype=DataType.INT8),
        FieldSchema(name="json_data", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields, "Crash Test Schema")
    collection = Collection(COLLECTION_NAME, schema)
    print("Collection created.")

    t0 = time.time()
    # 插入 500万条 flag=0 的数据
    for i in range(0, TOTAL_COUNT, BATCH_SIZE):
        ids = [k for k in range(i, i + BATCH_SIZE)]
        vectors = np.random.random((BATCH_SIZE, DIM)).astype(np.float32)
        group_ids = np.random.randint(0, 1000, size=BATCH_SIZE).tolist()
        flags = np.zeros(BATCH_SIZE, dtype=np.int8).tolist() # 全是 0
        json_data = [{"tag": "test"} for _ in range(BATCH_SIZE)]

        collection.insert([ids, vectors, group_ids, flags, json_data])
        print(f"Inserted {i + BATCH_SIZE} / {TOTAL_COUNT}...")

    # --- 新增部分：插入唯一的幸存者 ---
    print(">>> Inserting the LONE SURVIVOR (flag=1)...")
    collection.insert([
        [TOTAL_COUNT + 1],    # ID
        [np.random.random(DIM).astype(np.float32)], # Vector (注意 numpy维度)
        [999],                # Group ID
        [1],                  # Flag = 1 <--- 关键！
        [{"tag": "survivor"}] 
    ])
    # -------------------------------

    print(f"Insertion done in {time.time()-t0:.2f}s")

    print("Building Index (HNSW)...")
    # 修改为 HNSW 索引配置
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",  # <--- 变成了 HNSW
        "params": {
            "M": 32,           # 图的最大连接数 (常见值 16~64)
            "efConstruction": 200 # 建图精度
        }
    }
    collection.create_index("vector", index_params)
    
    collection.load()
    print("Collection Loaded. Ready for attack.")

if __name__ == "__main__":
    prepare_data()