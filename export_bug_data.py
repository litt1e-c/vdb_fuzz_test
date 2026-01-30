import json
import numpy as np
from pymilvus import connections, Collection

# --- 配置 ---
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "fuzz_stable_v3"  # 你的 Fuzzer 表名
EXPORT_FILE = "milvus_full_data.json"

# --- JSON 编码器：增强版 ---
class MilvusDataEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理 Numpy 数组
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 处理 Numpy 标量
        if isinstance(obj, np.generic):
            return obj.item()
        # 🔔 修复点：处理 Protobuf 的 RepeatedScalarContainer (针对 ARRAY 类型)
        # 通过类名判断，避免引入额外依赖
        if "RepeatedScalarContainer" in type(obj).__name__:
            return list(obj)
        return super().default(obj)

def export_all_data():
    print(f"🔌 Connecting to {HOST}:{PORT}...")
    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    print(f"📥 Loading collection '{COLLECTION_NAME}'...")
    try:
        col = Collection(COLLECTION_NAME)
        col.load()
    except Exception as e:
        print(f"❌ Collection not found: {e}")
        return

    print("🔍 Fetching ALL data (limit=10000)...")
    
    # 使用迭代器或大 Limit 获取全量数据
    # 这里直接用 query 拿 5000 条没问题
    try:
        # id >= -1 是一个恒真条件，用于全表扫描
        res = col.query(expr="id > -1", output_fields=["*"], limit=10000)
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return

    print(f"✅ Retrieved {len(res)} rows. Serializing to JSON...")

    try:
        with open(EXPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(res, f, cls=MilvusDataEncoder, indent=None) # indent=None 减小文件体积
        
        print("\n" + "="*50)
        print(f"🎉 Success! Full dataset saved to: {EXPORT_FILE}")
        print(f"   Total Count: {len(res)}")
        print(f"   File size will be large, perfect for reproducing volume-dependent bugs.")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

if __name__ == "__main__":
    export_all_data()