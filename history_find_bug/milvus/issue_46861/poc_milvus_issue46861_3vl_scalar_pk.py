"""

Bug Description:
  NULL values incorrectly satisfy comparison operators when combined with 
  primary key filters. This violates SQL-92 NULL semantics.

Test Case:
  Query: (id == 3588) AND (c4 <= 7.3154750146117635)
  Row 3588 has c4 = NULL
  Expected result: 0 rows (NULL should NOT satisfy <=)
  Actual result: 1 row (BUG!)

Environment:
  - Python: 3.11.5
  - PyMilvus: 2.6.5
  - Pandas: 1.5.3
  - NumPy: 1.26.4
  - Milvus Server: v2.6.7

Usage:
  python repro_from_file.py [--host 127.0.0.1] [--port 19530]
  
Requires:
  - milvus_bug_dataset_formatted.json (5000 records, 22 fields)
  - Milvus server running
"""

import json
import time
import random
import string
import numpy as np
import pandas as pd
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import sys
import pymilvus

# ================= 配置 =================
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "repro_full_structure" 
JSON_FILE = "milvus_bug_dataset_formatted.json"
SEED = 999
TARGET_ID = 3588
THRESHOLD = 7.3154750146117635
N = 5000
DIM = 128

def get_type_name(dtype):
    """Map Milvus DataType enum to string"""
    type_map = {
        DataType.BOOL: "BOOL",
        DataType.INT64: "INT64",
        DataType.DOUBLE: "DOUBLE",
        DataType.VARCHAR: "VARCHAR",
        DataType.JSON: "JSON",
        DataType.ARRAY: "ARRAY",
        DataType.FLOAT_VECTOR: "FLOAT_VECTOR"
    }
    return type_map.get(dtype, str(dtype))

class DataManager:
    """Data generation and schema management for Milvus fuzzing"""
    def __init__(self):
        self.df = None
        self.vectors = None
        self.schema_config = []
        self.null_ratio = 0.1
        self.array_capacity = 50
        self.json_max_depth = 3
        self.int_range = 180000
        self.double_scale = 10000

    KEY_POOL = [f"k_{i}" for i in range(20)] + ["user", "log", "data", "price", "config", "history", "active", "color"]

    def _gen_random_json_structure(self, rng, depth=3):
        """递归生成随机 JSON 结构"""
        if depth == 0 or rng.random() < 0.3:
            r = rng.random()
            if r < 0.2: return int(rng.integers(-100000, 100000))
            elif r < 0.4: return float(rng.random() * 1000)
            elif r < 0.6: return self._random_string(0, 8)
            elif r < 0.8: return bool(rng.choice([True, False]))
            else: return None

        if rng.random() < 0.2:
            length = rng.integers(1, 5)
            return [self._gen_random_json_structure(rng, depth - 1) for _ in range(length)]
        
        num_keys = rng.integers(1, 6)
        obj = {}
        selected_keys = rng.choice(self.KEY_POOL, size=min(num_keys, len(self.KEY_POOL)), replace=False)
        for k in selected_keys:
            key_str = str(k)
            obj[key_str] = self._gen_random_json_structure(rng, depth - 1)
        return obj

    def _random_string(self, min_len=0, max_len=10):
        """生成随机字符串"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=random.randint(min_len, max_len)))

    def generate_schema(self):
        """生成 schema，确保与 fuzzer 一致"""
        print("🎲 1. Defining Dynamic Schema...")
        self.schema_config = []
        
        num_fields = 20
        types_pool = [DataType.INT64, DataType.DOUBLE, DataType.BOOL, DataType.VARCHAR]
        
        # 固定随机种子，确保重现
        field_rng = np.random.default_rng(999)
        for i in range(num_fields):
            # 按顺序随机选择类型，与 fuzzer 保持一致
            ftype = field_rng.choice(types_pool)
            self.schema_config.append({"name": f"c{i}", "type": ftype})
        
        self.schema_config.append({"name": "meta_json", "type": DataType.JSON})
        self.schema_config.append({
            "name": "tags_array",
            "type": DataType.ARRAY,
            "element_type": DataType.INT64,
            "max_capacity": self.array_capacity
        })
        
        print(f"   -> Generated {len(self.schema_config)} dynamic fields (plus id & vector).")
        print("   -> Schema Structure:")
        for f in self.schema_config:
            t_name = get_type_name(f["type"])
            if f["type"] == DataType.ARRAY:
                ele_name = get_type_name(f["element_type"])
                print(f"      - {f['name']:<12} : {t_name}<{ele_name}>")
            else:
                print(f"      - {f['name']:<12} : {t_name}")

    def generate_data(self):
        """生成数据"""
        print(f"🌊 2. Generating {N} rows (Vector Dim={DIM})...")
        rng = np.random.default_rng(42)
        
        self.vectors = rng.random((N, DIM), dtype=np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1)[:, np.newaxis]

        data = {"id": np.arange(N, dtype=np.int64)}
        
        print("   -> Filling scalar attributes...")
        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]
            
            if ftype == DataType.INT64:
                data[fname] = rng.integers(-self.int_range, self.int_range, size=N)
            elif ftype == DataType.DOUBLE:
                data[fname] = rng.random(N) * self.double_scale
            elif ftype == DataType.BOOL:
                data[fname] = rng.choice([True, False], size=N)
            elif ftype == DataType.VARCHAR:
                data[fname] = [self._random_string(0, random.randint(5, 50)) for _ in range(N)]
            elif ftype == DataType.JSON:
                json_list = []
                for _ in range(N):
                    base_obj = {
                        "price": int(rng.integers(0, 1000)),
                        "color": rng.choice(["Red", "Blue", "Green"]),
                        "active": bool(rng.choice([True, False])),
                    }
                    if rng.random() < 0.1:
                        base_obj["price"] = None
                    if rng.random() < 0.8:
                        base_obj["config"] = {"version": int(rng.integers(1, 10))}
                    if rng.random() < 0.8:
                        base_obj["history"] = [int(x) for x in rng.integers(0, 100, size=3)]
                    
                    random_part = self._gen_random_json_structure(rng, depth=self.json_max_depth)
                    if isinstance(random_part, dict):
                        base_obj.update(random_part)
                    else:
                        base_obj["random_payload"] = random_part
                    
                    json_list.append(base_obj)
                data[fname] = json_list
            elif ftype == DataType.ARRAY:
                arr_list = []
                for _ in range(N):
                    length = random.randint(0, 5)
                    arr_list.append(list(rng.integers(0, 100, size=length)))
                data[fname] = arr_list

            mask = rng.random(N) < self.null_ratio
            temp_arr = np.array(data[fname], dtype=object)
            temp_arr[mask] = None
            data[fname] = temp_arr

        self.df = pd.DataFrame(data)
        print("✅ Data Generation Complete.")

def wait_for_index(collection_name):
    print(f"   ⏳ Waiting for index build...", end="", flush=True)
    while True:
        res = utility.index_building_progress(collection_name)
        total = res.get("total_rows", 0)
        indexed = res.get("indexed_rows", 0)
        if total > 0 and indexed >= total:
            print(" ✅ Done.")
            break
        time.sleep(1)
        print(".", end="", flush=True)

def infer_milvus_type(pd_dtype, sample_values=None):
    """根据 pandas dtype 和实际值推断 Milvus 类型"""
    if pd.api.types.is_integer_dtype(pd_dtype):
        return DataType.INT64
    elif pd.api.types.is_float_dtype(pd_dtype):
        return DataType.DOUBLE
    elif pd.api.types.is_bool_dtype(pd_dtype):
        return DataType.BOOL
    elif pd.api.types.is_object_dtype(pd_dtype):
        # 检查实际值来判断是否为 bool/dict/list
        if sample_values is not None:
            for v in sample_values:
                if v is not None:
                    if isinstance(v, bool):
                        return DataType.BOOL
                    elif isinstance(v, dict):
                        return DataType.JSON
                    elif isinstance(v, list):
                        return DataType.ARRAY
        return DataType.VARCHAR
    else:
        return DataType.VARCHAR

def main():
    # Print environment info at start
    print("\n" + "=" * 80)
    print("ENVIRONMENT INFORMATION")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyMilvus: {pymilvus.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")
    print("=" * 80 + "\n")
    
    print("=" * 80)
    print("Milvus Bug Repro: NULL Comparison Issue")
    print("=" * 80)

    connections.connect("default", host=HOST, port=PORT)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # Step 1: 读取 JSON 文件
    print(f"\n[Step 1] Loading data from '{JSON_FILE}'...")
    try:
        with open(JSON_FILE, "r") as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File '{JSON_FILE}' not found.")
        return

    print(f"   ✅ Loaded {len(full_data)} records")

    # Step 2: 转换为 DataFrame 并根据实际数据推断类型
    print(f"\n[Step 2] Converting to DataFrame and inferring schema...")
    df = pd.DataFrame(full_data)
    print(f"   DataFrame shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")

    # Step 3: 生成 Milvus Schema
    print(f"\n[Step 3] Creating Milvus Collection...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),  # 向量字段
    ]
    
    # 根据实际数据类型推断字段类型
    for col_name in df.columns:
        if col_name in ["id", "vector"]:
            continue  # 已经处理过
        
        col_dtype = df[col_name].dtype
        
        # 传递实际列值给推断函数，以正确检测类型
        milvus_type = infer_milvus_type(col_dtype, df[col_name].values)
        
        if milvus_type == DataType.JSON:
            fields.append(FieldSchema(name=col_name, dtype=DataType.JSON, nullable=True))
        elif milvus_type == DataType.ARRAY:
            fields.append(FieldSchema(
                name=col_name, dtype=DataType.ARRAY, 
                element_type=DataType.INT64, max_capacity=100,
                nullable=True
            ))
        elif milvus_type == DataType.VARCHAR:
            fields.append(FieldSchema(name=col_name, dtype=milvus_type, nullable=True, max_length=512))
        else:
            fields.append(FieldSchema(name=col_name, dtype=milvus_type, nullable=True))

    schema = CollectionSchema(fields, enable_dynamic_field=True)
    col = Collection(COLLECTION_NAME, schema)
    print(f"   ✅ Collection '{COLLECTION_NAME}' created")

    # Step 4: 插入数据
    print(f"\n[Step 4] Inserting {len(full_data)} rows...")
    # 去掉 vector 字段（如果有的话），因为 dynamic field 会处理
    insert_data = []
    for record in full_data:
        insert_record = {k: v for k, v in record.items()}
        insert_data.append(insert_record)
    
    for start in range(0, len(insert_data), 200):
        end = min(start + 200, len(insert_data))
        batch = insert_data[start:end]
        col.insert(batch)
        print(f"   Inserted {end}/{len(insert_data)}...", end="\r")
    
    col.flush()
    print(f"   ✅ Insert & Flush complete        ")

    # Step 5: 建索引
    print(f"\n[Step 5] Creating vector index...")
    if "vector" in df.columns:
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        col.create_index("vector", index_params)
        col.load()
        print(f"   ✅ Index created and collection loaded")

# ==================================================
    # 🕵️‍♀️ 插入验证环节：验尸报告
    # ==================================================
    print(f"\n[Sanity Check] Inspecting raw data for ID {TARGET_ID}...")
    # 只查 ID，不加任何标量过滤，获取最真实的存储状态
    check_res = col.query(f"id == {TARGET_ID}", output_fields=["c4"], consistency_level="Strong")
    
    if check_res:
        actual_val = check_res[0]["c4"]
        print(f"   Raw Value stored in Milvus: {actual_val}")
        print(f"   Python Type: {type(actual_val)}")
        
        if actual_val is None:
            print("   ✅ CONFIRMED: Data is truly NULL (NoneType).")
            print("      Therefore, any inequality query matching this row IS A BUG.")
        elif isinstance(actual_val, (int, float)):
            print(f"   ⚠️ WARNING: Data was converted to number: {actual_val}")
            print("      If this is a small number, the query might be technically 'correct' but data loading was wrong.")
    else:
        print("   ❌ Critical: ID not found in verification step.")
    print("=" * 80)
    # ==================================================

    # Step 6: 执行 bug 查询
    print(f"\n[Step 6] Executing bug query...")
    expr = f"(id == {TARGET_ID}) and (c4 <= {THRESHOLD})"
    print(f"   Expression: {expr}")
    
    try:
        res = col.query(expr, output_fields=["id", "c4"], consistency_level="Strong")
        
        print(f"\n[Step 7] Result Analysis:")
        print(f"   Returned rows: {len(res)}")
        
        if len(res) > 0:
            row = res[0]
            print(f"\n" + "=" * 80)
            print(f"❌ BUG REPRODUCED!")
            print("=" * 80)
            print(f"\n   Row data:")
            print(f"      id={row['id']}")
            print(f"      c4={row['c4']}")
            print(f"\n   Expected: 0 rows (NULL should NOT satisfy <=)")
            print(f"   Actual:   {len(res)} row(s)")
            print(f"\n   Root Cause: c4=NULL is incorrectly treated as satisfying")
            print(f"              the comparison (c4 <= {THRESHOLD})")
            print(f"\n   This violates SQL-92 NULL semantics.")
            print(f"   Milvus should return UNKNOWN (False) for NULL comparisons.")
            print("=" * 80 + "\n")
        else:
            print(f"   ✅ No rows returned (correct behavior)")
    except Exception as e:
        print(f"   ❌ Query error: {e}")
        print(f"   (This may indicate c4 has a different type in the JSON file)")
if __name__ == "__main__":
    main()