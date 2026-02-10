"""
Milvus Dynamic Fuzzing System (Phase 3: High Load Stable Version)
Features:
1. Dynamic Schema: Random 5-15 fields + JSON + ARRAY.
2. High Load Data: 100k Vectors * 768 Dim.
3. Stable Insert: Low batch size + Throttling + Frequent Flush to prevent Server OOM.
4. Robust Query: Handles Bool/String types correctly.
"""
import time
import random
import string
import numpy as np
import pandas as pd
import json
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)
import sys
import argparse

# --- Configuration (User Specified) ---
HOST = "127.0.0.1"
PORT = "19531"           # 你的自定义端口
COLLECTION_NAME = "fuzz_stable_v3"
N = 5000                 # 数据量（磁盘满了，先用 1000 测试功能）
DIM = 128                # 维度 128
BATCH_SIZE = 200         # 批次大小（内存模式可以大一些）
SLEEP_INTERVAL = 0.01    # 每次插入后暂停 10ms（内存模式更快）
FLUSH_INTERVAL = 500     # 每 500 条刷盘

# 稳定的索引类型列表（移除不稳定或需要特殊配置的索引）
ALL_INDEX_TYPES = [
    "FLAT", "HNSW", "IVF_FLAT", "IVF_SQ8", "IVF_PQ"
]
# 注意：INDEX_TYPE 等随机变量移到 run() 内部，在种子设置后初始化，保证可重复性
INDEX_TYPE = None  # 延迟初始化

# 全局度量类型列表（浮点向量支持的全部 metric）
# L2: 欧氏距离 (越小越相似), IP: 内积 (越大越相似), COSINE: 余弦相似度 (越大越相似)
ALL_METRIC_TYPES = ["L2", "IP", "COSINE"]
METRIC_TYPE = None  # 延迟初始化（在 run() 内种子设置后随机选取）

# 记录当前索引类型（用于索引重建时避免重复）
CURRENT_INDEX_TYPE = None  # 延迟初始化
VECTOR_CHECK_RATIO = None  # 延迟初始化
VECTOR_TOPK = None         # 延迟初始化

# 混淆开关（默认关闭）。当 >0 时，在 JSON 下钻策略中按该概率触发类型混淆
CHAOS_RATE = 0.0

# --- 1. Data Manager ---

def get_type_name(dtype):
    """Map Milvus DataType enum to string"""
    type_map = {
        DataType.BOOL: "BOOL",
        DataType.INT8: "INT8",
        DataType.INT16: "INT16",
        DataType.INT32: "INT32",
        DataType.INT64: "INT64",
        DataType.FLOAT: "FLOAT",
        DataType.DOUBLE: "DOUBLE",
        DataType.VARCHAR: "VARCHAR",
        DataType.JSON: "JSON",
        DataType.ARRAY: "ARRAY",
        DataType.FLOAT_VECTOR: "FLOAT_VECTOR"
    }
    return type_map.get(dtype, str(dtype))

# 全局 ID 计数器（避免使用 time.time() 导致不可重复）
_GLOBAL_ID_COUNTER = 0

def generate_unique_id():
    """生成唯一 ID（基于计数器 + 随机数，保证可重复性）"""
    global _GLOBAL_ID_COUNTER
    _GLOBAL_ID_COUNTER += 1
    return 10000000 + _GLOBAL_ID_COUNTER * 1000 + random.randint(1, 999)

class DataManager:
    # 用于动态生成唯一 ID 的计数器（避免使用 time.time() 导致不可重复）
    _id_counter = 0
    
    def generate_single_row(self, id_override=None):
        """生成一行与schema一致的新数据（含唯一id），与generate_data逻辑保持一致"""
        # 使用 np.random 的全局状态（已被 np.random.seed() 设置），保证可重复
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        row = {}
        # 生成唯一id（使用计数器 + 随机数，避免 time.time() 导致不可重复）
        if id_override is not None:
            row["id"] = int(id_override)
        else:
            DataManager._id_counter += 1
            row["id"] = 10000000 + DataManager._id_counter * 1000 + random.randint(1, 999)
        
        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]
            
            if ftype == DataType.INT64:
                row[fname] = int(rng.integers(-self.int_range, self.int_range))
            elif ftype == DataType.DOUBLE:
                row[fname] = float(rng.random() * self.double_scale)
            elif ftype == DataType.BOOL:
                row[fname] = bool(rng.choice([True, False]))
            elif ftype == DataType.VARCHAR:
                row[fname] = self._random_string(0, random.randint(5, 50))
            elif ftype == DataType.JSON:
                # 与generate_data中的JSON结构保持一致
                base_obj = {
                    "price": int(rng.integers(0, 1000)),
                    "color": rng.choice(["Red", "Blue", "Green"]),
                    "active": bool(rng.choice([True, False])),
                }
                if rng.random() < 0.8:
                    base_obj["config"] = {"version": int(rng.integers(1, 10))}
                if rng.random() < 0.8:
                    base_obj["history"] = [int(x) for x in rng.integers(0, 100, size=3)]
                random_part = self._gen_random_json_structure(rng, depth=self.json_max_depth)
                if isinstance(random_part, dict):
                    base_obj.update(random_part)
                else:
                    base_obj["random_payload"] = random_part
                row[fname] = base_obj
            elif ftype == DataType.ARRAY:
                # 与schema中的array_capacity一致
                arr_len = rng.integers(1, min(self.array_capacity, 10) + 1)
                row[fname] = [int(x) for x in rng.integers(0, 100, size=arr_len)]
        return row

    def generate_single_vector(self):
        """生成一条单位化的向量"""
        vec = np.random.randn(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    def __init__(self):
        self.df = None
        self.vectors = None
        self.schema_config = []
        self.null_ratio = random.uniform(0.05, 0.15)
        self.array_capacity = random.randint(5, 50)
        self.json_max_depth = random.randint(1, 5)
        self.int_range = random.randint(5000, 100000)
        self.double_scale = random.uniform(100, 10000)

    KEY_POOL = [f"k_{i}" for i in range(20)] + ["user", "log", "data", "a b", "test.key"]

    def _gen_random_json_structure(self, rng, depth=3):
        """递归生成随机 JSON 结构"""
        if depth == 0 or rng.random() < 0.3:
            r = rng.random()
            if r < 0.2: return int(rng.integers(-100000, 100000))
            elif r < 0.4: return float(rng.random() * 1000)
            elif r < 0.6: return self._random_string(0, 8)
            # changed delete null generation
            #elif r < 0.8: return bool(rng.choice([True, False]))
            #else: return None
            else: return bool(rng.choice([True, False]))

        if rng.random() < 0.2:
            length = rng.integers(1, 5)
            return [self._gen_random_json_structure(rng, depth - 1) for _ in range(length)]

        num_keys = rng.integers(1, 6)
        obj = {}
        selected_keys = rng.choice(self.KEY_POOL, size=num_keys, replace=False)
        for k in selected_keys:
            key_str = str(k)
            obj[key_str] = self._gen_random_json_structure(rng, depth - 1)
        return obj

    def _random_string(self, min_len=0, max_len=10):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=random.randint(min_len, max_len)))

    def generate_schema(self):
        print("🎲 1. Defining Dynamic Schema...")
        self.schema_config = []
        num_fields = random.randint(3, 20)
        types_pool = [DataType.INT64, DataType.DOUBLE, DataType.BOOL, DataType.VARCHAR]

        for i in range(num_fields):
            ftype = random.choice(types_pool)
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
                    ### changed delete null generation
                    # if rng.random() < 0.1:
                    #     base_obj["price"] = None
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

            ### changed delete null generation
            #mask = rng.random(N) < self.null_ratio
            #temp_arr = np.array(data[fname], dtype=object)
            #temp_arr[mask] = None
            #data[fname] = temp_arr

        self.df = pd.DataFrame(data)
        print("✅ Data Generation Complete.")

# --- 2. Milvus Manager (Stable Insert Mode) ---

class MilvusManager:
    def __init__(self):
        self.col = None

    def connect(self):
        print(f"🔌 Connecting to Milvus at {HOST}:{PORT}...")
        try:
            # Increase connection timeout
            connections.connect("default", host=HOST, port=PORT, timeout=30)
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("   (Please check if Docker container is running)")
            exit(1)

    def reset_collection(self, schema_config):
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        ]

        for fc in schema_config:
            if fc["type"] == DataType.ARRAY:
                fields.append(FieldSchema(
                    name=fc["name"], dtype=DataType.ARRAY,
                    element_type=fc["element_type"], max_capacity=fc["max_capacity"],
                    nullable=True
                ))
            else:
                fields.append(FieldSchema(name=fc["name"], dtype=fc["type"], nullable=True, max_length=512))

        schema = CollectionSchema(fields, enable_dynamic_field=True)
        self.col = Collection(COLLECTION_NAME, schema)
        print("🛠️ Collection Created.")

    def insert(self, dm):
        print(f"⚡ 3. Inserting Data (Batch={BATCH_SIZE}, Sleep={SLEEP_INTERVAL}s)...")
        records = dm.df.to_dict(orient="records")
        total = len(records)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_data = records[start:end]

            # Prepare rows
            insert_rows = []
            for i, row in enumerate(batch_data):
                row_with_vec = row.copy()
                row_with_vec["vector"] = dm.vectors[start + i].tolist()

                # Sanitize numpy types
                for k, v in row_with_vec.items():
                    if hasattr(v, "item"):
                        row_with_vec[k] = v.item()

                insert_rows.append(row_with_vec)

            # --- Retry Logic for Stability ---
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.col.insert(insert_rows)
                    break # Success
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"\n❌ Critical Error at batch {start}-{end}: {e}")
                        raise e # Stop if all retries fail

                    print(f"   ⚠️ Retry {attempt+1}/{max_retries} due to error...", end="\r")
                    time.sleep(2) # Wait longer before retry

                    # Reconnect attempt
                    try:
                        connections.connect("default", host=HOST, port=PORT, timeout=10)
                    except:
                        pass

            # --- Progress & Throttling ---
            print(f"   Inserted {end}/{total}...", end="\r")

            # 1. Throttling: Sleep to let server CPU breathe
            time.sleep(SLEEP_INTERVAL)

            # 2. Frequent Flush: Free up server memory
            if end % FLUSH_INTERVAL == 0:
                try:
                    self.col.flush()
                except:
                    pass

        print("\n✅ Insert Complete. Building Index...")
        self.col.flush()

        index_params = {}

        if INDEX_TYPE == "FLAT":
            print("🔨 Building FLAT Index (Exact Search)...")
            index_params = {
                "metric_type": METRIC_TYPE,
                "index_type": "FLAT",
                "params": {}
            }

        elif INDEX_TYPE == "HNSW":
            print("🔨 Building HNSW Index (Graph Based)...")
            index_params = {
                "metric_type": METRIC_TYPE,
                "index_type": "HNSW",
                "params": {
                    "M": 32,              # 图的最大连接数
                    "efConstruction": 256 # 构建时的搜索深度
                }
            }

        elif INDEX_TYPE == "IVF_FLAT":
            print("🔨 Building IVF_FLAT Index...")
            index_params = {
                "metric_type": METRIC_TYPE,
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

        elif INDEX_TYPE == "IVF_SQ8":
            print("🔨 Building IVF_SQ8 Index...")
            index_params = {
                "metric_type": METRIC_TYPE,
                "index_type": "IVF_SQ8",
                "params": {"nlist": 128}
            }

        elif INDEX_TYPE == "IVF_PQ":
            print("🔨 Building IVF_PQ Index...")
            index_params = {
                "metric_type": METRIC_TYPE,
                "index_type": "IVF_PQ",
                "params": {"nlist": 128, "m": 8, "nbits": 8}
            }

        try:
            # 创建索引
            self.col.create_index("vector", index_params)

            # 加载数据 (必须在建索引后)
            print("📥 Loading collection into memory...")
            self.col.load()

        except Exception as e:
            print(f"❌ Index build failed (Likely OOM or Config Error): {e}")
            exit(1)

# --- 3. Robust Query Generator ---

class OracleQueryGenerator:
    def __init__(self, dm):
        self.schema = dm.schema_config
        self._dm = dm  # 保存 dm 引用，而不是 df 引用
    
    @property
    def df(self):
        """动态获取最新的 DataFrame"""
        return self._dm.df

    def _random_string(self, min_len=5, max_len=10):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=random.randint(min_len, max_len)))

    def _get_json_val(self, obj, keys):
        try:
            if obj is None: return None
            val = obj
            for k in keys:
                if val is None: return None
                # 处理数组下标
                if isinstance(k, int):
                    if isinstance(val, list) and len(val) > k: val = val[k]
                    else: return None
                # 处理字典 Key
                else:
                    if isinstance(val, dict) and k in val: val = val[k]
                    else: return None
            return val
        except: return None


    # 新增高级 JSON 生成器
    def gen_json_advanced_expr(self):
        """
        专门生成高级 JSON 查询
        """
        json_fields = [f for f in self.schema if f["type"] == DataType.JSON]
        if not json_fields:
            # 兜底表达式，永远为真
            return ("id > 0", None)
        field = random.choice(json_fields); name = field["name"]; series = self.df[name]
        strategy = random.choice(["range", "nested", "index", "multi_key"])

        # --- 策略 1: Range ---
        if strategy == "range":
            low = random.randint(100, 500); high = low + random.randint(50, 200)
            expr = f'({name}["price"] > {low} and {name}["price"] < {high})'

            # 【修复】使用默认参数绑定，避免闭包捕获问题
            def check_range(x, _low=low, _high=high):
                try:
                    v = self._get_json_val(x, ["price"])
                    if v is None: return False
                    if isinstance(v, bool): return False
                    return (isinstance(v, (int, float)) and v > _low and v < _high)
                except: return False

            return (expr, series.apply(check_range))

        # --- 策略 2: Nested ---
        elif strategy == "nested":
            val = random.randint(1, 9)
            expr = f'{name}["config"]["version"] == {val}'

            # 【修复】使用默认参数绑定
            def check_nested(x, _val=val):
                try:
                    v = self._get_json_val(x, ["config", "version"])
                    if v is None: return False
                    if isinstance(v, bool): return False
                    return v == _val
                except: return False

            return (expr, series.apply(check_nested))

        # --- 策略 3: Index ---
        elif strategy == "index":
            idx = 0; val = random.randint(20, 80)
            expr = f'{name}["history"][{idx}] > {val}'

            # 【修复】使用默认参数绑定
            def check_index(x, _idx=idx, _val=val):
                try:
                    v = self._get_json_val(x, ["history", _idx])
                    if v is None: return False
                    if isinstance(v, bool): return False
                    return (isinstance(v, (int, float)) and v > _val)
                except: return False

            return (expr, series.apply(check_index))

        # --- 策略 4: Multi-key ---
        elif strategy == "multi_key":
            color = random.choice(["Red", "Blue"])
            expr = f'({name}["active"] == true and {name}["color"] == "{color}")'

            # 【修复】使用默认参数绑定
            def check_multi(x, _color=color):
                try:
                    active = self._get_json_val(x, ["active"])
                    col_val = self._get_json_val(x, ["color"])
                    if active is not True: return False
                    return col_val == _color
                except: return False

            return (expr, series.apply(check_multi))

        # 兜底表达式，永远为真
        return ("id > 0", None)
    def get_value_for_query(self, fname, ftype):
        """
        获取一个用于查询的值，有 10% 的概率返回该类型的一个“不存在的值”。
        如果返回 None，表示该字段适合生成 IS NULL/IS NOT NULL 查询。
        """
        # 1. 尝试从真实数据中采样一个值 (90% 概率)
        valid_series = self.df[fname].dropna()
        if not valid_series.empty and random.random() < 0.8: # 90% 概率从真实数据采样,这里先暂时改成0.8进行测试
            val = random.choice(valid_series.values)
            if hasattr(val, "item"): val = val.item()
            return val


        # 2. 10% 概率生成一个“不存在”的值
        # Milvus 没有查询值的 API，所以我们只能通过生成远离现有值的方式
        # 确保这个值在现有数据中极大概率不存在

        if ftype == DataType.INT64:
            # 采样一个现有值的范围，然后生成一个超出范围的值
            if not valid_series.empty:
                min_val, max_val = int(valid_series.min()), int(valid_series.max())
                # 生成一个比最大值大很多，或者比最小值小很多的值
                return random.choice([max_val + 100000, min_val - 100000])
            return random.randint(-200000, 200000) # 极端值

        elif ftype == DataType.DOUBLE:
            if not valid_series.empty:
                min_val, max_val = float(valid_series.min()), float(valid_series.max())
                val = random.choice([max_val + 100000.0, min_val - 100000.0])
            else:
                val = random.random() * 200000.0

            # 【修复点 2】生成的随机浮点数也要做精度截断
            return float(np.float32(val))

        elif ftype == DataType.BOOL:
            # Bool 只有 True/False，无法生成“不存在”值，只能采样
            # 这种情况下，10% 概率会 fallback 到 None，由 gen_atomic_expr 处理
            return random.choice([True, False])

        elif ftype == DataType.VARCHAR:
            # 生成一个随机的、很长的、或者使用特殊字符的字符串，使其不太可能存在
            chars = string.ascii_letters + string.digits + "!@#$%^&*"
            length = random.randint(15, 30) # 更长，更随机
            return ''.join(random.choices(chars, k=length))

        elif ftype == DataType.JSON:
            # 随机生成一个不太可能存在的 JSON key或value
            new_key = self._random_string(10, 15) # 不存在的 key
            new_val = self._random_string(10, 15) # 不存在的 value
            return {"_non_exist_key": new_key, "_non_exist_val": new_val} # 随便返回一个不可能存在的结构

        elif ftype == DataType.ARRAY:
            # 随机生成一个不在任何数组中的数字
            return random.randint(50000, 100000) # 极大概率不在 0-100 的范围

        return None # 无法生成时返回 None，由上层处理 Null


    def gen_atomic_expr(self):
        f = random.choice(self.schema)
        name, ftype = f["name"], f["type"]
        series = self.df[name]

        # 1. Null Check (保持不变 - 这是绝对安全的)
        if random.random() < 0.15:
            if random.random() < 0.5: return (f"{name} is null", series.isnull())
            else: return (f"{name} is not null", series.notnull())

        # 获取查询值
        val = self.get_value_for_query(name, ftype)
        if val is None:
            # 兜底表达式，永远为真
            return ("id > 0", series.notnull())

        mask = None
        expr = ""

        # --- 🛡️ 标量安全比较 (SQL 标准：Null 比较永远为 False) ---
        def safe_compare_scalar(op, target_val):
            def comp(x):
                # 探测结果证明：Scalar Null 永远不等于/不大于/不小于 任何值
                if x is None: return False
                # Pandas 的 Float Series 可能包含 NaN
                if isinstance(x, float) and np.isnan(x): return False
                try:
                    if op == "==": return x == target_val
                    if op == "!=": return x != target_val
                    if op == ">": return x > target_val
                    if op == "<": return x < target_val
                    if op == ">=": return x >= target_val
                    if op == "<=": return x <= target_val
                    return False
                except:
                    return False
            return comp

        # 2. Value Comparison (分流处理)
        if ftype == DataType.BOOL:
            val_bool = bool(val)
            expr = f"{name} == {str(val_bool).lower()}"
            mask = series.apply(safe_compare_scalar("==", val_bool))

        elif ftype == DataType.INT64:
            val_int = int(val)
            op = random.choice([">", "<", "==", "!=", ">=", "<="])
            expr = f"{name} {op} {val_int}"
            mask = series.apply(safe_compare_scalar(op, val_int))

        elif ftype == DataType.DOUBLE:
            val_float = float(val)
            # 🚨 关键回退：移除 "==", "!=" 以避免 IEEE 754 精度误报
            op = random.choice([">", "<", ">=", "<="])
            expr = f"{name} {op} {val_float}"
            mask = series.apply(safe_compare_scalar(op, val_float))

        elif ftype == DataType.VARCHAR:
            op = random.choice(["==", "!=", ">", "<", "like"])
            if op == "like":
                raw_p = val[0] if val else 'a'
                p = 'a' if raw_p in ['%', '_'] else raw_p
                expr = f'{name} like "{p}%"'
                mask = series.notnull() & series.astype(str).str.startswith(p)
            else:
                expr = f'{name} {op} "{val}"'
                mask = series.apply(safe_compare_scalar(op, val))

        elif ftype == DataType.JSON:

            # --- 策略 A: Exists 查询 ---
            if random.random() < 0.2:
                tk = "price" if random.random() < 0.5 else "non_exist"
                expr = f'exists({name}["{tk}"])'
                # 【修复】使用默认参数绑定
                def check_exists(x, _tk=tk):
                    if not isinstance(x, dict): return False
                    # 探测结果 V1: exists 排除了显式 Null 值
                    return _tk in x and x[_tk] is not None
                mask = series.apply(check_exists)

            # --- 策略 B: json_contains 查询 ---
            elif random.random() < 0.2 and isinstance(val, dict):
                list_keys = [k for k, v in val.items() if isinstance(v, list) and len(v) > 0]
                if list_keys:
                    k = random.choice(list_keys)
                    target_list = val[k]
                    target_item = random.choice(target_list)
                    if target_item is not None and isinstance(target_item, (int, float, str, bool)):
                        item_str = json.dumps(target_item)
                        if isinstance(target_item, bool):
                            item_str = item_str.lower()
                        expr = f'json_contains({name}["{k}"], {item_str})'
                        # 【修复】使用默认参数绑定
                        def check_list_contains(x, _k=k, _target_item=target_item):
                            if not isinstance(x, dict): return False
                            if _k not in x or not isinstance(x[_k], list): return False
                            return _target_item in x[_k]
                        mask = series.apply(check_list_contains)
                    else: return (f"{name} is not null", series.notnull())
                else: return (f"{name} is not null", series.notnull())

            # --- 策略 C: 下钻 (Drill-down) ---
            else:
                target_val = val
                path_keys = []
                depth = 3
                while isinstance(target_val, dict) and target_val and depth > 0:
                    k = random.choice(list(target_val.keys()))
                    path_keys.append(k)
                    target_val = target_val[k]
                    depth -= 1

                if isinstance(target_val, list) and target_val:
                    idx = random.randint(0, len(target_val)-1)
                    path_keys.append(idx)
                    target_val = target_val[idx]

                is_chaos = (CHAOS_RATE > 0) and (random.random() < CHAOS_RATE)
                query_val = target_val

                if is_chaos or not isinstance(target_val, (int, float, str, bool)):
                    if isinstance(target_val, int): query_val = "chaos_str"
                    elif isinstance(target_val, str): query_val = 99999
                    else: query_val = 1

                if isinstance(query_val, (int, float, str, bool)):
                    path_str = "".join([f'["{p}"]' if isinstance(p, str) else f'[{p}]' for p in path_keys])

                    if isinstance(query_val, bool):
                        op = random.choice(["==", "!="])
                    else:
                        op = random.choice(["==", "!=", ">", "<", ">=", "<="])

                    if isinstance(query_val, bool): val_str = str(query_val).lower()
                    elif isinstance(query_val, str): val_str = f'"{query_val}"'
                    else: val_str = str(query_val)

                    expr = f'{name}{path_str} {op} {val_str}'

                    # --- 🛡️ JSON 安全比较 ---
                    # 【关键修复】使用默认参数绑定，避免闭包捕获问题
                    def safe_check_json(x, _path_keys=path_keys, _op=op, _query_val=query_val):
                        try:
                            v = self._get_json_val(x, _path_keys)

                            # (A) 基础判空
                            # 探测结果 V1: Missing/Null != Value -> True
                            # 探测结果 V2: Missing/Null > Value -> False
                            if v is None:
                                if _op == "!=": return True
                                return False

                            is_v_num = isinstance(v, (int, float)) and not isinstance(v, bool)
                            is_q_num = isinstance(_query_val, (int, float)) and not isinstance(_query_val, bool)

                            if is_v_num and is_q_num:
                                if _op == "==": return v == _query_val
                                if _op == "!=": return v != _query_val
                                if _op == ">": return v > _query_val
                                if _op == "<": return v < _query_val
                                if _op == ">=": return v >= _query_val
                                if _op == "<=": return v <= _query_val
                                return False

                            if type(v) != type(_query_val):
                                if _op == "!=": return True
                                return False

                            if _op == "==": return v == _query_val
                            if _op == "!=": return v != _query_val
                            if _op == ">": return v > _query_val
                            if _op == "<": return v < _query_val
                            if _op == ">=": return v >= _query_val
                            if _op == "<=": return v <= _query_val
                            return False
                        except:
                            return False

                    mask = series.apply(safe_check_json)
                else:
                    return (f"{name} is not null", series.notnull())

        if mask is not None:
            # 🚨 最终返回规则：
            # 1. JSON 类型：不做 notnull 过滤 (因为 != 包含 None)
            if ftype == DataType.JSON:
                return (expr, mask)

            # 2. 标量类型：必须做 notnull 过滤
            return (expr, mask & series.notnull())

        return ("", None)

    def gen_constant_expr(self):
        """
        生成常量表达式。

        【约束条件 - 已通过实验验证】
        Milvus 对常数表达式的支持极其有限。探测结果（test_constant_expr.py）：

        ✅ 安全（2/26 通过）：
        - null is null          ← 恒真，安全
        - null is not null      ← 恒假，安全

        ❌ 不安全（全部崩溃）：
        - 纯数值常数: 1==1, 1.5>2.1, 等       → PhyFilterBitsNode ColumnVector 类型错误
        - 纯字符串常数: "abc"=="abc"，等      → PhyFilterBitsNode ColumnVector 类型错误
        - 混合类型: "123"==123, true==1, 等   → 表达式解析失败（类型检查拒绝）

        因此本函数仅生成 NULL 相关的常数表达式。
        """
        # 创建全 True/False 的 mask
        true_mask = pd.Series(True, index=self.df.index)
        false_mask = pd.Series(False, index=self.df.index)

        strategy = random.choice(["null_is_null", "null_is_not_null"])

        if strategy == "null_is_null":
            # NULL is null (恒真、安全)
            expr = "null is null"
            return (expr, true_mask)
        else:
            # NULL is not null (恒假、安全)
            expr = "null is not null"
            return (expr, false_mask)

    def _apply_not_mask(self, mask, expr_str):
        """
        三值逻辑 NOT 核心方法。
        
        SQL/Milvus 三值逻辑规则：
        - NOT(True)  = False
        - NOT(False) = True  
        - NOT(NULL)  = NULL → 在布尔上下文中视为 False

        因此: not(col > 5) 当 col=NULL → NOT(NULL) → NULL → False (不返回该行)
        这意味着: not(A) 的结果 ≠ ~A，而是 ~A & (所有涉及字段非NULL)
        
        特殊情况:
        - "is null" / "is not null" 表达式: 结果已经是纯布尔(不含NULL)，可以直接 ~mask
        - JSON 字段的 != : JSON 中 NULL != val → True，但 NOT(NULL != val) → False
        """
        import re

        # 特殊情况 1: 纯 NULL 检查表达式 (is null / is not null)
        # 这些表达式的结果是纯布尔，不存在 NULL 中间态
        # "col is null" → True/False (没有 NULL)，所以 NOT 直接取反
        stripped = expr_str.strip()
        if re.match(r'^\w+\s+is\s+(not\s+)?null$', stripped, re.IGNORECASE):
            return ~mask

        # 特殊情况 2: 常量表达式 (null is null / null is not null)
        if stripped.lower().startswith('null '):
            return ~mask

        # 一般情况: 需要识别表达式中涉及的字段，排除 NULL 行
        # 提取表达式中出现的 schema 字段名
        involved_fields = []
        for f in self.schema:
            fname = f["name"]
            # 检查字段名是否出现在表达式中（作为独立的 token）
            # 使用 word boundary 避免 "c1" 匹配 "c10"
            if re.search(r'\b' + re.escape(fname) + r'\b', expr_str):
                involved_fields.append(f)

        if not involved_fields:
            # 无法识别字段，保守处理：直接取反
            return ~mask

        # 构建 "所有涉及字段均非NULL" 的 mask
        notnull_mask = pd.Series(True, index=self.df.index)
        for f in involved_fields:
            fname = f["name"]
            ftype = f["type"]
            series = self.df[fname]

            if ftype == DataType.JSON:
                # JSON 字段: 字段本身非 NULL 即可
                # (JSON 内部的 key missing 已经在原始 mask 中处理过了)
                notnull_mask = notnull_mask & series.notnull()
            else:
                # 标量字段: 必须非 NULL
                notnull_mask = notnull_mask & series.notnull()

        # NOT(A) = ~A & all_fields_notnull
        # 解释: 只有当所有涉及字段都有值时，NOT 才会把 False 翻转为 True
        #        如果任何字段是 NULL，NOT(NULL) = NULL → False
        return ~mask & notnull_mask

    def gen_not_atomic_expr(self):
        """
        生成专门针对 NOT 的原子表达式。
        返回 (expr_str, pandas_mask)
        
        这些表达式专门设计来触发 Milvus 的 NOT + NULL 交互路径，
        是发现三值逻辑 Bug 的高效手段。
        """
        f = random.choice(self.schema)
        name, ftype = f["name"], f["type"]
        series = self.df[name]

        strategy = random.choice([
            "not_compare",       # not (col > X)
            "not_eq",            # not (col == X)
            "not_is_null",       # not (col is null) ↔ col is not null
            "not_is_not_null",   # not (col is not null) ↔ col is null
            "not_and",           # not (col > X and col < Y)
            "not_or",            # not (col == X or col == Y)
        ])

        # --- 策略 1: NOT + 比较运算 ---
        if strategy == "not_compare":
            if ftype in [DataType.INT64, DataType.DOUBLE]:
                val = self.get_value_for_query(name, ftype)
                if val is None:
                    return (f"not ({name} is null)", series.notnull())
                op = random.choice([">", "<", ">=", "<="])
                inner_expr = f"{name} {op} {val}"
                expr = f"not ({inner_expr})"
                
                # Oracle: 计算内部表达式的 mask
                def safe_cmp(x, _op=op, _val=val):
                    if x is None: return False
                    if isinstance(x, float) and np.isnan(x): return False
                    try:
                        if _op == ">": return x > _val
                        if _op == "<": return x < _val
                        if _op == ">=": return x >= _val
                        if _op == "<=": return x <= _val
                        return False
                    except: return False
                inner_mask = series.apply(safe_cmp)
                # 3VL NOT: ~inner & notnull
                mask = ~inner_mask & series.notnull()
                return (expr, mask)
            
            elif ftype == DataType.VARCHAR:
                val = self.get_value_for_query(name, ftype)
                if val is None:
                    return (f"not ({name} is null)", series.notnull())
                inner_expr = f'{name} == "{val}"'
                expr = f"not ({inner_expr})"
                def safe_eq(x, _val=val):
                    if x is None: return False
                    if isinstance(x, float) and np.isnan(x): return False
                    try: return x == _val
                    except: return False
                inner_mask = series.apply(safe_eq)
                mask = ~inner_mask & series.notnull()
                return (expr, mask)

            elif ftype == DataType.BOOL:
                val_bool = random.choice([True, False])
                inner_expr = f"{name} == {str(val_bool).lower()}"
                expr = f"not ({inner_expr})"
                def safe_eq_bool(x, _val=val_bool):
                    if x is None: return False
                    if isinstance(x, float) and np.isnan(x): return False
                    try: return x == _val
                    except: return False
                inner_mask = series.apply(safe_eq_bool)
                mask = ~inner_mask & series.notnull()
                return (expr, mask)

            elif ftype == DataType.JSON:
                # NOT + JSON 下钻比较
                val = random.randint(100, 500)
                inner_expr = f'{name}["price"] > {val}'
                expr = f"not ({inner_expr})"
                def check_json_gt(x, _val=val):
                    try:
                        v = self._get_json_val(x, ["price"])
                        if v is None: return False
                        if isinstance(v, bool): return False
                        return isinstance(v, (int, float)) and v > _val
                    except: return False
                inner_mask = series.apply(check_json_gt)
                # JSON NOT: ~inner，但 JSON 字段本身为 NULL 的行返回 False
                mask = ~inner_mask & series.notnull()
                return (expr, mask)

            # 兜底
            return (f"not ({name} is null)", series.notnull())

        # --- 策略 2: NOT + 等值 ---
        elif strategy == "not_eq":
            if ftype == DataType.INT64:
                val = self.get_value_for_query(name, ftype)
                if val is None:
                    return (f"not ({name} is null)", series.notnull())
                val_int = int(val)
                expr = f"not ({name} == {val_int})"
                def safe_eq_int(x, _val=val_int):
                    if x is None: return False
                    if isinstance(x, float) and np.isnan(x): return False
                    try: return x == _val
                    except: return False
                inner_mask = series.apply(safe_eq_int)
                mask = ~inner_mask & series.notnull()
                return (expr, mask)
            # 其他类型 fallthrough
            return (f"not ({name} is null)", series.notnull())

        # --- 策略 3: NOT (is null) ↔ is not null ---
        elif strategy == "not_is_null":
            expr = f"not ({name} is null)"
            mask = series.notnull()
            return (expr, mask)

        # --- 策略 4: NOT (is not null) ↔ is null ---
        elif strategy == "not_is_not_null":
            expr = f"not ({name} is not null)"
            mask = series.isnull()
            return (expr, mask)

        # --- 策略 5: NOT + AND (德摩根定律测试核心) ---
        elif strategy == "not_and":
            if ftype == DataType.INT64:
                val = self.get_value_for_query(name, ftype)
                if val is None:
                    return (f"not ({name} is null)", series.notnull())
                val_int = int(val)
                low = val_int - random.randint(10, 100)
                high = val_int + random.randint(10, 100)
                inner_expr = f"({name} > {low} and {name} < {high})"
                expr = f"not {inner_expr}"
                # Oracle: not (col > low and col < high)
                # 3VL: NULL 行 → inner = False (因为 NULL>low = False)
                #       所以 NOT(False) = True? 不！
                #       实际上 NULL>low = NULL, NULL<high = NULL
                #       NULL AND NULL = NULL, NOT(NULL) = NULL → False
                def check_range(x, _low=low, _high=high):
                    if x is None: return False
                    if isinstance(x, float) and np.isnan(x): return False
                    try: return x > _low and x < _high
                    except: return False
                inner_mask = series.apply(check_range)
                mask = ~inner_mask & series.notnull()
                return (expr, mask)
            return (f"not ({name} is null)", series.notnull())

        # --- 策略 6: NOT + OR ---
        elif strategy == "not_or":
            if ftype == DataType.INT64:
                val1 = self.get_value_for_query(name, ftype)
                val2 = self.get_value_for_query(name, ftype)
                if val1 is None or val2 is None:
                    return (f"not ({name} is null)", series.notnull())
                val1_int, val2_int = int(val1), int(val2)
                inner_expr = f"({name} == {val1_int} or {name} == {val2_int})"
                expr = f"not {inner_expr}"
                def check_or(x, _v1=val1_int, _v2=val2_int):
                    if x is None: return False
                    if isinstance(x, float) and np.isnan(x): return False
                    try: return x == _v1 or x == _v2
                    except: return False
                inner_mask = series.apply(check_or)
                mask = ~inner_mask & series.notnull()
                return (expr, mask)
            return (f"not ({name} is null)", series.notnull())

        # 终极兜底
        return (f"not ({name} is null)", series.notnull())


    def gen_complex_expr(self, depth):
        """递归生成：同时返回 Milvus 字符串 和 Pandas Mask"""
        # 递归终止
        if depth == 0 or random.random() < 0.2:
            # 【新增】2% 概率生成安全的常数表达式（NULL is/is not null）
            # 【注意】纯数字常数表达式（1<2 等）会导致 Milvus 崩溃，已排除
            if random.random() < 0.02:
                res = self.gen_constant_expr()
                if res[0]:
                    return res

            if random.random() < 0.3:
                res = self.gen_json_advanced_expr()
                # 只有当成功生成了非空表达式时才返回
                if res[0]:
                    return res
            expr, mask = self.gen_atomic_expr()
            if expr: return expr, mask
            # 如果生成失败，重试
            return self.gen_complex_expr(depth)

        # 递归生成子节点
        expr_l, mask_l = self.gen_complex_expr(depth - 1)
        expr_r, mask_r = self.gen_complex_expr(depth - 1)

        if not expr_l: return expr_r, mask_r
        if not expr_r: return expr_l, mask_l

        op = random.choices(["and", "or", "not"], weights=[0.4, 0.4, 0.2], k=1)[0]

        if op == "not":
            # --- NOT 分支 (三值逻辑) ---
            # 只使用左子表达式，对其取反
            # 20% 概率使用专门的 NOT 原子表达式（更精准打击 Bug）
            if random.random() < 0.2:
                not_res = self.gen_not_atomic_expr()
                if not_res and not_res[0]:
                    return not_res

            # 否则对普通子表达式取反
            expr_not = f"not ({expr_l})"
            mask_not = self._apply_not_mask(mask_l, expr_l)
            return expr_not, mask_not

        elif op == "and":
            # Milvus: (A and B)
            # Pandas: A & B
            return f"({expr_l} and {expr_r})", (mask_l & mask_r)
        else:
            # Milvus: (A or B)
            # Pandas: A | B
            return f"({expr_l} or {expr_r})", (mask_l | mask_r)


class EquivalenceQueryGenerator(OracleQueryGenerator):
    """
    等价查询生成器：不依赖 Pandas，而是生成两个逻辑等价的 Milvus 查询，
    比较它们在 Milvus 中的返回结果是否一致。
    """
    def __init__(self, dm):
        super().__init__(dm)
        # 建立字段名到类型的快速查找表
        self.field_types = {f["name"]: f["type"] for f in dm.schema_config}

    def _gen_guaranteed_false_expr(self):
        """
        根据 Schema 构造一个【必然为假】但【计算复杂】的表达式。
        用于替换简单的 id == -1，强制 Milvus 调动更多执行路径。
        """
        # 随机选一个字段，不要每次都用 id
        field = random.choice(self.schema)
        name = field["name"]
        dtype = field["type"]

        # 策略 1: 数组长度不可能为负 (针对 Array)
        if dtype == DataType.ARRAY:
            # 语义：数组长度小于 0 -> 恒假
            # 攻击点：强制触发 ArrayLength 算子
            return f"array_length({name}) < 0"

        # 策略 2: 字符串不可能包含的特殊值 (针对 Varchar)
        elif dtype == DataType.VARCHAR:
            # 语义：等于一个极其复杂的随机串
            # 攻击点：强制触发字符串 Hash 或 Trie 树搜索
            complex_str = "fuzz_impossible_" + "".join(random.choices(string.ascii_letters, k=10))
            return f'{name} == "{complex_str}"'

        # 策略 3: 数学矛盾 (针对 Int/Float)
        elif dtype in [DataType.INT64, DataType.INT32, DataType.INT16, DataType.INT8]:
            # 语义：(x > MAX) AND (x < MIN) -> 恒假
            # 攻击点：强制执行两次比较运算并做 AND 合并
            return f"({name} > 200000 and {name} < -200000)"

        # 策略 4: 浮点数 NaN 检测 (针对 Float/Double)
        elif dtype in [DataType.DOUBLE, DataType.FLOAT]:
            # 语义：两个很大的数做且运算，或者利用逻辑矛盾
            # 注意：Milvus 中 float == float 比较危险，用范围矛盾
            return f"({name} > 1e20 and {name} < -1e20)"
            
        # 策略 5: JSON 包含不存在的 Key (针对 JSON)
        elif dtype == DataType.JSON:
             # 语义：JSON 包含一个极长的不存在 Key
             # 攻击点：强制解析 JSON 结构
             return f'exists({name}["milvus_fuzz_ghost_key_v3"])'

        # 兜底：如果上面都没命中 (比如 Bool)，还是用 ID
        return "id == -999999"

    def mutate_expr(self, base_expr):
        """
        输入一个基础表达式，返回一组逻辑等价的变形表达式列表。
        """
        mutations = []
        
        # 1. 双重否定 (Double Negation)
        # 逻辑：A <=> not (not A)
        mutations.append({
            "type": "DoubleNegation",
            "expr": f"not (not ({base_expr}))"
        })

        # 2. 恒真条件注入 (Tautology Injection)
        # 逻辑：A <=> A AND True
        # 注意：Milvus 不支持纯常量 "1==1"，必须用数据相关的恒真式 "id > -1"
        mutations.append({
            "type": "TautologyAnd",
            "expr": f"({base_expr}) and (id > -1)"
        })

        # 3. ID 切分 (Partitioning) - 强力测试
        # 逻辑：A <=> (A AND condition) OR (A AND NOT condition)
        # 使用 Modulo 模拟随机切分
        mutations.append({
            "type": "PartitionById",
            "expr": f"(({base_expr}) and (id % 2 == 0)) or (({base_expr}) and (id % 2 != 0))"
        })

        # 4. 冗余 OR (Idempotency)
        # 逻辑：A <=> A OR A
        # 这可以测试去重逻辑
        mutations.append({
            "type": "SelfOr",
            "expr": f"({base_expr}) or ({base_expr})"
        })

        # 5. 特殊操作符展开 (针对 IN)
        # 只有当基础表达式极其简单且包含 IN 时才触发
        if " in [" in base_expr and " and " not in base_expr:
            try:
                # 解析简单的 field in [a, b]
                parts = base_expr.split(" in ")
                field = parts[0]
                # 这是一个非常简陋的解析，仅演示思路
                list_str = parts[1].strip("[]")
                items = [x.strip() for x in list_str.split(",")]
                if len(items) > 0 and len(items) < 10:
                    or_expr = " or ".join([f"{field} == {item}" for item in items])
                    mutations.append({
                        "type": "InExpandToOr",
                        "expr": f"({or_expr})"
                    })
            except:
                pass

        #6. 德·摩根定律包装 (De Morgan Wrapper)
        # 逻辑: A <=> NOT ( (NOT A) OR (id == -1) )
        # 假设 id 都是正数，(id == -1) 为 False
        # 这迫使查询引擎执行：全集 - ( (全集 - A) U 空集 )
        complex_false = self._gen_guaranteed_false_expr()
        mutations.append({
            "type": "DeMorganWrapper",
            # 逻辑：not ( (not A) or False ) <=> not (not A) <=> A
            "expr": f"not ( (not ({base_expr})) or ({complex_false}) )"
        })

        # 7. 交换律测试 (Commutativity)
        # 原有逻辑是 A and True，现在增加 True and A
        # 测试执行引擎的左值/右值求值顺序
        mutations.append({
            "type": "TautologyAnd_Left",
            "expr": f"(id > -1) and ({base_expr})"
        })

        # 8. 噪声 OR 注入 (Empty Set Merge)
        # 逻辑: A <=> A OR False
        # 测试结果集并集操作 (Union) 的稳定性
        complex_false_2 = self._gen_guaranteed_false_expr()
        mutations.append({
            "type": "NoiseOr",
            "expr": f"({base_expr}) or ({complex_false_2})"
        })

        
        # 9. 整数不等式微调 (Integer Range Shift)
        # 逻辑: (col > 10) <=> (col >= 11) 对于整数
        # 这是一个针对 Milvus 范围查询(Range Query) 极其有效的测试
        # 我们尝试用正则寻找 " > 数字" 的模式
        import re
        
        # 定义整数类型集合
        int_types = [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64]

        # 匹配 "> 整数"
        # 注意：这里正则稍微改一下，确保后面不是小数点，防止匹配到浮点数的整数部分
        # r'(?<!\.)' 表示前面不能有小数点（虽然后面有 field 一般没事）
        # r'(?!\.)' 表示数字后面不能紧跟小数点
        match_gt = re.search(r'([a-zA-Z0-9_]+)\s*>\s*(\d+)(?!\.)', base_expr)
        if match_gt:
            col = match_gt.group(1)
            val = int(match_gt.group(2))
            
            # 【关键修复】检查该字段是否真的是整数类型
            col_type = self.field_types.get(col)
            if col_type in int_types:
                if val < 2**60: 
                    new_expr = base_expr.replace(match_gt.group(0), f"{col} >= {val + 1}")
                    mutations.append({
                        "type": "IntRangeShift_GT",
                        "expr": new_expr
                    })

        # 匹配 "< 整数"
        match_lt = re.search(r'([a-zA-Z0-9_]+)\s*<\s*(\d+)(?!\.)', base_expr)
        if match_lt:
            col = match_lt.group(1)
            val = int(match_lt.group(2))
            
            # 【关键修复】检查该字段是否真的是整数类型
            col_type = self.field_types.get(col)
            if col_type in int_types:
                if val > -2**60:
                    new_expr = base_expr.replace(match_lt.group(0), f"{col} <= {val - 1}")
                    mutations.append({
                        "type": "IntRangeShift_LT",
                        "expr": new_expr
                    })
        return mutations

def run_equivalence_mode(rounds=100, seed=None, enable_dynamic_ops=True):
    """
    运行等价性模糊测试
    """
    global INDEX_TYPE, CURRENT_INDEX_TYPE, VECTOR_CHECK_RATIO, VECTOR_TOPK, _GLOBAL_ID_COUNTER, METRIC_TYPE
    
    if seed is not None:
        print(f"\n🔒 Equivalence 模式使用固定种子 {seed}")
        random.seed(seed)
        np.random.seed(seed)
    else:
        seed = random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)

    # 【关键】重置 ID 计数器，保证完全可重复
    _GLOBAL_ID_COUNTER = 0
    DataManager._id_counter = 0

    # 【关键】在种子设置之后初始化随机变量，保证可重复性
    INDEX_TYPE = random.choice(ALL_INDEX_TYPES)
    CURRENT_INDEX_TYPE = INDEX_TYPE
    METRIC_TYPE = random.choice(ALL_METRIC_TYPES)
    VECTOR_CHECK_RATIO = random.uniform(0.2, 0.8)
    VECTOR_TOPK = random.randint(50, 200)
    print(f"   索引类型: {INDEX_TYPE}, 度量类型: {METRIC_TYPE}")

    # --- 日志设置 ---
    timestamp = int(time.time())
    log_filename = f"equiv_test_{timestamp}.log"
    print("\n" + "="*60)
    print(f"👯 启动 Equivalence Mode (等价性测试)")
    print(f"   原理: Query(A) 应该等于 Query(Transformation(A))")
    print(f"   Seed: {seed}")
    print(f"📄 日志: {log_filename}")
    print("="*60)

    # 1. 初始化
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()

    mm = MilvusManager()
    mm.connect()
    mm.reset_collection(dm.schema_config)
    mm.insert(dm)

    qg = EquivalenceQueryGenerator(dm)
    failed_cases = []

    with open(log_filename, "w", encoding="utf-8") as f:
        def file_log(msg):
            f.write(msg + "\n")
            f.flush()
        
        file_log(f"Equivalence Test Started | Seed: {seed}")

        for i in range(rounds):
            print(f"\r⚖️  Test {i+1}/{rounds}...", end="", flush=True)
            
            # --- 【动态数据变动】每轮有 20% 概率触发 ---
            if enable_dynamic_ops and i > 0 and random.random() < 0.2:
                op = random.choices(["insert", "delete", "upsert"], weights=[0.4, 0.4, 0.2], k=1)[0]
                batch_count = random.randint(1, 5)

                if op == "insert":
                    new_rows = []
                    rows_with_vec = []
                    for _ in range(batch_count):
                        row = dm.generate_single_row()
                        vec = dm.generate_single_vector()
                        row_with_vec = row.copy()
                        row_with_vec["vector"] = vec
                        new_rows.append(row)
                        rows_with_vec.append(row_with_vec)
                    try:
                        mm.col.insert(rows_with_vec)
                        mm.col.flush()
                        dm.df = pd.concat([dm.df, pd.DataFrame(new_rows)], ignore_index=True)
                        dm.vectors = np.vstack([dm.vectors, np.array([r["vector"] for r in rows_with_vec])])
                        file_log(f"[Dynamic] Inserted {len(new_rows)} rows")
                    except Exception as e:
                        file_log(f"[Dynamic] Insert failed: {e}")

                elif op == "delete" and len(dm.df) > 100:
                    del_count = min(batch_count, len(dm.df) - 100)
                    del_ids = random.sample(dm.df["id"].tolist(), del_count)
                    try:
                        expr = f"id in {del_ids}"
                        mm.col.delete(expr)
                        mm.col.flush()
                        idx = dm.df[dm.df["id"].isin(del_ids)].index.to_numpy()
                        dm.df = dm.df.drop(idx).reset_index(drop=True)
                        dm.vectors = np.delete(dm.vectors, idx, axis=0)
                        file_log(f"[Dynamic] Deleted {len(del_ids)} rows")
                    except Exception as e:
                        file_log(f"[Dynamic] Delete failed: {e}")

                else:  # upsert
                    upsert_rows = []
                    rows_with_vec = []
                    new_rows_list = []
                    new_vectors = []
                    for _ in range(batch_count):
                        use_existing = (not dm.df.empty) and (random.random() < 0.7)
                        if use_existing:
                            target_id = random.choice(dm.df["id"].tolist())
                        else:
                            target_id = generate_unique_id()
                        row = dm.generate_single_row(id_override=target_id)
                        vec = dm.generate_single_vector()
                        row_with_vec = row.copy()
                        row_with_vec["vector"] = vec
                        upsert_rows.append(row)
                        rows_with_vec.append(row_with_vec)
                    try:
                        mm.col.upsert(rows_with_vec)
                        mm.col.flush()
                        for row, row_with_vec in zip(upsert_rows, rows_with_vec):
                            rid = row["id"]
                            match_idx = dm.df.index[dm.df["id"] == rid].tolist()
                            if match_idx:
                                idx = match_idx[0]
                                for k, v in row.items():
                                    dm.df.at[idx, k] = v
                                dm.vectors[idx] = row_with_vec["vector"]
                            else:
                                new_rows_list.append(row)
                                new_vectors.append(row_with_vec["vector"])
                        if new_rows_list:
                            dm.df = pd.concat([dm.df, pd.DataFrame(new_rows_list)], ignore_index=True)
                            dm.vectors = np.vstack([dm.vectors, np.array(new_vectors)])
                        file_log(f"[Dynamic] Upserted {len(upsert_rows)} rows")
                    except Exception as e:
                        file_log(f"[Dynamic] Upsert failed: {e}")

            # 1. 生成一个基础查询 (Base Query)
            # 使用较小的深度，方便人类阅读 debug
            base_expr = ""
            while not base_expr:
                base_expr, _ = qg.gen_complex_expr(depth=random.randint(1, 15))
            
            # 2. 生成变体 (Mutations)
            mutations = qg.mutate_expr(base_expr)
            if not mutations: continue

            # 3. 执行 Base Query
            try:
                base_res = mm.col.query(base_expr, output_fields=["id"], consistency_level="Strong")
                base_ids = set([x["id"] for x in base_res])
            except Exception as e:
                # 基础查询挂了，可能是语法问题，跳过
                file_log(f"[Test {i}] Base Query Failed: {e}")
                continue

            log_header = f"[Test {i}] Base: {base_expr} (Hits: {len(base_ids)})"
            file_log(f"\n{log_header}")

            # 4. 执行并对比所有变体
            for m in mutations:
                m_type = m["type"]
                m_expr = m["expr"]
                
                try:
                    mut_res = mm.col.query(m_expr, output_fields=["id"], consistency_level="Strong")
                    mut_ids = set([x["id"] for x in mut_res])

                    if base_ids == mut_ids:
                        file_log(f"  ✅ [{m_type}] Match")
                    else:
                        # --- 发现 BUG ---
                        print(f"\n\n❌ EQUIVALENCE FAILURE [Test {i}]")
                        print(f"   Type: {m_type}")
                        print(f"   Base Expr: {base_expr}")
                        print(f"   Mut  Expr: {m_expr}")
                        print(f"   Base Hits: {len(base_ids)} | Mut Hits: {len(mut_ids)}")
                        
                        diff_msg = ""
                        missing = base_ids - mut_ids
                        extra = mut_ids - base_ids
                        if missing: diff_msg += f"Mutation Missing: {list(missing)} "
                        if extra: diff_msg += f"Mutation Extra: {list(extra)} "
                        print(f"   Diff: {diff_msg}")
                        print("-" * 50)
                        
                        # ==================== [新增代码开始] ====================
                        # 🔍 深度取证：打印出问题行的实际数据
                        def print_evidence(ids, label):
                            if not ids: return
                            print(f"   🕵️‍♀️ {label} Row Analysis (Top 3):")
                            # 从 DataManager 的 DataFrame 中提取行
                            subset = dm.df[dm.df["id"].isin(list(ids)[:3])]
                            
                            for _, row in subset.iterrows():
                                row_id = row["id"]
                                print(f"      👉 ID: {row_id}")
                                
                                # 智能提取表达式中涉及的字段 (简化版：打印所有非空字段，重点关注 JSON)
                                # 1. 打印标量中 None 的情况
                                null_fields = [k for k, v in row.items() if v is None and k != "vector"]
                                if null_fields:
                                    print(f"         ⚠️ Null Fields: {null_fields}")
                                
                                # 2. 打印 JSON 内容 (如果表达式里涉及)
                                if "meta_json" in base_expr:
                                    import json
                                    # 格式化打印 JSON
                                    js_val = row.get("meta_json")
                                    if js_val:
                                        print(f"         📄 meta_json: {json.dumps(js_val, ensure_ascii=False)}")
                                    else:
                                        print(f"         📄 meta_json: None")

                                # 3. 打印其他关键标量 (为了不刷屏，只打印 String/Int/Array)
                                for k, v in row.items():
                                    if k in ["id", "vector", "meta_json"]: continue
                                    # 简单的启发式：如果字段名出现在表达式里，就打印
                                    if k in base_expr: 
                                        print(f"         🔹 {k}: {v}")
                                print("      " + "-"*30)

                        print_evidence(missing, "MISSING (Base hit, Mut missed)")
                        print_evidence(extra, "EXTRA (Base missed, Mut hit)")
                        # ==================== [新增代码结束] ====================

                        file_log(f"  ❌ [{m_type}] FAIL! {diff_msg}")
                        print(f"   Diff: {diff_msg}")
                        print("-" * 50)

                        file_log(f"  ❌ [{m_type}] FAIL! {diff_msg}")
                        file_log(f"     Mut Expr: {m_expr}")

                        failed_cases.append({
                            "id": i,
                            "type": m_type,
                            "base": base_expr,
                            "mut": m_expr,
                            "diff": diff_msg
                        })

                except Exception as e:
                    print(f"\n\n⚠️ Mutation Crash [Test {i}]")
                    print(f"   Type: {m_type}")
                    print(f"   Expr: {m_expr}")
                    print(f"   Error: {e}")
                    file_log(f"  ⚠️ [{m_type}] CRASH: {e}")

    print("\n" + "="*60)
    if failed_cases:
        print(f"🚫 发现 {len(failed_cases)} 个等价性错误！请检查日志。")
    else:
        print(f"✅ 所有等价性测试通过。")

# --- 4. Main Execution ---

def run(rounds = 100, seed=None, enable_dynamic_ops=True):
    """
    seed=None: 随机数据，每次不同（默认行为）
    seed=<数字>: 固定种子，完全复现之前的测试
    """
    global INDEX_TYPE, CURRENT_INDEX_TYPE, VECTOR_CHECK_RATIO, VECTOR_TOPK, _GLOBAL_ID_COUNTER, METRIC_TYPE
    
    # 记录当前使用的种子（如果有的话），方便后续复现
    current_seed = seed
    if current_seed is None:
        # 即使没有指定seed，也生成一个用于记录
        current_seed = random.randint(0, 2**31 - 1)
        random.seed(current_seed)
        np.random.seed(current_seed)
        print(f"🎲 随机生成种子: {current_seed}")
    else:
        print(f"🔒 使用固定种子 {current_seed} - 可复现的数据")
        random.seed(seed)
        np.random.seed(seed)

    # 【关键】重置 ID 计数器，保证完全可重复
    _GLOBAL_ID_COUNTER = 0
    DataManager._id_counter = 0

    # 【关键】在种子设置之后初始化随机变量，保证可重复性
    INDEX_TYPE = random.choice(ALL_INDEX_TYPES)
    CURRENT_INDEX_TYPE = INDEX_TYPE
    METRIC_TYPE = random.choice(ALL_METRIC_TYPES)
    VECTOR_CHECK_RATIO = random.uniform(0.2, 0.8)
    VECTOR_TOPK = random.randint(50, 200)
    print(f"   索引类型: {INDEX_TYPE}, 度量类型: {METRIC_TYPE}, 向量校验比例: {VECTOR_CHECK_RATIO:.2f}, TopK: {VECTOR_TOPK}")

    # 1. 初始化
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()

    mm = MilvusManager()
    mm.connect()
    mm.reset_collection(dm.schema_config)
    mm.insert(dm)

    # 2. 日志设置
    timestamp = int(time.time())
    log_filename = f"fuzz_test_{timestamp}.log"
    print(f"\n📝 详细日志将写入: {log_filename}")
    print(f"   🔑 如需复现此次测试，运行: python milvus_fuzz_oracle.py --seed {current_seed}")
    print(f"🚀 开始测试 (控制台仅显示失败案例)...")

    qg = OracleQueryGenerator(dm)
    failed_cases = []
    total_test = rounds

    # 打开文件 (使用 'w' 模式)
    with open(log_filename, "w", encoding="utf-8") as f:

        # 辅助：写入文件并刷新
        def file_log(msg):
            f.write(msg + "\n")
            f.flush()

        # 辅助：提取指定 ID 的行，避免控制台输出过大
        def sample_rows(id_set, limit=5):
            if not id_set:
                return []
            subset = dm.df[dm.df["id"].isin(list(id_set))]
            rows = subset.to_dict(orient="records")
            return rows[:limit]

        file_log(f"Start Testing: {total_test} rounds")
        file_log("=" * 50)

        for i in range(total_test):
            print(f"\r⏳ Running Test {i+1}/{total_test}...", end="", flush=True)

            # --- 随机触发 compaction ---
            if i > 0 and i % 25 == 0:
                try:
                    mm.col.compact()
                    file_log(f"[Maintenance] Triggered compaction at round {i}")
                except Exception as e:
                    file_log(f"[Maintenance] Compaction failed at round {i}: {e}")

            # --- 随机触发索引重建 ---
            if i > 0 and i % 40 == 0:
                try:
                    # 选择一个不同于当前的索引类型
                    candidates = [t for t in ALL_INDEX_TYPES if t != CURRENT_INDEX_TYPE]
                    if not candidates:
                        candidates = ALL_INDEX_TYPES
                    new_index_type = random.choice(candidates)
                    old_index_type = CURRENT_INDEX_TYPE
                    
                    # 根据索引类型设置参数（注意 Milvus 参数结构）
                    if new_index_type == "HNSW":
                        create_params = {
                            "index_type": "HNSW",
                            "metric_type": METRIC_TYPE,
                            "params": {"M": 8, "efConstruction": 64}
                        }
                    elif new_index_type == "IVF_PQ":
                        # IVF_PQ 需要额外的 m 和 nbits 参数
                        create_params = {
                            "index_type": "IVF_PQ",
                            "metric_type": METRIC_TYPE,
                            "params": {"nlist": 128, "m": 8, "nbits": 8}
                        }
                    elif new_index_type.startswith("IVF"):
                        # IVF_FLAT, IVF_SQ8 等
                        create_params = {
                            "index_type": new_index_type,
                            "metric_type": METRIC_TYPE,
                            "params": {"nlist": 128}
                        }
                    else:
                        # FLAT 不需要额外参数
                        create_params = {
                            "index_type": new_index_type,
                            "metric_type": METRIC_TYPE,
                            "params": {}
                        }
                    
                    # 必须先 release 才能 drop_index
                    mm.col.release()
                    file_log(f"[Maintenance] Released collection before drop index at round {i}")
                    
                    # drop_index() 不需要 field_name 参数，直接删除默认索引
                    mm.col.drop_index()
                    file_log(f"[Maintenance] Dropped vector index (was {old_index_type}) at round {i}")
                    
                    mm.col.create_index(
                        field_name="vector",
                        index_params=create_params
                    )
                    # 等待索引构建完成
                    utility.wait_for_index_building_complete(COLLECTION_NAME)
                    file_log(f"[Maintenance] Rebuilt index to {new_index_type} at round {i}")
                    CURRENT_INDEX_TYPE = new_index_type
                    mm.col.load()
                    file_log(f"[Maintenance] Reloaded collection after index rebuild at round {i}")
                except Exception as e:
                    file_log(f"[Maintenance] Index rebuild failed at round {i}: {e}")
                    # 尝试恢复：重新 load collection
                    try:
                        file_log(f"[Maintenance] Attempting recovery - reloading collection at round {i}")
                        mm.col.load()
                        file_log(f"[Maintenance] Recovery successful - collection reloaded at round {i}")
                    except Exception as e2:
                        file_log(f"[Maintenance] Recovery failed at round {i}: {e2}")

            # --- 动态插入/删除/Upsert ---
            if enable_dynamic_ops and i > 0 and i % 10 == 0:
                op = random.choices(["insert", "delete", "upsert"], weights=[0.4, 0.4, 0.2], k=1)[0]
                batch_count = random.randint(1, 5)

                if op == "insert":
                    new_rows = []
                    rows_with_vec = []
                    for _ in range(batch_count):
                        row = dm.generate_single_row()
                        vec = dm.generate_single_vector()
                        row_with_vec = row.copy()
                        row_with_vec["vector"] = vec
                        new_rows.append(row)
                        rows_with_vec.append(row_with_vec)

                    try:
                        mm.col.insert(rows_with_vec)
                        mm.col.flush()
                        dm.df = pd.concat([dm.df, pd.DataFrame(new_rows)], ignore_index=True)
                        dm.vectors = np.vstack([dm.vectors, np.array([r["vector"] for r in rows_with_vec])])
                        inserted_ids = [r["id"] for r in new_rows]
                        file_log(f"[Dynamic] Inserted {len(new_rows)} rows: ids={inserted_ids}")
                    except Exception as e:
                        file_log(f"[Dynamic] Insert failed: {e}")

                elif op == "delete":
                    if not dm.df.empty:
                        del_count = min(batch_count, len(dm.df))
                        del_ids = random.sample(dm.df["id"].tolist(), del_count)
                        try:
                            expr = f"id in {del_ids}"
                            mm.col.delete(expr)
                            mm.col.flush()
                            idx = dm.df[dm.df["id"].isin(del_ids)].index.to_numpy()
                            dm.df = dm.df.drop(idx).reset_index(drop=True)
                            dm.vectors = np.delete(dm.vectors, idx, axis=0)
                            file_log(f"[Dynamic] Deleted {len(del_ids)} rows: ids={del_ids}")
                            
                            # 验证删除是否生效
                            for did in del_ids:
                                verify_res = mm.col.query(
                                    f"id == {did}",
                                    output_fields=["id"],
                                    consistency_level="Strong"
                                )
                                if verify_res:
                                    file_log(f"[DELETE_WARN] ID {did} still exists in Milvus after delete!")
                                    
                        except Exception as e:
                            file_log(f"[Dynamic] Delete failed: {e}")

                else:
                    # upsert
                    upsert_rows = []
                    rows_with_vec = []
                    updated_ids = []
                    new_rows = []
                    new_vectors = []

                    for _ in range(batch_count):
                        use_existing = (not dm.df.empty) and (random.random() < 0.7)
                        if use_existing:
                            target_id = random.choice(dm.df["id"].tolist())
                            updated_ids.append(target_id)
                        else:
                            target_id = generate_unique_id()

                        row = dm.generate_single_row(id_override=target_id)
                        vec = dm.generate_single_vector()
                        row_with_vec = row.copy()
                        row_with_vec["vector"] = vec
                        upsert_rows.append(row)
                        rows_with_vec.append(row_with_vec)

                    try:
                        mm.col.upsert(rows_with_vec)
                        mm.col.flush()

                        for row, row_with_vec in zip(upsert_rows, rows_with_vec):
                            rid = row["id"]
                            match_idx = dm.df.index[dm.df["id"] == rid].tolist()
                            if match_idx:
                                idx = match_idx[0]
                                for k, v in row.items():
                                    dm.df.at[idx, k] = v
                                dm.vectors[idx] = row_with_vec["vector"]
                            else:
                                new_rows.append(row)
                                new_vectors.append(row_with_vec["vector"])

                        if new_rows:
                            dm.df = pd.concat([dm.df, pd.DataFrame(new_rows)], ignore_index=True)
                            dm.vectors = np.vstack([dm.vectors, np.array(new_vectors)])

                        # 增强日志：记录 upsert 的具体 ID 和关键数据
                        upsert_ids = [r["id"] for r in upsert_rows]
                        upsert_details = [{
                            "id": r["id"],
                            "history": r.get("meta_json", {}).get("history"),
                            "c19": r.get("c19"),
                            "price": r.get("meta_json", {}).get("price")
                        } for r in upsert_rows]
                        
                        file_log(
                            f"[Dynamic] Upserted {len(upsert_rows)} rows: ids={upsert_ids}, updated_ids={updated_ids}, new_ids={[r['id'] for r in new_rows]}"
                        )
                        file_log(f"[Dynamic] Upsert details: {upsert_details}")
                        
                        # 验证 upsert 数据同步
                        for row in upsert_rows:
                            rid = row["id"]
                            # 检查 Milvus 数据
                            try:
                                milvus_res = mm.col.query(
                                    f"id == {rid}",
                                    output_fields=["id", "c19", "meta_json"],
                                    consistency_level="Strong"
                                )
                                if milvus_res:
                                    milvus_history = milvus_res[0].get("meta_json", {}).get("history")
                                    pandas_row = dm.df[dm.df["id"] == rid]
                                    if not pandas_row.empty:
                                        pandas_history = pandas_row.iloc[0]["meta_json"].get("history")
                                        if milvus_history != pandas_history:
                                            file_log(f"[SYNC_WARN] ID {rid} history mismatch after upsert: Milvus={milvus_history} vs Pandas={pandas_history}")
                            except Exception as ve:
                                file_log(f"[SYNC_WARN] ID {rid} verification error: {ve}")
                                
                    except Exception as e:
                        file_log(f"[Dynamic] Upsert failed: {e}")

            # 生成查询
            depth = random.randint(1, 15)
            expr_str = ""
            while not expr_str:
                expr_str, pandas_mask = qg.gen_complex_expr(depth)

            log_header = f"[Test {i}]"
            file_log(f"\n{log_header} Expr: {expr_str}")

            # Pandas 计算
            expected_ids = set(dm.df[pandas_mask.fillna(False)]["id"].values.tolist())

            try:
                start_t = time.time()

                iterator = mm.col.query_iterator(
                    batch_size=10000,
                    expr=expr_str,
                    output_fields=["id"],
                    consistency_level="Strong"
                )

                actual_ids = set()
                while True:
                    result = iterator.next()
                    if not result:
                        iterator.close()
                        break
                    for item in result:
                        actual_ids.add(item["id"])

                cost = (time.time() - start_t) * 1000

                file_log(f"  Pandas: {len(expected_ids)} | Milvus: {len(actual_ids)} | Time: {cost:.1f}ms")

                if expected_ids == actual_ids:
                    file_log("  -> MATCH")
                else:
                    # --- 发现错误！控制台打印！---
                    print(f"\n❌ [Test {i}] MISMATCH!")
                    print(f"   Expr: {expr_str}")
                    print(f"   Expected: {len(expected_ids)} vs Actual: {len(actual_ids)}")

                    # 计算差异
                    missing = expected_ids - actual_ids
                    extra = actual_ids - expected_ids
                    diff_msg = ""
                    if missing: diff_msg += f"Missing IDs: {list(missing)} "
                    if extra: diff_msg += f"Extra IDs: {list(extra)}"

                    print(f"   Diff: {diff_msg}")
                    print(f"   🔑 复现此bug: python milvus_fuzz_oracle.py --seed {current_seed}\n")

                    file_log(f"  -> MISMATCH! {diff_msg}")
                    file_log(f"  -> REPRODUCTION SEED: {current_seed}")

                    # 记录与展示具体数据行，便于一眼确认异常
                    if missing:
                        missing_rows = sample_rows(missing)
                        file_log(f"  Missing rows sample ({len(missing_rows)}/{len(missing)}): {missing_rows}")
                        print("   Missing rows (sample):")
                        for r in missing_rows:
                            print(f"     {r}")
                    if extra:
                        extra_rows = sample_rows(extra)
                        file_log(f"  Extra rows sample ({len(extra_rows)}/{len(extra)}): {extra_rows}")
                        print("   Extra rows (sample):")
                        for r in extra_rows:
                            print(f"     {r}")

                        # 【验证】单独查询 Extra IDs，确认它们是否真的被 Milvus 返回
                        print("\n   🔍 Verifying Extra IDs individually:")
                        for eid in list(extra)[:5]:  # 只验证前5个，避免过多输出
                            try:
                                verify_res = mm.col.query(
                                    f"id == {eid}",
                                    output_fields=["id"],
                                    limit=1,
                                    consistency_level="Strong"
                                )
                                exists_in_db = len(verify_res) > 0

                                # 再用原始表达式 + id 过滤，看是否返回
                                combined_expr = f"({expr_str}) and (id == {eid})"
                                match_res = mm.col.query(
                                    combined_expr,
                                    output_fields=["id"],
                                    limit=1,
                                    consistency_level="Strong"
                                )
                                matches_expr = len(match_res) > 0

                                print(f"     ID {eid}: exists={exists_in_db}, matches_original_expr={matches_expr}")
                                file_log(f"  Extra ID {eid} verification: exists={exists_in_db}, matches={matches_expr}")

                            except Exception as ve:
                                print(f"     ID {eid}: verification failed - {ve}")
                                file_log(f"  Extra ID {eid} verification error: {ve}")

                    # 加入错误列表
                    failed_cases.append({
                        "id": i,
                        "expr": expr_str,
                        "detail": f"Exp: {len(expected_ids)} vs Act: {len(actual_ids)}. {diff_msg}",
                        "seed": current_seed
                    })
                # --- 向量 + 标量联合校验（HNSW/IVF/FLAT 均可），可选执行 ---
                # 仅当存在满足标量条件的数据时才做检查，且按比例抽样，避免对公用服务器造成额外压力。
                if expected_ids and random.random() < VECTOR_CHECK_RATIO:
                    try:
                        # 随机挑一条已有向量作为查询向量，确保搜索能命中实际数据
                        q_idx = random.randint(0, len(dm.vectors) - 1)
                        q_vec = dm.vectors[q_idx].tolist()

                        search_k = min(VECTOR_TOPK, len(dm.df))
                        
                        # 从当前索引动态获取 metric type
                        current_metric_type = METRIC_TYPE
                        try:
                            for idx in mm.col.indexes:
                                if idx.field_name == "vector":
                                    current_metric_type = idx.params.get("metric_type", METRIC_TYPE)
                                    break
                        except:
                            pass
                        
                        # 根据当前索引类型动态设置搜索参数
                        if CURRENT_INDEX_TYPE == "HNSW":
                            ef_param = max(64, search_k + 8)  # ef must be > k for HNSW
                            search_params = {"metric_type": current_metric_type, "params": {"ef": ef_param}}
                        elif CURRENT_INDEX_TYPE.startswith("IVF"):
                            search_params = {"metric_type": current_metric_type, "params": {"nprobe": 16}}
                        else:  # FLAT 或其他
                            search_params = {"metric_type": current_metric_type, "params": {}}

                        search_res = mm.col.search(
                            data=[q_vec],
                            anns_field="vector",
                            param=search_params,
                            limit=search_k,
                            expr=expr_str,
                            output_fields=["id"],
                        )

                        returned_ids = set()
                        if search_res and len(search_res) > 0:
                            for hit in search_res[0]:
                                # Hit 对象兼容属性/下标两种访问方式
                                returned_ids.add(hit.get("id") if isinstance(hit, dict) else hit.id)

                        if not returned_ids:
                            file_log("  VectorCheck: PASS (no ANN hits)")
                        elif returned_ids.issubset(expected_ids):
                            file_log(f"  VectorCheck: PASS ({len(returned_ids)} hits subset of scalar filter)")
                        else:
                            extra_vec = returned_ids - expected_ids
                            file_log(f"  VectorCheck: FAIL extra ids {list(extra_vec)[:3]}...")
                            print(f"\n⚠️ [Test {i}] Vector subset violated: extras {list(extra_vec)[:3]}...")
                            failed_cases.append({
                                "id": i,
                                "expr": expr_str,
                                "detail": f"VectorCheck extras: {list(extra_vec)[:3]}..."
                            })
                    except Exception as e:
                        file_log(f"  VectorCheck: ERROR {e}")
                        print(f"\n⚠️ [Test {i}] VectorCheck error: {e}")
            except Exception as e:
                # --- 发生异常！控制台打印！---
                print(f"\n⚠️ [Test {i}] CRASHED!")
                print(f"   Expr: {expr_str}")
                print(f"   Error: {e}\n")
                file_log(f"  -> ERROR: {e}")

                failed_cases.append({
                    "id": i,
                    "expr": expr_str,
                    "detail": f"Exception: {e}"
                })

    # --- 最终总结 ---
    print("\n" + "="*60)
    if not failed_cases:
        print(f"✅ 所有 {total_test} 轮测试全部通过！")
        print(f"📄 详细记录请查看: {log_filename}")
    else:
        print(f"🚫 发现 {len(failed_cases)} 个失败案例！(已保存至日志)")
        print("-" * 60)
        # 再次列出错误，防止遗漏
        for case in failed_cases:
            print(f"🔴 Case {case['id']}:")
            print(f"   Expr: {case['expr']}")
            print(f"   Issue: {case['detail']}")
            if 'seed' in case:
                print(f"   🔑 复现: python milvus_fuzz_oracle.py --seed {case['seed']}")
            print("-" * 30)
        print(f"📄 请查看 {log_filename} 获取完整上下文。")
        print(f"🔑 全局复现命令: python milvus_fuzz_oracle.py --seed {current_seed}")

class PQSQueryGenerator(OracleQueryGenerator):
    """
    PQS 生成器：专注于生成“必须能查到指定行”的查询。
    继承自 OracleQueryGenerator，复用其工具函数。
    """
    def __init__(self, dm):
        super().__init__(dm)

    def _has_valid_content(self, obj):
        """
        递归检查对象是否包含至少一个非 Null 的标量。
        解决 {"a": null} 或 [null] 被 Milvus 视为不存在的问题。
        """
        if obj is None:
            return False
        if isinstance(obj, (int, float, str, bool)):
            return True
        if hasattr(obj, "item"): # Numpy scalar
            return True

        # 递归检查容器
        if isinstance(obj, dict):
            return any(self._has_valid_content(v) for v in obj.values())
        if isinstance(obj, (list, np.ndarray)):
            return any(self._has_valid_content(v) for v in obj)

        return False

    def gen_multi_field_true_expr(self, row, n=None, min_fields=2, prefer_non_id=True):
        """
        基于pivot_row自动拼接n个字段的必真表达式，优先包含id字段。
        """
        if n is None:
            n = random.randint(2, 4)
        if min_fields < 1:
            min_fields = 1
        
        valid_fields = []
        # schema 是 list of dict, e.g. [{'name': 'id', 'type': <DataType.INT64: 5>}, ...]
        for f in self.schema:
            fname = f["name"]
            ftype = f["type"]
            if fname == "vector": continue
            if fname not in row: continue
            
            val = row[fname]
            # 简单的非空校验
            if val is None: continue
            if isinstance(val, float) and np.isnan(val): continue
            
            # 目前只处理标量，简单起见
            if ftype in [DataType.INT64, DataType.VARCHAR, DataType.BOOL, DataType.DOUBLE]:
                 valid_fields.append(f)

        if not valid_fields:
            pivot_id = row.get("id")
            return f"id == {pivot_id}"

        # 必须包含 id (如果有)
        id_field = next((f for f in valid_fields if f["name"] == "id"), None)
        other_fields = [f for f in valid_fields if f["name"] != "id"]

        chosen = []

        # 优先选择非 id 字段，确保表达式不止于 id
        target_count = max(min_fields, n)
        if other_fields:
            count = min(target_count, len(other_fields))
            chosen.extend(random.sample(other_fields, count))

        # 如果字段不够，再补充 id
        if id_field and len(chosen) < target_count:
            if (not prefer_non_id) or not other_fields:
                chosen.append(id_field)
            else:
                # 有其他字段但数量不够时，仍允许补 id 保证最小字段数
                chosen.append(id_field)
        
        exprs = []
        for f in chosen:
            fname = f["name"]
            ftype = f["type"]
            val = row[fname]
            if hasattr(val, "item"): val = val.item() # numpy scalar to python

            try:
                if ftype == DataType.BOOL:
                    exprs.append(f"{fname} == {str(bool(val)).lower()}")
                elif ftype == DataType.INT64:
                    exprs.append(self._gen_boundary_int(fname, val))
                elif ftype == DataType.DOUBLE:
                    exprs.append(self._gen_boundary_float(fname, val))
                elif ftype == DataType.VARCHAR:
                    exprs.append(self._gen_boundary_str(fname, val))
            except:
                pass # 忽略转换错误
        
        if not exprs:
            pivot_id = row.get("id")
            return f"id == {pivot_id}"
             
        return " and ".join(exprs)

    def gen_pqs_expr(self, pivot_row, depth):
        """基于pivot行生成必真的复合表达式"""
        # 获取pivot_id用于兜底
        pivot_id = pivot_row.get("id") if hasattr(pivot_row, "get") else pivot_row["id"]
        # fallback = f"id == {pivot_id}"
        # 【修改】使用多字段复杂兜底
        fallback = self.gen_multi_field_true_expr(pivot_row, min_fields=2)
        
        force_recursion = False
        if depth > 3:
            force_recursion = True

        # 递归终止
        if depth <= 0 or (not force_recursion and random.random() < 0.3):
            result = self.gen_true_atomic_expr(pivot_row)
            # 确保返回非空表达式
            if not result or not result.strip():
                return fallback
            return result

        op = random.choice(["and", "or", "nested_not"])

        if op == "nested_not":
            inner = self.gen_pqs_expr(pivot_row, depth)
            if not inner or not inner.strip(): inner = fallback
            return f"(not (not ({inner})))"

        elif op == "and":
            expr_l = self.gen_pqs_expr(pivot_row, depth - 1)
            expr_r = self.gen_pqs_expr(pivot_row, depth - 1)
            if not expr_l or not expr_l.strip(): expr_l = fallback
            if not expr_r or not expr_r.strip(): expr_r = fallback
            return f"({expr_l} and {expr_r})"

        else: # OR
            expr_l = self.gen_pqs_expr(pivot_row, depth - 1)
            noise_depth = depth + random.randint(0, 3)
            expr_r = self.gen_complex_noise(noise_depth, pivot_id)

            if not expr_l or not expr_l.strip(): expr_l = fallback
            if not expr_r or not expr_r.strip(): expr_r = fallback

            if random.random() < 0.5:
                return f"({expr_l} or {expr_r})"
            else:
                return f"({expr_r} or {expr_l})"

    def gen_true_atomic_expr(self, row):
        """
        核心逻辑：针对单行数据，生成必真的原子条件
        【增强版】支持边界值、精度测试、否定形式、交叉字段组合
        """
        # 随机选一个字段
        field = random.choice(self.schema)
        fname = field["name"]
        ftype = field["type"]
        val = row[fname]

        # 1. 处理 Null
        # 注意：pandas 的 NaN/None 处理
        is_null = False
        try:
            if val is None: is_null = True
            # 处理 Numpy 的 NaN
            if isinstance(val, float) and np.isnan(val): is_null = True
            # 处理 Pandas/Numpy 的特殊对象
            if isinstance(val, (np.ndarray, list)) and len(val) == 0:
                # 空数组不算 Null，但我们在 PQS 里不针对空数组生成值匹配
                pass
        except: pass

        if is_null:
            return f"{fname} is null"

        # 2. 针对不同类型的必真构造
        if ftype == DataType.BOOL:
            return f"{fname} == {str(bool(val)).lower()}"

        elif ftype == DataType.INT64:
            # 30% 概率生成算术表达式，否则使用边界表达式
            if random.random() < 0.3:
                res = self.gen_arithmetic_expr(fname, val)
                if res: return res
            return self._gen_boundary_int(fname, val)

        elif ftype == DataType.DOUBLE:
            # 30% 概率生成算术表达式，否则使用边界表达式
            if random.random() < 0.3:
                res = self.gen_arithmetic_expr(fname, val)
                if res: return res
            return self._gen_boundary_float(fname, val)

        # --- 集成高级 LIKE 测试 ---
        elif ftype == DataType.VARCHAR:
            if random.random() < 0.3:
                res = self.gen_advanced_like(fname, val)
                if res: return res
            return self._gen_boundary_str(fname, val)

        elif ftype == DataType.JSON:
            # 【增强】使用深层 JSON 下钻 (50% 概率)
            if random.random() < 0.5:
                deep_expr = self._gen_deep_json_path(fname, val)
                if deep_expr:
                    return deep_expr
            # 回退到标准 JSON 处理
            return self._gen_pqs_json(fname, val)

        elif ftype == DataType.ARRAY:
            if not isinstance(val, list):
                return f"{fname} is not null"

            length = len(val)
            # 过滤有效标量
            valid_items = [x for x in val if x is not None and not (isinstance(x, float) and np.isnan(x))]

            strategies = []

            # 策略 1: 基础 contains (已有)
            if valid_items:
                target = random.choice(valid_items)
                strategies.append(f'array_contains({fname}, {json.dumps(target)})')

            # 策略 2: [新增] array_length
            # 构造: len == X, len > X-1, len < X+1
            strategies.append(f'array_length({fname}) == {length}')
            strategies.append(f'array_length({fname}) >= {length}')

            # 策略 3: [新增] array_contains_all (子集)
            if len(valid_items) >= 2:
                subset = random.sample(valid_items, random.randint(2, min(4, len(valid_items))))
                strategies.append(f'array_contains_all({fname}, {json.dumps(subset)})')

            # 策略 4: [新增] array_contains_any (子集+噪音)
            if valid_items:
                subset = [random.choice(valid_items)]
                # 添加一个肯定不存在的值作为噪音
                subset.append("fake_val" if isinstance(subset[0], str) else 999999)
                strategies.append(f'array_contains_any({fname}, {json.dumps(subset)})')

            # 策略 5: [新增] 数组元素直接索引 (Milvus 支持 array[0])
            if length > 0:
                idx = random.randint(0, length - 1)
                item = val[idx]
                if isinstance(item, (int, float)):
                    strategies.append(f'{fname}[{idx}] == {item}')
                elif isinstance(item, str):
                    strategies.append(f'{fname}[{idx}] == {json.dumps(item)}')

            if strategies:
                return random.choice(strategies)
            return f"{fname} is not null"

        # 【关键修复】兜底：对于未覆盖的类型或高级方法返回None的情况
        # 使用基于当前行数据的简单表达式
        pivot_id = row.get("id")
        if pivot_id is not None:
            return self.gen_multi_field_true_expr(row, min_fields=2)
        
        # 最终兜底：使用当前字段的简单比较
        if ftype == DataType.INT64:
            return f"{fname} == {int(val)}"
        elif ftype == DataType.DOUBLE:
            return f"{fname} >= {float(val) - 0.001}"
        elif ftype == DataType.VARCHAR:
            safe_val = str(val).replace('"', '\\"')
            return f'{fname} == "{safe_val}"'
        else:
            return f"{fname} is not null"


    # ===== 【新增】边界值生成器 =====

    def _gen_boundary_int(self, fname, val):
        val = int(val)
        # 只有当 val 满足特定条件时，才生成特定边界查询，否则回退到通用逻辑
        strategies = []

        # 1. 恒等/范围 (基础)
        strategies.append(f"{fname} == {val}")
        strategies.append(f"{fname} >= {val}")
        strategies.append(f"{fname} <= {val}")
        # 1.1 逻辑否定（与 >=/<= 等价），增强操作符多样性
        strategies.append(f"not ({fname} < {val})")
        strategies.append(f"not ({fname} > {val})")

        # 2. 0 的特殊处理
        if val == 0:
            strategies.append(f"{fname} == 0")
        else:
            strategies.append(f"{fname} != 0") # 非 0 值肯定不等于 0

        # 3. 符号处理
        if val > 0:
            strategies.append(f"{fname} > -1")
        elif val < 0:
            strategies.append(f"{fname} < 1")

        # 4. ±1 范围覆盖 (强力推荐)
        # 构造一个刚好包围 val 的区间
        strategies.append(f"({fname} > {val - 1} and {fname} < {val + 1})")

        # 5. 排除法
        fake = val + random.choice([-100, 100])
        strategies.append(f"{fname} != {fake}")

        return random.choice(strategies)

    def _gen_boundary_float(self, fname, val):

        val = float(val)
        strategies = []

        # 1. 基础范围 (浮点数不建议用 ==)
        epsilon = 1e-5
        strategies.append(f"({fname} > {val - epsilon} and {fname} < {val + epsilon})")

        # 2. 宽松范围
        strategies.append(f"{fname} >= {val - epsilon}")
        strategies.append(f"{fname} <= {val + epsilon}")

        # 3. 排除明显错误的值
        strategies.append(f"{fname} != {val + 1.0}")

        # 4. 绝对值/符号逻辑
        if val > 0:
            strategies.append(f"{fname} > -0.0001")
        elif val < 0:
            strategies.append(f"{fname} < 0.0001")

        # 5. 逻辑否定（与 >=/<= 等价），增强操作符多样性
        strategies.append(f"not ({fname} > {val})")
        strategies.append(f"not ({fname} < {val})")

        return random.choice(strategies)

    def _gen_boundary_str(self, fname, val):

        val = str(val)
        if '"' in val:
            val = val.replace('"', '\\"')
        strategies = []

        # 1. 等值
        strategies.append(f'{fname} == "{val}"')

        # 2. 包含逻辑
        if len(val) > 0:
            strategies.append(f'{fname} >= "{val}"')
            strategies.append(f'{fname} <= "{val}"')
            strategies.append(f'{fname} != "{val}_fake"')

            # Like 前缀
            prefix = val[:random.randint(1, len(val))]
            # 只有当前缀不包含特殊字符时才用 like，防止意外匹配错
            if "%" not in prefix and "_" not in prefix:
                strategies.append(f'{fname} like "{prefix}%"')
                # 额外：后缀单字符匹配（始终为真，因为字符串必以其最后一个字符结尾）
            suffix_char = val[-1]
            if suffix_char not in ["%", "_"]:
                strategies.append(f'{fname} like "%{suffix_char}"')

        # 3. 列表包含
        dummies = [self._random_string(5) for _ in range(3)]
        candidates = dummies + [val]
        random.shuffle(candidates)
        in_list = ", ".join([f'"{s}"' for s in candidates])
        strategies.append(f'{fname} in [{in_list}]')

        return random.choice(strategies)


    def gen_complex_noise(self, depth, pivot_id=None):
        """生成复杂噪声表达式，pivot_id用于兜底"""
        # 基于pivot_id的兜底表达式
        fallback = f"id == {pivot_id}" if pivot_id is not None else "id >= 0"
        
        if depth <= 0:
            if random.random() < 0.8:
                res = self.gen_atomic_expr()
                # gen_atomic_expr 返回 (expr, mask)，可能为 None
                if res and res[0] and res[0].strip():
                    return res[0]
                return fallback
            return fallback

        op_type = random.choice(["and", "or", "not", "nested"])

        # 递归调用
        sub_1 = self.gen_complex_noise(depth - 1, pivot_id)
        if not sub_1 or not sub_1.strip(): sub_1 = fallback

        if op_type == "not":
            return f"(not ({sub_1}))"

        elif op_type == "nested":
            return f"({sub_1})"

        elif op_type == "and" or op_type == "or":
            sub_2 = self.gen_complex_noise(depth - 1, pivot_id)
            if not sub_2 or not sub_2.strip(): sub_2 = fallback

            return f"({sub_1} {op_type} {sub_2})"

        return fallback

    def _gen_deep_json_path(self, fname, json_obj):
        """
        融合版 JSON 路径生成器 (V3 - Phantom Container Fix)：
        1. 修复了全 Null 容器 ({"a": null}, [null]) 被误判为 exists=True 的 Bug。
        """
        # 0. 基础校验与类型转换
        if json_obj is None:
            return f"{fname} is null"

        if hasattr(json_obj, "tolist"): json_obj = json_obj.tolist()
        if hasattr(json_obj, "item"): json_obj = json_obj.item()

        current = json_obj
        path_str = ""

        # --- 1. 深度优先下钻 ---
        max_depth = random.randint(1, 12)
        depth = 0

        while depth < max_depth:
            # A. Dict 下钻
            if isinstance(current, dict) and len(current) > 0:
                k = random.choice(list(current.keys()))
                safe_k = k.replace('"', '\\"')
                path_str += f'["{safe_k}"]'
                current = current[k]
                depth += 1

                # 【关键修正】：只有当当前容器包含有效内容时，才允许随机停止测试 exists
                # 如果是 {"a": null}，禁止停止，必须钻到底
                if self._has_valid_content(current) and random.random() < 0.2:
                    break

            # B. List 下钻
            elif isinstance(current, list) and len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                path_str += f'[{idx}]'
                current = current[idx]
                depth += 1
                # List 同样适用：如果是 [null]，禁止停止

            # C. 标量或空 -> 停止
            else:
                break

        full_field = f"{fname}{path_str}"

        # --- 2. 生成断言 ---

        if hasattr(current, "item"): current = current.item()

        # Case 1: Null 值 (最终钻到了具体的 Null)
        if current is None:
            return f"{full_field} is null"

        # Case 2: 复杂类型 (Dict/List) 停在了中间
        if isinstance(current, (dict, list)):
            if len(current) == 0:
                return f"{full_field} is not null"

            # 【双重保险】：只有包含有效标量，才生成 exists
            if self._has_valid_content(current):
                return f'exists({full_field})'
            else:
                # 理论上 While 循环的逻辑应避免进入这里。
                # 但如果到了这里（例如 max_depth 耗尽），且内容全是 null，
                # 我们只能放弃生成 exists，因为它在 Milvus 里是 False。
                # 我们可以尝试生成 is null (针对容器本身)，但这很危险。
                # 最安全的是放弃本次生成。
                return None

        # Case 3: 标量值 (Scalar) -> 调用高强度边界测试
        if isinstance(current, bool):
            val_str = str(current).lower()
            if random.random() < 0.5:
                return f"{full_field} == {val_str}"
            else:
                return f"{full_field} != {str(not current).lower()}"

        elif isinstance(current, int):
            return self._gen_boundary_int(full_field, current)

        elif isinstance(current, float):
            return self._gen_boundary_float(full_field, current)

        elif isinstance(current, str):
            return self._gen_boundary_str(full_field, current)

        return None

    def gen_arithmetic_expr(self, fname, val):
        """
        针对数值类型生成算术运算查询。
        目标：测试 +, -, *, %, ** 是否导致 Crash 或计算错误。
        """
        if not isinstance(val, (int, float)): return None

        is_float = isinstance(val, float)

        if is_float:
            ops = ["+", "-", "*"] # 浮点数不测取模
        else:
            ops = ["+", "-", "*", "%"]
        op = random.choice(ops)

        # 策略 A: 字段 op 常量
        operand = 0
        if op == "%":
            if val == 0: return None
            operand = random.randint(1, max(2, abs(int(val))))
        else:
            operand = random.randint(1, 100)

        # 计算预期结果 (Python)
        try:
            if op == "+": res = val + operand
            elif op == "-": res = val - operand
            elif op == "*": res = val * operand
            elif op == "%": res = val % operand
        except:
            return None

        # 构造 Milvus 表达式
        # 注意：浮点数比较需要用范围
        if isinstance(res, float):
            return f"({fname} {op} {operand} > {res - 0.001} and {fname} {op} {operand} < {res + 0.001})"
        else:
            # 整数可以直接比较，也可以偶尔用 != 混淆
            if random.random() < 0.5:
                return f"({fname} {op} {operand} == {res})"
            else:
                return f"({fname} {op} {operand} != {res + 9999})" # 必真逻辑：结果肯定不等于一个错误值

    def gen_advanced_like(self, fname, val):
        """
        针对字符串生成复杂的 LIKE 查询。
        覆盖: 前缀、后缀、包含、单字符通配符(_)。
        """
        # 基础检查和预处理
        if not isinstance(val, str):
            return None

        # 简化处理：长度太短或包含复杂转义字符时，使用相等查询
        if len(val) < 3:
            return None

    # 【关键修正】处理特殊字符：引号、反斜杠和通配符
    # 如果包含这些特殊字符，直接使用相等查询避免复杂转义
        if any(c in val for c in ['"', "'", "\\", "%", "_"]):
            safe_val = val.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
            return f'{fname} == "{safe_val}"'

        # 随机选择 LIKE 策略
        strategy = random.choice(["prefix", "suffix", "contains", "underscore", "mixed"])

        # 根据策略生成模式
        try:
            if strategy == "prefix":
                # "abcde" -> "ab%"
                cut_len = random.randint(1, len(val) - 1)
                pat = val[:cut_len] + "%"

            elif strategy == "suffix":
                # "abcde" -> "%de"
                cut_len = random.randint(1, len(val) - 1)
                pat = "%" + val[-cut_len:]

            elif strategy == "contains":
                # "abcde" -> "%bc%"
                if len(val) >= 4:
                    start = random.randint(0, len(val) - 3)
                    end = random.randint(start + 2, len(val))
                    pat = "%" + val[start:end] + "%"
                else:
                    pat = val[:len(val)-1] + "%"

            elif strategy == "underscore":
                # "abcde" -> "ab_de"
                if len(val) >= 2:
                    pos = random.randint(0, len(val) - 1)
                    pat = val[:pos] + "_" + val[pos+1:]
                else:
                    pat = "_"

            else:  # mixed
                # "abcde" -> "_b%e"
                if len(val) >= 4:
                    parts = []
                    remaining = val
                    split_point = random.randint(1, len(val) - 2)

                    # 1. 用 _ 替换首字符: "A..." -> "_..."
                    parts.append("_" + remaining[1:split_point])

                    # 2. 中间部分，50%概率插入 %
                    if random.choice([True, False]):
                        parts.append(remaining[split_point:-1])
                    else:
                        parts.append(remaining[split_point:-1] + "%")

                    # 3. 尾字符
                    parts.append(remaining[-1])
                    pat = "".join(parts)
                else:
                    pat = val[0] + "%" + val[-1] if len(val) >= 2 else "%"

        except Exception:
            # 兜底：任何生成错误都回退到相等
            safe_val = val.replace('"', '\\"')
            return f'{fname} == "{safe_val}"'

        # 安全转义构造出的模式
        safe_pat = pat.replace('"', '\\"')

        # 确保模式有效
        if not safe_pat.strip("%_"):
            safe_val = val.replace('"', '\\"')
            return f'{fname} == "{safe_val}"'

        return f'{fname} like "{safe_pat}"'

    def _gen_pqs_json(self, fname, json_obj):
        """
        PQS JSON 主入口：
        1. 尝试 json_contains (针对 List)
        2. 尝试 json_contains_all/any (针对 List)
        3. 回退到 Deep Path (针对 Dict 或 List 下钻)
        """
        # 基础检查
        if json_obj is None: return f"{fname} is null"
        if not isinstance(json_obj, dict): return f"{fname} is not null"

        # --- 策略 1: 针对 List 的包含测试 (保留原逻辑) ---
        list_keys = [k for k, v in json_obj.items() if isinstance(v, list) and len(v) > 0]

        # 40% 概率测 json_contains，前提是有列表
        if list_keys and random.random() < 0.4:
            k = random.choice(list_keys)
            safe_k = k.replace('"', '\\"')
            val_list = json_obj[k]

            # 提取有效标量
            valid_items = []
            for x in val_list:
                if x is not None and isinstance(x, (int, float, str, bool)):
                    valid_items.append(x)
                elif hasattr(x, "item"):
                    valid_items.append(x.item())

            if valid_items:
                target = random.choice(valid_items)

                # 随机选择操作符
                op_type = random.choice(["contains", "contains_all", "contains_any"])

                if op_type == "contains":
                    val_str = json.dumps(target)
                    return f'json_contains({fname}["{safe_k}"], {val_str})'

                elif op_type == "contains_all" and len(valid_items) >= 2:
                    subset = random.sample(valid_items, min(len(valid_items), 2))
                    return f'json_contains_all({fname}["{safe_k}"], {json.dumps(subset)})'

                elif op_type == "contains_any":
                    subset = [target, "fake_val_999"] # 混入噪音
                    return f'json_contains_any({fname}["{safe_k}"], {json.dumps(subset)})'

        # --- 策略 2: 深度下钻 (调用上面的融合版函数) ---
        # 这里会触发 _gen_deep_json_path -> 进而触发 _gen_boundary_xxx
        res = self._gen_deep_json_path(fname, json_obj)
        if res: return res

        # 兜底
        return f"{fname} is not null"

def run_pqs_mode(rounds=100, seed=None, enable_dynamic_ops=True):
    global INDEX_TYPE, CURRENT_INDEX_TYPE, VECTOR_CHECK_RATIO, VECTOR_TOPK, _GLOBAL_ID_COUNTER, METRIC_TYPE
    
    if seed is not None:
        print(f"\n🔒 PQS模式使用固定种子 {seed}")
        random.seed(seed)
        np.random.seed(seed)
    else:
        # PQS 模式也需要生成种子用于可重复性
        seed = random.randint(0, 2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        print(f"\n🎲 PQS模式随机生成种子: {seed}")

    # 【关键】重置 ID 计数器，保证完全可重复
    _GLOBAL_ID_COUNTER = 0
    DataManager._id_counter = 0

    # 【关键】在种子设置之后初始化随机变量，保证可重复性
    INDEX_TYPE = random.choice(ALL_INDEX_TYPES)
    CURRENT_INDEX_TYPE = INDEX_TYPE
    METRIC_TYPE = random.choice(ALL_METRIC_TYPES)
    VECTOR_CHECK_RATIO = random.uniform(0.2, 0.8)
    VECTOR_TOPK = random.randint(50, 200)
    print(f"   索引类型: {INDEX_TYPE}, 度量类型: {METRIC_TYPE}")

    # --- 日志设置 ---
    timestamp = int(time.time())
    log_filename = f"pqs_test_{timestamp}.log"
    print("\n" + "="*60)
    print(f"🚀 启动 PQS (Pivot Query Synthesis) 模式测试")
    print(f"📄 详细日志将写入: {log_filename}")
    print("="*60)

    # 1. 初始化
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()

    mm = MilvusManager()
    mm.connect()
    mm.reset_collection(dm.schema_config)
    mm.insert(dm)

    pqs_gen = PQSQueryGenerator(dm)
    errors = []

    # 定义一个辅助函数：安全地转换数据用于打印（处理 Numpy 和截断长数组）
    def safe_format_row(row_series):
        row_dict = row_series.to_dict()
        safe_data = {}
        for k, v in row_dict.items():
            # 排除向量，太长了
            if k == "vector": continue

            # 处理 Numpy 标量
            if hasattr(v, "item"):
                v = v.item()

            # 处理 NaN/None
            if v is None or (isinstance(v, float) and np.isnan(v)):
                safe_data[k] = None
                continue

            # 处理超长列表/数组 (截断)
            if isinstance(v, (list, np.ndarray)) and len(v) > 20:
                safe_data[k] = f"<Array length={len(v)}...>"
            else:
                safe_data[k] = v
        return safe_data

    with open(log_filename, "w", encoding="utf-8") as f:
        def file_log(msg):
            f.write(msg + "\n")
            f.flush()

        file_log(f"PQS Test Started | Rounds: {rounds} | Seed: {seed}")
        file_log("=" * 80)

        for i in range(rounds):
            # --- 【动态数据变动】每轮有 20% 概率触发 ---
            if enable_dynamic_ops and i > 0 and random.random() < 0.2:
                op = random.choices(["insert", "delete", "upsert"], weights=[0.4, 0.4, 0.2], k=1)[0]
                batch_count = random.randint(1, 5)

                if op == "insert":
                    new_rows = []
                    rows_with_vec = []
                    for _ in range(batch_count):
                        row = dm.generate_single_row()
                        vec = dm.generate_single_vector()
                        row_with_vec = row.copy()
                        row_with_vec["vector"] = vec
                        new_rows.append(row)
                        rows_with_vec.append(row_with_vec)
                    try:
                        mm.col.insert(rows_with_vec)
                        mm.col.flush()
                        dm.df = pd.concat([dm.df, pd.DataFrame(new_rows)], ignore_index=True)
                        dm.vectors = np.vstack([dm.vectors, np.array([r["vector"] for r in rows_with_vec])])
                        file_log(f"[Dynamic] Inserted {len(new_rows)} rows")
                    except Exception as e:
                        file_log(f"[Dynamic] Insert failed: {e}")

                elif op == "delete" and len(dm.df) > 100:
                    del_count = min(batch_count, len(dm.df) - 100)
                    del_ids = random.sample(dm.df["id"].tolist(), del_count)
                    try:
                        expr = f"id in {del_ids}"
                        mm.col.delete(expr)
                        mm.col.flush()
                        idx = dm.df[dm.df["id"].isin(del_ids)].index.to_numpy()
                        dm.df = dm.df.drop(idx).reset_index(drop=True)
                        dm.vectors = np.delete(dm.vectors, idx, axis=0)
                        file_log(f"[Dynamic] Deleted {len(del_ids)} rows")
                    except Exception as e:
                        file_log(f"[Dynamic] Delete failed: {e}")

                else:  # upsert
                    upsert_rows = []
                    rows_with_vec = []
                    updated_ids = []
                    new_rows_list = []
                    new_vectors = []
                    for _ in range(batch_count):
                        use_existing = (not dm.df.empty) and (random.random() < 0.7)
                        if use_existing:
                            target_id = random.choice(dm.df["id"].tolist())
                            updated_ids.append(target_id)
                        else:
                            target_id = generate_unique_id()
                        row = dm.generate_single_row(id_override=target_id)
                        vec = dm.generate_single_vector()
                        row_with_vec = row.copy()
                        row_with_vec["vector"] = vec
                        upsert_rows.append(row)
                        rows_with_vec.append(row_with_vec)
                    try:
                        mm.col.upsert(rows_with_vec)
                        mm.col.flush()
                        for row, row_with_vec in zip(upsert_rows, rows_with_vec):
                            rid = row["id"]
                            match_idx = dm.df.index[dm.df["id"] == rid].tolist()
                            if match_idx:
                                idx = match_idx[0]
                                for k, v in row.items():
                                    dm.df.at[idx, k] = v
                                dm.vectors[idx] = row_with_vec["vector"]
                            else:
                                new_rows_list.append(row)
                                new_vectors.append(row_with_vec["vector"])
                        if new_rows_list:
                            dm.df = pd.concat([dm.df, pd.DataFrame(new_rows_list)], ignore_index=True)
                            dm.vectors = np.vstack([dm.vectors, np.array(new_vectors)])
                        file_log(f"[Dynamic] Upserted {len(upsert_rows)} rows")
                    except Exception as e:
                        file_log(f"[Dynamic] Upsert failed: {e}")

            random_idx = random.randint(0, len(dm.df) - 1)
            pivot_row = dm.df.iloc[random_idx]
            pivot_id = pivot_row["id"]

            expr = ""
            for _ in range(5):
                try:
                    expr = pqs_gen.gen_pqs_expr(pivot_row, depth=random.randint(1, 13))
                    if expr and expr.strip(): break
                except: pass
            # 兜底：如果表达式为空或全空格，使用多字段兜底表达式
            if not expr or not expr.strip():
                expr = pqs_gen.gen_multi_field_true_expr(pivot_row, min_fields=2)
                file_log(f"[Round {i}] Used fallback expr (complex): {expr}")

            print(f"\r🔍 [Round {i+1}/{rounds}] Check ID: {pivot_id}...", end="", flush=True)

            log_header = f"[Round {i}] Target ID: {pivot_id}"
            file_log(f"\n{log_header}")
            file_log(f"  Expr: {expr}")

            try:
                start_t = time.time()
                res = mm.col.query(expr, output_fields=["id"], limit=10000)
                cost = (time.time() - start_t) * 1000
                found_ids = set([r["id"] for r in res])

                # --- PQS 验证 ---
                if pivot_id in found_ids:
                    file_log(f"  -> PASS | Found: {len(found_ids)} hits | Time: {cost:.2f}ms")
                else:
                    # ❌ 失败！(False Negative)
                    # 1. 准备打印数据
                    safe_row = safe_format_row(pivot_row)
                    json_data = safe_row.get("meta_json", {})
                    scalar_data = {k: v for k, v in safe_row.items() if k != "meta_json"}

                    # 2. 控制台醒目打印
                    print(f"\n\n❌ PQS ERROR DETECTED [Round {i}]")
                    print(f"   Target ID: {pivot_id}")
                    print(f"   Expression: {expr}")
                    print(f"   Found Count: {len(found_ids)} (Target NOT found)")
                    print("-" * 50)
                    print(f"   🔎 EVIDENCE (Target Row Data):")

                    # 漂亮地打印 JSON 字段 (通常是问题核心)
                    print(f"   [meta_json]:")
                    print(json.dumps(json_data, indent=4, ensure_ascii=False))

                    # 打印其他标量字段
                    print(f"   [Scalars]:")
                    for k, v in scalar_data.items():
                        print(f"     {k:<10}: {v}")
                    print("-" * 50)

                    # 3. 记录日志
                    file_log(f"  -> ❌ FAIL! Target ID {pivot_id} NOT found.")
                    file_log(f"  -> Row Data: {safe_row}")

                    errors.append({"id": pivot_id, "expr": expr})

            except Exception as e:
                print(f"\n\n⚠️ Execution Error [Round {i}]: {e}")
                file_log(f"  -> EXECUTION ERROR: {e}")

    print("\n" + "="*60)
    if not errors:
        print(f"✅ PQS 测试完成。未发现错误。")
    else:
        print(f"🚫 PQS 测试完成。发现 {len(errors)} 个潜在 Bug！")
        print(f"📄 详细数据已记录至日志: {log_filename}")

def run_groupby_test(rounds=50, seed=None, enable_dynamic_ops=True):
    """
    专门用于检测GroupBy Key分裂和 strict_group_size 失效的问题
    """
    global METRIC_TYPE, INDEX_TYPE, CURRENT_INDEX_TYPE, VECTOR_CHECK_RATIO, VECTOR_TOPK, _GLOBAL_ID_COUNTER
    if seed is not None:
        print(f"\n🔒 GroupBy 模式使用固定种子 {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    # 【关键】初始化随机变量
    _GLOBAL_ID_COUNTER = 0
    DataManager._id_counter = 0
    INDEX_TYPE = random.choice(ALL_INDEX_TYPES)
    CURRENT_INDEX_TYPE = INDEX_TYPE
    METRIC_TYPE = random.choice(ALL_METRIC_TYPES)
    print(f"   索引类型: {INDEX_TYPE}, 度量类型: {METRIC_TYPE}")
    
    timestamp = int(time.time())
    log_filename = f"groupby_test_{timestamp}.log"
    print("\n" + "="*60)
    print(f"📊 启动 GroupBy 逻辑专项测试")
    print(f"   日志: {log_filename}")
    print("="*60)

    # 1. 初始化 (复用现有的 Manager)
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data() # 注意：这里已经应用了你刚才修改的无Null逻辑

    mm = MilvusManager()
    mm.connect()
    mm.reset_collection(dm.schema_config)
    mm.insert(dm)

    # 识别可用于分组的字段
    # 1. 所有的 Int/String/Bool 标量字段
    # 2. JSON 字段中的常见 Key (我们在 DataManager 里写死的那些)
    potential_group_fields = []
    
    # A. 标量字段
    for f in dm.schema_config:
        if f["type"] in [DataType.INT64, DataType.VARCHAR, DataType.BOOL]:
            potential_group_fields.append(f["name"])
    
    # B. JSON 字段 (硬编码 DataManager 中生成的 Key)
    # DataManager 生成逻辑中包含: price, color, active, config["version"]
    json_fields = [f["name"] for f in dm.schema_config if f["type"] == DataType.JSON]
    for jf in json_fields:
        potential_group_fields.append(f'{jf}["color"]')   # 字符串，高频重复，适合分组
        potential_group_fields.append(f'{jf}["active"]')  # Bool，只有两组
        potential_group_fields.append(f'{jf}["price"]')   # Int，值较多

    errors = []

    with open(log_filename, "w", encoding="utf-8") as f:
        def file_log(msg):
            f.write(msg + "\n")
            f.flush()
        
        file_log(f"GroupBy Test Started | Rounds: {rounds}")

        for i in range(rounds):
            print(f"\r📊 GroupBy Test {i+1}/{rounds}...", end="", flush=True)

            # --- 【动态数据变动】每轮有 20% 概率触发 ---
            if enable_dynamic_ops and i > 0 and random.random() < 0.2:
                op = random.choices(["insert", "delete", "upsert"], weights=[0.4, 0.4, 0.2], k=1)[0]
                batch_count = random.randint(1, 5)
                if op == "insert":
                    new_rows = []
                    rows_with_vec = []
                    for _ in range(batch_count):
                        row = dm.generate_single_row()
                        vec = dm.generate_single_vector()
                        row_with_vec = row.copy()
                        row_with_vec["vector"] = vec
                        new_rows.append(row)
                        rows_with_vec.append(row_with_vec)
                    try:
                        mm.col.insert(rows_with_vec)
                        mm.col.flush()
                        dm.df = pd.concat([dm.df, pd.DataFrame(new_rows)], ignore_index=True)
                        dm.vectors = np.vstack([dm.vectors, np.array([r["vector"] for r in rows_with_vec])])
                        file_log(f"[Dynamic] Inserted {len(new_rows)} rows")
                    except Exception as e:
                        file_log(f"[Dynamic] Insert failed: {e}")
                elif op == "delete" and len(dm.df) > 100:
                    del_count = min(batch_count, len(dm.df) - 100)
                    del_ids = random.sample(dm.df["id"].tolist(), del_count)
                    try:
                        expr = f"id in {del_ids}"
                        mm.col.delete(expr)
                        mm.col.flush()
                        idx = dm.df[dm.df["id"].isin(del_ids)].index.to_numpy()
                        dm.df = dm.df.drop(idx).reset_index(drop=True)
                        dm.vectors = np.delete(dm.vectors, idx, axis=0)
                        file_log(f"[Dynamic] Deleted {len(del_ids)} rows")
                    except Exception as e:
                        file_log(f"[Dynamic] Delete failed: {e}")
                else:  # upsert
                    upsert_rows = []
                    rows_with_vec = []
                    updated_ids = []
                    new_rows_list = []
                    new_vectors = []
                    for _ in range(batch_count):
                        use_existing = (not dm.df.empty) and (random.random() < 0.7)
                        if use_existing:
                            target_id = random.choice(dm.df["id"].tolist())
                            updated_ids.append(target_id)
                        else:
                            target_id = generate_unique_id()
                        row = dm.generate_single_row(id_override=target_id)
                        vec = dm.generate_single_vector()
                        row_with_vec = row.copy()
                        row_with_vec["vector"] = vec
                        upsert_rows.append(row)
                        rows_with_vec.append(row_with_vec)
                    try:
                        mm.col.upsert(rows_with_vec)
                        mm.col.flush()
                        for row, row_with_vec in zip(upsert_rows, rows_with_vec):
                            rid = row["id"]
                            match_idx = dm.df.index[dm.df["id"] == rid].tolist()
                            if match_idx:
                                idx = match_idx[0]
                                for k, v in row.items():
                                    dm.df.at[idx, k] = v
                                dm.vectors[idx] = row_with_vec["vector"]
                            else:
                                new_rows_list.append(row)
                                new_vectors.append(row_with_vec["vector"])
                        if new_rows_list:
                            dm.df = pd.concat([dm.df, pd.DataFrame(new_rows_list)], ignore_index=True)
                            dm.vectors = np.vstack([dm.vectors, np.array(new_vectors)])
                        file_log(f"[Dynamic] Upserted {len(upsert_rows)} rows")
                    except Exception as e:
                        file_log(f"[Dynamic] Upsert failed: {e}")

            # --- 1. 构造测试参数 ---
            group_field = random.choice(potential_group_fields)
            group_size = random.randint(1, 5)
            limit_groups = random.randint(2, 20) # 想要返回多少个组
            strict = random.choice([True, False])
            
            # 随机挑选一个向量进行搜索
            q_idx = random.randint(0, len(dm.vectors) - 1)
            q_vec = dm.vectors[q_idx].tolist()

            # 构造 Output Fields (必须包含分组字段本身，否则无法验证)
            # 如果是 JSON 嵌套字段 (meta["color"])，output_fields 只能填 meta
            output_fields = ["id"]
            if "[" in group_field:
                base_field = group_field.split("[")[0]
                output_fields.append(base_field)
            else:
                output_fields.append(group_field)

            # --- 2. 执行 Milvus Search ---
            current_metric_type = METRIC_TYPE
            try:
                for idx in mm.col.indexes:
                    if idx.field_name == "vector":
                        current_metric_type = idx.params.get("metric_type", METRIC_TYPE)
                        break
            except:
                pass

            idx_params_dict = {}
            if INDEX_TYPE == "HNSW":
                idx_params_dict = {"ef": 64}
            elif INDEX_TYPE.startswith("IVF"):
                idx_params_dict = {"nprobe": 10}
            
            search_params = {
                "data": [q_vec],
                "anns_field": "vector",
                "param": {"metric_type": current_metric_type, "params": idx_params_dict},
                "limit": limit_groups, 
                "group_by_field": group_field,
                "group_size": group_size,
                "strict_group_size": strict,
                "output_fields": output_fields
            }

            try:
                # 注意：PyMilvus 2.4+ 支持 group_by_field
                res = mm.col.search(**search_params)
            except Exception as e:
                # 某些旧版本或特殊配置可能不支持，记录并跳过
                file_log(f"[Round {i}] Search Failed: {e}")
                continue

            # --- 3. 验证逻辑 ---
            if not res or len(res) == 0:
                continue

            hits = res[0] # 单向量搜索，取第一个结果集
            
            # 提取所有组的 Key 值
            seen_group_keys = set()
            group_keys_list = []
            
            # PyMilvus 返回的 GroupBy 结果通常是一个迭代器，每个 item 代表一个 Entity
            # 但在 GroupBy 模式下，返回的数据结构可能扁平化或者分层
            # 通常：Hits 包含 N 个 entity，每个 entity 属于某个组。
            # ⚠️ 关键：Milvus SDK 在 GroupBy 时，res[0] 通常返回的是“打平”的列表，
            # 或者是按组聚合。我们需要根据实际返回验证。
            # 假设返回的是打平的 top entity (每组 group_size 个)
            
            # 为了准确获取 Group Key，我们需要解析 output_fields
            def get_val_from_hit(hit, g_field):
                # hit.entity.get()
                if "[" in g_field:
                    # JSON 处理: meta["color"] -> entity.meta -> ["color"]
                    base, path = g_field.split("[", 1)
                    path = path.rstrip("]").strip('"').strip("'")
                    json_val = hit.entity.get(base)
                    if isinstance(json_val, dict):
                        return json_val.get(path)
                    return None
                else:
                    return hit.entity.get(g_field)

            # 遍历检查
            # 由于 PyMilvus 并没有显式暴露 "Group对象"，它是把所有组的结果拼在一起返回
            # 或者在某些版本里是 list of hits。
            # 我们主要检查：
            # 同样的值，被判定为不同的组？
            # 这在 client 端很难直接探测，因为 client 拿到的是 entity list。
            # 但我们可以检查：如果我们自己在 Client 端按 Key 分组，组的数量是否 > limit？
            
            # 更简单的验证：Strict Group Size
            # 我们统计每个 Key 出现的次数
            key_counts = {}
            for hit in hits:
                val = get_val_from_hit(hit, group_field)
                # JSON 可能返回 List 等不可哈希类型，转字符串
                val_key = str(val) 
                
                key_counts[val_key] = key_counts.get(val_key, 0) + 1

            # 🔍 CHECK 1: Limit 检查
            # 返回的不同 Key 的数量（即组的数量）不应超过 limit
            if len(key_counts) > limit_groups:
                msg = f"❌ Limit Exceeded: Asked for {limit_groups} groups, got {len(key_counts)}"
                print(f"\n{msg}")
                print(f"   ⚠️ Field: {group_field}")
                print(f"   ⚠️ Returned Keys: {list(key_counts.keys())}")
                # 检查是否有重复但类型不同的 Key (例如 '7' 和 7)
                raw_keys = [str(k) for k in key_counts.keys()]
                if len(raw_keys) != len(set(raw_keys)):
                     print(f"   🕵️‍♀️ DETECTED DUPLICATE STRING REPRESENTATION! (Type Confusion likely)")
                file_log(f"[Round {i}] {msg}")
                errors.append(msg)

            # 🔍 CHECK 2: Strict Group Size 检查
            # 如果 strict=True，且我们确定数据库里数据够多（DataGen生成的数据分布很均匀）
            # 那么每个 Key 的 count 应该等于 group_size
            if strict:
                for k, count in key_counts.items():
                    # 容错：如果 DataGen 随机生成导致该 key 总数真的很少，忽略
                    # 但如果是 "Red", "Blue" 这种 DataGen 生成了几百个的，不应少于 group_size
                    if count < group_size:
                        # 这是一个可疑点 (Warning)
                        # 为了确认是 Bug 还是数据真不够，我们可以去 pandas 查一下真实 count
                        # 这里为了性能，只记录 Warning
                        # msg = f"⚠️ Strict Size Warning: Group '{k}' has {count} items, expected {group_size}"
                        # file_log(f"[Round {i}] {msg}")
                        pass
                    
                    if count > group_size:
                        msg = f"❌ Strict Size Violation: Group '{k}' has {count} items, expected max {group_size}"
                        print(f"\n{msg}")
                        file_log(f"[Round {i}] {msg}")
                        errors.append(msg)

            file_log(f"[Round {i}] Field: {group_field} | Groups: {len(key_counts)} | Strict: {strict} -> PASS")

    print("\n" + "="*60)
    if not errors:
        print(f"✅ GroupBy 测试完成。未发现显式逻辑违规。")
    else:
        print(f"🚫 GroupBy 测试发现 {len(errors)} 个问题！请查看日志。")

# 如果你想直接运行这个模式，可以在下面调用
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus Fuzz Oracle")
    
    # Common arguments
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--rounds", type=int, default=1000, help="Number of rounds for main/test modes")
    parser.add_argument("--collection", type=str, default="fuzz_stable_v3", help="Milvus collection name")
    parser.add_argument("--no-dynamic-ops", action="store_true", help="Disable dynamic operations (insert/delete/upsert)")
    
    # Chaos engineering
    parser.add_argument("--chaos", action="store_true", help="Enable default chaos rate (0.1)")
    parser.add_argument("--chaos-rate", type=float, default=0.0, help="Set custom chaos rate (0.0 - 1.0)")
    parser.add_argument("--metric", type=str, choices=["L2", "IP", "COSINE"], default=None,
                        help="Force a specific metric type (default: random from L2/IP/COSINE)")

    # Execution Modes (Mutually Exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pqs", action="store_true", help="Run PQS (Predicate Query Search) Mode")
    group.add_argument("--equiv", action="store_true", help="Run Equivalence Mode")
    group.add_argument("--groupby-test", action="store_true", help="Run GroupBy Test Mode")

    parser.add_argument("--pqs-rounds", type=int, default=1000, help="Rounds for PQS mode (implies --pqs if set explicitly, but better to use --pqs)")

    args = parser.parse_args()

    # 1. Apply Configuration
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    COLLECTION_NAME = args.collection
    enable_dynamic_ops = not args.no_dynamic_ops

    if args.chaos:
        CHAOS_RATE = 0.1
    if args.chaos_rate > 0:
        CHAOS_RATE = args.chaos_rate

    # 2. Determine Mode and Execute
    print("=" * 80)
    print(f"🚀 Milvus Fuzz Oracle Startup")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Seed: {args.seed if args.seed is not None else '(Random)'}")
    print(f"   Metric: {args.metric if args.metric else '(Random from L2/IP/COSINE)'}")
    print(f"   Dynamic Ops: {enable_dynamic_ops}")
    print(f"   Chaos Rate: {CHAOS_RATE}")

    if args.pqs or (args.pqs_rounds != 1000 and not args.equiv and not args.groupby_test):
        # Implicitly enable PQS if pqs_rounds is modified from default, or explicit --pqs
        # Note: The logic 'args.pqs_rounds != 1000' is a heuristic for backward compat if the user just set --pqs-rounds
        print(f"   Mode: PQS (Predicate Query Search)")
        print(f"   Rounds: {args.pqs_rounds}")
        print("=" * 80)
        run_pqs_mode(rounds=args.pqs_rounds, seed=args.seed, enable_dynamic_ops=enable_dynamic_ops)
    
    elif args.equiv:
        print(f"   Mode: Equivalence Test")
        print(f"   Rounds: {args.rounds}")
        print("=" * 80)
        run_equivalence_mode(rounds=args.rounds, seed=args.seed, enable_dynamic_ops=enable_dynamic_ops)

    elif args.groupby_test:
        print(f"   Mode: GroupBy Test")
        print(f"   Rounds: {args.rounds}")
        print("=" * 80)
        run_groupby_test(rounds=args.rounds, seed=args.seed, enable_dynamic_ops=enable_dynamic_ops)

    else:
        print(f"   Mode: Standard Fuzzing")
        print(f"   Rounds: {args.rounds}")
        print("=" * 80)
        run(rounds=args.rounds, seed=args.seed, enable_dynamic_ops=enable_dynamic_ops)