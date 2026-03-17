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
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient
)
import sys
import argparse

# --- Configuration (User Specified) ---
HOST = "127.0.0.1"
PORT = "19531"           # 你的自定义端口
COLLECTION_NAME = "fuzz_stable_v3"
VECTOR_INDEX_NAME = "vector_idx"  # 显式命名向量索引，避免多索引时 AmbiguousIndexName
N = 5000                 # 数据量（磁盘满了，先用 1000 测试功能）
DIM = 128                # 维度 128
BATCH_SIZE = 200         # 批次大小（内存模式可以大一些）
SLEEP_INTERVAL = 0.01    # 每次插入后暂停 10ms（内存模式更快）
FLUSH_INTERVAL = 500     # 每 500 条刷盘

# 稳定的索引类型列表（移除不稳定或需要特殊配置的索引）
ALL_INDEX_TYPES = [
    "FLAT", "HNSW", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "DISKANN"
]
# 注意：INDEX_TYPE 等随机变量移到 run() 内部，在种子设置后初始化，保证可重复性
INDEX_TYPE = None  # 延迟初始化

# 全局度量类型列表（浮点向量支持的全部 metric）
# L2: 欧氏距离 (越小越相似), IP: 内积 (越大越相似), COSINE: 余弦相似度 (越大越相似)
ALL_METRIC_TYPES = ["L2", "IP", "COSINE"]
METRIC_TYPE = None  # 延迟初始化（在 run() 内种子设置后随机选取）

# 数值类型分组常量
ALL_INT_TYPES = [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64]
ALL_FLOAT_TYPES = [DataType.FLOAT, DataType.DOUBLE]
ALL_NUMERIC_TYPES = ALL_INT_TYPES + ALL_FLOAT_TYPES

# 整数类型的值范围 (用于数据生成和值采样的 clamp)
INT_RANGES = {
    DataType.INT8:  (-128, 127),
    DataType.INT16: (-32768, 32767),
    DataType.INT32: (-2147483648, 2147483647),
    DataType.INT64: (-2**63, 2**63 - 1),
}

# 记录当前索引类型（用于索引重建时避免重复）
CURRENT_INDEX_TYPE = None  # 延迟初始化
VECTOR_CHECK_RATIO = None  # 延迟初始化
VECTOR_TOPK = None         # 延迟初始化

# 标量字段索引类型映射 (根据 Milvus 文档)
# 每种标量字段类型对应可用的索引类型列表
SCALAR_INDEX_TYPES = {
    DataType.INT8:    ["INVERTED", "STL_SORT"],
    DataType.INT16:   ["INVERTED", "STL_SORT"],
    DataType.INT32:   ["INVERTED", "STL_SORT"],
    DataType.INT64:   ["INVERTED", "STL_SORT"],
    DataType.FLOAT:   ["INVERTED"],
    DataType.DOUBLE:  ["INVERTED"],
    DataType.BOOL:    ["BITMAP", "INVERTED"],
    DataType.VARCHAR: ["INVERTED", "BITMAP", "TRIE"],
    # JSON 索引需要 json_cast_type + json_path 参数，配置复杂，暂不支持随机 fuzzing
    # DataType.JSON:  ["INVERTED"],
    DataType.ARRAY:   ["BITMAP", "INVERTED"],
}

# 支持的数组元素类型（用于随机生成多类型数组字段）
ALL_ARRAY_ELEMENT_TYPES = [
    DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
    DataType.FLOAT, DataType.DOUBLE,
    DataType.BOOL, DataType.VARCHAR,
]

# 标量索引创建概率（每个字段有多大概率被创建索引，延迟初始化）
SCALAR_INDEX_PROBABILITY = None

# 混淆开关（默认关闭）。当 >0 时，在 JSON 下钻策略中按该概率触发类型混淆
CHAOS_RATE = 0.0

# Schema Evolution: 最大可动态添加的字段数
MAX_EVOLVED_FIELDS = 5

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

# --- Milvus 空值语义工具 ---
# Milvus 将空 JSON {} 和空数组 [] 视为 null，pandas 不会。
# 此工具函数确保 Oracle 模型与 Milvus 行为一致。
def milvus_is_empty(val):
    """判断一个值在 Milvus 语义下是否为 null（None / NaN / [] / {}）"""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, list) and len(val) == 0:
        return True
    if isinstance(val, dict) and len(val) == 0:
        return True
    return False

def milvus_null_mask(series):
    """返回一个 boolean mask，True 表示该值在 Milvus 中被视为 null。
    用于替代 series.isnull()，额外处理 [] 和 {} 的情况。"""
    return series.apply(milvus_is_empty).astype("boolean")

def milvus_notnull_mask(series):
    """返回一个 boolean mask，True 表示该值在 Milvus 中被视为非 null。
    用于替代 series.notnull()，额外处理 [] 和 {} 的情况。"""
    return (~series.apply(milvus_is_empty)).astype("boolean")

def normalize_empty_to_none(df, schema_config):
    """将 DataFrame 中的空数组 [] 和空 JSON {} 规范化为 None。
    这确保 pandas Oracle 与 Milvus 的空值语义一致。"""
    for fc in schema_config:
        fname = fc["name"]
        if fname not in df.columns:
            continue
        ftype = fc["type"]
        if ftype in (DataType.ARRAY, DataType.JSON):
            df[fname] = df[fname].apply(lambda x: None if milvus_is_empty(x) else x)
    return df

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
            
            # 演进字段: 随机生成值（含一定比例的 null）
            # 在 schema 演进后新插入的数据会携带该字段值
            # null 时不传该字段: Milvus 自动设为 null, pandas 通过 pd.concat 补 NaN
            if fname.startswith("evo_"):
                if random.random() >= 0.3:  # 70% 概率生成真实值
                    row[fname] = self.generate_field_value(field)
                # else: 不放入 row, Milvus 自动 null, pandas NaN
                continue
            
            if ftype in ALL_INT_TYPES:
                lo, hi = INT_RANGES[ftype]
                # 使用较小的范围避免数据过于稀疏
                effective_lo = max(lo, -self.int_range)
                effective_hi = min(hi, self.int_range)
                row[fname] = int(rng.integers(effective_lo, effective_hi))
            elif ftype in ALL_FLOAT_TYPES:
                val = float(rng.random() * self.double_scale)
                if ftype == DataType.FLOAT:
                    val = float(np.float32(val))  # clamp 到 float32 精度
                row[fname] = val
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
                elem_type = field.get("element_type", DataType.INT64)
                arr_len = rng.integers(1, min(self.array_capacity, 10) + 1)
                arr_val = self._gen_array_data(rng, elem_type, arr_len)
                # Milvus 将 [] 视为 null，不放入 row（同 None 处理）
                if arr_val:
                    row[fname] = arr_val
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
            elif r < 0.8: return bool(rng.choice([True, False]))
            else: return None

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

    def _gen_array_data(self, rng, element_type, length):
        """根据元素类型生成数组数据"""
        length = int(length)
        if element_type in ALL_INT_TYPES:
            lo, hi = INT_RANGES.get(element_type, (-100, 100))
            effective_lo = max(lo, -100)
            effective_hi = min(hi, 100)
            return [int(x) for x in rng.integers(effective_lo, effective_hi, size=length)]
        elif element_type in ALL_FLOAT_TYPES:
            vals = rng.random(length) * 200 - 100  # [-100, 100]
            if element_type == DataType.FLOAT:
                vals = vals.astype(np.float32).astype(np.float64)
            return [float(x) for x in vals]
        elif element_type == DataType.BOOL:
            return [bool(rng.choice([True, False])) for _ in range(length)]
        elif element_type == DataType.VARCHAR:
            return [self._random_string(1, 8) for _ in range(length)]
        else:
            return [int(x) for x in rng.integers(0, 100, size=length)]

    def generate_schema(self):
        print("🎲 1. Defining Dynamic Schema...")
        self.schema_config = []
        num_fields = random.randint(3, 25)
        types_pool = ALL_INT_TYPES + ALL_FLOAT_TYPES + [DataType.BOOL, DataType.VARCHAR]

        for i in range(num_fields):
            ftype = random.choice(types_pool)
            self.schema_config.append({"name": f"c{i}", "type": ftype})

        self.schema_config.append({"name": "meta_json", "type": DataType.JSON})

        # 生成1-3个不同元素类型的数组字段（增强覆盖率）
        num_arrays = random.randint(1, 3)
        used_element_types = set()
        for arr_i in range(num_arrays):
            remaining = [t for t in ALL_ARRAY_ELEMENT_TYPES if t not in used_element_types]
            if remaining:
                elem_type = random.choice(remaining)
            else:
                elem_type = random.choice(ALL_ARRAY_ELEMENT_TYPES)
            used_element_types.add(elem_type)
            elem_name = get_type_name(elem_type).lower()
            arr_name = f"arr_{elem_name}_{arr_i}"
            arr_config = {
                "name": arr_name,
                "type": DataType.ARRAY,
                "element_type": elem_type,
                "max_capacity": self.array_capacity
            }
            self.schema_config.append(arr_config)

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

            if ftype in ALL_INT_TYPES:
                lo, hi = INT_RANGES[ftype]
                effective_lo = max(lo, -self.int_range)
                effective_hi = min(hi, self.int_range)
                data[fname] = rng.integers(effective_lo, effective_hi, size=N)
            elif ftype in ALL_FLOAT_TYPES:
                vals = rng.random(N) * self.double_scale
                if ftype == DataType.FLOAT:
                    vals = vals.astype(np.float32).astype(np.float64)  # clamp 精度
                data[fname] = vals
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
                elem_type = field.get("element_type", DataType.INT64)
                arr_list = []
                for _ in range(N):
                    length = random.randint(0, 5)
                    arr_list.append(self._gen_array_data(rng, elem_type, length))
                data[fname] = arr_list

            mask = rng.random(N) < self.null_ratio
            temp_arr = np.array(data[fname], dtype=object)
            temp_arr[mask] = None
            data[fname] = temp_arr

        self.df = pd.DataFrame(data)
        # 【关键】Milvus 将空数组 [] 和空 JSON {} 视为 null
        # 规范化 pandas 数据使 Oracle 模型与 Milvus 一致
        normalize_empty_to_none(self.df, self.schema_config)
        print("✅ Data Generation Complete.")

    # --- Schema Evolution Methods ---

    def generate_field_value(self, field_config):
        """为指定字段配置生成一个随机值（不含 None）。使用全局 random 状态保证可重复性。"""
        ftype = field_config["type"]

        if ftype in ALL_INT_TYPES:
            lo, hi = INT_RANGES[ftype]
            effective_lo = max(lo, -self.int_range)
            effective_hi = min(hi, self.int_range)
            return random.randint(effective_lo, effective_hi)
        elif ftype in ALL_FLOAT_TYPES:
            val = random.random() * self.double_scale
            if ftype == DataType.FLOAT:
                val = float(np.float32(val))
            return float(val)
        elif ftype == DataType.BOOL:
            return random.choice([True, False])
        elif ftype == DataType.VARCHAR:
            return self._random_string(0, random.randint(5, 50))
        elif ftype == DataType.JSON:
            # 演进字段使用简单 JSON 结构，避免复杂嵌套
            base_obj = {
                "evo_val": random.randint(0, 1000),
                "evo_tag": random.choice(["alpha", "beta", "gamma", "delta"]),
                "evo_flag": random.choice([True, False]),
            }
            return base_obj
        elif ftype == DataType.ARRAY:
            elem_type = field_config.get("element_type", DataType.INT64)
            rng = np.random.default_rng(random.randint(0, 2**31))
            arr_len = random.randint(1, min(self.array_capacity, 10))
            return self._gen_array_data(rng, elem_type, int(arr_len))
        return None

    def evolve_schema_add_field(self):
        """
        Schema Evolution: 随机添加一个新字段到 schema 配置和 pandas DataFrame。
        新字段对所有现有数据默认为 None（与 Milvus nullable=True 行为一致）。

        Returns:
            field_config dict if successful, None if limit reached.
        """
        # 统计已演进的字段数
        evolved_count = sum(1 for f in self.schema_config if f["name"].startswith("evo_"))
        if evolved_count >= MAX_EVOLVED_FIELDS:
            return None

        # 选择随机类型（标量为主，ARRAY 和 JSON 概率较低）
        type_choices = list(ALL_INT_TYPES) + list(ALL_FLOAT_TYPES) + [DataType.BOOL, DataType.VARCHAR]
        ftype = random.choice(type_choices)

        # 15% 概率改为 ARRAY，10% 概率改为 JSON（简单结构）
        elem_type = None
        roll = random.random()
        if roll < 0.15:
            ftype = DataType.ARRAY
            elem_type = random.choice(ALL_ARRAY_ELEMENT_TYPES)
        elif roll < 0.25:
            ftype = DataType.JSON

        type_name = get_type_name(ftype).lower()
        field_name = f"evo_{evolved_count}_{type_name}"

        field_config = {"name": field_name, "type": ftype}

        if ftype == DataType.ARRAY:
            if elem_type is None:
                elem_type = random.choice(ALL_ARRAY_ELEMENT_TYPES)
            field_config["element_type"] = elem_type
            field_config["max_capacity"] = self.array_capacity

        # 添加到 schema_config（OracleQueryGenerator 通过引用自动可见）
        self.schema_config.append(field_config)

        # 更新 pandas DataFrame: 所有现有行设为 None
        self.df[field_name] = None

        return field_config

    def backfill_evolved_field(self, field_config, fill_ratio=None):
        """
        为现有数据的新演进字段生成回填值，并同步更新 pandas DataFrame。

        回填策略：随机选取部分现有行，为新字段填充对应类型的随机值。
        未被选中的行保持 None（与 Milvus 中 nullable 字段的原始状态一致）。

        Args:
            field_config: 字段配置 dict
            fill_ratio: 填充比例 (0.0 - 1.0)，None 则随机选取

        Returns:
            list of (pandas_index, row_id, value) tuples — 需要回填的具体数据
        """
        if fill_ratio is None:
            fill_ratio = random.uniform(0.1, 0.4)

        n_rows = len(self.df)
        if n_rows == 0:
            return []

        field_name = field_config["name"]

        # 确定哪些行被填充
        fill_mask = np.random.random(n_rows) < fill_ratio
        fill_indices = np.where(fill_mask)[0]

        backfill_data = []
        for idx in fill_indices:
            value = self.generate_field_value(field_config)
            row_id = int(self.df.at[idx, "id"])
            backfill_data.append((int(idx), row_id, value))
            # 同步更新 pandas DataFrame
            self.df.at[idx, field_name] = value

        return backfill_data

# --- 2. Milvus Manager (Stable Insert Mode) ---

class MilvusManager:
    def __init__(self):
        self.col = None
        self.scalar_indexes = {}  # 记录当前标量索引: {field_name: index_type}

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
                arr_kwargs = {
                    "name": fc["name"],
                    "dtype": DataType.ARRAY,
                    "element_type": fc["element_type"],
                    "max_capacity": fc["max_capacity"],
                    "nullable": True
                }
                if fc["element_type"] == DataType.VARCHAR:
                    arr_kwargs["max_length"] = 256
                fields.append(FieldSchema(**arr_kwargs))
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

        elif INDEX_TYPE == "DISKANN":
            print("🔨 Building DISKANN Index (Disk-based ANN)...")
            index_params = {
                "metric_type": METRIC_TYPE,
                "index_type": "DISKANN",
                "params": {}
            }

        try:
            # 创建向量索引
            self.col.create_index("vector", index_params, index_name=VECTOR_INDEX_NAME)

            # 创建标量字段索引
            self.create_scalar_indexes(dm.schema_config)

            # 加载数据 (必须在建索引后)
            print("📥 Loading collection into memory...")
            self.col.load()

        except Exception as e:
            print(f"❌ Index build failed (Likely OOM or Config Error): {e}")
            exit(1)

    def create_scalar_indexes(self, schema_config):
        """
        为标量字段随机创建索引。
        根据 SCALAR_INDEX_PROBABILITY 概率决定每个字段是否建索引，
        从 SCALAR_INDEX_TYPES 映射表中随机选择索引类型。
        """
        print("🏗️  Creating Scalar Field Indexes...")
        self.scalar_indexes = {}

        for field in schema_config:
            fname = field["name"]
            ftype = field["type"]

            # 检查该类型是否支持标量索引
            if ftype not in SCALAR_INDEX_TYPES:
                continue

            # 按概率决定是否为该字段创建索引
            if random.random() > SCALAR_INDEX_PROBABILITY:
                continue

            # 从适用的索引类型中随机选一个
            applicable_types = list(SCALAR_INDEX_TYPES[ftype])

            # 【修复】ARRAY 字段的 BITMAP 索引仅支持 bool/int/string 元素类型
            if ftype == DataType.ARRAY:
                elem_type = field.get("element_type", DataType.INT64)
                if elem_type in ALL_FLOAT_TYPES:
                    applicable_types = [t for t in applicable_types if t != "BITMAP"]
                if not applicable_types:
                    applicable_types = ["INVERTED"]

            chosen_type = random.choice(applicable_types)

            try:
                index_params = {"index_type": chosen_type}
                # 使用显式 index_name 避免 drop_index 时的 AmbiguousIndexName
                idx_name = f"idx_{fname}"
                self.col.create_index(field_name=fname, index_params=index_params, index_name=idx_name)
                self.scalar_indexes[fname] = chosen_type
                print(f"   ✅ {fname} ({get_type_name(ftype)}) -> {chosen_type}")
            except Exception as e:
                print(f"   ⚠️ {fname} ({get_type_name(ftype)}) -> {chosen_type} FAILED: {e}")

        if self.scalar_indexes:
            print(f"   -> Created {len(self.scalar_indexes)} scalar indexes: {self.scalar_indexes}")
        else:
            print(f"   -> No scalar indexes created (probabilistic skip)")

    def add_evolved_field(self, field_config):
        """
        通过 MilvusClient 向集合添加新字段（Schema Evolution）。
        添加完成后断开/重连并刷新 Collection 对象以获取新 Schema。
        """
        client = MilvusClient(uri=f"http://{HOST}:{PORT}")

        field_name = field_config["name"]
        ftype = field_config["type"]

        kwargs = {
            "collection_name": COLLECTION_NAME,
            "field_name": field_name,
            "data_type": ftype,
            "nullable": True,
        }

        if ftype == DataType.VARCHAR:
            kwargs["max_length"] = 512
        elif ftype == DataType.JSON:
            pass  # JSON 不需要额外参数
        elif ftype == DataType.ARRAY:
            kwargs["element_type"] = field_config["element_type"]
            kwargs["max_capacity"] = field_config["max_capacity"]
            if field_config["element_type"] == DataType.VARCHAR:
                kwargs["max_length"] = 256

        client.add_collection_field(**kwargs)

        # 强制断开/重连以刷新 pymilvus ORM 的 Schema 缓存
        self.col.release()
        try:
            connections.disconnect("default")
        except Exception:
            pass
        connections.connect("default", host=HOST, port=PORT)

        self.col = Collection(COLLECTION_NAME)
        self.col.load()

        # 验证新 schema 已被正确加载
        field_names = [f.name for f in self.col.schema.fields]
        if field_name not in field_names:
            print(f"⚠️ Schema refresh issue: {field_name} not found in {field_names}")

    def backfill_field_data(self, dm, field_config, backfill_data):
        """
        通过 MilvusClient.upsert(partial_update=True) 回填部分现有行的新字段值。
        只需传递 主键ID + 新字段值，无需向量和其他字段。
        pandas DataFrame 已由 dm.backfill_evolved_field() 提前同步更新。

        Args:
            dm: DataManager
            field_config: 字段配置
            backfill_data: list of (pandas_idx, row_id, value) tuples
        """
        if not backfill_data:
            return 0

        field_name = field_config["name"]
        client = MilvusClient(uri=f"http://{HOST}:{PORT}")
        success_count = 0

        # 逐批 partial upsert
        for start in range(0, len(backfill_data), BATCH_SIZE):
            batch = backfill_data[start:start + BATCH_SIZE]
            rows_to_upsert = []

            for pandas_idx, row_id, value in batch:
                # partial_update=True: 只需要主键 + 要更新的字段
                row = {"id": int(row_id), field_name: value}
                rows_to_upsert.append(row)

            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    data=rows_to_upsert,
                    partial_update=True
                )
                success_count += len(rows_to_upsert)
            except Exception as e:
                print(f"   ⚠️ Backfill partial upsert failed: {e}")

        if success_count > 0:
            self.col.flush()
        return success_count

    def rebuild_scalar_indexes(self, schema_config):
        """
        随机重建标量字段索引（必须在 collection release 之后调用）。
        策略：
        1. 移除所有旧的标量索引
        2. 重新按随机概率和类型创建索引
        3. 尽量选择与旧索引不同的类型，最大化状态空间覆盖
        """
        old_indexes = dict(self.scalar_indexes)
        self.scalar_indexes = {}

        # 1. 根据 Python 字典记录删除旧索引（避免访问 col.indexes 触发 AmbiguousIndexName）
        for field_name in old_indexes:
            try:
                # 使用显式 index_name 删除索引
                idx_name = f"idx_{field_name}"
                self.col.drop_index(index_name=idx_name)
            except Exception:
                pass  # 索引可能已不存在

        # 2. 重新随机创建
        for field in schema_config:
            fname = field["name"]
            ftype = field["type"]

            if ftype not in SCALAR_INDEX_TYPES:
                continue

            # 重建时使用随机概率（可能与初始不同，增加变异性）
            rebuild_prob = random.uniform(0.3, 0.9)
            if random.random() > rebuild_prob:
                continue

            applicable_types = list(SCALAR_INDEX_TYPES[ftype])

            # 【修复】ARRAY 字段的 BITMAP 索引仅支持 bool/int/string 元素类型
            if ftype == DataType.ARRAY:
                elem_type = field.get("element_type", DataType.INT64)
                if elem_type in ALL_FLOAT_TYPES:
                    applicable_types = [t for t in applicable_types if t != "BITMAP"]
                if not applicable_types:
                    applicable_types = ["INVERTED"]

            # 尝试选择一个不同于之前的索引类型（最大化覆盖）
            old_type = old_indexes.get(fname)
            candidates = [t for t in applicable_types if t != old_type]
            if not candidates:
                candidates = applicable_types
            chosen_type = random.choice(candidates)

            try:
                index_params = {"index_type": chosen_type}
                idx_name = f"idx_{fname}"
                self.col.create_index(field_name=fname, index_params=index_params, index_name=idx_name)
                self.scalar_indexes[fname] = chosen_type
            except Exception:
                pass  # 某些组合可能不兼容当前 Milvus 版本

        return old_indexes, dict(self.scalar_indexes)

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

    @staticmethod
    def _float_safe_range(result, is_float32):
        """
        生成经过 Milvus FLOAT 字面量隐式降精度后仍然正确的范围边界。

        [BUG WORKAROUND] Milvus 会将表达式中的浮点字面量隐式转换为字段存储类型
        (FLOAT → float32) 再与运算结果比较。若容差 < float32 ULP，上下界会坍缩
        到同一个 float32 值，导致范围查询返回空结果。

        解决方案: 对 FLOAT 字段使用 float32 nextafter 边界 (跳过 2 个 ULP)。
        TODO: Milvus 修复字面量精度处理后移除此 workaround。
        """
        if is_float32:
            f32 = np.float32(result)
            lo = f32
            hi = f32
            for _ in range(2):
                lo = np.nextafter(lo, np.float32(-np.inf))
                hi = np.nextafter(hi, np.float32(np.inf))
            return float(lo), float(hi)
        else:
            epsilon = max(0.001, abs(result) * 1e-12)
            return result - epsilon, result + epsilon

    def _get_field_type(self, fname):
        """根据字段名查找字段类型（用于 PQS 模式中不直接传递 ftype 的场景）"""
        for f in self.schema:
            if f["name"] == fname:
                return f["type"]
        return None

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
        # 【关键】使用 milvus_notnull_mask：Milvus 将 {} 视为 null
        # [WORKAROUND: Milvus NOT(UNKNOWN) Bug]
        # Milvus 对 NULL JSON 字段执行 NOT(UNKNOWN) 时可能错误返回 TRUE
        # 原始代码: 不使用 notnull_mask, 直接 return (expr, mask)
        # 现在: 所有 JSON 表达式包裹 (name is not null and (expr)), mask & notnull_mask
        # 【关键】使用 milvus_notnull_mask：Milvus 将 {} 视为 null
        notnull_mask = milvus_notnull_mask(series)
        strategy = random.choice(["range", "nested", "index", "multi_key"])

        # --- 策略 1: Range ---
        if strategy == "range":
            low = random.randint(100, 500); high = low + random.randint(50, 200)
            expr = f'({name}["price"] > {low} and {name}["price"] < {high})'

            # 【修复】使用默认参数绑定，避免闭包捕获问题
            def check_range(x, _low=low, _high=high):
                try:
                    if x is None: return None   # 整个JSON字段为NULL → UNKNOWN (3VL)
                    v = self._get_json_val(x, ["price"])
                    if v is None: return False  # 键不存在于非空JSON → FALSE (2VL)
                    if isinstance(v, bool): return False
                    return (isinstance(v, (int, float)) and v > _low and v < _high)
                except: return False

            raw_mask = series.apply(check_range).astype("boolean")
            # [WORKAROUND: NOT(UNKNOWN)] 原始: return (expr, raw_mask)
            return (f"({name} is not null and ({expr}))", raw_mask & notnull_mask)

        # --- 策略 2: Nested ---
        elif strategy == "nested":
            val = random.randint(1, 9)
            expr = f'{name}["config"]["version"] == {val}'

            # 【修复】使用默认参数绑定
            def check_nested(x, _val=val):
                try:
                    if x is None: return None   # 整个JSON字段为NULL → UNKNOWN (3VL)
                    v = self._get_json_val(x, ["config", "version"])
                    if v is None: return False  # 键不存在于非空JSON → FALSE (2VL)
                    if isinstance(v, bool): return False
                    return v == _val
                except: return False

            raw_mask = series.apply(check_nested).astype("boolean")
            # [WORKAROUND: NOT(UNKNOWN)] 原始: return (expr, raw_mask)
            return (f"({name} is not null and ({expr}))", raw_mask & notnull_mask)

        # --- 策略 3: Index ---
        elif strategy == "index":
            idx = 0; val = random.randint(20, 80)
            expr = f'{name}["history"][{idx}] > {val}'

            # 【修复】使用默认参数绑定
            def check_index(x, _idx=idx, _val=val):
                try:
                    if x is None: return None   # 整个JSON字段为NULL → UNKNOWN (3VL)
                    v = self._get_json_val(x, ["history", _idx])
                    if v is None: return False  # 键不存在于非空JSON → FALSE (2VL)
                    if isinstance(v, bool): return False
                    return (isinstance(v, (int, float)) and v > _val)
                except: return False

            raw_mask = series.apply(check_index).astype("boolean")
            # [WORKAROUND: NOT(UNKNOWN)] 原始: return (expr, raw_mask)
            return (f"({name} is not null and ({expr}))", raw_mask & notnull_mask)

        # --- 策略 4: Multi-key ---
        elif strategy == "multi_key":
            color = random.choice(["Red", "Blue"])
            expr = f'({name}["active"] == true and {name}["color"] == "{color}")'

            # 【修复】使用默认参数绑定
            def check_multi(x, _color=color):
                try:
                    if x is None: return None   # 整个JSON字段为NULL → UNKNOWN (3VL)
                    active = self._get_json_val(x, ["active"])
                    col_val = self._get_json_val(x, ["color"])
                    if active is None or col_val is None:
                        return False  # 键不存在于非空JSON → FALSE (2VL)
                    if active is not True: return False # Active must be True
                    return col_val == _color
                except: return False

            raw_mask = series.apply(check_multi).astype("boolean")
            # [WORKAROUND: NOT(UNKNOWN)] 原始: return (expr, raw_mask)
            return (f"({name} is not null and ({expr}))", raw_mask & notnull_mask)

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

        if ftype in ALL_INT_TYPES:
            # 采样一个现有值的范围，然后生成一个超出范围的值
            lo, hi = INT_RANGES[ftype]
            if not valid_series.empty:
                min_val, max_val = int(valid_series.min()), int(valid_series.max())
                # 生成一个比最大值大很多，或者比最小值小很多的值，但 clamp 到类型范围
                candidates = []
                upper = min(max_val + 100000, hi)
                lower = max(min_val - 100000, lo)
                if upper > max_val: candidates.append(upper)
                if lower < min_val: candidates.append(lower)
                if candidates:
                    return random.choice(candidates)
                return int(np.clip(random.randint(lo, hi), lo, hi))
            return int(np.clip(random.randint(-200000, 200000), lo, hi))

        elif ftype in ALL_FLOAT_TYPES:
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

        # 1. Null Check
        # 【关键】使用 milvus_null_mask / milvus_notnull_mask 代替 pandas 原生方法
        # 因为 Milvus 将 [] 和 {} 视为 null，但 pandas 不会
        if random.random() < 0.15:
            if random.random() < 0.5:
                return (f"{name} is null", milvus_null_mask(series))
            else:
                return (f"{name} is not null", milvus_notnull_mask(series))

        # 获取查询值
        val = self.get_value_for_query(name, ftype)
        if val is None:
            # 兜底表达式，永远为真
            id_mask = self.df["id"] > 0
            return ("id > 0", id_mask.astype("boolean"))

        mask = None
        expr = ""

        # --- 🛡️ 标量安全比较 (SQL 3VL) ---
        def safe_compare_scalar(op, target_val):
            # Normalize target container for membership tests
            if op in {"in", "not in"}:
                try:
                    target_set = set(
                        (x.item() if hasattr(x, "item") else x)
                        for x in (target_val or [])
                    )
                except Exception:
                    target_set = set()
            else:
                target_set = None
            def comp(x):
                # 3VL: NULL 比较返回 Unknown
                if x is None: return None
                # Pandas 的 Float Series 可能包含 NaN
                if isinstance(x, float) and np.isnan(x): return None
                try:
                    if op == "in":
                        return x in target_set
                    if op == "not in":
                        return x not in target_set
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

        def _normalize_scalar(v):
            return v.item() if hasattr(v, "item") else v

        def _sample_in_list(max_items=5):
            valid_series = series.dropna()
            if valid_series.empty:
                return []
            values = [_normalize_scalar(v) for v in valid_series.values]
            # 去重但尽量保持顺序稳定
            seen = set()
            uniq = []
            for v in values:
                if v in seen:
                    continue
                seen.add(v)
                uniq.append(v)
            k = random.randint(1, min(max_items, len(uniq)))
            return random.sample(uniq, k)

        # 2. Value Comparison (分流处理)
        if ftype == DataType.BOOL:
            # 加入 in / not in 覆盖（等价于多值枚举）
            if random.random() < 0.25:
                candidates = [True, False]
                chosen = [random.choice(candidates)]
                expr_in = f"{name} in [{str(chosen[0]).lower()}]"
                if random.random() < 0.5:
                    expr = expr_in
                    mask = series.apply(safe_compare_scalar("in", chosen)).astype("boolean")
                else:
                    expr = f"not ({expr_in})"
                    mask = ~series.apply(safe_compare_scalar("in", chosen)).astype("boolean")
            else:
                val_bool = bool(val)
                expr = f"{name} == {str(val_bool).lower()}"
                mask = series.apply(safe_compare_scalar("==", val_bool)).astype("boolean")

        elif ftype in ALL_INT_TYPES:
            # 【新增】15% 概率生成算术表达式
            if random.random() < 0.15:
                arith_res = self._gen_arithmetic_oracle_expr(name, ftype, series)
                if arith_res:
                    return arith_res
            # 加入 in / not in 覆盖
            if random.random() < 0.25:
                items = _sample_in_list(max_items=5)
                # 兜底：确保至少有一个值
                if not items:
                    items = [int(val)]
                items = [int(_normalize_scalar(x)) for x in items]
                in_list_str = ", ".join(str(x) for x in items)
                expr_in = f"{name} in [{in_list_str}]"
                if random.random() < 0.5:
                    expr = expr_in
                    mask = series.apply(safe_compare_scalar("in", items)).astype("boolean")
                else:
                    # 用 not (x in [...]) 表达 not in，兼容性更好
                    expr = f"not ({expr_in})"
                    mask = ~series.apply(safe_compare_scalar("in", items)).astype("boolean")
            else:
                val_int = int(val)
                op = random.choice([">", "<", "==", "!=", ">=", "<="])
                expr = f"{name} {op} {val_int}"
                mask = series.apply(safe_compare_scalar(op, val_int)).astype("boolean")

        elif ftype in ALL_FLOAT_TYPES:
            # 【新增】15% 概率生成算术表达式
            if random.random() < 0.15:
                arith_res = self._gen_arithmetic_oracle_expr(name, ftype, series)
                if arith_res:
                    return arith_res
            val_float = float(val)
            # 🚨 关键回退：移除 "==", "!=" 以避免 IEEE 754 精度误报
            op = random.choice([">", "<", ">=", "<="])
            expr = f"{name} {op} {val_float}"
            mask = series.apply(safe_compare_scalar(op, val_float)).astype("boolean")

        elif ftype == DataType.VARCHAR:
            op = random.choice(["==", "!=", ">", "<", "like", "in", "not in"])
            if op == "like":
                raw_p = val[0] if val else 'a'
                p = 'a' if raw_p in ['%', '_'] else raw_p
                expr = f'{name} like "{p}%"'
                base_mask = series.astype(str).str.startswith(p)
                mask = base_mask.astype("boolean").where(series.notnull(), pd.NA)
            elif op in {"in", "not in"}:
                items = _sample_in_list(max_items=5)
                if not items:
                    items = [str(val)]

                def esc(s: str) -> str:
                    return s.replace("\\", "\\\\").replace('"', '\\"')

                raw_items = [str(_normalize_scalar(x)) for x in items]
                escaped_items = [esc(s) for s in raw_items]
                in_list_str = ", ".join(f'"{s}"' for s in escaped_items)
                expr_in = f'{name} in [{in_list_str}]'
                if op == "in":
                    expr = expr_in
                    mask = series.apply(safe_compare_scalar("in", raw_items)).astype("boolean")
                else:
                    expr = f"not ({expr_in})"
                    mask = ~series.apply(safe_compare_scalar("in", raw_items)).astype("boolean")
            else:
                expr = f'{name} {op} "{val}"'
                mask = series.apply(safe_compare_scalar(op, val)).astype("boolean")

        elif ftype == DataType.JSON:

            # --- 策略 A: Exists 查询 ---
            if random.random() < 0.2:
                tk = "price" if random.random() < 0.5 else "non_exist"
                expr = f'exists({name}["{tk}"])'
                # 【修复】使用默认参数绑定
                def check_exists(x, _tk=tk):
                    if x is None: return False
                    if not isinstance(x, dict): return False
                    # 探测结果 V1: exists 排除了显式 Null 值
                    return _tk in x and x[_tk] is not None
                mask = series.apply(check_exists).astype("boolean")

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
                        # 【修复】3VL: NULL / 缺失键 / null值 → UNKNOWN (None → <NA>)
                        def check_list_contains(x, _k=k, _target_item=target_item):
                            if x is None: return None           # NULL JSON → UNKNOWN
                            if not isinstance(x, dict): return None
                            if _k not in x: return None         # 缺失键 → UNKNOWN
                            val = x[_k]
                            if val is None: return None          # 显式 null → UNKNOWN
                            if not isinstance(val, list): return False  # 非数组 → FALSE
                            return _target_item in val
                        mask = series.apply(check_list_contains).astype("boolean")
                    else: return (f"{name} is not null", milvus_notnull_mask(series))
                else: return (f"{name} is not null", milvus_notnull_mask(series))

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

                    # [WORKAROUND: Milvus JSON != Bug]
                    # Milvus 已确认 Bug: JSON 键缺失/值为 null 时 != 返回 TRUE (应为 FALSE/UNKNOWN)
                    # 原始代码:
                    #   if isinstance(query_val, bool):
                    #       op = random.choice(["==", "!="])
                    #   else:
                    #       op = random.choice(["==", "!=", ">", "<", ">=", "<="])
                    if isinstance(query_val, bool):
                        op = "=="
                    else:
                        op = random.choice(["==", ">", "<", ">=", "<="])

                    if isinstance(query_val, bool): val_str = str(query_val).lower()
                    elif isinstance(query_val, str): val_str = f'"{query_val}"'
                    else: val_str = str(query_val)

                    expr = f'{name}{path_str} {op} {val_str}'

                    # --- 🛡️ JSON 安全比较 ---
                    # 【关键修复】使用默认参数绑定，避免闭包捕获问题
                    def safe_check_json(x, _path_keys=path_keys, _op=op, _query_val=query_val):
                        try:
                            if x is None: return None  # 整个JSON字段为NULL → UNKNOWN (3VL)
                            v = self._get_json_val(x, _path_keys)

                            # (A) 基础判空 — 键不存在于非空JSON → Milvus返回FALSE(2VL)
                            if v is None:
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
                                return False  # 类型不匹配 → Milvus返回FALSE

                            if _op == "==": return v == _query_val
                            if _op == "!=": return v != _query_val
                            if _op == ">": return v > _query_val
                            if _op == "<": return v < _query_val
                            if _op == ">=": return v >= _query_val
                            if _op == "<=": return v <= _query_val
                            return False
                        except:
                            return False  # JSON访问异常 → Milvus返回FALSE

                    mask = series.apply(safe_check_json).astype("boolean")
                else:
                    return (f"{name} is not null", milvus_notnull_mask(series))

        elif ftype == DataType.ARRAY:
            # 【新增】ARRAY 表达式处理（从官方测试脚本借鉴）
            elem_type = None
            for fc in self.schema:
                if fc["name"] == name:
                    elem_type = fc.get("element_type", DataType.INT64)
                    break
            if elem_type is None:
                elem_type = DataType.INT64

            arr_result = self._gen_array_oracle_expr(name, series, elem_type)
            if arr_result and arr_result[0]:
                return arr_result
            return (f"{name} is not null", milvus_notnull_mask(series))

        if mask is not None:
            # 对原子比较统一加非空护栏，规避 Milvus 的 NULL 比较歧义
            if ftype in [DataType.BOOL] + ALL_INT_TYPES + ALL_FLOAT_TYPES + [DataType.VARCHAR]:
                notnull_mask = milvus_notnull_mask(series)
                guarded_mask = mask.astype("boolean") & notnull_mask
                guarded_expr = f"({name} is not null and ({expr}))"
                return (guarded_expr, guarded_mask)

            if ftype == DataType.JSON:
                if expr and not expr.startswith("exists(") and not expr.startswith("json_contains("):
                    notnull_mask = milvus_notnull_mask(series)
                    guarded_mask = mask.astype("boolean") & notnull_mask
                    guarded_expr = f"({name} is not null and ({expr}))"
                    return (guarded_expr, guarded_mask)
                return (expr, mask)

            return (expr, mask)

        return ("", None)

    def _gen_array_oracle_expr(self, name, series, element_type):
        """
        【新增】为 ARRAY 字段生成带 Oracle mask 的表达式。
        覆盖: array_contains, array_length, 数组下标访问, array_contains_all/any
        灵感来源: 官方测试脚本中的数组操作覆盖模式。
        """
        strategy = random.choice([
            "contains", "length", "subscript", "contains_all", "contains_any"
        ])
        # 【关键】使用 milvus_notnull_mask：Milvus 将 [] 视为 null
        notnull_mask = milvus_notnull_mask(series)

        def _sample_element(s, etype):
            """从现有数据中采样一个数组元素值"""
            for val in s.dropna().values:
                if isinstance(val, list) and len(val) > 0:
                    items = [x for x in val if x is not None]
                    if items:
                        return random.choice(items)
            # 兜底: 生成一个默认值
            if etype in ALL_INT_TYPES:
                return random.randint(-50, 50)
            elif etype in ALL_FLOAT_TYPES:
                return round(random.uniform(-50, 50), 4)
            elif etype == DataType.BOOL:
                return random.choice([True, False])
            elif etype == DataType.VARCHAR:
                return ''.join(random.choices(string.ascii_lowercase, k=4))
            return 0

        def _format_val(v, etype):
            """格式化值为 Milvus 表达式字面量"""
            if isinstance(v, bool):
                return str(v).lower()
            if isinstance(v, str):
                safe = v.replace('\\', '\\\\').replace('"', '\\"')
                return f'"{safe}"'
            if isinstance(v, float):
                return str(v)
            return str(v)

        # --- 策略 1: array_contains ---
        if strategy == "contains":
            target = _sample_element(series, element_type)
            val_str = _format_val(target, element_type)
            expr = f"array_contains({name}, {val_str})"

            def check_contains(x, _target=target):
                if x is None: return None
                if not isinstance(x, list): return False
                return _target in x

            mask = series.apply(check_contains).astype("boolean")
            return (f"({name} is not null and ({expr}))", mask & notnull_mask)

        # --- 策略 2: array_length ---
        elif strategy == "length":
            length_val = random.randint(0, 5)
            op = random.choice(["==", ">", "<", ">=", "<="])
            expr = f"array_length({name}) {op} {length_val}"

            def check_length(x, _op=op, _val=length_val):
                if x is None: return None
                if not isinstance(x, list): return False
                l = len(x)
                if _op == "==": return l == _val
                if _op == ">": return l > _val
                if _op == "<": return l < _val
                if _op == ">=": return l >= _val
                if _op == "<=": return l <= _val
                return False

            mask = series.apply(check_length).astype("boolean")
            return (f"({name} is not null and ({expr}))", mask & notnull_mask)

        # --- 策略 3: 数组下标访问 (arr[idx] op val) ---
        elif strategy == "subscript":
            idx = random.randint(0, 2)
            target = _sample_element(series, element_type)
            val_str = _format_val(target, element_type)

            if element_type in ALL_INT_TYPES + ALL_FLOAT_TYPES:
                op = random.choice([">", "<", "==", ">=", "<="])
                expr = f"{name}[{idx}] {op} {val_str}"

                def check_subscript_num(x, _idx=idx, _op=op, _target=target):
                    if x is None: return None
                    if not isinstance(x, list) or len(x) <= _idx: return False
                    v = x[_idx]
                    if v is None: return False
                    try:
                        is_v_num = isinstance(v, (int, float)) and not isinstance(v, bool)
                        is_t_num = isinstance(_target, (int, float)) and not isinstance(_target, bool)
                        if not (is_v_num and is_t_num): return False
                        if _op == "==": return v == _target
                        if _op == ">": return v > _target
                        if _op == "<": return v < _target
                        if _op == ">=": return v >= _target
                        if _op == "<=": return v <= _target
                    except: return False
                    return False

                mask = series.apply(check_subscript_num).astype("boolean")
                return (f"({name} is not null and ({expr}))", mask & notnull_mask)

            elif element_type == DataType.VARCHAR:
                sub_op = random.choice(["eq", "like"])
                if sub_op == "eq":
                    expr = f'{name}[{idx}] == {val_str}'

                    def check_str_sub(x, _idx=idx, _target=str(target)):
                        if x is None: return None
                        if not isinstance(x, list) or len(x) <= _idx: return False
                        return x[_idx] == _target

                    mask = series.apply(check_str_sub).astype("boolean")
                    return (f"({name} is not null and ({expr}))", mask & notnull_mask)
                else:
                    prefix = str(target)[:2] if len(str(target)) >= 2 else str(target)[:1]
                    if any(c in prefix for c in ['%', '_', '"', '\\\\']):
                        prefix = 'a'
                    expr = f'{name}[{idx}] like "{prefix}%"'

                    def check_like_sub(x, _idx=idx, _prefix=prefix):
                        if x is None: return None
                        if not isinstance(x, list) or len(x) <= _idx: return False
                        v = x[_idx]
                        if not isinstance(v, str): return False
                        return v.startswith(_prefix)

                    mask = series.apply(check_like_sub).astype("boolean")
                    return (f"({name} is not null and ({expr}))", mask & notnull_mask)

            elif element_type == DataType.BOOL:
                val_bool = random.choice([True, False])
                expr = f"{name}[{idx}] == {str(val_bool).lower()}"

                def check_bool_sub(x, _idx=idx, _val=val_bool):
                    if x is None: return None
                    if not isinstance(x, list) or len(x) <= _idx: return False
                    return x[_idx] == _val

                mask = series.apply(check_bool_sub).astype("boolean")
                return (f"({name} is not null and ({expr}))", mask & notnull_mask)

            return (f"{name} is not null", notnull_mask)

        # --- 策略 4: array_contains_all ---
        elif strategy == "contains_all":
            # 从同一行采样多个元素作为子集
            targets = []
            for val in series.dropna().values:
                if isinstance(val, list) and len(val) >= 2:
                    items = [x for x in val if x is not None]
                    if len(items) >= 2:
                        k = random.randint(2, min(3, len(items)))
                        targets = random.sample(items, k)
                        break
            if not targets:
                target = _sample_element(series, element_type)
                targets = [target]

            targets_str = json.dumps(targets)
            expr = f"array_contains_all({name}, {targets_str})"

            def check_contains_all(x, _targets=targets):
                if x is None: return None
                if not isinstance(x, list): return False
                return all(t in x for t in _targets)

            mask = series.apply(check_contains_all).astype("boolean")
            return (f"({name} is not null and ({expr}))", mask & notnull_mask)

        # --- 策略 5: array_contains_any ---
        elif strategy == "contains_any":
            target = _sample_element(series, element_type)
            # 混入一个不存在的值 (需要避免溢出窄整型)
            if element_type == DataType.VARCHAR:
                noise = "zzzz_nonexist"
            elif element_type == DataType.BOOL:
                noise = not target
            elif element_type in (DataType.INT8, DataType.INT16):
                lo, hi = INT_RANGES[element_type]
                # 取一个在范围内但 ≠ target 的极端值
                noise = lo if target != lo else hi
            else:
                noise = 999999
            targets = [target, noise]
            targets_str = json.dumps(targets)
            expr = f"array_contains_any({name}, {targets_str})"

            def check_contains_any(x, _targets=targets):
                if x is None: return None
                if not isinstance(x, list): return False
                return any(t in x for t in _targets)

            mask = series.apply(check_contains_any).astype("boolean")
            return (f"({name} is not null and ({expr}))", mask & notnull_mask)

        return (f"{name} is not null", notnull_mask)

    def _gen_arithmetic_oracle_expr(self, name, ftype, series):
        """
        【新增】为数值类型生成算术表达式（Oracle 模式）。
        覆盖: field + const, field - const, field * const, field % const
        这些操作在 Milvus 内部走不同的执行路径，是 bug 高发区。
        """
        if ftype in ALL_INT_TYPES:
            ops = ["+", "-", "*", "%"]
        elif ftype in ALL_FLOAT_TYPES:
            ops = ["+", "-", "*"]
        else:
            return None

        op = random.choice(ops)
        # 使用小操作数避免溢出
        if op == "%":
            operand = random.randint(2, 20)
        elif op == "*":
            operand = random.randint(1, 5)
        else:
            operand = random.randint(1, 50)

        # 从实际数据采样一个值来计算阈值
        valid_series = series.dropna()
        if valid_series.empty:
            return None

        sample_val = valid_series.iloc[random.randint(0, len(valid_series) - 1)]
        if hasattr(sample_val, 'item'):
            sample_val = sample_val.item()

        try:
            if op == "+": expected = sample_val + operand
            elif op == "-": expected = sample_val - operand
            elif op == "*": expected = sample_val * operand
            elif op == "%":
                if operand == 0: return None
                # 【关键】使用 C/Go 风格的截断除法取模（而非 Python 的欧几里得取模）
                # Python: -47 % 20 = 13 (always non-negative)
                # Go/C++: -47 % 20 = -7 (sign follows dividend)
                expected = int(np.fmod(sample_val, operand))
        except:
            return None

        if ftype in ALL_FLOAT_TYPES:
            cmp_op = random.choice([">", "<", ">=", "<="])
            if ftype == DataType.FLOAT:
                # [WORKAROUND] FLOAT 字段: 确保 delta 远大于 float32 ULP，
                # 并将 threshold 对齐到 float32 以匹配 Milvus 隐式降精度行为
                f32_expected = np.float32(expected)
                ulp = float(np.nextafter(np.abs(f32_expected), np.float32(np.inf))) - float(np.abs(f32_expected))
                min_delta = max(ulp * 4, 0.01)
                delta = random.uniform(min_delta, min_delta + 1.0)
                if cmp_op in [">", ">="]:
                    threshold = float(np.float32(expected - delta))
                else:
                    threshold = float(np.float32(expected + delta))
            else:
                delta = random.uniform(0.01, 1.0)
                if cmp_op in [">", ">="]:
                    threshold = expected - delta
                else:
                    threshold = expected + delta
        else:
            cmp_op = random.choice([">", "<", "==", ">=", "<="])
            if cmp_op == "==":
                threshold = expected
            elif cmp_op in [">", ">="]:
                threshold = expected - random.randint(1, 10)
            else:
                threshold = expected + random.randint(1, 10)

        expr = f"{name} {op} {operand} {cmp_op} {threshold}"

        def check_arith(x, _op=op, _operand=operand, _cmp_op=cmp_op, _threshold=threshold):
            if x is None: return None
            if isinstance(x, float) and np.isnan(x): return None
            try:
                if _op == "+": result = x + _operand
                elif _op == "-": result = x - _operand
                elif _op == "*": result = x * _operand
                elif _op == "%":
                    if _operand == 0: return False
                    # C/Go 风格取模（截断除法）
                    result = int(np.fmod(x, _operand))
                else: return False

                if _cmp_op == "==": return result == _threshold
                if _cmp_op == ">": return result > _threshold
                if _cmp_op == "<": return result < _threshold
                if _cmp_op == ">=": return result >= _threshold
                if _cmp_op == "<=": return result <= _threshold
                return False
            except:
                return False

        mask = series.apply(check_arith).astype("boolean")
        notnull_mask = milvus_notnull_mask(series)
        guarded_mask = mask & notnull_mask
        guarded_expr = f"({name} is not null and ({expr}))"
        return (guarded_expr, guarded_mask)

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
        # 创建全 True/False 的 mask (使用 nullable boolean 兼容 3VL)
        true_mask = pd.Series(True, index=self.df.index, dtype="boolean")
        false_mask = pd.Series(False, index=self.df.index, dtype="boolean")

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
        # 3VL 逻辑核心:
        # Pandas 的 nullable boolean 类型 (dtype="boolean") 支持 <NA> (即 Null)
        # ~<NA> -> <NA>
        # True & <NA> -> <NA>
        # False & <NA> -> False
        # 这完全符合 SQL 三值逻辑。
        # 因此，只要保证 mask 本身是 nullable boolean 类型，直接取反即可。
        return ~mask.astype("boolean")

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
            if ftype in ALL_INT_TYPES + ALL_FLOAT_TYPES:
                val = self.get_value_for_query(name, ftype)
                if val is None:
                    return (f"not ({name} is null)", milvus_notnull_mask(series))
                op = random.choice([">", "<", ">=", "<="])
                inner_expr = f"{name} {op} {val}"
                expr = f"not ({inner_expr})"
                
                # Oracle: 计算内部表达式的 mask
                def safe_cmp(x, _op=op, _val=val):
                    if x is None: return None
                    if isinstance(x, float) and np.isnan(x): return None
                    try:
                        if _op == ">": return x > _val
                        if _op == "<": return x < _val
                        if _op == ">=": return x >= _val
                        if _op == "<=": return x <= _val
                        return False
                    except: return False
                inner_mask = series.apply(safe_cmp).astype("boolean")
                # 3VL NOT: ~inner
                mask = ~inner_mask
                return (expr, mask)
            
            elif ftype == DataType.VARCHAR:
                val = self.get_value_for_query(name, ftype)
                if val is None:
                    return (f"not ({name} is null)", milvus_notnull_mask(series))
                inner_expr = f'{name} == "{val}"'
                expr = f"not ({inner_expr})"
                def safe_eq(x, _val=val):
                    if x is None: return None
                    if isinstance(x, float) and np.isnan(x): return None
                    try: return x == _val
                    except: return False
                inner_mask = series.apply(safe_eq).astype("boolean")
                mask = ~inner_mask
                return (expr, mask)

            elif ftype == DataType.BOOL:
                val_bool = random.choice([True, False])
                inner_expr = f"{name} == {str(val_bool).lower()}"
                expr = f"not ({inner_expr})"
                def safe_eq_bool(x, _val=val_bool):
                    if x is None: return None
                    if isinstance(x, float) and np.isnan(x): return None
                    try: return x == _val
                    except: return False
                inner_mask = series.apply(safe_eq_bool).astype("boolean")
                mask = ~inner_mask
                return (expr, mask)

            elif ftype == DataType.JSON:
                # NOT + JSON 下钻比较
                val = random.randint(100, 500)
                inner_expr = f'{name}["price"] > {val}'
                # [WORKAROUND: Milvus NOT(UNKNOWN) Bug]
                # Milvus 对 NULL JSON 字段 NOT(UNKNOWN) 可能错误返回 TRUE
                # 原始代码: expr = f"not ({inner_expr})"
                inner_guarded = f'({name} is not null and ({inner_expr}))'
                expr = f"not ({inner_guarded})"
                def check_json_gt(x, _val=val):
                    try:
                        if x is None: return None   # 整个JSON字段为NULL → UNKNOWN (3VL)
                        v = self._get_json_val(x, ["price"])
                        if v is None: return False  # 键不存在于非空JSON → FALSE (2VL)
                        if isinstance(v, bool): return False
                        return isinstance(v, (int, float)) and v > _val
                    except: return False  # JSON访问异常 → Milvus返回FALSE
                inner_mask = series.apply(check_json_gt).astype("boolean")
                # [WORKAROUND: NOT(UNKNOWN)] 原始: mask = ~inner_mask; return (expr, mask)
                notnull_mask = milvus_notnull_mask(series)
                inner_mask_guarded = inner_mask & notnull_mask
                mask = ~inner_mask_guarded
                return (expr, mask)

            # 兜底
            return (f"not ({name} is null)", milvus_notnull_mask(series))

        # --- 策略 2: NOT + 等值 ---
        elif strategy == "not_eq":
            if ftype in ALL_INT_TYPES:
                val = self.get_value_for_query(name, ftype)
                if val is None:
                    return (f"not ({name} is null)", milvus_notnull_mask(series))
                val_int = int(val)
                expr = f"not ({name} == {val_int})"
                def safe_eq_int(x, _val=val_int):
                    if x is None: return None
                    if isinstance(x, float) and np.isnan(x): return None
                    try: return x == _val
                    except: return False
                inner_mask = series.apply(safe_eq_int).astype("boolean")
                mask = ~inner_mask
                return (expr, mask)
            # 其他类型 fallthrough
            return (f"not ({name} is null)", milvus_notnull_mask(series))

        # --- 策略 3: NOT (is null) ↔ is not null ---
        elif strategy == "not_is_null":
            expr = f"not ({name} is null)"
            mask = milvus_notnull_mask(series)
            return (expr, mask)

        # --- 策略 4: NOT (is not null) ↔ is null ---
        elif strategy == "not_is_not_null":
            expr = f"not ({name} is not null)"
            mask = milvus_null_mask(series)
            return (expr, mask)

        # --- 策略 5: NOT + AND (德摩根定律测试核心) ---
        elif strategy == "not_and":
            if ftype in ALL_INT_TYPES:
                val = self.get_value_for_query(name, ftype)
                if val is None:
                    return (f"not ({name} is null)", milvus_notnull_mask(series))
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
                    if x is None: return None
                    if isinstance(x, float) and np.isnan(x): return None
                    try: return x > _low and x < _high
                    except: return False
                inner_mask = series.apply(check_range).astype("boolean")
                mask = ~inner_mask
                return (expr, mask)
            return (f"not ({name} is null)", milvus_notnull_mask(series))

        # --- 策略 6: NOT + OR ---
        elif strategy == "not_or":
            if ftype in ALL_INT_TYPES:
                val1 = self.get_value_for_query(name, ftype)
                val2 = self.get_value_for_query(name, ftype)
                if val1 is None or val2 is None:
                    return (f"not ({name} is null)", milvus_notnull_mask(series))
                val1_int, val2_int = int(val1), int(val2)
                inner_expr = f"({name} == {val1_int} or {name} == {val2_int})"
                expr = f"not {inner_expr}"
                def check_or(x, _v1=val1_int, _v2=val2_int):
                    if x is None: return None
                    if isinstance(x, float) and np.isnan(x): return None
                    try: return x == _v1 or x == _v2
                    except: return False
                inner_mask = series.apply(check_or).astype("boolean")
                mask = ~inner_mask
                return (expr, mask)
            return (f"not ({name} is null)", milvus_notnull_mask(series))

        # 终极兜底
        return (f"not ({name} is null)", milvus_notnull_mask(series))


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

def run_equivalence_mode(rounds=100, seed=None, enable_dynamic_ops=True, consistency=None):
    """
    运行等价性模糊测试
    """
    global INDEX_TYPE, CURRENT_INDEX_TYPE, VECTOR_CHECK_RATIO, VECTOR_TOPK, _GLOBAL_ID_COUNTER, METRIC_TYPE, SCALAR_INDEX_PROBABILITY
    
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
    SCALAR_INDEX_PROBABILITY = random.uniform(0.3, 0.8)

    # 一致性等级: 固定或随机 (独立 RNG, 不消耗主 PRNG 序列)
    ALL_CONSISTENCY_LEVELS = ["Strong", "Bounded", "Eventually", "Session"]
    if consistency:
        consistency_rng = None  # 不需要 RNG, 使用固定值
        print(f"   一致性等级: {consistency} (固定)")
    else:
        consistency_rng = random.Random(seed ^ 0xC0A51573)
        print(f"   一致性等级: 随机 {ALL_CONSISTENCY_LEVELS}")

    def pick_consistency():
        if consistency:
            return consistency
        return consistency_rng.choice(ALL_CONSISTENCY_LEVELS)

    print(f"   索引类型: {INDEX_TYPE}, 度量类型: {METRIC_TYPE}")
    print(f"   标量索引概率: {SCALAR_INDEX_PROBABILITY:.2f}")

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
                        mm.col.compact()
                        mm.col.wait_for_compaction_completed()
                        mm.col.release()
                        mm.col.load()
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
                        mm.col.compact()
                        mm.col.wait_for_compaction_completed()
                        mm.col.release()
                        mm.col.load()
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

            # --- 随机触发标量索引重建 ---
            if i > 0 and i % 50 == 0:
                try:
                    mm.col.release()
                    file_log(f"[Maintenance] Released collection for scalar index rebuild at round {i}")
                    old_idx, new_idx = mm.rebuild_scalar_indexes(dm.schema_config)
                    file_log(f"[Maintenance] Scalar index rebuild at round {i}: {old_idx} -> {new_idx}")
                    mm.col.load()
                    file_log(f"[Maintenance] Reloaded after scalar index rebuild at round {i}")
                except Exception as e:
                    file_log(f"[Maintenance] Scalar index rebuild failed at round {i}: {e}")
                    try:
                        mm.col.load()
                    except:
                        pass

            # --- Schema Evolution ---
            if enable_dynamic_ops and i > 0 and i % 30 == 0:
                try:
                    field_config = dm.evolve_schema_add_field()
                    if field_config:
                        ftype_name = get_type_name(field_config["type"])
                        if field_config["type"] == DataType.ARRAY:
                            ftype_name += f"<{get_type_name(field_config['element_type'])}>"
                        file_log(f"[SchemaEvolution] Adding field '{field_config['name']}' ({ftype_name}) at round {i}")
                        mm.add_evolved_field(field_config)
                        file_log(f"[SchemaEvolution] Field '{field_config['name']}' added to Milvus successfully")

                        # 回填: 使用 partial_update=True 为部分现有行填充新字段值
                        backfill_data = dm.backfill_evolved_field(field_config)
                        if backfill_data:
                            fill_count = mm.backfill_field_data(dm, field_config, backfill_data)
                            file_log(f"[SchemaEvolution] Backfilled {fill_count}/{len(backfill_data)} rows for '{field_config['name']}'")
                        else:
                            file_log(f"[SchemaEvolution] No rows selected for backfill of '{field_config['name']}'")
                except Exception as e:
                    file_log(f"[SchemaEvolution] Failed at round {i}: {e}")
                    try:
                        mm.col.load()
                    except:
                        pass

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
                cl_mut = pick_consistency()
                base_res = mm.col.query(base_expr, output_fields=["id"], consistency_level=cl_mut)
                base_ids = set([x["id"] for x in base_res])
            except Exception as e:
                # 基础查询挂了，可能是语法问题，跳过
                file_log(f"[Test {i}] Base Query Failed: {e}")
                continue

            log_header = f"[Test {i}] Base: {base_expr} (Hits: {len(base_ids)}) [CL={cl_mut}]"
            file_log(f"\n{log_header}")

            # 4. 执行并对比所有变体
            for m in mutations:
                m_type = m["type"]
                m_expr = m["expr"]
                
                try:
                    mut_res = mm.col.query(m_expr, output_fields=["id"], consistency_level=cl_mut)
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
                        
                        # ==================== [数据对比取证] ====================
                        # 🔍 深度取证：展示 Pandas 数据 + Milvus 实际数据，便于判断 bug 来源
                        def print_evidence(ids, label, ref_expr):
                            if not ids: return
                            ids_sample = list(ids)[:5]
                            print(f"   🕵️ {label} 数据对比 (共{len(ids)}条, 展示{len(ids_sample)}条):")
                            file_log(f"  {label} 数据对比 (共{len(ids)}条, 展示{len(ids_sample)}条):")

                            # Pandas 数据
                            pandas_subset = dm.df[dm.df["id"].isin(ids_sample)].to_dict(orient="records")
                            # Milvus 数据
                            all_output_fields = ["id"] + [fc["name"] for fc in dm.schema_config]
                            try:
                                milvus_rows = mm.col.query(
                                    f"id in {ids_sample}",
                                    output_fields=all_output_fields,
                                    limit=len(ids_sample),
                                    consistency_level="Strong"
                                )
                            except Exception:
                                milvus_rows = []
                            milvus_by_id = {r["id"]: r for r in milvus_rows}

                            for rid in ids_sample:
                                print(f"      ─── ID: {rid} ───")
                                file_log(f"    ─── ID: {rid} ───")

                                # Pandas 侧
                                p_rows = [r for r in pandas_subset if r.get("id") == rid]
                                if p_rows:
                                    p_data = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, 'item') else v))
                                              for k, v in p_rows[0].items() if k != "vector"}
                                    null_fields = [k for k, v in p_data.items() if v is None and k != "id"]
                                    print(f"        [Pandas Oracle]")
                                    if null_fields:
                                        print(f"          ⚠️ Null 字段: {null_fields}")
                                    for k, v in p_data.items():
                                        if k == "id": continue
                                        if k in ref_expr or v is None or isinstance(v, (dict, list)):
                                            v_str = json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else str(v)
                                            if len(v_str) > 200: v_str = v_str[:200] + "..."
                                            print(f"          {k}: {v_str}")
                                    file_log(f"      [Pandas] {json.dumps(p_data, ensure_ascii=False, default=str)}")
                                else:
                                    print(f"        [Pandas Oracle] ❌ 不存在")
                                    file_log(f"      [Pandas] NOT FOUND")

                                # Milvus 侧
                                if rid in milvus_by_id:
                                    m_data = {k: v for k, v in milvus_by_id[rid].items() if k != "vector"}
                                    null_fields_m = [k for k, v in m_data.items() if v is None and k != "id"]
                                    print(f"        [Milvus 实际]")
                                    if null_fields_m:
                                        print(f"          ⚠️ Null 字段: {null_fields_m}")
                                    for k, v in m_data.items():
                                        if k == "id": continue
                                        if k in ref_expr or v is None or isinstance(v, (dict, list)):
                                            v_str = json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else str(v)
                                            if len(v_str) > 200: v_str = v_str[:200] + "..."
                                            print(f"          {k}: {v_str}")
                                    file_log(f"      [Milvus] {json.dumps(m_data, ensure_ascii=False, default=str)}")

                                    # Null 不一致高亮
                                    if p_rows:
                                        diffs = []
                                        for k in set(list(p_data.keys()) + list(m_data.keys())):
                                            if k in ("id", "vector"): continue
                                            pv, mv = p_data.get(k), m_data.get(k)
                                            if (pv is None) != (mv is None):
                                                diffs.append(f"{k}: Pandas={'null' if pv is None else repr(pv)[:50]}, Milvus={'null' if mv is None else repr(mv)[:50]}")
                                        if diffs:
                                            print(f"        🔴 Null 不一致: {'; '.join(diffs)}")
                                            file_log(f"      NULL_MISMATCH: {'; '.join(diffs)}")
                                else:
                                    print(f"        [Milvus 实际] ❌ 查询未返回")
                                    file_log(f"      [Milvus] NOT FOUND")

                        print_evidence(missing, "MISSING (Base hit, Mut missed)", base_expr)
                        print_evidence(extra, "EXTRA (Base missed, Mut hit)", base_expr)
                        # ==================== [数据对比取证结束] ====================

                        file_log(f"  ❌ [{m_type}] FAIL! {diff_msg}")
                        file_log(f"     Mut Expr: {m_expr}")
                        print(f"   Diff: {diff_msg}")
                        print("-" * 50)

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

def run(rounds = 100, seed=None, enable_dynamic_ops=True, consistency=None):
    """
    seed=None: 随机数据，每次不同（默认行为）
    seed=<数字>: 固定种子，完全复现之前的测试
    """
    global INDEX_TYPE, CURRENT_INDEX_TYPE, VECTOR_CHECK_RATIO, VECTOR_TOPK, _GLOBAL_ID_COUNTER, METRIC_TYPE, SCALAR_INDEX_PROBABILITY
    
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
    SCALAR_INDEX_PROBABILITY = random.uniform(0.3, 0.8)

    # 一致性等级: 固定或随机 (独立 RNG, 不消耗主 PRNG 序列, 不影响表达式/数据生成)
    ALL_CONSISTENCY_LEVELS = ["Strong", "Bounded", "Eventually", "Session"]
    if consistency:
        consistency_rng = None
        print(f"   一致性等级: {consistency} (固定)")
    else:
        consistency_rng = random.Random(current_seed ^ 0xC0A51573)
        print(f"   一致性等级: 随机 {ALL_CONSISTENCY_LEVELS}")

    def pick_consistency():
        if consistency:
            return consistency
        return consistency_rng.choice(ALL_CONSISTENCY_LEVELS)

    print(f"   索引类型: {INDEX_TYPE}, 度量类型: {METRIC_TYPE}, 向量校验比例: {VECTOR_CHECK_RATIO:.2f}, TopK: {VECTOR_TOPK}")
    print(f"   标量索引概率: {SCALAR_INDEX_PROBABILITY:.2f}")

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

        # 辅助：提取指定 ID 的 Pandas 行
        def sample_rows(id_set, limit=5):
            if not id_set:
                return []
            subset = dm.df[dm.df["id"].isin(list(id_set))]
            rows = subset.to_dict(orient="records")
            return rows[:limit]

        # 辅助：从 Milvus 查询完整行数据（用于差异对比）
        def query_milvus_rows(id_list, limit=5):
            """查询 Milvus 中指定 ID 的完整行数据，返回 [{field: value, ...}, ...]"""
            if not id_list:
                return []
            ids = list(id_list)[:limit]
            all_output_fields = ["id"] + [fc["name"] for fc in dm.schema_config]
            try:
                res = mm.col.query(
                    f"id in {ids}",
                    output_fields=all_output_fields,
                    limit=limit,
                    consistency_level="Strong"
                )
                return res
            except Exception as e:
                file_log(f"  [query_milvus_rows] Failed: {e}")
                return []

        # 辅助：格式化行数据（去向量、处理 numpy 类型）
        def format_row_for_display(row_dict):
            """格式化单行数据，去掉向量、处理 numpy/NaN"""
            out = {}
            for k, v in row_dict.items():
                if k == "vector":
                    continue
                if hasattr(v, "item"):
                    v = v.item()
                if isinstance(v, float) and np.isnan(v):
                    v = None
                out[k] = v
            return out

        # 辅助：打印 Pandas vs Milvus 对比数据
        def print_diff_evidence(diff_ids, label, expr_str):
            """对差异 ID 展示 Pandas Oracle 数据 + Milvus 实际数据，便于判断 bug 来源"""
            if not diff_ids:
                return
            ids_sample = list(diff_ids)[:5]
            # 1. Pandas 数据
            pandas_rows = dm.df[dm.df["id"].isin(ids_sample)].to_dict(orient="records")
            # 2. Milvus 数据
            milvus_rows = query_milvus_rows(ids_sample)
            milvus_by_id = {r["id"]: r for r in milvus_rows}

            print(f"\n   📊 {label} 数据对比 (共{len(diff_ids)}条, 展示{len(ids_sample)}条):")
            file_log(f"  {label} 数据对比 (共{len(diff_ids)}条, 展示{len(ids_sample)}条):")

            for rid in ids_sample:
                print(f"   ─── ID: {rid} ───")
                file_log(f"  ─── ID: {rid} ───")

                # Pandas 侧
                p_rows = [r for r in pandas_rows if r.get("id") == rid]
                if p_rows:
                    p_data = format_row_for_display(p_rows[0])
                    # 提取与表达式相关的字段（启发式）
                    null_fields = [k for k, v in p_data.items() if v is None and k != "id"]
                    print(f"     [Pandas Oracle]")
                    if null_fields:
                        print(f"       ⚠️ Null 字段: {null_fields}")
                    for k, v in p_data.items():
                        if k == "id":
                            continue
                        # 只打印在表达式中出现的字段 + null 字段 + JSON/Array 字段
                        if k in expr_str or v is None or isinstance(v, (dict, list)):
                            v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                            if len(v_str) > 200:
                                v_str = v_str[:200] + "..."
                            print(f"       {k}: {v_str}")
                    file_log(f"    [Pandas] {json.dumps(p_data, ensure_ascii=False, default=str)}")
                else:
                    print(f"     [Pandas Oracle] ❌ 不存在 (ID {rid} 不在 DataFrame 中)")
                    file_log(f"    [Pandas] NOT FOUND")

                # Milvus 侧
                if rid in milvus_by_id:
                    m_data = format_row_for_display(milvus_by_id[rid])
                    null_fields_m = [k for k, v in m_data.items() if v is None and k != "id"]
                    print(f"     [Milvus 实际]")
                    if null_fields_m:
                        print(f"       ⚠️ Null 字段: {null_fields_m}")
                    for k, v in m_data.items():
                        if k == "id":
                            continue
                        if k in expr_str or v is None or isinstance(v, (dict, list)):
                            v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                            if len(v_str) > 200:
                                v_str = v_str[:200] + "..."
                            print(f"       {k}: {v_str}")
                    file_log(f"    [Milvus] {json.dumps(m_data, ensure_ascii=False, default=str)}")

                    # 关键差异高亮: 比较 Pandas vs Milvus 中 null/非null 不一致的字段
                    if p_rows:
                        p_data = format_row_for_display(p_rows[0])
                        diffs = []
                        for k in set(list(p_data.keys()) + list(m_data.keys())):
                            if k in ("id", "vector"):
                                continue
                            pv = p_data.get(k)
                            mv = m_data.get(k)
                            p_null = pv is None
                            m_null = mv is None
                            if p_null != m_null:
                                diffs.append(f"{k}: Pandas={'null' if p_null else repr(pv)}, Milvus={'null' if m_null else repr(mv)}")
                        if diffs:
                            print(f"     🔴 Null 不一致: {'; '.join(diffs)}")
                            file_log(f"    NULL_MISMATCH: {'; '.join(diffs)}")
                else:
                    print(f"     [Milvus 实际] ❌ 不存在 (查询未返回)")
                    file_log(f"    [Milvus] NOT FOUND")

        file_log(f"Start Testing: {total_test} rounds,seed : {current_seed}")
        file_log("=" * 50)

        for i in range(total_test):
            cl = pick_consistency()
            print(f"\r⏳ Running Test {i+1}/{total_test} [CL={cl}]...          ", end="", flush=True)

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
                    elif new_index_type == "DISKANN":
                        create_params = {
                            "index_type": "DISKANN",
                            "metric_type": METRIC_TYPE,
                            "params": {}
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
                    
                    # 使用显式 index_name 避免 AmbiguousIndexName
                    mm.col.drop_index(index_name=VECTOR_INDEX_NAME)
                    file_log(f"[Maintenance] Dropped vector index (was {old_index_type}) at round {i}")
                    
                    mm.col.create_index(
                        field_name="vector",
                        index_params=create_params,
                        index_name=VECTOR_INDEX_NAME
                    )
                    utility.wait_for_index_building_complete(COLLECTION_NAME, index_name=VECTOR_INDEX_NAME)
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

            # --- 随机触发标量索引重建 ---
            if i > 0 and i % 35 == 0:
                try:
                    mm.col.release()
                    file_log(f"[Maintenance] Released collection for scalar index rebuild at round {i}")

                    old_idx, new_idx = mm.rebuild_scalar_indexes(dm.schema_config)
                    file_log(f"[Maintenance] Scalar index rebuild at round {i}: {old_idx} -> {new_idx}")

                    mm.col.load()
                    file_log(f"[Maintenance] Reloaded collection after scalar index rebuild at round {i}")
                except Exception as e:
                    file_log(f"[Maintenance] Scalar index rebuild failed at round {i}: {e}")
                    try:
                        mm.col.load()
                        file_log(f"[Maintenance] Recovery after scalar rebuild - collection reloaded at round {i}")
                    except Exception as e2:
                        file_log(f"[Maintenance] Scalar rebuild recovery failed at round {i}: {e2}")

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
                            mm.col.compact()
                            mm.col.wait_for_compaction_completed()
                            mm.col.release()
                            mm.col.load()
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
                        mm.col.compact()
                        mm.col.wait_for_compaction_completed()
                        mm.col.release()
                        mm.col.load()

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

            # --- Schema Evolution: 随机添加新字段 ---
            if enable_dynamic_ops and i > 0 and i % 30 == 0:
                try:
                    field_config = dm.evolve_schema_add_field()
                    if field_config:
                        ftype_name = get_type_name(field_config["type"])
                        if field_config["type"] == DataType.ARRAY:
                            ftype_name += f"<{get_type_name(field_config['element_type'])}>"
                        file_log(f"[SchemaEvolution] Adding field '{field_config['name']}' ({ftype_name}) at round {i}")

                        # 1. 向 Milvus 添加新字段（现有数据的新字段值均为 null）
                        mm.add_evolved_field(field_config)
                        file_log(f"[SchemaEvolution] Field '{field_config['name']}' added to Milvus successfully")

                        # 回填: 使用 partial_update=True 为部分现有行填充新字段值
                        backfill_data = dm.backfill_evolved_field(field_config)
                        if backfill_data:
                            fill_count = mm.backfill_field_data(dm, field_config, backfill_data)
                            file_log(f"[SchemaEvolution] Backfilled {fill_count}/{len(backfill_data)} rows for '{field_config['name']}'")
                        else:
                            file_log(f"[SchemaEvolution] No rows selected for backfill of '{field_config['name']}'")

                        # 验证: 查询新字段的 is null / is not null
                        time.sleep(0.5)  # 等待新 schema 生效
                        try:
                            null_res = mm.col.query(
                                f"{field_config['name']} is null",
                                output_fields=["id"],
                                consistency_level="Strong"
                            )
                            not_null_res = mm.col.query(
                                f"{field_config['name']} is not null",
                                output_fields=["id"],
                                consistency_level="Strong"
                            )
                            total_in_milvus = len(null_res) + len(not_null_res)
                            total_in_pandas = len(dm.df)
                            pandas_null_count = int(dm.df[field_config['name']].isnull().sum())
                            pandas_notnull_count = total_in_pandas - pandas_null_count

                            file_log(f"[SchemaEvolution] Verify '{field_config['name']}': "
                                     f"Milvus(null={len(null_res)}, not_null={len(not_null_res)}, total={total_in_milvus}) "
                                     f"Pandas(null={pandas_null_count}, not_null={pandas_notnull_count}, total={total_in_pandas})")

                            if len(null_res) != pandas_null_count or len(not_null_res) != pandas_notnull_count:
                                msg = (f"SchemaEvolution MISMATCH for '{field_config['name']}': "
                                       f"Milvus null={len(null_res)} vs Pandas null={pandas_null_count}, "
                                       f"Milvus not_null={len(not_null_res)} vs Pandas not_null={pandas_notnull_count}")
                                print(f"\n❌ {msg}")
                                file_log(f"[SchemaEvolution] ❌ {msg}")
                                failed_cases.append({
                                    "id": i, "expr": f"SchemaEvolution({field_config['name']})",
                                    "detail": msg, "seed": current_seed
                                })
                            else:
                                file_log(f"[SchemaEvolution] ✅ Verify PASSED for '{field_config['name']}'")
                        except Exception as ve:
                            file_log(f"[SchemaEvolution] Verify failed: {ve}")

                except Exception as e:
                    file_log(f"[SchemaEvolution] Failed at round {i}: {e}")
                    import traceback
                    file_log(f"[SchemaEvolution] Traceback:\n{traceback.format_exc()}")
                    # 尝试恢复
                    try:
                        mm.col.load()
                    except:
                        pass

            # 生成查询
            depth = random.randint(1, 15)
            expr_str = ""
            while not expr_str:
                expr_str, pandas_mask = qg.gen_complex_expr(depth)

            log_header = f"[Test {i}]"
            file_log(f"\n{log_header} Expr: {expr_str}")
            file_log(f"  ConsistencyLevel: {cl}")

            # Pandas 计算
            expected_ids = set(dm.df[pandas_mask.fillna(False)]["id"].values.tolist())

            try:
                start_t = time.time()

                iterator = mm.col.query_iterator(
                    batch_size=10000,
                    expr=expr_str,
                    output_fields=["id"],
                    consistency_level=cl
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
                    # 计算差异
                    missing = expected_ids - actual_ids
                    extra = actual_ids - expected_ids
                    diff_msg = ""
                    if missing: diff_msg += f"Missing IDs: {list(missing)} "
                    if extra: diff_msg += f"Extra IDs: {list(extra)}"

                    # 非 Strong 一致性允许暂时不一致, 降级为警告
                    if cl != "Strong":
                        file_log(f"  -> WARN (non-Strong CL={cl}): {diff_msg}")
                        file_log(f"   Expr: {expr_str}")
                        print(f"   Expr: {expr_str}")
                        print(f"\n⚠️  [Test {i}] WARN (CL={cl}): Expected {len(expected_ids)} vs Actual {len(actual_ids)}")
                    else:
                        # --- Strong 级别: 真正的 MISMATCH ---
                        print(f"\n❌ [Test {i}] MISMATCH!")
                        print(f"   Expr: {expr_str}")
                        print(f"   Expected: {len(expected_ids)} vs Actual: {len(actual_ids)}")
                        print(f"   Diff: {diff_msg}")
                        print(f"   🔑 复现此bug: python milvus_fuzz_oracle.py --seed {current_seed}\n")

                        file_log(f"  -> MISMATCH! {diff_msg}")
                        file_log(f"  -> REPRODUCTION SEED: {current_seed}")

                    # 记录与展示具体数据行，便于一眼确认异常
                    if cl != "Strong":
                        # 非 Strong: 记日志 + 展示 Pandas vs Milvus 对比数据
                        if missing:
                            print_diff_evidence(missing, "Missing (Pandas有/Milvus无)", expr_str)
                        if extra:
                            print_diff_evidence(extra, "Extra (Milvus有/Pandas无)", expr_str)

                            print("\n   🔍 Verifying Extra IDs individually:")
                            for eid in list(extra)[:5]:
                                try:
                                    verify_res = mm.col.query(
                                        f"id == {eid}",
                                        output_fields=["id"],
                                        limit=1,
                                        consistency_level="Strong"
                                    )
                                    exists_in_db = len(verify_res) > 0

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
                    else:
                        # Strong: 完整的错误分析 + Pandas vs Milvus 数据对比
                        if missing:
                            print_diff_evidence(missing, "Missing (Pandas有/Milvus无)", expr_str)
                        if extra:
                            print_diff_evidence(extra, "Extra (Milvus有/Pandas无)", expr_str)

                            # 【验证】单独查询 Extra IDs，确认它们是否真的被 Milvus 返回
                            print("\n   🔍 Verifying Extra IDs individually:")
                            for eid in list(extra)[:5]:
                                try:
                                    verify_res = mm.col.query(
                                        f"id == {eid}",
                                        output_fields=["id"],
                                        limit=1,
                                        consistency_level="Strong"
                                    )
                                    exists_in_db = len(verify_res) > 0

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

                        # 加入错误列表 (仅 Strong 级别)
                        failed_cases.append({
                            "id": i,
                            "expr": expr_str,
                            "detail": f"Exp: {len(expected_ids)} vs Act: {len(actual_ids)}. {diff_msg}",
                            "seed": current_seed,
                            "consistency_level": cl
                        })
                # --- 向量 + 标量联合校验（HNSW/IVF/FLAT 均可），可选执行 ---
                # 仅当存在满足标量条件的数据时才做检查，且按比例抽样，避免对公用服务器造成额外压力。
                if expected_ids and random.random() < VECTOR_CHECK_RATIO:
                    try:
                        # 随机挑一条已有向量作为查询向量，确保搜索能命中实际数据
                        q_idx = random.randint(0, len(dm.vectors) - 1)
                        q_vec = dm.vectors[q_idx].tolist()

                        search_k = min(VECTOR_TOPK, len(dm.df))
                        
                        # 使用全局 METRIC_TYPE（避免 col.indexes 触发 AmbiguousIndexName）
                        current_metric_type = METRIC_TYPE
                        
                        # 根据当前索引类型动态设置搜索参数
                        if CURRENT_INDEX_TYPE == "HNSW":
                            ef_param = max(64, search_k + 8)  # ef must be > k for HNSW
                            search_params = {"metric_type": current_metric_type, "params": {"ef": ef_param}}
                        elif CURRENT_INDEX_TYPE.startswith("IVF"):
                            search_params = {"metric_type": current_metric_type, "params": {"nprobe": 16}}
                        elif CURRENT_INDEX_TYPE == "DISKANN":
                            search_params = {"metric_type": current_metric_type, "params": {"search_list": max(20, search_k + 8)}}
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
            if ftype in ALL_INT_TYPES + ALL_FLOAT_TYPES + [DataType.VARCHAR, DataType.BOOL]:
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
                elif ftype in ALL_INT_TYPES:
                    exprs.append(self._gen_boundary_int(fname, val))
                elif ftype in ALL_FLOAT_TYPES:
                    exprs.append(self._gen_boundary_float(fname, val, ftype))
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

        elif ftype in ALL_INT_TYPES:
            # 30% 概率生成算术表达式，否则使用边界表达式
            if random.random() < 0.3:
                res = self.gen_arithmetic_expr(fname, val, ftype)
                if res: return res
            return self._gen_boundary_int(fname, val)

        elif ftype in ALL_FLOAT_TYPES:
            # 30% 概率生成算术表达式，否则使用边界表达式
            if random.random() < 0.3:
                res = self.gen_arithmetic_expr(fname, val, ftype)
                if res: return res
            return self._gen_boundary_float(fname, val, ftype)

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
                if isinstance(subset[0], bool):
                    subset.append(not subset[0])
                elif isinstance(subset[0], str):
                    subset.append("fake_val")
                else:
                    subset.append(999999)
                strategies.append(f'array_contains_any({fname}, {json.dumps(subset)})')

            # 策略 5: [新增] 数组元素直接索引 (Milvus 支持 array[0])
            if length > 0:
                idx = random.randint(0, length - 1)
                item = val[idx]
                # 注意: isinstance(True, int) == True, 必须先检查 bool
                if isinstance(item, bool):
                    strategies.append(f'{fname}[{idx}] == {str(item).lower()}')
                elif isinstance(item, (int, float)):
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
        if ftype in ALL_INT_TYPES:
            return f"{fname} == {int(val)}"
        elif ftype in ALL_FLOAT_TYPES:
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
        # [WORKAROUND: Milvus NOT(UNKNOWN) Bug]
        # 原始代码:
        #   strategies.append(f"not ({fname} < {val})")
        #   strategies.append(f"not ({fname} > {val})")
        if '"' in fname:
            base_field = fname.split('[')[0]
            strategies.append(f"not ({base_field} is not null and ({fname} < {val}))")
            strategies.append(f"not ({base_field} is not null and ({fname} > {val}))")
        else:
            strategies.append(f"not ({fname} < {val})")
            strategies.append(f"not ({fname} > {val})")

        # 2. 0 的特殊处理
        if val == 0:
            strategies.append(f"{fname} == 0")
        else:
            # [WORKAROUND: Milvus JSON != Bug]
            # 原始代码: strategies.append(f"{fname} != 0")
            if '"' not in fname:
                strategies.append(f"{fname} != 0")
            else:
                strategies.append(f"{fname} > {val - 1}")

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
        # [WORKAROUND: Milvus JSON != Bug]
        # 原始代码: strategies.append(f"{fname} != {fake}")
        if '"' not in fname:
            strategies.append(f"{fname} != {fake}")
        else:
            strategies.append(f"({fname} > {val - 1} and {fname} < {val + 1})")

        return random.choice(strategies)

    def _gen_boundary_float(self, fname, val, ftype=None):

        val = float(val)
        strategies = []

        # 1. 基础范围 (浮点数不建议用 ==)
        # [WORKAROUND] FLOAT 字段: 使用 float32 ULP-aware 边界
        is_f32 = (ftype == DataType.FLOAT)
        lo, hi = self._float_safe_range(val, is_f32)
        strategies.append(f"({fname} > {lo} and {fname} < {hi})")

        # 2. 宽松范围 (>= / <= 对 float32 降精度天然安全，使用固定 epsilon 即可)
        epsilon = 1e-5
        strategies.append(f"{fname} >= {val - epsilon}")
        strategies.append(f"{fname} <= {val + epsilon}")

        # 3. 排除明显错误的值
        # [WORKAROUND: Milvus JSON != Bug]
        # 原始代码: strategies.append(f"{fname} != {val + 1.0}")
        if '"' not in fname:
            strategies.append(f"{fname} != {val + 1.0}")
        else:
            strategies.append(f"{fname} < {val + 1.0}")

        # 4. 绝对值/符号逻辑
        if val > 0:
            strategies.append(f"{fname} > -0.0001")
        elif val < 0:
            strategies.append(f"{fname} < 0.0001")

        # 5. 逻辑否定（与 >=/<= 等价），增强操作符多样性
        # [WORKAROUND: Milvus NOT(UNKNOWN) Bug]
        # 原始代码:
        #   strategies.append(f"not ({fname} > {val})")
        #   strategies.append(f"not ({fname} < {val})")
        if '"' in fname:
            base_field = fname.split('[')[0]
            strategies.append(f"not ({base_field} is not null and ({fname} > {val}))")
            strategies.append(f"not ({base_field} is not null and ({fname} < {val}))")
        else:
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
            # [WORKAROUND: Milvus JSON != Bug]
            # 原始代码: strategies.append(f'{fname} != "{val}_fake"')
            if '"' not in str(fname)[:5]:
                strategies.append(f'{fname} != "{val}_fake"')
            else:
                strategies.append(f'{fname} >= "{val}"')

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
            # [WORKAROUND: Milvus JSON != Bug]
            # 原始代码:
            #   op = random.choice(["==", "!="])
            #   if op == "!=": return f"{full_field} != {str(not current).lower()}"
            #   else: return f"{full_field} == {val_str}"
            return f"{full_field} == {val_str}"

        elif isinstance(current, int):
            return self._gen_boundary_int(full_field, current)

        elif isinstance(current, float):
            return self._gen_boundary_float(full_field, current)

        elif isinstance(current, str):
            return self._gen_boundary_str(full_field, current)

        return None

    def gen_arithmetic_expr(self, fname, val, ftype=None):
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

        # 计算预期结果 (使用 C/Go 风格取模，与 Milvus 一致)
        try:
            if op == "+": res = val + operand
            elif op == "-": res = val - operand
            elif op == "*": res = val * operand
            elif op == "%": res = int(np.fmod(val, operand))
        except:
            return None

        # 构造 Milvus 表达式
        # 注意：浮点数比较需要用范围
        # [WORKAROUND] FLOAT 字段的字面量会被 Milvus 隐式降精度到 float32
        if isinstance(res, float):
            is_f32 = (ftype == DataType.FLOAT)
            lo, hi = self._float_safe_range(res, is_f32)
            return f"({fname} {op} {operand} > {lo} and {fname} {op} {operand} < {hi})"
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
    global INDEX_TYPE, CURRENT_INDEX_TYPE, VECTOR_CHECK_RATIO, VECTOR_TOPK, _GLOBAL_ID_COUNTER, METRIC_TYPE, SCALAR_INDEX_PROBABILITY
    
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
    SCALAR_INDEX_PROBABILITY = random.uniform(0.3, 0.8)
    print(f"   索引类型: {INDEX_TYPE}, 度量类型: {METRIC_TYPE}")
    print(f"   标量索引概率: {SCALAR_INDEX_PROBABILITY:.2f}")

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
    vector_checks = 0
    vector_failures = 0

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

    def build_vector_search_params(search_k):
        """根据当前向量索引类型构建检索参数，尽量提高召回稳定性。"""
        metric = METRIC_TYPE
        if CURRENT_INDEX_TYPE == "HNSW":
            ef_param = max(128, search_k * 2)
            return {"metric_type": metric, "params": {"ef": ef_param}}
        if CURRENT_INDEX_TYPE and CURRENT_INDEX_TYPE.startswith("IVF"):
            nprobe = min(128, max(16, search_k // 2))
            return {"metric_type": metric, "params": {"nprobe": nprobe}}
        if CURRENT_INDEX_TYPE == "DISKANN":
            search_list = max(40, search_k * 2)
            return {"metric_type": metric, "params": {"search_list": search_list}}
        return {"metric_type": metric, "params": {}}

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
                        mm.col.compact()
                        mm.col.wait_for_compaction_completed()
                        mm.col.release()
                        mm.col.load()
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
                        mm.col.compact()
                        mm.col.wait_for_compaction_completed()
                        mm.col.release()
                        mm.col.load()
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

            # --- 随机触发标量索引重建 ---
            if i > 0 and i % 50 == 0:
                try:
                    mm.col.release()
                    file_log(f"[Maintenance] Released collection for scalar index rebuild at round {i}")
                    old_idx, new_idx = mm.rebuild_scalar_indexes(dm.schema_config)
                    file_log(f"[Maintenance] Scalar index rebuild at round {i}: {old_idx} -> {new_idx}")
                    mm.col.load()
                    file_log(f"[Maintenance] Reloaded after scalar index rebuild at round {i}")
                except Exception as e:
                    file_log(f"[Maintenance] Scalar index rebuild failed at round {i}: {e}")
                    try:
                        mm.col.load()
                    except:
                        pass

            # --- Schema Evolution (PQS mode) ---
            if enable_dynamic_ops and i > 0 and i % 30 == 0:
                try:
                    field_config = dm.evolve_schema_add_field()
                    if field_config:
                        ftype_name = get_type_name(field_config["type"])
                        if field_config["type"] == DataType.ARRAY:
                            ftype_name += f"<{get_type_name(field_config['element_type'])}>"
                        file_log(f"[SchemaEvolution] Adding field '{field_config['name']}' ({ftype_name}) at round {i}")
                        mm.add_evolved_field(field_config)
                        file_log(f"[SchemaEvolution] Field '{field_config['name']}' added to Milvus successfully")

                        # 回填: 使用 partial_update=True 为部分现有行填充新字段值
                        backfill_data = dm.backfill_evolved_field(field_config)
                        if backfill_data:
                            fill_count = mm.backfill_field_data(dm, field_config, backfill_data)
                            file_log(f"[SchemaEvolution] Backfilled {fill_count}/{len(backfill_data)} rows for '{field_config['name']}'")
                        else:
                            file_log(f"[SchemaEvolution] No rows selected for backfill of '{field_config['name']}'")
                except Exception as e:
                    file_log(f"[SchemaEvolution] Failed at round {i}: {e}")
                    try:
                        mm.col.load()
                    except:
                        pass

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
                res = mm.col.query(expr, output_fields=["id"], limit=10000, consistency_level="Strong")
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
                    print(f"   🔎 EVIDENCE (Pandas Oracle vs Milvus 数据对比):")

                    # --- Pandas 侧 ---
                    print(f"   [Pandas Oracle]:")
                    print(f"     [meta_json]: {json.dumps(json_data, indent=4, ensure_ascii=False)}")
                    for k, v in scalar_data.items():
                        print(f"     {k:<10}: {v}")

                    # --- Milvus 侧 ---
                    all_output_fields = ["id"] + [fc["name"] for fc in dm.schema_config]
                    try:
                        milvus_rows = mm.col.query(
                            f"id in [{pivot_id}]",
                            output_fields=all_output_fields,
                            limit=1,
                            consistency_level="Strong"
                        )
                    except Exception as mq_e:
                        milvus_rows = []
                        print(f"     ⚠️ Milvus 查询失败: {mq_e}")

                    if milvus_rows:
                        m_row = milvus_rows[0]
                        m_data = {k: v for k, v in m_row.items() if k != "vector"}
                        m_json = m_data.pop("meta_json", {})
                        print(f"   [Milvus 实际]:")
                        print(f"     [meta_json]: {json.dumps(m_json, indent=4, ensure_ascii=False, default=str)}")
                        for k, v in m_data.items():
                            v_str = str(v)
                            if len(v_str) > 200: v_str = v_str[:200] + "..."
                            print(f"     {k:<10}: {v_str}")
                        # Null 差异高亮
                        diffs = []
                        for k in set(list(scalar_data.keys()) + list(m_data.keys())):
                            if k in ("id", "vector"): continue
                            pv, mv = scalar_data.get(k), m_data.get(k)
                            if (pv is None) != (mv is None):
                                diffs.append(f"{k}: Pandas={'null' if pv is None else repr(pv)[:50]}, Milvus={'null' if mv is None else repr(mv)[:50]}")
                        if diffs:
                            print(f"   🔴 Null 不一致: {'; '.join(diffs)}")
                            file_log(f"  NULL_MISMATCH: {'; '.join(diffs)}")
                    else:
                        print(f"   [Milvus 实际] ❌ 查询未返回该行")
                    print("-" * 50)

                    # 3. 记录日志
                    file_log(f"  -> ❌ FAIL! Target ID {pivot_id} NOT found.")
                    file_log(f"  -> Pandas Row Data: {safe_row}")
                    if milvus_rows:
                        file_log(f"  -> Milvus Row Data: {m_data}")

                    errors.append({"id": pivot_id, "expr": expr})

                # --- PQS 向量检验 ---
                # 用 pivot 对应向量做 ANN 检索，期望结果中至少包含 pivot_id。
                # 为控制负载，按 VECTOR_CHECK_RATIO 比例抽样执行。
                if random.random() < VECTOR_CHECK_RATIO and len(dm.vectors) > random_idx:
                    vector_checks += 1
                    try:
                        pivot_vec = dm.vectors[random_idx].tolist()
                        search_k = min(VECTOR_TOPK, len(dm.df))
                        search_params = build_vector_search_params(search_k)
                        vector_res = mm.col.search(
                            data=[pivot_vec],
                            anns_field="vector",
                            param=search_params,
                            limit=search_k,
                            output_fields=["id"],
                        )

                        vector_ids = set()
                        if vector_res and len(vector_res) > 0:
                            for hit in vector_res[0]:
                                vector_ids.add(hit.get("id") if isinstance(hit, dict) else hit.id)

                        if pivot_id in vector_ids:
                            file_log(
                                f"  -> VectorSelfCheck PASS | k={search_k} | found={len(vector_ids)}"
                            )
                        else:
                            vector_failures += 1
                            top_ids = list(vector_ids)[:5]
                            print(f"\n⚠️ PQS Vector SelfCheck FAIL [Round {i}] | Target ID: {pivot_id} | TopIDs: {top_ids}")
                            file_log(
                                f"  -> VectorSelfCheck FAIL | target={pivot_id} | top_ids={top_ids} | params={search_params}"
                            )
                            errors.append({
                                "id": pivot_id,
                                "expr": expr,
                                "vector_check": True,
                                "detail": f"Vector self-check miss, top_ids={top_ids}",
                            })
                    except Exception as ve:
                        vector_failures += 1
                        print(f"\n⚠️ PQS Vector Check Error [Round {i}]: {ve}")
                        file_log(f"  -> VectorSelfCheck ERROR: {ve}")
                        errors.append({
                            "id": pivot_id,
                            "expr": expr,
                            "vector_check": True,
                            "detail": f"Vector check error: {ve}",
                        })

            except Exception as e:
                print(f"\n\n⚠️ Execution Error [Round {i}]: {e}")
                file_log(f"  -> EXECUTION ERROR: {e}")

    print("\n" + "="*60)
    print(f"📊 向量自检统计: {vector_checks} 次, 失败 {vector_failures} 次")
    if not errors:
        print(f"✅ PQS 测试完成。未发现错误。")
    else:
        print(f"🚫 PQS 测试完成。发现 {len(errors)} 个潜在 Bug！")
        print(f"📄 详细数据已记录至日志: {log_filename}")

def run_groupby_test(rounds=50, seed=None, enable_dynamic_ops=True):
    """
    专门用于检测GroupBy Key分裂和 strict_group_size 失效的问题
    """
    global METRIC_TYPE, INDEX_TYPE, CURRENT_INDEX_TYPE, VECTOR_CHECK_RATIO, VECTOR_TOPK, _GLOBAL_ID_COUNTER, SCALAR_INDEX_PROBABILITY
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
    SCALAR_INDEX_PROBABILITY = random.uniform(0.3, 0.8)
    print(f"   索引类型: {INDEX_TYPE}, 度量类型: {METRIC_TYPE}")
    print(f"   标量索引概率: {SCALAR_INDEX_PROBABILITY:.2f}")
    
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
        if f["type"] in ALL_INT_TYPES + [DataType.VARCHAR, DataType.BOOL]:
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
                        mm.col.compact()
                        mm.col.wait_for_compaction_completed()
                        mm.col.release()
                        mm.col.load()
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
                        mm.col.compact()
                        mm.col.wait_for_compaction_completed()
                        mm.col.release()
                        mm.col.load()
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

            # --- 随机触发标量索引重建 ---
            if i > 0 and i % 25 == 0:
                try:
                    mm.col.release()
                    file_log(f"[Maintenance] Released collection for scalar index rebuild at round {i}")
                    old_idx, new_idx = mm.rebuild_scalar_indexes(dm.schema_config)
                    file_log(f"[Maintenance] Scalar index rebuild at round {i}: {old_idx} -> {new_idx}")
                    mm.col.load()
                    file_log(f"[Maintenance] Reloaded after scalar index rebuild at round {i}")
                except Exception as e:
                    file_log(f"[Maintenance] Scalar index rebuild failed at round {i}: {e}")
                    try:
                        mm.col.load()
                    except:
                        pass

            # --- Schema Evolution (GroupBy mode) ---
            if enable_dynamic_ops and i > 0 and i % 30 == 0:
                try:
                    field_config = dm.evolve_schema_add_field()
                    if field_config:
                        ftype_name = get_type_name(field_config["type"])
                        if field_config["type"] == DataType.ARRAY:
                            ftype_name += f"<{get_type_name(field_config['element_type'])}>"
                        file_log(f"[SchemaEvolution] Adding field '{field_config['name']}' ({ftype_name}) at round {i}")
                        mm.add_evolved_field(field_config)
                        file_log(f"[SchemaEvolution] Field '{field_config['name']}' added to Milvus successfully")

                        # 回填: 使用 partial_update=True 为部分现有行填充新字段值
                        backfill_data = dm.backfill_evolved_field(field_config)
                        if backfill_data:
                            fill_count = mm.backfill_field_data(dm, field_config, backfill_data)
                            file_log(f"[SchemaEvolution] Backfilled {fill_count}/{len(backfill_data)} rows for '{field_config['name']}'")
                        else:
                            file_log(f"[SchemaEvolution] No rows selected for backfill of '{field_config['name']}'")
                except Exception as e:
                    file_log(f"[SchemaEvolution] Failed at round {i}: {e}")
                    try:
                        mm.col.load()
                    except:
                        pass

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
            # 使用全局 METRIC_TYPE（避免 col.indexes 触发 AmbiguousIndexName）
            current_metric_type = METRIC_TYPE

            idx_params_dict = {}
            if INDEX_TYPE == "HNSW":
                idx_params_dict = {"ef": 64}
            elif INDEX_TYPE.startswith("IVF"):
                idx_params_dict = {"nprobe": 10}
            elif INDEX_TYPE == "DISKANN":
                idx_params_dict = {"search_list": 30}
            
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
    parser.add_argument("--consistency", type=str, choices=["Strong", "Bounded", "Eventually", "Session"],
                        default=None, help="Force a specific consistency level (default: random)")

    # Execution Modes (Mutually Exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pqs", action="store_true", help="Run PQS (Predicate Query Search) Mode")
    group.add_argument("--equiv", action="store_true", help="Run Equivalence Mode")
    group.add_argument("--groupby-test", action="store_true", help="Run GroupBy Test Mode")

    parser.add_argument("--pqs-rounds", type=int, default=None, help="Override rounds for PQS mode (default: use --rounds)")

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
    if args.metric:
        ALL_METRIC_TYPES = [args.metric]

    # 2. Determine Mode and Execute
    print("=" * 80)
    print(f"🚀 Milvus Fuzz Oracle Startup")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Seed: {args.seed if args.seed is not None else '(Random)'}")
    print(f"   Metric: {args.metric if args.metric else '(Random from L2/IP/COSINE)'}")
    print(f"   Dynamic Ops: {enable_dynamic_ops}")
    print(f"   Chaos Rate: {CHAOS_RATE}")
    print(f"   Consistency: {args.consistency if args.consistency else '(Random)'}")

    pqs_rounds = args.pqs_rounds if args.pqs_rounds is not None else args.rounds
    if args.pqs or (args.pqs_rounds is not None and not args.equiv and not args.groupby_test):
        print(f"   Mode: PQS (Predicate Query Search)")
        print(f"   Rounds: {pqs_rounds}")
        print("=" * 80)
        run_pqs_mode(rounds=pqs_rounds, seed=args.seed, enable_dynamic_ops=enable_dynamic_ops)
    
    elif args.equiv:
        print(f"   Mode: Equivalence Test")
        print(f"   Rounds: {args.rounds}")
        print("=" * 80)
        run_equivalence_mode(rounds=args.rounds, seed=args.seed, enable_dynamic_ops=enable_dynamic_ops, consistency=args.consistency)

    elif args.groupby_test:
        print(f"   Mode: GroupBy Test")
        print(f"   Rounds: {args.rounds}")
        print("=" * 80)
        run_groupby_test(rounds=args.rounds, seed=args.seed, enable_dynamic_ops=enable_dynamic_ops)

    else:
        print(f"   Mode: Standard Fuzzing")
        print(f"   Rounds: {args.rounds}")
        print("=" * 80)
        run(rounds=args.rounds, seed=args.seed, enable_dynamic_ops=enable_dynamic_ops, consistency=args.consistency)