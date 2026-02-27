"""
Qdrant Dynamic Fuzzing System (Phase 1: Full Oracle Mode)
Features:
1. Dynamic Schema: Random 5-20 fields + JSON-like payload structure
2. Configurable Data: Adjustable vectors and dimensions
3. Robust Query Generation: Random filter expressions with type safety
4. Oracle Testing: Pandas comparison for correctness verification
5. PQS (Pivot Query Synthesis): Must-hit query testing
6. Equivalence Testing: Query transformation validation
"""
import time
import random
import string
import math
import numpy as np
import pandas as pd
import json
import sys
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny, MatchExcept,
    Range, DatetimeRange, PayloadField,
    IsNullCondition, IsEmptyCondition,
    HasIdCondition, PointIdsList,
    GeoPoint, GeoBoundingBox, GeoRadius,
    PayloadSchemaType
)

# --- Configuration (User Specified) ---
HOST = "127.0.0.1"
PORT = 6333                # Qdrant default port
COLLECTION_NAME = "fuzz_oracle_v1"
N = 5000                   # 数据量
DIM = 128                  # 向量维度
BATCH_SIZE = 500           # 批次大小
SLEEP_INTERVAL = 0.01      # 每次插入后暂停

# 全局度量类型列表（延迟初始化，在 run() 内种子设置后随机选取）
ALL_DISTANCE_TYPES = [Distance.EUCLID, Distance.COSINE, Distance.DOT, Distance.MANHATTAN]
DISTANCE_TYPE = None  # 延迟初始化

# 随机化配置
INDEX_TYPE = random.choice(["hnsw"])  # Qdrant 主要使用 HNSW
VECTOR_CHECK_RATIO = random.uniform(0.2, 0.8)  # 20-80% 概率做向量+标量联合校验
VECTOR_TOPK = random.randint(50, 200)          # TopK 在 50-200 之间随机

# 混淆开关（默认关闭）
CHAOS_RATE = 0.0

# 唯一 ID 计数器（用于动态插入）
_global_id_counter = N * 10

def generate_unique_id():
    """生成全局唯一 ID（用于动态 upsert 新行）"""
    global _global_id_counter
    _global_id_counter += 1
    return _global_id_counter

# --- 1. Data Manager ---

class FieldType:
    """模拟字段类型枚举"""
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    STRING = "STRING"
    JSON = "JSON"
    ARRAY_INT = "ARRAY_INT"
    ARRAY_STR = "ARRAY_STR"
    ARRAY_FLOAT = "ARRAY_FLOAT"
    DATETIME = "DATETIME"
    GEO = "GEO"

def get_type_name(dtype):
    """Map field type to string"""
    return dtype

class DataManager:
    _id_counter = 0  # 类级别 ID 计数器

    def __init__(self):
        self.df = None
        self.vectors = None
        self.schema_config = []
        self.null_ratio = random.uniform(0.05, 0.15)
        self.array_capacity = random.randint(5, 50)
        self.json_max_depth = random.randint(1, 5)
        self.int_range = random.randint(5000, 100000)
        self.double_scale = random.uniform(100, 10000)

    KEY_POOL = [f"k_{i}" for i in range(20)] + ["user", "log", "data", "a_b", "test_key"]

    def _gen_random_json_structure(self, rng, depth=3):
        """递归生成随机 JSON 结构"""
        if depth == 0 or rng.random() < 0.3:
            r = rng.random()
            if r < 0.2: return int(rng.integers(-100000, 100000))
            elif r < 0.4: return float(rng.random() * 1000)
            elif r < 0.6: return self._random_string(0, 8)
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
        types_pool = [FieldType.INT, FieldType.FLOAT, FieldType.BOOL, FieldType.STRING, FieldType.DATETIME]

        for i in range(num_fields):
            ftype = random.choice(types_pool)
            self.schema_config.append({"name": f"c{i}", "type": ftype})

        # 添加 JSON 字段（Qdrant 中作为嵌套 payload）
        self.schema_config.append({"name": "meta_json", "type": FieldType.JSON})
        
        # 添加 Array 字段
        self.schema_config.append({
            "name": "tags_array",
            "type": FieldType.ARRAY_INT,
            "max_capacity": self.array_capacity
        })
        
        # 添加字符串数组
        self.schema_config.append({
            "name": "labels_array",
            "type": FieldType.ARRAY_STR,
            "max_capacity": self.array_capacity
        })

        # 添加浮点数组
        self.schema_config.append({
            "name": "scores_array",
            "type": FieldType.ARRAY_FLOAT,
            "max_capacity": self.array_capacity
        })

        # 添加 GEO 字段（经纬度坐标）
        self.schema_config.append({"name": "location_geo", "type": FieldType.GEO})

        print(f"   -> Generated {len(self.schema_config)} dynamic fields (plus id & vector).")
        print("   -> Schema Structure:")
        for f in self.schema_config:
            t_name = get_type_name(f["type"])
            print(f"      - {f['name']:<15} : {t_name}")

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

            if ftype == FieldType.INT:
                values = rng.integers(-self.int_range, self.int_range, size=N).tolist()
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.FLOAT:
                values = (rng.random(N) * self.double_scale).tolist()
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.BOOL:
                values = rng.choice([True, False], size=N).tolist()
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.STRING:
                values = [self._random_string(0, random.randint(5, 50)) for _ in range(N)]
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.JSON:
                json_list = []
                for _ in range(N):
                    if rng.random() < self.null_ratio:
                        json_list.append(None)
                        continue
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
                    json_list.append(base_obj)
                data[fname] = json_list
            elif ftype == FieldType.ARRAY_INT:
                arr_list = []
                for _ in range(N):
                    length = random.randint(0, 5)
                    arr_list.append(list(rng.integers(0, 100, size=length)))
                data[fname] = self._apply_nulls(arr_list, rng)
            elif ftype == FieldType.ARRAY_STR:
                arr_list = []
                for _ in range(N):
                    length = random.randint(0, 5)
                    arr_list.append([self._random_string(2, 8) for _ in range(length)])
                data[fname] = self._apply_nulls(arr_list, rng)
            elif ftype == FieldType.ARRAY_FLOAT:
                arr_list = []
                for _ in range(N):
                    length = random.randint(0, 5)
                    arr_list.append([round(float(rng.random() * 1000), 2) for _ in range(length)])
                data[fname] = self._apply_nulls(arr_list, rng)
            elif ftype == FieldType.DATETIME:
                # 生成时间戳（epoch 秒数，整型）
                # 范围：2020-01-01 到 2025-01-01
                epoch_2020 = int(datetime(2020, 1, 1).timestamp())
                epoch_2025 = int(datetime(2025, 1, 1).timestamp())
                values = [int(rng.integers(epoch_2020, epoch_2025)) for _ in range(N)]
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.GEO:
                geo_list = []
                for _ in range(N):
                    if rng.random() < self.null_ratio:
                        geo_list.append(None)
                    else:
                        geo_list.append({
                            "lat": round(float(rng.uniform(-90, 90)), 6),
                            "lon": round(float(rng.uniform(-180, 180)), 6)
                        })
                data[fname] = geo_list

        self.df = pd.DataFrame(data)
        print("✅ Data Generation Complete.")

    def _apply_nulls(self, values, rng):
        """根据 null_ratio 用 None 替换部分值"""
        if not isinstance(values, list):
            values = list(values)
        for idx in range(len(values)):
            if rng.random() < self.null_ratio:
                values[idx] = None
        return values

    def generate_single_row(self, id_override=None):
        """生成单条数据行（用于动态插入/upsert）"""
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        row = {}
        # 生成唯一 id
        if id_override is not None:
            row["id"] = int(id_override)
        else:
            DataManager._id_counter += 1
            row["id"] = 10000000 + DataManager._id_counter * 1000 + random.randint(1, 999)

        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]

            if ftype == FieldType.INT:
                row[fname] = int(rng.integers(-self.int_range, self.int_range))
            elif ftype == FieldType.FLOAT:
                row[fname] = float(rng.random() * self.double_scale)
            elif ftype == FieldType.BOOL:
                row[fname] = bool(rng.choice([True, False]))
            elif ftype == FieldType.STRING:
                row[fname] = self._random_string(0, random.randint(5, 50))
            elif ftype == FieldType.JSON:
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
            elif ftype == FieldType.ARRAY_INT:
                arr_len = rng.integers(0, 6)
                row[fname] = [int(x) for x in rng.integers(0, 100, size=arr_len)]
            elif ftype == FieldType.ARRAY_STR:
                arr_len = rng.integers(0, 6)
                row[fname] = [self._random_string(2, 8) for _ in range(arr_len)]
            elif ftype == FieldType.ARRAY_FLOAT:
                arr_len = rng.integers(0, 6)
                row[fname] = [round(float(rng.random() * 1000), 2) for _ in range(arr_len)]
            elif ftype == FieldType.DATETIME:
                from datetime import datetime as dt_cls
                epoch_2020 = int(dt_cls(2020, 1, 1).timestamp())
                epoch_2025 = int(dt_cls(2025, 1, 1).timestamp())
                row[fname] = int(rng.integers(epoch_2020, epoch_2025))
            elif ftype == FieldType.GEO:
                row[fname] = {
                    "lat": round(float(rng.uniform(-90, 90)), 6),
                    "lon": round(float(rng.uniform(-180, 180)), 6)
                }
        return row

    def generate_single_vector(self):
        """生成一条单位化的向量"""
        vec = np.random.randn(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

# --- 2. Qdrant Manager ---

class QdrantManager:
    def __init__(self):
        self.client = None

    def connect(self):
        print(f"🔌 Connecting to Qdrant at {HOST}:{PORT}...")
        try:
            self.client = QdrantClient(host=HOST, port=PORT, timeout=30)
            # 测试连接
            self.client.get_collections()
            print("✅ Connected to Qdrant successfully.")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("   (Please check if Qdrant container is running)")
            exit(1)

    def reset_collection(self, schema_config):
        global DISTANCE_TYPE
        # 延迟初始化距离度量（在种子设置后随机选取）
        if DISTANCE_TYPE is None:
            DISTANCE_TYPE = random.choice(ALL_DISTANCE_TYPES)
        print(f"📐 Distance Metric: {DISTANCE_TYPE}")

        # 删除已存在的 collection
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except:
            pass

        # 创建新 collection
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=DIM, distance=DISTANCE_TYPE),
        )
        print("🛠️ Collection Created.")

    def insert(self, dm):
        print(f"⚡ 3. Inserting Data (Batch={BATCH_SIZE}, Sleep={SLEEP_INTERVAL}s)...")
        records = dm.df.to_dict(orient="records")
        total = len(records)
        
        # 构建字段类型映射
        field_type_map = {f["name"]: f["type"] for f in dm.schema_config}

        def convert_numpy_types(obj):
            """递归转换 numpy 类型为 Python 原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return [convert_numpy_types(x) for x in obj.tolist()]
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(x) for x in obj]
            return obj
        
        def convert_value_by_type(k, v, field_type_map):
            """根据 schema 类型正确转换值，确保 INT 字段存为 int"""
            # 处理 None 和 NaN
            if v is None:
                return None
            if isinstance(v, float) and np.isnan(v):
                return None
            
            ftype = field_type_map.get(k)
            
            # INT 字段必须转为 int（Pandas 有 None 时会把整列变成 float）
            if ftype == FieldType.INT:
                return int(v)
            # BOOL 字段确保是 Python bool
            elif ftype == FieldType.BOOL:
                return bool(v)
            # DATETIME 字段转为 int（epoch 秒数）
            elif ftype == FieldType.DATETIME:
                return int(v)
            # GEO 字段确保是正确的 dict
            elif ftype == FieldType.GEO:
                if isinstance(v, dict) and "lat" in v and "lon" in v:
                    return {"lat": float(v["lat"]), "lon": float(v["lon"])}
                return None
            # 其他类型递归转换
            else:
                return convert_numpy_types(v)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_data = records[start:end]

            points = []
            for i, row in enumerate(batch_data):
                point_id = int(row["id"])
                vector = dm.vectors[start + i].tolist()

                # 构建 payload
                payload = {}
                for k, v in row.items():
                    if k == "id":
                        continue
                    # 根据 schema 类型正确转换
                    payload[k] = convert_value_by_type(k, v, field_type_map)

                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))

            # 重试逻辑
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                        wait=True
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"\n❌ Critical Error at batch {start}-{end}: {e}")
                        raise e
                    print(f"   ⚠️ Retry {attempt+1}/{max_retries} due to error...", end="\r")
                    time.sleep(2)

            print(f"   Inserted {end}/{total}...", end="\r")
            time.sleep(SLEEP_INTERVAL)

        print("\n✅ Insert Complete.")

        # 创建索引 (可选，Qdrant 默认会自动创建)
        print("🔨 Creating payload indexes...")
        for field in dm.schema_config:
            fname = field["name"]
            ftype = field["type"]
            try:
                if ftype == FieldType.INT:
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=fname,
                        field_schema=PayloadSchemaType.INTEGER
                    )
                elif ftype == FieldType.FLOAT:
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=fname,
                        field_schema=PayloadSchemaType.FLOAT
                    )
                elif ftype == FieldType.STRING:
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=fname,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                elif ftype == FieldType.BOOL:
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=fname,
                        field_schema=PayloadSchemaType.BOOL
                    )
                elif ftype == FieldType.DATETIME:
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=fname,
                        field_schema=PayloadSchemaType.INTEGER
                    )
                elif ftype == FieldType.GEO:
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=fname,
                        field_schema=PayloadSchemaType.GEO
                    )
            except Exception as e:
                # 索引创建失败不是致命错误
                pass
        print("✅ Index creation complete.")

# --- 3. Query Generator (Oracle Mode) ---

class OracleQueryGenerator:
    """
    生成 Qdrant 过滤条件并同时生成对应的 Pandas mask
    用于 Oracle 对比测试
    """
    def __init__(self, dm):
        self.schema = dm.schema_config
        self._dm = dm  # 保存 dm 引用，动态获取最新 df

    @property
    def df(self):
        """动态获取最新的 DataFrame（支持动态插入/删除/upsert 后数据同步）"""
        return self._dm.df

    def _random_string(self, min_len=5, max_len=10):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=random.randint(min_len, max_len)))

    @staticmethod
    def _convert_to_native(val):
        """将 numpy 类型转换为 Python 原生类型"""
        if isinstance(val, np.integer):
            return int(val)
        elif isinstance(val, np.floating):
            return float(val)
        elif isinstance(val, np.bool_):
            return bool(val)
        elif isinstance(val, np.ndarray):
            return [OracleQueryGenerator._convert_to_native(x) for x in val.tolist()]
        elif isinstance(val, list):
            return [OracleQueryGenerator._convert_to_native(x) for x in val]
        return val

    def _payload_field(self, key):
        """Helper to build a PayloadField for null/empty conditions"""
        return PayloadField(key=key)

    def _is_null_condition(self, key):
        return IsNullCondition(is_null=self._payload_field(key))

    def _is_empty_condition(self, key):
        return IsEmptyCondition(is_empty=self._payload_field(key))

    def _empty_or_null_filter(self, key):
        """Treat null payloads as empty when checking array emptiness"""
        return Filter(should=[
            self._empty_filter(key),
            self._null_filter(key)
        ])

    def _condition_filter(self, condition):
        return Filter(must=[condition])

    def _null_filter(self, key):
        return self._condition_filter(self._is_null_condition(key))

    def _empty_filter(self, key):
        return self._condition_filter(self._is_empty_condition(key))

    def _get_json_val(self, obj, keys):
        """获取 JSON 嵌套值"""
        try:
            if obj is None: return None
            val = obj
            for k in keys:
                if val is None: return None
                if isinstance(k, int):
                    if isinstance(val, list) and len(val) > k: val = val[k]
                    else: return None
                else:
                    if isinstance(val, dict) and k in val: val = val[k]
                    else: return None
            return val
        except: return None

    def get_value_for_query(self, fname, ftype):
        """获取用于查询的值"""
        valid_series = self.df[fname].dropna()
        if not valid_series.empty and random.random() < 0.8:
            val = random.choice(valid_series.values)
            if hasattr(val, "item"): val = val.item()
            return val

        # 生成不存在的值
        if ftype == FieldType.INT:
            if not valid_series.empty:
                min_val, max_val = int(valid_series.min()), int(valid_series.max())
                return random.choice([max_val + 100000, min_val - 100000])
            return random.randint(-200000, 200000)

        elif ftype == FieldType.FLOAT:
            if not valid_series.empty:
                min_val, max_val = float(valid_series.min()), float(valid_series.max())
                val = random.choice([max_val + 100000.0, min_val - 100000.0])
            else:
                val = random.random() * 200000.0
            return float(np.float32(val))

        elif ftype == FieldType.BOOL:
            return random.choice([True, False])

        elif ftype == FieldType.STRING:
            chars = string.ascii_letters + string.digits + "!@#$%^&*"
            length = random.randint(15, 30)
            return ''.join(random.choices(chars, k=length))

        elif ftype == FieldType.JSON:
            new_key = self._random_string(10, 15)
            new_val = self._random_string(10, 15)
            return {"_non_exist_key": new_key, "_non_exist_val": new_val}

        elif ftype in [FieldType.ARRAY_INT, FieldType.ARRAY_STR]:
            return random.randint(50000, 100000)

        elif ftype == FieldType.ARRAY_FLOAT:
            return round(random.random() * 2000, 2)

        elif ftype == FieldType.DATETIME:
            # 生成一个随机时间戳（epoch 秒数）
            epoch_2020 = int(datetime(2020, 1, 1).timestamp())
            epoch_2025 = int(datetime(2025, 1, 1).timestamp())
            return random.randint(epoch_2020, epoch_2025)

        elif ftype == FieldType.GEO:
            return {"lat": round(random.uniform(-90, 90), 6), "lon": round(random.uniform(-180, 180), 6)}

        return None

    def gen_atomic_expr(self):
        """
        生成原子表达式
        返回: (qdrant_filter, pandas_mask, expr_str)
        """
        f = random.choice(self.schema)
        name, ftype = f["name"], f["type"]
        series = self.df[name]

        # 1. Null/Empty Check
        if random.random() < 0.15:
            if random.random() < 0.5:
                # is_null - 直接传布尔值
                filter_cond = self._null_filter(name)
                return (filter_cond, series.isnull(), f"{name} is null")
            else:
                # is_empty (for arrays) - 直接传布尔值
                if ftype in [FieldType.ARRAY_INT, FieldType.ARRAY_STR]:
                    filter_cond = self._empty_or_null_filter(name)
                    mask = series.apply(lambda x: x is None or (isinstance(x, list) and len(x) == 0))
                    return (filter_cond, mask, f"{name} is empty")
                else:
                    # Qdrant 不支持 is_null=False，需要用 must_not 包装
                    filter_cond = Filter(must_not=[self._is_null_condition(name)])
                    return (filter_cond, series.notnull(), f"{name} is not null")

        val = self.get_value_for_query(name, ftype)
        if val is None:
            filter_cond = self._null_filter(name)
            return (filter_cond, series.isnull(), f"{name} is null")

        # 安全比较函数 - 处理 None 和 NaN
        def safe_compare_scalar(op, target_val):
            def comp(x):
                # 使用 pd.isna() 统一处理 None 和 NaN
                if pd.isna(x): return False
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

        filter_cond = None
        mask = None
        expr_str = ""

        # 2. 根据类型生成条件
        if ftype == FieldType.BOOL:
            val_bool = bool(val)
            filter_cond = FieldCondition(
                key=name,
                match=MatchValue(value=val_bool)
            )
            mask = series.apply(safe_compare_scalar("==", val_bool))
            expr_str = f"{name} == {val_bool}"

        elif ftype == FieldType.INT:
            val_int = int(val)
            op = random.choice([">", "<", "==", "!=", ">=", "<="])
            
            if op == "==":
                filter_cond = FieldCondition(key=name, match=MatchValue(value=val_int))
            elif op == "!=":
                filter_cond = FieldCondition(key=name, match=MatchExcept(**{"except": [val_int]}))
            elif op == ">":
                filter_cond = FieldCondition(key=name, range=Range(gt=val_int))
            elif op == "<":
                filter_cond = FieldCondition(key=name, range=Range(lt=val_int))
            elif op == ">=":
                filter_cond = FieldCondition(key=name, range=Range(gte=val_int))
            elif op == "<=":
                filter_cond = FieldCondition(key=name, range=Range(lte=val_int))
            
            mask = series.apply(safe_compare_scalar(op, val_int))
            expr_str = f"{name} {op} {val_int}"

        elif ftype == FieldType.FLOAT:
            val_float = float(val)
            op = random.choice([">", "<", ">=", "<="])  # 避免 == 精度问题
            
            if op == ">":
                filter_cond = FieldCondition(key=name, range=Range(gt=val_float))
            elif op == "<":
                filter_cond = FieldCondition(key=name, range=Range(lt=val_float))
            elif op == ">=":
                filter_cond = FieldCondition(key=name, range=Range(gte=val_float))
            elif op == "<=":
                filter_cond = FieldCondition(key=name, range=Range(lte=val_float))
            
            mask = series.apply(safe_compare_scalar(op, val_float))
            expr_str = f"{name} {op} {val_float}"

        elif ftype == FieldType.STRING:
            op = random.choice(["==", "!=", "in"])
            
            if op == "==":
                filter_cond = FieldCondition(key=name, match=MatchValue(value=val))
                mask = series.apply(safe_compare_scalar("==", val))
                expr_str = f'{name} == "{val}"'
            elif op == "!=":
                filter_cond = FieldCondition(key=name, match=MatchExcept(**{"except": [val]}))
                mask = series.apply(safe_compare_scalar("!=", val))
                expr_str = f'{name} != "{val}"'
            else:  # in
                # 生成多个值的列表
                valid_vals = [str(x) for x in self.df[name].dropna().unique()[:5].tolist()]
                if val not in valid_vals:
                    valid_vals.append(val)
                filter_cond = FieldCondition(key=name, match=MatchAny(any=valid_vals))
                mask = series.apply(lambda x: x in valid_vals if x is not None else False)
                expr_str = f'{name} in {valid_vals}'

        elif ftype == FieldType.JSON:
            # JSON 嵌套查询
            return self.gen_json_expr(name, series, val)

        elif ftype in [FieldType.ARRAY_INT, FieldType.ARRAY_STR]:
            # 数组包含查询
            valid_series = self.df[name].dropna()
            all_items = []
            for arr in valid_series:
                if isinstance(arr, list):
                    all_items.extend(arr)
            
            if all_items:
                target = random.choice(all_items)
                # 转换 numpy 类型
                target = self._convert_to_native(target)
                filter_cond = FieldCondition(key=name, match=MatchAny(any=[target]))
                mask = series.apply(lambda x, t=target: t in x if isinstance(x, list) else False)
                expr_str = f'{name} contains {target}'
            else:
                # Qdrant 不支持 is_empty=False，需要用 must_not 包装
                filter_cond = Filter(must_not=[self._is_empty_condition(name)])
                mask = series.apply(lambda x: isinstance(x, list) and len(x) > 0)
                expr_str = f'{name} is not empty'

        elif ftype == FieldType.ARRAY_FLOAT:
            # 浮点数组 - 使用范围查询（检查数组中是否有元素在范围内）
            valid_series = self.df[name].dropna()
            all_items = []
            for arr in valid_series:
                if isinstance(arr, list):
                    all_items.extend(arr)
            if all_items:
                target = random.choice(all_items)
                target_f = float(target)
                epsilon = 0.5  # 较大的 epsilon 避免浮点精度问题
                filter_cond = FieldCondition(key=name, range=Range(gte=target_f - epsilon, lte=target_f + epsilon))
                mask = series.apply(lambda x, t=target_f, e=epsilon:
                    any(t - e <= float(v) <= t + e for v in x) if isinstance(x, list) and x else False)
                expr_str = f'{name} has element ~= {target_f}'
            else:
                filter_cond = Filter(must_not=[self._is_empty_condition(name)])
                mask = series.apply(lambda x: isinstance(x, list) and len(x) > 0)
                expr_str = f'{name} is not empty'

        elif ftype == FieldType.DATETIME:
            val_int = int(val)
            op = random.choice([">", "<", ">=", "<=", "==", "!="])
            if op == "==":
                filter_cond = FieldCondition(key=name, match=MatchValue(value=val_int))
            elif op == "!=":
                filter_cond = FieldCondition(key=name, match=MatchExcept(**{"except": [val_int]}))
            elif op == ">":
                filter_cond = FieldCondition(key=name, range=Range(gt=val_int))
            elif op == "<":
                filter_cond = FieldCondition(key=name, range=Range(lt=val_int))
            elif op == ">=":
                filter_cond = FieldCondition(key=name, range=Range(gte=val_int))
            elif op == "<=":
                filter_cond = FieldCondition(key=name, range=Range(lte=val_int))
            mask = series.apply(safe_compare_scalar(op, val_int))
            expr_str = f"{name} {op} {val_int} (timestamp)"

        elif ftype == FieldType.GEO:
            # GEO 使用 bounding box 查询
            if isinstance(val, dict) and "lat" in val and "lon" in val:
                center_lat = val["lat"]
                center_lon = val["lon"]
            else:
                center_lat = random.uniform(-80, 80)
                center_lon = random.uniform(-170, 170)
            delta_lat = random.uniform(1, 30)
            delta_lon = random.uniform(1, 30)
            top_lat = min(center_lat + delta_lat, 90)
            bottom_lat = max(center_lat - delta_lat, -90)
            left_lon = max(center_lon - delta_lon, -180)
            right_lon = min(center_lon + delta_lon, 180)
            filter_cond = FieldCondition(
                key=name,
                geo_bounding_box=GeoBoundingBox(
                    top_left=GeoPoint(lat=top_lat, lon=left_lon),
                    bottom_right=GeoPoint(lat=bottom_lat, lon=right_lon)
                )
            )
            mask = series.apply(lambda x: (
                x is not None and isinstance(x, dict)
                and "lat" in x and "lon" in x
                and bottom_lat <= x["lat"] <= top_lat
                and left_lon <= x["lon"] <= right_lon
            ))
            expr_str = f"{name} in bbox({bottom_lat:.2f},{left_lon:.2f} -> {top_lat:.2f},{right_lon:.2f})"

        if filter_cond is not None and mask is not None:
            return (filter_cond, mask & series.notnull(), expr_str)

        # 默认返回 - Qdrant 不支持 is_null=False，需要用 must_not 包装
        filter_cond = Filter(must_not=[self._is_null_condition(name)])
        return (filter_cond, series.notnull(), f"{name} is not null")

    def gen_json_expr(self, name, series, val):
        """生成 JSON 嵌套字段查询"""
        strategy = random.choice(["range", "nested", "index", "multi_key"])

        if strategy == "range":
            low = random.randint(100, 500)
            high = low + random.randint(50, 200)
            # Qdrant 使用点号访问嵌套字段
            filter_cond = FieldCondition(
                key=f"{name}.price",
                range=Range(gt=low, lt=high)
            )

            def check_range(x):
                try:
                    v = self._get_json_val(x, ["price"])
                    if v is None: return False
                    if isinstance(v, bool): return False
                    return (isinstance(v, (int, float)) and v > low and v < high)
                except: return False

            return (filter_cond, series.apply(check_range), f'{name}.price > {low} and < {high}')

        elif strategy == "nested":
            val = random.randint(1, 9)
            filter_cond = FieldCondition(
                key=f"{name}.config.version",
                match=MatchValue(value=val)
            )

            def check_nested(x):
                try:
                    v = self._get_json_val(x, ["config", "version"])
                    if v is None: return False
                    if isinstance(v, bool): return False
                    return v == val
                except: return False

            return (filter_cond, series.apply(check_nested), f'{name}.config.version == {val}')

        elif strategy == "index":
            idx = 0
            val = random.randint(20, 80)
            filter_cond = FieldCondition(
                key=f"{name}.history[{idx}]",
                range=Range(gt=val)
            )

            def check_index(x):
                try:
                    v = self._get_json_val(x, ["history", idx])
                    if v is None: return False
                    if isinstance(v, bool): return False
                    return (isinstance(v, (int, float)) and v > val)
                except: return False

            return (filter_cond, series.apply(check_index), f'{name}.history[{idx}] > {val}')

        else:  # multi_key
            color = random.choice(["Red", "Blue"])
            # Qdrant 的 must 条件组合
            filter_cond = models.Filter(
                must=[
                    FieldCondition(key=f"{name}.active", match=MatchValue(value=True)),
                    FieldCondition(key=f"{name}.color", match=MatchValue(value=color))
                ]
            )

            def check_multi(x):
                try:
                    active = self._get_json_val(x, ["active"])
                    col_val = self._get_json_val(x, ["color"])
                    if active is not True: return False
                    return col_val == color
                except: return False

            return (filter_cond, series.apply(check_multi), f'{name}.active == true and {name}.color == "{color}"')

    def gen_constant_expr(self):
        """生成常量表达式（Qdrant 支持有限）"""
        # Qdrant 中 id 是点 ID，不能用于 FieldCondition 过滤
        # 使用一个实际存在的 INT 字段来实现恒真/恒假条件
        
        # 找到一个 INT 类型的字段
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            field_name = int_fields[0]["name"]
            series = self.df[field_name]
            min_val = series.min()
            max_val = series.max()
        else:
            # 回退到使用 FLOAT 字段
            float_fields = [f for f in self.schema if f["type"] == FieldType.FLOAT]
            if float_fields:
                field_name = float_fields[0]["name"]
                series = self.df[field_name]
                min_val = series.min()
                max_val = series.max()
            else:
                return None, None, None
        
        # 注意：Qdrant 中 null 值不参与范围比较，所以"恒真"条件也要排除 null
        # true_mask 只对非 null 的行为 True
        true_mask = series.notna()
        false_mask = pd.Series(False, index=self.df.index)

        if random.random() < 0.5:
            # 必真：使用一个肯定包含所有非 null 数据的范围
            filter_cond = FieldCondition(key=field_name, range=Range(gte=min_val - 1000000, lte=max_val + 1000000))
            return (filter_cond, true_mask, f"{field_name} in wide range (always true)")
        else:
            # 必假：使用一个不可能的范围（大于最大值 + 大偏移）
            impossible_val = max_val + 10000000
            filter_cond = FieldCondition(key=field_name, range=Range(gt=impossible_val))
            return (filter_cond, false_mask, f"{field_name} > {impossible_val} (always false)")

    def gen_complex_expr(self, depth):
        """递归生成复杂表达式"""
        if depth == 0 or random.random() < 0.2:
            if random.random() < 0.02:
                res = self.gen_constant_expr()
                if res[0]:
                    # 确保返回 Filter 对象
                    filter_obj = res[0]
                    if isinstance(filter_obj, FieldCondition):
                        filter_obj = Filter(must=[filter_obj])
                    return filter_obj, res[1], res[2]

            if random.random() < 0.3:
                # 尝试生成 JSON 表达式
                json_fields = [f for f in self.schema if f["type"] == FieldType.JSON]
                if json_fields:
                    field = random.choice(json_fields)
                    name = field["name"]
                    series = self.df[name]
                    valid_series = series.dropna()
                    if not valid_series.empty:
                        val = random.choice(valid_series.values)
                        res = self.gen_json_expr(name, series, val)
                        if res[0]:
                            # 确保返回 Filter 对象
                            filter_obj = res[0]
                            if isinstance(filter_obj, FieldCondition):
                                filter_obj = Filter(must=[filter_obj])
                            return filter_obj, res[1], res[2]

            filter_cond, mask, expr_str = self.gen_atomic_expr()
            if filter_cond:
                # 确保返回 Filter 对象
                if isinstance(filter_cond, FieldCondition):
                    filter_cond = Filter(must=[filter_cond])
                return filter_cond, mask, expr_str
            return self.gen_complex_expr(depth)

        # 递归生成子节点
        filter_l, mask_l, expr_l = self.gen_complex_expr(depth - 1)
        if not filter_l: return self.gen_complex_expr(depth)

        op = random.choice(["and", "or", "not"])

        if op == "not":
            # NOT 逻辑：取反子表达式
            # Qdrant must_not: 排除匹配的点（null 值的点由于不匹配条件，会被保留）
            # Pandas ~mask: True -> False, False -> True（与 must_not 语义一致）
            not_filter = Filter(must_not=[filter_l])
            not_mask = ~mask_l.fillna(False)
            return not_filter, not_mask, f"NOT ({expr_l})"

        # AND/OR 需要第二个子表达式
        filter_r, mask_r, expr_r = self.gen_complex_expr(depth - 1)
        if not filter_r: return filter_l, mask_l, expr_l

        if op == "and":
            # AND 逻辑：将两个 Filter 作为整体放入 must 列表
            # 不能直接提取 must 字段，因为 Filter 可能是用 should/must_not 构建的
            combined_filter = Filter(must=[filter_l, filter_r])
            return combined_filter, (mask_l & mask_r), f"({expr_l} AND {expr_r})"
        else:
            # OR 逻辑：将两个 Filter 作为整体放入 should 列表
            combined_filter = Filter(should=[filter_l, filter_r])
            return combined_filter, (mask_l | mask_r), f"({expr_l} OR {expr_r})"


# --- 4. Equivalence Query Generator ---

class EquivalenceQueryGenerator(OracleQueryGenerator):
    """
    等价查询生成器：生成逻辑等价的查询变体，验证结果一致性
    """
    def __init__(self, dm):
        super().__init__(dm)
        self.field_types = {f["name"]: f["type"] for f in dm.schema_config}
        # 缓存用于恒真条件的字段
        self._tautology_field = self._find_tautology_field()

    def _find_tautology_field(self):
        """找到用于生成恒真条件的字段（INT 或 FLOAT）"""
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            name = int_fields[0]["name"]
            series = self.df[name]
            return {"name": name, "min": series.min(), "max": series.max(), "type": "int"}
        
        float_fields = [f for f in self.schema if f["type"] == FieldType.FLOAT]
        if float_fields:
            name = float_fields[0]["name"]
            series = self.df[name]
            return {"name": name, "min": series.min(), "max": series.max(), "type": "float"}
        
        return None

    def _gen_tautology_filter(self):
        """生成恒真条件（返回所有数据，包括 null 值）"""
        if self._tautology_field:
            name = self._tautology_field["name"]
            min_val = self._tautology_field["min"]
            max_val = self._tautology_field["max"]
            # 使用 (宽范围 OR 字段为null) 来确保包含所有数据（包括 null）
            # 因为 Qdrant 的 Range 条件不会匹配 null 值
            range_cond = Filter(must=[FieldCondition(key=name, range=Range(gte=min_val - 1000000, lte=max_val + 1000000))])
            null_cond = Filter(must=[IsNullCondition(is_null=PayloadField(key=name))])
            return (
                Filter(should=[range_cond, null_cond]),
                f"({name} in wide range OR {name} is null) (always true)"
            )
        # 如果没有数值字段，返回 None（不应该发生）
        return None, None

    def _gen_guaranteed_false_filter(self):
        """生成必然为假的过滤条件"""
        field = random.choice(self.schema)
        name = field["name"]
        dtype = field["type"]

        if dtype in [FieldType.ARRAY_INT, FieldType.ARRAY_STR, FieldType.ARRAY_FLOAT]:
            impossible_val = "qdrant_fuzz_impossible_" + self._random_string(10)
            return (
                FieldCondition(key=name, match=MatchValue(value=impossible_val)),
                f'{name} == "{impossible_val}"'
            )

        elif dtype == FieldType.STRING:
            complex_str = "fuzz_impossible_" + self._random_string(10, 20)
            return (
                FieldCondition(key=name, match=MatchValue(value=complex_str)),
                f'{name} == "{complex_str}"'
            )

        elif dtype in [FieldType.INT]:
            return (
                Filter(must=[
                    FieldCondition(key=name, range=Range(gt=200000)),
                    FieldCondition(key=name, range=Range(lt=-200000))
                ]),
                f"({name} > 200000 and {name} < -200000)"
            )

        elif dtype == FieldType.FLOAT:
            return (
                Filter(must=[
                    FieldCondition(key=name, range=Range(gt=1e20)),
                    FieldCondition(key=name, range=Range(lt=-1e20))
                ]),
                f"({name} > 1e20 and {name} < -1e20)"
            )

        elif dtype == FieldType.JSON:
            return (
                FieldCondition(
                    key=f"{name}.qdrant_fuzz_ghost_key_v3",
                    match=MatchValue(value="non_exist")
                ),
                f'{name}.qdrant_fuzz_ghost_key_v3 == "non_exist"'
            )

        elif dtype == FieldType.DATETIME:
            return (
                FieldCondition(key=name, range=Range(gt=9999999999)),
                f'{name} > 9999999999 (impossible timestamp)'
            )

        elif dtype == FieldType.GEO:
            # 使用极小的不可能区域
            return (
                FieldCondition(key=name, geo_bounding_box=GeoBoundingBox(
                    top_left=GeoPoint(lat=89.99999, lon=179.99998),
                    bottom_right=GeoPoint(lat=89.99998, lon=179.99999)
                )),
                f'{name} in impossible tiny bbox near pole'
            )

        # 默认：使用 INT 字段的不可能范围
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            int_name = int_fields[0]["name"]
            return (
                Filter(must=[
                    FieldCondition(key=int_name, range=Range(gt=200000)),
                    FieldCondition(key=int_name, range=Range(lt=-200000))
                ]),
                f"({int_name} > 200000 and {int_name} < -200000)"
            )
        
        # 回退到 FLOAT
        float_fields = [f for f in self.schema if f["type"] == FieldType.FLOAT]
        if float_fields:
            float_name = float_fields[0]["name"]
            return (
                Filter(must=[
                    FieldCondition(key=float_name, range=Range(gt=1e20)),
                    FieldCondition(key=float_name, range=Range(lt=-1e20))
                ]),
                f"({float_name} > 1e20 and {float_name} < -1e20)"
            )
        
        # 最后回退到 STRING
        str_fields = [f for f in self.schema if f["type"] == FieldType.STRING]
        if str_fields:
            str_name = str_fields[0]["name"]
            return (
                FieldCondition(key=str_name, match=MatchValue(value="__impossible_value_fuzz__")),
                f'{str_name} == "__impossible_value_fuzz__"'
            )

    def mutate_filter(self, base_filter, base_expr):
        """
        输入基础过滤器，返回逻辑等价的变体列表
        """
        mutations = []

        # 1. 双重否定 (Double Negation)
        # not (not A) => A
        mutations.append({
            "type": "DoubleNegation",
            "filter": Filter(must_not=[Filter(must_not=[base_filter] if isinstance(base_filter, FieldCondition) else [base_filter])]),
            "expr": f"not (not ({base_expr}))"
        })

        # 2. 恒真条件注入 (Tautology AND)
        # A AND True => A
        tautology_filter, tautology_expr = self._gen_tautology_filter()
        if tautology_filter:
            mutations.append({
                "type": "TautologyAnd",
                "filter": Filter(must=[base_filter, tautology_filter]),
                "expr": f"({base_expr}) AND ({tautology_expr})"
            })

        # 3. 数值字段切分 (Partitioning)
        # A => (A AND field < mid) OR (A AND field >= mid) OR (A AND field is null)
        # 使用 INT 字段进行切分，需要包含 null 值以保持等价性
        if self._tautology_field:
            part_name = self._tautology_field["name"]
            mid_val = (self._tautology_field["min"] + self._tautology_field["max"]) / 2
            part1 = Filter(must=[base_filter, FieldCondition(key=part_name, range=Range(lt=mid_val))])
            part2 = Filter(must=[base_filter, FieldCondition(key=part_name, range=Range(gte=mid_val))])
            part3 = Filter(must=[base_filter, IsNullCondition(is_null=PayloadField(key=part_name))])  # null 分区
            
            mutations.append({
                "type": "PartitionByField",
                "filter": Filter(should=[part1, part2, part3]),
                "expr": f"(({base_expr}) AND {part_name}<{mid_val}) OR (({base_expr}) AND {part_name}>={mid_val}) OR (({base_expr}) AND {part_name} is null)"
            })

        # 4. 冗余 OR (Self OR)
        # A OR A => A
        if isinstance(base_filter, Filter):
            mutations.append({
                "type": "SelfOr",
                "filter": Filter(should=[base_filter, base_filter]),
                "expr": f"({base_expr}) OR ({base_expr})"
            })
        else:
            mutations.append({
                "type": "SelfOr",
                "filter": Filter(should=[Filter(must=[base_filter]), Filter(must=[base_filter])]),
                "expr": f"({base_expr}) OR ({base_expr})"
            })

        # 5. 德·摩根包装
        # not ( (not A) OR False ) => A
        false_filter, false_expr = self._gen_guaranteed_false_filter()
        inner_not_a = Filter(must_not=[base_filter] if isinstance(base_filter, FieldCondition) else [base_filter])
        inner_or = Filter(should=[inner_not_a, Filter(must=[false_filter] if isinstance(false_filter, FieldCondition) else [false_filter])])
        mutations.append({
            "type": "DeMorganWrapper",
            "filter": Filter(must_not=[inner_or]),
            "expr": f"not ( (not ({base_expr})) OR ({false_expr}) )"
        })

        # 6. 噪声 OR 注入 (Empty Set Merge)
        # A OR False => A
        false_filter2, false_expr2 = self._gen_guaranteed_false_filter()
        if isinstance(base_filter, Filter):
            mutations.append({
                "type": "NoiseOr",
                "filter": Filter(should=[base_filter, Filter(must=[false_filter2] if isinstance(false_filter2, FieldCondition) else [false_filter2])]),
                "expr": f"({base_expr}) OR ({false_expr2})"
            })
        else:
            mutations.append({
                "type": "NoiseOr",
                "filter": Filter(should=[Filter(must=[base_filter]), Filter(must=[false_filter2] if isinstance(false_filter2, FieldCondition) else [false_filter2])]),
                "expr": f"({base_expr}) OR ({false_expr2})"
            })

        return mutations


# --- 5. PQS Query Generator ---

class PQSQueryGenerator(OracleQueryGenerator):
    """
    PQS 生成器：专注于生成"必须能查到指定行"的查询
    """
    def __init__(self, dm):
        super().__init__(dm)
        # 缓存用于恒真条件的字段
        self._tautology_field = self._find_tautology_field()

    def _find_tautology_field(self):
        """找到用于生成恒真条件的字段（INT 或 FLOAT）"""
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            name = int_fields[0]["name"]
            series = self.df[name]
            return {"name": name, "min": series.min(), "max": series.max(), "type": "int"}
        
        float_fields = [f for f in self.schema if f["type"] == FieldType.FLOAT]
        if float_fields:
            name = float_fields[0]["name"]
            series = self.df[name]
            return {"name": name, "min": series.min(), "max": series.max(), "type": "float"}
        
        return None

    def _gen_tautology_filter(self):
        """生成恒真条件（返回所有数据，包括 null 值）"""
        if self._tautology_field:
            name = self._tautology_field["name"]
            min_val = self._tautology_field["min"]
            max_val = self._tautology_field["max"]
            # 使用 (宽范围 OR 字段为null) 来确保包含所有数据（包括 null）
            # 因为 Qdrant 的 Range 条件不会匹配 null 值
            range_cond = Filter(must=[FieldCondition(key=name, range=Range(gte=min_val - 1000000, lte=max_val + 1000000))])
            null_cond = Filter(must=[IsNullCondition(is_null=PayloadField(key=name))])
            return (
                Filter(should=[range_cond, null_cond]),
                f"({name} in wide range OR {name} is null) (always true)"
            )
        # 如果没有数值字段，返回 None（不应该发生）
        return None, None

    def _gen_guaranteed_false_filter(self):
        """生成必然为假的过滤条件"""
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            name = int_fields[0]["name"]
            return (
                Filter(must=[
                    FieldCondition(key=name, range=Range(gt=200000)),
                    FieldCondition(key=name, range=Range(lt=-200000))
                ]),
                f"({name} > 200000 and {name} < -200000)"
            )
        
        float_fields = [f for f in self.schema if f["type"] == FieldType.FLOAT]
        if float_fields:
            name = float_fields[0]["name"]
            return (
                Filter(must=[
                    FieldCondition(key=name, range=Range(gt=1e20)),
                    FieldCondition(key=name, range=Range(lt=-1e20))
                ]),
                f"({name} > 1e20 and {name} < -1e20)"
            )
        
        return None, None

    def _has_valid_content(self, obj):
        """递归检查对象是否包含至少一个非 Null 的标量"""
        if obj is None:
            return False
        if isinstance(obj, (int, float, str, bool)):
            return True
        if hasattr(obj, "item"):
            return True
        if isinstance(obj, dict):
            return any(self._has_valid_content(v) for v in obj.values())
        if isinstance(obj, (list, np.ndarray)):
            return any(self._has_valid_content(v) for v in obj)
        return False

    def gen_pqs_filter(self, pivot_row, depth):
        """生成针对指定行必真的过滤条件"""
        force_recursion = depth > 3

        if depth <= 0 or (not force_recursion and random.random() < 0.3):
            return self.gen_true_atomic_filter(pivot_row)

        op = random.choice(["and", "or", "nested_not"])

        # 获取恒真/恒假条件
        tautology_filter, tautology_expr = self._gen_tautology_filter()
        if not tautology_filter:
            tautology_filter, tautology_expr = self._gen_fallback_tautology()
        false_filter, false_expr = self._gen_guaranteed_false_filter()

        if op == "nested_not":
            inner_filter, inner_expr = self.gen_pqs_filter(pivot_row, depth - 1)
            if not inner_filter:
                inner_filter = tautology_filter
                inner_expr = tautology_expr
            # not (not A) => A
            double_not = Filter(must_not=[Filter(must_not=[inner_filter] if isinstance(inner_filter, FieldCondition) else [inner_filter])])
            return double_not, f"not (not ({inner_expr}))"

        elif op == "and":
            filter_l, expr_l = self.gen_pqs_filter(pivot_row, depth - 1)
            filter_r, expr_r = self.gen_pqs_filter(pivot_row, depth - 1)
            if not filter_l:
                filter_l = tautology_filter
                expr_l = tautology_expr
            if not filter_r:
                filter_r = tautology_filter
                expr_r = tautology_expr
            
            return Filter(must=[filter_l, filter_r]), f"({expr_l} AND {expr_r})"

        else:  # OR
            filter_l, expr_l = self.gen_pqs_filter(pivot_row, depth - 1)
            noise_filter, noise_expr = self.gen_complex_noise(min(depth + random.randint(0, 3), 8))

            if not filter_l:
                filter_l = tautology_filter
                expr_l = tautology_expr
            if not noise_filter:
                noise_filter = false_filter
                noise_expr = false_expr

            should_list = []
            if isinstance(filter_l, Filter):
                should_list.append(filter_l)
            else:
                should_list.append(Filter(must=[filter_l]))
            if isinstance(noise_filter, Filter):
                should_list.append(noise_filter)
            else:
                should_list.append(Filter(must=[noise_filter]))

            if random.random() < 0.5:
                return Filter(should=should_list), f"({expr_l} OR {noise_expr})"
            else:
                return Filter(should=should_list[::-1]), f"({noise_expr} OR {expr_l})"

    def _gen_fallback_tautology(self):
        """生成一个回退的恒真条件（包含 null 值）"""
        float_fields = [f for f in self.schema if f["type"] == FieldType.FLOAT]
        if float_fields:
            name = float_fields[0]["name"]
            series = self.df[name]
            # 使用 (宽范围 OR null) 确保包含所有数据
            range_cond = Filter(must=[FieldCondition(key=name, range=Range(gte=series.min() - 1e10, lte=series.max() + 1e10))])
            null_cond = Filter(must=[IsNullCondition(is_null=PayloadField(key=name))])
            return (
                Filter(should=[range_cond, null_cond]),
                f"({name} in wide range OR {name} is null)"
            )
        return None, None

    def gen_true_atomic_filter(self, row):
        """针对单行数据，生成必真的原子条件"""
        field = random.choice(self.schema)
        fname = field["name"]
        ftype = field["type"]
        val = row[fname]

        # 处理 Null
        is_null = False
        try:
            if val is None: is_null = True
            if isinstance(val, float) and np.isnan(val): is_null = True
        except: pass

        if is_null:
            return self._null_filter(fname), f"{fname} is null"

        # 根据类型生成条件
        if ftype == FieldType.BOOL:
            return FieldCondition(key=fname, match=MatchValue(value=bool(val))), f"{fname} == {bool(val)}"

        elif ftype == FieldType.INT:
            val_int = int(val) if hasattr(val, "item") else int(val)
            strategies = []
            strategies.append((
                FieldCondition(key=fname, match=MatchValue(value=val_int)),
                f"{fname} == {val_int}"
            ))
            strategies.append((
                FieldCondition(key=fname, range=Range(gte=val_int, lte=val_int)),
                f"{fname} >= {val_int} AND <= {val_int}"
            ))
            strategies.append((
                FieldCondition(key=fname, range=Range(gt=val_int-1, lt=val_int+1)),
                f"{fname} > {val_int-1} AND < {val_int+1}"
            ))
            return random.choice(strategies)

        elif ftype == FieldType.FLOAT:
            val_float = float(val)
            epsilon = 1e-5
            strategies = []
            strategies.append((
                FieldCondition(key=fname, range=Range(gt=val_float - epsilon, lt=val_float + epsilon)),
                f"{fname} > {val_float - epsilon} AND < {val_float + epsilon}"
            ))
            strategies.append((
                FieldCondition(key=fname, range=Range(gte=val_float - epsilon)),
                f"{fname} >= {val_float - epsilon}"
            ))
            return random.choice(strategies)

        elif ftype == FieldType.STRING:
            val_str = str(val)
            strategies = []
            strategies.append((
                FieldCondition(key=fname, match=MatchValue(value=val_str)),
                f'{fname} == "{val_str}"'
            ))
            # in 列表包含
            dummies = [self._random_string(5) for _ in range(3)]
            candidates = dummies + [val_str]
            random.shuffle(candidates)
            strategies.append((
                FieldCondition(key=fname, match=MatchAny(any=candidates)),
                f'{fname} in {candidates}'
            ))
            return random.choice(strategies)

        elif ftype == FieldType.JSON:
            return self._gen_pqs_json_filter(fname, val)

        elif ftype in [FieldType.ARRAY_INT, FieldType.ARRAY_STR]:
            if not isinstance(val, list) or len(val) == 0:
                return self._empty_or_null_filter(fname), f"{fname} is empty"
            
            valid_items = [x for x in val if x is not None]
            if valid_items:
                target = random.choice(valid_items)
                # 确保 target 是原生 Python 类型
                target = self._convert_to_native(target)
                return FieldCondition(key=fname, match=MatchAny(any=[target])), f"{fname} contains {target}"
            # Qdrant 不支持 is_null=False，需要用 must_not 包装
            return Filter(must_not=[self._is_null_condition(fname)]), f"{fname} is not null"

        elif ftype == FieldType.ARRAY_FLOAT:
            if not isinstance(val, list) or len(val) == 0:
                return self._empty_or_null_filter(fname), f"{fname} is empty"
            valid_items = [x for x in val if x is not None]
            if valid_items:
                target = float(random.choice(valid_items))
                epsilon = 0.5
                return FieldCondition(key=fname, range=Range(gte=target - epsilon, lte=target + epsilon)), f"{fname} has element ~= {target}"
            return Filter(must_not=[self._is_null_condition(fname)]), f"{fname} is not null"

        elif ftype == FieldType.DATETIME:
            val_int = int(val) if hasattr(val, "item") else int(val)
            # 使用包含该精确值的范围
            return FieldCondition(key=fname, range=Range(gte=val_int, lte=val_int)), f'{fname} timestamp == {val_int}'

        elif ftype == FieldType.GEO:
            if isinstance(val, dict) and "lat" in val and "lon" in val:
                lat, lon = val["lat"], val["lon"]
                delta = 0.001  # 极小范围确保命中
                return FieldCondition(
                    key=fname,
                    geo_bounding_box=GeoBoundingBox(
                        top_left=GeoPoint(lat=min(lat + delta, 90), lon=max(lon - delta, -180)),
                        bottom_right=GeoPoint(lat=max(lat - delta, -90), lon=min(lon + delta, 180))
                    )
                ), f"{fname} in tiny bbox around ({lat},{lon})"
            return Filter(must_not=[self._is_null_condition(fname)]), f"{fname} is not null"

        return Filter(must_not=[self._is_null_condition(fname)]), f"{fname} is not null"

    def _gen_pqs_json_filter(self, fname, json_obj):
        """生成针对 JSON 字段的必真条件"""
        if json_obj is None:
            return self._null_filter(fname), f"{fname} is null"
        if not isinstance(json_obj, dict):
            return Filter(must_not=[self._is_null_condition(fname)]), f"{fname} is not null"

        # 尝试下钻到标量值
        strategies = []

        # 策略 1: price 字段
        if "price" in json_obj:
            price = json_obj["price"]
            if isinstance(price, (int, float)):
                strategies.append((
                    FieldCondition(key=f"{fname}.price", match=MatchValue(value=int(price))),
                    f"{fname}.price == {int(price)}"
                ))

        # 策略 2: color 字段
        if "color" in json_obj:
            color = json_obj["color"]
            if isinstance(color, str):
                strategies.append((
                    FieldCondition(key=f"{fname}.color", match=MatchValue(value=color)),
                    f'{fname}.color == "{color}"'
                ))

        # 策略 3: active 字段
        if "active" in json_obj:
            active = json_obj["active"]
            if isinstance(active, bool):
                strategies.append((
                    FieldCondition(key=f"{fname}.active", match=MatchValue(value=active)),
                    f"{fname}.active == {active}"
                ))

        # 策略 4: nested config.version
        if "config" in json_obj and isinstance(json_obj["config"], dict):
            if "version" in json_obj["config"]:
                version = json_obj["config"]["version"]
                if isinstance(version, int):
                    strategies.append((
                        FieldCondition(key=f"{fname}.config.version", match=MatchValue(value=version)),
                        f"{fname}.config.version == {version}"
                    ))

        if strategies:
            return random.choice(strategies)

        # Qdrant 不支持 is_null=False，需要用 must_not 包装
        return Filter(must_not=[self._is_null_condition(fname)]), f"{fname} is not null"

    def gen_complex_noise(self, depth):
        """生成复杂的噪声表达式"""
        if depth <= 0 or random.random() < 0.3:
            if random.random() < 0.8:
                res = self.gen_atomic_expr()
                if res and res[0]:
                    return res[0], res[2]
                tautology_filter, tautology_expr = self._gen_tautology_filter()
                if tautology_filter:
                    return tautology_filter, tautology_expr
                return self._gen_fallback_tautology()
            tautology_filter, tautology_expr = self._gen_tautology_filter()
            if tautology_filter:
                return tautology_filter, tautology_expr
            return self._gen_fallback_tautology()

        op_type = random.choice(["and", "or", "not"])

        filter_1, expr_1 = self.gen_complex_noise(depth - 1)
        if not filter_1:
            tautology_filter, tautology_expr = self._gen_tautology_filter()
            if tautology_filter:
                filter_1, expr_1 = tautology_filter, tautology_expr
            else:
                filter_1, expr_1 = self._gen_fallback_tautology()

        if op_type == "not":
            return Filter(must_not=[filter_1] if isinstance(filter_1, FieldCondition) else [filter_1]), f"not ({expr_1})"

        elif op_type in ["and", "or"]:
            filter_2, expr_2 = self.gen_complex_noise(depth - 1)
            if not filter_2:
                tautology_filter, tautology_expr = self._gen_tautology_filter()
                if tautology_filter:
                    filter_2, expr_2 = tautology_filter, tautology_expr
                else:
                    filter_2, expr_2 = self._gen_fallback_tautology()

            if op_type == "and":
                return Filter(must=[filter_1, filter_2]), f"({expr_1} AND {expr_2})"
            else:
                should_list = []
                if isinstance(filter_1, Filter):
                    should_list.append(filter_1)
                else:
                    should_list.append(Filter(must=[filter_1]))
                if isinstance(filter_2, Filter):
                    should_list.append(filter_2)
                else:
                    should_list.append(Filter(must=[filter_2]))
                return Filter(should=should_list), f"({expr_1} OR {expr_2})"

        tautology_filter, tautology_expr = self._gen_tautology_filter()
        if tautology_filter:
            return tautology_filter, tautology_expr
        return self._gen_fallback_tautology()


# --- 6. Dynamic Data Operations Helper ---

def _do_dynamic_op(dm, qm, file_log, delete_min_rows=100):
    """
    执行一次随机动态数据操作（insert/delete/upsert）。
    同时同步 dm.df 和 dm.vectors。
    """
    op = random.choices(["insert", "delete", "upsert"], weights=[0.4, 0.4, 0.2], k=1)[0]
    batch_count = random.randint(1, 5)

    if op == "insert":
        new_rows = []
        new_vecs = []
        for _ in range(batch_count):
            row = dm.generate_single_row()
            vec = dm.generate_single_vector()
            new_rows.append(row)
            new_vecs.append(vec)
        try:
            points = []
            for row, vec in zip(new_rows, new_vecs):
                payload = {k: v for k, v in row.items() if k != "id"}
                points.append(PointStruct(id=row["id"], vector=vec, payload=payload))
            qm.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
            dm.df = pd.concat([dm.df, pd.DataFrame(new_rows)], ignore_index=True)
            dm.vectors = np.vstack([dm.vectors, np.array(new_vecs)])
            inserted_ids = [r["id"] for r in new_rows]
            file_log(f"[Dynamic] Inserted {len(new_rows)} rows: ids={inserted_ids}")
        except Exception as e:
            file_log(f"[Dynamic] Insert failed: {e}")

    elif op == "delete":
        if len(dm.df) > delete_min_rows:
            del_count = min(batch_count, len(dm.df) - delete_min_rows)
            del_ids = random.sample(dm.df["id"].tolist(), del_count)
            try:
                qm.client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=del_ids),
                    wait=True
                )
                idx = dm.df[dm.df["id"].isin(del_ids)].index.to_numpy()
                dm.df = dm.df.drop(idx).reset_index(drop=True)
                dm.vectors = np.delete(dm.vectors, idx, axis=0)
                file_log(f"[Dynamic] Deleted {len(del_ids)} rows: ids={del_ids}")
            except Exception as e:
                file_log(f"[Dynamic] Delete failed: {e}")

    else:  # upsert
        upsert_rows = []
        upsert_vecs = []
        for _ in range(batch_count):
            use_existing = (not dm.df.empty) and (random.random() < 0.7)
            if use_existing:
                target_id = random.choice(dm.df["id"].tolist())
            else:
                target_id = generate_unique_id()
            row = dm.generate_single_row(id_override=target_id)
            vec = dm.generate_single_vector()
            upsert_rows.append(row)
            upsert_vecs.append(vec)
        try:
            points = []
            for row, vec in zip(upsert_rows, upsert_vecs):
                payload = {k: v for k, v in row.items() if k != "id"}
                points.append(PointStruct(id=row["id"], vector=vec, payload=payload))
            qm.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
            new_rows = []
            new_vectors = []
            for row, vec in zip(upsert_rows, upsert_vecs):
                rid = row["id"]
                match_idx = dm.df.index[dm.df["id"] == rid].tolist()
                if match_idx:
                    idx = match_idx[0]
                    for k, v in row.items():
                        dm.df.at[idx, k] = v
                    dm.vectors[idx] = vec
                else:
                    new_rows.append(row)
                    new_vectors.append(vec)
            if new_rows:
                dm.df = pd.concat([dm.df, pd.DataFrame(new_rows)], ignore_index=True)
                dm.vectors = np.vstack([dm.vectors, np.array(new_vectors)])
            upsert_ids = [r["id"] for r in upsert_rows]
            file_log(f"[Dynamic] Upserted {len(upsert_rows)} rows: ids={upsert_ids}")
        except Exception as e:
            file_log(f"[Dynamic] Upsert failed: {e}")


# --- 7. Main Execution Functions ---

def run(rounds=100, seed=None, enable_dynamic_ops=False):
    """
    Oracle 模式主测试：Qdrant vs Pandas 对比
    """
    global DISTANCE_TYPE
    DISTANCE_TYPE = None  # 重置，随机选取
    current_seed = seed
    if current_seed is None:
        current_seed = random.randint(0, 2**31 - 1)
        random.seed(current_seed)
        np.random.seed(current_seed)
        print(f"🎲 随机生成种子: {current_seed}")
    else:
        print(f"🔒 使用固定种子 {current_seed} - 可复现的数据")
        random.seed(seed)
        np.random.seed(seed)

    # 1. 初始化
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()

    qm = QdrantManager()
    qm.connect()
    qm.reset_collection(dm.schema_config)
    qm.insert(dm)

    # 2. 日志设置
    timestamp = int(time.time())
    log_filename = f"qdrant_fuzz_test_{timestamp}.log"
    print(f"\n📝 详细日志将写入: {log_filename}")
    print(f"   🔑 如需复现此次测试，运行: python qdrant_fuzz_oracle.py --seed {current_seed}")
    print(f"🚀 开始测试 (控制台仅显示失败案例)...")

    qg = OracleQueryGenerator(dm)
    failed_cases = []
    total_test = rounds

    with open(log_filename, "w", encoding="utf-8") as f:
        def file_log(msg):
            f.write(msg + "\n")
            f.flush()

        def sample_rows(id_set, limit=5):
            if not id_set:
                return []
            subset = dm.df[dm.df["id"].isin(list(id_set))]
            rows = subset.to_dict(orient="records")
            return rows[:limit]

        file_log(f"Start Testing: {total_test} rounds | Seed: {current_seed}")
        file_log("=" * 50)

        for i in range(total_test):
            print(f"\r⏳ Running Test {i+1}/{total_test}...", end="", flush=True)

            # --- 动态插入/删除/Upsert ---
            if enable_dynamic_ops and i > 0 and i % 10 == 0:
                _do_dynamic_op(dm, qm, file_log)

            # 生成查询
            depth = random.randint(1, 10)
            filter_obj = None
            for _ in range(10):
                filter_obj, pandas_mask, expr_str = qg.gen_complex_expr(depth)
                if filter_obj: break

            if not filter_obj: continue

            log_header = f"[Test {i}]"
            file_log(f"\n{log_header} Expr: {expr_str}")

            # Pandas 计算
            expected_ids = set(dm.df[pandas_mask.fillna(False)]["id"].values.tolist())

            try:
                start_t = time.time()

                # Qdrant 查询
                scroll_limit = len(dm.df) + 1000  # 动态数据量自适应
                scroll_result = qm.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=filter_obj,
                    limit=scroll_limit,
                    with_payload=False,
                    with_vectors=False
                )

                actual_ids = set()
                for point in scroll_result[0]:
                    actual_ids.add(point.id)

                cost = (time.time() - start_t) * 1000

                file_log(f"  Pandas: {len(expected_ids)} | Qdrant: {len(actual_ids)} | Time: {cost:.1f}ms")

                if expected_ids == actual_ids:
                    file_log("  -> MATCH")
                else:
                    print(f"\n❌ [Test {i}] MISMATCH!")
                    print(f"   Expr: {expr_str}")
                    print(f"   Expected: {len(expected_ids)} vs Actual: {len(actual_ids)}")

                    missing = expected_ids - actual_ids
                    extra = actual_ids - expected_ids
                    diff_msg = ""
                    if missing: diff_msg += f"Missing IDs: {list(missing)[:10]} "
                    if extra: diff_msg += f"Extra IDs: {list(extra)[:10]}"

                    print(f"   Diff: {diff_msg}")
                    print(f"   🔑 复现此bug: python qdrant_fuzz_oracle.py --seed {current_seed}\n")

                    file_log(f"  -> MISMATCH! {diff_msg}")
                    file_log(f"  -> REPRODUCTION SEED: {current_seed}")

                    if missing:
                        missing_rows = sample_rows(missing)
                        file_log(f"  Missing rows sample: {missing_rows}")
                        print("   Missing rows (sample):")
                        for r in missing_rows[:3]:
                            print(f"     {r}")
                    if extra:
                        extra_rows = sample_rows(extra)
                        file_log(f"  Extra rows sample: {extra_rows}")
                        print("   Extra rows (sample):")
                        for r in extra_rows[:3]:
                            print(f"     {r}")

                    failed_cases.append({
                        "id": i,
                        "expr": expr_str,
                        "detail": f"Exp: {len(expected_ids)} vs Act: {len(actual_ids)}. {diff_msg}",
                        "seed": current_seed
                    })

                # 向量 + 标量联合校验
                if expected_ids and random.random() < VECTOR_CHECK_RATIO:
                    try:
                        q_idx = random.randint(0, len(dm.vectors) - 1)
                        q_vec = dm.vectors[q_idx].tolist()
                        search_k = min(VECTOR_TOPK, len(dm.df))

                        search_res = qm.client.search(
                            collection_name=COLLECTION_NAME,
                            query_vector=q_vec,
                            query_filter=filter_obj,
                            limit=search_k,
                            with_payload=False
                        )

                        returned_ids = set(p.id for p in search_res)

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

            except Exception as e:
                print(f"\n⚠️ [Test {i}] CRASHED!")
                print(f"   Expr: {expr_str}")
                print(f"   Error: {e}\n")
                file_log(f"  -> ERROR: {e}")

                failed_cases.append({
                    "id": i,
                    "expr": expr_str,
                    "detail": f"Exception: {e}"
                })

    print("\n" + "="*60)
    if not failed_cases:
        print(f"✅ 所有 {total_test} 轮测试全部通过！")
        print(f"📄 详细记录请查看: {log_filename}")
    else:
        print(f"🚫 发现 {len(failed_cases)} 个失败案例！(已保存至日志)")
        print("-" * 60)
        for case in failed_cases:
            print(f"🔴 Case {case['id']}:")
            print(f"   Expr: {case['expr']}")
            print(f"   Issue: {case['detail']}")
            if 'seed' in case:
                print(f"   🔑 复现: python qdrant_fuzz_oracle.py --seed {case['seed']}")
            print("-" * 30)
        print(f"📄 请查看 {log_filename} 获取完整上下文。")
        print(f"🔑 全局复现命令: python qdrant_fuzz_oracle.py --seed {current_seed}")


def run_equivalence_mode(rounds=100, seed=None, enable_dynamic_ops=False):
    """
    等价性测试模式
    """
    global DISTANCE_TYPE
    DISTANCE_TYPE = None  # 重置
    if seed is not None:
        print(f"\n🔒 Equivalence 模式使用固定种子 {seed}")
        random.seed(seed)
        np.random.seed(seed)
    else:
        seed = random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)

    timestamp = int(time.time())
    log_filename = f"qdrant_equiv_test_{timestamp}.log"
    print("\n" + "="*60)
    print(f"👯 启动 Equivalence Mode (等价性测试)")
    print(f"   原理: Query(A) 应该等于 Query(Transformation(A))")
    print(f"   Seed: {seed}")
    print(f"📄 日志: {log_filename}")
    print("="*60)

    # 初始化
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()

    qm = QdrantManager()
    qm.connect()
    qm.reset_collection(dm.schema_config)
    qm.insert(dm)

    qg = EquivalenceQueryGenerator(dm)
    failed_cases = []

    with open(log_filename, "w", encoding="utf-8") as f:
        def file_log(msg):
            f.write(msg + "\n")
            f.flush()

        file_log(f"Equivalence Test Started | Seed: {seed}")

        for i in range(rounds):
            print(f"\r⚖️  Test {i+1}/{rounds}...", end="", flush=True)

            # --- 动态操作（20% 概率）---
            if enable_dynamic_ops and i > 0 and random.random() < 0.2:
                _do_dynamic_op(dm, qm, file_log)

            # 生成基础查询
            base_filter = None
            for _ in range(10):
                base_filter, _, base_expr = qg.gen_complex_expr(depth=random.randint(1, 12))
                if base_filter: break
            
            if not base_filter: continue

            # 生成变体
            mutations = qg.mutate_filter(base_filter, base_expr)
            if not mutations: continue

            # 执行基础查询
            try:
                scroll_limit = len(dm.df) + 1000
                base_res = qm.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=base_filter,
                    limit=scroll_limit,
                    with_payload=False
                )
                base_ids = set(p.id for p in base_res[0])
            except Exception as e:
                file_log(f"[Test {i}] Base Query Failed: {e}")
                continue

            log_header = f"[Test {i}] Base: {base_expr} (Hits: {len(base_ids)})"
            file_log(f"\n{log_header}")

            # 执行并对比所有变体
            for m in mutations:
                m_type = m["type"]
                m_filter = m["filter"]
                m_expr = m["expr"]

                try:
                    mut_res = qm.client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=m_filter,
                        limit=scroll_limit,
                        with_payload=False
                    )
                    mut_ids = set(p.id for p in mut_res[0])

                    if base_ids == mut_ids:
                        file_log(f"  ✅ [{m_type}] Match")
                    else:
                        print(f"\n\n❌ EQUIVALENCE FAILURE [Test {i}]")
                        print(f"   Type: {m_type}")
                        print(f"   Base Expr: {base_expr}")
                        print(f"   Mut  Expr: {m_expr}")
                        print(f"   Base Hits: {len(base_ids)} | Mut Hits: {len(mut_ids)}")

                        missing = base_ids - mut_ids
                        extra = mut_ids - base_ids
                        diff_msg = ""
                        if missing: diff_msg += f"Mutation Missing: {list(missing)[:5]} "
                        if extra: diff_msg += f"Mutation Extra: {list(extra)[:5]} "
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


def run_pqs_mode(rounds=100, seed=None, enable_dynamic_ops=False):
    """
    PQS 模式测试：验证生成的查询必须命中指定行
    """
    global DISTANCE_TYPE
    DISTANCE_TYPE = None  # 重置
    if seed is not None:
        print(f"\n🔒 PQS模式使用固定种子 {seed}")
        random.seed(seed)
        np.random.seed(seed)

    timestamp = int(time.time())
    log_filename = f"qdrant_pqs_test_{timestamp}.log"
    print("\n" + "="*60)
    print(f"🚀 启动 PQS (Pivot Query Synthesis) 模式测试")
    print(f"📄 详细日志将写入: {log_filename}")
    print("="*60)

    # 初始化
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()

    qm = QdrantManager()
    qm.connect()
    qm.reset_collection(dm.schema_config)
    qm.insert(dm)

    pqs_gen = PQSQueryGenerator(dm)
    errors = []
    successful_tests = 0
    skipped_tests = 0

    def safe_format_row(row_series):
        row_dict = row_series.to_dict()
        safe_data = {}
        for k, v in row_dict.items():
            if k == "vector": continue
            if hasattr(v, "item"):
                v = v.item()
            if v is None or (isinstance(v, float) and np.isnan(v)):
                safe_data[k] = None
                continue
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
            # --- 动态操作（20% 概率）---
            if enable_dynamic_ops and i > 0 and random.random() < 0.2:
                _do_dynamic_op(dm, qm, file_log)

            random_idx = random.randint(0, len(dm.df) - 1)
            pivot_row = dm.df.iloc[random_idx]
            pivot_id = int(pivot_row["id"])

            filter_obj = None
            expr = ""
            # 增加重试次数到 20 次
            for retry in range(20):
                try:
                    filter_obj, expr = pqs_gen.gen_pqs_filter(pivot_row, depth=random.randint(5, 13))
                    if filter_obj: break
                except Exception as e:
                    if retry == 19:  # 最后一次重试时记录错误
                        file_log(f"[Round {i}] Filter generation failed after 20 retries: {e}")

            if not filter_obj:
                skipped_tests += 1
                print(f"\r⏭️  [Round {i+1}/{rounds}] Skipped (filter gen failed)...", end="", flush=True)
                file_log(f"[Round {i}] SKIPPED - Could not generate valid filter for ID {pivot_id}")
                continue

            print(f"\r🔍 [Round {i+1}/{rounds}] Check ID: {pivot_id}...", end="", flush=True)

            log_header = f"[Round {i}] Target ID: {pivot_id}"
            file_log(f"\n{log_header}")
            file_log(f"  Expr: {expr}")

            try:
                start_t = time.time()
                scroll_limit = len(dm.df) + 1000
                res = qm.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=filter_obj,
                    limit=scroll_limit,
                    with_payload=False
                )
                cost = (time.time() - start_t) * 1000
                found_ids = set(p.id for p in res[0])

                successful_tests += 1
                if pivot_id in found_ids:
                    file_log(f"  -> PASS | Found: {len(found_ids)} hits | Time: {cost:.2f}ms")
                else:
                    safe_row = safe_format_row(pivot_row)
                    json_data = safe_row.get("meta_json", {})
                    scalar_data = {k: v for k, v in safe_row.items() if k != "meta_json"}

                    print(f"\n\n❌ PQS ERROR DETECTED [Round {i}]")
                    print(f"   Target ID: {pivot_id}")
                    print(f"   Expression: {expr}")
                    print(f"   Found Count: {len(found_ids)} (Target NOT found)")
                    print("-" * 50)
                    print(f"   🔎 EVIDENCE (Target Row Data):")
                    print(f"   [meta_json]:")
                    print(json.dumps(json_data, indent=4, ensure_ascii=False))
                    print(f"   [Scalars]:")
                    for k, v in scalar_data.items():
                        print(f"     {k:<15}: {v}")
                    print("-" * 50)

                    file_log(f"  -> ❌ FAIL! Target ID {pivot_id} NOT found.")
                    file_log(f"  -> Row Data: {safe_row}")

                    errors.append({"id": pivot_id, "expr": expr})

            except Exception as e:
                print(f"\n\n⚠️ Execution Error [Round {i}]: {e}")
                file_log(f"  -> EXECUTION ERROR: {e}")

    print("\n" + "="*60)
    print(f"📊 测试统计:")
    print(f"   总轮数: {rounds}")
    print(f"   成功执行: {successful_tests}")
    print(f"   跳过: {skipped_tests}")
    print(f"   发现错误: {len(errors)}")
    print("="*60)
    if not errors:
        if successful_tests > 0:
            print(f"✅ PQS 测试完成。{successful_tests} 个测试全部通过！")
        else:
            print(f"⚠️  警告：没有成功执行任何测试（全部跳过）")
    else:
        print(f"🚫 PQS 测试完成。发现 {len(errors)} 个潜在 Bug！")
        print(f"📄 详细数据已记录至日志: {log_filename}")


def run_group_test(rounds=50, seed=None, enable_dynamic_ops=False):
    """
    Group By 测试模式 (Qdrant 使用 group_by 参数)
    """
    global DISTANCE_TYPE
    DISTANCE_TYPE = None  # 重置
    if seed is not None:
        print(f"\n🔒 GroupBy 模式使用固定种子 {seed}")
        random.seed(seed)
        np.random.seed(seed)

    timestamp = int(time.time())
    log_filename = f"qdrant_group_test_{timestamp}.log"
    print("\n" + "="*60)
    print(f"📊 启动 GroupBy 逻辑专项测试")
    print(f"   日志: {log_filename}")
    print("="*60)

    # 初始化
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()

    qm = QdrantManager()
    qm.connect()
    qm.reset_collection(dm.schema_config)
    qm.insert(dm)

    # 识别可用于分组的字段
    potential_group_fields = []
    for f in dm.schema_config:
        if f["type"] in [FieldType.INT, FieldType.STRING, FieldType.BOOL]:
            potential_group_fields.append(f["name"])

    # JSON 嵌套字段
    json_fields = [f["name"] for f in dm.schema_config if f["type"] == FieldType.JSON]
    for jf in json_fields:
        potential_group_fields.append(f'{jf}.color')
        potential_group_fields.append(f'{jf}.active')
        potential_group_fields.append(f'{jf}.price')

    errors = []

    with open(log_filename, "w", encoding="utf-8") as f:
        def file_log(msg):
            f.write(msg + "\n")
            f.flush()

        file_log(f"GroupBy Test Started | Rounds: {rounds}")

        for i in range(rounds):
            print(f"\r📊 GroupBy Test {i+1}/{rounds}...", end="", flush=True)

            # --- 动态操作（20% 概率）---
            if enable_dynamic_ops and i > 0 and random.random() < 0.2:
                _do_dynamic_op(dm, qm, file_log)

            group_field = random.choice(potential_group_fields)
            group_size = random.randint(1, 5)
            limit_groups = random.randint(2, 20)

            q_idx = random.randint(0, len(dm.vectors) - 1)
            q_vec = dm.vectors[q_idx].tolist()

            try:
                # Qdrant 的 group_by 搜索 - 使用 search_groups
                search_res = qm.client.search_groups(
                    collection_name=COLLECTION_NAME,
                    query_vector=q_vec,
                    group_by=group_field,
                    limit=limit_groups,
                    group_size=group_size,
                    with_payload=False
                )

                # 验证返回的组数
                num_groups = len(search_res.groups)
                
                if num_groups > limit_groups:
                    msg = f"❌ Limit Exceeded: Asked for {limit_groups} groups, got {num_groups}"
                    print(f"\n{msg}")
                    print(f"   ⚠️ Field: {group_field}")
                    file_log(f"[Round {i}] {msg}")
                    errors.append(msg)

                # 验证每组大小
                for group in search_res.groups:
                    if len(group.hits) > group_size:
                        msg = f"❌ Group Size Violation: Group has {len(group.hits)} items, expected max {group_size}"
                        print(f"\n{msg}")
                        file_log(f"[Round {i}] {msg}")
                        errors.append(msg)

                file_log(f"[Round {i}] Field: {group_field} | Groups: {num_groups} | GroupSize: {group_size} -> PASS")

            except Exception as e:
                file_log(f"[Round {i}] Search Failed: {e}")
                continue

    print("\n" + "="*60)
    if not errors:
        print(f"✅ GroupBy 测试完成。未发现显式逻辑违规。")
    else:
        print(f"🚫 GroupBy 测试发现 {len(errors)} 个问题！请查看日志。")


# --- Main Entry Point ---

if __name__ == "__main__":
    seed = None
    rounds = 1000
    pqs_rounds = 1000
    mode = "oracle"  # 默认模式
    enable_dynamic = False  # 动态数据操作开关

    # 解析命令行参数
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--seed" and i + 1 < len(args):
                seed = int(args[i + 1])
                i += 2
            elif arg == "--rounds" and i + 1 < len(args):
                rounds = int(args[i + 1])
                i += 2
            elif arg == "--pqs-rounds" and i + 1 < len(args):
                pqs_rounds = int(args[i + 1])
                i += 2
            elif arg == "--equiv":
                mode = "equiv"
                i += 1
            elif arg == "--pqs":
                mode = "pqs"
                i += 1
            elif arg == "--group":
                mode = "group"
                i += 1
            elif arg == "--dynamic":
                enable_dynamic = True
                i += 1
            elif arg == "--chaos":
                CHAOS_RATE = 0.1
                i += 1
            elif arg == "--chaos-rate" and i + 1 < len(args):
                try:
                    CHAOS_RATE = float(args[i + 1])
                except:
                    pass
                i += 2
            elif arg == "--help":
                print("""
Qdrant Fuzz Oracle - 动态模糊测试工具

用法:
  python qdrant_fuzz_oracle.py [选项]

选项:
  --seed <数字>       使用固定种子以复现测试
  --rounds <数字>     主测试轮数 (默认: 1000)
  --pqs-rounds <数字> PQS 测试轮数 (默认: 1000)
  --equiv             运行等价性测试模式
  --pqs               运行 PQS (必中查询) 测试模式
  --group             运行 GroupBy 测试模式
  --dynamic           开启动态数据操作 (insert/delete/upsert)
  --chaos             开启混淆模式 (10%)
  --chaos-rate <比率> 自定义混淆概率 (0.0-1.0)
  --help              显示此帮助信息

示例:
  python qdrant_fuzz_oracle.py                    # 默认 Oracle 模式
  python qdrant_fuzz_oracle.py --seed 12345       # 使用种子复现
  python qdrant_fuzz_oracle.py --equiv --rounds 500
  python qdrant_fuzz_oracle.py --pqs --pqs-rounds 200
  python qdrant_fuzz_oracle.py --dynamic --rounds 300  # 动态操作模式
""")
                sys.exit(0)
            else:
                i += 1

    print("=" * 80)
    print(f"🚀 Qdrant Fuzz Oracle 启动")
    print(f"   模式: {mode}")
    print(f"   主测试轮数: {rounds}")
    print(f"   PQS测试轮数: {pqs_rounds}")
    print(f"   随机种子: {seed if seed else '(随机)'}")
    print(f"   动态操作: {'开启' if enable_dynamic else '关闭'}")
    print(f"   混淆概率: {CHAOS_RATE}")
    print("=" * 80)

    if mode == "equiv":
        run_equivalence_mode(rounds=rounds, seed=seed, enable_dynamic_ops=enable_dynamic)
    elif mode == "pqs":
        run_pqs_mode(rounds=pqs_rounds, seed=seed, enable_dynamic_ops=enable_dynamic)
    elif mode == "group":
        run_group_test(rounds=rounds, seed=seed, enable_dynamic_ops=enable_dynamic)
    else:
        # 默认运行 Oracle 模式
        run(rounds=rounds, seed=seed, enable_dynamic_ops=enable_dynamic)
        # 可选：同时运行 PQS 模式
        # run_pqs_mode(rounds=pqs_rounds, seed=seed)
