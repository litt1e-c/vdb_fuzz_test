"""
Weaviate Dynamic Fuzzing System (Full Oracle Mode)
Features:
1. Dynamic Schema: Random 5-20 fields + nested object structure
2. Configurable Data: Adjustable vectors and dimensions
3. Robust Query Generation: Random filter expressions with type safety
4. Oracle Testing: Pandas comparison for correctness verification
5. PQS (Pivot Query Synthesis): Must-hit query testing
6. Equivalence Testing: Query transformation validation
7. Null Value Handling: Comprehensive null/empty value testing
"""
import time
import random
import string
import numpy as np
import pandas as pd
import json
import sys
import uuid
import weaviate
from weaviate.classes.config import Configure, Property, DataType, Reconfigure
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject

# 是否启用 Null 过滤测试 (需要 Weaviate 支持 indexNullState)
ENABLE_NULL_FILTER = False  # 设为 True 如果你的 Weaviate 版本支持

# Weaviate 停用词列表 (英文常见停用词)
WEAVIATE_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in',
    'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the',
    'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'its', 'itself', 'what', 'which', 'who', 'whom', 'when',
    'where', 'why', 'how', 'all', 'am', 'been', 'being', 'both', 'can', 'could',
    'did', 'do', 'does', 'doing', 'done', 'down', 'during', 'each', 'few', 'from',
    'further', 'had', 'has', 'have', 'having', 'here', 'just', 'may', 'might',
    'more', 'most', 'must', 'nor', 'now', 'only', 'other', 'ought', 'own', 'same',
    'shall', 'should', 'so', 'some', 'than', 'too', 'until', 'up', 'very', 'were',
    'would', 'about', 'above', 'after', 'again', 'against', 'any', 'because',
    'before', 'below', 'between', 'cannot', 'could', 'either', 'else', 'ever',
    'every', 'get', 'got', 'had', 'hardly', 'however', 'indeed', 'isn', 'let',
    'likely', 'may', 'neither', 'never', 'nobody', 'nothing', 'often', 'once',
    'over', 'perhaps', 'rather', 'really', 'said', 'say', 'seem', 'seemed',
    'seems', 'since', 'still', 'sure', 'though', 'through', 'thus', 'under',
    'unless', 'upon', 'very', 'want', 'wants', 'way', 'ways', 'well', 'went',
    'whether', 'while', 'within', 'without', 'yet', 'am', 'are', 'aren', 'arent',
    'isn', 'isnt', 'wasn', 'wasnt', 'weren', 'werent', 'won', 'wont', 'wouldn',
    'wouldnt', 'can', 'cant', 'cannot', 'couldn', 'couldnt', 'didn', 'didnt',
    'doesn', 'doesnt', 'don', 'dont', 'hadn', 'hadnt', 'hasn', 'hasnt', 'haven',
    'havent', 'mightn', 'mightnt', 'mustn', 'mustnt', 'needn', 'neednt', 'shan',
    'shant', 'shouldn', 'shouldnt'
}

def is_stopword(word):
    """检查一个词是否是停用词"""
    if not isinstance(word, str):
        return False
    return word.lower().strip() in WEAVIATE_STOPWORDS

# --- Configuration (User Specified) ---
HOST = "127.0.0.1"
PORT = 8080                # Weaviate default port
CLASS_NAME = "FuzzOracleV1"
N = 3000                   # 数据量 (Weaviate 可能比 Qdrant 慢，适当减少)
DIM = 128                  # 向量维度
BATCH_SIZE = 200           # 批次大小
SLEEP_INTERVAL = 0.02      # 每次插入后暂停

# 随机化配置
VECTOR_CHECK_RATIO = random.uniform(0.2, 0.8)  # 20-80% 概率做向量+标量联合校验
VECTOR_TOPK = random.randint(50, 200)          # TopK 在 50-200 之间随机

# --- 1. Data Manager ---

class FieldType:
    """模拟字段类型枚举"""
    INT = "INT"
    NUMBER = "NUMBER"  # Weaviate uses NUMBER for floats
    BOOL = "BOOL"
    TEXT = "TEXT"      # Weaviate uses TEXT/STRING
    OBJECT = "OBJECT"  # Nested objects
    INT_ARRAY = "INT_ARRAY"
    TEXT_ARRAY = "TEXT_ARRAY"
    NUMBER_ARRAY = "NUMBER_ARRAY"

def get_weaviate_datatype(ftype):
    """Map field type to Weaviate DataType"""
    mapping = {
        FieldType.INT: DataType.INT,
        FieldType.NUMBER: DataType.NUMBER,
        FieldType.BOOL: DataType.BOOL,
        FieldType.TEXT: DataType.TEXT,
        FieldType.INT_ARRAY: DataType.INT_ARRAY,
        FieldType.TEXT_ARRAY: DataType.TEXT_ARRAY,
        FieldType.NUMBER_ARRAY: DataType.NUMBER_ARRAY,
    }
    return mapping.get(ftype)

class DataManager:
    def __init__(self):
        self.df = None
        self.vectors = None
        self.schema_config = []
        self.null_ratio = random.uniform(0.05, 0.15)  # 空值比例
        self.array_capacity = random.randint(5, 20)
        self.int_range = random.randint(5000, 100000)
        self.double_scale = random.uniform(100, 10000)

    def _random_string(self, min_len=1, max_len=10):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=random.randint(min_len, max_len)))

    def generate_schema(self):
        print("🎲 1. Defining Dynamic Schema...")
        self.schema_config = []
        num_fields = random.randint(3, 15)
        types_pool = [FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT]

        for i in range(num_fields):
            ftype = random.choice(types_pool)
            self.schema_config.append({"name": f"c{i}", "type": ftype})

        # 添加数组字段
        self.schema_config.append({
            "name": "tagsArray",
            "type": FieldType.INT_ARRAY,
            "max_capacity": self.array_capacity
        })
        
        self.schema_config.append({
            "name": "labelsArray",
            "type": FieldType.TEXT_ARRAY,
            "max_capacity": self.array_capacity
        })

        # 添加数字数组
        self.schema_config.append({
            "name": "scoresArray",
            "type": FieldType.NUMBER_ARRAY,
            "max_capacity": self.array_capacity
        })

        print(f"   -> Generated {len(self.schema_config)} dynamic fields (plus id & vector).")
        print("   -> Schema Structure:")
        for f in self.schema_config:
            t_name = f["type"]
            print(f"      - {f['name']:<20} : {t_name}")

    def generate_data(self):
        print(f"🌊 2. Generating {N} rows (Vector Dim={DIM})...")
        rng = np.random.default_rng(42)
        self.vectors = rng.random((N, DIM), dtype=np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1)[:, np.newaxis]

        # 生成 UUID 作为 ID
        data = {"id": [str(uuid.uuid4()) for _ in range(N)]}
        data["row_num"] = list(range(N))  # 用于调试的行号
        
        print("   -> Filling scalar attributes...")
        
        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]

            if ftype == FieldType.INT:
                values = rng.integers(-self.int_range, self.int_range, size=N).tolist()
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.NUMBER:
                values = (rng.random(N) * self.double_scale).tolist()
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.BOOL:
                values = rng.choice([True, False], size=N).tolist()
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.TEXT:
                values = [self._random_string(1, random.randint(5, 30)) for _ in range(N)]
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.INT_ARRAY:
                arr_list = []
                for _ in range(N):
                    if rng.random() < self.null_ratio:
                        arr_list.append(None)
                    else:
                        length = random.randint(0, 5)
                        arr_list.append([int(x) for x in rng.integers(0, 100, size=length)])
                data[fname] = arr_list
            elif ftype == FieldType.TEXT_ARRAY:
                arr_list = []
                for _ in range(N):
                    if rng.random() < self.null_ratio:
                        arr_list.append(None)
                    else:
                        length = random.randint(0, 5)
                        arr_list.append([self._random_string(2, 8) for _ in range(length)])
                data[fname] = arr_list
            elif ftype == FieldType.NUMBER_ARRAY:
                arr_list = []
                for _ in range(N):
                    if rng.random() < self.null_ratio:
                        arr_list.append(None)
                    else:
                        length = random.randint(0, 5)
                        arr_list.append([float(x) for x in rng.random(length) * 100])
                data[fname] = arr_list

        self.df = pd.DataFrame(data)
        print("✅ Data Generation Complete.")
        
        # 打印空值统计
        null_counts = {col: self.df[col].isna().sum() for col in self.df.columns}
        print(f"   -> Null value counts (sample): {dict(list(null_counts.items())[:5])}")

    def _apply_nulls(self, values, rng):
        """根据 null_ratio 用 None 替换部分值"""
        if not isinstance(values, list):
            values = list(values)
        for idx in range(len(values)):
            if rng.random() < self.null_ratio:
                values[idx] = None
        return values


# --- 2. Weaviate Manager ---

class WeaviateManager:
    def __init__(self):
        self.client = None

    def connect(self):
        print(f"🔌 Connecting to Weaviate at {HOST}:{PORT}...")
        try:
            self.client = weaviate.connect_to_local(
                host=HOST,
                port=PORT,
                grpc_port=50051  # 默认 gRPC 端口
            )
            # 测试连接
            if self.client.is_ready():
                print("✅ Connected to Weaviate successfully.")
            else:
                print("❌ Weaviate is not ready.")
                exit(1)
        except Exception as e:
            # 尝试不使用 gRPC 连接
            try:
                self.client = weaviate.connect_to_local(
                    host=HOST,
                    port=PORT,
                    skip_init_checks=True
                )
                print("✅ Connected to Weaviate (without gRPC).")
            except Exception as e2:
                print(f"❌ Connection failed: {e2}")
                print("   (Please check if Weaviate container is running)")
                exit(1)

    def close(self):
        if self.client:
            self.client.close()

    def reset_collection(self, schema_config):
        """删除并重新创建 collection"""
        # 删除已存在的 class
        try:
            self.client.collections.delete(CLASS_NAME)
            print(f"   -> Deleted existing class: {CLASS_NAME}")
        except Exception as e:
            pass

        # 构建属性列表
        properties = []
        for field in schema_config:
            fname = field["name"]
            ftype = field["type"]
            
            wv_type = get_weaviate_datatype(ftype)
            if wv_type:
                properties.append(
                    Property(name=fname, data_type=wv_type)
                )

        # 添加 row_num 属性用于调试
        properties.append(Property(name="row_num", data_type=DataType.INT))

        # 创建新 collection (启用 null 状态索引以支持 is_none 过滤)
        try:
            self.client.collections.create(
                name=CLASS_NAME,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),
                inverted_index_config=Configure.inverted_index(
                    index_null_state=True  # 启用 null 状态索引
                ) if ENABLE_NULL_FILTER else None
            )
        except Exception as e:
            # 回退到不带 null 索引的创建方式
            print(f"   -> Note: Creating without null index (error: {e})")
            self.client.collections.create(
                name=CLASS_NAME,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),
            )
        print("🛠️ Collection Created.")

    def insert(self, dm):
        print(f"⚡ 3. Inserting Data (Batch={BATCH_SIZE}, Sleep={SLEEP_INTERVAL}s)...")
        
        collection = self.client.collections.get(CLASS_NAME)
        records = dm.df.to_dict(orient="records")
        total = len(records)
        
        # 构建字段类型映射
        field_type_map = {f["name"]: f["type"] for f in dm.schema_config}
        field_type_map["row_num"] = FieldType.INT

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
            """根据 schema 类型正确转换值"""
            if v is None:
                return None
            if isinstance(v, float) and np.isnan(v):
                return None
            
            ftype = field_type_map.get(k)
            
            if ftype == FieldType.INT:
                return int(v)
            elif ftype == FieldType.BOOL:
                return bool(v)
            elif ftype == FieldType.NUMBER:
                return float(v)
            elif ftype == FieldType.TEXT:
                return str(v)
            else:
                return convert_numpy_types(v)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_data = records[start:end]

            data_objects = []
            for i, row in enumerate(batch_data):
                obj_uuid = row["id"]
                vector = dm.vectors[start + i].tolist()

                # 构建属性
                properties = {}
                for k, v in row.items():
                    if k == "id":
                        continue
                    converted_val = convert_value_by_type(k, v, field_type_map)
                    # 只添加非 None 的值 (Weaviate 中不存储 null)
                    if converted_val is not None:
                        properties[k] = converted_val

                data_objects.append(
                    DataObject(
                        uuid=obj_uuid,
                        properties=properties,
                        vector=vector
                    )
                )

            # 重试逻辑
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    collection.data.insert_many(data_objects)
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


# --- 3. Query Generator (Oracle Mode) ---

class OracleQueryGenerator:
    """
    生成 Weaviate 过滤条件并同时生成对应的 Pandas mask
    用于 Oracle 对比测试
    """
    def __init__(self, dm):
        self.schema = dm.schema_config
        self.df = dm.df
        self.dm = dm  # 保存 DataManager 引用
        
        # 预计算字段统计信息用于边界值测试
        self._field_stats = {}
        for field in self.schema:
            fname = field["name"]
            ftype = field["type"]
            series = self.df[fname].dropna()
            if ftype in [FieldType.INT, FieldType.NUMBER] and not series.empty:
                self._field_stats[fname] = {
                    "min": series.min(),
                    "max": series.max(),
                    "median": series.median(),
                    "q1": series.quantile(0.25),
                    "q3": series.quantile(0.75)
                }

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

    def get_value_for_query(self, fname, ftype):
        """获取用于查询的值"""
        valid_series = self.df[fname].dropna()
        if not valid_series.empty and random.random() < 0.8:
            val = random.choice(valid_series.values)
            if hasattr(val, "item"):
                val = val.item()
            return val

        # 生成不存在的值
        if ftype == FieldType.INT:
            if not valid_series.empty:
                min_val, max_val = int(valid_series.min()), int(valid_series.max())
                return random.choice([max_val + 100000, min_val - 100000])
            return random.randint(-200000, 200000)

        elif ftype == FieldType.NUMBER:
            if not valid_series.empty:
                min_val, max_val = float(valid_series.min()), float(valid_series.max())
                val = random.choice([max_val + 100000.0, min_val - 100000.0])
            else:
                val = random.random() * 200000.0
            return float(np.float32(val))

        elif ftype == FieldType.BOOL:
            return random.choice([True, False])

        elif ftype == FieldType.TEXT:
            chars = string.ascii_letters + string.digits + "!@#$%^&*"
            length = random.randint(15, 30)
            return ''.join(random.choices(chars, k=length))

        elif ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY]:
            return random.randint(50000, 100000)

        return None

    def gen_boundary_expr(self):
        """
        生成边界值测试表达式
        测试: min, max, min-1, max+1, 中位数等边界条件
        """
        # 找一个有统计信息的数值字段
        numeric_fields = [f for f in self.schema if f["name"] in self._field_stats]
        if not numeric_fields:
            return None, None, None
        
        field = random.choice(numeric_fields)
        fname = field["name"]
        ftype = field["type"]
        stats = self._field_stats[fname]
        series = self.df[fname]
        
        # 选择边界测试类型
        boundary_type = random.choice([
            "exact_min", "exact_max", "below_min", "above_max",
            "at_median", "range_q1_q3", "range_narrow"
        ])
        
        def safe_compare(op, val):
            def comp(x):
                if pd.isna(x):
                    return False
                try:
                    if op == "==": return x == val
                    if op == ">": return x > val
                    if op == "<": return x < val
                    if op == ">=": return x >= val
                    if op == "<=": return x <= val
                except:
                    return False
                return False
            return comp
        
        if ftype == FieldType.INT:
            min_v, max_v = int(stats["min"]), int(stats["max"])
            median_v = int(stats["median"])
            q1, q3 = int(stats["q1"]), int(stats["q3"])
            
            if boundary_type == "exact_min":
                filter_cond = Filter.by_property(fname).equal(min_v)
                mask = series.apply(safe_compare("==", min_v))
                return filter_cond, mask, f"{fname} == {min_v} (exact min)"
            
            elif boundary_type == "exact_max":
                filter_cond = Filter.by_property(fname).equal(max_v)
                mask = series.apply(safe_compare("==", max_v))
                return filter_cond, mask, f"{fname} == {max_v} (exact max)"
            
            elif boundary_type == "below_min":
                val = min_v - 1
                filter_cond = Filter.by_property(fname).less_than(val)
                mask = series.apply(safe_compare("<", val))
                return filter_cond, mask, f"{fname} < {val} (below min, expect empty)"
            
            elif boundary_type == "above_max":
                val = max_v + 1
                filter_cond = Filter.by_property(fname).greater_than(val)
                mask = series.apply(safe_compare(">", val))
                return filter_cond, mask, f"{fname} > {val} (above max, expect empty)"
            
            elif boundary_type == "at_median":
                filter_cond = Filter.by_property(fname).equal(median_v)
                mask = series.apply(safe_compare("==", median_v))
                return filter_cond, mask, f"{fname} == {median_v} (median)"
            
            elif boundary_type == "range_q1_q3":
                # 四分位范围查询
                filter_cond = (
                    Filter.by_property(fname).greater_or_equal(q1) &
                    Filter.by_property(fname).less_or_equal(q3)
                )
                mask = series.apply(lambda x: q1 <= x <= q3 if pd.notna(x) else False)
                return filter_cond, mask, f"{fname} BETWEEN {q1} AND {q3} (IQR)"
            
            else:  # range_narrow
                # 窄范围查询
                mid = (min_v + max_v) // 2
                delta = max(1, (max_v - min_v) // 20)
                filter_cond = (
                    Filter.by_property(fname).greater_or_equal(mid - delta) &
                    Filter.by_property(fname).less_or_equal(mid + delta)
                )
                mask = series.apply(lambda x: (mid - delta) <= x <= (mid + delta) if pd.notna(x) else False)
                return filter_cond, mask, f"{fname} BETWEEN {mid-delta} AND {mid+delta} (narrow)"
        
        elif ftype == FieldType.NUMBER:
            min_v, max_v = float(stats["min"]), float(stats["max"])
            median_v = float(stats["median"])
            
            if boundary_type in ["exact_min", "exact_max", "at_median"]:
                # 浮点数精确匹配困难，改用范围
                epsilon = 1e-6
                if boundary_type == "exact_min":
                    target = min_v
                elif boundary_type == "exact_max":
                    target = max_v
                else:
                    target = median_v
                
                filter_cond = (
                    Filter.by_property(fname).greater_or_equal(target - epsilon) &
                    Filter.by_property(fname).less_or_equal(target + epsilon)
                )
                mask = series.apply(lambda x: (target - epsilon) <= x <= (target + epsilon) if pd.notna(x) else False)
                return filter_cond, mask, f"{fname} ≈ {target} (float boundary)"
            
            elif boundary_type == "below_min":
                val = min_v - 1.0
                filter_cond = Filter.by_property(fname).less_than(val)
                mask = series.apply(safe_compare("<", val))
                return filter_cond, mask, f"{fname} < {val} (below min)"
            
            elif boundary_type == "above_max":
                val = max_v + 1.0
                filter_cond = Filter.by_property(fname).greater_than(val)
                mask = series.apply(safe_compare(">", val))
                return filter_cond, mask, f"{fname} > {val} (above max)"
            
            else:  # range queries
                mid = (min_v + max_v) / 2
                delta = max(0.001, (max_v - min_v) / 10)
                filter_cond = (
                    Filter.by_property(fname).greater_or_equal(mid - delta) &
                    Filter.by_property(fname).less_or_equal(mid + delta)
                )
                mask = series.apply(lambda x: (mid - delta) <= x <= (mid + delta) if pd.notna(x) else False)
                return filter_cond, mask, f"{fname} BETWEEN {mid-delta:.4f} AND {mid+delta:.4f}"
        
        return None, None, None

    def gen_multi_array_expr(self):
        """
        生成数组的多值查询表达式
        测试: contains_any with multiple values, 空数组边界
        """
        array_fields = [f for f in self.schema if f["type"] in [
            FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY
        ]]
        
        if not array_fields:
            return None, None, None
        
        field = random.choice(array_fields)
        fname = field["name"]
        ftype = field["type"]
        series = self.df[fname]
        
        # 收集所有数组元素
        valid_series = series.dropna()
        all_items = []
        for arr in valid_series:
            if isinstance(arr, list):
                all_items.extend(arr)
        
        # 对 TEXT_ARRAY 过滤停用词
        if ftype == FieldType.TEXT_ARRAY:
            all_items = [item for item in all_items if not is_stopword(item)]
        
        if not all_items:
            return None, None, None
        
        # 随机选择 2-4 个值进行多值查询
        num_targets = min(random.randint(2, 4), len(set(all_items)))
        targets = random.sample(list(set(all_items)), num_targets)
        targets = [self._convert_to_native(t) for t in targets]
        
        # contains_any with multiple values
        filter_cond = Filter.by_property(fname).contains_any(targets)
        
        if ftype == FieldType.TEXT_ARRAY:
            targets_lower = [str(t).lower() for t in targets]
            mask = series.apply(
                lambda x: any(str(item).lower() in targets_lower for item in x) 
                if isinstance(x, list) else False
            )
        else:
            mask = series.apply(
                lambda x: any(item in targets for item in x) 
                if isinstance(x, list) else False
            )
        
        return filter_cond, mask, f"{fname} contains_any {targets}"

    def gen_atomic_expr(self):
        """
        生成原子表达式
        返回: (weaviate_filter, pandas_mask, expr_str)
        """
        f = random.choice(self.schema)
        name, ftype = f["name"], f["type"]
        series = self.df[name]

        # 1. Null Check - 只在启用时生成
        if ENABLE_NULL_FILTER and random.random() < 0.15:
            if random.random() < 0.5:
                # is null
                filter_cond = Filter.by_property(name).is_none(True)
                return (filter_cond, series.isnull(), f"{name} is null")
            else:
                # is not null
                filter_cond = Filter.by_property(name).is_none(False)
                return (filter_cond, series.notnull(), f"{name} is not null")

        val = self.get_value_for_query(name, ftype)
        if val is None:
            filter_cond = Filter.by_property(name).is_none(True)
            return (filter_cond, series.isnull(), f"{name} is null")

        # 安全比较函数 - 处理 None 和 NaN
        def safe_compare_scalar(op, target_val):
            def comp(x):
                if pd.isna(x):
                    return False
                try:
                    if op == "==":
                        return x == target_val
                    if op == "!=":
                        return x != target_val
                    if op == ">":
                        return x > target_val
                    if op == "<":
                        return x < target_val
                    if op == ">=":
                        return x >= target_val
                    if op == "<=":
                        return x <= target_val
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
            filter_cond = Filter.by_property(name).equal(val_bool)
            mask = series.apply(safe_compare_scalar("==", val_bool))
            expr_str = f"{name} == {val_bool}"

        elif ftype == FieldType.INT:
            val_int = int(val)
            # 注意：Weaviate 的 not_equal 会返回 null 值的行，所以我们避免使用 !=
            op = random.choice([">", "<", "==", ">=", "<="])
            
            if op == "==":
                filter_cond = Filter.by_property(name).equal(val_int)
            elif op == ">":
                filter_cond = Filter.by_property(name).greater_than(val_int)
            elif op == "<":
                filter_cond = Filter.by_property(name).less_than(val_int)
            elif op == ">=":
                filter_cond = Filter.by_property(name).greater_or_equal(val_int)
            elif op == "<=":
                filter_cond = Filter.by_property(name).less_or_equal(val_int)
            
            mask = series.apply(safe_compare_scalar(op, val_int))
            expr_str = f"{name} {op} {val_int}"

        elif ftype == FieldType.NUMBER:
            val_float = float(val)
            op = random.choice([">", "<", ">=", "<="])  # 避免 == 精度问题
            
            if op == ">":
                filter_cond = Filter.by_property(name).greater_than(val_float)
            elif op == "<":
                filter_cond = Filter.by_property(name).less_than(val_float)
            elif op == ">=":
                filter_cond = Filter.by_property(name).greater_or_equal(val_float)
            elif op == "<=":
                filter_cond = Filter.by_property(name).less_or_equal(val_float)
            
            mask = series.apply(safe_compare_scalar(op, val_float))
            expr_str = f"{name} {op} {val_float}"

        elif ftype == FieldType.TEXT:
            # 注意：Weaviate 对 TEXT 字段进行不区分大小写的匹配
            op = random.choice(["==", "like"])
            
            if op == "==":
                filter_cond = Filter.by_property(name).equal(val)
                # Weaviate TEXT == 是不区分大小写的
                val_lower = str(val).lower()
                mask = series.apply(lambda x, v=val_lower: str(x).lower() == v if pd.notna(x) else False)
                expr_str = f'{name} == "{val}"'
            else:  # like
                # 使用前缀匹配，确保前缀是有效的
                prefix = val[:3] if len(val) >= 3 else val
                # 确保前缀只包含字母数字
                prefix = ''.join(c for c in prefix if c.isalnum())
                if not prefix:
                    prefix = "a"  # 默认前缀
                filter_cond = Filter.by_property(name).like(f"{prefix}*")
                mask = series.apply(lambda x, p=prefix: x is not None and str(x).lower().startswith(p.lower()) if pd.notna(x) else False)
                expr_str = f'{name} like "{prefix}*"'

        elif ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY]:
            # 数组包含查询
            valid_series = self.df[name].dropna()
            all_items = []
            for arr in valid_series:
                if isinstance(arr, list):
                    all_items.extend(arr)
            
            # 对于 TEXT_ARRAY，过滤停用词
            if ftype == FieldType.TEXT_ARRAY:
                all_items = [item for item in all_items if not is_stopword(item)]
            
            if all_items:
                target = random.choice(all_items)
                target = self._convert_to_native(target)
                filter_cond = Filter.by_property(name).contains_any([target])
                
                # Weaviate 对字符串数组进行不区分大小写的匹配
                if ftype == FieldType.TEXT_ARRAY:
                    target_lower = str(target).lower()
                    mask = series.apply(lambda x, t=target_lower: any(str(item).lower() == t for item in x) if isinstance(x, list) else False)
                else:
                    mask = series.apply(lambda x, t=target: t in x if isinstance(x, list) else False)
                expr_str = f'{name} contains {target}'
            else:
                if ENABLE_NULL_FILTER:
                    filter_cond = Filter.by_property(name).is_none(True)
                    mask = series.isnull()
                    expr_str = f'{name} is null'
                else:
                    # 如果禁用 null 过滤，返回一个恒假条件
                    filter_cond = Filter.by_property(name).contains_any(["__impossible_fuzz_value__"])
                    mask = pd.Series(False, index=self.df.index)
                    expr_str = f'{name} contains __impossible__'

        if filter_cond is not None and mask is not None:
            return (filter_cond, mask, expr_str)

        # 默认返回 - 只在启用 null 过滤时使用
        if ENABLE_NULL_FILTER:
            filter_cond = Filter.by_property(name).is_none(False)
            return (filter_cond, series.notnull(), f"{name} is not null")
        else:
            # 返回一个简单的恒真条件
            int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
            if int_fields:
                fname = int_fields[0]["name"]
                s = self.df[fname]
                min_v = int(s.min()) if not s.isna().all() else 0
                max_v = int(s.max()) if not s.isna().all() else 0
                filter_cond = (
                    Filter.by_property(fname).greater_or_equal(min_v - 1000000) &
                    Filter.by_property(fname).less_or_equal(max_v + 1000000)
                )
                return (filter_cond, s.notna(), f"{fname} in range (fallback)")
            return None, None, None

    def gen_constant_expr(self):
        """生成常量表达式"""
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            field_name = int_fields[0]["name"]
            series = self.df[field_name]
            min_val = int(series.min()) if not series.isna().all() else 0
            max_val = int(series.max()) if not series.isna().all() else 0
        else:
            float_fields = [f for f in self.schema if f["type"] == FieldType.NUMBER]
            if float_fields:
                field_name = float_fields[0]["name"]
                series = self.df[field_name]
                min_val = float(series.min()) if not series.isna().all() else 0
                max_val = float(series.max()) if not series.isna().all() else 0
            else:
                return None, None, None
        
        # 恒真条件（对非 null 值）
        true_mask = series.notna()
        false_mask = pd.Series(False, index=self.df.index)

        if random.random() < 0.5:
            # 必真：大范围
            filter_cond = (
                Filter.by_property(field_name).greater_or_equal(min_val - 1000000) &
                Filter.by_property(field_name).less_or_equal(max_val + 1000000)
            )
            return (filter_cond, true_mask, f"{field_name} in wide range (always true)")
        else:
            # 必假：不可能的范围
            impossible_val = max_val + 10000000
            filter_cond = Filter.by_property(field_name).greater_than(impossible_val)
            return (filter_cond, false_mask, f"{field_name} > {impossible_val} (always false)")

    def gen_complex_expr(self, depth):
        """递归生成复杂表达式，增强版包含更多测试类型"""
        if depth == 0 or random.random() < 0.3:
            # 随机选择表达式类型，增加边界值和多值数组测试的权重
            expr_choice = random.random()
            
            if expr_choice < 0.02:
                # 2% 概率生成常量表达式
                res = self.gen_constant_expr()
                if res[0]:
                    return res
            
            elif expr_choice < 0.12:
                # 10% 概率生成边界值表达式
                res = self.gen_boundary_expr()
                if res[0] is not None:
                    return res
            
            elif expr_choice < 0.20:
                # 8% 概率生成多值数组表达式
                res = self.gen_multi_array_expr()
                if res[0] is not None:
                    return res

            # 默认生成原子表达式
            filter_cond, mask, expr_str = self.gen_atomic_expr()
            if filter_cond:
                return filter_cond, mask, expr_str
            return self.gen_complex_expr(depth)

        # 递归生成子节点
        filter_l, mask_l, expr_l = self.gen_complex_expr(depth - 1)
        filter_r, mask_r, expr_r = self.gen_complex_expr(depth - 1)

        if not filter_l:
            return filter_r, mask_r, expr_r
        if not filter_r:
            return filter_l, mask_l, expr_l

        op = random.choice(["and", "or"])

        if op == "and":
            combined_filter = filter_l & filter_r
            return combined_filter, (mask_l & mask_r), f"({expr_l} AND {expr_r})"
        else:
            combined_filter = filter_l | filter_r
            return combined_filter, (mask_l | mask_r), f"({expr_l} OR {expr_r})"


# --- 4. Equivalence Query Generator ---

class EquivalenceQueryGenerator(OracleQueryGenerator):
    """
    等价查询生成器：生成逻辑等价的查询变体，验证结果一致性
    """
    def __init__(self, dm):
        super().__init__(dm)
        self.field_types = {f["name"]: f["type"] for f in dm.schema_config}
        self._tautology_field = self._find_tautology_field()

    def _find_tautology_field(self):
        """找到用于生成恒真条件的字段（INT 或 NUMBER）"""
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            name = int_fields[0]["name"]
            series = self.df[name].dropna()
            if not series.empty:
                return {"name": name, "min": int(series.min()), "max": int(series.max()), "type": "int"}
        
        float_fields = [f for f in self.schema if f["type"] == FieldType.NUMBER]
        if float_fields:
            name = float_fields[0]["name"]
            series = self.df[name].dropna()
            if not series.empty:
                return {"name": name, "min": float(series.min()), "max": float(series.max()), "type": "float"}
        
        return None

    def _gen_tautology_filter(self):
        """生成恒真条件（返回所有非 null 数据）"""
        if self._tautology_field:
            name = self._tautology_field["name"]
            min_val = self._tautology_field["min"]
            max_val = self._tautology_field["max"]
            # 宽范围
            filter_cond = (
                Filter.by_property(name).greater_or_equal(min_val - 1000000) &
                Filter.by_property(name).less_or_equal(max_val + 1000000)
            )
            return (filter_cond, f"({name} >= {min_val - 1000000} AND {name} <= {max_val + 1000000})")
        return None, None

    def _gen_guaranteed_false_filter(self):
        """生成必然为假的过滤条件"""
        field = random.choice(self.schema)
        name = field["name"]
        dtype = field["type"]

        if dtype == FieldType.TEXT:
            complex_str = "fuzz_impossible_" + self._random_string(10, 20)
            return (
                Filter.by_property(name).equal(complex_str),
                f'{name} == "{complex_str}"'
            )

        elif dtype == FieldType.INT:
            return (
                Filter.by_property(name).greater_than(200000) &
                Filter.by_property(name).less_than(-200000),
                f"({name} > 200000 AND {name} < -200000)"
            )

        elif dtype == FieldType.NUMBER:
            return (
                Filter.by_property(name).greater_than(1e20) &
                Filter.by_property(name).less_than(-1e20),
                f"({name} > 1e20 AND {name} < -1e20)"
            )

        # 默认使用 INT 字段
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            int_name = int_fields[0]["name"]
            return (
                Filter.by_property(int_name).greater_than(200000) &
                Filter.by_property(int_name).less_than(-200000),
                f"({int_name} > 200000 AND {int_name} < -200000)"
            )
        
        # 回退到 TEXT
        str_fields = [f for f in self.schema if f["type"] == FieldType.TEXT]
        if str_fields:
            str_name = str_fields[0]["name"]
            return (
                Filter.by_property(str_name).equal("__impossible_value_fuzz__"),
                f'{str_name} == "__impossible_value_fuzz__"'
            )
        
        return None, None

    def mutate_filter(self, base_filter, base_expr):
        """输入基础过滤器，返回逻辑等价的变体列表"""
        mutations = []

        # 1. 恒真条件注入 (Tautology AND)
        # 只有当 tautology 字段没有 null 值时才安全使用
        tautology_filter, tautology_expr = self._gen_tautology_filter()
        if tautology_filter and self._tautology_field and self._tautology_field.get("no_nulls", False):
            mutations.append({
                "type": "TautologyAnd",
                "filter": base_filter & tautology_filter,
                "expr": f"({base_expr}) AND ({tautology_expr})"
            })

        # 2. 冗余 OR (Self OR)
        mutations.append({
            "type": "SelfOr",
            "filter": base_filter | base_filter,
            "expr": f"({base_expr}) OR ({base_expr})"
        })

        # 3. 噪声 OR 注入 (Empty Set Merge)
        false_filter, false_expr = self._gen_guaranteed_false_filter()
        if false_filter:
            mutations.append({
                "type": "NoiseOr",
                "filter": base_filter | false_filter,
                "expr": f"({base_expr}) OR ({false_expr})"
            })

        return mutations


# --- 5. PQS Query Generator ---

class PQSQueryGenerator(OracleQueryGenerator):
    """
    PQS 生成器：专注于生成"必须能查到指定行"的查询
    """
    def __init__(self, dm):
        super().__init__(dm)
        self._tautology_field = self._find_tautology_field()

    def _find_tautology_field(self):
        """找到用于生成恒真条件的字段 - 优先选择没有 null 值的字段"""
        # 优先寻找没有 null 值的 INT 字段
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        for field in int_fields:
            name = field["name"]
            null_count = self.df[name].isna().sum()
            if null_count == 0:
                series = self.df[name]
                return {"name": name, "min": int(series.min()), "max": int(series.max()), "type": "int", "no_nulls": True}
        
        # 如果所有 INT 字段都有 null，使用第一个（会影响等价测试）
        if int_fields:
            name = int_fields[0]["name"]
            series = self.df[name].dropna()
            if not series.empty:
                return {"name": name, "min": int(series.min()), "max": int(series.max()), "type": "int", "no_nulls": False}
        
        # 尝试 NUMBER 字段
        float_fields = [f for f in self.schema if f["type"] == FieldType.NUMBER]
        for field in float_fields:
            name = field["name"]
            null_count = self.df[name].isna().sum()
            if null_count == 0:
                series = self.df[name]
                return {"name": name, "min": float(series.min()), "max": float(series.max()), "type": "float", "no_nulls": True}
        
        if float_fields:
            name = float_fields[0]["name"]
            series = self.df[name].dropna()
            if not series.empty:
                return {"name": name, "min": float(series.min()), "max": float(series.max()), "type": "float", "no_nulls": False}
        
        return None

    def _gen_tautology_filter(self):
        """生成恒真条件"""
        if self._tautology_field:
            name = self._tautology_field["name"]
            min_val = self._tautology_field["min"]
            max_val = self._tautology_field["max"]
            filter_cond = (
                Filter.by_property(name).greater_or_equal(min_val - 1000000) &
                Filter.by_property(name).less_or_equal(max_val + 1000000)
            )
            return (filter_cond, f"({name} in wide range)")
        return None, None

    def _gen_guaranteed_false_filter(self):
        """生成必然为假的过滤条件"""
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            name = int_fields[0]["name"]
            return (
                Filter.by_property(name).greater_than(200000) &
                Filter.by_property(name).less_than(-200000),
                f"({name} > 200000 AND {name} < -200000)"
            )
        
        float_fields = [f for f in self.schema if f["type"] == FieldType.NUMBER]
        if float_fields:
            name = float_fields[0]["name"]
            return (
                Filter.by_property(name).greater_than(1e20) &
                Filter.by_property(name).less_than(-1e20),
                f"({name} > 1e20 AND {name} < -1e20)"
            )
        
        return None, None

    def gen_pqs_filter(self, pivot_row, depth):
        """生成针对指定行必真的过滤条件"""
        force_recursion = depth > 3

        if depth <= 0 or (not force_recursion and random.random() < 0.3):
            result = self.gen_true_atomic_filter(pivot_row)
            if result[0] is None:
                # 如果生成失败（比如字段为 null），尝试用 tautology 替代
                tautology_filter, tautology_expr = self._gen_tautology_filter()
                if tautology_filter:
                    return tautology_filter, tautology_expr
            return result

        op = random.choice(["and", "or"])

        tautology_filter, tautology_expr = self._gen_tautology_filter()
        false_filter, false_expr = self._gen_guaranteed_false_filter()

        if op == "and":
            filter_l, expr_l = self.gen_pqs_filter(pivot_row, depth - 1)
            filter_r, expr_r = self.gen_pqs_filter(pivot_row, depth - 1)
            if not filter_l:
                filter_l, expr_l = tautology_filter, tautology_expr
            if not filter_r:
                filter_r, expr_r = tautology_filter, tautology_expr
            
            if filter_l and filter_r:
                return filter_l & filter_r, f"({expr_l} AND {expr_r})"
            return filter_l, expr_l

        else:  # OR
            filter_l, expr_l = self.gen_pqs_filter(pivot_row, depth - 1)
            noise_filter, noise_expr = self._gen_guaranteed_false_filter()

            if not filter_l:
                filter_l, expr_l = tautology_filter, tautology_expr
            if not noise_filter:
                noise_filter, noise_expr = false_filter, false_expr

            if filter_l and noise_filter:
                return filter_l | noise_filter, f"({expr_l} OR {noise_expr})"
            return filter_l, expr_l

    def gen_true_atomic_filter(self, row):
        """针对单行数据，生成必真的原子条件"""
        # 尝试多个字段，避免一直选到 null 字段
        shuffled_schema = random.sample(self.schema, len(self.schema))
        
        for field in shuffled_schema:
            fname = field["name"]
            ftype = field["type"]
            val = row.get(fname)

            # 处理 Null
            is_null = False
            try:
                if val is None:
                    is_null = True
                if isinstance(val, float) and np.isnan(val):
                    is_null = True
            except:
                pass

            if is_null:
                if ENABLE_NULL_FILTER:
                    return Filter.by_property(fname).is_none(True), f"{fname} is null"
                else:
                    # 当禁用 null 过滤时，尝试下一个字段
                    continue

            # 根据类型生成条件
            if ftype == FieldType.BOOL:
                return Filter.by_property(fname).equal(bool(val)), f"{fname} == {bool(val)}"

            elif ftype == FieldType.INT:
                val_int = int(val) if hasattr(val, "item") else int(val)
                strategies = []
                strategies.append((
                    Filter.by_property(fname).equal(val_int),
                    f"{fname} == {val_int}"
                ))
                strategies.append((
                    Filter.by_property(fname).greater_or_equal(val_int) &
                    Filter.by_property(fname).less_or_equal(val_int),
                    f"{fname} >= {val_int} AND <= {val_int}"
                ))
                return random.choice(strategies)

            elif ftype == FieldType.NUMBER:
                val_float = float(val)
                epsilon = 1e-5
                return (
                    Filter.by_property(fname).greater_than(val_float - epsilon) &
                    Filter.by_property(fname).less_than(val_float + epsilon),
                    f"{fname} > {val_float - epsilon} AND < {val_float + epsilon}"
                )

            elif ftype == FieldType.TEXT:
                val_str = str(val)
                return Filter.by_property(fname).equal(val_str), f'{fname} == "{val_str}"'

            elif ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY]:
                if not isinstance(val, list) or len(val) == 0:
                    if ENABLE_NULL_FILTER:
                        return Filter.by_property(fname).is_none(True), f"{fname} is null/empty"
                    else:
                        continue
                
                valid_items = [x for x in val if x is not None]
                # 对于 TEXT_ARRAY，过滤停用词
                if ftype == FieldType.TEXT_ARRAY:
                    valid_items = [x for x in valid_items if not is_stopword(x)]
                if valid_items:
                    target = random.choice(valid_items)
                    target = self._convert_to_native(target)
                    return Filter.by_property(fname).contains_any([target]), f"{fname} contains {target}"
                else:
                    # 如果所有项都被过滤掉（都是停用词），尝试下一个字段
                    continue

        # 如果所有字段都是 null，返回 None
        if ENABLE_NULL_FILTER:
            # 选一个字段返回 is not null（可能不匹配，但这是最后手段）
            fname = self.schema[0]["name"]
            return Filter.by_property(fname).is_none(False), f"{fname} is not null"
        else:
            return None, None


# --- 6. Main Execution Functions ---

def run(rounds=100, seed=None):
    """
    Oracle 模式主测试：Weaviate vs Pandas 对比
    """
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

    wm = WeaviateManager()
    wm.connect()
    try:
        wm.reset_collection(dm.schema_config)
        wm.insert(dm)

        # 2. 日志设置
        timestamp = int(time.time())
        log_filename = f"weaviate_fuzz_test_{timestamp}.log"
        print(f"\n📝 详细日志将写入: {log_filename}")
        print(f"   🔑 如需复现此次测试，运行: python weaviate_fuzz_oracle.py --seed {current_seed}")
        print(f"🚀 开始测试 (控制台仅显示失败案例)...")

        qg = OracleQueryGenerator(dm)
        collection = wm.client.collections.get(CLASS_NAME)
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

                # 生成查询
                depth = random.randint(1, 6)
                filter_obj = None
                for _ in range(10):
                    filter_obj, pandas_mask, expr_str = qg.gen_complex_expr(depth)
                    if filter_obj:
                        break

                if not filter_obj:
                    continue

                log_header = f"[Test {i}]"
                file_log(f"\n{log_header} Expr: {expr_str}")

                # Pandas 计算
                expected_ids = set(dm.df[pandas_mask.fillna(False)]["id"].values.tolist())

                try:
                    start_t = time.time()

                    # Weaviate 查询
                    response = collection.query.fetch_objects(
                        filters=filter_obj,
                        limit=N
                    )

                    actual_ids = set()
                    for obj in response.objects:
                        actual_ids.add(str(obj.uuid))

                    cost = (time.time() - start_t) * 1000

                    file_log(f"  Pandas: {len(expected_ids)} | Weaviate: {len(actual_ids)} | Time: {cost:.1f}ms")

                    if expected_ids == actual_ids:
                        file_log("  -> MATCH")
                    else:
                        print(f"\n❌ [Test {i}] MISMATCH!")
                        print(f"   Expr: {expr_str}")
                        print(f"   Expected: {len(expected_ids)} vs Actual: {len(actual_ids)}")

                        missing = expected_ids - actual_ids
                        extra = actual_ids - expected_ids
                        diff_msg = ""
                        if missing:
                            diff_msg += f"Missing IDs: {len(missing)} "
                        if extra:
                            diff_msg += f"Extra IDs: {len(extra)}"

                        print(f"   Diff: {diff_msg}")
                        print(f"   🔑 复现此bug: python weaviate_fuzz_oracle.py --seed {current_seed}\n")

                        file_log(f"  -> MISMATCH! {diff_msg}")
                        file_log(f"  -> REPRODUCTION SEED: {current_seed}")

                        if missing:
                            missing_rows = sample_rows(missing)
                            file_log(f"  Missing rows sample: {missing_rows}")
                        if extra:
                            extra_rows = sample_rows(extra)
                            file_log(f"  Extra rows sample: {extra_rows}")

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

                            search_res = collection.query.near_vector(
                                near_vector=q_vec,
                                filters=filter_obj,
                                limit=search_k
                            )

                            returned_ids = set(str(obj.uuid) for obj in search_res.objects)

                            if not returned_ids:
                                file_log("  VectorCheck: PASS (no ANN hits)")
                            elif returned_ids.issubset(expected_ids):
                                file_log(f"  VectorCheck: PASS ({len(returned_ids)} hits subset of scalar filter)")
                            else:
                                extra_vec = returned_ids - expected_ids
                                file_log(f"  VectorCheck: FAIL extra ids count: {len(extra_vec)}")
                                print(f"\n⚠️ [Test {i}] Vector subset violated: extras count {len(extra_vec)}")
                                failed_cases.append({
                                    "id": i,
                                    "expr": expr_str,
                                    "detail": f"VectorCheck extras count: {len(extra_vec)}"
                                })
                        except Exception as e:
                            file_log(f"  VectorCheck: ERROR {e}")

                    # 分页一致性测试 (10% 概率，且结果集大于 20)
                    # 注意: Weaviate 的 offset 分页在无排序时可能不稳定，这是已知行为
                    # 我们使用 cursor-based 分页来测试
                    if len(expected_ids) > 20 and random.random() < 0.1:
                        try:
                            page_size = random.randint(5, 15)
                            all_paged_ids = set()
                            cursor = None
                            max_pages = 10
                            
                            for page in range(max_pages):
                                if cursor:
                                    paged_res = collection.query.fetch_objects(
                                        filters=filter_obj,
                                        limit=page_size,
                                        after=cursor
                                    )
                                else:
                                    paged_res = collection.query.fetch_objects(
                                        filters=filter_obj,
                                        limit=page_size
                                    )
                                
                                if not paged_res.objects:
                                    break
                                
                                page_ids = set(str(obj.uuid) for obj in paged_res.objects)
                                
                                # 检查是否有重复
                                duplicates = all_paged_ids & page_ids
                                if duplicates:
                                    file_log(f"  CursorPaginationCheck: FAIL - duplicates found at page {page}")
                                    print(f"\n⚠️ [Test {i}] Cursor pagination duplicates at page {page}")
                                    failed_cases.append({
                                        "id": i,
                                        "expr": expr_str,
                                        "detail": f"Cursor pagination duplicates: {len(duplicates)} at page {page}"
                                    })
                                    break
                                
                                all_paged_ids.update(page_ids)
                                
                                # 获取最后一个 UUID 作为 cursor
                                cursor = paged_res.objects[-1].uuid
                                
                                if len(page_ids) < page_size:
                                    break
                            
                            # 验证分页结果是预期结果的子集
                            if all_paged_ids and not all_paged_ids.issubset(expected_ids):
                                extra_paged = all_paged_ids - expected_ids
                                file_log(f"  CursorPaginationCheck: FAIL - {len(extra_paged)} extra IDs")
                            else:
                                file_log(f"  CursorPaginationCheck: PASS ({len(all_paged_ids)} IDs via cursor pagination)")
                                
                        except Exception as e:
                            file_log(f"  CursorPaginationCheck: ERROR {e}")

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
            for case in failed_cases[:10]:  # 只显示前10个
                print(f"🔴 Case {case['id']}:")
                print(f"   Expr: {case['expr'][:100]}...")
                print(f"   Issue: {case['detail']}")
                if 'seed' in case:
                    print(f"   🔑 复现: python weaviate_fuzz_oracle.py --seed {case['seed']}")
                print("-" * 30)
            print(f"📄 请查看 {log_filename} 获取完整上下文。")
            print(f"🔑 全局复现命令: python weaviate_fuzz_oracle.py --seed {current_seed}")

    finally:
        wm.close()


def run_equivalence_mode(rounds=100, seed=None):
    """等价性测试模式"""
    if seed is not None:
        print(f"\n🔒 Equivalence 模式使用固定种子 {seed}")
        random.seed(seed)
        np.random.seed(seed)
    else:
        seed = random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)

    timestamp = int(time.time())
    log_filename = f"weaviate_equiv_test_{timestamp}.log"
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

    wm = WeaviateManager()
    wm.connect()
    
    try:
        wm.reset_collection(dm.schema_config)
        wm.insert(dm)

        qg = EquivalenceQueryGenerator(dm)
        collection = wm.client.collections.get(CLASS_NAME)
        failed_cases = []

        with open(log_filename, "w", encoding="utf-8") as f:
            def file_log(msg):
                f.write(msg + "\n")
                f.flush()

            file_log(f"Equivalence Test Started | Seed: {seed}")

            for i in range(rounds):
                print(f"\r⏳ Running Equivalence Test {i+1}/{rounds}...", end="", flush=True)

                # 生成基础查询
                depth = random.randint(1, 4)
                base_filter, base_mask, base_expr = qg.gen_complex_expr(depth)
                if not base_filter:
                    continue

                file_log(f"\n[Test {i}] Base: {base_expr}")

                try:
                    # 执行基础查询
                    base_result = collection.query.fetch_objects(
                        filters=base_filter,
                        limit=N
                    )
                    base_ids = set(str(obj.uuid) for obj in base_result.objects)
                    file_log(f"  Base result: {len(base_ids)} rows")

                    # 生成并测试变体
                    mutations = qg.mutate_filter(base_filter, base_expr)
                    
                    for mut in mutations:
                        mut_type = mut["type"]
                        mut_filter = mut["filter"]
                        mut_expr = mut["expr"]

                        try:
                            mut_result = collection.query.fetch_objects(
                                filters=mut_filter,
                                limit=N
                            )
                            mut_ids = set(str(obj.uuid) for obj in mut_result.objects)

                            if base_ids == mut_ids:
                                file_log(f"  {mut_type}: PASS ({len(mut_ids)} rows)")
                            else:
                                diff = len(base_ids.symmetric_difference(mut_ids))
                                file_log(f"  {mut_type}: FAIL (diff: {diff})")
                                print(f"\n❌ [Test {i}] Equivalence MISMATCH!")
                                print(f"   Type: {mut_type}")
                                print(f"   Base: {len(base_ids)} vs Mutant: {len(mut_ids)}")
                                
                                failed_cases.append({
                                    "id": i,
                                    "type": mut_type,
                                    "base_expr": base_expr,
                                    "mut_expr": mut_expr,
                                    "detail": f"Base: {len(base_ids)} vs Mutant: {len(mut_ids)}"
                                })
                        except Exception as e:
                            file_log(f"  {mut_type}: ERROR - {e}")

                except Exception as e:
                    file_log(f"  Base query ERROR: {e}")

        print("\n" + "="*60)
        if failed_cases:
            print(f"🚫 发现 {len(failed_cases)} 个等价性错误！请检查日志。")
        else:
            print(f"✅ 所有等价性测试通过。")
        print(f"📄 详细日志: {log_filename}")

    finally:
        wm.close()


def run_pqs_mode(rounds=100, seed=None):
    """PQS 模式测试：验证生成的查询必须命中指定行"""
    if seed is not None:
        print(f"\n🔒 PQS模式使用固定种子 {seed}")
        random.seed(seed)
        np.random.seed(seed)
    else:
        seed = random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)

    timestamp = int(time.time())
    log_filename = f"weaviate_pqs_test_{timestamp}.log"
    print("\n" + "="*60)
    print(f"🚀 启动 PQS (Pivot Query Synthesis) 模式测试")
    print(f"   Seed: {seed}")
    print(f"📄 详细日志将写入: {log_filename}")
    print("="*60)

    # 初始化
    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()

    wm = WeaviateManager()
    wm.connect()
    
    try:
        wm.reset_collection(dm.schema_config)
        wm.insert(dm)

        pqs_gen = PQSQueryGenerator(dm)
        collection = wm.client.collections.get(CLASS_NAME)
        errors = []
        successful_tests = 0
        skipped_tests = 0

        with open(log_filename, "w", encoding="utf-8") as f:
            def file_log(msg):
                f.write(msg + "\n")
                f.flush()

            file_log(f"PQS Test Started | Rounds: {rounds} | Seed: {seed}")
            file_log("=" * 80)

            for i in range(rounds):
                print(f"\r⏳ Running PQS Test {i+1}/{rounds}...", end="", flush=True)

                # 随机选择一行作为 pivot
                pivot_idx = random.randint(0, N - 1)
                pivot_row = dm.df.iloc[pivot_idx].to_dict()
                pivot_id = pivot_row["id"]

                # 生成针对这行的 PQS 查询
                depth = random.randint(2, 6)
                try:
                    pqs_filter, pqs_expr = pqs_gen.gen_pqs_filter(pivot_row, depth)
                except Exception as e:
                    file_log(f"[Test {i}] Filter generation error: {e}")
                    skipped_tests += 1
                    continue

                if not pqs_filter:
                    skipped_tests += 1
                    continue

                file_log(f"\n[Test {i}] Pivot ID: {pivot_id}")
                file_log(f"  Filter: {pqs_expr[:200]}...")

                try:
                    # 执行查询
                    result = collection.query.fetch_objects(
                        filters=pqs_filter,
                        limit=N
                    )
                    result_ids = set(str(obj.uuid) for obj in result.objects)

                    if pivot_id in result_ids:
                        file_log(f"  -> PASS (pivot found, total: {len(result_ids)})")
                        successful_tests += 1
                    else:
                        file_log(f"  -> FAIL! Pivot {pivot_id} NOT in results (got {len(result_ids)} rows)")
                        print(f"\n❌ [Test {i}] PQS FAIL - Pivot not found!")
                        print(f"   Pivot ID: {pivot_id}")
                        print(f"   Results count: {len(result_ids)}")
                        
                        errors.append({
                            "id": i,
                            "pivot_id": pivot_id,
                            "expr": pqs_expr,
                            "result_count": len(result_ids)
                        })

                except Exception as e:
                    file_log(f"  -> ERROR: {e}")
                    errors.append({
                        "id": i,
                        "pivot_id": pivot_id,
                        "expr": pqs_expr,
                        "error": str(e)
                    })

        print("\n" + "="*60)
        print(f"📊 测试统计:")
        print(f"   总轮数: {rounds}")
        print(f"   成功执行: {successful_tests}")
        print(f"   跳过: {skipped_tests}")
        print(f"   发现错误: {len(errors)}")
        print("="*60)
        
        if not errors:
            if successful_tests > 0:
                print(f"✅ PQS 测试全部通过！")
            else:
                print(f"⚠️ 没有成功执行的测试。")
        else:
            print(f"🚫 PQS 测试完成。发现 {len(errors)} 个潜在 Bug！")
            print(f"📄 详细数据已记录至日志: {log_filename}")

    finally:
        wm.close()


# --- Main Entry Point ---

if __name__ == "__main__":
    seed = None
    rounds = 500
    pqs_rounds = 500
    mode = "oracle"  # 默认模式

    # 解析命令行参数
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            if args[i] == "--seed" and i + 1 < len(args):
                seed = int(args[i + 1])
                i += 2
            elif args[i] == "--rounds" and i + 1 < len(args):
                rounds = int(args[i + 1])
                i += 2
            elif args[i] == "--pqs-rounds" and i + 1 < len(args):
                pqs_rounds = int(args[i + 1])
                i += 2
            elif args[i] == "--mode" and i + 1 < len(args):
                mode = args[i + 1]
                i += 2
            elif args[i] in ["-h", "--help"]:
                print("Usage: python weaviate_fuzz_oracle.py [options]")
                print("Options:")
                print("  --seed <int>       Set random seed for reproducibility")
                print("  --rounds <int>     Number of test rounds (default: 500)")
                print("  --pqs-rounds <int> Number of PQS test rounds (default: 500)")
                print("  --mode <str>       Test mode: oracle, equiv, pqs (default: oracle)")
                print("  -h, --help         Show this help message")
                exit(0)
            else:
                i += 1

    print("=" * 80)
    print(f"🚀 Weaviate Fuzz Oracle 启动")
    print(f"   模式: {mode}")
    print(f"   主测试轮数: {rounds}")
    print(f"   PQS测试轮数: {pqs_rounds}")
    print(f"   随机种子: {seed if seed else '(随机)'}")
    print("=" * 80)

    if mode == "equiv":
        run_equivalence_mode(rounds=rounds, seed=seed)
    elif mode == "pqs":
        run_pqs_mode(rounds=pqs_rounds, seed=seed)
    else:
        # 默认运行 Oracle 模式
        run(rounds=rounds, seed=seed)
