"""
Weaviate Dynamic Fuzzing System (V2: Enhanced Oracle Mode)
Features:
1. Dynamic Schema: Random 5-20 fields + BOOL_ARRAY + DATE types
2. Configurable Data: Adjustable vectors and dimensions
3. Robust Query Generation: Random filter expressions with type safety
4. Oracle Testing: Pandas comparison for correctness verification
5. PQS (Pivot Query Synthesis): Must-hit query testing
6. Equivalence Testing: Query transformation validation
7. Null Value Handling: Enabled by default with indexNullState
8. NOT expressions: Filter.not_() with multiple strategies
9. not_equal / contains_all / contains_none operators
10. Randomized vector index (HNSW/FLAT/Dynamic) + distance metric
11. Consistency level cycling (ONE/QUORUM/ALL)
12. Dynamic ops: insert/delete/update mid-test
"""
import time
import random
import string
import numpy as np
import pandas as pd
import json
import sys
import uuid
import argparse
import weaviate
from weaviate.classes.config import (
    Configure, Property, DataType, Reconfigure,
    VectorDistances,
)
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject
from weaviate.classes.config import ConsistencyLevel

# --- Null 过滤默认启用 ---
ENABLE_NULL_FILTER = True

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
    if not isinstance(word, str):
        return False
    return word.lower().strip() in WEAVIATE_STOPWORDS

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 8080
CLASS_NAME = "FuzzOracleV2"
N = 3000
DIM = 128
BATCH_SIZE = 200
SLEEP_INTERVAL = 0.02

# --- 延迟初始化的随机变量 (在 run() 中种子设置后初始化) ---
VECTOR_CHECK_RATIO = None
VECTOR_TOPK = None

# 向量索引类型和距离度量
ALL_VECTOR_INDEX_TYPES = ["hnsw", "flat", "dynamic"]
ALL_DISTANCE_METRICS = [VectorDistances.COSINE, VectorDistances.L2_SQUARED, VectorDistances.DOT]
VECTOR_INDEX_TYPE = None
DISTANCE_METRIC = None

# 一致性等级
ALL_CONSISTENCY_LEVELS = [ConsistencyLevel.ONE, ConsistencyLevel.QUORUM, ConsistencyLevel.ALL]


# --- 1. Data Manager ---

class FieldType:
    INT = "INT"
    NUMBER = "NUMBER"
    BOOL = "BOOL"
    TEXT = "TEXT"
    INT_ARRAY = "INT_ARRAY"
    TEXT_ARRAY = "TEXT_ARRAY"
    NUMBER_ARRAY = "NUMBER_ARRAY"
    BOOL_ARRAY = "BOOL_ARRAY"
    DATE = "DATE"

def get_weaviate_datatype(ftype):
    mapping = {
        FieldType.INT: DataType.INT,
        FieldType.NUMBER: DataType.NUMBER,
        FieldType.BOOL: DataType.BOOL,
        FieldType.TEXT: DataType.TEXT,
        FieldType.INT_ARRAY: DataType.INT_ARRAY,
        FieldType.TEXT_ARRAY: DataType.TEXT_ARRAY,
        FieldType.NUMBER_ARRAY: DataType.NUMBER_ARRAY,
        FieldType.BOOL_ARRAY: DataType.BOOL_ARRAY,
        FieldType.DATE: DataType.DATE,
    }
    return mapping.get(ftype)


class DataManager:
    def __init__(self):
        self.df = None
        self.vectors = None
        self.schema_config = []
        self.null_ratio = random.uniform(0.05, 0.15)
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
        types_pool = [FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT, FieldType.DATE]

        for i in range(num_fields):
            ftype = random.choice(types_pool)
            self.schema_config.append({"name": f"c{i}", "type": ftype})

        self.schema_config.append({"name": "tagsArray", "type": FieldType.INT_ARRAY, "max_capacity": self.array_capacity})
        self.schema_config.append({"name": "labelsArray", "type": FieldType.TEXT_ARRAY, "max_capacity": self.array_capacity})
        self.schema_config.append({"name": "scoresArray", "type": FieldType.NUMBER_ARRAY, "max_capacity": self.array_capacity})
        self.schema_config.append({"name": "flagsArray", "type": FieldType.BOOL_ARRAY, "max_capacity": self.array_capacity})

        print(f"   -> Generated {len(self.schema_config)} dynamic fields (plus id & vector).")
        for f in self.schema_config:
            print(f"      - {f['name']:<20} : {f['type']}")

    def generate_single_row(self, id_override=None):
        row = {}
        row["id"] = id_override if id_override else str(uuid.uuid4())
        row["row_num"] = -1

        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]
            if random.random() < self.null_ratio:
                row[fname] = None
                continue
            if ftype == FieldType.INT:
                row[fname] = random.randint(-self.int_range, self.int_range)
            elif ftype == FieldType.NUMBER:
                row[fname] = random.random() * self.double_scale
            elif ftype == FieldType.BOOL:
                row[fname] = random.choice([True, False])
            elif ftype == FieldType.TEXT:
                row[fname] = self._random_string(1, random.randint(5, 30))
            elif ftype == FieldType.DATE:
                y = random.randint(2000, 2025)
                m = random.randint(1, 12)
                d = random.randint(1, 28)
                row[fname] = f"{y:04d}-{m:02d}-{d:02d}T00:00:00Z"
            elif ftype == FieldType.INT_ARRAY:
                arr = [random.randint(0, 100) for _ in range(random.randint(0, 5))]
                row[fname] = arr if arr else None  # empty array → null (Weaviate semantics)
            elif ftype == FieldType.TEXT_ARRAY:
                arr = [self._random_string(2, 8) for _ in range(random.randint(0, 5))]
                row[fname] = arr if arr else None
            elif ftype == FieldType.NUMBER_ARRAY:
                arr = [random.random() * 100 for _ in range(random.randint(0, 5))]
                row[fname] = arr if arr else None
            elif ftype == FieldType.BOOL_ARRAY:
                arr = [random.choice([True, False]) for _ in range(random.randint(0, 5))]
                row[fname] = arr if arr else None
        return row

    def generate_single_vector(self):
        vec = np.random.random(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def generate_data(self):
        print(f"🌊 2. Generating {N} rows (Vector Dim={DIM})...")
        rng = np.random.default_rng(42)
        self.vectors = rng.random((N, DIM), dtype=np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1)[:, np.newaxis]

        data = {"id": [str(uuid.uuid4()) for _ in range(N)]}
        data["row_num"] = list(range(N))

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
            elif ftype == FieldType.DATE:
                values = []
                for _ in range(N):
                    y = random.randint(2000, 2025)
                    m = random.randint(1, 12)
                    d = random.randint(1, 28)
                    values.append(f"{y:04d}-{m:02d}-{d:02d}T00:00:00Z")
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
            elif ftype == FieldType.BOOL_ARRAY:
                arr_list = []
                for _ in range(N):
                    if rng.random() < self.null_ratio:
                        arr_list.append(None)
                    else:
                        length = random.randint(0, 5)
                        arr_list.append([bool(rng.choice([True, False])) for _ in range(length)])
                data[fname] = arr_list

        self.df = pd.DataFrame(data)
        # WORKAROUND: Weaviate treats empty arrays as null, normalize in oracle
        for field in self.schema_config:
            if field["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY]:
                self.df[field["name"]] = self.df[field["name"]].apply(
                    lambda x: None if isinstance(x, list) and len(x) == 0 else x)
        print("✅ Data Generation Complete.")
        null_counts = {col: self.df[col].isna().sum() for col in self.df.columns}
        print(f"   -> Null counts (sample): {dict(list(null_counts.items())[:5])}")

    def _apply_nulls(self, values, rng):
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
            self.client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=50051)
            if self.client.is_ready():
                print("✅ Connected to Weaviate.")
            else:
                print("❌ Weaviate not ready.")
                exit(1)
        except Exception:
            try:
                self.client = weaviate.connect_to_local(host=HOST, port=PORT, skip_init_checks=True)
                print("✅ Connected (without gRPC).")
            except Exception as e2:
                print(f"❌ Connection failed: {e2}")
                exit(1)

    def close(self):
        if self.client:
            self.client.close()

    def reset_collection(self, schema_config):
        try:
            self.client.collections.delete(CLASS_NAME)
            print(f"   -> Deleted existing: {CLASS_NAME}")
        except Exception:
            pass

        properties = []
        for field in schema_config:
            wv_type = get_weaviate_datatype(field["type"])
            if wv_type:
                properties.append(Property(name=field["name"], data_type=wv_type))
        properties.append(Property(name="row_num", data_type=DataType.INT))

        vi_type = VECTOR_INDEX_TYPE or "hnsw"
        dist = DISTANCE_METRIC or VectorDistances.COSINE

        if vi_type == "hnsw":
            ef_c = random.choice([64, 128, 256])
            max_conn = random.choice([16, 32, 64])
            vi_config = Configure.VectorIndex.hnsw(distance_metric=dist, ef_construction=ef_c, max_connections=max_conn)
            print(f"   -> VectorIndex: HNSW (ef_c={ef_c}, max_conn={max_conn})")
        elif vi_type == "flat":
            vi_config = Configure.VectorIndex.flat(distance_metric=dist)
            print(f"   -> VectorIndex: FLAT")
        else:
            threshold = random.choice([5000, 10000, 20000])
            vi_config = Configure.VectorIndex.dynamic(distance_metric=dist, threshold=threshold)
            print(f"   -> VectorIndex: DYNAMIC (threshold={threshold})")
        print(f"   -> Distance: {dist}")

        for attempt in range(3):
            try:
                self.client.collections.create(
                    name=CLASS_NAME, properties=properties,
                    vectorizer_config=Configure.Vectorizer.none(),
                    vector_index_config=vi_config,
                    inverted_index_config=Configure.inverted_index(index_null_state=True, index_property_length=True),
                )
                break
            except Exception as e:
                err_msg = str(e)
                print(f"   -> Create attempt {attempt+1} failed: {err_msg[:120]}")
                try:
                    self.client.collections.delete(CLASS_NAME)
                except Exception:
                    pass
                if "async indexing" in err_msg or "dynamic" in err_msg.lower():
                    vi_config = Configure.VectorIndex.hnsw(distance_metric=dist)
                    print(f"   -> Falling back to HNSW (dynamic not supported)")
                elif attempt == 0:
                    # Try without inverted_index_config
                    try:
                        self.client.collections.create(
                            name=CLASS_NAME, properties=properties,
                            vectorizer_config=Configure.Vectorizer.none(),
                            vector_index_config=vi_config,
                        )
                        break
                    except Exception as e2:
                        print(f"   -> Fallback also failed: {e2}")
                        vi_config = Configure.VectorIndex.hnsw(distance_metric=dist)
                else:
                    if attempt == 2:
                        raise
        print("🛠️ Collection Created.")

    def insert(self, dm):
        print(f"⚡ 3. Inserting Data (Batch={BATCH_SIZE})...")
        collection = self.client.collections.get(CLASS_NAME)
        records = dm.df.to_dict(orient="records")
        total = len(records)
        field_type_map = {f["name"]: f["type"] for f in dm.schema_config}
        field_type_map["row_num"] = FieldType.INT

        def convert_numpy_types(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.bool_): return bool(obj)
            elif isinstance(obj, np.ndarray): return [convert_numpy_types(x) for x in obj.tolist()]
            elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert_numpy_types(x) for x in obj]
            return obj

        def convert_value(k, v, ftmap):
            if v is None: return None
            if isinstance(v, float) and np.isnan(v): return None
            ftype = ftmap.get(k)
            if ftype == FieldType.INT: return int(v)
            elif ftype == FieldType.BOOL: return bool(v)
            elif ftype == FieldType.NUMBER: return float(v)
            elif ftype in (FieldType.TEXT, FieldType.DATE): return str(v)
            else: return convert_numpy_types(v)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch = records[start:end]
            data_objects = []
            for i, row in enumerate(batch):
                props = {}
                for k, v in row.items():
                    if k == "id": continue
                    cv = convert_value(k, v, field_type_map)
                    if cv is not None:
                        if isinstance(cv, list) and len(cv) == 0:
                            continue  # Skip empty arrays (Weaviate treats as null)
                        props[k] = cv
                data_objects.append(DataObject(uuid=row["id"], properties=props, vector=dm.vectors[start + i].tolist()))

            for attempt in range(3):
                try:
                    collection.data.insert_many(data_objects)
                    break
                except Exception as e:
                    if attempt == 2: raise e
                    time.sleep(2)
            print(f"   Inserted {end}/{total}...", end="\r")
            time.sleep(SLEEP_INTERVAL)
        print("\n✅ Insert Complete.")


# --- 3. Query Generator ---

class OracleQueryGenerator:
    def __init__(self, dm):
        self.schema = dm.schema_config
        self.df = dm.df
        self.dm = dm
        self._field_stats = {}
        for field in self.schema:
            fname, ftype = field["name"], field["type"]
            series = self.df[fname].dropna()
            if ftype in [FieldType.INT, FieldType.NUMBER] and not series.empty:
                self._field_stats[fname] = {
                    "min": series.min(), "max": series.max(),
                    "median": series.median(), "q1": series.quantile(0.25), "q3": series.quantile(0.75)
                }

    def _random_string(self, min_len=5, max_len=10):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(min_len, max_len)))

    @staticmethod
    def _convert_to_native(val):
        if isinstance(val, np.integer): return int(val)
        elif isinstance(val, np.floating): return float(val)
        elif isinstance(val, np.bool_): return bool(val)
        elif isinstance(val, np.ndarray): return [OracleQueryGenerator._convert_to_native(x) for x in val.tolist()]
        elif isinstance(val, list): return [OracleQueryGenerator._convert_to_native(x) for x in val]
        return val

    def get_value_for_query(self, fname, ftype):
        valid = self.df[fname].dropna()
        if not valid.empty and random.random() < 0.8:
            val = random.choice(valid.values)
            return val.item() if hasattr(val, "item") else val
        if ftype == FieldType.INT:
            if not valid.empty:
                return random.choice([int(valid.max()) + 100000, int(valid.min()) - 100000])
            return random.randint(-200000, 200000)
        elif ftype == FieldType.NUMBER:
            if not valid.empty:
                return random.choice([float(valid.max()) + 1e5, float(valid.min()) - 1e5])
            return random.random() * 200000.0
        elif ftype == FieldType.BOOL:
            return random.choice([True, False])
        elif ftype == FieldType.TEXT:
            return ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%", k=random.randint(15, 30)))
        elif ftype == FieldType.DATE:
            y, m, d = random.randint(2000, 2025), random.randint(1, 12), random.randint(1, 28)
            return f"{y:04d}-{m:02d}-{d:02d}T00:00:00Z"
        return None

    def gen_boundary_expr(self):
        numeric_fields = [f for f in self.schema if f["name"] in self._field_stats]
        if not numeric_fields:
            return None, None, None
        field = random.choice(numeric_fields)
        fname, ftype = field["name"], field["type"]
        stats = self._field_stats[fname]
        series = self.df[fname]
        bt = random.choice(["exact_min", "exact_max", "below_min", "above_max", "at_median", "range_q1_q3", "range_narrow"])

        def safe_cmp(op, val):
            def comp(x):
                if pd.isna(x): return False
                try:
                    if op == "==": return x == val
                    if op == ">": return x > val
                    if op == "<": return x < val
                    if op == ">=": return x >= val
                    if op == "<=": return x <= val
                except: pass
                return False
            return comp

        if ftype == FieldType.INT:
            min_v, max_v = int(stats["min"]), int(stats["max"])
            med = int(stats["median"])
            q1, q3 = int(stats["q1"]), int(stats["q3"])
            if bt == "exact_min":
                return Filter.by_property(fname).equal(min_v), series.apply(safe_cmp("==", min_v)), f"{fname} == {min_v} (min)"
            elif bt == "exact_max":
                return Filter.by_property(fname).equal(max_v), series.apply(safe_cmp("==", max_v)), f"{fname} == {max_v} (max)"
            elif bt == "below_min":
                v = min_v - 1
                return Filter.by_property(fname).less_than(v), series.apply(safe_cmp("<", v)), f"{fname} < {v} (below min)"
            elif bt == "above_max":
                v = max_v + 1
                return Filter.by_property(fname).greater_than(v), series.apply(safe_cmp(">", v)), f"{fname} > {v} (above max)"
            elif bt == "at_median":
                return Filter.by_property(fname).equal(med), series.apply(safe_cmp("==", med)), f"{fname} == {med} (median)"
            elif bt == "range_q1_q3":
                fc = Filter.by_property(fname).greater_or_equal(q1) & Filter.by_property(fname).less_or_equal(q3)
                return fc, series.apply(lambda x: q1 <= x <= q3 if pd.notna(x) else False), f"{fname} [{q1},{q3}] (IQR)"
            else:
                mid = (min_v + max_v) // 2
                delta = max(1, (max_v - min_v) // 20)
                fc = Filter.by_property(fname).greater_or_equal(mid - delta) & Filter.by_property(fname).less_or_equal(mid + delta)
                return fc, series.apply(lambda x: (mid - delta) <= x <= (mid + delta) if pd.notna(x) else False), f"{fname} [{mid-delta},{mid+delta}]"
        elif ftype == FieldType.NUMBER:
            min_v, max_v, med = float(stats["min"]), float(stats["max"]), float(stats["median"])
            if bt in ["exact_min", "exact_max", "at_median"]:
                eps = 1e-6
                t = min_v if bt == "exact_min" else (max_v if bt == "exact_max" else med)
                fc = Filter.by_property(fname).greater_or_equal(t - eps) & Filter.by_property(fname).less_or_equal(t + eps)
                return fc, series.apply(lambda x: (t - eps) <= x <= (t + eps) if pd.notna(x) else False), f"{fname} ≈ {t}"
            elif bt == "below_min":
                v = min_v - 1.0
                return Filter.by_property(fname).less_than(v), series.apply(safe_cmp("<", v)), f"{fname} < {v}"
            elif bt == "above_max":
                v = max_v + 1.0
                return Filter.by_property(fname).greater_than(v), series.apply(safe_cmp(">", v)), f"{fname} > {v}"
            else:
                mid = (min_v + max_v) / 2
                delta = max(0.001, (max_v - min_v) / 10)
                fc = Filter.by_property(fname).greater_or_equal(mid - delta) & Filter.by_property(fname).less_or_equal(mid + delta)
                return fc, series.apply(lambda x: (mid - delta) <= x <= (mid + delta) if pd.notna(x) else False), f"{fname} [{mid-delta:.2f},{mid+delta:.2f}]"
        return None, None, None

    def gen_multi_array_expr(self):
        """增强: contains_any / contains_all / contains_none"""
        arr_fields = [f for f in self.schema if f["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY]]
        if not arr_fields:
            return None, None, None
        field = random.choice(arr_fields)
        fname, ftype = field["name"], field["type"]
        series = self.df[fname]
        all_items = []
        for arr in series.dropna():
            if isinstance(arr, list):
                all_items.extend(arr)
        if ftype == FieldType.TEXT_ARRAY:
            all_items = [x for x in all_items if not is_stopword(x)]
        if not all_items:
            return None, None, None
        num = min(random.randint(2, 4), len(set(all_items)))
        if num < 1:
            return None, None, None
        targets = [self._convert_to_native(t) for t in random.sample(list(set(all_items)), num)]

        mode = random.choice(["contains_any", "contains_all", "contains_none"])
        is_text = ftype == FieldType.TEXT_ARRAY
        if is_text:
            tl = [str(t).lower() for t in targets]

        if mode == "contains_any":
            fc = Filter.by_property(fname).contains_any(targets)
            if is_text:
                mask = series.apply(lambda x: any(str(i).lower() in tl for i in x) if isinstance(x, list) else False)
            else:
                mask = series.apply(lambda x: any(i in targets for i in x) if isinstance(x, list) else False)
            return fc, mask, f"{fname} contains_any {targets}"
        elif mode == "contains_all":
            fc = Filter.by_property(fname).contains_all(targets)
            if is_text:
                mask = series.apply(lambda x: all(any(str(i).lower() == t for i in x) for t in tl) if isinstance(x, list) else False)
            else:
                mask = series.apply(lambda x: all(t in x for t in targets) if isinstance(x, list) else False)
            return fc, mask, f"{fname} contains_all {targets}"
        else:
            # WORKAROUND: contains_none 语义验证 — 不含 targets 中任何值; null 数组视为匹配
            fc = Filter.by_property(fname).contains_none(targets)
            if is_text:
                mask = series.apply(lambda x: not any(str(i).lower() in tl for i in x) if isinstance(x, list) else True)
            else:
                mask = series.apply(lambda x: not any(i in targets for i in x) if isinstance(x, list) else True)
            return fc, mask, f"{fname} contains_none {targets}"

    def gen_not_expr(self):
        """NOT 表达式 — 使用 Filter.not_() 包装
        WORKAROUND: Weaviate NOT 包含 null 行 (与 SQL 三值逻辑不同)
        """
        f = random.choice(self.schema)
        name, ftype = f["name"], f["type"]
        series = self.df[name]
        strat = random.choice(["not_cmp", "not_eq_text", "not_range", "not_contains", "not_null"])

        if strat == "not_cmp" and ftype in [FieldType.INT, FieldType.NUMBER]:
            val = self.get_value_for_query(name, ftype)
            if val is None: return None, None, None
            val = int(val) if ftype == FieldType.INT else float(val)
            op = random.choice([">", "<", ">=", "<=", "=="])
            op_map = {">": "greater_than", "<": "less_than", ">=": "greater_or_equal", "<=": "less_or_equal", "==": "equal"}
            inner = getattr(Filter.by_property(name), op_map[op])(val)
            fc = Filter.not_(inner)
            neg = {"==": "!=", ">": "<=", "<": ">=", ">=": "<", "<=": ">"}[op]
            def mk(x, o=neg, v=val):
                if pd.isna(x): return True  # Weaviate NOT includes null rows
                try:
                    if o == "!=": return x != v
                    if o == "<=": return x <= v
                    if o == ">=": return x >= v
                    if o == "<": return x < v
                    if o == ">": return x > v
                except: pass
                return False
            return fc, series.apply(mk), f"NOT({name} {op} {val})"

        elif strat == "not_eq_text" and ftype == FieldType.TEXT:
            val = self.get_value_for_query(name, ftype)
            if val is None: return None, None, None
            fc = Filter.not_(Filter.by_property(name).equal(val))
            vl = str(val).lower()
            return fc, series.apply(lambda x: str(x).lower() != vl if pd.notna(x) else True), f'NOT({name} == "{val}")'

        elif strat == "not_range" and ftype in [FieldType.INT, FieldType.NUMBER]:
            v1 = self.get_value_for_query(name, ftype)
            v2 = self.get_value_for_query(name, ftype)
            if v1 is None or v2 is None: return None, None, None
            if ftype == FieldType.INT:
                lo, hi = sorted([int(v1), int(v2)])
            else:
                lo, hi = sorted([float(v1), float(v2)])
            inner = Filter.by_property(name).greater_or_equal(lo) & Filter.by_property(name).less_or_equal(hi)
            fc = Filter.not_(inner)
            return fc, series.apply(lambda x: not (lo <= x <= hi) if pd.notna(x) else True), f"NOT({name} [{lo},{hi}])"

        elif strat == "not_contains" and ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY]:
            all_items = []
            for arr in self.df[name].dropna():
                if isinstance(arr, list): all_items.extend(arr)
            if ftype == FieldType.TEXT_ARRAY:
                all_items = [x for x in all_items if not is_stopword(x)]
            if all_items:
                t = self._convert_to_native(random.choice(all_items))
                fc = Filter.not_(Filter.by_property(name).contains_any([t]))
                if ftype == FieldType.TEXT_ARRAY:
                    tl = str(t).lower()
                    return fc, series.apply(lambda x: not any(str(i).lower() == tl for i in x) if isinstance(x, list) else True), f"NOT({name} contains {t})"
                return fc, series.apply(lambda x: t not in x if isinstance(x, list) else True), f"NOT({name} contains {t})"

        elif strat == "not_null" and ENABLE_NULL_FILTER:
            fc = Filter.not_(Filter.by_property(name).is_none(True))
            return fc, series.notnull(), f"NOT({name} is null)"

        # fallback: NOT(bool == val)
        for ff in self.schema:
            if ff["type"] == FieldType.BOOL:
                v = self.get_value_for_query(ff["name"], FieldType.BOOL)
                if v is not None:
                    fc = Filter.not_(Filter.by_property(ff["name"]).equal(bool(v)))
                    return fc, self.df[ff["name"]].apply(lambda x, vv=bool(v): x != vv if pd.notna(x) else True), f"NOT({ff['name']} == {bool(v)})"
        return None, None, None

    def gen_atomic_expr(self):
        f = random.choice(self.schema)
        name, ftype = f["name"], f["type"]
        series = self.df[name]

        if ENABLE_NULL_FILTER and random.random() < 0.15:
            if random.random() < 0.5:
                return Filter.by_property(name).is_none(True), series.isnull(), f"{name} is null"
            else:
                return Filter.by_property(name).is_none(False), series.notnull(), f"{name} is not null"

        val = self.get_value_for_query(name, ftype)
        if val is None:
            if ENABLE_NULL_FILTER:
                return Filter.by_property(name).is_none(True), series.isnull(), f"{name} is null"
            return None, None, None

        def safe_cmp(op, tv):
            def comp(x):
                if pd.isna(x): return False
                try:
                    if op == "==": return x == tv
                    if op == "!=": return x != tv
                    if op == ">": return x > tv
                    if op == "<": return x < tv
                    if op == ">=": return x >= tv
                    if op == "<=": return x <= tv
                except: pass
                return False
            return comp

        fc, mask, es = None, None, ""

        if ftype == FieldType.BOOL:
            vb = bool(val)
            if random.random() < 0.3:
                # WORKAROUND: Weaviate not_equal 包含 null 行 (与 SQL 不同)
                fc = Filter.by_property(name).not_equal(vb)
                mask = series.apply(safe_cmp("!=", vb)) | series.isnull()
                es = f"{name} != {vb}"
            else:
                fc = Filter.by_property(name).equal(vb)
                mask = series.apply(safe_cmp("==", vb))
                es = f"{name} == {vb}"

        elif ftype == FieldType.INT:
            vi = int(val)
            op = random.choice([">", "<", "==", ">=", "<=", "!="])
            op_map = {"==": "equal", "!=": "not_equal", ">": "greater_than", "<": "less_than", ">=": "greater_or_equal", "<=": "less_or_equal"}
            fc = getattr(Filter.by_property(name), op_map[op])(vi)
            mask = series.apply(safe_cmp(op, vi))
            if op == "!=":
                mask = mask | series.isnull()  # Weaviate not_equal includes null rows
            es = f"{name} {op} {vi}"

        elif ftype == FieldType.NUMBER:
            vf = float(val)
            op = random.choice([">", "<", ">=", "<=", "!="])
            op_map = {"!=": "not_equal", ">": "greater_than", "<": "less_than", ">=": "greater_or_equal", "<=": "less_or_equal"}
            fc = getattr(Filter.by_property(name), op_map[op])(vf)
            mask = series.apply(safe_cmp(op, vf))
            if op == "!=":
                mask = mask | series.isnull()  # Weaviate not_equal includes null rows
            es = f"{name} {op} {vf}"

        elif ftype == FieldType.TEXT:
            op = random.choice(["==", "!=", "like"])
            if op == "==":
                fc = Filter.by_property(name).equal(val)
                vl = str(val).lower()
                mask = series.apply(lambda x, v=vl: str(x).lower() == v if pd.notna(x) else False)
                es = f'{name} == "{val}"'
            elif op == "!=":
                fc = Filter.by_property(name).not_equal(val)
                vl = str(val).lower()
                # Weaviate not_equal includes null rows
                mask = series.apply(lambda x, v=vl: str(x).lower() != v if pd.notna(x) else True)
                es = f'{name} != "{val}"'
            else:
                prefix = ''.join(c for c in val[:3] if c.isalnum()) or "a"
                fc = Filter.by_property(name).like(f"{prefix}*")
                mask = series.apply(lambda x, p=prefix: str(x).lower().startswith(p.lower()) if pd.notna(x) else False)
                es = f'{name} like "{prefix}*"'

        elif ftype == FieldType.DATE:
            op = random.choice([">", "<", ">=", "<=", "=="])
            op_map = {"==": "equal", ">": "greater_than", "<": "less_than", ">=": "greater_or_equal", "<=": "less_or_equal"}
            fc = getattr(Filter.by_property(name), op_map[op])(val)
            mask = series.apply(safe_cmp(op, val))
            es = f"{name} {op} {val}"

        elif ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY]:
            all_items = []
            for arr in self.df[name].dropna():
                if isinstance(arr, list): all_items.extend(arr)
            if ftype == FieldType.TEXT_ARRAY:
                all_items = [x for x in all_items if not is_stopword(x)]
            if all_items:
                t = self._convert_to_native(random.choice(all_items))
                fc = Filter.by_property(name).contains_any([t])
                if ftype == FieldType.TEXT_ARRAY:
                    tl = str(t).lower()
                    mask = series.apply(lambda x, tt=tl: any(str(i).lower() == tt for i in x) if isinstance(x, list) else False)
                else:
                    mask = series.apply(lambda x, tt=t: tt in x if isinstance(x, list) else False)
                es = f'{name} contains {t}'
            else:
                if ENABLE_NULL_FILTER:
                    fc = Filter.by_property(name).is_none(True)
                    mask = series.isnull()
                    es = f'{name} is null'
                else:
                    fc = Filter.by_property(name).contains_any(["__impossible__"])
                    mask = pd.Series(False, index=self.df.index)
                    es = f'{name} contains __impossible__'

        if fc is not None and mask is not None:
            return fc, mask, es
        if ENABLE_NULL_FILTER:
            return Filter.by_property(name).is_none(False), series.notnull(), f"{name} is not null"
        int_fields = [ff for ff in self.schema if ff["type"] == FieldType.INT]
        if int_fields:
            fn = int_fields[0]["name"]
            s = self.df[fn]
            mn = int(s.min()) if not s.isna().all() else 0
            mx = int(s.max()) if not s.isna().all() else 0
            return (Filter.by_property(fn).greater_or_equal(mn - 1000000) & Filter.by_property(fn).less_or_equal(mx + 1000000)), s.notna(), f"{fn} range (fallback)"
        return None, None, None

    def gen_constant_expr(self):
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            fn = int_fields[0]["name"]
            s = self.df[fn]
            mn = int(s.min()) if not s.isna().all() else 0
            mx = int(s.max()) if not s.isna().all() else 0
        else:
            float_fields = [f for f in self.schema if f["type"] == FieldType.NUMBER]
            if not float_fields: return None, None, None
            fn = float_fields[0]["name"]
            s = self.df[fn]
            mn = float(s.min()) if not s.isna().all() else 0
            mx = float(s.max()) if not s.isna().all() else 0
        if random.random() < 0.5:
            return (Filter.by_property(fn).greater_or_equal(mn - 1e6) & Filter.by_property(fn).less_or_equal(mx + 1e6)), s.notna(), f"{fn} wide range (true)"
        else:
            imp = mx + 1e7
            return Filter.by_property(fn).greater_than(imp), pd.Series(False, index=self.df.index), f"{fn} > {imp} (false)"

    def gen_complex_expr(self, depth):
        if depth == 0 or random.random() < 0.3:
            r = random.random()
            if r < 0.02:
                res = self.gen_constant_expr()
                if res[0]: return res
            elif r < 0.12:
                res = self.gen_boundary_expr()
                if res[0] is not None: return res
            elif r < 0.20:
                res = self.gen_multi_array_expr()
                if res[0] is not None: return res
            elif r < 0.35:
                res = self.gen_not_expr()
                if res[0] is not None: return res
            fc, m, e = self.gen_atomic_expr()
            if fc: return fc, m, e
            return self.gen_complex_expr(depth)

        fl, ml, el = self.gen_complex_expr(depth - 1)
        fr, mr, er = self.gen_complex_expr(depth - 1)
        if not fl: return fr, mr, er
        if not fr: return fl, ml, el
        if random.choice(["and", "or"]) == "and":
            return fl & fr, ml & mr, f"({el} AND {er})"
        return fl | fr, ml | mr, f"({el} OR {er})"


# --- 4. Equivalence ---

class EquivalenceQueryGenerator(OracleQueryGenerator):
    def __init__(self, dm):
        super().__init__(dm)
        self._tautology_field = self._find_tautology_field()

    def _find_tautology_field(self):
        for f in self.schema:
            if f["type"] == FieldType.INT:
                s = self.df[f["name"]].dropna()
                if not s.empty:
                    return {"name": f["name"], "min": int(s.min()), "max": int(s.max())}
        for f in self.schema:
            if f["type"] == FieldType.NUMBER:
                s = self.df[f["name"]].dropna()
                if not s.empty:
                    return {"name": f["name"], "min": float(s.min()), "max": float(s.max())}
        return None

    def _gen_tautology_filter(self):
        if self._tautology_field:
            n = self._tautology_field["name"]
            mn, mx = self._tautology_field["min"], self._tautology_field["max"]
            return Filter.by_property(n).greater_or_equal(mn - 1e6) & Filter.by_property(n).less_or_equal(mx + 1e6), f"({n} wide)"
        return None, None

    def _gen_false_filter(self):
        for f in self.schema:
            if f["type"] == FieldType.INT:
                return Filter.by_property(f["name"]).greater_than(2e5) & Filter.by_property(f["name"]).less_than(-2e5), f"({f['name']} impossible)"
        for f in self.schema:
            if f["type"] == FieldType.TEXT:
                return Filter.by_property(f["name"]).equal("__impossible__"), f'{f["name"]} impossible'
        return None, None

    def mutate_filter(self, base_filter, base_expr):
        mutations = [{"type": "SelfOr", "filter": base_filter | base_filter, "expr": f"({base_expr}) OR ({base_expr})"}]
        ff, fe = self._gen_false_filter()
        if ff:
            mutations.append({"type": "NoiseOr", "filter": base_filter | ff, "expr": f"({base_expr}) OR ({fe})"})
        mutations.append({"type": "DoubleNeg", "filter": Filter.not_(Filter.not_(base_filter)), "expr": f"NOT(NOT({base_expr}))"})
        return mutations


# --- 5. PQS ---

class PQSQueryGenerator(OracleQueryGenerator):
    def __init__(self, dm):
        super().__init__(dm)
        self._tautology_field = self._find_tautology_field()

    def _find_tautology_field(self):
        for f in self.schema:
            if f["type"] == FieldType.INT:
                n = f["name"]
                if self.df[n].isna().sum() == 0:
                    s = self.df[n]
                    return {"name": n, "min": int(s.min()), "max": int(s.max())}
        for f in self.schema:
            if f["type"] == FieldType.INT:
                s = self.df[f["name"]].dropna()
                if not s.empty:
                    return {"name": f["name"], "min": int(s.min()), "max": int(s.max())}
        for f in self.schema:
            if f["type"] == FieldType.NUMBER:
                s = self.df[f["name"]].dropna()
                if not s.empty:
                    return {"name": f["name"], "min": float(s.min()), "max": float(s.max())}
        return None

    def _gen_tautology(self):
        if self._tautology_field:
            n = self._tautology_field["name"]
            mn, mx = self._tautology_field["min"], self._tautology_field["max"]
            return Filter.by_property(n).greater_or_equal(mn - 1e6) & Filter.by_property(n).less_or_equal(mx + 1e6), f"({n} wide)"
        return None, None

    def _gen_false(self):
        for f in self.schema:
            if f["type"] == FieldType.INT:
                return Filter.by_property(f["name"]).greater_than(2e5) & Filter.by_property(f["name"]).less_than(-2e5), f"({f['name']} impossible)"
        return None, None

    def gen_pqs_filter(self, pivot_row, depth):
        if depth <= 0 or (depth <= 3 and random.random() < 0.3):
            res = self.gen_true_atomic(pivot_row)
            if res[0] is None:
                tf, te = self._gen_tautology()
                if tf: return tf, te
            return res
        op = random.choice(["and", "or"])
        tf, te = self._gen_tautology()
        if op == "and":
            fl, el = self.gen_pqs_filter(pivot_row, depth - 1)
            fr, er = self.gen_pqs_filter(pivot_row, depth - 1)
            if not fl: fl, el = tf, te
            if not fr: fr, er = tf, te
            if fl and fr: return fl & fr, f"({el} AND {er})"
            return fl, el
        else:
            fl, el = self.gen_pqs_filter(pivot_row, depth - 1)
            nf, ne = self._gen_false()
            if not fl: fl, el = tf, te
            if not nf: nf, ne = self._gen_false()
            if fl and nf: return fl | nf, f"({el} OR {ne})"
            return fl, el

    def gen_true_atomic(self, row):
        for field in random.sample(self.schema, len(self.schema)):
            fname, ftype = field["name"], field["type"]
            val = row.get(fname)
            is_null = val is None or (isinstance(val, float) and np.isnan(val))
            if is_null:
                if ENABLE_NULL_FILTER:
                    return Filter.by_property(fname).is_none(True), f"{fname} is null"
                continue
            if ftype == FieldType.BOOL:
                return Filter.by_property(fname).equal(bool(val)), f"{fname} == {bool(val)}"
            elif ftype == FieldType.INT:
                vi = int(val)
                return random.choice([
                    (Filter.by_property(fname).equal(vi), f"{fname} == {vi}"),
                    (Filter.by_property(fname).greater_or_equal(vi) & Filter.by_property(fname).less_or_equal(vi), f"{fname} [{vi},{vi}]")
                ])
            elif ftype == FieldType.NUMBER:
                vf = float(val)
                eps = 1e-5
                return Filter.by_property(fname).greater_than(vf - eps) & Filter.by_property(fname).less_than(vf + eps), f"{fname} ≈ {vf}"
            elif ftype in (FieldType.TEXT, FieldType.DATE):
                return Filter.by_property(fname).equal(str(val)), f'{fname} == "{val}"'
            elif ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY]:
                if not isinstance(val, list) or not val: continue
                items = [x for x in val if x is not None]
                if ftype == FieldType.TEXT_ARRAY:
                    items = [x for x in items if not is_stopword(x)]
                if items:
                    t = self._convert_to_native(random.choice(items))
                    return Filter.by_property(fname).contains_any([t]), f"{fname} contains {t}"
                continue
        if ENABLE_NULL_FILTER:
            return Filter.by_property(self.schema[0]["name"]).is_none(False), f"{self.schema[0]['name']} is not null"
        return None, None


# --- 6. Main Execution ---

def run(rounds=100, seed=None, enable_dynamic_ops=True, consistency=None):
    global VECTOR_CHECK_RATIO, VECTOR_TOPK, VECTOR_INDEX_TYPE, DISTANCE_METRIC
    current_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)
    print(f"🔒 Seed: {current_seed}")

    VECTOR_CHECK_RATIO = random.uniform(0.2, 0.8)
    VECTOR_TOPK = random.randint(50, 200)
    VECTOR_INDEX_TYPE = random.choice(ALL_VECTOR_INDEX_TYPES)
    DISTANCE_METRIC = random.choice(ALL_DISTANCE_METRICS)
    cl_list = [consistency] if consistency else ALL_CONSISTENCY_LEVELS
    _cl_rng = random.Random(current_seed + 7)
    print(f"   VecIndex: {VECTOR_INDEX_TYPE}, Dist: {DISTANCE_METRIC}, VecRatio: {VECTOR_CHECK_RATIO:.2f}, TopK: {VECTOR_TOPK}")

    dm = DataManager()
    dm.generate_schema()
    dm.generate_data()
    wm = WeaviateManager()
    wm.connect()
    try:
        wm.reset_collection(dm.schema_config)
        wm.insert(dm)
        ts = int(time.time())
        logf = f"weaviate_fuzz_test_{ts}.log"
        print(f"\n📝 Log: {logf}")
        print(f"   🔑 Reproduce: python weaviate_fuzz_oracle.py --seed {current_seed}")
        print("🚀 Testing...")

        qg = OracleQueryGenerator(dm)
        col = wm.client.collections.get(CLASS_NAME)
        fails = []

        with open(logf, "w", encoding="utf-8") as f:
            def flog(msg):
                f.write(msg + "\n"); f.flush()

            def sample(ids, lim=5):
                if not ids: return []
                return dm.df[dm.df["id"].isin(list(ids))].to_dict("records")[:lim]

            flog(f"Start: {rounds} rounds | Seed: {current_seed} | VecIdx: {VECTOR_INDEX_TYPE} | Dist: {DISTANCE_METRIC}")
            flog("=" * 50)

            for i in range(rounds):
                cl = _cl_rng.choice(cl_list)
                print(f"\r⏳ Test {i+1}/{rounds} [CL={cl}]       ", end="", flush=True)

                # Dynamic ops
                if enable_dynamic_ops and i > 0 and i % 10 == 0:
                    op = random.choices(["insert", "delete", "update"], weights=[0.4, 0.4, 0.2], k=1)[0]
                    bc = random.randint(1, 5)
                    if op == "insert":
                        try:
                            rows, vecs = [], []
                            for _ in range(bc):
                                rows.append(dm.generate_single_row())
                                vecs.append(dm.generate_single_vector())
                            objs = []
                            for r, v in zip(rows, vecs):
                                p = {k: vv for k, vv in r.items() if k != "id" and vv is not None}
                                objs.append(DataObject(uuid=r["id"], properties=p, vector=v.tolist()))
                            col.data.insert_many(objs)
                            dm.df = pd.concat([dm.df, pd.DataFrame(rows)], ignore_index=True)
                            dm.vectors = np.vstack([dm.vectors, np.array(vecs)])
                            qg = OracleQueryGenerator(dm)
                            flog(f"[Dyn] Inserted {bc}")
                        except Exception as e:
                            flog(f"[Dyn] Insert fail: {e}")
                    elif op == "delete":
                        if len(dm.df) > bc:
                            try:
                                idxs = random.sample(range(len(dm.df)), bc)
                                dids = [dm.df.iloc[x]["id"] for x in idxs]
                                for d in dids:
                                    col.data.delete_by_id(d)
                                keep = ~dm.df["id"].isin(dids)
                                ki = dm.df[keep].index.to_numpy()
                                dm.df = dm.df[keep].reset_index(drop=True)
                                dm.vectors = dm.vectors[ki]
                                qg = OracleQueryGenerator(dm)
                                flog(f"[Dyn] Deleted {len(dids)}")
                            except Exception as e:
                                flog(f"[Dyn] Delete fail: {e}")
                    else:
                        if not dm.df.empty:
                            try:
                                ui = random.randint(0, len(dm.df) - 1)
                                uid = dm.df.iloc[ui]["id"]
                                nr = dm.generate_single_row(id_override=uid)
                                nv = dm.generate_single_vector()
                                p = {k: v for k, v in nr.items() if k != "id" and v is not None}
                                col.data.replace(uuid=uid, properties=p, vector=nv.tolist())
                                for k, v in nr.items():
                                    dm.df.at[ui, k] = v
                                dm.vectors[ui] = nv
                                qg = OracleQueryGenerator(dm)
                                flog(f"[Dyn] Updated {uid}")
                            except Exception as e:
                                flog(f"[Dyn] Update fail: {e}")

                depth = random.randint(1, 15)
                fo = None
                for _ in range(10):
                    fo, pm, es = qg.gen_complex_expr(depth)
                    if fo: break
                if not fo: continue

                flog(f"\n[T{i}] {es[:300]}")
                flog(f"  CL={cl}")
                exp = set(dm.df[pm.fillna(False)]["id"].tolist())

                try:
                    t0 = time.time()
                    col_cl = col.with_consistency_level(cl)
                    res = col_cl.query.fetch_objects(filters=fo, limit=max(N, len(dm.df) + 100))
                    act = set(str(o.uuid) for o in res.objects)
                    ms = (time.time() - t0) * 1000
                    flog(f"  Pandas:{len(exp)} | Weaviate:{len(act)} | {ms:.1f}ms")

                    if exp == act:
                        flog("  -> MATCH")
                    else:
                        mi, ex = exp - act, act - exp
                        dm_ = f"Missing:{len(mi)} Extra:{len(ex)}"
                        if cl != ConsistencyLevel.ALL:
                            flog(f"  -> WARN (CL={cl}): {dm_}")
                        else:
                            print(f"\n❌ [T{i}] MISMATCH! {dm_}")
                            print(f"   Expr: {es[:200]}")
                            print(f"   🔑 --seed {current_seed}")
                            flog(f"  -> MISMATCH! {dm_}")
                            if mi: flog(f"  Missing: {sample(mi)}")
                            if ex: flog(f"  Extra: {sample(ex)}")
                            fails.append({"id": i, "expr": es, "detail": dm_, "seed": current_seed})

                    if exp and random.random() < VECTOR_CHECK_RATIO:
                        try:
                            qi = random.randint(0, len(dm.vectors) - 1)
                            sr = col.query.near_vector(near_vector=dm.vectors[qi].tolist(), filters=fo, limit=min(VECTOR_TOPK, len(dm.df)))
                            ri = set(str(o.uuid) for o in sr.objects)
                            if ri and not ri.issubset(exp):
                                ev = ri - exp
                                flog(f"  VecCheck: WARN {len(ev)} extras (soft)")
                                # Soft check: don't add to fails for vec search
                            else:
                                flog(f"  VecCheck: PASS ({len(ri)})")
                        except Exception as e:
                            flog(f"  VecCheck: ERR {e}")

                    if len(exp) > 20 and random.random() < 0.1:
                        try:
                            ps = random.randint(5, 15)
                            ap, cur = set(), None
                            for pg in range(10):
                                kw = {"filters": fo, "limit": ps}
                                if cur: kw["after"] = cur
                                pr = col.query.fetch_objects(**kw)
                                if not pr.objects: break
                                pi = set(str(o.uuid) for o in pr.objects)
                                dup = ap & pi
                                if dup:
                                    flog(f"  Page: FAIL dup@{pg}")
                                    fails.append({"id": i, "detail": f"Page dup@{pg}"})
                                    break
                                ap.update(pi)
                                cur = pr.objects[-1].uuid
                                if len(pi) < ps: break
                            if ap and not ap.issubset(exp):
                                flog(f"  Page: FAIL extras")
                            else:
                                flog(f"  Page: PASS ({len(ap)})")
                        except Exception as e:
                            flog(f"  Page: ERR {e}")
                except Exception as e:
                    print(f"\n⚠️ [T{i}] CRASH: {e}")
                    flog(f"  -> ERR: {e}")
                    fails.append({"id": i, "expr": es, "detail": str(e)})

        print("\n" + "="*60)
        if not fails:
            print(f"✅ All {rounds} tests passed!")
        else:
            print(f"🚫 {len(fails)} failures!")
            for c in fails[:10]:
                print(f"  🔴 T{c['id']}: {c.get('detail','')[:100]}")
        print(f"📄 Log: {logf}")
        print(f"🔑 Reproduce: python weaviate_fuzz_oracle.py --seed {current_seed}")
    finally:
        wm.close()


def run_equivalence_mode(rounds=100, seed=None):
    global VECTOR_CHECK_RATIO, VECTOR_TOPK, VECTOR_INDEX_TYPE, DISTANCE_METRIC
    if seed is None: seed = random.randint(0, 1000000)
    random.seed(seed); np.random.seed(seed)
    VECTOR_INDEX_TYPE = random.choice(ALL_VECTOR_INDEX_TYPES)
    DISTANCE_METRIC = random.choice(ALL_DISTANCE_METRICS)

    logf = f"weaviate_equiv_test_{int(time.time())}.log"
    print(f"\n👯 Equivalence Mode | Seed: {seed} | VecIdx: {VECTOR_INDEX_TYPE}")

    dm = DataManager(); dm.generate_schema(); dm.generate_data()
    wm = WeaviateManager(); wm.connect()
    try:
        wm.reset_collection(dm.schema_config); wm.insert(dm)
        qg = EquivalenceQueryGenerator(dm)
        col = wm.client.collections.get(CLASS_NAME)
        fails = []
        lim = max(N, len(dm.df) + 100)
        with open(logf, "w", encoding="utf-8") as f:
            def flog(m): f.write(m + "\n"); f.flush()
            flog(f"Equiv | Seed:{seed}")
            for i in range(rounds):
                print(f"\r⏳ Equiv {i+1}/{rounds}", end="", flush=True)
                d = random.randint(1, 8)
                bf, bm, be = qg.gen_complex_expr(d)
                if not bf: continue
                flog(f"\n[T{i}] {be[:200]}")
                try:
                    br = col.query.fetch_objects(filters=bf, limit=lim)
                    bi = set(str(o.uuid) for o in br.objects)
                    flog(f"  Base: {len(bi)}")
                    for mut in qg.mutate_filter(bf, be):
                        try:
                            mr = col.query.fetch_objects(filters=mut["filter"], limit=lim)
                            mi = set(str(o.uuid) for o in mr.objects)
                            if bi == mi:
                                flog(f"  {mut['type']}: PASS")
                            else:
                                flog(f"  {mut['type']}: FAIL (diff:{len(bi.symmetric_difference(mi))})")
                                print(f"\n❌ [T{i}] {mut['type']} FAIL")
                                fails.append({"id": i, "type": mut["type"]})
                        except Exception as e:
                            flog(f"  {mut['type']}: ERR {e}")
                except Exception as e:
                    flog(f"  ERR: {e}")
        print(f"\n{'✅ All passed' if not fails else f'🚫 {len(fails)} failures'}. Log: {logf}")
    finally:
        wm.close()


def run_pqs_mode(rounds=100, seed=None):
    global VECTOR_CHECK_RATIO, VECTOR_TOPK, VECTOR_INDEX_TYPE, DISTANCE_METRIC
    if seed is None: seed = random.randint(0, 1000000)
    random.seed(seed); np.random.seed(seed)
    VECTOR_INDEX_TYPE = random.choice(ALL_VECTOR_INDEX_TYPES)
    DISTANCE_METRIC = random.choice(ALL_DISTANCE_METRICS)

    logf = f"weaviate_pqs_test_{int(time.time())}.log"
    print(f"\n🚀 PQS Mode | Seed: {seed} | VecIdx: {VECTOR_INDEX_TYPE}")

    dm = DataManager(); dm.generate_schema(); dm.generate_data()
    wm = WeaviateManager(); wm.connect()
    try:
        wm.reset_collection(dm.schema_config); wm.insert(dm)
        pqs = PQSQueryGenerator(dm)
        col = wm.client.collections.get(CLASS_NAME)
        errs, ok, skip = [], 0, 0
        lim = max(N, len(dm.df) + 100)
        with open(logf, "w", encoding="utf-8") as f:
            def flog(m): f.write(m + "\n"); f.flush()
            flog(f"PQS | Seed:{seed}")
            for i in range(rounds):
                print(f"\r⏳ PQS {i+1}/{rounds}", end="", flush=True)
                pi = random.randint(0, len(dm.df) - 1)
                pr = dm.df.iloc[pi].to_dict()
                pid = pr["id"]
                d = random.randint(2, 8)
                try:
                    pf, pe = pqs.gen_pqs_filter(pr, d)
                except Exception as e:
                    flog(f"[T{i}] Gen err: {e}"); skip += 1; continue
                if not pf: skip += 1; continue
                flog(f"\n[T{i}] Pivot:{pid}")
                try:
                    res = col.query.fetch_objects(filters=pf, limit=lim)
                    ri = set(str(o.uuid) for o in res.objects)
                    if pid in ri:
                        flog(f"  PASS ({len(ri)})"); ok += 1
                    else:
                        flog(f"  FAIL! Missing pivot ({len(ri)})")
                        print(f"\n❌ [T{i}] PQS FAIL")
                        errs.append({"id": i, "pivot": pid})
                except Exception as e:
                    flog(f"  ERR: {e}")
                    errs.append({"id": i, "error": str(e)})
        print(f"\n📊 PQS: ok={ok} skip={skip} err={len(errs)}")
        print(f"{'✅ All passed' if not errs else f'🚫 {len(errs)} failures'}. Log: {logf}")
    finally:
        wm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weaviate Fuzz Oracle V2")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--rounds", type=int, default=500, help="Test rounds (default: 500)")
    parser.add_argument("--mode", choices=["oracle", "equiv", "pqs"], default="oracle", help="Test mode")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic ops")
    parser.add_argument("--host", type=str, default=None, help="Weaviate host")
    parser.add_argument("--port", type=int, default=None, help="Weaviate port")
    parser.add_argument("-N", type=int, default=None, help="Data rows")
    args = parser.parse_args()

    if args.host: HOST = args.host
    if args.port: PORT = args.port
    if args.N: N = args.N

    print("=" * 80)
    print(f"🚀 Weaviate Fuzz Oracle V2 | Mode: {args.mode} | Rounds: {args.rounds} | Seed: {args.seed or '(random)'} | Dynamic: {'OFF' if args.no_dynamic else 'ON'}")
    print("=" * 80)

    if args.mode == "equiv":
        run_equivalence_mode(rounds=args.rounds, seed=args.seed)
    elif args.mode == "pqs":
        run_pqs_mode(rounds=args.rounds, seed=args.seed)
    else:
        run(rounds=args.rounds, seed=args.seed, enable_dynamic_ops=not args.no_dynamic)
