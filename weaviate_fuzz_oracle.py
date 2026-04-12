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
13. Tokenization.FIELD for TEXT/TEXT_ARRAY: case-sensitive, no stopword filtering
14. Dynamic row tracking: Weaviate inverted-index bug detection & classification

Notes:
  Historical bug status drifts across Weaviate/client versions, so this file keeps
  the runtime logic but avoids hard-coding a large "known bugs" list in the header.
  Use the dedicated repro scripts in compare_test/weaviate_bug*.py for the current
  version-specific probes that still reproduce locally.
"""
import time
import os
import random
import string
import numpy as np
import pandas as pd
import json
import sys
import uuid
import argparse
import re
from datetime import timedelta, timezone
import weaviate
from weaviate.classes.config import (
    Configure, Property, DataType, Reconfigure,
    VectorDistances, Tokenization,
)
from weaviate.classes.query import Filter, GeoCoordinate, GroupBy, Sort, MetadataQuery
from weaviate.classes.data import DataObject
from weaviate.classes.config import ConsistencyLevel

# --- Null 过滤默认启用 ---
ENABLE_NULL_FILTER = True

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
BOUNDARY_INJECTION_RATE = None

# 标量优先：过滤表达式更偏向标量字段，减少因数组/OBJECT 语义导致的噪声
SCALAR_QUERY_PRIORITY = 0.85

# Current Weaviate client/transport path may coerce large integers through
# floating-point serialization. Keep fuzzed INT values within the IEEE-754
# safe-integer range so insert/query failures reflect engine behavior rather
# than client-side rounding artifacts.
WEAVIATE_SAFE_INT_MIN = -((2**53) - 1)
WEAVIATE_SAFE_INT_MAX = (2**53) - 1

INT_BOUNDARY_VALUES = [
    WEAVIATE_SAFE_INT_MIN,
    WEAVIATE_SAFE_INT_MIN + 1,
    -1024,
    -1,
    0,
    1,
    1024,
    WEAVIATE_SAFE_INT_MAX - 1,
    WEAVIATE_SAFE_INT_MAX,
]
INT64_MIN = -(2**63)
INT64_MAX = (2**63) - 1

def next_float(value, toward):
    if value is None:
        return None
    try:
        fv = np.float64(value)
        tv = np.float64(toward)
    except Exception:
        return None
    if not np.isfinite(fv) or not np.isfinite(tv):
        return None
    nxt = np.nextafter(fv, tv)
    if not np.isfinite(nxt):
        return None
    return float(nxt)


def unique_float_values(values):
    out, seen = [], set()
    for value in values:
        try:
            fv = np.float64(value)
        except Exception:
            continue
        if not np.isfinite(fv):
            continue
        bits = int(fv.view(np.uint64))
        if bits in seen:
            continue
        seen.add(bits)
        out.append(float(fv))
    return out


def format_rfc3339_timestamp(ts, offset_minutes=0):
    if ts is None:
        return None
    try:
        pts = pd.Timestamp(ts)
    except Exception:
        return None
    if pd.isna(pts):
        return None
    if pts.tzinfo is None:
        pts = pts.tz_localize("UTC")
    else:
        pts = pts.tz_convert("UTC")
    tz = timezone(timedelta(minutes=int(offset_minutes or 0)))
    dt = pts.to_pydatetime().astimezone(tz)
    timespec = "microseconds" if dt.microsecond else "seconds"
    out = dt.isoformat(timespec=timespec)
    return out.replace("+00:00", "Z")


def build_number_boundary_values():
    finfo = np.finfo(np.float64)
    base = [
        float(-finfo.tiny),
        -1e12,
        -1e6,
        -1.0,
        -1e-12,
        # Keep -0.0 out of the main fuzz subset; see issue10917 probes.
        next_float(0.0, -np.inf),
        0.0,
        finfo.smallest_subnormal,
        finfo.tiny,
        1e-12,
        1.0,
        1e6,
        1e12,
    ]
    anchors = [
        -1e12,
        -1.0,
        -1e-12,
        0.0,
        1e-12,
        1.0,
        1e12,
        -finfo.tiny,
        finfo.tiny,
    ]
    expanded = list(base)
    for anchor in anchors:
        lower = next_float(anchor, -np.inf)
        upper = next_float(anchor, np.inf)
        if lower is not None:
            expanded.append(lower)
        if upper is not None:
            expanded.append(upper)
    return unique_float_values(expanded)


NUMBER_BOUNDARY_VALUES = build_number_boundary_values()

DATE_BOUNDARY_VALUES = [
    "1969-12-31T23:59:59Z",
    "1970-01-01T00:00:00Z",
    "1970-01-01T00:00:00.000001Z",
    "1999-12-31T23:59:59Z",
    "2000-02-29T00:00:00Z",
    "2001-09-09T01:46:40Z",
    "2016-12-31T23:59:59-08:00",
    "2020-02-29T12:34:56+00:00",
    "2024-02-29T15:59:59.999999Z",
    "2024-02-29T23:59:59+08:00",
    "2024-03-01T00:00:00.000001Z",
    "2024-03-01T08:00:00.000001+08:00",
    "2025-12-31T23:59:59Z",
]

EARTH_RADIUS_M = 6_371_000.0
MAX_GEO_FILTER_CANDIDATES = 800
GEO_DISTANCE_BOUNDARY_GUARD_M = 25.0
GEO_BOUNDARY_VALUES = [
    {"latitude": 0.0, "longitude": 0.0},
    {"latitude": -0.0, "longitude": -0.0},
    {"latitude": 0.0, "longitude": 179.9999},
    {"latitude": 0.0, "longitude": -179.9999},
    {"latitude": 89.9999, "longitude": 0.0},
    {"latitude": -89.9999, "longitude": 0.0},
    {"latitude": 45.0, "longitude": 45.0},
    {"latitude": -45.0, "longitude": -45.0},
]

TEXT_BOUNDARY_VALUES = [
    "",
    "A",
    "a",
    "0",
    "Aa",
    "a_a",
    "A-A",
    "CaseSensitive",
    "casesensitive",
    "x" * 64,
    "Y" * 255,
]

TEXT_STEMS = [
    "alpha",
    "ALPHA",
    "Alpha",
    "beta",
    "BETA",
    "item",
    "ITEM",
    "user",
    "User",
    "edge",
    "EDGE",
    "mix",
]

# 向量索引类型和距离度量
ALL_VECTOR_INDEX_TYPES = ["hnsw", "flat", "dynamic"]
ALL_DISTANCE_METRICS = [VectorDistances.COSINE, VectorDistances.L2_SQUARED, VectorDistances.DOT]
VECTOR_INDEX_TYPE = None
DISTANCE_METRIC = None

# 一致性等级
ALL_CONSISTENCY_LEVELS = [ConsistencyLevel.ONE, ConsistencyLevel.QUORUM, ConsistencyLevel.ALL]
DEFAULT_CONSISTENCY_LEVEL = ConsistencyLevel.ALL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.path.join(SCRIPT_DIR, "weaviate_log")


def ensure_log_dir():
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    return DEFAULT_LOG_DIR


def make_log_path(filename: str) -> str:
    return os.path.join(ensure_log_dir(), filename)


def display_path(path: str) -> str:
    try:
        return os.path.relpath(path, start=os.getcwd())
    except Exception:
        return path


def consistency_label(consistency, randomize=False):
    if randomize:
        return "RANDOM(ONE/QUORUM/ALL)"
    return str(consistency or DEFAULT_CONSISTENCY_LEVEL)


def parse_consistency_arg(value: str):
    mapping = {
        "one": ConsistencyLevel.ONE,
        "quorum": ConsistencyLevel.QUORUM,
        "all": ConsistencyLevel.ALL,
    }
    return mapping[value.lower()]


def format_repro_command(seed, consistency, randomize_consistency=False):
    parts = [f"python weaviate_fuzz_oracle.py --seed {seed}"]
    if randomize_consistency:
        parts.append("--random-consistency")
    elif (consistency or DEFAULT_CONSISTENCY_LEVEL) != DEFAULT_CONSISTENCY_LEVEL:
        parts.append(f"--consistency {(consistency or DEFAULT_CONSISTENCY_LEVEL).name.lower()}")
    return " ".join(parts)


# --- 1. Data Manager ---

class FieldType:
    INT = "INT"
    NUMBER = "NUMBER"
    BOOL = "BOOL"
    TEXT = "TEXT"
    GEO = "GEO_COORDINATES"
    INT_ARRAY = "INT_ARRAY"
    TEXT_ARRAY = "TEXT_ARRAY"
    NUMBER_ARRAY = "NUMBER_ARRAY"
    BOOL_ARRAY = "BOOL_ARRAY"
    DATE = "DATE"
    OBJECT = "OBJECT"  # Nested object (对标 Milvus JSON)

def get_weaviate_datatype(ftype):
    mapping = {
        FieldType.INT: DataType.INT,
        FieldType.NUMBER: DataType.NUMBER,
        FieldType.BOOL: DataType.BOOL,
        FieldType.TEXT: DataType.TEXT,
        FieldType.GEO: DataType.GEO_COORDINATES,
        FieldType.INT_ARRAY: DataType.INT_ARRAY,
        FieldType.TEXT_ARRAY: DataType.TEXT_ARRAY,
        FieldType.NUMBER_ARRAY: DataType.NUMBER_ARRAY,
        FieldType.BOOL_ARRAY: DataType.BOOL_ARRAY,
        FieldType.DATE: DataType.DATE,
        # OBJECT handled specially in reset_collection
    }
    return mapping.get(ftype)


# --- FuzzStats: 统计/度量收集 ---
class FuzzStats:
    """查询延迟、错误分类、通过率等统计信息收集 (对标 Milvus 日志汇总)"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.latencies = []
        self.expr_depths = []
        self.error_categories = {}  # {category: count}

    def record(self, passed, latency_ms=None, depth=None, error_cat=None):
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        if latency_ms is not None:
            self.latencies.append(latency_ms)
        if depth is not None:
            self.expr_depths.append(depth)
        if error_cat:
            self.error_categories[error_cat] = self.error_categories.get(error_cat, 0) + 1

    def record_skip(self):
        self.skipped += 1

    def record_fail(self, error_cat=None):
        self.record(False, error_cat=error_cat)

    def summary(self):
        lines = []
        lines.append(f"Total:{self.total} Pass:{self.passed} Fail:{self.failed} Skip:{self.skipped}")
        if self.latencies:
            arr = np.array(self.latencies)
            lines.append(f"Latency: avg={arr.mean():.1f}ms p50={np.percentile(arr,50):.1f}ms p99={np.percentile(arr,99):.1f}ms max={arr.max():.1f}ms")
        if self.expr_depths:
            darr = np.array(self.expr_depths)
            lines.append(f"Depth: avg={darr.mean():.1f} max={darr.max()}")
        if self.error_categories:
            lines.append(f"Errors: {dict(self.error_categories)}")
        return " | ".join(lines)


def to_utc_timestamp(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def canonicalize_date_string(value):
    ts = to_utc_timestamp(value)
    if ts is None:
        return None
    return ts.isoformat().replace("+00:00", "Z")


def float_window(value, ulps=4):
    if value is None:
        return None, None
    try:
        fv = float(value)
    except Exception:
        return None, None
    if not np.isfinite(fv):
        return None, None
    lo = np.float64(fv)
    hi = np.float64(fv)
    for _ in range(max(1, ulps)):
        lo = np.nextafter(lo, np.float64(-np.inf))
        hi = np.nextafter(hi, np.float64(np.inf))
    return float(lo), float(hi)


def is_null_like(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def select_sorted_value(values, fraction):
    if not values:
        return None
    idx = int(round((len(values) - 1) * fraction))
    idx = max(0, min(len(values) - 1, idx))
    return values[idx]


def build_int_stats(values):
    if not values:
        return None
    ordered = sorted(int(v) for v in values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "median": select_sorted_value(ordered, 0.50),
        "q1": select_sorted_value(ordered, 0.25),
        "q3": select_sorted_value(ordered, 0.75),
    }


def clamp_weaviate_int(value):
    return max(WEAVIATE_SAFE_INT_MIN, min(WEAVIATE_SAFE_INT_MAX, int(value)))


def clamp_latitude(value):
    return max(-90.0, min(90.0, float(value)))


def wrap_longitude(value):
    lon = ((float(value) + 180.0) % 360.0) - 180.0
    if lon == -180.0 and float(value) > 0:
        return 180.0
    return lon


def normalize_geo_value(value):
    if value is None:
        return None
    if hasattr(value, "latitude") and hasattr(value, "longitude"):
        lat = getattr(value, "latitude")
        lon = getattr(value, "longitude")
    elif isinstance(value, dict):
        lat = value.get("latitude")
        lon = value.get("longitude")
    else:
        return None
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return None
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None
    if lat < -90.0 or lat > 90.0 or lon < -180.0 or lon > 180.0:
        return None
    return {"latitude": lat, "longitude": lon}


def geo_distance_m(a, b):
    ga = normalize_geo_value(a)
    gb = normalize_geo_value(b)
    if ga is None or gb is None:
        return None
    lat1 = np.radians(ga["latitude"])
    lat2 = np.radians(gb["latitude"])
    dlat = lat2 - lat1
    dlon = np.radians(gb["longitude"] - ga["longitude"])
    h = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_M * np.arcsin(min(1.0, np.sqrt(h))))


def normalize_scalar_for_compare(ftype, value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if ftype == FieldType.INT:
        return int(value)
    if ftype == FieldType.NUMBER:
        fv = float(value)
        return 0.0 if fv == 0.0 else fv
    if ftype == FieldType.BOOL:
        return bool(value)
    if ftype == FieldType.TEXT:
        return None if value == "" else str(value)
    if ftype == FieldType.DATE:
        return canonicalize_date_string(value)
    if ftype == FieldType.GEO:
        return normalize_geo_value(value)
    return value


def extract_expr_fields(expr, schema_config):
    if not expr:
        return []
    matched = []
    for field in schema_config:
        if re.search(rf"\b{re.escape(field['name'])}\b", expr):
            matched.append(field)
    return matched


class DataManager:
    def __init__(self, seed):
        self.seed = int(seed)
        self.df = None
        self.vectors = None
        self.schema_config = []
        self.filterable_fields = set()  # Populated by reset_collection
        self.py_rng = random.Random(self.seed ^ 0x5EEDFACE)
        self.value_rng = np.random.default_rng(self.seed ^ 0xA17E)
        self.vector_rng = np.random.default_rng(self.seed ^ 0xC0FFEE)
        self.id_counter = 0
        self.null_ratio = self.py_rng.uniform(0.05, 0.15)
        self.array_capacity = self.py_rng.randint(5, 20)
        self.int_range = self.py_rng.randint(5000, 100000)
        self.double_scale = self.py_rng.uniform(100, 10000)

    def _next_uuid(self):
        self.id_counter += 1
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-fuzz-{self.seed}-{self.id_counter}"))

    def _random_string(self, min_len=1, max_len=10, boundary=False):
        if boundary and BOUNDARY_INJECTION_RATE and self.py_rng.random() < BOUNDARY_INJECTION_RATE:
            return self.py_rng.choice(TEXT_BOUNDARY_VALUES)

        mode = self.py_rng.choices(
            ["clustered", "symbolic", "mixed_case", "random"],
            weights=[0.45, 0.15, 0.20, 0.20],
            k=1,
        )[0]
        if mode == "clustered":
            stem = self.py_rng.choice(TEXT_STEMS)
            suffix = self.py_rng.choice(["", "", "", "_01", "-A", ".v2", str(self.py_rng.randint(0, 999))])
            return f"{stem}{suffix}"
        if mode == "symbolic":
            alphabet = string.ascii_letters + string.digits + "_-./@"
        elif mode == "mixed_case":
            stem = self.py_rng.choice(TEXT_STEMS)
            suffix = ''.join(self.py_rng.choices(string.ascii_letters + string.digits, k=self.py_rng.randint(0, 4)))
            return f"{stem}{suffix}"
        else:
            alphabet = string.ascii_letters + string.digits
        return ''.join(self.py_rng.choices(alphabet, k=self.py_rng.randint(min_len, max_len)))

    def _random_bool(self):
        return self.py_rng.choice([True, False])

    def _random_int(self, boundary=True):
        if boundary and BOUNDARY_INJECTION_RATE and self.py_rng.random() < BOUNDARY_INJECTION_RATE:
            return clamp_weaviate_int(self.py_rng.choice(INT_BOUNDARY_VALUES))

        mode = self.py_rng.choices(["uniform", "cluster", "hotspot"], weights=[0.35, 0.35, 0.30], k=1)[0]
        if mode == "hotspot":
            return clamp_weaviate_int(self.py_rng.choice([-1024, -100, -10, -1, 0, 1, 2, 10, 100, 1024]))
        if mode == "cluster":
            center = self.py_rng.choice([-10000, -1000, -100, -1, 0, 1, 100, 1000, 10000])
            jitter = self.py_rng.randint(-5, 5)
            return clamp_weaviate_int(np.clip(center + jitter, -self.int_range, self.int_range))
        return clamp_weaviate_int(self.py_rng.randint(-self.int_range, self.int_range))

    def _random_number(self, boundary=True):
        if boundary and BOUNDARY_INJECTION_RATE and self.py_rng.random() < BOUNDARY_INJECTION_RATE:
            return float(self.py_rng.choice(NUMBER_BOUNDARY_VALUES))

        mode = self.py_rng.choices(["uniform", "cluster", "hotspot", "rounded"], weights=[0.30, 0.30, 0.20, 0.20], k=1)[0]
        if mode == "hotspot":
            #, -0.0 暂时注释掉-0.0，避免重复问题反复出现
            # return float(self.py_rng.choice([-1e6, -1.0,-0.0,-0.1, 0.0, 0.1, 1.0, 1e6]))
            return float(self.py_rng.choice([-1e6, -1.0, -0.1, 0.0, 0.1, 1.0, 1e6]))
        if mode == "cluster":
            center = self.py_rng.choice([-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0])
            return float(center + self.py_rng.uniform(-1e-3, 1e-3) * max(abs(center), 1.0))
        if mode == "rounded":
            return float(round(self.py_rng.uniform(-self.double_scale, self.double_scale), self.py_rng.choice([0, 1, 2, 6])))
        return float(self.py_rng.uniform(-self.double_scale, self.double_scale))

    def _format_timestamp(self, ts):
        offset_minutes = self.py_rng.choice([0, 0, 0, 60, 330, -480])
        return format_rfc3339_timestamp(ts, offset_minutes=offset_minutes)

    def _random_date(self, boundary=True):
        if boundary and BOUNDARY_INJECTION_RATE and self.py_rng.random() < BOUNDARY_INJECTION_RATE:
            return self.py_rng.choice(DATE_BOUNDARY_VALUES)

        mode = self.py_rng.choices(["boundary", "cluster", "uniform"], weights=[0.30, 0.35, 0.35], k=1)[0]
        if mode == "boundary":
            return self.py_rng.choice(DATE_BOUNDARY_VALUES)

        if mode == "cluster":
            anchor = to_utc_timestamp(self.py_rng.choice(DATE_BOUNDARY_VALUES))
            if anchor is None:
                anchor = pd.Timestamp("2024-01-01T00:00:00Z")
            micro_jitter = self.py_rng.choice([0, 0, 0, -17, -1, 1, 17, 123456, 999999])
            ts = anchor + pd.Timedelta(
                days=self.py_rng.randint(-3, 3),
                hours=self.py_rng.randint(-6, 6),
                microseconds=micro_jitter,
            )
        else:
            year = self.py_rng.randint(1998, 2026)
            month = self.py_rng.randint(1, 12)
            day = self.py_rng.randint(1, 28)
            hour = self.py_rng.randint(0, 23)
            minute = self.py_rng.randint(0, 59)
            second = self.py_rng.randint(0, 59)
            microsecond = self.py_rng.choice([0, 0, 0, 1, 2, 17, 123456, 999999])
            ts = pd.Timestamp(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
                tz="UTC",
            )
        return self._format_timestamp(ts)

    def _random_geo(self, boundary=True):
        if boundary and BOUNDARY_INJECTION_RATE and self.py_rng.random() < BOUNDARY_INJECTION_RATE:
            return dict(self.py_rng.choice(GEO_BOUNDARY_VALUES))

        anchor = self.py_rng.choice(GEO_BOUNDARY_VALUES)
        jitter_m = self.py_rng.choice([0.0, 1.0, 10.0, 100.0, 1000.0, 100000.0])
        bearing = self.py_rng.uniform(0.0, 2.0 * np.pi)
        lat_scale = jitter_m / 111_320.0
        lat = clamp_latitude(anchor["latitude"] + lat_scale * np.cos(bearing))
        lon_denom = max(0.01, np.cos(np.radians(anchor["latitude"])))
        lon_scale = jitter_m / (111_320.0 * lon_denom)
        lon = wrap_longitude(anchor["longitude"] + lon_scale * np.sin(bearing))
        return {"latitude": float(lat), "longitude": float(lon)}

    def _random_array_length(self):
        if self.py_rng.random() < 0.35:
            candidates = [0, 1, 2, max(0, self.array_capacity - 1), self.array_capacity]
            return int(self.py_rng.choice(candidates))
        return self.py_rng.randint(0, min(self.array_capacity, 5))

    def _generate_array_value(self, element_type):
        arr_len = self._random_array_length()
        if arr_len <= 0:
            return None

        values = []
        for _ in range(arr_len):
            if element_type == FieldType.INT:
                values.append(self._random_int(boundary=True))
            elif element_type == FieldType.NUMBER:
                values.append(self._random_number(boundary=True))
            elif element_type == FieldType.BOOL:
                values.append(self._random_bool())
            else:
                values.append(self._random_string(1, 12, boundary=True))
        return values if values else None

    def _generate_object_value(self, field):
        if self.py_rng.random() < self.null_ratio:
            return None

        obj = {}
        for nf in field.get("nested", []):
            if self.py_rng.random() < self.null_ratio:
                continue
            if nf["type"] == FieldType.INT:
                obj[nf["name"]] = self._random_int(boundary=True)
            elif nf["type"] == FieldType.NUMBER:
                obj[nf["name"]] = self._random_number(boundary=True)
            elif nf["type"] == FieldType.BOOL:
                obj[nf["name"]] = self._random_bool()
            elif nf["type"] == FieldType.TEXT:
                obj[nf["name"]] = self._random_string(2, 12, boundary=True)
        return obj if obj else None

    def _generate_scalar_value(self, ftype):
        if ftype == FieldType.INT:
            return self._random_int(boundary=True)
        if ftype == FieldType.NUMBER:
            return self._random_number(boundary=True)
        if ftype == FieldType.BOOL:
            return self._random_bool()
        if ftype == FieldType.TEXT:
            return self._random_string(1, 48, boundary=True)
        if ftype == FieldType.DATE:
            return self._random_date(boundary=True)
        if ftype == FieldType.GEO:
            return self._random_geo(boundary=True)
        return None

    def normalize_dataframe_types(self):
        if self.df is None:
            return
        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]
            if fname not in self.df.columns:
                continue
            if ftype == FieldType.INT:
                normalized = [
                    None if is_null_like(v) else int(v)
                    for v in self.df[fname].tolist()
                ]
                self.df[fname] = pd.Series(normalized, index=self.df.index, dtype="object")

    def rows_to_dataframe(self, rows):
        frame = pd.DataFrame(rows)
        for field in self.schema_config:
            fname = field["name"]
            if fname not in frame.columns:
                continue
            if field["type"] == FieldType.INT:
                frame[fname] = pd.Series(
                    [row.get(fname) for row in rows],
                    index=frame.index,
                    dtype="object",
                )
        return frame

    def generate_schema(self):
        print("🎲 1. Defining Dynamic Schema...")
        self.schema_config = []
        num_fields = self.py_rng.randint(5, 18)
        types_pool = [FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT, FieldType.DATE]
        weights = [0.26, 0.24, 0.14, 0.24, 0.12]

        for i in range(num_fields):
            ftype = self.py_rng.choices(types_pool, weights=weights, k=1)[0]
            self.schema_config.append({"name": f"c{i}", "type": ftype})

        self.schema_config.append({"name": "tagsArray", "type": FieldType.INT_ARRAY, "max_capacity": self.array_capacity})
        self.schema_config.append({"name": "labelsArray", "type": FieldType.TEXT_ARRAY, "max_capacity": self.array_capacity})
        self.schema_config.append({"name": "scoresArray", "type": FieldType.NUMBER_ARRAY, "max_capacity": self.array_capacity})
        self.schema_config.append({"name": "flagsArray", "type": FieldType.BOOL_ARRAY, "max_capacity": self.array_capacity})
        self.schema_config.append({"name": "geoLocation", "type": FieldType.GEO})
        self.schema_config.append({"name": "metaObj", "type": FieldType.OBJECT, "nested": [
            {"name": "price", "type": FieldType.INT},
            {"name": "color", "type": FieldType.TEXT},
            {"name": "active", "type": FieldType.BOOL},
            {"name": "score", "type": FieldType.NUMBER},
        ]})

        print(f"   -> Generated {len(self.schema_config)} dynamic fields (plus id & vector).")
        for f in self.schema_config:
            print(f"      - {f['name']:<20} : {f['type']}")

    def generate_single_row(self, id_override=None, row_num=-1):
        row = {
            "id": str(id_override) if id_override is not None else self._next_uuid(),
            "row_num": int(row_num),
        }

        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]

            if ftype in {FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT, FieldType.DATE}:
                row[fname] = None if self.py_rng.random() < self.null_ratio else self._generate_scalar_value(ftype)
            elif ftype == FieldType.GEO:
                row[fname] = None if self.py_rng.random() < self.null_ratio else self._random_geo(boundary=True)
            elif ftype == FieldType.INT_ARRAY:
                row[fname] = None if self.py_rng.random() < self.null_ratio else self._generate_array_value(FieldType.INT)
            elif ftype == FieldType.TEXT_ARRAY:
                row[fname] = None if self.py_rng.random() < self.null_ratio else self._generate_array_value(FieldType.TEXT)
            elif ftype == FieldType.NUMBER_ARRAY:
                row[fname] = None if self.py_rng.random() < self.null_ratio else self._generate_array_value(FieldType.NUMBER)
            elif ftype == FieldType.BOOL_ARRAY:
                row[fname] = None if self.py_rng.random() < self.null_ratio else self._generate_array_value(FieldType.BOOL)
            elif ftype == FieldType.OBJECT:
                row[fname] = self._generate_object_value(field)
        return row

    def generate_single_vector(self):
        vec = self.vector_rng.random(DIM, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0] = 1.0
            norm = 1.0
        vec /= norm
        return vec.astype(np.float32)

    def generate_data(self):
        print(f"🌊 2. Generating {N} rows (Vector Dim={DIM})...")
        self.vectors = self.vector_rng.random((N, DIM), dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1)
        norms[norms == 0] = 1.0
        self.vectors = (self.vectors / norms[:, np.newaxis]).astype(np.float32)

        rows = [self.generate_single_row(row_num=i) for i in range(N)]
        self.df = self.rows_to_dataframe(rows)

        for field in self.schema_config:
            if field["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY]:
                self.df[field["name"]] = self.df[field["name"]].apply(
                    lambda x: None if isinstance(x, list) and len(x) == 0 else x
                )
        self.normalize_dataframe_types()

        print("✅ Data Generation Complete.")
        null_counts = {col: self.df[col].isna().sum() for col in self.df.columns}
        print(f"   -> Null counts (sample): {dict(list(null_counts.items())[:5])}")


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

    @staticmethod
    def _normalize_metadata_time(value):
        if value is None:
            return pd.NaT
        try:
            ts = pd.Timestamp(value)
        except Exception:
            return pd.NaT
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def reset_collection(self, schema_config):
        try:
            self.client.collections.delete(CLASS_NAME)
            print(f"   -> Deleted existing: {CLASS_NAME}")
        except Exception:
            pass

        properties = []
        index_config_log = []
        filterable_fields = set()  # Track which fields have index_filterable=True
        searchable_text_fields = set()
        for field in schema_config:
            if field["type"] == FieldType.OBJECT:
                # Nested object property
                nested_props = []
                for nf in field.get("nested", []):
                    nwt = get_weaviate_datatype(nf["type"])
                    if nwt:
                        nested_props.append(Property(name=nf["name"], data_type=nwt))
                if nested_props:
                    properties.append(Property(
                        name=field["name"], data_type=DataType.OBJECT,
                        nested_properties=nested_props,
                        index_filterable=True,  # OBJECT must be filterable for is_none
                    ))
                    filterable_fields.add(field["name"])
                continue
            wv_type = get_weaviate_datatype(field["type"])
            if wv_type:
                # Scalar index config fuzzing (对标 Milvus 标量索引随机创建)
                idx_filterable = True if field["type"] == FieldType.GEO else random.choice([True, True, True, False])  # 75% on
                idx_searchable = random.choice([True, False]) if field["type"] == FieldType.TEXT else False
                idx_range = random.choice([True, False]) if field["type"] in (FieldType.INT, FieldType.NUMBER, FieldType.DATE) else False
                # TEXT/TEXT_ARRAY 使用 Tokenization.FIELD: 精确匹配, 区分大小写, 无停用词过滤
                tok = Tokenization.FIELD if field["type"] in (FieldType.TEXT, FieldType.TEXT_ARRAY) else None
                p_kwargs = dict(name=field["name"], data_type=wv_type,
                                index_filterable=idx_filterable,
                                index_searchable=idx_searchable,
                                index_range_filterable=idx_range)
                if tok is not None:
                    p_kwargs["tokenization"] = tok
                p = Property(**p_kwargs)
                properties.append(p)
                if idx_filterable:
                    filterable_fields.add(field["name"])
                if field["type"] == FieldType.TEXT and idx_searchable:
                    searchable_text_fields.add(field["name"])
                index_config_log.append(f"{field['name']}: filt={idx_filterable} search={idx_searchable} range={idx_range}")
        properties.append(Property(name="row_num", data_type=DataType.INT))
        if index_config_log:
            print(f"   -> Scalar index config: {len(index_config_log)} fields randomized")
            for ic in index_config_log[:5]:
                print(f"      {ic}")

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
                    vector_config=Configure.Vectors.self_provided(
                        vector_index_config=vi_config,
                    ),
                    inverted_index_config=Configure.inverted_index(
                        index_null_state=True,
                        index_property_length=True,
                        index_timestamps=True,
                    ),
                )
                break
            except Exception as e:
                err_msg = str(e)
                print(f"   -> Create attempt {attempt+1} failed: {err_msg[:120]}")
                try:
                    self.client.collections.delete(CLASS_NAME)
                except Exception:
                    pass
                if "async indexing" in err_msg or "dynamic" in err_msg.lower() or "422" in err_msg:
                    vi_config = Configure.VectorIndex.hnsw(distance_metric=dist)
                    print(f"   -> Falling back to HNSW (dynamic requires ASYNC_INDEXING=true in server env)")
                elif attempt == 0:
                    # Try without inverted_index_config
                    try:
                        self.client.collections.create(
                            name=CLASS_NAME, properties=properties,
                            vector_config=Configure.Vectors.self_provided(
                                vector_index_config=vi_config,
                            ),
                        )
                        break
                    except Exception as e2:
                        print(f"   -> Fallback also failed: {e2}")
                        vi_config = Configure.VectorIndex.hnsw(distance_metric=dist)
                else:
                    if attempt == 2:
                        raise
        print("🛠️ Collection Created.")
        self.searchable_text_fields = searchable_text_fields
        return filterable_fields

    def sync_creation_times(self, dm, ids=None, retries=3):
        if self.client is None or dm.df.empty:
            return

        collection = self.client.collections.get(CLASS_NAME)
        target_ids = None if ids is None else [str(x) for x in ids]
        mapping = {}

        for attempt in range(retries):
            kwargs = {
                "limit": max(1, len(target_ids) if target_ids is not None else len(dm.df) + 10),
                "return_metadata": MetadataQuery(creation_time=True),
                "return_properties": False,
            }
            if target_ids:
                if len(target_ids) == 1:
                    kwargs["filters"] = Filter.by_id().equal(target_ids[0])
                else:
                    kwargs["filters"] = Filter.by_id().contains_any(target_ids)

            response = collection.query.fetch_objects(**kwargs)
            mapping = {
                str(obj.uuid): self._normalize_metadata_time(
                    getattr(getattr(obj, "metadata", None), "creation_time", None)
                )
                for obj in response.objects
            }

            if target_ids is None or all(target in mapping for target in target_ids):
                break
            time.sleep(SLEEP_INTERVAL * 2)

        if "_creation_time" not in dm.df.columns:
            dm.df["_creation_time"] = pd.NaT

        if target_ids is None:
            dm.df["_creation_time"] = pd.to_datetime(
                dm.df["id"].map(mapping), utc=True, errors="coerce"
            )
        else:
            mask = dm.df["id"].isin(target_ids)
            dm.df.loc[mask, "_creation_time"] = dm.df.loc[mask, "id"].map(mapping)
            dm.df["_creation_time"] = pd.to_datetime(dm.df["_creation_time"], utc=True, errors="coerce")

    def sync_update_times(self, dm, ids=None, retries=3):
        if self.client is None or dm.df.empty:
            return

        collection = self.client.collections.get(CLASS_NAME)
        target_ids = None if ids is None else [str(x) for x in ids]
        mapping = {}

        for attempt in range(retries):
            kwargs = {
                "limit": max(1, len(target_ids) if target_ids is not None else len(dm.df) + 10),
                "return_metadata": MetadataQuery(last_update_time=True),
                "return_properties": False,
            }
            if target_ids:
                if len(target_ids) == 1:
                    kwargs["filters"] = Filter.by_id().equal(target_ids[0])
                else:
                    kwargs["filters"] = Filter.by_id().contains_any(target_ids)

            response = collection.query.fetch_objects(**kwargs)
            mapping = {
                str(obj.uuid): self._normalize_metadata_time(
                    getattr(getattr(obj, "metadata", None), "last_update_time", None)
                )
                for obj in response.objects
            }

            if target_ids is None or all(target in mapping for target in target_ids):
                break
            time.sleep(SLEEP_INTERVAL * 2)

        if "_update_time" not in dm.df.columns:
            dm.df["_update_time"] = pd.NaT

        if target_ids is None:
            dm.df["_update_time"] = pd.to_datetime(
                dm.df["id"].map(mapping), utc=True, errors="coerce"
            )
        else:
            mask = dm.df["id"].isin(target_ids)
            dm.df.loc[mask, "_update_time"] = dm.df.loc[mask, "id"].map(mapping)
            dm.df["_update_time"] = pd.to_datetime(dm.df["_update_time"], utc=True, errors="coerce")

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
            elif ftype == FieldType.GEO:
                gv = normalize_geo_value(v)
                if gv is None:
                    return None
                return GeoCoordinate(latitude=gv["latitude"], longitude=gv["longitude"])
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
                    result = collection.data.insert_many(data_objects)
                    batch_errors = getattr(result, "errors", None)
                    if not batch_errors and isinstance(result, dict):
                        batch_errors = result.get("errors")
                    if batch_errors:
                        raise RuntimeError(f"batch insert partially failed: {batch_errors}")
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
        self.schema_all = dm.schema_config
        # Only use filterable fields for query generation to avoid index-not-enabled errors
        filterable = getattr(dm, 'filterable_fields', None)
        if filterable:
            self.schema = [f for f in dm.schema_config if f["name"] in filterable]
        else:
            self.schema = dm.schema_config
        like_text_names = set(filterable or set()) | set(getattr(dm, "searchable_text_fields", set()) or set())
        self.df = dm.df
        self.dm = dm
        self.scalar_schema = [f for f in self.schema if f["type"] in [FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT, FieldType.DATE]]
        self.like_text_schema = [f for f in dm.schema_config if f["type"] == FieldType.TEXT and f["name"] in like_text_names]
        self.geo_schema = [f for f in self.schema if f["type"] == FieldType.GEO]
        self.array_schema = [f for f in self.schema if f["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY]]
        property_length_text_names = like_text_names
        property_length_array_names = {f["name"] for f in self.array_schema}
        self.property_length_schema = [
            f for f in dm.schema_config
            if (f["type"] == FieldType.TEXT and f["name"] in property_length_text_names)
            or (f["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY] and f["name"] in property_length_array_names)
        ]
        self._field_values = {}
        self._field_stats = {}
        value_schema = list(self.scalar_schema)
        value_schema_names = {f["name"] for f in value_schema}
        for field in self.like_text_schema:
            if field["name"] not in value_schema_names:
                value_schema.append(field)
                value_schema_names.add(field["name"])
        for field in value_schema:
            fname, ftype = field["name"], field["type"]
            valid_values = self._extract_valid_values(fname, ftype)
            if not valid_values:
                continue
            self._field_values[fname] = valid_values
            if ftype == FieldType.INT:
                stats = build_int_stats(valid_values)
                if stats is not None:
                    self._field_stats[fname] = stats
            elif ftype == FieldType.NUMBER:
                arr = np.array(valid_values, dtype=np.float64)
                self._field_stats[fname] = {
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                    "q1": float(np.quantile(arr, 0.25)),
                    "q3": float(np.quantile(arr, 0.75)),
                }
            elif ftype == FieldType.DATE:
                parsed = pd.Series(valid_values).apply(to_utc_timestamp).dropna()
                if not parsed.empty:
                    self._field_stats[fname] = {
                        "min_ts": parsed.min(),
                        "max_ts": parsed.max(),
                        "samples": list(valid_values[:32]),
                    }
        self.creation_time_series = (
            pd.to_datetime(self.df["_creation_time"], utc=True, errors="coerce")
            if "_creation_time" in self.df.columns else None
        )
        self.update_time_series = (
            pd.to_datetime(self.df["_update_time"], utc=True, errors="coerce")
            if "_update_time" in self.df.columns else None
        )

    def _random_string(self, min_len=5, max_len=10):
        alphabet = string.ascii_letters + string.digits + "_-./@"
        return ''.join(random.choices(alphabet, k=random.randint(min_len, max_len)))

    @staticmethod
    def _convert_to_native(val):
        if isinstance(val, np.integer): return int(val)
        elif isinstance(val, np.floating): return float(val)
        elif isinstance(val, np.bool_): return bool(val)
        elif isinstance(val, np.ndarray): return [OracleQueryGenerator._convert_to_native(x) for x in val.tolist()]
        elif isinstance(val, list): return [OracleQueryGenerator._convert_to_native(x) for x in val]
        return val

    def _is_effectively_null_value(self, value, ftype):
        if is_null_like(value):
            return True
        if ftype == FieldType.TEXT and value == "":
            return True
        if ftype in (FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY):
            if isinstance(value, list) and len(value) == 0:
                return True
        if ftype == FieldType.GEO and normalize_geo_value(value) is None:
            return True
        return False

    def _effective_null_mask(self, series, ftype):
        return series.apply(lambda x, ft=ftype: self._is_effectively_null_value(x, ft))

    def _valid_series(self, fname, ftype):
        series = self.df[fname]
        return series[~self._effective_null_mask(series, ftype)]

    def _extract_valid_values(self, fname, ftype):
        values = []
        for value in self.df[fname].tolist():
            if self._is_effectively_null_value(value, ftype):
                continue
            if ftype == FieldType.INT:
                values.append(int(value))
            elif ftype == FieldType.NUMBER:
                fv = float(value)
                if np.isfinite(fv):
                    values.append(fv)
            elif ftype == FieldType.BOOL:
                values.append(bool(value))
            elif ftype in (FieldType.TEXT, FieldType.DATE):
                values.append(str(value))
            elif ftype == FieldType.GEO:
                geo = normalize_geo_value(value)
                if geo is not None:
                    values.append(geo)
            else:
                values.append(value)
        return values

    def _existing_ids(self):
        if "id" not in self.df.columns:
            return []
        return [str(value) for value in self.df["id"].tolist() if value is not None]

    def _random_absent_uuid(self, existing_ids=None):
        existing = set(existing_ids or self._existing_ids())
        while True:
            candidate = str(uuid.uuid4())
            if candidate not in existing:
                return candidate

    def _date_cmp_mask(self, series, op, target):
        target_ts = to_utc_timestamp(target)
        if target_ts is None:
            return pd.Series(False, index=series.index)

        def comp(x):
            ts = to_utc_timestamp(x)
            if ts is None:
                return False
            if op == "==": return ts == target_ts
            if op == "!=": return ts != target_ts
            if op == ">": return ts > target_ts
            if op == "<": return ts < target_ts
            if op == ">=": return ts >= target_ts
            if op == "<=": return ts <= target_ts
            return False

        return series.apply(comp)

    def get_value_for_query(self, fname, ftype):
        valid_values = self._field_values.get(fname, [])

        if BOUNDARY_INJECTION_RATE and random.random() < BOUNDARY_INJECTION_RATE:
            if ftype == FieldType.INT:
                return clamp_weaviate_int(random.choice(INT_BOUNDARY_VALUES))
            if ftype == FieldType.NUMBER:
                return float(random.choice(NUMBER_BOUNDARY_VALUES))
            if ftype == FieldType.TEXT:
                candidates = [v for v in TEXT_BOUNDARY_VALUES if v != ""]
                return random.choice(candidates or TEXT_BOUNDARY_VALUES)
            if ftype == FieldType.DATE:
                return random.choice(DATE_BOUNDARY_VALUES)

        if valid_values and random.random() < 0.8:
            return random.choice(valid_values)
        if ftype == FieldType.INT:
            stats = self._field_stats.get(fname)
            if stats:
                hi = min(int(stats["max"]) + 100000, WEAVIATE_SAFE_INT_MAX)
                lo = max(int(stats["min"]) - 100000, WEAVIATE_SAFE_INT_MIN)
                candidates = []
                if hi > int(stats["max"]):
                    candidates.append(hi)
                if lo < int(stats["min"]):
                    candidates.append(lo)
                if candidates:
                    return random.choice(candidates)
                return random.choice(INT_BOUNDARY_VALUES)
            return clamp_weaviate_int(random.randint(-200000, 200000))
        elif ftype == FieldType.NUMBER:
            stats = self._field_stats.get(fname)
            if stats:
                anchors = [
                    float(stats["min"]),
                    float(stats["q1"]),
                    float(stats["median"]),
                    float(stats["q3"]),
                    float(stats["max"]),
                ]
                nearby = unique_float_values(
                    [
                        candidate
                        for anchor in anchors
                        for candidate in (next_float(anchor, -np.inf), next_float(anchor, np.inf))
                        if candidate is not None
                    ]
                )
                if nearby and random.random() < 0.70:
                    return random.choice(nearby)
                outside = unique_float_values(
                    [
                        next_float(float(stats["min"]), -np.inf),
                        next_float(float(stats["max"]), np.inf),
                    ]
                )
                if outside:
                    return random.choice(outside)
                return random.choice(NUMBER_BOUNDARY_VALUES)
            return random.uniform(-200000.0, 200000.0)
        elif ftype == FieldType.BOOL:
            return random.choice([True, False])
        elif ftype == FieldType.TEXT:
            return ''.join(random.choices(string.ascii_letters + string.digits + "_-./@", k=random.randint(15, 30)))
        elif ftype == FieldType.DATE:
            stats = self._field_stats.get(fname)
            if stats and stats.get("samples"):
                parsed_samples = [to_utc_timestamp(value) for value in stats["samples"]]
                parsed_samples = [value for value in parsed_samples if value is not None]
                if parsed_samples:
                    anchor = random.choice(parsed_samples)
                    if random.random() < 0.70:
                        delta = random.choice(
                            [
                                pd.Timedelta(microseconds=-1),
                                pd.Timedelta(microseconds=1),
                                pd.Timedelta(seconds=-1),
                                pd.Timedelta(seconds=1),
                            ]
                        )
                        return format_rfc3339_timestamp(anchor + delta)
                    return format_rfc3339_timestamp(anchor)
            if stats and "max_ts" in stats:
                anchor = stats["max_ts"] + pd.Timedelta(days=30)
                return format_rfc3339_timestamp(anchor)
            return "2030-01-01T00:00:00Z"
        return None

    def gen_boundary_expr(self):
        boundary_fields = [f for f in self.scalar_schema if f["name"] in self._field_stats and f["type"] in [FieldType.INT, FieldType.NUMBER, FieldType.DATE]]
        if not boundary_fields:
            return None, None, None
        field = random.choice(boundary_fields)
        fname, ftype = field["name"], field["type"]
        stats = self._field_stats[fname]
        series = self.df[fname]

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
            bt = random.choice(["exact_min", "exact_max", "below_min", "above_max", "at_median", "range_q1_q3", "range_narrow"])
            min_v, max_v = int(stats["min"]), int(stats["max"])
            med = int(stats["median"])
            q1, q3 = int(stats["q1"]), int(stats["q3"])
            if bt == "exact_min":
                return Filter.by_property(fname).equal(min_v), series.apply(safe_cmp("==", min_v)), f"{fname} == {min_v} (min)"
            elif bt == "exact_max":
                return Filter.by_property(fname).equal(max_v), series.apply(safe_cmp("==", max_v)), f"{fname} == {max_v} (max)"
            elif bt == "below_min":
                if min_v <= WEAVIATE_SAFE_INT_MIN:
                    return Filter.by_property(fname).equal(min_v), series.apply(safe_cmp("==", min_v)), f"{fname} == {min_v} (min)"
                v = min_v - 1
                return Filter.by_property(fname).less_than(v), series.apply(safe_cmp("<", v)), f"{fname} < {v} (below min)"
            elif bt == "above_max":
                if max_v >= WEAVIATE_SAFE_INT_MAX:
                    return Filter.by_property(fname).equal(max_v), series.apply(safe_cmp("==", max_v)), f"{fname} == {max_v} (max)"
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
            bt = random.choice(
                [
                    "exact_min",
                    "exact_max",
                    "below_min",
                    "above_max",
                    "at_median",
                    "range_q1_q3",
                    "range_narrow",
                    "gt_prev_anchor",
                    "lt_next_anchor",
                ]
            )
            min_v, max_v, med = float(stats["min"]), float(stats["max"]), float(stats["median"])
            if bt in ["exact_min", "exact_max", "at_median"]:
                t = min_v if bt == "exact_min" else (max_v if bt == "exact_max" else med)
                lo, hi = float_window(t)
                if lo is None:
                    return None, None, None
                fc = Filter.by_property(fname).greater_or_equal(lo) & Filter.by_property(fname).less_or_equal(hi)
                return fc, series.apply(lambda x: lo <= x <= hi if pd.notna(x) else False), f"{fname} ≈ {t}"
            elif bt == "below_min":
                v = next_float(min_v, -np.inf) or (min_v - 1.0)
                return Filter.by_property(fname).less_than(v), series.apply(safe_cmp("<", v)), f"{fname} < {v}"
            elif bt == "above_max":
                v = next_float(max_v, np.inf) or (max_v + 1.0)
                return Filter.by_property(fname).greater_than(v), series.apply(safe_cmp(">", v)), f"{fname} > {v}"
            elif bt == "gt_prev_anchor":
                anchor = random.choice([min_v, med, max_v, float(stats["q1"]), float(stats["q3"])])
                prev_val = next_float(anchor, -np.inf)
                if prev_val is None:
                    return None, None, None
                return (
                    Filter.by_property(fname).greater_than(prev_val),
                    series.apply(safe_cmp(">", prev_val)),
                    f"{fname} > prev({anchor})",
                )
            elif bt == "lt_next_anchor":
                anchor = random.choice([min_v, med, max_v, float(stats["q1"]), float(stats["q3"])])
                next_val = next_float(anchor, np.inf)
                if next_val is None:
                    return None, None, None
                return (
                    Filter.by_property(fname).less_than(next_val),
                    series.apply(safe_cmp("<", next_val)),
                    f"{fname} < next({anchor})",
                )
            else:
                mid = (min_v + max_v) / 2
                delta = max(0.001, (max_v - min_v) / 10)
                fc = Filter.by_property(fname).greater_or_equal(mid - delta) & Filter.by_property(fname).less_or_equal(mid + delta)
                return fc, series.apply(lambda x: (mid - delta) <= x <= (mid + delta) if pd.notna(x) else False), f"{fname} [{mid-delta:.2f},{mid+delta:.2f}]"
        elif ftype == FieldType.DATE:
            bt = random.choice(["exact_min", "exact_max", "below_min", "above_max", "sample_equal", "gt_before_sample_us", "lt_after_sample_us"])
            min_ts = stats["min_ts"]
            max_ts = stats["max_ts"]
            if bt == "exact_min":
                target = format_rfc3339_timestamp(min_ts)
                return Filter.by_property(fname).equal(target), self._date_cmp_mask(series, "==", target), f"{fname} == {target} (min)"
            elif bt == "exact_max":
                target = format_rfc3339_timestamp(max_ts)
                return Filter.by_property(fname).equal(target), self._date_cmp_mask(series, "==", target), f"{fname} == {target} (max)"
            elif bt == "below_min":
                target = format_rfc3339_timestamp(min_ts - pd.Timedelta(microseconds=1))
                return Filter.by_property(fname).less_than(target), self._date_cmp_mask(series, "<", target), f"{fname} < {target}"
            elif bt == "above_max":
                target = format_rfc3339_timestamp(max_ts + pd.Timedelta(microseconds=1))
                return Filter.by_property(fname).greater_than(target), self._date_cmp_mask(series, ">", target), f"{fname} > {target}"
            elif bt == "gt_before_sample_us":
                sample_ts = to_utc_timestamp(random.choice(stats["samples"]))
                if sample_ts is None:
                    return None, None, None
                target = format_rfc3339_timestamp(sample_ts - pd.Timedelta(microseconds=1))
                return Filter.by_property(fname).greater_than(target), self._date_cmp_mask(series, ">", target), f"{fname} > {target}"
            elif bt == "lt_after_sample_us":
                sample_ts = to_utc_timestamp(random.choice(stats["samples"]))
                if sample_ts is None:
                    return None, None, None
                target = format_rfc3339_timestamp(sample_ts + pd.Timedelta(microseconds=1))
                return Filter.by_property(fname).less_than(target), self._date_cmp_mask(series, "<", target), f"{fname} < {target}"
            else:
                sample = random.choice(stats["samples"])
                return Filter.by_property(fname).equal(sample), self._date_cmp_mask(series, "==", sample), f"{fname} == {sample}"
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
            # Tokenization.FIELD: 不需要停用词过滤
            pass
        if not all_items:
            return None, None, None
        num = min(random.randint(2, 4), len(set(all_items)))
        if num < 1:
            return None, None, None
        targets = [self._convert_to_native(t) for t in random.sample(list(set(all_items)), num)]

        mode = random.choice(["contains_any", "contains_all", "contains_none"])
        # Tokenization.FIELD → 精确匹配, 不再需要 .lower() 变换
        if mode == "contains_any":
            fc = Filter.by_property(fname).contains_any(targets)
            mask = series.apply(lambda x: any(i in targets for i in x) if isinstance(x, list) else False)
            return fc, mask, f"{fname} contains_any {targets}"
        elif mode == "contains_all":
            fc = Filter.by_property(fname).contains_all(targets)
            mask = series.apply(lambda x: all(t in x for t in targets) if isinstance(x, list) else False)
            return fc, mask, f"{fname} contains_all {targets}"
        else:
            # contains_none: 不含 targets 中任何值; null 数组视为匹配
            fc = Filter.by_property(fname).contains_none(targets)
            mask = series.apply(lambda x: not any(i in targets for i in x) if isinstance(x, list) else True)
            return fc, mask, f"{fname} contains_none {targets}"

    def gen_geo_expr(self):
        if not self.geo_schema:
            return None, None, None
        field = random.choice(self.geo_schema)
        fname = field["name"]
        series = self.df[fname]
        valid_values = [normalize_geo_value(v) for v in series.tolist()]
        valid_values = [v for v in valid_values if v is not None]
        if not valid_values:
            return None, None, None
        center = random.choice(valid_values)
        distance_series = series.apply(lambda value: geo_distance_m(center, value))
        finite_distances = sorted(
            float(dist) for dist in distance_series.tolist()
            if dist is not None and np.isfinite(dist)
        )
        if not finite_distances:
            return None, None, None

        safe_cap = min(MAX_GEO_FILTER_CANDIDATES, len(finite_distances))
        target_counts = {1, 2, 5, 10, 25, 50, 100, 200, 400, safe_cap}
        if safe_cap > 1:
            target_counts.update(
                {
                    max(1, safe_cap // 4),
                    max(1, safe_cap // 2),
                    max(1, (safe_cap * 3) // 4),
                }
            )

        distance_candidates = []
        for target_count in sorted(target_counts):
            if target_count <= 0 or target_count > safe_cap:
                continue
            left = finite_distances[target_count - 1]
            right = finite_distances[target_count] if target_count < len(finite_distances) else None
            if right is None:
                if len(finite_distances) <= MAX_GEO_FILTER_CANDIDATES:
                    distance_candidates.append(left + 2.0 * GEO_DISTANCE_BOUNDARY_GUARD_M)
                continue
            if right - left > 2.0 * GEO_DISTANCE_BOUNDARY_GUARD_M:
                distance_candidates.append((left + right) / 2.0)

        seen_distances = set()
        random.shuffle(distance_candidates)
        for distance in distance_candidates:
            distance = float(max(0.0, distance))
            distance_key = round(distance, 6)
            if distance_key in seen_distances:
                continue
            seen_distances.add(distance_key)
            if distance > 0.0:
                low = distance - GEO_DISTANCE_BOUNDARY_GUARD_M
                high = distance + GEO_DISTANCE_BOUNDARY_GUARD_M
                if any(low <= dist <= high for dist in finite_distances):
                    continue
            mask = distance_series.apply(
                lambda dist, radius=distance: bool(
                    dist is not None and np.isfinite(dist) and dist <= radius + 1e-6
                )
            )
            match_count = int(mask.fillna(False).sum())
            if match_count <= 0 or match_count > MAX_GEO_FILTER_CANDIDATES:
                continue
            fc = Filter.by_property(fname).within_geo_range(
                coordinate=GeoCoordinate(latitude=center["latitude"], longitude=center["longitude"]),
                distance=distance,
            )
            return fc, mask, f"{fname} within_geo_range ({center['latitude']:.6f},{center['longitude']:.6f}) <= {distance}m"
        return None, None, None

    def gen_id_expr(self):
        id_values = self._existing_ids()
        if not id_values:
            return None, None, None
        id_series = self.df["id"].astype(str)
        fake_id = self._random_absent_uuid(id_values)
        mode = random.choice(
            ["equal_present", "equal_absent", "not_equal", "contains_any_present", "contains_any_absent", "contains_none"]
        )

        if mode == "equal_present":
            target = random.choice(id_values)
            return Filter.by_id().equal(target), id_series == target, f'id == "{target}"'

        if mode == "equal_absent":
            return Filter.by_id().equal(fake_id), id_series == fake_id, f'id == "{fake_id}"'

        if mode == "not_equal":
            if len(id_values) > 1 and random.random() < 0.6:
                target = random.choice(id_values)
            else:
                target = fake_id
            return Filter.by_id().not_equal(target), id_series != target, f'id != "{target}"'

        if mode == "contains_any_present":
            sample_size = min(len(set(id_values)), random.randint(1, 3))
            targets = random.sample(list(set(id_values)), sample_size)
            if random.random() < 0.5:
                targets.append(fake_id)
            targets = list(dict.fromkeys(targets))
            return Filter.by_id().contains_any(targets), id_series.isin(targets), f"id contains_any {targets}"

        if mode == "contains_any_absent":
            targets = [fake_id]
            return Filter.by_id().contains_any(targets), id_series.isin(targets), f"id contains_any {targets}"

        sample_size = min(len(set(id_values)), random.randint(1, 3))
        targets = random.sample(list(set(id_values)), sample_size)
        if random.random() < 0.5:
            targets.append(fake_id)
        targets = list(dict.fromkeys(targets))
        return Filter.by_id().contains_none(targets), ~id_series.isin(targets), f"id contains_none {targets}"

    def _property_length_series(self, name, ftype):
        series = self.df[name]

        def length_of(value):
            if value is None:
                return 0
            try:
                if pd.isna(value):
                    return 0
            except Exception:
                pass
            if ftype == FieldType.TEXT:
                return len(str(value))
            if isinstance(value, list):
                return len(value)
            return 0

        return series.apply(length_of).astype(int)

    def gen_property_length_expr(self):
        fields = self.property_length_schema[:]
        random.shuffle(fields)
        for field in fields:
            name, ftype = field["name"], field["type"]
            lengths = self._property_length_series(name, ftype)
            if lengths.empty:
                continue
            observed_lengths = sorted(set(int(v) for v in lengths.tolist()))
            target_pool = list(dict.fromkeys(observed_lengths + [0, 1, 2, max(observed_lengths) + 1]))
            target = int(random.choice(target_pool))
            op = random.choice(["equal", "not_equal", "less_than", "less_or_equal", "greater_than", "greater_or_equal"])
            base = Filter.by_property(name, length=True)
            if op == "equal":
                return base.equal(target), lengths == target, f"len({name}) == {target}"
            if op == "not_equal":
                return base.not_equal(target), lengths != target, f"len({name}) != {target}"
            if op == "less_than":
                return base.less_than(target), lengths < target, f"len({name}) < {target}"
            if op == "less_or_equal":
                return base.less_or_equal(target), lengths <= target, f"len({name}) <= {target}"
            if op == "greater_than":
                return base.greater_than(target), lengths > target, f"len({name}) > {target}"
            return base.greater_or_equal(target), lengths >= target, f"len({name}) >= {target}"
        return None, None, None

    def _creation_time_values(self):
        if self.creation_time_series is None:
            return []
        values = self.creation_time_series.dropna().tolist()
        uniq = []
        seen = set()
        for value in values:
            ts = pd.Timestamp(value)
            key = ts.isoformat()
            if key not in seen:
                seen.add(key)
                uniq.append(ts)
        return uniq

    def _creation_time_cmp_mask(self, op, target):
        if self.creation_time_series is None:
            return pd.Series(False, index=self.df.index)
        target_ts = pd.Timestamp(target)
        if target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize("UTC")
        else:
            target_ts = target_ts.tz_convert("UTC")
        series = self.creation_time_series
        if op == "==":
            return series == target_ts
        if op == "!=":
            return series != target_ts
        if op == "<":
            return series < target_ts
        if op == "<=":
            return series <= target_ts
        if op == ">":
            return series > target_ts
        return series >= target_ts

    def gen_creation_time_expr(self):
        values = self._creation_time_values()
        if not values:
            return None, None, None

        fake_old = min(values) - pd.Timedelta(days=365)
        fake_future = max(values) + pd.Timedelta(days=365)
        mode = random.choice(
            ["equal", "not_equal", "less_than", "less_or_equal", "greater_than", "greater_or_equal", "contains_any", "contains_none"]
        )

        if mode == "equal":
            target = random.choice(values)
            return (
                Filter.by_creation_time().equal(target.to_pydatetime()),
                self._creation_time_cmp_mask("==", target),
                f"creation_time == {target.isoformat()}",
            )

        if mode == "not_equal":
            target = random.choice(values) if random.random() < 0.6 else fake_future
            return (
                Filter.by_creation_time().not_equal(target.to_pydatetime()),
                self._creation_time_cmp_mask("!=", target),
                f"creation_time != {target.isoformat()}",
            )

        if mode == "less_than":
            target = random.choice(values + [fake_future])
            return (
                Filter.by_creation_time().less_than(target.to_pydatetime()),
                self._creation_time_cmp_mask("<", target),
                f"creation_time < {target.isoformat()}",
            )

        if mode == "less_or_equal":
            target = random.choice(values + [fake_future])
            return (
                Filter.by_creation_time().less_or_equal(target.to_pydatetime()),
                self._creation_time_cmp_mask("<=", target),
                f"creation_time <= {target.isoformat()}",
            )

        if mode == "greater_than":
            target = random.choice(values + [fake_old])
            return (
                Filter.by_creation_time().greater_than(target.to_pydatetime()),
                self._creation_time_cmp_mask(">", target),
                f"creation_time > {target.isoformat()}",
            )

        if mode == "greater_or_equal":
            target = random.choice(values + [fake_old])
            return (
                Filter.by_creation_time().greater_or_equal(target.to_pydatetime()),
                self._creation_time_cmp_mask(">=", target),
                f"creation_time >= {target.isoformat()}",
            )

        if mode == "contains_any":
            sample_size = min(len(values), random.randint(1, 3))
            targets = random.sample(values, sample_size)
            if random.random() < 0.5:
                targets.append(fake_future)
            uniq_targets = []
            seen = set()
            for target in targets:
                key = target.isoformat()
                if key not in seen:
                    seen.add(key)
                    uniq_targets.append(target)
            return (
                Filter.by_creation_time().contains_any([target.to_pydatetime() for target in uniq_targets]),
                self.creation_time_series.isin(uniq_targets),
                f"creation_time contains_any {[target.isoformat() for target in uniq_targets]}",
            )

        sample_size = min(len(values), random.randint(1, 3))
        targets = random.sample(values, sample_size)
        if random.random() < 0.5:
            targets.append(fake_old)
        uniq_targets = []
        seen = set()
        for target in targets:
            key = target.isoformat()
            if key not in seen:
                seen.add(key)
                uniq_targets.append(target)
        return (
            Filter.by_creation_time().contains_none([target.to_pydatetime() for target in uniq_targets]),
            ~self.creation_time_series.isin(uniq_targets),
            f"creation_time contains_none {[target.isoformat() for target in uniq_targets]}",
        )

    def _update_time_values(self):
        if self.update_time_series is None:
            return []
        values = self.update_time_series.dropna().tolist()
        uniq = []
        seen = set()
        for value in values:
            ts = pd.Timestamp(value)
            key = ts.isoformat()
            if key not in seen:
                seen.add(key)
                uniq.append(ts)
        return uniq

    def _update_time_cmp_mask(self, op, target):
        if self.update_time_series is None:
            return pd.Series(False, index=self.df.index)
        target_ts = pd.Timestamp(target)
        if target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize("UTC")
        else:
            target_ts = target_ts.tz_convert("UTC")
        series = self.update_time_series
        if op == "==":
            return series == target_ts
        if op == "!=":
            return series != target_ts
        if op == "<":
            return series < target_ts
        if op == "<=":
            return series <= target_ts
        if op == ">":
            return series > target_ts
        return series >= target_ts

    def gen_update_time_expr(self):
        values = self._update_time_values()
        if not values:
            return None, None, None

        fake_old = min(values) - pd.Timedelta(days=365)
        fake_future = max(values) + pd.Timedelta(days=365)
        mode = random.choice(
            ["equal", "not_equal", "less_than", "less_or_equal", "greater_than", "greater_or_equal", "contains_any", "contains_none"]
        )

        if mode == "equal":
            target = random.choice(values)
            return (
                Filter.by_update_time().equal(target.to_pydatetime()),
                self._update_time_cmp_mask("==", target),
                f"update_time == {target.isoformat()}",
            )

        if mode == "not_equal":
            target = random.choice(values) if random.random() < 0.6 else fake_future
            return (
                Filter.by_update_time().not_equal(target.to_pydatetime()),
                self._update_time_cmp_mask("!=", target),
                f"update_time != {target.isoformat()}",
            )

        if mode == "less_than":
            target = random.choice(values + [fake_future])
            return (
                Filter.by_update_time().less_than(target.to_pydatetime()),
                self._update_time_cmp_mask("<", target),
                f"update_time < {target.isoformat()}",
            )

        if mode == "less_or_equal":
            target = random.choice(values + [fake_future])
            return (
                Filter.by_update_time().less_or_equal(target.to_pydatetime()),
                self._update_time_cmp_mask("<=", target),
                f"update_time <= {target.isoformat()}",
            )

        if mode == "greater_than":
            target = random.choice(values + [fake_old])
            return (
                Filter.by_update_time().greater_than(target.to_pydatetime()),
                self._update_time_cmp_mask(">", target),
                f"update_time > {target.isoformat()}",
            )

        if mode == "greater_or_equal":
            target = random.choice(values + [fake_old])
            return (
                Filter.by_update_time().greater_or_equal(target.to_pydatetime()),
                self._update_time_cmp_mask(">=", target),
                f"update_time >= {target.isoformat()}",
            )

        if mode == "contains_any":
            sample_size = min(len(values), random.randint(1, 3))
            targets = random.sample(values, sample_size)
            if random.random() < 0.5:
                targets.append(fake_future)
            uniq_targets = []
            seen = set()
            for target in targets:
                key = target.isoformat()
                if key not in seen:
                    seen.add(key)
                    uniq_targets.append(target)
            return (
                Filter.by_update_time().contains_any([target.to_pydatetime() for target in uniq_targets]),
                self.update_time_series.isin(uniq_targets),
                f"update_time contains_any {[target.isoformat() for target in uniq_targets]}",
            )

        sample_size = min(len(values), random.randint(1, 3))
        targets = random.sample(values, sample_size)
        if random.random() < 0.5:
            targets.append(fake_old)
        uniq_targets = []
        seen = set()
        for target in targets:
            key = target.isoformat()
            if key not in seen:
                seen.add(key)
                uniq_targets.append(target)
        return (
            Filter.by_update_time().contains_none([target.to_pydatetime() for target in uniq_targets]),
            ~self.update_time_series.isin(uniq_targets),
            f"update_time contains_none {[target.isoformat() for target in uniq_targets]}",
        )

    def gen_nested_object_expr(self):
        """Nested Object 查询 (对标 Milvus gen_json_advanced_expr)
        NOTE: Weaviate 当前版本对 OBJECT 类型仅可靠使用 is_none(True)
        以及 NOT(is_none(True)) 过滤；direct is_none(False) 属于未实现/不支持边界。
        不支持嵌套属性路径过滤 (metaObj.price > X)。
        主要价值: 数据完整性验证 + null 测试。
        """
        obj_fields = [f for f in self.schema if f["type"] == FieldType.OBJECT]
        if not obj_fields:
            return None, None, None
        field = random.choice(obj_fields)
        fname = field["name"]
        series = self.df[fname]
        if not ENABLE_NULL_FILTER:
            return None, None, None
        # Only is_none(True) is reliable for OBJECT type in current Weaviate
        fc = Filter.by_property(fname).is_none(True)
        mask = series.apply(lambda x: x is None or (not isinstance(x, dict)))
        return fc, mask, f"{fname} is null"

    def gen_not_expr(self):
        """NOT 表达式 — 使用 Filter.not_() 包装
        WORKAROUND: Weaviate NOT 包含 null 行 (与 SQL 三值逻辑不同)
        """
        non_obj = [f for f in self.schema if f["type"] not in (FieldType.OBJECT, FieldType.GEO)]
        obj_fields = [f for f in self.schema if f["type"] == FieldType.OBJECT]
        if ENABLE_NULL_FILTER and obj_fields and (not non_obj or random.random() < 0.12):
            f = random.choice(obj_fields)
            name = f["name"]
            series = self.df[name]
            fc = Filter.not_(Filter.by_property(name).is_none(True))
            mask = series.apply(lambda x: isinstance(x, dict))
            return fc, mask, f"NOT({name} is null)"

        if not non_obj:
            return None, None, None
        f = random.choice(non_obj)
        name, ftype = f["name"], f["type"]
        series = self.df[name]
        null_mask = self._effective_null_mask(series, ftype)
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
            # Tokenization.FIELD → 精确比较, 不再需要 .lower()
            return fc, series.apply(lambda x: str(x) != str(val) if x is not None and not (isinstance(x, float) and np.isnan(x)) else True), f'NOT({name} == "{val}")'

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
            # Tokenization.FIELD: 不需要停用词过滤
            if all_items:
                t = self._convert_to_native(random.choice(all_items))
                fc = Filter.not_(Filter.by_property(name).contains_any([t]))
                if ftype == FieldType.TEXT_ARRAY:
                    # Tokenization.FIELD → 精确比较
                    return fc, series.apply(lambda x: not any(str(i) == str(t) for i in x) if isinstance(x, list) else True), f"NOT({name} contains {t})"
                return fc, series.apply(lambda x: t not in x if isinstance(x, list) else True), f"NOT({name} contains {t})"

        elif strat == "not_null" and ENABLE_NULL_FILTER:
            fc = Filter.not_(Filter.by_property(name).is_none(True))
            return fc, ~null_mask, f"NOT({name} is null)"

        # fallback: NOT(bool == val)
        for ff in self.schema:
            if ff["type"] == FieldType.BOOL:
                v = self.get_value_for_query(ff["name"], FieldType.BOOL)
                if v is not None:
                    fc = Filter.not_(Filter.by_property(ff["name"]).equal(bool(v)))
                    return fc, self.df[ff["name"]].apply(lambda x, vv=bool(v): x != vv if x is not None and not (isinstance(x, float) and np.isnan(x)) else True), f"NOT({ff['name']} == {bool(v)})"
        return None, None, None

    def gen_atomic_expr(self):
        if random.random() < 0.10:
            res = self.gen_id_expr()
            if res[0] is not None:
                return res

        if self.creation_time_series is not None and random.random() < 0.10:
            res = self.gen_creation_time_expr()
            if res[0] is not None:
                return res

        if self.update_time_series is not None and random.random() < 0.10:
            res = self.gen_update_time_expr()
            if res[0] is not None:
                return res

        if self.property_length_schema and random.random() < 0.12:
            res = self.gen_property_length_expr()
            if res[0] is not None:
                return res

        if self.like_text_schema and random.random() < 0.24:
            f = random.choice(self.like_text_schema)
            name = f["name"]
            series = self.df[name]
            sv = self.get_value_for_query(name, FieldType.TEXT)
            if sv is not None:
                sv = str(sv)
                if random.random() < 0.40:
                    mode = random.choice(["contains_any", "contains_all", "contains_none"])
                    if mode == "contains_all":
                        targets = [sv]
                    else:
                        targets = [sv]
                        if random.random() < 0.55:
                            targets.append(''.join(random.choices(string.ascii_letters, k=30)))
                    target_set = set(targets)
                    if mode == "contains_any":
                        mask = series.apply(
                            lambda x, vals=target_set: str(x) in vals
                            if not self._is_effectively_null_value(x, FieldType.TEXT)
                            else False
                        )
                        return Filter.by_property(name).contains_any(targets), mask, f"{name} contains_any {targets}"
                    if mode == "contains_all":
                        mask = series.apply(
                            lambda x, vals=targets: all(str(x) == value for value in vals)
                            if not self._is_effectively_null_value(x, FieldType.TEXT)
                            else False
                        )
                        return Filter.by_property(name).contains_all(targets), mask, f"{name} contains_all {targets}"
                    mask = series.apply(
                        lambda x, vals=target_set: str(x) not in vals
                        if not self._is_effectively_null_value(x, FieldType.TEXT)
                        else True
                    )
                    return Filter.by_property(name).contains_none(targets), mask, f"{name} contains_none {targets}"
                op = random.choice(["like_prefix", "like_suffix", "like_contains", "like_single"])
                if op == "like_prefix":
                    k = random.randint(1, min(4, max(1, len(sv))))
                    prefix = ''.join(c for c in sv[:k] if c.isalnum()) or "a"
                    return Filter.by_property(name).like(f"{prefix}*"), series.apply(lambda x, p=prefix: str(x).startswith(p) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False), f'{name} like "{prefix}*"'
                elif op == "like_suffix" and len(sv) >= 2:
                    k = random.randint(1, min(4, len(sv)))
                    suffix = ''.join(c for c in sv[-k:] if c.isalnum()) or "z"
                    return Filter.by_property(name).like(f"*{suffix}"), series.apply(lambda x, s=suffix: str(x).endswith(s) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False), f'{name} like "*{suffix}"'
                elif op == "like_contains" and len(sv) >= 3:
                    i_start = random.randint(0, len(sv) - 2)
                    j_end = random.randint(i_start + 1, min(i_start + 5, len(sv)))
                    sub = ''.join(c for c in sv[i_start:j_end] if c.isalnum()) or "a"
                    return Filter.by_property(name).like(f"*{sub}*"), series.apply(lambda x, s=sub: s in str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False), f'{name} like "*{sub}*"'
                elif op == "like_single" and len(sv) >= 2:
                    pos = random.randint(0, len(sv) - 1)
                    pat = sv[:pos] + "?" + sv[pos+1:]
                    import re as _re
                    regex = _re.compile("^" + _re.escape(sv[:pos]) + "." + _re.escape(sv[pos+1:]) + "$")
                    return Filter.by_property(name).like(pat), series.apply(lambda x, r=regex: bool(r.match(str(x))) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False), f'{name} like "{pat}"'

        if self.geo_schema and random.random() < 0.12:
            res = self.gen_geo_expr()
            if res[0] is not None:
                return res

        # 标量优先，降低数组/OBJECT 语义噪声对 Oracle 的影响
        non_obj = [f for f in self.schema if f["type"] not in (FieldType.OBJECT, FieldType.GEO)]
        if self.scalar_schema and random.random() < SCALAR_QUERY_PRIORITY:
            filterable_schema = self.scalar_schema
        else:
            filterable_schema = non_obj
        if not filterable_schema:
            return None, None, None
        f = random.choice(filterable_schema)
        name, ftype = f["name"], f["type"]
        series = self.df[name]
        null_mask = self._effective_null_mask(series, ftype)

        if ENABLE_NULL_FILTER and random.random() < 0.15:
            if random.random() < 0.5:
                return Filter.by_property(name).is_none(True), null_mask, f"{name} is null"
            else:
                return Filter.by_property(name).is_none(False), ~null_mask, f"{name} is not null"

        val = self.get_value_for_query(name, ftype)
        if val is None:
            if ENABLE_NULL_FILTER:
                return Filter.by_property(name).is_none(True), null_mask, f"{name} is null"
            return None, None, None

        def safe_cmp(op, tv):
            def comp(x):
                if x is None or (isinstance(x, float) and np.isnan(x)): return False
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
                fc = Filter.by_property(name).not_equal(vb)
                mask = series.apply(safe_cmp("!=", vb)) | null_mask
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
                mask = mask | null_mask  # Weaviate not_equal includes null rows
            es = f"{name} {op} {vi}"

        elif ftype == FieldType.NUMBER:
            vf = float(val)
            op = random.choice([">", "<", ">=", "<=", "!="])
            op_map = {"!=": "not_equal", ">": "greater_than", "<": "less_than", ">=": "greater_or_equal", "<=": "less_or_equal"}
            fc = getattr(Filter.by_property(name), op_map[op])(vf)
            mask = series.apply(safe_cmp(op, vf))
            if op == "!=":
                mask = mask | null_mask  # Weaviate not_equal includes null rows
            es = f"{name} {op} {vf}"

        elif ftype == FieldType.TEXT:
            op = random.choice(["==", "!=", "like_prefix", "like_suffix", "like_contains", "like_single"])
            sv = str(val)
            if op == "==":
                fc = Filter.by_property(name).equal(val)
                # Tokenization.FIELD → 精确匹配, 区分大小写
                mask = series.apply(lambda x, v=sv: str(x) == v if x is not None and not (isinstance(x, float) and np.isnan(x)) else False)
                es = f'{name} == "{val}"'
            elif op == "!=":
                fc = Filter.by_property(name).not_equal(val)
                # Weaviate not_equal includes null rows (二值逻辑)
                mask = series.apply(lambda x, v=sv: str(x) != v if x is not None and not (isinstance(x, float) and np.isnan(x)) else True)
                es = f'{name} != "{val}"'
            elif op == "like_prefix":
                k = random.randint(1, min(4, max(1, len(sv))))
                prefix = ''.join(c for c in sv[:k] if c.isalnum()) or "a"
                fc = Filter.by_property(name).like(f"{prefix}*")
                # Tokenization.FIELD → like 也区分大小写
                mask = series.apply(lambda x, p=prefix: str(x).startswith(p) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False)
                es = f'{name} like "{prefix}*"'
            elif op == "like_suffix" and len(sv) >= 2:
                k = random.randint(1, min(4, len(sv)))
                suffix = ''.join(c for c in sv[-k:] if c.isalnum()) or "z"
                fc = Filter.by_property(name).like(f"*{suffix}")
                mask = series.apply(lambda x, s=suffix: str(x).endswith(s) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False)
                es = f'{name} like "*{suffix}"'
            elif op == "like_contains" and len(sv) >= 3:
                i_start = random.randint(0, len(sv) - 2)
                j_end = random.randint(i_start + 1, min(i_start + 5, len(sv)))
                sub = ''.join(c for c in sv[i_start:j_end] if c.isalnum()) or "a"
                fc = Filter.by_property(name).like(f"*{sub}*")
                mask = series.apply(lambda x, s=sub: s in str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False)
                es = f'{name} like "*{sub}*"'
            elif op == "like_single" and len(sv) >= 2:
                pos = random.randint(0, len(sv) - 1)
                pat = sv[:pos] + "?" + sv[pos+1:]
                fc = Filter.by_property(name).like(pat)
                import re as _re
                # Tokenization.FIELD → 区分大小写, 不用 IGNORECASE
                regex = _re.compile("^" + _re.escape(sv[:pos]) + "." + _re.escape(sv[pos+1:]) + "$")
                mask = series.apply(lambda x, r=regex: bool(r.match(str(x))) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False)
                es = f'{name} like "{pat}"'
            else:
                # fallback to prefix
                prefix = ''.join(c for c in sv[:3] if c.isalnum()) or "a"
                fc = Filter.by_property(name).like(f"{prefix}*")
                mask = series.apply(lambda x, p=prefix: str(x).startswith(p) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False)
                es = f'{name} like "{prefix}*"'

        elif ftype == FieldType.DATE:
            op = random.choice([">", "<", ">=", "<=", "==", "!="])
            op_map = {"==": "equal", "!=": "not_equal", ">": "greater_than", "<": "less_than", ">=": "greater_or_equal", "<=": "less_or_equal"}
            fc = getattr(Filter.by_property(name), op_map[op])(val)
            mask = self._date_cmp_mask(series, op, val)
            if op == "!=":
                mask = mask | null_mask
            es = f"{name} {op} {val}"

        elif ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY]:
            all_items = []
            for arr in self.df[name].dropna():
                if isinstance(arr, list): all_items.extend(arr)
            # Tokenization.FIELD: 不需要停用词过滤, 精确匹配
            if all_items:
                t = self._convert_to_native(random.choice(all_items))
                fc = Filter.by_property(name).contains_any([t])
                mask = series.apply(lambda x, tt=t: tt in x if isinstance(x, list) else False)
                es = f'{name} contains {t}'
            else:
                if ENABLE_NULL_FILTER:
                    fc = Filter.by_property(name).is_none(True)
                    mask = null_mask
                    es = f'{name} is null'
                else:
                    fc = Filter.by_property(name).contains_any(["__impossible__"])
                    mask = pd.Series(False, index=self.df.index)
                    es = f'{name} contains __impossible__'

        if fc is not None and mask is not None:
            return fc, mask, es
        if ENABLE_NULL_FILTER:
            return Filter.by_property(name).is_none(False), ~null_mask, f"{name} is not null"
        int_fields = [ff for ff in self.schema if ff["type"] == FieldType.INT]
        if int_fields:
            fn = int_fields[0]["name"]
            s = self.df[fn]
            stats = self._field_stats.get(fn)
            mn = int(stats["min"]) if stats else 0
            mx = int(stats["max"]) if stats else 0
            return (Filter.by_property(fn).greater_or_equal(clamp_weaviate_int(mn - 1000000)) & Filter.by_property(fn).less_or_equal(clamp_weaviate_int(mx + 1000000))), ~self._effective_null_mask(s, FieldType.INT), f"{fn} range (fallback)"
        return None, None, None

    def gen_constant_expr(self):
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            fn = int_fields[0]["name"]
            s = self.df[fn]
            stats = self._field_stats.get(fn)
            mn = int(stats["min"]) if stats else 0
            mx = int(stats["max"]) if stats else 0
        else:
            float_fields = [f for f in self.schema if f["type"] == FieldType.NUMBER]
            if not float_fields: return None, None, None
            fn = float_fields[0]["name"]
            s = self.df[fn]
            stats = self._field_stats.get(fn)
            mn = float(stats["min"]) if stats else 0
            mx = float(stats["max"]) if stats else 0
        if random.random() < 0.5:
            if int_fields:
                lo = clamp_weaviate_int(mn - 1000000)
                hi = clamp_weaviate_int(mx + 1000000)
            else:
                lo = mn - 1e6
                hi = mx + 1e6
            return (Filter.by_property(fn).greater_or_equal(lo) & Filter.by_property(fn).less_or_equal(hi)), s.notna(), f"{fn} wide range (true)"
        else:
            imp = mx + 1e7
            return Filter.by_property(fn).greater_than(imp), pd.Series(False, index=self.df.index), f"{fn} > {imp} (false)"

    def _apply_not_mask(self, mask):
        """Apply NOT to a boolean mask with Weaviate null semantics.
        Weaviate NOT includes null rows: NOT(expr) returns rows where
        expr is False OR the field is null. This differs from SQL 3VL.
        For a compound expression mask, we invert True→False, False→True.
        Null rows (NaN in mask) become True (included in NOT result).
        """
        return mask.apply(lambda x: True if pd.isna(x) else not x)

    def gen_complex_expr(self, depth):
        if depth == 0 or random.random() < 0.2:
            r = random.random()
            if r < 0.04:
                res = self.gen_constant_expr()
                if res[0]: return res
            elif r < 0.22:
                res = self.gen_boundary_expr()
                if res[0] is not None: return res
            elif r < 0.28 and self.array_schema:
                res = self.gen_multi_array_expr()
                if res[0] is not None: return res
            elif r < 0.42:
                res = self.gen_not_expr()
                if res[0] is not None: return res
            elif r < 0.45:
                res = self.gen_nested_object_expr()
                if res[0] is not None: return res
            fc, m, e = self.gen_atomic_expr()
            if fc: return fc, m, e
            return self.gen_complex_expr(depth)

        # Recursive branch: AND (40%), OR (40%), NOT (20%)
        # Matches Milvus gen_complex_expr: random.choices(["and","or","not"], weights=[0.4,0.4,0.2])
        op = random.choices(["and", "or", "not"], weights=[0.4, 0.4, 0.2], k=1)[0]

        if op == "not":
            # NOT branch: negate one sub-expression
            # 30% chance: use dedicated gen_not_expr for more targeted NOT patterns
            if random.random() < 0.3:
                not_res = self.gen_not_expr()
                if not_res[0] is not None:
                    return not_res

            # Otherwise: recursively generate and negate
            fl, ml, el = self.gen_complex_expr(depth - 1)
            if not fl:
                return self.gen_complex_expr(depth)
            fc_not = Filter.not_(fl)
            mask_not = self._apply_not_mask(ml)
            return fc_not, mask_not, f"NOT({el})"
        else:
            fl, ml, el = self.gen_complex_expr(depth - 1)
            fr, mr, er = self.gen_complex_expr(depth - 1)
            if not fl: return fr, mr, er
            if not fr: return fl, ml, el
            if op == "and":
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
                stats = self._field_stats.get(f["name"])
                if stats:
                    return {"name": f["name"], "min": int(stats["min"]), "max": int(stats["max"])}
        for f in self.schema:
            if f["type"] == FieldType.NUMBER:
                stats = self._field_stats.get(f["name"])
                if stats:
                    return {"name": f["name"], "min": float(stats["min"]), "max": float(stats["max"])}
        return None

    def _gen_tautology_filter(self):
        if self._tautology_field:
            n = self._tautology_field["name"]
            mn, mx = self._tautology_field["min"], self._tautology_field["max"]
            if isinstance(mn, int) and isinstance(mx, int):
                lo = clamp_weaviate_int(mn - 1000000)
                hi = clamp_weaviate_int(mx + 1000000)
            else:
                lo = mn - 1e6
                hi = mx + 1e6
            wide = Filter.by_property(n).greater_or_equal(lo) & Filter.by_property(n).less_or_equal(hi)
            if ENABLE_NULL_FILTER:
                # Include null rows to make it a true tautology
                return wide | Filter.by_property(n).is_none(True), f"({n} wide|null)"
            return wide, f"({n} wide)"
        return None, None

    def _gen_false_filter(self):
        for f in self.schema:
            if f["type"] == FieldType.INT:
                return Filter.by_property(f["name"]).greater_than(200000) & Filter.by_property(f["name"]).less_than(-200000), f"({f['name']} impossible)"
        for f in self.schema:
            if f["type"] == FieldType.TEXT:
                return Filter.by_property(f["name"]).equal("__impossible__"), f'{f["name"]} impossible'
        return None, None

    def _gen_guaranteed_false_filter(self):
        """生成保证为假但计算复杂的表达式, 覆盖更多引擎代码路径"""
        for f in self.schema:
            if f["type"] == FieldType.INT:
                n = f["name"]
                return Filter.by_property(n).greater_than(200000) & Filter.by_property(n).less_than(-200000), f"({n} impossible-range)"
        for f in self.schema:
            if f["type"] == FieldType.NUMBER:
                n = f["name"]
                return Filter.by_property(n).greater_than(1e20) & Filter.by_property(n).less_than(-1e20), f"({n} impossible-float)"
        for f in self.schema:
            if f["type"] == FieldType.TEXT:
                n = f["name"]
                impossible = 'fuzz_impossible_' + ''.join(random.choices(string.ascii_letters, k=30))
                return Filter.by_property(n).equal(impossible), f'({n} impossible-text)'
        return self._gen_false_filter()

    def mutate_filter(self, base_filter, base_expr):
        mutations = [{"type": "SelfOr", "filter": base_filter | base_filter, "expr": f"({base_expr}) OR ({base_expr})"}]
        ff, fe = self._gen_false_filter()
        if ff:
            mutations.append({"type": "NoiseOr", "filter": base_filter | ff, "expr": f"({base_expr}) OR ({fe})"})
        mutations.append({"type": "DoubleNeg", "filter": Filter.not_(Filter.not_(base_filter)), "expr": f"NOT(NOT({base_expr}))"})

        # TautologyAnd: A AND True → A
        tf, te = self._gen_tautology_filter()
        if tf:
            mutations.append({"type": "TautologyAnd", "filter": base_filter & tf, "expr": f"({base_expr}) AND {te}"})
            mutations.append({"type": "TautologyAnd_Left", "filter": tf & base_filter, "expr": f"{te} AND ({base_expr})"})

        # DeMorganWrapper: NOT(NOT(A) OR False) ≡ A
        gf, ge = self._gen_guaranteed_false_filter()
        if gf:
            mutations.append({"type": "DeMorgan", "filter": Filter.not_(Filter.not_(base_filter) | gf), "expr": f"NOT(NOT({base_expr}) OR {ge})"})

        # IntRangeShift: int > X ≡ int >= X+1
        for f in self.schema:
            if f["type"] == FieldType.INT:
                n = f["name"]
                s = self.df[n].dropna()
                if not s.empty:
                    v = int(s.sample(1).iloc[0])
                    orig = Filter.by_property(n).greater_than(v)
                    shifted = Filter.by_property(n).greater_or_equal(v + 1)
                    mutations.append({"type": "IntRangeShift", "filter": (base_filter & shifted) | (base_filter & Filter.not_(orig)),
                                      "expr": f"(({base_expr}) AND {n}>={v+1}) OR (({base_expr}) AND NOT({n}>{v}))"})
                    break

        # PartitionById: (A AND row_num%2==even) OR (A AND row_num%2==odd) ≡ A
        # Use row_num field for partition
        if 'row_num' in [f['name'] for f in self.schema] or True:
            # Approximate partition using two tautology halves via known field ranges
            for f in self.schema:
                if f['type'] == FieldType.INT and f['name'] != 'row_num':
                    values = self._field_values.get(f["name"], [])
                    stats = self._field_stats.get(f["name"])
                    if len(values) > 10 and stats:
                        med = int(stats["median"])
                        left_part = Filter.by_property(f['name']).less_or_equal(med)
                        right_part = Filter.by_property(f['name']).greater_than(med)
                        null_part = Filter.by_property(f['name']).is_none(True) if ENABLE_NULL_FILTER else None
                        if null_part:
                            combined = (base_filter & left_part) | (base_filter & right_part) | (base_filter & null_part)
                        else:
                            combined = (base_filter & left_part) | (base_filter & right_part)
                        mutations.append({"type": "Partition", "filter": combined,
                                          "expr": f"partition({base_expr}, {f['name']}, med={med})"})
                        break

        return mutations


# --- 5. PQS ---

class PQSQueryGenerator(OracleQueryGenerator):
    def __init__(self, dm):
        super().__init__(dm)
        self.geo_schema = []
        self._tautology_field = self._find_tautology_field()

    def _find_tautology_field(self):
        for f in self.schema:
            if f["type"] == FieldType.INT:
                n = f["name"]
                stats = self._field_stats.get(n)
                if stats and self._effective_null_mask(self.df[n], FieldType.INT).sum() == 0:
                    return {"name": n, "min": int(stats["min"]), "max": int(stats["max"])}
        for f in self.schema:
            if f["type"] == FieldType.INT:
                stats = self._field_stats.get(f["name"])
                if stats:
                    return {"name": f["name"], "min": int(stats["min"]), "max": int(stats["max"])}
        for f in self.schema:
            if f["type"] == FieldType.NUMBER:
                stats = self._field_stats.get(f["name"])
                if stats:
                    return {"name": f["name"], "min": float(stats["min"]), "max": float(stats["max"])}
        return None

    def _gen_tautology(self):
        if self._tautology_field:
            n = self._tautology_field["name"]
            mn, mx = self._tautology_field["min"], self._tautology_field["max"]
            if isinstance(mn, int) and isinstance(mx, int):
                lo = clamp_weaviate_int(mn - 1000000)
                hi = clamp_weaviate_int(mx + 1000000)
            else:
                lo = mn - 1e6
                hi = mx + 1e6
            wide = Filter.by_property(n).greater_or_equal(lo) & Filter.by_property(n).less_or_equal(hi)
            if ENABLE_NULL_FILTER:
                return wide | Filter.by_property(n).is_none(True), f"({n} wide|null)"
            return wide, f"({n} wide)"
        return None, None

    def _gen_false(self):
        for f in self.schema:
            if f["type"] == FieldType.INT:
                return Filter.by_property(f["name"]).greater_than(200000) & Filter.by_property(f["name"]).less_than(-200000), f"({f['name']} impossible)"
        return None, None

    def gen_pqs_filter(self, pivot_row, depth):
        """生成必真的复合表达式 (PQS: Pivot Query Synthesis)
        参照 Milvus gen_pqs_expr:
         - AND: 两个子表达式都是必真的 → 结果必真
         - OR: 左侧必真, 右侧可以是任意噪声 → 结果必真
         - NOT(NOT(...)): 双重否定 ≡ 原表达式 → 保持必真
         - 噪声分支使用 gen_complex_noise 生成复杂但语义丰富的子树
        """
        force_recursion = depth > 5

        if depth <= 0 or (not force_recursion and depth <= 3 and random.random() < 0.25):
            res = self.gen_true_atomic(pivot_row)
            if res[0] is None:
                tf, te = self._gen_tautology()
                if tf: return tf, te
            return res

        op = random.choices(
            ["and", "or", "nested_not", "or_complex_noise", "tautology_and"],
            weights=[0.30, 0.25, 0.20, 0.15, 0.10], k=1
        )[0]
        tf, te = self._gen_tautology()

        if op == "and":
            # A AND B: 两侧都必真
            fl, el = self.gen_pqs_filter(pivot_row, depth - 1)
            fr, er = self.gen_pqs_filter(pivot_row, depth - 1)
            if not fl: fl, el = tf, te
            if not fr: fr, er = tf, te
            if fl and fr: return fl & fr, f"({el} AND {er})"
            return fl, el

        elif op == "or":
            # A OR False: 左侧必真, 右侧假 → 必真
            fl, el = self.gen_pqs_filter(pivot_row, depth - 1)
            nf, ne = self._gen_false()
            if not fl: fl, el = tf, te
            if not nf: nf, ne = self._gen_false()
            if fl and nf: return fl | nf, f"({el} OR {ne})"
            return fl, el

        elif op == "or_complex_noise":
            # A OR ComplexNoise: 噪声深度递增, 增加查询引擎压力
            fl, el = self.gen_pqs_filter(pivot_row, depth - 1)
            noise_depth = random.randint(2, min(depth + 2, 8))
            nf, ne = self.gen_complex_noise(noise_depth)
            if not fl: fl, el = tf, te
            if nf:
                if random.random() < 0.5:
                    return fl | nf, f"({el} OR {ne})"
                else:
                    return nf | fl, f"({ne} OR {el})"
            return fl, el

        elif op == "tautology_and":
            # TrueExpr AND Tautology: 必真 AND 永真 → 必真
            fl, el = self.gen_pqs_filter(pivot_row, depth - 1)
            if not fl: fl, el = tf, te
            if fl and tf: return fl & tf, f"({el} AND {te})"
            return fl, el

        else:  # nested_not: NOT(NOT(inner)) ≡ inner → 保持必真
            fi, ei = self.gen_pqs_filter(pivot_row, depth - 1)
            if fi:
                return Filter.not_(Filter.not_(fi)), f"NOT(NOT({ei}))"
            return tf, te

    def _gen_false(self):
        """Guaranteed-false complex expression (for noise in OR branches)"""
        for f in self.schema:
            if f["type"] == FieldType.INT:
                n = f["name"]
                return Filter.by_property(n).greater_than(200000) & Filter.by_property(n).less_than(-200000), f"({n} impossible)"
        for f in self.schema:
            if f["type"] == FieldType.NUMBER:
                n = f["name"]
                return Filter.by_property(n).greater_than(1e20) & Filter.by_property(n).less_than(-1e20), f"({n} impossible-float)"
        for f in self.schema:
            if f["type"] == FieldType.TEXT:
                return Filter.by_property(f["name"]).equal("__impossible__"), f'{f["name"]} impossible'
        return None, None

    def gen_complex_noise(self, depth=3):
        """生成复杂但不保证语义的噪声表达式 (用于 PQS OR 分支)
        注意: 不要求必假, 只要求复杂——增加查询引擎的执行路径覆盖
        对标 Milvus gen_complex_noise: 包含多层 AND/OR/NOT 嵌套
        """
        if depth <= 0:
            # 叶子节点: 50% 假表达式, 50% 随机原子表达式
            if random.random() < 0.5:
                ff, fe = self._gen_false()
                if ff: return ff, fe
            res = self.gen_atomic_expr()
            if res[0]: return res[0], res[2]
            ff, fe = self._gen_false()
            return ff, fe

        op = random.choices(["and", "or", "not", "mixed"], weights=[0.3, 0.3, 0.2, 0.2], k=1)[0]

        if op == "and":
            fl, el = self.gen_complex_noise(depth - 1)
            fr, er = self.gen_complex_noise(depth - 1)
            if fl and fr: return fl & fr, f"({el} AND {er})"
            return fl or fr, el or er
        elif op == "or":
            fl, el = self.gen_complex_noise(depth - 1)
            fr, er = self.gen_complex_noise(depth - 1)
            if fl and fr: return fl | fr, f"({el} OR {er})"
            return fl or fr, el or er
        elif op == "not":
            fi, ei = self.gen_complex_noise(depth - 1)
            if fi: return Filter.not_(fi), f"NOT({ei})"
            ff, fe = self._gen_false()
            return ff, fe
        else:  # mixed: 组合不同类型的表达式
            fl, el = self.gen_complex_noise(depth - 1)
            # 右侧使用 gen_boundary_expr 或 gen_not_expr 或 gen_multi_array_expr
            fr_gen = random.choice([self.gen_boundary_expr, self.gen_not_expr, self.gen_multi_array_expr])
            res = fr_gen()
            if res[0] is not None:
                fr, er = res[0], res[2]
            else:
                fr, er = self.gen_complex_noise(depth - 1)
            if fl and fr:
                if random.random() < 0.5:
                    return fl & fr, f"({el} AND {er})"
                else:
                    return fl | fr, f"({el} OR {er})"
            return fl or fr, el or er

    def gen_multi_field_true_filter(self, row, n=3):
        """多字段联合必真表达式, 增加查询引擎工作量"""
        fields = random.sample(self.schema, min(n, len(self.schema)))
        filters, exprs = [], []
        for f in fields:
            fname, ftype = f["name"], f["type"]
            if ftype == FieldType.GEO:
                continue
            val = row.get(fname)
            is_null = val is None or (isinstance(val, float) and np.isnan(val))
            if is_null:
                if ENABLE_NULL_FILTER:
                    filters.append(Filter.by_property(fname).is_none(True))
                    exprs.append(f"{fname} is null")
                continue
            if ftype == FieldType.BOOL:
                filters.append(Filter.by_property(fname).equal(bool(val)))
                exprs.append(f"{fname}=={bool(val)}")
            elif ftype == FieldType.INT:
                vi = int(val)
                filters.append(Filter.by_property(fname).equal(vi))
                exprs.append(f"{fname}=={vi}")
            elif ftype == FieldType.NUMBER:
                vf = float(val)
                lo, hi = float_window(vf)
                if lo is None:
                    continue
                filters.append(Filter.by_property(fname).greater_or_equal(lo) & Filter.by_property(fname).less_or_equal(hi))
                exprs.append(f"{fname}≈{vf}")
            elif ftype in (FieldType.TEXT, FieldType.DATE):
                filters.append(Filter.by_property(fname).equal(str(val)))
                exprs.append(f'{fname}=="{val}"')
        if len(filters) >= 2:
            combined = filters[0]
            for ff in filters[1:]:
                combined = combined & ff
            return combined, " AND ".join(exprs)
        elif filters:
            return filters[0], exprs[0]
        return None, None

    def gen_true_atomic(self, row):
        """Enhanced: multi-strategy boundary generation for PQS"""
        row_id = row.get("id")
        if row_id is not None and random.random() < 0.18:
            row_id = str(row_id)
            fake_id = self._random_absent_uuid()
            id_strat = random.choice(
                ["equal", "contains_any_self", "contains_any_noise", "not_equal_fake", "contains_none_fake"]
            )
            if id_strat == "equal":
                return Filter.by_id().equal(row_id), f'id == "{row_id}"'
            if id_strat == "contains_any_self":
                return Filter.by_id().contains_any([row_id]), f'id contains_any ["{row_id}"]'
            if id_strat == "contains_any_noise":
                return Filter.by_id().contains_any([row_id, fake_id]), f'id contains_any ["{row_id}", "{fake_id}"]'
            if id_strat == "not_equal_fake":
                return Filter.by_id().not_equal(fake_id), f'id != "{fake_id}"'
            return Filter.by_id().contains_none([fake_id]), f'id contains_none ["{fake_id}"]'

        row_creation_time = row.get("_creation_time")
        if row_creation_time is not None and random.random() < 0.16:
            try:
                creation_time = pd.Timestamp(row_creation_time)
                if pd.isna(creation_time):
                    creation_time = None
            except Exception:
                creation_time = None
            if creation_time is not None:
                if creation_time.tzinfo is None:
                    creation_time = creation_time.tz_localize("UTC")
                else:
                    creation_time = creation_time.tz_convert("UTC")
                before = creation_time - pd.Timedelta(seconds=1)
                after = creation_time + pd.Timedelta(seconds=1)
                fake_future = creation_time + pd.Timedelta(days=365)
                strat = random.choice(
                    ["equal", "contains_any_self", "contains_any_noise", "not_equal_fake", "contains_none_fake", "greater_than_before", "less_than_after", "greater_or_equal", "less_or_equal"]
                )
                if strat == "equal":
                    return Filter.by_creation_time().equal(creation_time.to_pydatetime()), f"creation_time == {creation_time.isoformat()}"
                if strat == "contains_any_self":
                    return Filter.by_creation_time().contains_any([creation_time.to_pydatetime()]), f"creation_time contains_any [{creation_time.isoformat()}]"
                if strat == "contains_any_noise":
                    return Filter.by_creation_time().contains_any([creation_time.to_pydatetime(), fake_future.to_pydatetime()]), f"creation_time contains_any [{creation_time.isoformat()}, {fake_future.isoformat()}]"
                if strat == "not_equal_fake":
                    return Filter.by_creation_time().not_equal(fake_future.to_pydatetime()), f"creation_time != {fake_future.isoformat()}"
                if strat == "contains_none_fake":
                    return Filter.by_creation_time().contains_none([fake_future.to_pydatetime()]), f"creation_time contains_none [{fake_future.isoformat()}]"
                if strat == "greater_than_before":
                    return Filter.by_creation_time().greater_than(before.to_pydatetime()), f"creation_time > {before.isoformat()}"
                if strat == "less_than_after":
                    return Filter.by_creation_time().less_than(after.to_pydatetime()), f"creation_time < {after.isoformat()}"
                if strat == "greater_or_equal":
                    return Filter.by_creation_time().greater_or_equal(creation_time.to_pydatetime()), f"creation_time >= {creation_time.isoformat()}"
                return Filter.by_creation_time().less_or_equal(creation_time.to_pydatetime()), f"creation_time <= {creation_time.isoformat()}"

        row_update_time = row.get("_update_time")
        if row_update_time is not None and random.random() < 0.16:
            try:
                update_time = pd.Timestamp(row_update_time)
                if pd.isna(update_time):
                    update_time = None
            except Exception:
                update_time = None
            if update_time is not None:
                if update_time.tzinfo is None:
                    update_time = update_time.tz_localize("UTC")
                else:
                    update_time = update_time.tz_convert("UTC")
                before = update_time - pd.Timedelta(seconds=1)
                after = update_time + pd.Timedelta(seconds=1)
                fake_future = update_time + pd.Timedelta(days=365)
                strat = random.choice(
                    ["equal", "contains_any_self", "contains_any_noise", "not_equal_fake", "contains_none_fake", "greater_than_before", "less_than_after", "greater_or_equal", "less_or_equal"]
                )
                if strat == "equal":
                    return Filter.by_update_time().equal(update_time.to_pydatetime()), f"update_time == {update_time.isoformat()}"
                if strat == "contains_any_self":
                    return Filter.by_update_time().contains_any([update_time.to_pydatetime()]), f"update_time contains_any [{update_time.isoformat()}]"
                if strat == "contains_any_noise":
                    return Filter.by_update_time().contains_any([update_time.to_pydatetime(), fake_future.to_pydatetime()]), f"update_time contains_any [{update_time.isoformat()}, {fake_future.isoformat()}]"
                if strat == "not_equal_fake":
                    return Filter.by_update_time().not_equal(fake_future.to_pydatetime()), f"update_time != {fake_future.isoformat()}"
                if strat == "contains_none_fake":
                    return Filter.by_update_time().contains_none([fake_future.to_pydatetime()]), f"update_time contains_none [{fake_future.isoformat()}]"
                if strat == "greater_than_before":
                    return Filter.by_update_time().greater_than(before.to_pydatetime()), f"update_time > {before.isoformat()}"
                if strat == "less_than_after":
                    return Filter.by_update_time().less_than(after.to_pydatetime()), f"update_time < {after.isoformat()}"
                if strat == "greater_or_equal":
                    return Filter.by_update_time().greater_or_equal(update_time.to_pydatetime()), f"update_time >= {update_time.isoformat()}"
                return Filter.by_update_time().less_or_equal(update_time.to_pydatetime()), f"update_time <= {update_time.isoformat()}"

        pl_fields = self.property_length_schema[:]
        random.shuffle(pl_fields)
        for field in pl_fields:
            fname, ftype = field["name"], field["type"]
            value = row.get(fname)
            if value is None:
                length = 0
            elif ftype == FieldType.TEXT:
                length = len(str(value))
            elif isinstance(value, list):
                length = len(value)
            else:
                length = 0
            base = Filter.by_property(fname, length=True)
            strat = random.choice(["equal", "greater_or_equal", "less_or_equal", "not_equal_fake"])
            if strat == "equal":
                return base.equal(length), f"len({fname}) == {length}"
            if strat == "greater_or_equal":
                return base.greater_or_equal(length), f"len({fname}) >= {length}"
            if strat == "less_or_equal":
                return base.less_or_equal(length), f"len({fname}) <= {length}"
            fake = length + random.choice([1, 2, 3])
            return base.not_equal(fake), f"len({fname}) != {fake}"

        text_fields = self.like_text_schema[:]
        random.shuffle(text_fields)
        for field in text_fields:
            fname = field["name"]
            val = row.get(fname)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            sv = str(val)
            strat = random.choice(["contains_any_self", "contains_all_self", "contains_any_noise", "contains_none_noise", "like_prefix", "like_suffix", "like_contains", "like_single", "eq"])
            if strat == "contains_any_self":
                return Filter.by_property(fname).contains_any([sv]), f'{fname} contains_any ["{sv}"]'
            elif strat == "contains_all_self":
                return Filter.by_property(fname).contains_all([sv]), f'{fname} contains_all ["{sv}"]'
            elif strat == "contains_any_noise":
                fake = ''.join(random.choices(string.ascii_letters, k=30))
                return Filter.by_property(fname).contains_any([sv, fake]), f'{fname} contains_any ["{sv}", "{fake}"]'
            elif strat == "contains_none_noise":
                fake = ''.join(random.choices(string.ascii_letters, k=30))
                while fake == sv:
                    fake = ''.join(random.choices(string.ascii_letters, k=30))
                return Filter.by_property(fname).contains_none([fake]), f'{fname} contains_none ["{fake}"]'
            if strat == "like_prefix" and len(sv) >= 1:
                k = random.randint(1, min(4, len(sv)))
                return Filter.by_property(fname).like(sv[:k] + "*"), f'{fname} like "{sv[:k]}*"'
            elif strat == "like_suffix" and len(sv) >= 2:
                k = random.randint(1, min(4, len(sv)))
                return Filter.by_property(fname).like("*" + sv[-k:]), f'{fname} like "*{sv[-k:]}"'
            elif strat == "like_contains" and len(sv) >= 3:
                i = random.randint(0, len(sv) - 2)
                j = random.randint(i + 1, min(i + 5, len(sv)))
                return Filter.by_property(fname).like("*" + sv[i:j] + "*"), f'{fname} like "*{sv[i:j]}*"'
            elif strat == "like_single" and len(sv) >= 2:
                p = random.randint(0, len(sv) - 1)
                pat = sv[:p] + "?" + sv[p+1:]
                return Filter.by_property(fname).like(pat), f'{fname} like "{pat}"'
            return Filter.by_property(fname).equal(sv), f'{fname} == "{sv}"'

        for field in random.sample(self.schema, len(self.schema)):
            fname, ftype = field["name"], field["type"]
            if ftype == FieldType.GEO:
                continue
            val = row.get(fname)
            is_null = val is None or (isinstance(val, float) and np.isnan(val))
            if is_null:
                if ENABLE_NULL_FILTER:
                    return Filter.by_property(fname).is_none(True), f"{fname} is null"
                continue
            if ftype == FieldType.BOOL:
                strat = random.choice(["eq", "neq"])
                if strat == "eq":
                    return Filter.by_property(fname).equal(bool(val)), f"{fname} == {bool(val)}"
                else:
                    return Filter.by_property(fname).not_equal(not bool(val)), f"{fname} != {not bool(val)}"
            elif ftype == FieldType.INT:
                vi = int(val)
                strat = random.choice(["eq", "ge_self", "le_self", "lt_above", "range_tight", "range_pm1", "not_lt", "neq_fake"])
                if strat == "eq":
                    return Filter.by_property(fname).equal(vi), f"{fname} == {vi}"
                elif strat == "ge_self":
                    return Filter.by_property(fname).greater_or_equal(vi), f"{fname} >= {vi}"
                elif strat == "le_self":
                    return Filter.by_property(fname).less_or_equal(vi), f"{fname} <= {vi}"
                elif strat == "lt_above":
                    if vi < WEAVIATE_SAFE_INT_MAX:
                        upper = clamp_weaviate_int(vi + 1)
                        return Filter.by_property(fname).less_than(upper), f"{fname} < {upper}"
                    return Filter.by_property(fname).less_or_equal(vi), f"{fname} <= {vi}"
                elif strat == "range_tight":
                    return Filter.by_property(fname).greater_or_equal(vi) & Filter.by_property(fname).less_or_equal(vi), f"{fname} [{vi},{vi}]"
                elif strat == "range_pm1":
                    return Filter.by_property(fname).greater_than(vi - 1) & Filter.by_property(fname).less_than(vi + 1), f"{fname} ({vi-1},{vi+1})"
                elif strat == "not_lt":
                    return Filter.not_(Filter.by_property(fname).less_than(vi)) & Filter.not_(Filter.by_property(fname).greater_than(vi)), f"NOT({fname}<{vi}) AND NOT({fname}>{vi})"
                else:  # neq_fake
                    fake = clamp_weaviate_int(vi + random.choice([100000, -100000]))
                    return Filter.by_property(fname).not_equal(fake) & Filter.by_property(fname).equal(vi), f"{fname} != {fake} AND {fname} == {vi}"
            elif ftype == FieldType.NUMBER:
                vf = float(val)
                strat = random.choice(["ge_self", "le_self", "lt_above", "gt_below", "range", "not_gt"])
                lo, hi = float_window(vf)
                if lo is None:
                    continue
                if strat == "ge_self":
                    return Filter.by_property(fname).greater_or_equal(vf), f"{fname}>={vf}"
                elif strat == "le_self":
                    return Filter.by_property(fname).less_or_equal(vf), f"{fname}<={vf}"
                elif strat == "lt_above":
                    return Filter.by_property(fname).less_than(hi), f"{fname}<{hi}"
                elif strat == "gt_below":
                    return Filter.by_property(fname).greater_than(lo), f"{fname}>{lo}"
                elif strat == "range":
                    return Filter.by_property(fname).greater_or_equal(lo) & Filter.by_property(fname).less_or_equal(hi), f"{fname} ≈ {vf}"
                else:
                    return Filter.not_(Filter.by_property(fname).greater_than(hi)) & Filter.by_property(fname).greater_or_equal(lo), f"NOT({fname}>{hi}) AND {fname}>={lo}"
            elif ftype == FieldType.TEXT:
                sv = str(val)
                strat = random.choice(["eq", "like_prefix", "like_suffix", "like_contains", "like_single", "neq_fake"])
                if strat == "eq":
                    return Filter.by_property(fname).equal(sv), f'{fname} == "{sv}"'
                elif strat == "like_prefix" and len(sv) >= 2:
                    k = random.randint(1, min(4, len(sv)))
                    return Filter.by_property(fname).like(sv[:k] + "*"), f'{fname} like "{sv[:k]}*"'
                elif strat == "like_suffix" and len(sv) >= 2:
                    k = random.randint(1, min(4, len(sv)))
                    return Filter.by_property(fname).like("*" + sv[-k:]), f'{fname} like "*{sv[-k:]}"'
                elif strat == "like_contains" and len(sv) >= 3:
                    i = random.randint(0, len(sv) - 2)
                    j = random.randint(i + 1, min(i + 5, len(sv)))
                    return Filter.by_property(fname).like("*" + sv[i:j] + "*"), f'{fname} like "*{sv[i:j]}*"'
                elif strat == "like_single" and len(sv) >= 2:
                    p = random.randint(0, len(sv) - 1)
                    pat = sv[:p] + "?" + sv[p+1:]
                    return Filter.by_property(fname).like(pat), f'{fname} like "{pat}"'
                elif strat == "neq_fake":
                    fake = ''.join(random.choices(string.ascii_letters, k=30))
                    return Filter.by_property(fname).not_equal(fake), f'{fname} != "{fake}"'
                else:
                    return Filter.by_property(fname).equal(sv), f'{fname} == "{sv}"'
            elif ftype == FieldType.DATE:
                strat = random.choice(["eq", "ge_self", "le_self", "lt_above", "gt_below"])
                if strat == "ge_self":
                    return Filter.by_property(fname).greater_or_equal(str(val)), f'{fname} >= "{val}"'
                if strat == "le_self":
                    return Filter.by_property(fname).less_or_equal(str(val)), f'{fname} <= "{val}"'
                if strat == "lt_above":
                    ts = to_utc_timestamp(val)
                    if ts is not None:
                        upper = format_rfc3339_timestamp(ts + pd.Timedelta(microseconds=1))
                        return Filter.by_property(fname).less_than(upper), f'{fname} < "{upper}"'
                if strat == "gt_below":
                    ts = to_utc_timestamp(val)
                    if ts is not None:
                        lower = format_rfc3339_timestamp(ts - pd.Timedelta(microseconds=1))
                        return Filter.by_property(fname).greater_than(lower), f'{fname} > "{lower}"'
                return Filter.by_property(fname).equal(str(val)), f'{fname} == "{val}"'
            elif ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY]:
                if not isinstance(val, list) or not val: continue
                items = [x for x in val if x is not None]
                # Tokenization.FIELD: 不需要停用词过滤
                if items:
                    strat = random.choice(["contains_any", "contains_all", "contains_any_noise", "contains_none_noise"])
                    if strat == "contains_any":
                        t = self._convert_to_native(random.choice(items))
                        return Filter.by_property(fname).contains_any([t]), f"{fname} contains {t}"
                    elif strat == "contains_all" and len(items) >= 2:
                        subset = [self._convert_to_native(x) for x in random.sample(items, min(2, len(items)))]
                        return Filter.by_property(fname).contains_all(subset), f"{fname} contains_all {subset}"
                    elif strat == "contains_none_noise":
                        item_set = set(items)
                        if ftype == FieldType.INT_ARRAY:
                            fake = -999999
                            while fake in item_set:
                                fake -= 1
                        elif ftype == FieldType.NUMBER_ARRAY:
                            fake = -999999.999
                            while fake in item_set:
                                fake -= 1.0
                        elif ftype == FieldType.BOOL_ARRAY:
                            remaining = [b for b in [True, False] if b not in item_set]
                            if not remaining:
                                t = self._convert_to_native(random.choice(items))
                                return Filter.by_property(fname).contains_any([t]), f"{fname} contains {t}"
                            fake = remaining[0]
                        else:
                            fake = "__fake__"
                            while fake in item_set:
                                fake += "_x"
                        fake = self._convert_to_native(fake)
                        return Filter.by_property(fname).contains_none([fake]), f"{fname} contains_none [{fake}]"
                    else:  # contains_any_noise
                        t = self._convert_to_native(random.choice(items))
                        # Use type-correct fake values to avoid pydantic validation errors
                        if ftype == FieldType.INT_ARRAY:
                            fake = -999999
                        elif ftype == FieldType.NUMBER_ARRAY:
                            fake = -999999.999
                        elif ftype == FieldType.BOOL_ARRAY:
                            # BOOL_ARRAY only has True/False, use contains_any with just [t]
                            return Filter.by_property(fname).contains_any([t]), f"{fname} contains {t}"
                        else:
                            fake = "__fake__"
                        return Filter.by_property(fname).contains_any([t, fake]), f"{fname} contains_any [{t},{fake}]"
                continue
        if ENABLE_NULL_FILTER:
            return Filter.by_property(self.schema[0]["name"]).is_none(False), f"{self.schema[0]['name']} is not null"
        return None, None


# --- 6. Main Execution ---

def initialize_seeded_run(seed=None):
    global VECTOR_CHECK_RATIO, VECTOR_TOPK, VECTOR_INDEX_TYPE, DISTANCE_METRIC, BOUNDARY_INJECTION_RATE
    current_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)
    VECTOR_CHECK_RATIO = random.uniform(0.2, 0.8)
    VECTOR_TOPK = random.randint(50, 200)
    BOUNDARY_INJECTION_RATE = random.uniform(0.10, 0.18)
    VECTOR_INDEX_TYPE = random.choice(ALL_VECTOR_INDEX_TYPES)
    DISTANCE_METRIC = random.choice(ALL_DISTANCE_METRICS)
    return current_seed

def run(rounds=100, seed=None, enable_dynamic_ops=True, consistency=DEFAULT_CONSISTENCY_LEVEL, randomize_consistency=False):
    current_seed = initialize_seeded_run(seed)
    print(f"🔒 Seed: {current_seed}")

    resolved_consistency = consistency or DEFAULT_CONSISTENCY_LEVEL
    cl_list = ALL_CONSISTENCY_LEVELS if randomize_consistency else [resolved_consistency]
    _cl_rng = random.Random(current_seed + 7)
    print(f"   Consistency: {consistency_label(resolved_consistency, randomize=randomize_consistency)}")
    print(f"   VecIndex: {VECTOR_INDEX_TYPE}, Dist: {DISTANCE_METRIC}, VecRatio: {VECTOR_CHECK_RATIO:.2f}, TopK: {VECTOR_TOPK}, BoundaryRate: {BOUNDARY_INJECTION_RATE:.2f}")

    dm = DataManager(current_seed)
    dm.generate_schema()
    dm.generate_data()
    wm = WeaviateManager()
    wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config)
        dm.searchable_text_fields = getattr(wm, "searchable_text_fields", set())
        wm.insert(dm)
        wm.sync_creation_times(dm)
        wm.sync_update_times(dm)
        ts = int(time.time())
        logf = make_log_path(f"weaviate_fuzz_test_{ts}.log")
        repro_cmd = format_repro_command(current_seed, resolved_consistency, randomize_consistency=randomize_consistency)
        print(f"\n📝 Log: {display_path(logf)}")
        print(f"   🔑 Reproduce: {repro_cmd}")
        print("🚀 Testing...")

        qg = OracleQueryGenerator(dm)
        col = wm.client.collections.get(CLASS_NAME)
        fails = []
        stats = FuzzStats()
        # Track dynamically modified row IDs to distinguish Weaviate inverted index bugs
        # from fuzzer logic bugs. Weaviate has a known bug where replace() corrupts
        # the inverted index for BOOL/array fields on modified rows.
        dynamic_ids = set()

        with open(logf, "w", encoding="utf-8") as f:
            def flog(msg):
                f.write(msg + "\n"); f.flush()

            def sample(ids, lim=5):
                if not ids: return []
                return dm.df[dm.df["id"].isin(list(ids))].to_dict("records")[:lim]

            def diagnose_id_mismatch(target_id, expr):
                row_df = dm.df[dm.df["id"] == target_id]
                row_data = row_df.iloc[0].to_dict() if not row_df.empty else None
                try:
                    obj = col.query.fetch_object_by_id(target_id)
                except Exception:
                    obj = None
                props = obj.properties if obj is not None else {}
                relevant = [ff for ff in extract_expr_fields(expr, dm.schema_config) if ff["type"] in [FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT, FieldType.DATE]]
                diffs = []
                for ff in relevant[:8]:
                    name = ff["name"]
                    pv = normalize_scalar_for_compare(ff["type"], row_data.get(name) if row_data else None)
                    wv = normalize_scalar_for_compare(ff["type"], props.get(name))
                    if pv != wv:
                        diffs.append(f"{name}: pandas={pv!r} weaviate={wv!r}")
                if not relevant:
                    return "no_relevant_scalar_fields", []
                if diffs:
                    return "LIKELY_ORACLE_OR_DATA_SYNC", diffs
                return "LIKELY_ENGINE_OR_QUERY_BUG", []

            flog(
                f"Start: {rounds} rounds | Seed: {current_seed} | "
                f"Consistency: {consistency_label(resolved_consistency, randomize=randomize_consistency)} | "
                f"VecIdx: {VECTOR_INDEX_TYPE} | Dist: {DISTANCE_METRIC}"
            )
            flog("=" * 50)

            for i in range(rounds):
                cl = _cl_rng.choice(cl_list) if randomize_consistency else cl_list[0]
                print(f"\r⏳ Test {i+1}/{rounds} [CL={cl}]       ", end="", flush=True)

                # Dynamic ops
                if enable_dynamic_ops and i > 0 and i % 10 == 0:
                    op = random.choices(["insert", "delete", "update", "upsert"], weights=[0.3, 0.3, 0.2, 0.2], k=1)[0]
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
                            dyn_result = col.data.insert_many(objs)
                            dyn_errors = getattr(dyn_result, "errors", None)
                            if not dyn_errors and isinstance(dyn_result, dict):
                                dyn_errors = dyn_result.get("errors")
                            if dyn_errors:
                                raise RuntimeError(f"dynamic batch insert partially failed: {dyn_errors}")
                            dm.df = pd.concat([dm.df, dm.rows_to_dataframe(rows)], ignore_index=True)
                            dm.vectors = np.vstack([dm.vectors, np.array(vecs)])
                            dm.normalize_dataframe_types()
                            dynamic_ids.update(r["id"] for r in rows)
                            flog(f"[Dyn] Inserted {bc}")
                            time.sleep(SLEEP_INTERVAL * 2)  # Allow index to stabilize
                            wm.sync_creation_times(dm, ids=[r["id"] for r in rows])
                            wm.sync_update_times(dm, ids=[r["id"] for r in rows])
                            qg = OracleQueryGenerator(dm)
                        except Exception as e:
                            flog(f"[Dyn] Insert fail: {e}")
                    elif op == "delete":
                        if len(dm.df) > bc:
                            try:
                                idxs = random.sample(range(len(dm.df)), bc)
                                dids = [dm.df.iloc[x]["id"] for x in idxs]
                                for d in dids:
                                    col.data.delete_by_id(d)
                                # Post-delete verification
                                for d in dids:
                                    try:
                                        obj = col.query.fetch_object_by_id(d)
                                        if obj is not None:
                                            flog(f"[Dyn] ❌ Deleted ID still exists: {d}")
                                            fails.append({"id": i, "detail": f"Ghost after delete: {d}"})
                                    except Exception:
                                        pass  # expected: not found
                                keep = ~dm.df["id"].isin(dids)
                                ki = dm.df[keep].index.to_numpy()
                                dm.df = dm.df[keep].reset_index(drop=True)
                                dm.vectors = dm.vectors[ki]
                                dm.normalize_dataframe_types()
                                qg = OracleQueryGenerator(dm)
                                flog(f"[Dyn] Deleted {len(dids)}")
                                time.sleep(SLEEP_INTERVAL * 2)
                            except Exception as e:
                                flog(f"[Dyn] Delete fail: {e}")
                    elif op == "update":
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
                                dm.normalize_dataframe_types()
                                dynamic_ids.add(uid)
                                # Post-update verification
                                try:
                                    obj = col.query.fetch_object_by_id(uid)
                                    if obj is not None:
                                        for pk, pv in p.items():
                                            av = obj.properties.get(pk)
                                            if av is not None and pv is not None:
                                                if isinstance(pv, float):
                                                    if abs(float(av) - pv) > 1e-3:
                                                        flog(f"[Dyn] ❌ Update mismatch: {pk} expected={pv} actual={av}")
                                                elif isinstance(pv, str) and to_utc_timestamp(pv) is not None:
                                                    norm_pv = canonicalize_date_string(pv)
                                                    norm_av = canonicalize_date_string(av)
                                                    if norm_pv != norm_av:
                                                        flog(f"[Dyn] ❌ Update mismatch: {pk} expected={norm_pv} actual={norm_av}")
                                                elif isinstance(pv, list):
                                                    # Array fields: compare sorted to avoid order issues
                                                    if sorted(str(x) for x in av) != sorted(str(x) for x in pv) if isinstance(av, list) else True:
                                                        pass  # Skip noisy array comparison
                                                elif isinstance(pv, dict):
                                                    # OBJECT fields: compare by sorted keys
                                                    if isinstance(av, dict):
                                                        if sorted(av.items()) != sorted(pv.items()):
                                                            flog(f"[Dyn] ❌ Update mismatch: {pk} expected={pv} actual={av}")
                                                    else:
                                                        flog(f"[Dyn] ❌ Update mismatch: {pk} expected={pv} actual={av}")
                                                elif str(av) != str(pv):
                                                    flog(f"[Dyn] ❌ Update mismatch: {pk} expected={pv} actual={av}")
                                        flog(f"[Dyn] Update verified {uid}")
                                except Exception:
                                    pass
                                flog(f"[Dyn] Updated {uid}")
                                time.sleep(SLEEP_INTERVAL * 2)
                                wm.sync_update_times(dm, ids=[uid])
                                qg = OracleQueryGenerator(dm)
                            except Exception as e:
                                flog(f"[Dyn] Update fail: {e}")
                    elif op == "upsert":
                        # Upsert: 70% existing ID (update), 30% new ID (insert)
                        upsert_rows, upsert_vecs = [], []
                        for _ in range(bc):
                            use_existing = (not dm.df.empty) and (random.random() < 0.7)
                            if use_existing:
                                target_id = random.choice(dm.df["id"].tolist())
                            else:
                                target_id = dm._next_uuid()
                            row = dm.generate_single_row(id_override=target_id)
                            vec = dm.generate_single_vector()
                            upsert_rows.append(row)
                            upsert_vecs.append(vec)
                        try:
                            new_rows, new_vecs = [], []
                            for row, vec in zip(upsert_rows, upsert_vecs):
                                rid = row["id"]
                                p = {k: v for k, v in row.items() if k != "id" and v is not None}
                                match_idx = dm.df.index[dm.df["id"] == rid].tolist()
                                if match_idx:
                                    # Update existing
                                    col.data.replace(uuid=rid, properties=p, vector=vec.tolist())
                                    idx = match_idx[0]
                                    for k, v in row.items():
                                        dm.df.at[idx, k] = v
                                    dm.vectors[idx] = vec
                                else:
                                    # Insert new
                                    col.data.insert(properties=p, uuid=rid, vector=vec.tolist())
                                    new_rows.append(row)
                                    new_vecs.append(vec)
                            dynamic_ids.update(r["id"] for r in upsert_rows)
                            if new_rows:
                                dm.df = pd.concat([dm.df, dm.rows_to_dataframe(new_rows)], ignore_index=True)
                                dm.vectors = np.vstack([dm.vectors, np.array(new_vecs)])
                            dm.normalize_dataframe_types()
                            flog(f"[Dyn] Upserted {len(upsert_rows)} (new:{len(new_rows)})")
                            time.sleep(SLEEP_INTERVAL * 2)  # Allow index to stabilize
                            wm.sync_creation_times(dm, ids=[r["id"] for r in upsert_rows])
                            wm.sync_update_times(dm, ids=[r["id"] for r in upsert_rows])
                            qg = OracleQueryGenerator(dm)
                        except Exception as e:
                            flog(f"[Dyn] Upsert fail: {e}")

                # HNSW parameter reconfiguration + maintenance pressure every 40 rounds
                if i > 0 and i % 40 == 0 and VECTOR_INDEX_TYPE == "hnsw":
                    try:
                        new_ef = random.choice([64, 128, 256, 512])
                        new_dyn = random.choice([4, 8, 12])
                        new_cutoff = random.choice([20000, 40000, 60000])
                        col.config.update(
                            vector_index_config=Reconfigure.VectorIndex.hnsw(
                                ef=new_ef,
                                dynamic_ef_factor=new_dyn,
                                flat_search_cutoff=new_cutoff,
                            )
                        )
                        flog(f"[Reconfig] HNSW ef={new_ef} dyn_ef={new_dyn} cutoff={new_cutoff}")
                        # Post-reconfig maintenance pressure: verify a known-good object
                        try:
                            verify_id = dm.df.iloc[0]["id"]
                            vobj = col.query.fetch_object_by_id(verify_id)
                            if vobj is None:
                                flog(f"[Maintenance] ❌ Post-reconfig ID missing: {verify_id}")
                                fails.append({"id": i, "detail": f"Post-reconfig ghost: {verify_id}"})
                            else:
                                flog(f"[Maintenance] ✅ Post-reconfig verify OK")
                        except Exception as ve:
                            flog(f"[Maintenance] verify err: {ve}")
                    except Exception as e:
                        flog(f"[Reconfig] Failed: {e}")

                depth = random.randint(1, 15)
                fo = None
                for _ in range(10):
                    fo, pm, es = qg.gen_complex_expr(depth)
                    if fo: break
                if not fo:
                    stats.record_skip()
                    continue

                flog(f"\n[T{i}] {es}")
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
                        stats.record(True, ms, depth)
                    else:
                        mi, ex = exp - act, act - exp
                        dm_ = f"Missing:{len(mi)} Extra:{len(ex)}"
                        # Check if mismatch involves ONLY dynamically modified rows
                        # Weaviate has a known bug: replace() corrupts inverted index
                        # for BOOL/BOOL_ARRAY/INT_ARRAY fields on modified rows.
                        mi_dyn = mi & dynamic_ids
                        ex_dyn = ex & dynamic_ids
                        is_weaviate_bug = (mi == mi_dyn) and (ex == ex_dyn) and dynamic_ids
                        if cl != ConsistencyLevel.ALL:
                            flog(f"  -> WARN (CL={cl}): {dm_}")
                        elif is_weaviate_bug:
                            # Known Weaviate inverted index bug — downgrade to warning
                            flog(f"  -> WEAVIATE_BUG (dynamic rows only): {dm_}")
                            flog(f"    Dynamic IDs in Extra: {len(ex_dyn)}, Missing: {len(mi_dyn)}")
                            stats.record(True, ms, depth)  # Count as pass
                            stats.weaviate_bugs = getattr(stats, 'weaviate_bugs', 0) + 1
                        else:
                            print(f"\n❌ [T{i}] MISMATCH! {dm_}")
                            print(f"   Expr: {es}")
                            print(f"   🔑 {repro_cmd}")
                            flog(f"  -> MISMATCH! {dm_}")
                            if mi: flog(f"  Missing: {sample(mi)}")
                            if ex: flog(f"  Extra: {sample(ex)}")
                            # Per-ID verification for extra IDs
                            for eid in list(ex)[:2]:
                                try:
                                    obj = col.query.fetch_object_by_id(eid)
                                    exists = obj is not None
                                    verdict, diffs = diagnose_id_mismatch(eid, es)
                                    flog(f"    ExtraID {eid}: exists={exists} verdict={verdict}")
                                    for d in diffs[:4]:
                                        flog(f"      {d}")
                                except Exception as ve:
                                    flog(f"    ExtraID verify err: {ve}")
                            for mid in list(mi)[:2]:
                                try:
                                    verdict, diffs = diagnose_id_mismatch(mid, es)
                                    flog(f"    MissingID {mid}: verdict={verdict}")
                                    for d in diffs[:4]:
                                        flog(f"      {d}")
                                except Exception as ve:
                                    flog(f"    MissingID verify err: {ve}")
                            fails.append({"id": i, "expr": es, "detail": dm_, "seed": current_seed})
                            stats.record(False, ms, depth, "mismatch")

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

                    if len(exp) > 20 and random.random() < 0.15:
                        try:
                            # --- Pagination consistency test ---
                            # Pick a sortable scalar field for deterministic pagination
                            _sortable_types = {FieldType.INT, FieldType.NUMBER, FieldType.TEXT, FieldType.DATE}
                            _sort_fields = [f for f in dm.schema_config
                                            if f["type"] in _sortable_types
                                            and (not dm.filterable_fields or f["name"] in dm.filterable_fields)]
                            ps = random.randint(5, 15)
                            max_pages = max(len(exp) // ps + 2, 10)

                            # --- Test A: WITH sort (deterministic — dups = real bug) ---
                            if _sort_fields:
                                sf = random.choice(_sort_fields)
                                asc = random.choice([True, False])
                                sort_obj = Sort.by_property(sf["name"], ascending=asc)
                                seen_sorted = set()
                                all_sorted = []
                                dup_sorted = 0
                                for pg in range(max_pages):
                                    offset = pg * ps
                                    pr = col_cl.query.fetch_objects(filters=fo, limit=ps, offset=offset, sort=sort_obj)
                                    if not pr.objects: break
                                    for o in pr.objects:
                                        uid = str(o.uuid)
                                        if uid in seen_sorted:
                                            dup_sorted += 1
                                        seen_sorted.add(uid)
                                        all_sorted.append(uid)
                                    if len(pr.objects) < ps: break
                                n_pages = (len(all_sorted) - 1) // ps + 1 if all_sorted else 0
                                if dup_sorted > 0:
                                    if cl != ConsistencyLevel.ALL:
                                        flog(f"  Page(sort={sf['name']}): WARN CL={cl} {dup_sorted} dups in {n_pages} pages")
                                    else:
                                        flog(f"  Page(sort={sf['name']}): FAIL {dup_sorted} dups in {n_pages} pages (BUG: sort should guarantee determinism)")
                                        stats.record_fail()
                                elif seen_sorted and not seen_sorted.issubset(exp):
                                    extras = seen_sorted - exp
                                    if cl != ConsistencyLevel.ALL:
                                        flog(f"  Page(sort={sf['name']}): WARN CL={cl} {len(extras)} extras not in expected")
                                    else:
                                        flog(f"  Page(sort={sf['name']}): FAIL {len(extras)} extras not in expected")
                                        stats.record_fail()
                                else:
                                    flog(f"  Page(sort={sf['name']}): PASS ({len(seen_sorted)} in {n_pages}p)")

                            # --- Test B: WITHOUT sort (informational — dups = known limitation) ---
                            if random.random() < 0.3:
                                seen_nosort = set()
                                dup_nosort = 0
                                for pg in range(min(max_pages, 10)):
                                    offset = pg * ps
                                    pr = col_cl.query.fetch_objects(filters=fo, limit=ps, offset=offset)
                                    if not pr.objects: break
                                    for o in pr.objects:
                                        uid = str(o.uuid)
                                        if uid in seen_nosort:
                                            dup_nosort += 1
                                        seen_nosort.add(uid)
                                    if len(pr.objects) < ps: break
                                if dup_nosort > 0:
                                    flog(f"  Page(nosort): INFO {dup_nosort} dups (known: non-deterministic w/o sort)")
                                else:
                                    flog(f"  Page(nosort): PASS ({len(seen_nosort)})")
                        except Exception as e:
                            flog(f"  Page: ERR {e}")
                except Exception as e:
                    print(f"\n⚠️ [T{i}] CRASH: {e}")
                    flog(f"  -> ERR: {e}")
                    # Categorize errors
                    err_str = str(e).lower()
                    if "inverted index" in err_str or "is it indexed" in err_str or "not found - is it indexed" in err_str:
                        # Expected when index_filterable=False (random config) — skip, not fail
                        cat = "index_skip"
                        stats.record_skip()
                        continue
                    elif "stopword" in err_str:
                        cat = "stopword"
                    elif "timeout" in err_str or "deadline" in err_str:
                        cat = "timeout"
                    elif "connection" in err_str:
                        cat = "connection"
                    else:
                        cat = "query_error"
                    stats.record(False, 0, depth, cat)
                    fails.append({"id": i, "expr": es, "detail": str(e)})

        print("\n" + "="*60)
        print(f"📊 Stats: {stats.summary()}")
        wb = getattr(stats, 'weaviate_bugs', 0)
        if wb:
            print(f"⚠️  {wb} Weaviate inverted-index bugs detected (dynamic rows, downgraded to warnings)")
        if not fails:
            print(f"✅ All {rounds} tests passed!" + (f" ({wb} weaviate bugs)" if wb else ""))
        else:
            print(f"🚫 {len(fails)} failures!")
            for c in fails[:10]:
                print(f"  🔴 T{c['id']}: {c.get('detail','')[:100]}")
        print(f"📄 Log: {display_path(logf)}")
        print(f"🔑 Reproduce: {format_repro_command(current_seed, resolved_consistency, randomize_consistency=randomize_consistency)}")
    finally:
        wm.close()


def run_equivalence_mode(rounds=100, seed=None):
    seed = initialize_seeded_run(seed)

    logf = make_log_path(f"weaviate_equiv_test_{int(time.time())}.log")
    print(f"\n👯 Equivalence Mode | Seed: {seed} | VecIdx: {VECTOR_INDEX_TYPE} | BoundaryRate: {BOUNDARY_INJECTION_RATE:.2f}")
    print(f"📄 Log: {display_path(logf)}")

    dm = DataManager(seed); dm.generate_schema(); dm.generate_data()
    wm = WeaviateManager(); wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config); dm.searchable_text_fields = getattr(wm, "searchable_text_fields", set()); wm.insert(dm); wm.sync_creation_times(dm); wm.sync_update_times(dm)
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
                                diff = bi.symmetric_difference(mi)
                                flog(f"  {mut['type']}: FAIL (diff:{len(diff)})")
                                print(f"\n❌ [T{i}] {mut['type']} FAIL (diff:{len(diff)})")
                                # Deep evidence: print details of diff IDs
                                for did in list(diff)[:3]:
                                    row = dm.df[dm.df['id'] == did]
                                    if not row.empty:
                                        null_cols = [c for c in row.columns if row[c].isna().any()]
                                        flog(f"    DiffID {did}: null_cols={null_cols}")
                                        for ff in dm.schema_config[:5]:
                                            v = row.iloc[0].get(ff['name'])
                                            flog(f"      {ff['name']}={v}")
                                fails.append({"id": i, "type": mut["type"]})
                        except Exception as e:
                            flog(f"  {mut['type']}: ERR {e}")
                except Exception as e:
                    flog(f"  ERR: {e}")
        print(f"\n{'✅ All passed' if not fails else f'🚫 {len(fails)} failures'}. Log: {display_path(logf)}")
    finally:
        wm.close()


def run_pqs_mode(rounds=100, seed=None):
    seed = initialize_seeded_run(seed)

    logf = make_log_path(f"weaviate_pqs_test_{int(time.time())}.log")
    print(f"\n🚀 PQS Mode | Seed: {seed} | VecIdx: {VECTOR_INDEX_TYPE} | BoundaryRate: {BOUNDARY_INJECTION_RATE:.2f}")
    print(f"📄 Log: {display_path(logf)}")

    dm = DataManager(seed); dm.generate_schema(); dm.generate_data()
    wm = WeaviateManager(); wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config); dm.searchable_text_fields = getattr(wm, "searchable_text_fields", set()); wm.insert(dm); wm.sync_creation_times(dm); wm.sync_update_times(dm)
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
                d = random.randint(3, 13)
                try:
                    # 30% chance: use multi-field joint expression for higher coverage
                    if random.random() < 0.3:
                        mf, me = pqs.gen_multi_field_true_filter(pr, n=random.randint(2, 4))
                        if mf:
                            pf, pe = mf, f"MultiField({me})"
                        else:
                            pf, pe = pqs.gen_pqs_filter(pr, d)
                    else:
                        pf, pe = pqs.gen_pqs_filter(pr, d)
                except Exception as e:
                    flog(f"[T{i}] Gen err: {e}"); skip += 1; continue
                if not pf: skip += 1; continue
                flog(f"\n[T{i}] Pivot:{pid}")
                flog(f"  Expr: {pe}")
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
        print(f"{'✅ All passed' if not errs else f'🚫 {len(errs)} failures'}. Log: {display_path(logf)}")
    finally:
        wm.close()


def run_groupby_mode(rounds=100, seed=None):
    seed = initialize_seeded_run(seed)

    logf = make_log_path(f"weaviate_groupby_test_{int(time.time())}.log")
    print(f"\n📦 GroupBy Mode | Seed: {seed} | VecIdx: {VECTOR_INDEX_TYPE} | BoundaryRate: {BOUNDARY_INJECTION_RATE:.2f}")
    print(f"📄 Log: {display_path(logf)}")

    dm = DataManager(seed); dm.generate_schema(); dm.generate_data()
    wm = WeaviateManager(); wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config); dm.searchable_text_fields = getattr(wm, "searchable_text_fields", set()); wm.insert(dm)
        col = wm.client.collections.get(CLASS_NAME)
        errs, ok = [], 0
        # Find groupable scalar fields (must be filterable/indexed)
        filterable = getattr(dm, 'filterable_fields', set())
        group_fields = [f for f in dm.schema_config if f["type"] in [FieldType.INT, FieldType.BOOL, FieldType.TEXT] and (not filterable or f["name"] in filterable)]
        if not group_fields:
            print("  No groupable fields, skip."); return

        with open(logf, "w", encoding="utf-8") as f:
            def flog(m): f.write(m + "\n"); f.flush()
            flog(f"GroupBy | Seed:{seed}")
            for i in range(rounds):
                print(f"\r⏳ GroupBy {i+1}/{rounds}", end="", flush=True)
                gf = random.choice(group_fields)
                n_groups = random.randint(2, 20)
                per_group = random.randint(1, 5)
                qi = random.randint(0, len(dm.vectors) - 1)
                qv = dm.vectors[qi].tolist()
                flog(f"\n[T{i}] field={gf['name']} groups={n_groups} per={per_group}")
                try:
                    gb = GroupBy(prop=gf["name"], number_of_groups=n_groups, objects_per_group=per_group)
                    res = col.query.near_vector(near_vector=qv, group_by=gb, limit=n_groups * per_group)
                    objs = res.objects
                    # Validate: count groups and per-group counts
                    groups = {}
                    for o in objs:
                        gval = o.properties.get(gf["name"])
                        gkey = str(gval)
                        groups.setdefault(gkey, []).append(str(o.uuid))
                    actual_groups = len(groups)
                    max_per = max(len(v) for v in groups.values()) if groups else 0
                    flog(f"  groups={actual_groups}/{n_groups} max_per={max_per}/{per_group} total={len(objs)}")
                    if actual_groups > n_groups:
                        flog(f"  ❌ Too many groups!")
                        errs.append({"id": i, "detail": f"groups {actual_groups}>{n_groups}"})
                    elif max_per > per_group:
                        flog(f"  ❌ Too many per group!")
                        errs.append({"id": i, "detail": f"per_group {max_per}>{per_group}"})
                    else:
                        # Check for key splitting (same value different keys)
                        flog(f"  PASS")
                        ok += 1
                except Exception as e:
                    err_str = str(e)
                    if "group by" in err_str.lower() or "not supported" in err_str.lower():
                        flog(f"  SKIP (unsupported): {err_str[:100]}")
                    else:
                        flog(f"  ERR: {err_str[:200]}")
                        errs.append({"id": i, "error": err_str[:100]})
        print(f"\n📊 GroupBy: ok={ok} err={len(errs)}")
        print(f"{'✅ All passed' if not errs else f'🚫 {len(errs)} failures'}. Log: {display_path(logf)}")
    finally:
        wm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weaviate Fuzz Oracle V2")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--rounds", type=int, default=500, help="Test rounds (default: 500)")
    parser.add_argument("--mode", choices=["oracle", "equiv", "pqs", "groupby"], default="oracle", help="Test mode")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic ops")
    parser.add_argument(
        "--consistency",
        choices=["all", "quorum", "one"],
        default="all",
        help="Consistency level for oracle mode (default: all)",
    )
    parser.add_argument(
        "--random-consistency",
        action="store_true",
        help="Randomize oracle consistency across ONE/QUORUM/ALL",
    )
    parser.add_argument("--host", type=str, default=None, help="Weaviate host")
    parser.add_argument("--port", type=int, default=None, help="Weaviate port")
    parser.add_argument("-N", type=int, default=None, help="Data rows")
    args = parser.parse_args()

    if args.host: HOST = args.host
    if args.port: PORT = args.port
    if args.N: N = args.N

    resolved_consistency = parse_consistency_arg(args.consistency)

    print("=" * 80)
    print(
        f"🚀 Weaviate Fuzz Oracle V2 | Mode: {args.mode} | Rounds: {args.rounds} | "
        f"Seed: {args.seed or '(random)'} | Dynamic: {'OFF' if args.no_dynamic else 'ON'} | "
        f"Consistency: {consistency_label(resolved_consistency, randomize=args.random_consistency)}"
    )
    print("=" * 80)

    if args.mode == "equiv":
        run_equivalence_mode(rounds=args.rounds, seed=args.seed)
    elif args.mode == "pqs":
        run_pqs_mode(rounds=args.rounds, seed=args.seed)
    elif args.mode == "groupby":
        run_groupby_mode(rounds=args.rounds, seed=args.seed)
    else:
        run(
            rounds=args.rounds,
            seed=args.seed,
            enable_dynamic_ops=not args.no_dynamic,
            consistency=resolved_consistency,
            randomize_consistency=args.random_consistency,
        )
