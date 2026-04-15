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
import os
import random
import string
import math
import copy
import re
import uuid
import numpy as np
import pandas as pd
import json
import sys
import argparse
import shlex
from datetime import datetime, timedelta, timezone
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny, MatchExcept,
    Range, DatetimeRange, PayloadField,
    IsNullCondition, IsEmptyCondition,
    HasIdCondition, PointIdsList,
    GeoPoint, GeoBoundingBox, GeoRadius,
    PayloadSchemaType,
    KeywordIndexParams, KeywordIndexType,
    IntegerIndexParams, IntegerIndexType,
    FloatIndexParams, FloatIndexType,
    GeoIndexParams, GeoIndexType,
    BoolIndexParams, BoolIndexType,
    DatetimeIndexParams, DatetimeIndexType,
    UuidIndexParams, UuidIndexType,
    SearchParams,
    NestedCondition, Nested,
    MinShould,
)

# --- Configuration (User Specified) ---
HOST = "127.0.0.1"
PORT = 6333                # REST port
GRPC_PORT = 6334           # gRPC port
PREFER_GRPC = False        # 默认使用 REST，按需可切到 gRPC
COLLECTION_NAME = "fuzz_oracle_v1"
FORCED_COLLECTION_NAME = None
N = 5000                   # 数据量
DIM = 128                  # 向量维度
BATCH_SIZE = 500           # 批次大小
SLEEP_INTERVAL = 0.01      # 每次插入后暂停
READ_CONSISTENCY = "all"   # 读一致性默认最高；可通过 CLI 覆盖或设为 random
WRITE_ORDERING = "strong"  # 写 ordering 默认最高；可通过 CLI 覆盖或设为 random
ALL_READ_CONSISTENCY_LEVELS = ["all", "majority", "quorum", 1]
ALL_WRITE_ORDERING_LEVELS = ["weak", "medium", "strong"]

# 全局度量类型列表（延迟初始化，在 run() 内种子设置后随机选取）
ALL_DISTANCE_TYPES = [Distance.EUCLID, Distance.COSINE, Distance.DOT, Distance.MANHATTAN]
DISTANCE_NAME_MAP = {
    "euclid": Distance.EUCLID,
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "manhattan": Distance.MANHATTAN,
}
DISTANCE_TYPE = None  # 延迟初始化
FORCED_DISTANCE_TYPE = None

# 随机化配置（延迟初始化，在 run() 内种子设置后赋值，确保可复现）
INDEX_TYPE = "hnsw"  # Qdrant 主要使用 HNSW
VECTOR_CHECK_RATIO = None  # 延迟初始化
VECTOR_TOPK = None          # 延迟初始化

def _init_vector_check_config():
    """在种子设置后调用，确保 VECTOR_CHECK_RATIO/VECTOR_TOPK 可复现"""
    global VECTOR_CHECK_RATIO, VECTOR_TOPK
    VECTOR_CHECK_RATIO = random.uniform(0.2, 0.8)
    VECTOR_TOPK = random.randint(50, 200)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACT_ROOT = os.environ.get(
    "QDRANT_EXPERIMENT_ROOT",
    os.path.join(os.path.expanduser("~"), "qdrant_artifacts"),
)
DEFAULT_LOG_DIR = os.path.join(DEFAULT_ARTIFACT_ROOT, "qdrant_log")
LOG_DIR = DEFAULT_LOG_DIR
RUN_ID = None
DISTANCE_ARG = "random"
SCROLL_MODE = "mixed"


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


def make_log_path(filename: str) -> str:
    return os.path.join(ensure_log_dir(), filename)


def display_path(path: str) -> str:
    try:
        return os.path.relpath(path, start=os.getcwd())
    except Exception:
        return path


def slugify(value: object, max_len: int = 96) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def sample_ids(values, limit: int = 10) -> list:
    try:
        return sorted(values)[:limit]
    except TypeError:
        return sorted(values, key=lambda item: str(item))[:limit]


def choose_scroll_strategy(total_rows: int) -> dict[str, object]:
    total_rows = max(1, int(total_rows))
    if SCROLL_MODE == "bulk":
        return {"mode": "bulk", "limit": total_rows + 1000}
    if SCROLL_MODE == "paged":
        divisor = random.choice([6, 8, 12, 16, 24, 32])
        return {"mode": "paged", "page_size": max(8, min(256, total_rows // divisor))}
    if random.random() < 0.35:
        divisor = random.choice([6, 8, 12, 16, 24, 32])
        return {"mode": "paged", "page_size": max(8, min(256, total_rows // divisor))}
    return {"mode": "bulk", "limit": total_rows + 1000}


def format_scroll_plan(plan: dict[str, object]) -> str:
    mode = str(plan.get("mode", "bulk"))
    if mode == "paged":
        parts = [f"mode=paged page_size={plan.get('page_size')}"]
        if "pages" in plan:
            parts.append(f"pages={plan.get('pages')}")
        if "duplicates" in plan:
            parts.append(f"duplicates={plan.get('duplicates')}")
        if "last_offset" in plan:
            parts.append(f"last_offset={plan.get('last_offset')}")
        return " ".join(parts)
    return f"mode=bulk limit={plan.get('limit')}"


def execute_scroll_query(
    qm,
    collection_name: str,
    scroll_filter,
    total_rows: int,
    *,
    with_payload=False,
    with_vectors=False,
):
    plan = choose_scroll_strategy(total_rows)
    if plan["mode"] == "bulk":
        points, next_offset = qm.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=int(plan["limit"]),
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        plan["pages"] = 1
        plan["duplicates"] = 0
        plan["last_offset"] = next_offset
        return list(points), plan

    page_size = int(plan["page_size"])
    max_pages = max(8, min(256, (max(1, total_rows) // max(1, page_size)) + 32))
    offset = None
    pages = 0
    all_points = []
    seen_ids = set()
    duplicate_ids = set()

    while True:
        points, next_offset = qm.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=page_size,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        pages += 1
        page_ids = [point.id for point in points]
        for pid in page_ids:
            if pid in seen_ids:
                duplicate_ids.add(pid)
            seen_ids.add(pid)
        all_points.extend(points)
        if next_offset is None:
            plan["pages"] = pages
            plan["duplicates"] = len(duplicate_ids)
            plan["last_offset"] = None
            return all_points, plan
        if pages >= max_pages:
            raise RuntimeError(
                f"Paged scroll exceeded safety bound: pages={pages}, page_size={page_size}, last_offset={next_offset!r}"
            )
        if next_offset == offset:
            raise RuntimeError(f"Paged scroll did not advance: offset={offset!r}")
        offset = next_offset


def should_run_count_cross_check(scroll_plan: dict[str, object]) -> bool:
    if str(scroll_plan.get("mode")) == "paged":
        return True
    return random.random() < 0.2


def make_run_label(mode: str, seed) -> str:
    if RUN_ID:
        return slugify(RUN_ID)
    seed_label = f"seed{seed}" if seed is not None else "seed-random"
    return slugify(f"{mode}-{seed_label}")


def make_mode_log_path(mode: str, seed) -> str:
    return make_log_path(f"qdrant_{slugify(mode, max_len=32)}_{make_run_label(mode, seed)}.log")


def make_collection_name(mode: str, seed) -> str:
    if FORCED_COLLECTION_NAME:
        return str(FORCED_COLLECTION_NAME)
    mode_part = slugify(mode, max_len=24)
    label_part = slugify(make_run_label(mode, seed), max_len=56)
    return f"qdrant_{mode_part}_{label_part}"


def set_collection_name(mode: str, seed) -> str:
    global COLLECTION_NAME
    COLLECTION_NAME = make_collection_name(mode, seed)
    return COLLECTION_NAME


def build_reproduce_command(mode: str, seed, rounds: int, enable_dynamic_ops: bool) -> str:
    mode_flag = {
        "oracle": "--oracle",
        "equiv": "--equiv",
        "pqs": "--pqs",
        "group": "--group",
    }.get(mode, "--oracle")
    cmd = [
        "python",
        "qdrant_fuzz_oracle.py",
        mode_flag,
        "--seed",
        str(seed),
    ]
    if mode == "pqs":
        cmd += ["--pqs-rounds", str(rounds)]
    else:
        cmd += ["--rounds", str(rounds)]
    cmd += [
        "-N",
        str(N),
        "--dim",
        str(DIM),
        "--batch-size",
        str(BATCH_SIZE),
        "--sleep-interval",
        str(SLEEP_INTERVAL),
        "--scroll-mode",
        SCROLL_MODE,
        "--distance",
        DISTANCE_ARG,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--grpc-port",
        str(GRPC_PORT),
        "--collection-name",
        COLLECTION_NAME,
        "--read-consistency",
        str(READ_CONSISTENCY),
        "--write-ordering",
        str(WRITE_ORDERING),
        "--log-dir",
        LOG_DIR,
    ]
    if PREFER_GRPC:
        cmd.append("--prefer-grpc")
    if enable_dynamic_ops:
        cmd.append("--dynamic")
    if PAYLOAD_MUTATIONS_ENABLED:
        cmd.append("--payload-mutations")
    if CHAOS_RATE > 0:
        cmd += ["--chaos-rate", str(CHAOS_RATE)]
    if SCHEMA_EVOLUTION_EXPLICIT_NULL_SYNC:
        cmd.append("--evo-null-sync")
    if INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES:
        cmd.append("--include-known-int64-boundaries")
    if RUN_ID:
        cmd += ["--run-id", RUN_ID]
    return shlex.join(cmd)


def write_run_manifest(
    log_filename: str,
    mode: str,
    seed,
    rounds: int,
    reproduce_command: str,
    enable_dynamic_ops: bool,
    schema_config=None,
) -> str:
    manifest_path = os.path.splitext(log_filename)[0] + ".manifest.json"
    payload = {
        "mode": mode,
        "seed": seed,
        "rounds": rounds,
        "run_id": RUN_ID,
        "log_path": log_filename,
        "reproduce_command": reproduce_command,
        "host": HOST,
        "port": PORT,
        "grpc_port": GRPC_PORT,
        "prefer_grpc": PREFER_GRPC,
        "collection_name": COLLECTION_NAME,
        "rows": N,
        "dim": DIM,
        "batch_size": BATCH_SIZE,
        "sleep_interval": SLEEP_INTERVAL,
        "scroll_mode": SCROLL_MODE,
        "distance_arg": DISTANCE_ARG,
        "read_consistency": READ_CONSISTENCY,
        "write_ordering": WRITE_ORDERING,
        "dynamic": bool(enable_dynamic_ops),
        "payload_mutations": bool(PAYLOAD_MUTATIONS_ENABLED),
        "chaos_rate": CHAOS_RATE,
        "schema_config": copy.deepcopy(schema_config) if schema_config is not None else None,
    }
    with open(manifest_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)
        fout.write("\n")
    return manifest_path


def is_qdrant_format_error(exc):
    """识别 Qdrant 400 JSON filter 格式错误（通常由无效条件或越界值触发）。"""
    msg = str(exc)
    return ("Unexpected Response: 400" in msg) and ("Format error in JSON body" in msg)

# 混淆开关（默认关闭）
CHAOS_RATE = 0.0

# Schema Evolution 最大演进字段数
MAX_EVOLVED_FIELDS = 5

# Schema Evolution: 是否为未回填行显式回写 NULL
# 默认关闭：演进字段查询统一走 is_empty 语义，避免对全量点逐条 upsert 导致性能/一致性漂移。
SCHEMA_EVOLUTION_EXPLICIT_NULL_SYNC = False

# Dynamic payload mutation 子路径（默认关闭；通过 --payload-mutations 显式开启）。
PAYLOAD_MUTATIONS_ENABLED = False

# 已知 Qdrant v1.17.0 在 int64 极值附近的 range 比较存在异常。
# 通用 fuzz 默认不注入这些值，避免把已知 server bug 误判成新的逻辑错误。
INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES = False
KNOWN_UNSTABLE_INT64_VALUES = {
    -(2 ** 63),
    -(2 ** 63) + 1,
    -(2 ** 63) + 2,
    (2 ** 63) - 3,
    (2 ** 63) - 2,
    (2 ** 63) - 1,
}

# 已知 Qdrant v1.17.0 在 ±float32 最大有限值附近的 range 比较存在异常。
# 通用 fuzz 保留这些值以维持边界覆盖；mismatch 日志会标记为已知边界候选，便于归因。
KNOWN_UNSTABLE_FLOAT32_ABS = float(np.finfo(np.float32).max)


def is_known_unstable_int64_value(value) -> bool:
    try:
        iv = int(value)
    except Exception:
        return False
    return iv in KNOWN_UNSTABLE_INT64_VALUES


def is_known_unstable_float32_value(value) -> bool:
    try:
        fv = float(value)
    except Exception:
        return False
    if not math.isfinite(fv):
        return False
    try:
        return bool(abs(np.float32(fv)) == np.float32(KNOWN_UNSTABLE_FLOAT32_ABS))
    except Exception:
        return False


def expr_mentions_known_unstable_int64(expr: str) -> bool:
    if not expr:
        return False
    for token in re.findall(r"-?\d+", expr):
        if is_known_unstable_int64_value(token):
            return True
    return False


def expr_mentions_known_unstable_float32(expr: str) -> bool:
    if not expr:
        return False
    float_literal_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for token in re.findall(float_literal_re, expr):
        if is_known_unstable_float32_value(token):
            return True
    return False


def expr_mentions_known_unstable_boundary(expr: str) -> bool:
    return expr_mentions_known_unstable_int64(expr) or expr_mentions_known_unstable_float32(expr)


def evolved_field_use_empty_semantics() -> bool:
    """
    演进字段（evo_*）的空值语义选择：
    - True:  使用 IsEmpty / NOT IsEmpty（字段缺失、null、[] 视为空）
    - False: 使用 IsNull  / NOT IsNull（严格区分字段缺失与显式 null）
    """
    return not SCHEMA_EVOLUTION_EXPLICIT_NULL_SYNC

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
    UUID = "UUID"
    JSON = "JSON"
    ARRAY_INT = "ARRAY_INT"
    ARRAY_STR = "ARRAY_STR"
    ARRAY_FLOAT = "ARRAY_FLOAT"
    DATETIME = "DATETIME"
    GEO = "GEO"
    ARRAY_OBJECT = "ARRAY_OBJECT"


SAFE_INDEX_PROFILES = {
    FieldType.INT: [
        "integer_default",
        "integer_principal",
        "integer_on_disk",
        "integer_lookup_only_on_disk",
        "integer_range_only_principal_on_disk",
    ],
    FieldType.FLOAT: ["float_default", "float_principal", "float_on_disk"],
    FieldType.BOOL: ["bool_default", "bool_on_disk"],
    FieldType.STRING: ["keyword_default", "keyword_tenant", "keyword_on_disk"],
    FieldType.UUID: ["uuid_default", "uuid_tenant", "uuid_on_disk"],
    FieldType.DATETIME: ["datetime_default", "datetime_principal", "datetime_on_disk"],
    FieldType.GEO: ["geo_default", "geo_on_disk"],
    FieldType.ARRAY_INT: ["integer_default", "integer_on_disk"],
    FieldType.ARRAY_STR: ["keyword_default", "keyword_on_disk"],
    FieldType.ARRAY_FLOAT: ["float_default", "float_on_disk"],
}


def choose_index_profile(ftype):
    choices = SAFE_INDEX_PROFILES.get(ftype)
    if not choices:
        return None
    return random.choice(choices)


def field_config_to_payload_schema(field_config):
    ftype = field_config.get("type")
    profile = field_config.get("index_profile")

    if ftype in (FieldType.STRING, FieldType.ARRAY_STR):
        if profile == "keyword_tenant":
            return KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True)
        if profile == "keyword_on_disk":
            return KeywordIndexParams(type=KeywordIndexType.KEYWORD, on_disk=True)
        return PayloadSchemaType.KEYWORD

    if ftype in (FieldType.INT, FieldType.ARRAY_INT):
        if profile == "integer_lookup_only_on_disk":
            return IntegerIndexParams(
                type=IntegerIndexType.INTEGER,
                lookup=True,
                range=False,
                on_disk=True,
            )
        if profile == "integer_range_only_principal_on_disk":
            return IntegerIndexParams(
                type=IntegerIndexType.INTEGER,
                lookup=False,
                range=True,
                is_principal=True,
                on_disk=True,
            )
        if profile == "integer_principal":
            return IntegerIndexParams(type=IntegerIndexType.INTEGER, is_principal=True)
        if profile == "integer_on_disk":
            return IntegerIndexParams(type=IntegerIndexType.INTEGER, on_disk=True)
        return PayloadSchemaType.INTEGER

    if ftype in (FieldType.FLOAT, FieldType.ARRAY_FLOAT):
        if profile == "float_principal":
            return FloatIndexParams(type=FloatIndexType.FLOAT, is_principal=True)
        if profile == "float_on_disk":
            return FloatIndexParams(type=FloatIndexType.FLOAT, on_disk=True)
        return PayloadSchemaType.FLOAT

    if ftype == FieldType.UUID:
        if profile == "uuid_tenant":
            return UuidIndexParams(type=UuidIndexType.UUID, is_tenant=True)
        if profile == "uuid_on_disk":
            return UuidIndexParams(type=UuidIndexType.UUID, on_disk=True)
        return PayloadSchemaType.UUID

    if ftype == FieldType.BOOL:
        if profile == "bool_on_disk":
            return BoolIndexParams(type=BoolIndexType.BOOL, on_disk=True)
        return PayloadSchemaType.BOOL

    if ftype == FieldType.DATETIME:
        if profile == "datetime_principal":
            return DatetimeIndexParams(type=DatetimeIndexType.DATETIME, is_principal=True)
        if profile == "datetime_on_disk":
            return DatetimeIndexParams(type=DatetimeIndexType.DATETIME, on_disk=True)
        return PayloadSchemaType.DATETIME

    if ftype == FieldType.GEO:
        if profile == "geo_on_disk":
            return GeoIndexParams(type=GeoIndexType.GEO, on_disk=True)
        return PayloadSchemaType.GEO

    return None

def get_type_name(dtype):
    """Map field type to string"""
    return dtype

class DataManager:
    _id_counter = 0  # 类级别 ID 计数器

    def __init__(self, data_seed=None):
        self.df = None
        self.vectors = None
        self.schema_config = []
        self.payload_presence = None
        self.INT64_MIN = -(2 ** 63)
        self.INT64_MAX = (2 ** 63) - 1
        if data_seed is None:
            data_seed = int(np.random.randint(0, 2**31 - 1))
        self.data_seed = int(data_seed)
        self.null_ratio = random.uniform(0.05, 0.15)
        self.boundary_injection_rate = random.uniform(0.08, 0.22)
        self.query_boundary_rate = min(0.35, self.boundary_injection_rate + 0.08)
        self.array_capacity = random.randint(5, 50)
        self.json_max_depth = random.randint(1, 5)
        self.int_range = random.randint(5000, 100000)
        self.double_scale = random.uniform(100, 10000)

    KEY_POOL = [f"k_{i}" for i in range(20)] + ["user", "log", "data", "a_b", "test_key"]
    COUNTRY_CITY_POOL = {
        "Germany": [
            {"name": "Berlin", "population": 3.7, "sightseeing": ["Brandenburg Gate", "Reichstag"]},
            {"name": "Munich", "population": 1.5, "sightseeing": ["Marienplatz", "Olympiapark"]},
        ],
        "Japan": [
            {"name": "Tokyo", "population": 9.3, "sightseeing": ["Tokyo Tower", "Tokyo Skytree"]},
            {"name": "Osaka", "population": 2.7, "sightseeing": ["Osaka Castle", "Universal Studios Japan"]},
        ],
        "France": [
            {"name": "Paris", "population": 2.1, "sightseeing": ["Eiffel Tower", "Louvre Museum"]},
            {"name": "Lyon", "population": 0.5, "sightseeing": ["Basilica of Notre-Dame", "Parc de la Tete d'Or"]},
        ],
        "Brazil": [
            {"name": "Sao Paulo", "population": 12.3, "sightseeing": ["Ibirapuera Park", "Paulista Avenue"]},
            {"name": "Rio de Janeiro", "population": 6.7, "sightseeing": ["Christ the Redeemer", "Copacabana Beach"]},
        ],
    }
    FLOAT32_MAX = float(np.finfo(np.float32).max)
    FLOAT32_MIN_NORMAL = float(np.finfo(np.float32).tiny)
    FLOAT32_MIN_SUBNORMAL = float(np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32))

    @staticmethod
    def _format_datetime_utc(ts):
        if ts is None or pd.isna(ts):
            return None
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        ts = ts.floor("s")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _normalize_datetime_value(self, value):
        if self._is_missing_scalar(value):
            return None
        if isinstance(value, (int, np.integer, float, np.floating)):
            return None
        try:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
        except Exception:
            return None
        if ts is None or pd.isna(ts):
            return None
        return self._format_datetime_utc(ts)

    def _random_datetime_value(self, rng):
        epoch_2020 = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
        epoch_2025 = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        raw = int(rng.integers(epoch_2020, epoch_2025))
        return self._format_datetime_utc(pd.Timestamp(raw, unit="s", tz="UTC"))

    def _boundary_pool(self, ftype):
        """按字段类型返回边界值候选池。"""
        if ftype == FieldType.INT:
            pool = [
                -self.int_range, -1, 0, 1, self.int_range,
                -2147483648, 2147483647,
                -9223372036854775807, 9223372036854775806
            ]
            if not INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES:
                pool = [v for v in pool if not is_known_unstable_int64_value(v)]
            return pool
        if ftype == FieldType.FLOAT:
            return [
                -0.0, 0.0, -1.0, 1.0,
                -self.double_scale, self.double_scale,
                -self.FLOAT32_MIN_SUBNORMAL, self.FLOAT32_MIN_SUBNORMAL,
                -self.FLOAT32_MIN_NORMAL, self.FLOAT32_MIN_NORMAL,
                -self.FLOAT32_MAX, self.FLOAT32_MAX
            ]
        if ftype == FieldType.BOOL:
            return [False, True]
        if ftype == FieldType.STRING:
            return [
                "", " ", "0", "A", "a" * 64,
                "Z" * 256, "!@#$%^&*()_+-=[]{}|;:,.<>?/"
            ]
        if ftype == FieldType.UUID:
            return [
                "00000000-0000-0000-0000-000000000000",
                "00000000-0000-0000-0000-000000000001",
                "ffffffff-ffff-ffff-ffff-ffffffffffff",
            ]
        if ftype == FieldType.JSON:
            return [
                {
                    "price": 0,
                    "color": "Red",
                    "active": True,
                    "config": {"version": 1},
                    "history": [0, 0, 0],
                    "country": copy.deepcopy({
                        "name": "Germany",
                        "cities": [
                            {"name": "Berlin", "population": 3.7, "sightseeing": ["Brandenburg Gate", "Reichstag"]},
                            {"name": "Munich", "population": 1.5, "sightseeing": ["Marienplatz", "Olympiapark"]},
                        ],
                    }),
                },
                {
                    "price": 999,
                    "color": "Blue",
                    "active": False,
                    "config": {"version": 9},
                    "history": [99, 50, 1],
                    "country": copy.deepcopy({
                        "name": "Japan",
                        "cities": [
                            {"name": "Tokyo", "population": 9.3, "sightseeing": ["Tokyo Tower", "Tokyo Skytree"]},
                            {"name": "Osaka", "population": 2.7, "sightseeing": ["Osaka Castle", "Universal Studios Japan"]},
                        ],
                    }),
                },
                {
                    "price": None,
                    "color": "Green",
                    "active": False,
                    "config": {"version": 0},
                    "history": [-1, 100, 2147483647],
                    "country": copy.deepcopy({
                        "name": "France",
                        "cities": [
                            {"name": "Paris", "population": 2.1, "sightseeing": ["Eiffel Tower", "Louvre Museum"]},
                        ],
                    }),
                    "random_payload": ""
                },
                {
                    "price": -1,
                    "color": "",
                    "active": True,
                    "test_key": "edge",
                    "country": copy.deepcopy({
                        "name": "Brazil",
                        "cities": [
                            {"name": "Sao Paulo", "population": 12.3, "sightseeing": ["Ibirapuera Park"]},
                        ],
                    }),
                    "random_payload": {"k": "v"}
                },
            ]
        if ftype == FieldType.ARRAY_INT:
            pool = [
                [], [0], [-1, 0, 1],
                [-2147483648, 2147483647],
                [-9223372036854775807, 9223372036854775806]
            ]
            if not INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES:
                filtered = []
                for arr in pool:
                    if isinstance(arr, list) and any(is_known_unstable_int64_value(v) for v in arr):
                        continue
                    filtered.append(arr)
                pool = filtered
            return pool
        if ftype == FieldType.ARRAY_STR:
            return [
                [], [""], [" ", "0", "A"],
                ["x" * 64, "y" * 128],
                ["!@#", "[]{}"]
            ]
        if ftype == FieldType.ARRAY_FLOAT:
            return [
                [], [0.0], [-0.0],
                [-self.FLOAT32_MIN_SUBNORMAL, self.FLOAT32_MIN_SUBNORMAL],
                [-self.FLOAT32_MAX, self.FLOAT32_MAX]
            ]
        if ftype == FieldType.DATETIME:
            return [
                "1970-01-01T00:00:00Z",
                "2020-01-01T00:00:00Z",
                "2024-02-29T12:34:56Z",
                "2024-12-31T23:59:59Z",
                "2099-12-31T23:59:59Z",
            ]
        if ftype == FieldType.GEO:
            return [
                {"lat": 0.0, "lon": 0.0},
                {"lat": -90.0, "lon": -180.0},
                {"lat": 90.0, "lon": 180.0},
                {"lat": -89.999999, "lon": 179.999999},
            ]
        if ftype == FieldType.ARRAY_OBJECT:
            return [
                [],
                [{"score": 0, "label": "alpha", "active": True}],
                [{"score": -100, "label": "", "active": False}],
                [{"score": 99, "label": "gamma", "active": True},
                 {"score": -99, "label": "delta", "active": False}],
            ]
        return []

    def _pick_boundary_value(self, ftype, rng=None):
        pool = self._boundary_pool(ftype)
        if not pool:
            return None
        if rng is None:
            val = random.choice(pool)
        else:
            val = pool[int(rng.integers(0, len(pool)))]
        return copy.deepcopy(val)

    def sample_boundary_value(self, ftype):
        """供查询生成器使用。"""
        return self._pick_boundary_value(ftype, rng=None)

    def _maybe_use_boundary_value(self, value, ftype, rng):
        if value is None:
            return None
        if float(rng.random()) < self.boundary_injection_rate:
            boundary_val = self._pick_boundary_value(ftype, rng=rng)
            if boundary_val is not None:
                return boundary_val
        return value

    def _inject_boundary_values(self, values, ftype, rng):
        if not isinstance(values, list):
            values = list(values)
        for idx, val in enumerate(values):
            if val is None:
                continue
            if float(rng.random()) < self.boundary_injection_rate:
                boundary_val = self._pick_boundary_value(ftype, rng=rng)
                if boundary_val is not None:
                    values[idx] = boundary_val
        return values

    @staticmethod
    def _is_missing_scalar(v):
        if v is None or v is pd.NA:
            return True
        if isinstance(v, (float, np.floating)):
            return bool(np.isnan(v))
        return False

    def _clip_int64(self, v):
        iv = int(v)
        if iv < self.INT64_MIN:
            return self.INT64_MIN
        if iv > self.INT64_MAX:
            return self.INT64_MAX
        return iv

    def _to_python_nested(self, obj):
        """递归将 numpy/pandas 标量转换为 Python 原生类型，并规范缺失值。"""
        if obj is None or obj is pd.NA:
            return None
        if isinstance(obj, (float, np.floating)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: self._to_python_nested(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [self._to_python_nested(v) for v in list(obj)]
        return obj

    def _normalize_scalar_for_field(self, v, ftype):
        if self._is_missing_scalar(v):
            return None
        if ftype == FieldType.INT:
            try:
                return self._clip_int64(v)
            except Exception:
                return None
        if ftype == FieldType.DATETIME:
            return self._normalize_datetime_value(v)
        if ftype == FieldType.BOOL:
            try:
                return bool(v)
            except Exception:
                return None
        if ftype == FieldType.UUID:
            try:
                return str(uuid.UUID(str(v)))
            except Exception:
                return None
        if ftype == FieldType.FLOAT:
            try:
                fv = float(v)
                if np.isnan(fv):
                    return None
                return fv
            except Exception:
                return None
        if ftype == FieldType.ARRAY_INT:
            if self._is_missing_scalar(v):
                return None
            if not isinstance(v, (list, tuple, np.ndarray)):
                v = [v]
            out = []
            for x in list(v):
                if self._is_missing_scalar(x):
                    continue
                try:
                    out.append(self._clip_int64(x))
                except Exception:
                    continue
            return out
        if ftype == FieldType.ARRAY_FLOAT:
            if self._is_missing_scalar(v):
                return None
            if not isinstance(v, (list, tuple, np.ndarray)):
                v = [v]
            out = []
            for x in list(v):
                if self._is_missing_scalar(x):
                    continue
                try:
                    out.append(float(x))
                except Exception:
                    continue
            return out
        if ftype == FieldType.ARRAY_STR:
            if self._is_missing_scalar(v):
                return None
            if not isinstance(v, (list, tuple, np.ndarray)):
                v = [v]
            return [str(x) for x in list(v) if not self._is_missing_scalar(x)]
        if ftype == FieldType.GEO:
            if self._is_missing_scalar(v):
                return None
            if not isinstance(v, dict):
                return None
            lat = v.get("lat")
            lon = v.get("lon")
            if self._is_missing_scalar(lat) or self._is_missing_scalar(lon):
                return None
            try:
                return {"lat": float(lat), "lon": float(lon)}
            except Exception:
                return None
        if ftype == FieldType.JSON:
            return self._to_python_nested(v)
        if ftype == FieldType.ARRAY_OBJECT:
            if self._is_missing_scalar(v):
                return None
            if not isinstance(v, (list, tuple)):
                return None
            return [self._to_python_nested(item) for item in v]
        return v

    def normalize_row_for_storage(self, row):
        """
        按 schema 规范化单行数据，避免 payload 在写入 Qdrant 前发生类型漂移。
        """
        norm = {}
        rid = row.get("id")
        norm["id"] = self._clip_int64(rid) if rid is not None else None
        field_type_map = {f["name"]: f["type"] for f in self.schema_config}
        for k, v in row.items():
            if k == "id":
                continue
            ftype = field_type_map.get(k)
            if ftype is None:
                norm[k] = self._to_python_nested(v)
            else:
                norm[k] = self._normalize_scalar_for_field(v, ftype)
        return norm

    def coerce_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按 schema 将任意 DataFrame 显式转换为稳定 dtype。
        供主表与动态批量 DataFrame 复用，避免“先推断为 float 再回转 int”的精度漂移。
        """
        if df is None:
            return df
        out = df.copy()

        if "id" in out.columns:
            out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")

        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]
            if fname not in out.columns:
                continue

            if ftype == FieldType.INT:
                raw_vals = out[fname].tolist()
                coerced = [pd.NA if self._is_missing_scalar(x) else self._clip_int64(x) for x in raw_vals]
                out[fname] = pd.Series(pd.array(coerced, dtype="Int64"), index=out.index)
            elif ftype == FieldType.DATETIME:
                out[fname] = out[fname].apply(lambda x: self._normalize_datetime_value(x))
            elif ftype == FieldType.BOOL:
                raw_vals = out[fname].tolist()
                coerced = [pd.NA if self._is_missing_scalar(x) else bool(x) for x in raw_vals]
                out[fname] = pd.Series(pd.array(coerced, dtype="boolean"), index=out.index)
            elif ftype == FieldType.FLOAT:
                out[fname] = pd.to_numeric(out[fname], errors="coerce")
            elif ftype in [FieldType.UUID, FieldType.ARRAY_INT, FieldType.ARRAY_STR, FieldType.ARRAY_FLOAT, FieldType.GEO, FieldType.JSON, FieldType.ARRAY_OBJECT]:
                out[fname] = out[fname].apply(lambda x, t=ftype: self._normalize_scalar_for_field(x, t))

        return out

    def normalize_dataframe_types(self):
        """
        统一主 DataFrame 中标量字段类型，避免 INT 在带 NULL 场景下漂移成 float，
        并将 DATETIME 规范化为 UTC 字符串，降低 Oracle 比较语义失真。
        """
        if self.df is None:
            return
        self.df = self.coerce_dataframe_types(self.df)

    def schema_field_names(self):
        return [field["name"] for field in self.schema_config]

    def initialize_payload_presence(self, present: bool = True):
        """
        Track whether a payload key is physically present for each row.

        Pandas stores both missing payload keys and explicit JSON null as
        None/NA. Qdrant's `is_null` distinguishes them, so payload mutation
        tests need a side-channel to keep the oracle precise.
        """
        if self.df is None:
            return
        self.payload_presence = pd.DataFrame(
            bool(present),
            index=self.df.index,
            columns=self.schema_field_names(),
            dtype=bool,
        )

    def ensure_payload_presence(self):
        if self.df is None:
            return
        field_names = self.schema_field_names()
        if self.payload_presence is None:
            self.initialize_payload_presence(present=True)
            return

        self.payload_presence = self.payload_presence.reindex(index=self.df.index)
        for field_name in field_names:
            if field_name not in self.payload_presence.columns:
                self.payload_presence[field_name] = True
        extra_columns = [col for col in self.payload_presence.columns if col not in field_names]
        if extra_columns:
            self.payload_presence = self.payload_presence.drop(columns=extra_columns)
        self.payload_presence = self.payload_presence[field_names].fillna(True).astype(bool)

    def add_payload_presence_field(self, field_name: str, present: bool = False):
        if self.df is None:
            return
        self.ensure_payload_presence()
        self.payload_presence[field_name] = bool(present)

    def set_payload_presence_by_indices(self, indices, field_names, present: bool):
        if self.df is None:
            return
        self.ensure_payload_presence()
        if isinstance(field_names, str):
            field_names = [field_names]
        clean_indices = [int(idx) for idx in indices if int(idx) in self.df.index]
        clean_fields = [field for field in field_names if field in self.payload_presence.columns]
        if clean_indices and clean_fields:
            self.payload_presence.loc[clean_indices, clean_fields] = bool(present)

    def append_payload_presence_rows(self, count: int, present: bool = True):
        if count <= 0:
            return
        self.ensure_payload_presence()
        start = len(self.payload_presence)
        new_index = range(start, start + count)
        new_presence = pd.DataFrame(
            bool(present),
            index=new_index,
            columns=self.schema_field_names(),
            dtype=bool,
        )
        self.payload_presence = pd.concat([self.payload_presence, new_presence], axis=0)

    def drop_payload_presence_indices(self, indices):
        self.ensure_payload_presence()
        clean_indices = [int(idx) for idx in indices if int(idx) in self.payload_presence.index]
        if clean_indices:
            self.payload_presence = self.payload_presence.drop(clean_indices).reset_index(drop=True)

    def _missing_value_for_field(self, field_name: str):
        field_type = next((field["type"] for field in self.schema_config if field["name"] == field_name), None)
        if field_type in (FieldType.INT, FieldType.BOOL):
            return pd.NA
        return None

    def mark_payload_fields_missing(self, indices, field_names):
        if isinstance(field_names, str):
            field_names = [field_names]
        clean_indices = [int(idx) for idx in indices if int(idx) in self.df.index]
        clean_fields = [field for field in field_names if field in self.df.columns]
        if not clean_indices or not clean_fields:
            return
        for field_name in clean_fields:
            self.df.loc[clean_indices, field_name] = self._missing_value_for_field(field_name)
        self.set_payload_presence_by_indices(clean_indices, clean_fields, False)
        self.normalize_dataframe_types()

    def mark_payload_fields_present(self, indices, field_names):
        self.set_payload_presence_by_indices(indices, field_names, True)

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

    def _random_uuid_string(self):
        return str(uuid.UUID(int=random.getrandbits(128)))

    def _build_country_payload(self, rng):
        country_name = str(rng.choice(list(self.COUNTRY_CITY_POOL.keys())))
        templates = self.COUNTRY_CITY_POOL[country_name]
        city_count = int(rng.integers(1, len(templates) + 1))
        selected_indices = rng.choice(len(templates), size=city_count, replace=False)
        if np.isscalar(selected_indices):
            selected_indices = [int(selected_indices)]
        cities = []
        for idx in list(selected_indices):
            city = copy.deepcopy(templates[int(idx)])
            if rng.random() < 0.2 and isinstance(city.get("sightseeing"), list) and len(city["sightseeing"]) > 1:
                city["sightseeing"] = city["sightseeing"][:1]
            cities.append(city)
        return {"name": country_name, "cities": cities}

    def _build_json_payload(self, rng):
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

        # Keep one stable documented nested-key shape in every non-null JSON object.
        base_obj["country"] = self._build_country_payload(rng)
        return base_obj

    def generate_schema(self):
        print("🎲 1. Defining Dynamic Schema...")
        self.schema_config = []
        num_fields = random.randint(3, 20)
        types_pool = [FieldType.INT, FieldType.FLOAT, FieldType.BOOL, FieldType.STRING, FieldType.UUID, FieldType.DATETIME]

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

        # 添加 Nested 对象数组字段（用于 NestedCondition 测试）
        self.schema_config.append({
            "name": "items_nested",
            "type": FieldType.ARRAY_OBJECT,
            "sub_fields": [
                {"name": "score", "type": "int"},
                {"name": "label", "type": "str"},
                {"name": "active", "type": "bool"},
            ]
        })

        for field in self.schema_config:
            field["index_profile"] = choose_index_profile(field["type"])

        print(f"   -> Generated {len(self.schema_config)} dynamic fields (plus id & vector).")
        print("   -> Schema Structure:")
        for f in self.schema_config:
            t_name = get_type_name(f["type"])
            profile = f.get("index_profile")
            if profile:
                print(f"      - {f['name']:<15} : {t_name} | index={profile}")
            else:
                print(f"      - {f['name']:<15} : {t_name}")

    def generate_data(self):
        print(f"🌊 2. Generating {N} rows (Vector Dim={DIM})...")
        print(f"   -> Data RNG seed: {self.data_seed}")
        print(f"   -> Boundary injection rate: {self.boundary_injection_rate:.2%}")
        rng = np.random.default_rng(self.data_seed)
        self.vectors = rng.random((N, DIM), dtype=np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1)[:, np.newaxis]

        data = {"id": np.arange(N, dtype=np.int64)}
        print("   -> Filling scalar attributes...")
        
        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]

            if ftype == FieldType.INT:
                values = rng.integers(-self.int_range, self.int_range, size=N).tolist()
                values = self._inject_boundary_values(values, ftype, rng)
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.FLOAT:
                values = (rng.random(N) * self.double_scale).tolist()
                values = self._inject_boundary_values(values, ftype, rng)
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.BOOL:
                values = rng.choice([True, False], size=N).tolist()
                values = self._inject_boundary_values(values, ftype, rng)
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.STRING:
                values = [self._random_string(0, random.randint(5, 50)) for _ in range(N)]
                values = self._inject_boundary_values(values, ftype, rng)
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.UUID:
                values = [self._random_uuid_string() for _ in range(N)]
                values = self._inject_boundary_values(values, ftype, rng)
                data[fname] = self._apply_nulls(values, rng)
            elif ftype == FieldType.JSON:
                json_list = []
                for _ in range(N):
                    if rng.random() < self.null_ratio:
                        json_list.append(None)
                        continue
                    json_list.append(self._build_json_payload(rng))
                json_list = self._inject_boundary_values(json_list, ftype, rng)
                data[fname] = json_list
            elif ftype == FieldType.ARRAY_INT:
                arr_list = []
                for _ in range(N):
                    length = random.randint(0, 5)
                    arr_list.append(list(rng.integers(0, 100, size=length)))
                arr_list = self._inject_boundary_values(arr_list, ftype, rng)
                data[fname] = self._apply_nulls(arr_list, rng)
            elif ftype == FieldType.ARRAY_STR:
                arr_list = []
                for _ in range(N):
                    length = random.randint(0, 5)
                    arr_list.append([self._random_string(2, 8) for _ in range(length)])
                arr_list = self._inject_boundary_values(arr_list, ftype, rng)
                data[fname] = self._apply_nulls(arr_list, rng)
            elif ftype == FieldType.ARRAY_FLOAT:
                arr_list = []
                for _ in range(N):
                    length = random.randint(0, 5)
                    arr_list.append([round(float(rng.random() * 1000), 2) for _ in range(length)])
                arr_list = self._inject_boundary_values(arr_list, ftype, rng)
                data[fname] = self._apply_nulls(arr_list, rng)
            elif ftype == FieldType.DATETIME:
                values = [self._random_datetime_value(rng) for _ in range(N)]
                values = self._inject_boundary_values(values, ftype, rng)
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
                geo_list = self._inject_boundary_values(geo_list, ftype, rng)
                data[fname] = geo_list
            elif ftype == FieldType.ARRAY_OBJECT:
                # 生成对象数组：每行含 0-5 个子对象，每个子对象带 score(int)/label(str)/active(bool)
                nested_labels = ["alpha", "beta", "gamma", "delta", "epsilon"]
                obj_list = []
                for _ in range(N):
                    if rng.random() < self.null_ratio:
                        obj_list.append(None)
                        continue
                    length = int(rng.integers(1, 6))  # 1-5 个子对象
                    arr = []
                    for _ in range(length):
                        arr.append({
                            "score": int(rng.integers(-100, 100)),
                            "label": str(rng.choice(nested_labels)),
                            "active": bool(rng.choice([True, False])),
                        })
                    obj_list.append(arr)
                obj_list = self._inject_boundary_values(obj_list, ftype, rng)
                data[fname] = obj_list

        # 在 DataFrame 构造前先固定关键标量列的 dtype，避免“int + null”先被推断成 float
        # 造成 64-bit 边界值在 normalize 前就发生精度漂移。
        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]
            if fname not in data:
                continue
            if ftype == FieldType.INT:
                data[fname] = pd.array(
                    [pd.NA if self._is_missing_scalar(x) else self._clip_int64(x) for x in data[fname]],
                    dtype="Int64"
                )
            elif ftype == FieldType.BOOL:
                data[fname] = pd.array(
                    [pd.NA if self._is_missing_scalar(x) else bool(x) for x in data[fname]],
                    dtype="boolean"
                )

        self.df = pd.DataFrame(data)
        self.normalize_dataframe_types()
        self.initialize_payload_presence(present=True)
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

            # 按 null_ratio 概率生成 NULL（与 generate_data 的 _apply_nulls 保持一致）
            # JSON 和 GEO 类型有自己的 null 处理逻辑，其它标量字段统一处理
            if ftype not in [FieldType.JSON, FieldType.GEO] and rng.random() < self.null_ratio:
                row[fname] = None
                continue

            if ftype == FieldType.INT:
                val = int(rng.integers(-self.int_range, self.int_range))
                row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.FLOAT:
                val = float(rng.random() * self.double_scale)
                row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.BOOL:
                val = bool(rng.choice([True, False]))
                row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.STRING:
                val = self._random_string(0, random.randint(5, 50))
                row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.UUID:
                val = self._random_uuid_string()
                row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.JSON:
                if rng.random() < self.null_ratio:
                    row[fname] = None
                    continue
                row[fname] = self._maybe_use_boundary_value(self._build_json_payload(rng), ftype, rng)
            elif ftype == FieldType.ARRAY_INT:
                arr_len = rng.integers(0, 6)
                val = [int(x) for x in rng.integers(0, 100, size=arr_len)]
                row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.ARRAY_STR:
                arr_len = rng.integers(0, 6)
                val = [self._random_string(2, 8) for _ in range(arr_len)]
                row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.ARRAY_FLOAT:
                arr_len = rng.integers(0, 6)
                val = [round(float(rng.random() * 1000), 2) for _ in range(arr_len)]
                row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.DATETIME:
                val = self._random_datetime_value(rng)
                row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.GEO:
                if rng.random() < self.null_ratio:
                    row[fname] = None
                else:
                    val = {
                        "lat": round(float(rng.uniform(-90, 90)), 6),
                        "lon": round(float(rng.uniform(-180, 180)), 6)
                    }
                    row[fname] = self._maybe_use_boundary_value(val, ftype, rng)
            elif ftype == FieldType.ARRAY_OBJECT:
                if rng.random() < self.null_ratio:
                    row[fname] = None
                else:
                    nested_labels = ["alpha", "beta", "gamma", "delta", "epsilon"]
                    length = int(rng.integers(1, 6))
                    arr = []
                    for _ in range(length):
                        arr.append({
                            "score": int(rng.integers(-100, 100)),
                            "label": str(rng.choice(nested_labels)),
                            "active": bool(rng.choice([True, False])),
                        })
                    row[fname] = self._maybe_use_boundary_value(arr, ftype, rng)
        return row

    def generate_single_vector(self):
        """生成一条单位化的向量"""
        vec = np.random.randn(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    def generate_field_value(self, field_config):
        """为给定字段配置生成一个随机值（用于 schema evolution 回填）"""
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        ftype = field_config["type"]
        value = None

        if ftype == FieldType.INT:
            value = int(rng.integers(-self.int_range, self.int_range))
        elif ftype == FieldType.FLOAT:
            value = float(rng.random() * self.double_scale)
        elif ftype == FieldType.BOOL:
            value = bool(rng.choice([True, False]))
        elif ftype == FieldType.STRING:
            value = self._random_string(0, random.randint(5, 50))
        elif ftype == FieldType.UUID:
            value = self._random_uuid_string()
        elif ftype == FieldType.DATETIME:
            value = self._random_datetime_value(rng)
        elif ftype == FieldType.JSON:
            value = self._build_json_payload(rng)
        elif ftype == FieldType.ARRAY_INT:
            arr_len = rng.integers(0, 6)
            value = [int(x) for x in rng.integers(0, 100, size=arr_len)]
        elif ftype == FieldType.ARRAY_STR:
            arr_len = rng.integers(0, 6)
            value = [self._random_string(2, 8) for _ in range(arr_len)]
        elif ftype == FieldType.ARRAY_FLOAT:
            arr_len = rng.integers(0, 6)
            value = [round(float(rng.random() * 1000), 2) for _ in range(arr_len)]
        elif ftype == FieldType.GEO:
            value = {
                "lat": round(float(rng.uniform(-90, 90)), 6),
                "lon": round(float(rng.uniform(-180, 180)), 6)
            }
        elif ftype == FieldType.ARRAY_OBJECT:
            nested_labels = ["alpha", "beta", "gamma", "delta", "epsilon"]
            length = int(rng.integers(1, 6))
            value = []
            for _ in range(length):
                value.append({
                    "score": int(rng.integers(-100, 100)),
                    "label": str(rng.choice(nested_labels)),
                    "active": bool(rng.choice([True, False])),
                })

        return self._maybe_use_boundary_value(value, ftype, rng)

    def evolve_schema_add_field(self):
        """
        Schema Evolution: 随机添加一个新字段到 schema 配置和 pandas DataFrame。
        新字段对所有现有数据默认为 None。

        Returns:
            field_config dict if successful, None if limit reached.
        """
        evolved_count = sum(1 for f in self.schema_config if f["name"].startswith("evo_"))
        if evolved_count >= MAX_EVOLVED_FIELDS:
            return None

        # 选择随机类型（标量为主，保持简单）
        type_choices = [FieldType.INT, FieldType.FLOAT, FieldType.BOOL, FieldType.STRING, FieldType.UUID, FieldType.DATETIME]
        ftype = random.choice(type_choices)

        type_name = get_type_name(ftype).lower()
        field_name = f"evo_{evolved_count}_{type_name}"

        field_config = {"name": field_name, "type": ftype}

        # 添加到 schema_config（OracleQueryGenerator 通过引用自动可见）
        self.schema_config.append(field_config)

        # 更新 pandas DataFrame: 所有现有行设为缺失值，并尽量保持类型稳定
        if ftype == FieldType.INT:
            self.df[field_name] = pd.Series([pd.NA] * len(self.df), dtype="Int64")
        elif ftype == FieldType.BOOL:
            self.df[field_name] = pd.Series([pd.NA] * len(self.df), dtype="boolean")
        else:
            self.df[field_name] = None
        self.add_payload_presence_field(field_name, present=False)
        self.normalize_dataframe_types()

        return field_config

    def backfill_evolved_field(self, field_config, fill_ratio=None):
        """
        为现有数据的新演进字段生成回填值，并同步更新 pandas DataFrame。

        Args:
            field_config: 字段配置 dict
            fill_ratio: 填充比例 (0.0 - 1.0)，None 则随机选取

        Returns:
            list of (pandas_index, row_id, value) tuples
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

        self.mark_payload_fields_present(fill_indices, field_name)
        self.normalize_dataframe_types()
        return backfill_data

# --- 2. Qdrant Manager ---

class QdrantManager:
    def __init__(self):
        self.client = None
        self._transport_info = {}

    def connect(self):
        proto = "gRPC" if PREFER_GRPC else "REST"
        print(f"🔌 Connecting to Qdrant ({proto}) at {HOST}:{PORT} (grpc:{GRPC_PORT})...")
        try:
            # 关键：port 始终是 REST 端口；gRPC 走独立 grpc_port。
            self.client = QdrantClient(
                host=HOST,
                port=PORT,
                grpc_port=GRPC_PORT,
                prefer_grpc=PREFER_GRPC,
                timeout=30
            )
            # 测试连接
            self.client.get_collections()
            backend = getattr(self.client, "_client", None)
            backend_cls = type(backend).__name__ if backend is not None else "unknown"
            actual_prefer_grpc = getattr(backend, "_prefer_grpc", None)
            self._transport_info = {
                "host": HOST,
                "rest_port": PORT,
                "grpc_port": GRPC_PORT,
                "requested_prefer_grpc": bool(PREFER_GRPC),
                "backend_class": backend_cls,
                "backend_prefer_grpc": actual_prefer_grpc,
            }
            print(
                "🔎 Transport Check: "
                f"backend={backend_cls}, requested_prefer_grpc={PREFER_GRPC}, "
                f"backend_prefer_grpc={actual_prefer_grpc}"
            )
            print("✅ Connected to Qdrant successfully.")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("   (Please check if Qdrant container is running)")
            exit(1)

    def get_transport_info(self):
        if self._transport_info:
            return dict(self._transport_info)
        return {
            "host": HOST,
            "rest_port": PORT,
            "grpc_port": GRPC_PORT,
            "requested_prefer_grpc": bool(PREFER_GRPC),
            "backend_class": "unknown",
            "backend_prefer_grpc": None,
        }

    def _pick_read_consistency(self):
        if READ_CONSISTENCY == "random":
            raw = random.choice(ALL_READ_CONSISTENCY_LEVELS)
        else:
            raw = READ_CONSISTENCY
        if isinstance(raw, int):
            return raw
        if isinstance(raw, models.ReadConsistencyType):
            return raw
        return models.ReadConsistencyType(raw)

    def _pick_write_ordering(self):
        if WRITE_ORDERING == "random":
            raw = random.choice(ALL_WRITE_ORDERING_LEVELS)
        else:
            raw = WRITE_ORDERING
        if isinstance(raw, models.WriteOrdering):
            return raw
        return models.WriteOrdering(raw)

    def retrieve(self, **kwargs):
        kwargs.setdefault("consistency", self._pick_read_consistency())
        return self.client.retrieve(**kwargs)

    def scroll(self, **kwargs):
        kwargs.setdefault("consistency", self._pick_read_consistency())
        return self.client.scroll(**kwargs)

    def query_points(self, **kwargs):
        kwargs.setdefault("consistency", self._pick_read_consistency())
        return self.client.query_points(**kwargs)

    def query_points_groups(self, **kwargs):
        kwargs.setdefault("consistency", self._pick_read_consistency())
        return self.client.query_points_groups(**kwargs)

    def count(self, **kwargs):
        consistency = kwargs.pop("consistency", None)
        if consistency is None:
            consistency = self._pick_read_consistency()
        try:
            return self.client.count(consistency=consistency, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if "Unknown arguments: ['consistency']" not in msg and "unexpected keyword argument 'consistency'" not in msg:
                raise
            return self.client.count(**kwargs)

    def upsert(self, **kwargs):
        kwargs.setdefault("ordering", self._pick_write_ordering())
        return self.client.upsert(**kwargs)

    def set_payload(self, **kwargs):
        kwargs.setdefault("ordering", self._pick_write_ordering())
        return self.client.set_payload(**kwargs)

    def overwrite_payload(self, **kwargs):
        kwargs.setdefault("ordering", self._pick_write_ordering())
        return self.client.overwrite_payload(**kwargs)

    def delete_payload(self, **kwargs):
        kwargs.setdefault("ordering", self._pick_write_ordering())
        return self.client.delete_payload(**kwargs)

    def clear_payload(self, **kwargs):
        kwargs.setdefault("ordering", self._pick_write_ordering())
        return self.client.clear_payload(**kwargs)

    def create_payload_index(self, **kwargs):
        kwargs.setdefault("ordering", self._pick_write_ordering())
        return self.client.create_payload_index(**kwargs)

    def delete_payload_index(self, **kwargs):
        kwargs.setdefault("ordering", self._pick_write_ordering())
        return self.client.delete_payload_index(**kwargs)

    def reset_collection(self, schema_config):
        global DISTANCE_TYPE
        # 延迟初始化距离度量（在种子设置后随机选取）
        if FORCED_DISTANCE_TYPE is not None:
            DISTANCE_TYPE = FORCED_DISTANCE_TYPE
        elif DISTANCE_TYPE is None:
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
        dm.normalize_dataframe_types()
        records = dm.df.to_dict(orient="records")
        total = len(records)
        
        # 构建字段类型映射
        field_type_map = {f["name"]: f["type"] for f in dm.schema_config}
        int64_min = dm.INT64_MIN
        int64_max = dm.INT64_MAX

        def is_missing_scalar(v):
            if v is None or v is pd.NA:
                return True
            if isinstance(v, (float, np.floating)):
                return bool(np.isnan(v))
            return False

        def clip_int64(v):
            iv = int(v)
            if iv < int64_min:
                return int64_min
            if iv > int64_max:
                return int64_max
            return iv

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
            ftype = field_type_map.get(k)
            if ftype is None:
                return convert_numpy_types(v)
            return dm._normalize_scalar_for_field(v, ftype)

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
                    self.upsert(
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
            field_schema = field_config_to_payload_schema(field)
            if field_schema is None:
                continue
            try:
                self.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=fname,
                    field_schema=field_schema
                )
            except Exception as e:
                # 索引创建失败不是致命错误
                pass
        print("✅ Index creation complete.")

    def sync_float_fields_from_storage(self, dm, batch_size=None):
        """
        在 gRPC 路径下，Qdrant 对极值 float 的序列化可能与本地 pandas 侧存在 ULP 级差异。
        该方法回读实际 payload，将 FLOAT / ARRAY_FLOAT 字段对齐到“库内真实值”，
        避免 oracle 在边界比较时出现假阳性 mismatch。
        """
        if batch_size is None:
            batch_size = BATCH_SIZE

        float_fields = [f["name"] for f in dm.schema_config if f["type"] == FieldType.FLOAT]
        array_float_fields = [f["name"] for f in dm.schema_config if f["type"] == FieldType.ARRAY_FLOAT]
        if not float_fields and not array_float_fields:
            return 0

        ids = [int(x) for x in dm.df["id"].tolist()]
        id_to_idx = {int(dm.df.at[i, "id"]): int(i) for i in range(len(dm.df))}
        touched = 0

        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            try:
                recs = self.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=batch_ids,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                continue

            for rec in recs:
                pid = int(rec.id)
                idx = id_to_idx.get(pid)
                if idx is None:
                    continue
                payload = rec.payload or {}

                for fname in float_fields:
                    v = payload.get(fname, None)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        dm.df.at[idx, fname] = float(v)
                    elif v is None:
                        dm.df.at[idx, fname] = None
                    else:
                        dm.df.at[idx, fname] = None

                for fname in array_float_fields:
                    v = payload.get(fname, None)
                    if v is None:
                        dm.df.at[idx, fname] = None
                    elif isinstance(v, list):
                        ok = True
                        arr = []
                        for x in v:
                            if isinstance(x, (int, float)) and not isinstance(x, bool):
                                arr.append(float(x))
                            else:
                                ok = False
                                break
                        dm.df.at[idx, fname] = arr if ok else None
                    else:
                        dm.df.at[idx, fname] = None

                touched += 1

        dm.normalize_dataframe_types()
        return touched

    def backfill_field_data(self, field_name, backfill_data):
        """
        通过 set_payload 为现有数据的新字段回填值。

        Args:
            field_name: 字段名
            backfill_data: list of (pandas_idx, row_id, value) tuples

        Returns:
            成功回填的行数
        """
        if not backfill_data:
            return 0

        success_count = 0
        for start in range(0, len(backfill_data), BATCH_SIZE):
            batch = backfill_data[start:start + BATCH_SIZE]
            for _, row_id, value in batch:
                try:
                    self.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={field_name: value},
                        points=[row_id],
                        wait=True
                    )
                    success_count += 1
                except Exception as e:
                    pass  # 回填单条失败不是致命错误
        return success_count

    def create_evolved_field_index(self, field_config):
        """为演进字段创建 payload 索引"""
        fname = field_config["name"]
        schema_cfg = dict(field_config)
        schema_cfg.setdefault("index_profile", choose_index_profile(schema_cfg.get("type")))
        schema_type = field_config_to_payload_schema(schema_cfg)
        if schema_type is not None:
            try:
                self.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=fname,
                    field_schema=schema_type,
                    wait=True
                )
            except Exception:
                pass  # 索引创建失败不致命

    def set_null_for_points(self, field_name, point_ids):
        """
        为指定批量 point 的某字段显式设置为 None。
        解决 Qdrant 中“字段不存在”≠“字段为 null”的语义差异。
        """
        if not point_ids:
            return 0
        success = 0
        ids = [int(x) for x in point_ids]

        # set_payload(field=None) 在 Qdrant 中不会持久化显式 null；
        # 使用“读取原向量+payload，再 upsert 合并写回”来确保 null 被显式保存。
        for start in range(0, len(ids), BATCH_SIZE):
            batch = ids[start:start + BATCH_SIZE]
            try:
                points = self.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=batch,
                    with_payload=True,
                    with_vectors=True,
                )
                points_by_id = {int(p.id): p for p in points}
            except Exception:
                points_by_id = {}

            for pid in batch:
                try:
                    p = points_by_id.get(pid)
                    if p is None:
                        continue

                    current_payload = dict(p.payload or {})
                    if current_payload.get(field_name, "__MISSING__") is None:
                        success += 1
                        continue

                    current_payload[field_name] = None

                    vec = p.vector
                    # 命名向量场景下，取第一个向量值
                    if isinstance(vec, dict):
                        if not vec:
                            continue
                        vec = next(iter(vec.values()))

                    self.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[PointStruct(id=pid, vector=vec, payload=current_payload)],
                        wait=True,
                    )
                    success += 1
                except Exception:
                    pass
        return success

    def rebuild_payload_indexes(self, schema_config):
        """
        随机删除并重建 payload 索引，测试索引重建后的数据一致性。
        每次随机选择 1~3 个字段进行索引重建。
        """
        indexable = [f for f in schema_config if f["type"] in (
            FieldType.INT, FieldType.FLOAT, FieldType.STRING, FieldType.UUID,
            FieldType.BOOL, FieldType.DATETIME, FieldType.GEO,
            FieldType.ARRAY_INT, FieldType.ARRAY_STR, FieldType.ARRAY_FLOAT,
        )]
        if not indexable:
            return []

        n_rebuild = random.randint(1, min(3, len(indexable)))
        fields_to_rebuild = random.sample(indexable, n_rebuild)
        rebuilt = []

        for field in fields_to_rebuild:
            fname = field["name"]
            field_schema = field_config_to_payload_schema(field)
            if field_schema is None:
                continue
            try:
                # 删除索引
                self.delete_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=fname,
                    wait=True
                )
                time.sleep(0.1)
                # 重建索引
                self.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=fname,
                    field_schema=field_schema,
                    wait=True
                )
                rebuilt.append(fname)
            except Exception as e:
                pass  # 索引重建失败不是致命错误

        return rebuilt

# --- 3. Query Generator (Oracle Mode) ---

class OracleQueryGenerator:
    """
    生成 Qdrant 过滤条件并同时生成对应的 Pandas mask
    用于 Oracle 对比测试
    """
    def __init__(self, dm):
        self.schema = dm.schema_config
        self._dm = dm  # 保存 dm 引用，动态获取最新 df
        self._query_boundary_rate = dm.query_boundary_rate
        self.INT64_MIN = -(2 ** 63)
        self.INT64_MAX = (2 ** 63) - 1

    @property
    def df(self):
        """动态获取最新的 DataFrame（支持动态插入/删除/upsert 后数据同步）"""
        return self._dm.df

    def _random_string(self, min_len=5, max_len=10):
        if max_len < min_len:
            max_len = min_len
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=random.randint(min_len, max_len)))

    def _random_uuid_string(self):
        return self._dm._random_uuid_string()

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

    def _is_evolved_field(self, name):
        """判断字段是否为 Schema Evolution 新增字段（evo_ 前缀）。
        演化字段未回填的行在 Qdrant 里"字段不存在"，需用 IsEmptyCondition 而非 IsNullCondition。"""
        return name.startswith("evo_")

    @staticmethod
    def _is_na_like(x):
        if x is None or x is pd.NA:
            return True
        if isinstance(x, (float, np.floating)):
            return bool(np.isnan(x))
        return False

    def _clip_int64(self, v):
        iv = int(v)
        if iv < self.INT64_MIN:
            return self.INT64_MIN
        if iv > self.INT64_MAX:
            return self.INT64_MAX
        return iv

    def _offset_int64(self, base, delta):
        return self._clip_int64(int(base) + int(delta))

    def _normalize_int_scalar(self, x):
        if self._is_na_like(x):
            return None
        try:
            return self._clip_int64(x)
        except Exception:
            return None

    @staticmethod
    def _format_datetime_utc(ts):
        if ts is None or pd.isna(ts):
            return None
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        ts = ts.floor("s")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _normalize_datetime_scalar(self, x):
        if self._is_na_like(x):
            return None
        if isinstance(x, (int, np.integer, float, np.floating)):
            return None
        try:
            ts = pd.to_datetime(x, utc=True, errors="coerce")
        except Exception:
            return None
        if ts is None or pd.isna(ts):
            return None
        return ts.floor("s")

    def _offset_datetime_str(self, value, *, seconds=0, days=0):
        ts = self._normalize_datetime_scalar(value)
        if ts is None:
            return None
        shifted = ts + pd.Timedelta(days=days, seconds=seconds)
        return self._format_datetime_utc(shifted)

    def _as_bool_mask(self, mask):
        """统一 mask 为 pandas nullable boolean Series，避免 object/float 掩码的隐式行为。"""
        if isinstance(mask, pd.Series):
            s = mask
        else:
            s = pd.Series(mask, index=self.df.index)
        if not s.index.equals(self.df.index):
            s = s.reindex(self.df.index)
        return s.astype("boolean")

    def _safe_apply(self, series, fn):
        """
        对 Series 执行逐元素函数时先转 object，规避 pandas nullable Int64
        在 apply/map 中隐式转为 float64 导致的 64-bit 边界值精度丢失。
        """
        return series.astype(object).apply(fn)

    def _field_presence(self, name):
        presence = getattr(self._dm, "payload_presence", None)
        if presence is None or name not in presence.columns:
            return pd.Series(True, index=self.df.index, dtype=bool)
        field_presence = presence[name]
        if not field_presence.index.equals(self.df.index):
            field_presence = field_presence.reindex(self.df.index, fill_value=True)
        return field_presence.fillna(False).astype(bool)

    def _series_is_null(self, series, name=None):
        null_mask = self._safe_apply(series, self._is_na_like).astype(bool)
        if name is None:
            return null_mask
        return self._field_presence(name) & null_mask

    def _series_not_null(self, series, name=None):
        return ~self._series_is_null(series, name)

    def _series_is_empty(self, series, name=None):
        # 对齐 Qdrant IsEmpty：missing/null/[]。在 pandas 中 missing 映射为 None/NaN。
        empty_mask = self._safe_apply(series, lambda x: self._is_na_like(x) or (isinstance(x, list) and len(x) == 0)).astype(bool)
        if name is None:
            return empty_mask
        return (~self._field_presence(name)) | empty_mask

    def _series_not_empty(self, series, name=None):
        return ~self._series_is_empty(series, name)

    def _uses_empty_semantics(self, name):
        """演进字段默认无法区分 missing 与显式 null，统一按 IsEmpty 语义处理。"""
        return self._is_evolved_field(name) and evolved_field_use_empty_semantics()

    def _series_qdrant_nullish(self, name, series):
        if self._uses_empty_semantics(name):
            return self._series_is_empty(series, name)
        return self._series_is_null(series, name)

    def _series_qdrant_not_nullish(self, name, series):
        if self._uses_empty_semantics(name):
            return self._series_not_empty(series, name)
        return self._series_not_null(series, name)

    def _series_values_count(self, name, series, cmp_op, threshold):
        present_mask = self._field_presence(name)

        def _count_match(idx, value):
            if not bool(present_mask.loc[idx]):
                return False
            if isinstance(value, list):
                cnt = len(value)
            elif self._is_na_like(value):
                cnt = 0
            else:
                cnt = 1
            if cmp_op == "gt":
                return cnt > threshold
            if cmp_op == "gte":
                return cnt >= threshold
            if cmp_op == "lt":
                return cnt < threshold
            return cnt <= threshold

        return pd.Series(
            [_count_match(idx, value) for idx, value in series.astype(object).items()],
            index=series.index,
            dtype=bool,
        )

    def _nullish_expr(self, name):
        return f"{name} is empty" if self._uses_empty_semantics(name) else f"{name} is null"

    def _not_nullish_expr(self, name):
        return f"{name} is not empty" if self._uses_empty_semantics(name) else f"{name} is not null"

    def _series_match_except(self, name, series, excluded_set, normalizer):
        """
        对齐 Qdrant MatchExcept:
        - 真实 Rust server 语义下，null / missing 均不会命中 MatchExcept。
        - 因此 pandas 侧只保留“可归一化后的非空值且不在 except 列表中”的行。
        """
        def _match(x, _excluded=excluded_set):
            if self._is_na_like(x):
                return False
            try:
                normalized = normalizer(x)
            except Exception:
                return False
            return normalized is not None and normalized not in _excluded

        return self._safe_apply(series, _match)

    def _series_match_except_array(self, series, excluded_set, normalizer):
        """
        对齐 Qdrant MatchExcept 的数组语义：
        - null / missing 不命中；
        - 数组中只要存在一个元素不在 except 列表中，即整体命中；
        - 空数组不命中。
        """
        def _match(x, _excluded=excluded_set):
            if self._is_na_like(x) or not isinstance(x, list):
                return False
            for item in x:
                try:
                    normalized = normalizer(item)
                except Exception:
                    continue
                if normalized is not None and normalized not in _excluded:
                    return True
            return False

        return self._safe_apply(series, _match)

    def _condition_filter(self, condition):
        return Filter(must=[condition])

    def _null_filter(self, key):
        return self._condition_filter(self._is_null_condition(key))

    def _empty_filter(self, key):
        return self._condition_filter(self._is_empty_condition(key))

    def _qdrant_is_empty_filter(self, key):
        return Filter(must=[self._is_empty_condition(key)])

    def _qdrant_not_empty_filter(self, key):
        return Filter(must_not=[self._is_empty_condition(key)])

    def _qdrant_nullish_filter(self, key):
        """nullish 语义：普通字段用 IsNull；演进字段用 IsEmpty（覆盖缺失）。"""
        if self._is_evolved_field(key) and evolved_field_use_empty_semantics():
            return self._qdrant_is_empty_filter(key)
        return self._qdrant_is_null_filter(key)

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

    def _get_json_projected_values(self, obj, prefix_keys, array_key, leaf_key, flatten_lists=False):
        """获取类似 country.cities[].population / country.cities[].sightseeing 的投影值。"""
        arr = self._get_json_val(obj, list(prefix_keys) + [array_key])
        if not isinstance(arr, list):
            return []
        out = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            val = item.get(leaf_key)
            if val is None:
                continue
            if flatten_lists and isinstance(val, list):
                out.extend(v for v in val if v is not None)
            else:
                out.append(val)
        return out

    def _iter_array_object_items(self, obj):
        if not isinstance(obj, list):
            return []
        return [item for item in obj if isinstance(item, dict)]

    def _array_object_leaf_values(self, obj, leaf_key):
        out = []
        for item in self._iter_array_object_items(obj):
            value = item.get(leaf_key)
            if value is not None:
                out.append(value)
        return out

    def _array_object_projected_path(self, name, leaf_key):
        return f"{name}[].{leaf_key}"

    def gen_array_object_projected_expr(self, name, series):
        """
        生成 ARRAY_OBJECT 的 projected-path 查询，例如 items[].score / items[].label。
        多个顶层字段条件在 Qdrant 中各自对投影值求值，因此允许 sibling 元素跨元素累积满足。
        """
        candidate_rows = []
        score_values = []
        label_values = []
        active_values = []
        cross_score_label_pairs = []

        for raw in series.dropna():
            items = self._iter_array_object_items(raw)
            if not items:
                continue
            candidate_rows.append(items)
            row_pairs = {
                (int(item["score"]), str(item["label"]))
                for item in items
                if item.get("score") is not None and item.get("label") is not None
            }
            for item in items:
                if item.get("score") is not None:
                    score_values.append(int(item["score"]))
                if item.get("label") is not None:
                    label_values.append(str(item["label"]))
                if item.get("active") is not None:
                    active_values.append(bool(item["active"]))
            if len(items) >= 2:
                for left in items:
                    left_score = left.get("score")
                    if left_score is None:
                        continue
                    for right in items:
                        if right is left:
                            continue
                        right_label = right.get("label")
                        if right_label is None:
                            continue
                        pair = (int(left_score), str(right_label))
                        if pair not in row_pairs:
                            cross_score_label_pairs.append(pair)

        if not candidate_rows:
            return self._qdrant_is_empty_filter(name), self._series_is_empty(series, name), f"{name} is empty"

        score_list = sorted(dict.fromkeys(score_values))
        label_list = sorted(dict.fromkeys(label_values))
        active_list = sorted(dict.fromkeys(active_values))
        cross_score_label_pairs = list(dict.fromkeys(cross_score_label_pairs))

        strategies = []
        if score_list:
            strategies.extend(["score_exact", "score_range"])
        if label_list:
            strategies.append("label_exact")
        if active_list:
            strategies.append("active_exact")
        if cross_score_label_pairs:
            strategies.append("cross_element_score_label_and")

        if not strategies:
            return self._qdrant_not_empty_filter(name), self._series_not_empty(series, name), f"{name} is not empty"

        strategy = random.choice(strategies)

        if strategy == "score_exact":
            score_val = random.choice(score_list)
            path = self._array_object_projected_path(name, "score")
            filter_obj = FieldCondition(key=path, match=MatchValue(value=score_val))
            mask = self._safe_apply(
                series,
                lambda x, target=score_val: any(v == target for v in self._array_object_leaf_values(x, "score")),
            )
            return filter_obj, mask, f"{path} == {score_val} (projected)"

        if strategy == "score_range":
            score_val = random.choice(score_list)
            cmp_op = random.choice(["gt", "lt", "gte", "lte"])
            path = self._array_object_projected_path(name, "score")
            filter_obj = FieldCondition(key=path, range=Range(**{cmp_op: score_val}))

            def _score_matches(values, op=cmp_op, target=score_val):
                for value in values:
                    if op == "gt" and value > target:
                        return True
                    if op == "lt" and value < target:
                        return True
                    if op == "gte" and value >= target:
                        return True
                    if op == "lte" and value <= target:
                        return True
                return False

            mask = self._safe_apply(
                series,
                lambda x: _score_matches(self._array_object_leaf_values(x, "score")),
            )
            op_symbol = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}[cmp_op]
            return filter_obj, mask, f"{path} {op_symbol} {score_val} (projected)"

        if strategy == "label_exact":
            label_val = random.choice(label_list)
            path = self._array_object_projected_path(name, "label")
            filter_obj = FieldCondition(key=path, match=MatchValue(value=label_val))
            mask = self._safe_apply(
                series,
                lambda x, target=label_val: any(v == target for v in self._array_object_leaf_values(x, "label")),
            )
            return filter_obj, mask, f'{path} == "{label_val}" (projected)'

        if strategy == "active_exact":
            active_val = random.choice(active_list)
            path = self._array_object_projected_path(name, "active")
            filter_obj = FieldCondition(key=path, match=MatchValue(value=active_val))
            mask = self._safe_apply(
                series,
                lambda x, target=active_val: any(v == target for v in self._array_object_leaf_values(x, "active")),
            )
            return filter_obj, mask, f"{path} == {active_val} (projected)"

        score_val, label_val = random.choice(cross_score_label_pairs)
        score_path = self._array_object_projected_path(name, "score")
        label_path = self._array_object_projected_path(name, "label")
        filter_obj = Filter(
            must=[
                FieldCondition(key=score_path, match=MatchValue(value=score_val)),
                FieldCondition(key=label_path, match=MatchValue(value=label_val)),
            ]
        )
        mask = self._safe_apply(
            series,
            lambda x, score_target=score_val, label_target=label_val: (
                any(v == score_target for v in self._array_object_leaf_values(x, "score"))
                and any(v == label_target for v in self._array_object_leaf_values(x, "label"))
            ),
        )
        expr = (
            f"{score_path} == {score_val} AND {label_path} == \"{label_val}\" "
            f"(projected cross-element)"
        )
        return filter_obj, mask, expr

    def get_value_for_query(self, fname, ftype):
        """获取用于查询的值"""
        # 在查询参数层注入边界值，提升各模式边界覆盖率
        if random.random() < self._query_boundary_rate:
            boundary_val = self._dm.sample_boundary_value(ftype)
            if boundary_val is not None:
                if ftype == FieldType.FLOAT:
                    return float(np.float32(boundary_val))
                if ftype == FieldType.ARRAY_INT and isinstance(boundary_val, list):
                    return boundary_val[0] if boundary_val else 0
                if ftype == FieldType.ARRAY_STR and isinstance(boundary_val, list):
                    return boundary_val[0] if boundary_val else ""
                if ftype == FieldType.ARRAY_FLOAT and isinstance(boundary_val, list):
                    return float(np.float32(boundary_val[0])) if boundary_val else 0.0
                return boundary_val

        valid_series = self.df[fname].dropna()
        if ftype == FieldType.INT and not valid_series.empty and not INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES:
            valid_series = valid_series[
                valid_series.apply(lambda x: not is_known_unstable_int64_value(self._convert_to_native(x)))
            ]
        if not valid_series.empty and random.random() < 0.8:
            val = random.choice(valid_series.values)
            if hasattr(val, "item"): val = val.item()
            return val

        # 生成不存在的值
        if ftype == FieldType.INT:
            if not valid_series.empty:
                min_val = self._clip_int64(valid_series.min())
                max_val = self._clip_int64(valid_series.max())
                return random.choice([
                    self._offset_int64(max_val, 100000),
                    self._offset_int64(min_val, -100000),
                ])
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

        elif ftype == FieldType.UUID:
            return self._random_uuid_string()

        elif ftype == FieldType.JSON:
            new_key = self._random_string(10, 15)
            new_val = self._random_string(10, 15)
            return {"_non_exist_key": new_key, "_non_exist_val": new_val}

        elif ftype in [FieldType.ARRAY_INT, FieldType.ARRAY_STR]:
            return random.randint(50000, 100000)

        elif ftype == FieldType.ARRAY_FLOAT:
            return round(random.random() * 2000, 2)

        elif ftype == FieldType.DATETIME:
            if not valid_series.empty:
                valid_strings = [self._convert_to_native(x) for x in valid_series.values if self._normalize_datetime_scalar(self._convert_to_native(x)) is not None]
                if valid_strings:
                    return random.choice(valid_strings)
            return random.choice([
                "2019-01-01T00:00:00Z",
                "2030-01-01T00:00:00Z",
            ])

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
        not_nullish_mask = self._series_qdrant_not_nullish(name, series)
        nullish_mask = self._series_qdrant_nullish(name, series)
        nullish_expr = self._nullish_expr(name)
        not_nullish_expr = self._not_nullish_expr(name)

        # 1. Null/Empty Check
        if random.random() < 0.15:
            # 演化字段默认走 IsEmpty（覆盖 missing/null/[]）；若开启显式 NULL 同步则走严格 IsNull。
            if self._is_evolved_field(name):
                if evolved_field_use_empty_semantics():
                    if random.random() < 0.5:
                        return (self._qdrant_is_empty_filter(name), self._series_is_empty(series, name), f"{name} is empty")
                    return (self._qdrant_not_empty_filter(name), self._series_not_empty(series, name), f"{name} is not empty")
                if random.random() < 0.5:
                    return (self._qdrant_is_null_filter(name), self._series_is_null(series, name), f"{name} is null")
                return (self._qdrant_not_null_filter(name), self._series_not_null(series, name), f"{name} is not null")

            if ftype in [FieldType.ARRAY_INT, FieldType.ARRAY_STR, FieldType.ARRAY_FLOAT, FieldType.ARRAY_OBJECT]:
                op = random.choice(["is_null", "is_not_null", "is_empty", "is_not_empty"])
                if op == "is_null":
                    return (self._qdrant_is_null_filter(name), self._series_is_null(series, name), f"{name} is null")
                if op == "is_not_null":
                    return (self._qdrant_not_null_filter(name), self._series_not_null(series, name), f"{name} is not null")
                if op == "is_empty":
                    return (self._qdrant_is_empty_filter(name), self._series_is_empty(series, name), f"{name} is empty")
                return (self._qdrant_not_empty_filter(name), self._series_not_empty(series, name), f"{name} is not empty")

            if random.random() < 0.5:
                return (self._qdrant_is_null_filter(name), self._series_is_null(series, name), f"{name} is null")
            return (self._qdrant_not_null_filter(name), self._series_not_null(series, name), f"{name} is not null")

        val = self.get_value_for_query(name, ftype)
        if val is None:
            if self._is_evolved_field(name):
                if evolved_field_use_empty_semantics():
                    return (self._qdrant_is_empty_filter(name), self._series_is_empty(series, name), f"{name} is empty")
                return (self._qdrant_is_null_filter(name), self._series_is_null(series, name), f"{name} is null")
            return (self._qdrant_is_null_filter(name), self._series_is_null(series, name), f"{name} is null")

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
            mask = self._safe_apply(series, safe_compare_scalar("==", val_bool))
            expr_str = f"{name} == {val_bool}"

        elif ftype == FieldType.INT:
            val_int = self._clip_int64(val)
            op = random.choice([">", "<", "==", "!=", ">=", "<=", "in", "not_in"])
            
            if op == "==":
                filter_cond = FieldCondition(key=name, match=MatchValue(value=val_int))
                mask = self._safe_apply(series, lambda x, t=val_int: self._normalize_int_scalar(x) == t)
                expr_str = f"{name} == {val_int}"
            elif op == "!=":
                filter_cond = FieldCondition(key=name, match=MatchExcept(**{"except": [val_int]}))
                mask = self._series_match_except(name, series, {val_int}, self._normalize_int_scalar)
                expr_str = f"{name} != {val_int}"
            elif op == ">":
                filter_cond = FieldCondition(key=name, range=Range(gt=val_int))
                mask = self._safe_apply(series, 
                    lambda x, t=val_int: (
                        self._normalize_int_scalar(x) is not None and self._normalize_int_scalar(x) > t
                    )
                )
                expr_str = f"{name} > {val_int}"
            elif op == "<":
                filter_cond = FieldCondition(key=name, range=Range(lt=val_int))
                mask = self._safe_apply(series, 
                    lambda x, t=val_int: (
                        self._normalize_int_scalar(x) is not None and self._normalize_int_scalar(x) < t
                    )
                )
                expr_str = f"{name} < {val_int}"
            elif op == ">=":
                filter_cond = FieldCondition(key=name, range=Range(gte=val_int))
                mask = self._safe_apply(series, 
                    lambda x, t=val_int: (
                        self._normalize_int_scalar(x) is not None and self._normalize_int_scalar(x) >= t
                    )
                )
                expr_str = f"{name} >= {val_int}"
            elif op == "<=":
                filter_cond = FieldCondition(key=name, range=Range(lte=val_int))
                mask = self._safe_apply(series, 
                    lambda x, t=val_int: (
                        self._normalize_int_scalar(x) is not None and self._normalize_int_scalar(x) <= t
                    )
                )
                expr_str = f"{name} <= {val_int}"
            elif op == "in":
                # IN 列表查询：从实际数据中采样 2-6 个值
                valid_vals = []
                for x in self.df[name].dropna().unique()[:50]:
                    nx = self._normalize_int_scalar(self._convert_to_native(x))
                    if nx is not None and (INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES or not is_known_unstable_int64_value(nx)):
                        valid_vals.append(nx)
                valid_vals = list(dict.fromkeys(valid_vals))
                sample_size = min(random.randint(2, 6), len(valid_vals))
                in_vals = random.sample(valid_vals, sample_size) if valid_vals else [val_int]
                if val_int not in in_vals:
                    in_vals[0] = val_int  # 确保至少包含查询基准值
                filter_cond = FieldCondition(key=name, match=MatchAny(any=in_vals))
                in_set = set(in_vals)
                mask = self._safe_apply(series, 
                    lambda x, s=in_set: (
                        self._normalize_int_scalar(x) is not None and self._normalize_int_scalar(x) in s
                    )
                )
                expr_str = f"{name} in {in_vals}"
            else:  # not_in
                # NOT IN 排除查询
                valid_vals = []
                for x in self.df[name].dropna().unique()[:50]:
                    nx = self._normalize_int_scalar(self._convert_to_native(x))
                    if nx is not None and (INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES or not is_known_unstable_int64_value(nx)):
                        valid_vals.append(nx)
                valid_vals = list(dict.fromkeys(valid_vals))
                sample_size = min(random.randint(2, 6), len(valid_vals))
                excl_vals = random.sample(valid_vals, sample_size) if valid_vals else [val_int]
                filter_cond = FieldCondition(key=name, match=MatchExcept(**{"except": excl_vals}))
                excl_set = set(excl_vals)
                mask = self._series_match_except(name, series, excl_set, self._normalize_int_scalar)
                expr_str = f"{name} not in {excl_vals}"

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
            
            mask = self._safe_apply(series, safe_compare_scalar(op, val_float))
            expr_str = f"{name} {op} {val_float}"

        elif ftype == FieldType.STRING:
            str_ops = ["==", "!=", "in", "not_in"]
            non_empty_strings = [str(x) for x in self.df[name].dropna().tolist() if isinstance(x, str) and x]
            text_any_sources = [x for x in non_empty_strings if any(part for part in x.split())]
            if hasattr(models, "MatchText") and non_empty_strings:
                str_ops.append("text")
            if hasattr(models, "MatchTextAny") and text_any_sources:
                str_ops.append("text_any")
            if hasattr(models, "MatchPhrase") and non_empty_strings:
                str_ops.append("phrase")
            op = random.choice(str_ops)
            
            if op == "==":
                filter_cond = FieldCondition(key=name, match=MatchValue(value=val))
                mask = self._safe_apply(series, safe_compare_scalar("==", val))
                expr_str = f'{name} == "{val}"'
            elif op == "!=":
                filter_cond = FieldCondition(key=name, match=MatchExcept(**{"except": [val]}))
                mask = self._series_match_except(
                    name,
                    series,
                    {val},
                    lambda x: x if isinstance(x, str) else None,
                )
                expr_str = f'{name} != "{val}"'
            elif op == "phrase":
                phrase_source = random.choice(non_empty_strings)
                if len(phrase_source) >= 4:
                    start = random.randint(0, max(0, len(phrase_source) - 3))
                    phrase = phrase_source[start:start + random.randint(2, min(6, len(phrase_source) - start))]
                else:
                    phrase = phrase_source
                filter_cond = FieldCondition(key=name, match=models.MatchPhrase(phrase=phrase))
                # 无全文索引时，Qdrant 退化为 substring 匹配
                mask = self._safe_apply(series, lambda x, p=phrase: isinstance(x, str) and p in x)
                expr_str = f'{name} phrase "{phrase}"'
            elif op == "text":
                text_source = random.choice(non_empty_strings)
                if len(text_source) >= 4:
                    start = random.randint(0, max(0, len(text_source) - 3))
                    query_text = text_source[start:start + random.randint(2, min(8, len(text_source) - start))]
                else:
                    query_text = text_source
                filter_cond = FieldCondition(key=name, match=models.MatchText(text=query_text))
                # 当前 fuzzer 对 STRING 只创建 KEYWORD 索引，不创建 full-text index，
                # 因而 text 条件按官方文档的“无 full-text index -> exact substring match”子集建模。
                mask = self._safe_apply(series, lambda x, q=query_text: isinstance(x, str) and q in x)
                expr_str = f'{name} text "{query_text}"'
            elif op == "text_any":
                source_terms = [part for part in random.choice(text_any_sources).split() if part]
                term_source = random.choice(source_terms)
                if len(term_source) >= 4:
                    start = random.randint(0, max(0, len(term_source) - 3))
                    real_term = term_source[start:start + random.randint(2, min(8, len(term_source) - start))]
                else:
                    real_term = term_source
                terms = [real_term, random.choice(["unlikelyterm", "absenttoken", "zzznomatch"])]
                random.shuffle(terms)
                query_text = " ".join(terms)
                filter_cond = FieldCondition(key=name, match=models.MatchTextAny(text_any=query_text))
                # 当前无 full-text index 子集下，本地 v1.17.0 表现为任一 query term 子串命中。
                mask = self._safe_apply(series, lambda x, ts=terms: isinstance(x, str) and any(t in x for t in ts))
                expr_str = f'{name} text_any "{query_text}"'
            elif op == "in":
                # 生成多个值的列表
                valid_vals = [str(x) for x in self.df[name].dropna().unique()[:5].tolist()]
                if val not in valid_vals:
                    valid_vals.append(val)
                filter_cond = FieldCondition(key=name, match=MatchAny(any=valid_vals))
                mask = self._safe_apply(series, lambda x: x in valid_vals if x is not None else False)
                expr_str = f'{name} in {valid_vals}'
            else:  # not_in
                valid_vals = [str(x) for x in self.df[name].dropna().unique()[:20].tolist()]
                valid_vals = list(dict.fromkeys(valid_vals))
                sample_size = min(random.randint(2, 6), len(valid_vals))
                excl_vals = random.sample(valid_vals, sample_size) if valid_vals else [str(val)]
                filter_cond = FieldCondition(key=name, match=MatchExcept(**{"except": excl_vals}))
                excl_set = set(excl_vals)
                mask = self._series_match_except(
                    name,
                    series,
                    excl_set,
                    lambda x: x if isinstance(x, str) else None,
                )
                expr_str = f'{name} not in {excl_vals}'

        elif ftype == FieldType.UUID:
            val_uuid = str(uuid.UUID(str(val)))
            filter_cond = FieldCondition(key=name, match=MatchValue(value=val_uuid))
            mask = self._safe_apply(
                series,
                lambda x, t=val_uuid: self._dm._normalize_scalar_for_field(x, FieldType.UUID) == t
            )
            expr_str = f'{name} uuid == "{val_uuid}"'

        elif ftype == FieldType.JSON:
            # JSON 嵌套查询
            return self.gen_json_expr(name, series, val)

        elif ftype in [FieldType.ARRAY_INT, FieldType.ARRAY_STR]:
            # 数组包含 / values_count 查询
            valid_series = self.df[name].dropna()
            all_items = []
            for arr in valid_series:
                if isinstance(arr, list):
                    all_items.extend(arr)
            unique_items = list(dict.fromkeys(
                self._convert_to_native(item) for item in all_items if item is not None
            ))

            array_ops = ["contains", "not_contains", "not_empty"]
            # Qdrant 对缺失字段不参与 values_count，而 pandas 中演化字段的 missing/null 不易严格区分，
            # 为避免 oracle 误报，演化字段不生成 values_count 谓词。
            if not self._is_evolved_field(name):
                array_ops.append("values_count")
            array_op = random.choice(array_ops)

            if array_op == "values_count":
                cmp_op = random.choice(["gt", "gte", "lt", "lte"])
                threshold = random.randint(0, 4)
                filter_cond = FieldCondition(
                    key=name,
                    values_count=models.ValuesCount(**{cmp_op: threshold})
                )

                op_map = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
                mask = self._series_values_count(name, series, cmp_op, threshold)
                expr_str = f"{name} values_count {op_map[cmp_op]} {threshold}"
            elif array_op == "contains" and unique_items:
                target = random.choice(unique_items)
                if len(unique_items) >= 2:
                    sample_size = min(random.randint(2, 4), len(unique_items))
                else:
                    sample_size = 1
                candidates = random.sample(unique_items, sample_size) if sample_size < len(unique_items) else list(unique_items)
                if target not in candidates:
                    candidates[0] = target
                candidates = list(dict.fromkeys(candidates))
                candidate_set = set(candidates)
                filter_cond = FieldCondition(key=name, match=MatchAny(any=candidates))
                mask = self._safe_apply(
                    series,
                    lambda x, s=candidate_set: any(v in s for v in x) if isinstance(x, list) else False
                )
                expr_str = f'{name} contains any of {candidates}'
            elif array_op == "not_contains" and unique_items:
                if len(unique_items) >= 2:
                    sample_size = min(random.randint(2, 4), len(unique_items))
                else:
                    sample_size = 1
                excl_vals = random.sample(unique_items, sample_size) if sample_size < len(unique_items) else list(unique_items)
                excl_vals = list(dict.fromkeys(excl_vals))
                excl_set = set(excl_vals)
                normalizer = self._normalize_int_scalar if ftype == FieldType.ARRAY_INT else (lambda x: x if isinstance(x, str) else None)
                filter_cond = FieldCondition(key=name, match=MatchExcept(**{"except": excl_vals}))
                mask = self._series_match_except_array(series, excl_set, normalizer)
                expr_str = f'{name} has element not in {excl_vals}'
            else:
                filter_cond = self._qdrant_not_empty_filter(name)
                mask = self._series_not_empty(series, name)
                expr_str = f'{name} is not empty'

        elif ftype == FieldType.ARRAY_FLOAT:
            # 浮点数组 - 元素范围 / values_count
            valid_series = self.df[name].dropna()
            all_items = []
            for arr in valid_series:
                if isinstance(arr, list):
                    all_items.extend(arr)
            array_ops = ["element_range", "not_empty"]
            if not self._is_evolved_field(name):
                array_ops.append("values_count")
            array_op = random.choice(array_ops)
            if array_op == "values_count":
                cmp_op = random.choice(["gt", "gte", "lt", "lte"])
                threshold = random.randint(0, 4)
                filter_cond = FieldCondition(
                    key=name,
                    values_count=models.ValuesCount(**{cmp_op: threshold})
                )

                op_map = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
                mask = self._series_values_count(name, series, cmp_op, threshold)
                expr_str = f"{name} values_count {op_map[cmp_op]} {threshold}"
            elif array_op == "element_range" and all_items:
                target = random.choice(all_items)
                target_f = float(target)
                epsilon = 0.5  # 较大的 epsilon 避免浮点精度问题
                filter_cond = FieldCondition(key=name, range=Range(gte=target_f - epsilon, lte=target_f + epsilon))
                mask = self._safe_apply(series, lambda x, t=target_f, e=epsilon:
                    any(t - e <= float(v) <= t + e for v in x) if isinstance(x, list) and x else False)
                expr_str = f'{name} has element ~= {target_f}'
            else:
                filter_cond = self._qdrant_not_empty_filter(name)
                mask = self._series_not_empty(series, name)
                expr_str = f'{name} is not empty'

        elif ftype == FieldType.DATETIME:
            target_ts = self._normalize_datetime_scalar(val)
            if target_ts is None:
                return (self._qdrant_is_null_filter(name), self._series_is_null(series, name), f"{name} is null")
            val_dt = self._format_datetime_utc(target_ts)
            op = random.choice([">", "<", ">=", "<="])
            if op == ">":
                filter_cond = FieldCondition(key=name, range=DatetimeRange(gt=val_dt))
                mask = self._safe_apply(series, 
                    lambda x, t=target_ts: (
                        self._normalize_datetime_scalar(x) is not None and self._normalize_datetime_scalar(x) > t
                    )
                )
                expr_str = f'{name} > "{val_dt}"'
            elif op == "<":
                filter_cond = FieldCondition(key=name, range=DatetimeRange(lt=val_dt))
                mask = self._safe_apply(series, 
                    lambda x, t=target_ts: (
                        self._normalize_datetime_scalar(x) is not None and self._normalize_datetime_scalar(x) < t
                    )
                )
                expr_str = f'{name} < "{val_dt}"'
            elif op == ">=":
                filter_cond = FieldCondition(key=name, range=DatetimeRange(gte=val_dt))
                mask = self._safe_apply(series, 
                    lambda x, t=target_ts: (
                        self._normalize_datetime_scalar(x) is not None and self._normalize_datetime_scalar(x) >= t
                    )
                )
                expr_str = f'{name} >= "{val_dt}"'
            elif op == "<=":
                filter_cond = FieldCondition(key=name, range=DatetimeRange(lte=val_dt))
                mask = self._safe_apply(series, 
                    lambda x, t=target_ts: (
                        self._normalize_datetime_scalar(x) is not None and self._normalize_datetime_scalar(x) <= t
                    )
                )
                expr_str = f'{name} <= "{val_dt}"'

        elif ftype == FieldType.GEO:
            if isinstance(val, dict) and "lat" in val and "lon" in val:
                center_lat = val["lat"]
                center_lon = val["lon"]
            else:
                center_lat = random.uniform(-80, 80)
                center_lon = random.uniform(-170, 170)

            geo_strategy = random.choice(["bbox", "radius"])

            if geo_strategy == "bbox":
                # GeoBoundingBox 查询
                delta_lat = random.uniform(1, 30)
                delta_lon = random.uniform(1, 30)
                top_lat = min(center_lat + delta_lat, 90)
                bottom_lat = max(center_lat - delta_lat, -90)
                left_lon = max(center_lon - delta_lon, -180)
                right_lon = min(center_lon + delta_lon, 180)
                pivot_on_bbox_edge = (
                    math.isclose(center_lat, top_lat)
                    or math.isclose(center_lat, bottom_lat)
                    or math.isclose(center_lon, left_lon)
                    or math.isclose(center_lon, right_lon)
                )
                bbox_clamped_to_global_edge = (
                    math.isclose(top_lat, 90.0)
                    or math.isclose(bottom_lat, -90.0)
                    or math.isclose(left_lon, -180.0)
                    or math.isclose(right_lon, 180.0)
                )
                if pivot_on_bbox_edge or bbox_clamped_to_global_edge:
                    geo_strategy = "radius"
                else:
                    filter_cond = FieldCondition(
                        key=name,
                        geo_bounding_box=GeoBoundingBox(
                            top_left=GeoPoint(lat=top_lat, lon=left_lon),
                            bottom_right=GeoPoint(lat=bottom_lat, lon=right_lon)
                        )
                    )
                    mask = self._safe_apply(series, lambda x: (
                        x is not None and isinstance(x, dict)
                        and "lat" in x and "lon" in x
                        and bottom_lat <= x["lat"] <= top_lat
                        and left_lon <= x["lon"] <= right_lon
                    ))
                    expr_str = f"{name} in bbox({bottom_lat:.2f},{left_lon:.2f} -> {top_lat:.2f},{right_lon:.2f})"
            if geo_strategy == "radius":
                # GeoRadius 查询 - Qdrant 独有的圆形地理范围查询
                radius_m = random.uniform(100_000, 5_000_000)  # 100km ~ 5000km
                filter_cond = FieldCondition(
                    key=name,
                    geo_radius=GeoRadius(
                        center=GeoPoint(lat=center_lat, lon=center_lon),
                        radius=radius_m
                    )
                )
                # Haversine 公式计算地球距离
                def _haversine_check(x, clat=center_lat, clon=center_lon, r=radius_m):
                    if x is None or not isinstance(x, dict) or "lat" not in x or "lon" not in x:
                        return False
                    lat1, lon1 = math.radians(clat), math.radians(clon)
                    lat2, lon2 = math.radians(x["lat"]), math.radians(x["lon"])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                    c = 2 * math.asin(min(1.0, math.sqrt(a)))
                    dist = 6_371_000 * c  # 地球平均半径(m)
                    return dist < r
                mask = self._safe_apply(series, _haversine_check)
                expr_str = f"{name} in radius({center_lat:.2f},{center_lon:.2f}, r={radius_m:.0f}m)"

        elif ftype == FieldType.ARRAY_OBJECT:
            # ARRAY_OBJECT 同时覆盖 nested 同元素语义和 items[].field 投影路径语义。
            if random.random() < 0.6:
                return self.gen_nested_expr(name, series)
            return self.gen_array_object_projected_expr(name, series)

        if filter_cond is not None and mask is not None:
            # 由各分支自行定义 null 语义，这里不再统一追加 notnull 截断。
            return (filter_cond, mask, expr_str)

        # 默认返回：普通字段 is not null；演化字段默认 is not empty。
        filter_cond = self._qdrant_not_null_filter(name)
        return (filter_cond, self._series_qdrant_not_nullish(name, series), self._not_nullish_expr(name))

    def gen_json_expr(self, name, series, val):
        """生成 JSON 嵌套字段查询"""
        # CHAOS_RATE: 按概率注入类型混淆（故意用错误类型查询）
        if CHAOS_RATE > 0 and random.random() < CHAOS_RATE:
            chaos_type = random.choice(["str_as_int", "int_as_str", "bool_as_int"])
            if chaos_type == "str_as_int":
                # 对 color(string) 用 int 查询
                fake_val = random.randint(0, 1000)
                filter_cond = FieldCondition(key=f"{name}.color", match=MatchValue(value=fake_val))
                mask = pd.Series(False, index=self.df.index)  # 不可能匹配
                return (filter_cond, mask, f'CHAOS: {name}.color == {fake_val} (type mismatch)')
            elif chaos_type == "int_as_str":
                # 对 price(int) 用 string 查询
                fake_str = self._random_string(3, 8)
                filter_cond = FieldCondition(key=f"{name}.price", match=MatchValue(value=fake_str))
                mask = pd.Series(False, index=self.df.index)
                return (filter_cond, mask, f'CHAOS: {name}.price == "{fake_str}" (type mismatch)')
            else:  # bool_as_int
                # 对 active(bool) 用 int 查询
                fake_int = random.randint(0, 1)
                filter_cond = FieldCondition(key=f"{name}.active", match=MatchValue(value=fake_int))
                mask = pd.Series(False, index=self.df.index)
                return (filter_cond, mask, f'CHAOS: {name}.active == {fake_int} (type mismatch)')

        strategy = random.choice([
            "range",
            "config_version",
            "multi_key",
            "country_name",
            "city_name",
            "city_population",
            "city_sightseeing",
        ])

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

            return (filter_cond, self._safe_apply(series, check_range), f'{name}.price > {low} and < {high}')

        elif strategy == "config_version":
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

            return (filter_cond, self._safe_apply(series, check_nested), f'{name}.config.version == {val}')

        elif strategy == "country_name":
            valid_vals = []
            for x in series.dropna():
                v = self._get_json_val(x, ["country", "name"])
                if isinstance(v, str):
                    valid_vals.append(v)
            country_val = random.choice(valid_vals) if valid_vals else "Germany"
            filter_cond = FieldCondition(
                key=f"{name}.country.name",
                match=MatchValue(value=country_val)
            )

            def check_country_name(x, _v=country_val):
                try:
                    return self._get_json_val(x, ["country", "name"]) == _v
                except:
                    return False

            return (
                filter_cond,
                self._safe_apply(series, check_country_name),
                f'{name}.country.name == "{country_val}"'
            )

        elif strategy == "city_name":
            valid_vals = []
            for x in series.dropna():
                valid_vals.extend(
                    str(v) for v in self._get_json_projected_values(
                        x, ["country"], "cities", "name"
                    ) if isinstance(v, str)
                )
            city_val = random.choice(valid_vals) if valid_vals else "Tokyo"
            filter_cond = FieldCondition(
                key=f"{name}.country.cities[].name",
                match=MatchValue(value=city_val)
            )

            def check_city_name(x, _v=city_val):
                try:
                    vals = self._get_json_projected_values(x, ["country"], "cities", "name")
                    return any(v == _v for v in vals)
                except:
                    return False

            return (
                filter_cond,
                self._safe_apply(series, check_city_name),
                f'{name}.country.cities[].name == "{city_val}"'
            )

        elif strategy == "city_population":
            valid_vals = []
            for x in series.dropna():
                vals = self._get_json_projected_values(x, ["country"], "cities", "population")
                for v in vals:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        valid_vals.append(float(v))
            threshold = random.choice(valid_vals) if valid_vals else 9.0
            filter_cond = FieldCondition(
                key=f"{name}.country.cities[].population",
                range=Range(gte=threshold)
            )

            def check_city_population(x, _threshold=threshold):
                try:
                    vals = self._get_json_projected_values(x, ["country"], "cities", "population")
                    return any(
                        isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) >= _threshold
                        for v in vals
                    )
                except:
                    return False

            return (
                filter_cond,
                self._safe_apply(series, check_city_population),
                f"{name}.country.cities[].population >= {threshold}"
            )

        elif strategy == "city_sightseeing":
            valid_vals = []
            for x in series.dropna():
                valid_vals.extend(
                    str(v) for v in self._get_json_projected_values(
                        x, ["country"], "cities", "sightseeing", flatten_lists=True
                    ) if isinstance(v, str)
                )
            sight_val = random.choice(valid_vals) if valid_vals else "Osaka Castle"
            filter_cond = FieldCondition(
                key=f"{name}.country.cities[].sightseeing",
                match=MatchValue(value=sight_val)
            )

            def check_city_sightseeing(x, _v=sight_val):
                try:
                    vals = self._get_json_projected_values(
                        x, ["country"], "cities", "sightseeing", flatten_lists=True
                    )
                    return any(v == _v for v in vals)
                except:
                    return False

            return (
                filter_cond,
                self._safe_apply(series, check_city_sightseeing),
                f'{name}.country.cities[].sightseeing == "{sight_val}"'
            )

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

            return (filter_cond, self._safe_apply(series, check_multi), f'{name}.active == true and {name}.color == "{color}"')

    # ------------------------------------------------------------------ #
    #  Nested Object Filter 查询生成器                                    #
    # ------------------------------------------------------------------ #
    def gen_nested_expr(self, name, series):
        """
        生成 NestedCondition 查询。
        Qdrant nested 语义：payload 中对象数组至少有一个元素同时满足所有条件。
        返回: (Filter, pandas_mask, expr_str)
        """
        # 从 DataFrame 中收集实际存在的值，用于构造有意义的查询；排序避免 set/hash 顺序引入不可复现性。
        valid_series = series.dropna()
        candidate_items = []
        score_values = []
        label_values = []
        active_values = []
        for arr in valid_series:
            for item in self._iter_array_object_items(arr):
                candidate_items.append(item)
                if item.get("score") is not None:
                    score_values.append(int(item["score"]))
                if item.get("label") is not None:
                    label_values.append(str(item["label"]))
                if item.get("active") is not None:
                    active_values.append(bool(item["active"]))

        score_list = sorted(dict.fromkeys(score_values)) or [-50, -10, 0, 10, 50]
        label_list = sorted(dict.fromkeys(label_values)) or ["alpha", "beta", "gamma"]
        active_list = sorted(dict.fromkeys(active_values)) or [False, True]

        strategy = random.choice([
            "single_match",   # 单字段精确匹配
            "single_range",   # 单字段范围
            "multi_and",      # 多字段 AND
            "should_or",      # should (OR) 内嵌
            "must_not",       # must_not 内嵌
            "inner_min_should",
            "top_level_must_not_min_should",
        ])

        def _score_cmp(value, op, target):
            if value is None:
                return False
            if op == "gt":
                return value > target
            if op == "lt":
                return value < target
            if op == "gte":
                return value >= target
            if op == "lte":
                return value <= target
            return False

        def _build_inner_min_should_bundle():
            chosen = random.choice(candidate_items) if candidate_items else {
                "score": random.choice(score_list),
                "label": random.choice(label_list),
                "active": random.choice(active_list),
            }
            raw_score = chosen.get("score")
            raw_label = chosen.get("label")
            raw_active = chosen.get("active")
            score_val = int(raw_score) if raw_score is not None else random.choice(score_list)
            label_val = str(raw_label) if raw_label is not None else random.choice(label_list)
            active_val = bool(raw_active) if raw_active is not None else random.choice(active_list)
            conditions = [
                FieldCondition(key="score", range=Range(gte=score_val)),
                FieldCondition(key="label", match=MatchValue(value=label_val)),
                FieldCondition(key="active", match=MatchValue(value=active_val)),
            ]
            min_count = random.choice([1, 2, len(conditions)])

            def _item_hits(item):
                if not isinstance(item, dict):
                    return False
                hits = 0
                if _score_cmp(item.get("score"), "gte", score_val):
                    hits += 1
                if item.get("label") == label_val:
                    hits += 1
                if item.get("active") == active_val:
                    hits += 1
                return hits >= min_count

            def _nested_positive_mask(x):
                return any(_item_hits(item) for item in self._iter_array_object_items(x))

            inner = Filter(min_should=MinShould(conditions=conditions, min_count=min_count))
            expr = (
                f'{name} nested(AT_LEAST_{min_count}(score >= {score_val}; '
                f'label == "{label_val}"; active == {active_val}))'
            )
            return inner, _nested_positive_mask, expr

        # --- 策略 1: 单字段精确匹配 ---
        if strategy == "single_match":
            sub_field = random.choice(["score", "label", "active"])
            if sub_field == "score":
                val = random.choice(score_list)
                inner_filter = Filter(must=[
                    FieldCondition(key="score", match=MatchValue(value=val))
                ])
                def _mask(x, _v=val):
                    if not isinstance(x, list) or not x:
                        return False
                    return any(
                        isinstance(item, dict) and item.get("score") == _v
                        for item in x
                    )
                expr_str = f"{name} nested(score == {val})"
            elif sub_field == "label":
                val = random.choice(label_list)
                inner_filter = Filter(must=[
                    FieldCondition(key="label", match=MatchValue(value=val))
                ])
                def _mask(x, _v=val):
                    if not isinstance(x, list) or not x:
                        return False
                    return any(
                        isinstance(item, dict) and item.get("label") == _v
                        for item in x
                )
                expr_str = f'{name} nested(label == "{val}")'
            else:  # active
                val = random.choice(active_list)
                inner_filter = Filter(must=[
                    FieldCondition(key="active", match=MatchValue(value=val))
                ])
                def _mask(x, _v=val):
                    if not isinstance(x, list) or not x:
                        return False
                    return any(
                        isinstance(item, dict) and item.get("active") == _v
                        for item in x
                    )
                expr_str = f"{name} nested(active == {val})"

        # --- 策略 2: 单字段范围 (score) ---
        elif strategy == "single_range":
            val = random.choice(score_list)
            op = random.choice(["gt", "lt", "gte", "lte"])
            inner_filter = Filter(must=[
                FieldCondition(key="score", range=Range(**{op: val}))
            ])
            op_sym = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}[op]
            def _mask(x, _op=op, _v=val):
                if not isinstance(x, list) or not x:
                    return False
                for item in x:
                    if not isinstance(item, dict) or "score" not in item or item["score"] is None:
                        continue
                    s = item["score"]
                    if _op == "gt" and s > _v:
                        return True
                    if _op == "lt" and s < _v:
                        return True
                    if _op == "gte" and s >= _v:
                        return True
                    if _op == "lte" and s <= _v:
                        return True
                return False
            expr_str = f"{name} nested(score {op_sym} {val})"

        # --- 策略 3: 多字段 AND (score + label 或 score + active) ---
        elif strategy == "multi_and":
            score_val = random.choice(score_list)
            score_op = random.choice(["gt", "lt", "gte", "lte"])
            score_op_sym = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}[score_op]

            second_field = random.choice(["label", "active"])
            if second_field == "label":
                label_val = random.choice(label_list)
                inner_filter = Filter(must=[
                    FieldCondition(key="score", range=Range(**{score_op: score_val})),
                    FieldCondition(key="label", match=MatchValue(value=label_val)),
                ])
                def _mask(x, _sop=score_op, _sv=score_val, _lv=label_val):
                    if not isinstance(x, list) or not x:
                        return False
                    for item in x:
                        if not isinstance(item, dict):
                            continue
                        s = item.get("score")
                        l = item.get("label")
                        if s is None or l is None:
                            continue
                        score_ok = False
                        if _sop == "gt": score_ok = s > _sv
                        elif _sop == "lt": score_ok = s < _sv
                        elif _sop == "gte": score_ok = s >= _sv
                        elif _sop == "lte": score_ok = s <= _sv
                        if score_ok and l == _lv:
                            return True
                    return False
                expr_str = f'{name} nested(score {score_op_sym} {score_val} AND label == "{label_val}")'
            else:  # active
                active_val = random.choice(active_list)
                inner_filter = Filter(must=[
                    FieldCondition(key="score", range=Range(**{score_op: score_val})),
                    FieldCondition(key="active", match=MatchValue(value=active_val)),
                ])
                def _mask(x, _sop=score_op, _sv=score_val, _av=active_val):
                    if not isinstance(x, list) or not x:
                        return False
                    for item in x:
                        if not isinstance(item, dict):
                            continue
                        s = item.get("score")
                        a = item.get("active")
                        if s is None or a is None:
                            continue
                        score_ok = False
                        if _sop == "gt": score_ok = s > _sv
                        elif _sop == "lt": score_ok = s < _sv
                        elif _sop == "gte": score_ok = s >= _sv
                        elif _sop == "lte": score_ok = s <= _sv
                        if score_ok and a == _av:
                            return True
                    return False
                expr_str = f"{name} nested(score {score_op_sym} {score_val} AND active == {active_val})"

        # --- 策略 4: should (OR) 内嵌 ---
        elif strategy == "should_or":
            # 至少一个元素满足 score==v1 OR score==v2
            vals = random.sample(score_list, min(2, len(score_list)))
            if len(vals) < 2:
                vals = [vals[0], vals[0] + 1]
            v1, v2 = vals[0], vals[1]
            inner_filter = Filter(should=[
                Filter(must=[FieldCondition(key="score", match=MatchValue(value=v1))]),
                Filter(must=[FieldCondition(key="score", match=MatchValue(value=v2))]),
            ])
            def _mask(x, _v1=v1, _v2=v2):
                if not isinstance(x, list) or not x:
                    return False
                # Qdrant nested + should: 至少一个元素满足 should 中任意一个条件
                for item in x:
                    if not isinstance(item, dict):
                        continue
                    s = item.get("score")
                    if s is not None and (s == _v1 or s == _v2):
                        return True
                return False
            expr_str = f"{name} nested(score == {v1} OR score == {v2})"

        # --- 策略 5: must_not 内嵌 ---
        elif strategy == "must_not":
            # 至少一个元素不满足 active==True（即 active 不为 True）
            active_val = random.choice(active_list)
            inner_filter = Filter(must_not=[
                FieldCondition(key="active", match=MatchValue(value=active_val)),
            ])
            def _mask(x, _av=active_val):
                if not isinstance(x, list) or not x:
                    return False
                # Qdrant nested + must_not: 只对 object 元素做检查；任一 object 未匹配被禁止条件即可命中。
                for item in x:
                    if not isinstance(item, dict):
                        continue
                    a = item.get("active")
                    if a != _av:
                        return True
                return False
            expr_str = f"{name} nested(NOT active == {active_val})"
        elif strategy == "inner_min_should":
            inner_filter, _mask, expr_str = _build_inner_min_should_bundle()
        else:  # top_level_must_not_min_should
            inner_filter, positive_mask_fn, inner_expr = _build_inner_min_should_bundle()
            nested_cond = NestedCondition(
                nested=Nested(key=name, filter=inner_filter)
            )
            filter_obj = Filter(must_not=[Filter(must=[nested_cond])])
            positive_mask = self._safe_apply(series, positive_mask_fn).fillna(False)
            return filter_obj, ~positive_mask, f"NOT ({inner_expr})"

        # 构造最终 NestedCondition + Filter
        nested_cond = NestedCondition(
            nested=Nested(key=name, filter=inner_filter)
        )
        filter_obj = Filter(must=[nested_cond])
        mask = self._safe_apply(series, _mask)
        return (filter_obj, mask, expr_str)

    def gen_has_id_expr(self):
        """
        生成 HasId 条件：使用点 ID 直接过滤。
        返回: (Filter, pandas_mask, expr_str)
        """
        all_ids = self.df["id"].tolist()
        if not all_ids:
            return None, None, None

        # 随机采样 1-10 个 ID
        sample_size = min(random.randint(1, 10), len(all_ids))
        selected_ids = [int(x) for x in random.sample(all_ids, sample_size)]

        # 50% 概率：正向 HasId，50%：NOT HasId
        if random.random() < 0.5:
            filter_obj = Filter(must=[HasIdCondition(has_id=selected_ids)])
            id_set = set(selected_ids)
            mask = self._safe_apply(self.df["id"], lambda x: int(x) in id_set)
            expr_str = f"id in {selected_ids}"
        else:
            filter_obj = Filter(must_not=[HasIdCondition(has_id=selected_ids)])
            id_set = set(selected_ids)
            mask = self._safe_apply(self.df["id"], lambda x: int(x) not in id_set)
            expr_str = f"id not in {selected_ids}"

        return filter_obj, mask, expr_str

    def _qdrant_is_null_filter(self, name):
        """严格 is_null：仅匹配“字段存在且值为 null”."""
        return Filter(must=[self._is_null_condition(name)])

    def _qdrant_not_null_filter(self, name):
        """为字段生成 is_not_null 的 Qdrant filter（自动适配演化字段）。
        - 演化字段（evo_）:
            * 默认: NOT IsEmpty（字段存在且有非 null/[] 值）
            * 开启显式 NULL 同步: NOT IsNull（严格 is_not_null）
        - 普通字段:         NOT IsNull
        """
        if self._is_evolved_field(name) and evolved_field_use_empty_semantics():
            return Filter(must_not=[self._is_empty_condition(name)])
        return Filter(must_not=[self._is_null_condition(name)])

    def gen_not_atomic_expr(self):
        """
        专项 NOT 原子表达式生成器（6种策略）。
        专门设计来触发 Qdrant 的 must_not + NULL 交互路径。
        返回 (filter_obj, pandas_mask, expr_str)
        """
        f = random.choice(self.schema)
        name, ftype = f["name"], f["type"]
        series = self.df[name]
        not_nullish_mask = self._series_qdrant_not_nullish(name, series)
        nullish_mask = self._series_qdrant_nullish(name, series)
        nullish_expr = self._nullish_expr(name)
        not_nullish_expr = self._not_nullish_expr(name)

        strategy = random.choice([
            "not_compare",     # NOT (col > X)
            "not_eq",          # NOT (col == X)
            "not_is_null",     # NOT (is null) ↔ is not null
            "not_is_not_null", # NOT (is not null) ↔ is null
            "not_and",         # NOT (col > X AND col < Y)
            "not_or",          # NOT (col == X OR col == Y)
        ])

        # --- 策略 1: NOT + 比较 ---
        if strategy == "not_compare":
            if ftype == FieldType.INT:
                valid = series.dropna()
                if valid.empty:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                val = self._normalize_int_scalar(self._convert_to_native(random.choice(valid.values)))
                if val is None:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                op = random.choice(["gt", "lt", "gte", "lte"])
                inner_cond = FieldCondition(key=name, range=Range(**{op: val}))
                f_obj = Filter(must_not=[inner_cond])

                def _mask_fn_int(x, _op=op, _val=val):
                    nx = self._normalize_int_scalar(x)
                    if nx is None:
                        return True  # null -> must_not 不命中 inner -> 保留
                    if _op == "gt":
                        return not (nx > _val)
                    if _op == "lt":
                        return not (nx < _val)
                    if _op == "gte":
                        return not (nx >= _val)
                    if _op == "lte":
                        return not (nx <= _val)
                    return True

                mask = self._safe_apply(series, _mask_fn_int)
                op_sym = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}[op]
                return f_obj, mask, f"NOT ({name} {op_sym} {val})"

            if ftype == FieldType.DATETIME:
                valid = series.dropna()
                if valid.empty:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                raw_val = self._convert_to_native(random.choice(valid.values))
                val_ts = self._normalize_datetime_scalar(raw_val)
                if val_ts is None:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                val_dt = self._format_datetime_utc(val_ts)
                op = random.choice(["gt", "lt", "gte", "lte"])
                inner_cond = FieldCondition(key=name, range=DatetimeRange(**{op: val_dt}))
                f_obj = Filter(must_not=[inner_cond])

                def _mask_fn_dt(x, _op=op, _val=val_ts):
                    nx = self._normalize_datetime_scalar(x)
                    if nx is None:
                        return True
                    if _op == "gt":
                        return not (nx > _val)
                    if _op == "lt":
                        return not (nx < _val)
                    if _op == "gte":
                        return not (nx >= _val)
                    if _op == "lte":
                        return not (nx <= _val)
                    return True

                mask = self._safe_apply(series, _mask_fn_dt)
                op_sym = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}[op]
                return f_obj, mask, f'NOT ({name} {op_sym} "{val_dt}")'

            if ftype == FieldType.FLOAT:
                valid = series.dropna()
                if valid.empty:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                val = self._convert_to_native(random.choice(valid.values))
                val = float(val)
                op = random.choice(["gt", "lt", "gte", "lte"])
                inner_cond = FieldCondition(key=name, range=Range(**{op: val}))
                f_obj = Filter(must_not=[inner_cond])
                # Oracle: Qdrant must_not 对 null → 不匹配 → 保留
                def _mask_fn(x, _op=op, _val=val):
                    if self._is_na_like(x):
                        return True  # null → must_not 不命中 → 保留
                    try:
                        xv = float(x)
                        if _op == "gt": return not (xv > _val)
                        if _op == "lt": return not (xv < _val)
                        if _op == "gte": return not (xv >= _val)
                        if _op == "lte": return not (xv <= _val)
                    except: return True
                    return True
                mask = self._safe_apply(series, _mask_fn)
                op_sym = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}[op]
                return f_obj, mask, f"NOT ({name} {op_sym} {val})"

            elif ftype == FieldType.STRING:
                valid = series.dropna()
                if valid.empty:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                val = str(random.choice(valid.values))
                inner_cond = FieldCondition(key=name, match=MatchValue(value=val))
                f_obj = Filter(must_not=[inner_cond])
                mask = self._safe_apply(series, lambda x: True if self._is_na_like(x) else x != val)
                return f_obj, mask, f'NOT ({name} == "{val}")'

            elif ftype == FieldType.BOOL:
                val_bool = random.choice([True, False])
                inner_cond = FieldCondition(key=name, match=MatchValue(value=val_bool))
                f_obj = Filter(must_not=[inner_cond])
                mask = self._safe_apply(series, lambda x: True if self._is_na_like(x) else x != val_bool)
                return f_obj, mask, f"NOT ({name} == {val_bool})"

            # fallback
            return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"

        # --- 策略 2: NOT + 等值 ---
        elif strategy == "not_eq":
            if ftype == FieldType.INT:
                valid = series.dropna()
                if valid.empty:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                val = self._normalize_int_scalar(self._convert_to_native(random.choice(valid.values)))
                if val is None:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                inner_cond = FieldCondition(key=name, match=MatchValue(value=val))
                f_obj = Filter(must_not=[inner_cond])
                mask = self._safe_apply(series, 
                    lambda x, t=val: (
                        self._normalize_int_scalar(x) is None or self._normalize_int_scalar(x) != t
                    )
                )
                return f_obj, mask, f"NOT ({name} == {val})"
            return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"

        # --- 策略 3: NOT (is null / is empty) → 非空行 ---
        elif strategy == "not_is_null":
            return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"

        # --- 策略 4: NOT (is not null / is not empty) → 空行 ---
        elif strategy == "not_is_not_null":
            # NOT(NOT(is_null)) = is_null
            inner = self._qdrant_not_null_filter(name)
            f_obj = Filter(must_not=[inner])
            return f_obj, nullish_mask, f"NOT ({not_nullish_expr})"

        # --- 策略 5: NOT + AND (德摩根定律) ---
        elif strategy == "not_and":
            if ftype == FieldType.INT:
                valid = series.dropna()
                if valid.empty:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                val = self._normalize_int_scalar(self._convert_to_native(random.choice(valid.values)))
                if val is None:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                low = self._offset_int64(val, -random.randint(10, 100))
                high = self._offset_int64(val, random.randint(10, 100))
                inner = Filter(must=[
                    FieldCondition(key=name, range=Range(gt=low)),
                    FieldCondition(key=name, range=Range(lt=high))
                ])
                f_obj = Filter(must_not=[inner])
                # Qdrant must_not: null → 不匹配 inner → 保留
                def _and_mask(x, _low=low, _high=high):
                    nx = self._normalize_int_scalar(x)
                    if nx is None:
                        return True
                    return not (nx > _low and nx < _high)
                mask = self._safe_apply(series, _and_mask)
                return f_obj, mask, f"NOT ({name} > {low} AND {name} < {high})"
            if ftype == FieldType.DATETIME:
                valid = series.dropna()
                if valid.empty:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                raw_val = self._convert_to_native(random.choice(valid.values))
                val_ts = self._normalize_datetime_scalar(raw_val)
                if val_ts is None:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                low_dt = self._format_datetime_utc(val_ts - pd.Timedelta(days=random.randint(1, 7)))
                high_dt = self._format_datetime_utc(val_ts + pd.Timedelta(days=random.randint(1, 7)))
                inner = Filter(must=[
                    FieldCondition(key=name, range=DatetimeRange(gt=low_dt)),
                    FieldCondition(key=name, range=DatetimeRange(lt=high_dt))
                ])
                f_obj = Filter(must_not=[inner])

                def _and_mask_dt(x, _low=low_dt, _high=high_dt):
                    nx = self._normalize_datetime_scalar(x)
                    low_ts = self._normalize_datetime_scalar(_low)
                    high_ts = self._normalize_datetime_scalar(_high)
                    if nx is None or low_ts is None or high_ts is None:
                        return True
                    return not (nx > low_ts and nx < high_ts)

                mask = self._safe_apply(series, _and_mask_dt)
                return f_obj, mask, f'NOT ({name} > "{low_dt}" AND {name} < "{high_dt}")'
            return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"

        # --- 策略 6: NOT + OR (德摩根延伸) ---
        elif strategy == "not_or":
            if ftype == FieldType.INT:
                valid = series.dropna()
                if len(valid) < 2:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                vals = []
                for v in random.sample(list(valid.values), min(8, len(valid))):
                    nv = self._normalize_int_scalar(self._convert_to_native(v))
                    if nv is not None:
                        vals.append(nv)
                    if len(vals) >= 2:
                        break
                if len(vals) < 2:
                    return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"
                v1, v2 = vals[0], vals[-1]
                inner = Filter(should=[
                    Filter(must=[FieldCondition(key=name, match=MatchValue(value=v1))]),
                    Filter(must=[FieldCondition(key=name, match=MatchValue(value=v2))])
                ])
                f_obj = Filter(must_not=[inner])
                def _or_mask(x, _v1=v1, _v2=v2):
                    nx = self._normalize_int_scalar(x)
                    if nx is None:
                        return True
                    return not (nx == _v1 or nx == _v2)
                mask = self._safe_apply(series, _or_mask)
                return f_obj, mask, f"NOT ({name} == {v1} OR {name} == {v2})"
            return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"

        # 终极兜底
        return self._qdrant_not_null_filter(name), not_nullish_mask, f"NOT ({nullish_expr})"

    def gen_constant_expr(self):

        """生成常量表达式（Qdrant 支持有限）"""
        # Qdrant 中 id 是点 ID，不能用于 FieldCondition 过滤
        # 使用一个实际存在的 INT 字段来实现恒真/恒假条件
        using_int = False

        # 找到一个 INT 类型的字段
        int_fields = [f for f in self.schema if f["type"] == FieldType.INT]
        if int_fields:
            field_name = int_fields[0]["name"]
            series = self.df[field_name]
            min_val = self._clip_int64(series.min())
            max_val = self._clip_int64(series.max())
            using_int = True
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
            if using_int:
                low = self._offset_int64(min_val, -1000000)
                high = self._offset_int64(max_val, 1000000)
            else:
                low = min_val - 1000000
                high = max_val + 1000000
            filter_cond = FieldCondition(key=field_name, range=Range(gte=low, lte=high))
            return (filter_cond, true_mask, f"{field_name} in wide range (always true)")
        else:
            # 必假：使用一个不可能的范围（大于最大值 + 大偏移）
            if using_int:
                impossible_val = self._offset_int64(max_val, 10000000)
                if impossible_val > max_val:
                    filter_cond = FieldCondition(key=field_name, range=Range(gt=impossible_val))
                    return (filter_cond, false_mask, f"{field_name} > {impossible_val} (always false)")
                # 触发饱和时，改用矛盾区间避免产生越界数字
                filter_cond = Filter(must=[
                    FieldCondition(key=field_name, range=Range(gt=max_val)),
                    FieldCondition(key=field_name, range=Range(lt=min_val)),
                ])
                return (filter_cond, false_mask, f"({field_name} > {max_val} and {field_name} < {min_val})")
            impossible_val = max_val + 10000000
            filter_cond = FieldCondition(key=field_name, range=Range(gt=impossible_val))
            return (filter_cond, false_mask, f"{field_name} > {impossible_val} (always false)")

    def gen_complex_expr(self, depth, _retry=0):
        """递归生成复杂表达式（_retry 用于防止无限递归）"""
        _MAX_RETRY = 20
        if _retry >= _MAX_RETRY:
            # 到达重试上限，用 HasId 兜底确保不会无限递归
            if not self.df.empty:
                res = self.gen_has_id_expr()
                if res and res[0]:
                    return res
            return None, None, None

        if depth == 0 or random.random() < 0.2:
            if random.random() < 0.02:
                res = self.gen_constant_expr()
                if res[0]:
                    # 确保返回 Filter 对象
                    filter_obj = res[0]
                    if isinstance(filter_obj, FieldCondition):
                        filter_obj = Filter(must=[filter_obj])
                    return filter_obj, res[1], res[2]

            # HasId 条件：10% 概率使用 ID 直接过滤
            if random.random() < 0.1 and not self.df.empty:
                res = self.gen_has_id_expr()
                if res[0]:
                    return res

            # 15% 概率生成专项 NOT 表达式
            if random.random() < 0.15:
                res = self.gen_not_atomic_expr()
                if res and res[0]:
                    f_obj = res[0]
                    if isinstance(f_obj, FieldCondition):
                        f_obj = Filter(must=[f_obj])
                    return f_obj, res[1], res[2]

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
            return self.gen_complex_expr(depth, _retry=_retry + 1)

        # 递归生成子节点
        filter_l, mask_l, expr_l = self.gen_complex_expr(depth - 1, _retry=0)
        if not filter_l: return self.gen_complex_expr(depth, _retry=_retry + 1)

        op = random.choices(
            ["and", "or", "not", "min_should"],
            weights=[0.34, 0.30, 0.26, 0.10],
            k=1,
        )[0]

        if op == "not":
            # NOT 逻辑：取反子表达式
            # Qdrant must_not: 排除匹配的点（null 值的点由于不匹配条件，会被保留）
            # Pandas ~mask: True -> False, False -> True（与 must_not 语义一致）
            not_filter = Filter(must_not=[filter_l])
            mask_lb = self._as_bool_mask(mask_l)
            not_mask = ~mask_lb.fillna(False)
            return not_filter, not_mask, f"NOT ({expr_l})"

        if op == "min_should":
            children = [(filter_l, mask_l, expr_l)]
            for _ in range(random.choice([1, 2])):
                child_filter, child_mask, child_expr = self.gen_complex_expr(depth - 1, _retry=0)
                if child_filter is not None:
                    children.append((child_filter, child_mask, child_expr))

            if len(children) < 2:
                return filter_l, mask_l, expr_l

            edge_counts = [1, len(children)]
            if len(children) > 1:
                edge_counts.append(max(1, len(children) - 1))
            min_count = random.choice(edge_counts)
            hit_count = pd.Series(0, index=self.df.index, dtype="int64")
            child_exprs = []
            child_filters = []
            for child_filter, child_mask, child_expr in children:
                child_filters.append(child_filter)
                child_exprs.append(child_expr)
                hit_count = hit_count + self._as_bool_mask(child_mask).fillna(False).astype("int64")
            combined_filter = Filter(
                min_should=MinShould(conditions=child_filters, min_count=min_count)
            )
            return combined_filter, (hit_count >= min_count), f"AT_LEAST_{min_count}({'; '.join(child_exprs)})"

        # AND/OR 需要第二个子表达式
        filter_r, mask_r, expr_r = self.gen_complex_expr(depth - 1)
        if not filter_r: return filter_l, mask_l, expr_l

        if op == "and":
            # AND 逻辑：将两个 Filter 作为整体放入 must 列表
            # 不能直接提取 must 字段，因为 Filter 可能是用 should/must_not 构建的
            combined_filter = Filter(must=[filter_l, filter_r])
            mask_lb = self._as_bool_mask(mask_l).fillna(False)
            mask_rb = self._as_bool_mask(mask_r).fillna(False)
            return combined_filter, (mask_lb & mask_rb), f"({expr_l} AND {expr_r})"
        else:
            # OR 逻辑：将两个 Filter 作为整体放入 should 列表
            combined_filter = Filter(should=[filter_l, filter_r])
            mask_lb = self._as_bool_mask(mask_l).fillna(False)
            mask_rb = self._as_bool_mask(mask_r).fillna(False)
            return combined_filter, (mask_lb | mask_rb), f"({expr_l} OR {expr_r})"


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
            if self._tautology_field.get("type") == "int":
                min_val = self._clip_int64(min_val)
                max_val = self._clip_int64(max_val)
                low = self._offset_int64(min_val, -1000000)
                high = self._offset_int64(max_val, 1000000)
            else:
                low = min_val - 1000000
                high = max_val + 1000000
            # 使用 (宽范围 OR 字段为null) 来确保包含所有数据（包括 null）
            # 因为 Qdrant 的 Range 条件不会匹配 null 值
            range_cond = Filter(must=[FieldCondition(key=name, range=Range(gte=low, lte=high))])
            null_cond = self._qdrant_nullish_filter(name)
            return (
                Filter(should=[range_cond, null_cond]),
                f"({name} in wide range OR {self._nullish_expr(name)}) (always true)"
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
                FieldCondition(key=name, range=DatetimeRange(gt="2100-01-01T00:00:00Z")),
                f'{name} > "2100-01-01T00:00:00Z"'
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
        # A => (A AND field < mid) OR (A AND field >= mid) OR (A AND field is null/is empty)
        # 使用 INT 字段进行切分，需要包含 null 值以保持等价性
        if self._tautology_field:
            part_name = self._tautology_field["name"]
            mid_val = (self._tautology_field["min"] + self._tautology_field["max"]) / 2
            part1 = Filter(must=[base_filter, FieldCondition(key=part_name, range=Range(lt=mid_val))])
            part2 = Filter(must=[base_filter, FieldCondition(key=part_name, range=Range(gte=mid_val))])
            part3 = Filter(must=[base_filter, self._qdrant_nullish_filter(part_name)])  # null/missing 分区（evo 适配）
            
            mutations.append({
                "type": "PartitionByField",
                "filter": Filter(should=[part1, part2, part3]),
                "expr": f"(({base_expr}) AND {part_name}<{mid_val}) OR (({base_expr}) AND {part_name}>={mid_val}) OR (({base_expr}) AND {self._nullish_expr(part_name)})"
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

        # 7. 恒真条件左侧注入 (TautologyAnd_Left)
        # True AND A => A
        if tautology_filter:
            mutations.append({
                "type": "TautologyAnd_Left",
                "filter": Filter(must=[tautology_filter, base_filter]),
                "expr": f"({tautology_expr}) AND ({base_expr})"
            })

        # 8. 整数范围微调 (IntRangeShift)
        # 对 INT 字段：gt=N <=> gte=N+1, lt=N <=> lte=N-1
        self._add_int_range_shift_mutations(base_filter, base_expr, mutations)

        # 9. InExpandToOr: MatchAny([a,b,c]) → should[MatchValue(a), MatchValue(b), MatchValue(c)]
        self._add_in_expand_to_or_mutations(base_filter, base_expr, mutations)

        return mutations

    def _add_int_range_shift_mutations(self, base_filter, base_expr, mutations):
        """
        尝试在 base_filter 中找到 INT 字段的 Range 条件，生成等价的微调变体。
        gt=N <=> gte=N+1, lt=N <=> lte=N-1 (仅对整数字段有效)
        """
        if not isinstance(base_filter, Filter):
            return

        # 查找 must 列表中的 FieldCondition + Range
        conditions = base_filter.must or []
        for cond in conditions:
            if not isinstance(cond, FieldCondition):
                continue
            if cond.range is None:
                continue

            # 检查是否为 INT 类型字段
            fname = cond.key
            ftype = self.field_types.get(fname)
            if ftype != FieldType.INT:
                continue

            rng = cond.range
            # gt=N → gte=N+1
            if rng.gt is not None and isinstance(rng.gt, (int, float)):
                shifted_val = int(rng.gt) + 1
                new_cond = FieldCondition(key=fname, range=Range(
                    gte=shifted_val, lt=rng.lt, lte=rng.lte
                ))
                new_must = [new_cond if c is cond else c for c in conditions]
                mutations.append({
                    "type": "IntRangeShift_gt_to_gte",
                    "filter": Filter(must=new_must, must_not=base_filter.must_not, should=base_filter.should, min_should=base_filter.min_should),
                    "expr": f"{base_expr} [gt={int(rng.gt)} → gte={shifted_val}]"
                })
                break

            # lt=N → lte=N-1
            if rng.lt is not None and isinstance(rng.lt, (int, float)):
                shifted_val = int(rng.lt) - 1
                new_cond = FieldCondition(key=fname, range=Range(
                    gt=rng.gt, gte=rng.gte, lte=shifted_val
                ))
                new_must = [new_cond if c is cond else c for c in conditions]
                mutations.append({
                    "type": "IntRangeShift_lt_to_lte",
                    "filter": Filter(must=new_must, must_not=base_filter.must_not, should=base_filter.should, min_should=base_filter.min_should),
                    "expr": f"{base_expr} [lt={int(rng.lt)} → lte={shifted_val}]"
                })
                break

            # gte=N → gt=N-1
            if rng.gte is not None and isinstance(rng.gte, (int, float)):
                shifted_val = int(rng.gte) - 1
                new_cond = FieldCondition(key=fname, range=Range(
                    gt=shifted_val, lt=rng.lt, lte=rng.lte
                ))
                new_must = [new_cond if c is cond else c for c in conditions]
                mutations.append({
                    "type": "IntRangeShift_gte_to_gt",
                    "filter": Filter(must=new_must, must_not=base_filter.must_not, should=base_filter.should, min_should=base_filter.min_should),
                    "expr": f"{base_expr} [gte={int(rng.gte)} → gt={shifted_val}]"
                })
                break

            # lte=N → lt=N+1
            if rng.lte is not None and isinstance(rng.lte, (int, float)):
                shifted_val = int(rng.lte) + 1
                new_cond = FieldCondition(key=fname, range=Range(
                    gt=rng.gt, gte=rng.gte, lt=shifted_val
                ))
                new_must = [new_cond if c is cond else c for c in conditions]
                mutations.append({
                    "type": "IntRangeShift_lte_to_lt",
                    "filter": Filter(must=new_must, must_not=base_filter.must_not, should=base_filter.should, min_should=base_filter.min_should),
                    "expr": f"{base_expr} [lte={int(rng.lte)} → lt={shifted_val}]"
                })
                break

        return mutations

    def _add_in_expand_to_or_mutations(self, base_filter, base_expr, mutations):
        """
        InExpandToOr: 将 MatchAny([a,b,c]) 展开为 should[MatchValue(a), MatchValue(b), MatchValue(c)]
        递归搜索 filter 树，找到第一个 MatchAny 条件并展开。
        """
        if not isinstance(base_filter, Filter):
            return

        # 递归搜索所有 must/must_not/should 中的 FieldCondition
        def _find_and_replace(filter_obj):
            """递归查找含 MatchAny 的 FieldCondition，返回替换后的新 Filter 或 None"""
            if isinstance(filter_obj, FieldCondition):
                if filter_obj.match and isinstance(filter_obj.match, MatchAny):
                    any_vals = filter_obj.match.any
                    if any_vals and 1 < len(any_vals) <= 10:
                        fname = filter_obj.key
                        should_list = [Filter(must=[FieldCondition(key=fname, match=MatchValue(value=v))]) for v in any_vals]
                        return Filter(should=should_list)
                return None
            if not isinstance(filter_obj, Filter):
                return None

            # 遍历 must
            if filter_obj.must:
                for i, cond in enumerate(filter_obj.must):
                    replacement = _find_and_replace(cond)
                    if replacement is not None:
                        new_must = list(filter_obj.must)
                        new_must[i] = replacement
                        return Filter(must=new_must, must_not=filter_obj.must_not, should=filter_obj.should, min_should=filter_obj.min_should)
            # 遍历 should
            if filter_obj.should:
                for i, cond in enumerate(filter_obj.should):
                    replacement = _find_and_replace(cond)
                    if replacement is not None:
                        new_should = list(filter_obj.should)
                        new_should[i] = replacement
                        return Filter(must=filter_obj.must, must_not=filter_obj.must_not, should=new_should, min_should=filter_obj.min_should)
            # 遍历 must_not
            if filter_obj.must_not:
                for i, cond in enumerate(filter_obj.must_not):
                    replacement = _find_and_replace(cond)
                    if replacement is not None:
                        new_must_not = list(filter_obj.must_not)
                        new_must_not[i] = replacement
                        return Filter(must=filter_obj.must, must_not=new_must_not, should=filter_obj.should, min_should=filter_obj.min_should)
            # 遍历 min_should
            if filter_obj.min_should:
                for i, cond in enumerate(filter_obj.min_should.conditions):
                    replacement = _find_and_replace(cond)
                    if replacement is not None:
                        new_conditions = list(filter_obj.min_should.conditions)
                        new_conditions[i] = replacement
                        new_min_should = MinShould(
                            conditions=new_conditions,
                            min_count=filter_obj.min_should.min_count,
                        )
                        return Filter(must=filter_obj.must, must_not=filter_obj.must_not, should=filter_obj.should, min_should=new_min_should)
            return None

        result = _find_and_replace(base_filter)
        if result is not None:
            mutations.append({
                "type": "InExpandToOr",
                "filter": result,
                "expr": f"{base_expr} [MatchAny→should+MatchValue]"
            })


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
            if self._tautology_field.get("type") == "int":
                min_val = self._clip_int64(min_val)
                max_val = self._clip_int64(max_val)
                low = self._offset_int64(min_val, -1000000)
                high = self._offset_int64(max_val, 1000000)
            else:
                low = min_val - 1000000
                high = max_val + 1000000
            # 使用 (宽范围 OR 字段为null) 来确保包含所有数据（包括 null）
            # 因为 Qdrant 的 Range 条件不会匹配 null 值
            range_cond = Filter(must=[FieldCondition(key=name, range=Range(gte=low, lte=high))])
            null_cond = self._qdrant_nullish_filter(name)
            return (
                Filter(should=[range_cond, null_cond]),
                f"({name} in wide range OR {self._nullish_expr(name)}) (always true)"
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

    def gen_multi_field_true_filter(self, pivot_row, n=3, min_fields=2):
        """
        多字段合取必真条件：从 pivot_row 中选 2-4 个非空标量字段，
        组合成 AND 条件。比纯 HasId 更有攻击性的 fallback。
        """
        valid_fields = []
        for f in self.schema:
            fname = f["name"]
            ftype = f["type"]
            val = pivot_row[fname]
            if val is None: continue
            if isinstance(val, float) and np.isnan(val): continue
            if ftype in [FieldType.INT, FieldType.FLOAT, FieldType.STRING, FieldType.UUID, FieldType.BOOL, FieldType.DATETIME]:
                valid_fields.append(f)

        if not valid_fields:
            # 纯 ID 兜底
            pid = int(pivot_row["id"])
            return HasIdCondition(has_id=[pid]), f"id == {pid}"

        count = min(random.randint(min_fields, n), len(valid_fields))
        chosen = random.sample(valid_fields, count)

        filters = []
        exprs = []
        for f in chosen:
            fname = f["name"]
            ftype = f["type"]
            val = pivot_row[fname]
            if hasattr(val, "item"): val = val.item()

            try:
                if ftype == FieldType.BOOL:
                    filters.append(FieldCondition(key=fname, match=MatchValue(value=bool(val))))
                    exprs.append(f"{fname}=={bool(val)}")
                elif ftype == FieldType.INT:
                    v = int(val)
                    filters.append(FieldCondition(key=fname, range=Range(gte=v, lte=v)))
                    exprs.append(f"{fname}=={v}")
                elif ftype == FieldType.DATETIME:
                    v = self._format_datetime_utc(self._normalize_datetime_scalar(val))
                    if v is None:
                        continue
                    filters.append(FieldCondition(key=fname, range=DatetimeRange(gte=v, lte=v)))
                    exprs.append(f'{fname}=="{v}"')
                elif ftype == FieldType.FLOAT:
                    v = float(val)
                    eps = 1e-5
                    filters.append(FieldCondition(key=fname, range=Range(gte=v-eps, lte=v+eps)))
                    exprs.append(f"{fname}~={v}")
                elif ftype == FieldType.STRING:
                    v = str(val)
                    filters.append(FieldCondition(key=fname, match=MatchValue(value=v)))
                    exprs.append(f'{fname}=="{v}"')
                elif ftype == FieldType.UUID:
                    v = str(uuid.UUID(str(val)))
                    filters.append(FieldCondition(key=fname, match=MatchValue(value=v)))
                    exprs.append(f'{fname} uuid=="{v}"')
            except:
                pass

        if not filters:
            pid = int(pivot_row["id"])
            return HasIdCondition(has_id=[pid]), f"id == {pid}"

        combined = Filter(must=filters)
        return combined, " AND ".join(exprs)

    def gen_pqs_filter(self, pivot_row, depth):
        """生成针对指定行必真的过滤条件"""
        force_recursion = depth > 3

        if depth <= 0 or (not force_recursion and random.random() < 0.3):
            # 30% 概率使用多字段合取 fallback（更强攻击性）
            if random.random() < 0.3:
                return self.gen_multi_field_true_filter(pivot_row)
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
            null_cond = self._qdrant_nullish_filter(name)
            return (
                Filter(should=[range_cond, null_cond]),
                f"({name} in wide range OR {self._nullish_expr(name)})"
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
            return self._qdrant_nullish_filter(fname), self._nullish_expr(fname)

        # 根据类型生成条件
        if ftype == FieldType.BOOL:
            return FieldCondition(key=fname, match=MatchValue(value=bool(val))), f"{fname} == {bool(val)}"

        elif ftype == FieldType.INT:
            val_int = self._clip_int64(val.item() if hasattr(val, "item") else val)
            strategies = []
            # 策略1: 精确匹配
            strategies.append((
                FieldCondition(key=fname, match=MatchValue(value=val_int)),
                f"{fname} == {val_int}"
            ))
            # 策略2: 闭区间包裹
            strategies.append((
                FieldCondition(key=fname, range=Range(gte=val_int, lte=val_int)),
                f"{fname} >= {val_int} AND <= {val_int}"
            ))
            # 策略3: 开区间包裹
            strategies.append((
                FieldCondition(
                    key=fname,
                    range=Range(
                        gt=self._offset_int64(val_int, -1),
                        lt=self._offset_int64(val_int, 1),
                    ),
                ),
                f"{fname} > {self._offset_int64(val_int, -1)} AND < {self._offset_int64(val_int, 1)}"
            ))
            # 策略4: MatchAny 包含真值+噪声
            dummies = [self._offset_int64(val_int, random.randint(100000, 200000)) for _ in range(3)]
            candidates = dummies + [val_int]
            random.shuffle(candidates)
            strategies.append((
                FieldCondition(key=fname, match=MatchAny(any=candidates)),
                f"{fname} in {candidates}"
            ))
            # 策略5: MatchExcept 排除不存在的假值
            fake_vals = [self._offset_int64(val_int, random.randint(100000, 200000)) for _ in range(3)]
            strategies.append((
                FieldCondition(key=fname, match=MatchExcept(**{"except": fake_vals})),
                f"{fname} not in {fake_vals}"
            ))
            # 策略6: must_not + Range 反面 (val < boundary → must_not val >= boundary)
            boundary = self._offset_int64(val_int, random.randint(1, 100))
            strategies.append((
                Filter(must_not=[FieldCondition(key=fname, range=Range(gte=boundary))]),
                f"NOT ({fname} >= {boundary})"
            ))
            return random.choice(strategies)

        elif ftype == FieldType.FLOAT:
            val_float = float(val)
            epsilon = 1e-5
            strategies = []
            # 策略1: 窄epsilon范围
            strategies.append((
                FieldCondition(key=fname, range=Range(gt=val_float - epsilon, lt=val_float + epsilon)),
                f"{fname} > {val_float - epsilon} AND < {val_float + epsilon}"
            ))
            # 策略2: 单边gte
            strategies.append((
                FieldCondition(key=fname, range=Range(gte=val_float - epsilon)),
                f"{fname} >= {val_float - epsilon}"
            ))
            # 策略3: 宽epsilon闭区间
            wide_eps = random.uniform(0.01, 1.0)
            strategies.append((
                FieldCondition(key=fname, range=Range(gte=val_float - wide_eps, lte=val_float + wide_eps)),
                f"{fname} in [{val_float - wide_eps:.6f}, {val_float + wide_eps:.6f}]"
            ))
            # 策略4: must_not 反面 (val > boundary → NOT val <= boundary)
            boundary = val_float - random.uniform(1.0, 100.0)
            strategies.append((
                Filter(must_not=[FieldCondition(key=fname, range=Range(lte=boundary))]),
                f"NOT ({fname} <= {boundary:.6f})"
            ))
            return random.choice(strategies)

        elif ftype == FieldType.STRING:
            val_str = str(val)
            strategies = []
            # 策略1: 精确匹配
            strategies.append((
                FieldCondition(key=fname, match=MatchValue(value=val_str)),
                f'{fname} == "{val_str}"'
            ))
            # 策略2: MatchAny 包含真值
            dummies = [self._random_string(5) for _ in range(3)]
            candidates = dummies + [val_str]
            random.shuffle(candidates)
            strategies.append((
                FieldCondition(key=fname, match=MatchAny(any=candidates)),
                f'{fname} in {candidates}'
            ))
            # 策略3: MatchExcept 排除不存在的假值
            fake_strs = [self._random_string(12) for _ in range(3)]
            strategies.append((
                FieldCondition(key=fname, match=MatchExcept(**{"except": fake_strs})),
                f'{fname} not in {fake_strs}'
            ))
            # 策略4: MatchPhrase（当前实现仅声明无 full-text index 的 substring 子集）
            if hasattr(models, "MatchPhrase") and val_str:
                if len(val_str) >= 4:
                    start = random.randint(0, max(0, len(val_str) - 3))
                    phrase = val_str[start:start + random.randint(2, min(6, len(val_str) - start))]
                else:
                    phrase = val_str
                strategies.append((
                    FieldCondition(key=fname, match=models.MatchPhrase(phrase=phrase)),
                    f'{fname} phrase "{phrase}"'
                ))
            # 策略5: MatchText（当前实现仅声明无 full-text index 的 substring 子集）
            if hasattr(models, "MatchText") and val_str:
                if len(val_str) >= 4:
                    start = random.randint(0, max(0, len(val_str) - 3))
                    query_text = val_str[start:start + random.randint(2, min(8, len(val_str) - start))]
                else:
                    query_text = val_str
                strategies.append((
                    FieldCondition(key=fname, match=models.MatchText(text=query_text)),
                    f'{fname} text "{query_text}"'
                ))
            # 策略6: MatchTextAny（当前实现仅声明无 full-text index 的任一 query term 子串命中子集）
            val_stripped = val_str.strip()
            if hasattr(models, "MatchTextAny") and val_stripped:
                source_terms = [part for part in val_stripped.split() if part]
                term_source = random.choice(source_terms)
                if len(term_source) >= 4:
                    start = random.randint(0, max(0, len(term_source) - 3))
                    real_term = term_source[start:start + random.randint(2, min(8, len(term_source) - start))]
                else:
                    real_term = term_source
                terms = [real_term, random.choice(["unlikelyterm", "absenttoken", "zzznomatch"])]
                random.shuffle(terms)
                query_text = " ".join(terms)
                strategies.append((
                    FieldCondition(key=fname, match=models.MatchTextAny(text_any=query_text)),
                    f'{fname} text_any "{query_text}"'
                ))
            return random.choice(strategies)

        elif ftype == FieldType.UUID:
            val_uuid = str(uuid.UUID(str(val)))
            return (
                FieldCondition(key=fname, match=MatchValue(value=val_uuid)),
                f'{fname} uuid == "{val_uuid}"'
            )

        elif ftype == FieldType.JSON:
            return self._gen_pqs_json_filter(fname, val)

        elif ftype in [FieldType.ARRAY_INT, FieldType.ARRAY_STR]:
            if not isinstance(val, list) or len(val) == 0:
                return self._qdrant_is_empty_filter(fname), f"{fname} is empty"
            
            valid_items = [x for x in val if x is not None]
            if valid_items:
                target = random.choice(valid_items)
                # 确保 target 是原生 Python 类型
                target = self._convert_to_native(target)
                strategies = []
                if ftype == FieldType.ARRAY_INT:
                    noise = [
                        self._offset_int64(int(target), random.randint(100000, 200000))
                        for _ in range(2)
                    ]
                else:
                    noise = [self._random_string(8, 12) for _ in range(2)]
                candidates = noise + [target]
                random.shuffle(candidates)
                candidates = list(dict.fromkeys(candidates))
                strategies.append((
                    FieldCondition(key=fname, match=MatchAny(any=candidates)),
                    f"{fname} contains any of {candidates}"
                ))
                if ftype == FieldType.ARRAY_INT:
                    fake = [
                        self._offset_int64(int(target), random.randint(200001, 400000))
                        for _ in range(3)
                    ]
                else:
                    fake = [self._random_string(12, 16) for _ in range(3)]
                fake = list(dict.fromkeys(fake))
                strategies.append((
                    FieldCondition(key=fname, match=MatchExcept(**{"except": fake})),
                    f"{fname} has element not in {fake}"
                ))
                return random.choice(strategies)
            return self._qdrant_not_null_filter(fname), self._not_nullish_expr(fname)

        elif ftype == FieldType.ARRAY_FLOAT:
            if not isinstance(val, list) or len(val) == 0:
                return self._qdrant_is_empty_filter(fname), f"{fname} is empty"
            valid_items = [x for x in val if x is not None]
            if valid_items:
                target = float(random.choice(valid_items))
                epsilon = 0.5
                return FieldCondition(key=fname, range=Range(gte=target - epsilon, lte=target + epsilon)), f"{fname} has element ~= {target}"
            return self._qdrant_not_null_filter(fname), self._not_nullish_expr(fname)

        elif ftype == FieldType.DATETIME:
            val_dt = self._format_datetime_utc(self._normalize_datetime_scalar(val))
            if val_dt is None:
                return self._qdrant_nullish_filter(fname), self._nullish_expr(fname)
            strategies = []
            prev_dt = self._offset_datetime_str(val_dt, seconds=-1)
            next_dt = self._offset_datetime_str(val_dt, seconds=1)
            future_dt = self._offset_datetime_str(val_dt, days=1)
            # 策略1: 精确闭区间
            strategies.append((
                FieldCondition(key=fname, range=DatetimeRange(gte=val_dt, lte=val_dt)),
                f'{fname} datetime == "{val_dt}"'
            ))
            if prev_dt is not None and next_dt is not None:
                strategies.append((
                    FieldCondition(key=fname, range=DatetimeRange(gt=prev_dt, lt=next_dt)),
                    f'{fname} datetime in ("{prev_dt}", "{next_dt}")'
                ))
            if future_dt is not None:
                strategies.append((
                    Filter(must_not=[FieldCondition(key=fname, range=DatetimeRange(gte=future_dt))]),
                    f'NOT ({fname} >= "{future_dt}")'
                ))
            return random.choice(strategies)

        elif ftype == FieldType.GEO:
            if isinstance(val, dict) and "lat" in val and "lon" in val:
                lat, lon = val["lat"], val["lon"]
                strategies = []
                # 策略1: 极小bbox
                delta = 0.001
                top_lat = min(lat + delta, 90)
                bottom_lat = max(lat - delta, -90)
                left_lon = max(lon - delta, -180)
                right_lon = min(lon + delta, 180)
                pivot_on_bbox_edge = (
                    math.isclose(lat, top_lat)
                    or math.isclose(lat, bottom_lat)
                    or math.isclose(lon, left_lon)
                    or math.isclose(lon, right_lon)
                )
                bbox_clamped_to_global_edge = (
                    math.isclose(top_lat, 90.0)
                    or math.isclose(bottom_lat, -90.0)
                    or math.isclose(left_lon, -180.0)
                    or math.isclose(right_lon, 180.0)
                )
                if not pivot_on_bbox_edge and not bbox_clamped_to_global_edge:
                    strategies.append((
                        FieldCondition(
                            key=fname,
                            geo_bounding_box=GeoBoundingBox(
                                top_left=GeoPoint(lat=top_lat, lon=left_lon),
                                bottom_right=GeoPoint(lat=bottom_lat, lon=right_lon)
                            )
                        ), f"{fname} in tiny bbox around ({lat},{lon})"
                    ))
                # 策略2: GeoRadius 极小半径确保命中
                strategies.append((
                    FieldCondition(
                        key=fname,
                        geo_radius=GeoRadius(
                            center=GeoPoint(lat=lat, lon=lon),
                            radius=1000.0  # 1km 足够小确保命中
                        )
                    ), f"{fname} in radius 1km around ({lat},{lon})"
                ))
                return random.choice(strategies)
            return self._qdrant_not_null_filter(fname), self._not_nullish_expr(fname)

        elif ftype == FieldType.ARRAY_OBJECT:
            # Nested 对象数组 PQS: 从 pivot_row 的实际数组中选取一个元素构造必真条件
            if not isinstance(val, list) or len(val) == 0:
                return self._qdrant_is_empty_filter(fname), f"{fname} is empty"
            # 从数组中选一个有效的子对象
            valid_items = self._iter_array_object_items(val)
            if not valid_items:
                return self._qdrant_is_empty_filter(fname), f"{fname} is empty"
            chosen = random.choice(valid_items)
            strategies = []
            # 策略1: 精确匹配 score
            if "score" in chosen and chosen["score"] is not None:
                sv = int(chosen["score"])
                strategies.append((
                    Filter(must=[NestedCondition(
                        nested=Nested(key=fname, filter=Filter(must=[
                            FieldCondition(key="score", match=MatchValue(value=sv))
                        ]))
                    )]),
                    f"{fname} nested(score == {sv})"
                ))
            # 策略2: 精确匹配 label
            if "label" in chosen and chosen["label"] is not None:
                lv = str(chosen["label"])
                strategies.append((
                    Filter(must=[NestedCondition(
                        nested=Nested(key=fname, filter=Filter(must=[
                            FieldCondition(key="label", match=MatchValue(value=lv))
                        ]))
                    )]),
                    f'{fname} nested(label == "{lv}")'
                ))
            # 策略3: 多字段 AND (score + label)
            if ("score" in chosen and chosen["score"] is not None
                    and "label" in chosen and chosen["label"] is not None):
                sv = int(chosen["score"])
                lv = str(chosen["label"])
                strategies.append((
                    Filter(must=[NestedCondition(
                        nested=Nested(key=fname, filter=Filter(must=[
                            FieldCondition(key="score", match=MatchValue(value=sv)),
                            FieldCondition(key="label", match=MatchValue(value=lv)),
                        ]))
                    )]),
                    f'{fname} nested(score == {sv} AND label == "{lv}")'
                ))
            # 策略4: 范围包裹 score
            if "score" in chosen and chosen["score"] is not None:
                sv = int(chosen["score"])
                strategies.append((
                    Filter(must=[NestedCondition(
                        nested=Nested(key=fname, filter=Filter(must=[
                            FieldCondition(key="score", range=Range(gte=sv, lte=sv))
                        ]))
                    )]),
                    f"{fname} nested(score >= {sv} AND score <= {sv})"
                ))
            # 策略5: should 内嵌
            if "score" in chosen and chosen["score"] is not None:
                sv = int(chosen["score"])
                alt = self._offset_int64(sv, random.choice([101, 203, 509]))
                if alt == sv:
                    alt = self._offset_int64(sv, 997)
                strategies.append((
                    Filter(must=[NestedCondition(
                        nested=Nested(key=fname, filter=Filter(should=[
                            Filter(must=[FieldCondition(key="score", match=MatchValue(value=sv))]),
                            Filter(must=[FieldCondition(key="score", match=MatchValue(value=alt))]),
                        ]))
                    )]),
                    f"{fname} nested(score == {sv} OR score == {alt})"
                ))
            # 策略6: inner must_not
            if "active" in chosen and chosen["active"] is not None:
                av = bool(chosen["active"])
                forbidden = not av
                strategies.append((
                    Filter(must=[NestedCondition(
                        nested=Nested(key=fname, filter=Filter(must_not=[
                            FieldCondition(key="active", match=MatchValue(value=forbidden))
                        ]))
                    )]),
                    f"{fname} nested(NOT active == {forbidden})"
                ))
            # 策略7: inner min_should（同一 nested object 内至少两个子谓词成立）
            if (
                "score" in chosen and chosen["score"] is not None
                and "label" in chosen and chosen["label"] is not None
                and "active" in chosen and chosen["active"] is not None
            ):
                sv = int(chosen["score"])
                lv = str(chosen["label"])
                av = bool(chosen["active"])
                strategies.append((
                    Filter(must=[NestedCondition(
                        nested=Nested(key=fname, filter=Filter(min_should=MinShould(
                            conditions=[
                                FieldCondition(key="score", range=Range(gte=sv)),
                                FieldCondition(key="label", match=MatchValue(value=lv)),
                                FieldCondition(key="active", match=MatchValue(value=av)),
                            ],
                            min_count=2,
                        )))
                    )]),
                    f'{fname} nested(AT_LEAST_2(score >= {sv}; label == "{lv}"; active == {av}))'
                ))
            # 策略8: projected key-path 跨 sibling 元素累积命中（与 nested 同元素语义形成对照）
            if len(valid_items) >= 2:
                pair_added = False
                row_pairs = {
                    (int(item["score"]), str(item["label"]))
                    for item in valid_items
                    if item.get("score") is not None and item.get("label") is not None
                }
                for left in valid_items:
                    left_score = left.get("score")
                    if left_score is None:
                        continue
                    for right in valid_items:
                        if right is left:
                            continue
                        right_label = right.get("label")
                        if right_label is None:
                            continue
                        pair = (int(left_score), str(right_label))
                        if pair in row_pairs:
                            continue
                        strategies.append((
                            Filter(must=[
                                FieldCondition(key=f"{fname}[].score", match=MatchValue(value=pair[0])),
                                FieldCondition(key=f"{fname}[].label", match=MatchValue(value=pair[1])),
                            ]),
                            f'{fname}[].score == {pair[0]} AND {fname}[].label == "{pair[1]}" (projected cross-element)'
                        ))
                        pair_added = True
                        break
                    if pair_added:
                        break
            if strategies:
                return random.choice(strategies)
            return self._qdrant_not_null_filter(fname), self._not_nullish_expr(fname)

        return self._qdrant_not_null_filter(fname), self._not_nullish_expr(fname)

    def _gen_pqs_json_filter(self, fname, json_obj):
        """
        融合版 JSON 必真条件生成器：
        支持递归深度下钻（最多12层），到达标量时生成边界条件。
        """
        if json_obj is None:
            return self._qdrant_nullish_filter(fname), self._nullish_expr(fname)
        if not isinstance(json_obj, dict) and not isinstance(json_obj, list):
            return self._qdrant_not_null_filter(fname), self._not_nullish_expr(fname)

        # 将 numpy 类型转换
        if hasattr(json_obj, "tolist"): json_obj = json_obj.tolist()
        if hasattr(json_obj, "item"): json_obj = json_obj.item()

        if isinstance(json_obj, dict):
            strategies = []
            country = json_obj.get("country")
            if isinstance(country, dict):
                country_name = country.get("name")
                if isinstance(country_name, str):
                    strategies.append((
                        FieldCondition(key=f"{fname}.country.name", match=MatchValue(value=country_name)),
                        f'{fname}.country.name == "{country_name}"'
                    ))
                cities = country.get("cities")
                if isinstance(cities, list):
                    valid_cities = [item for item in cities if isinstance(item, dict)]
                    if valid_cities:
                        chosen = random.choice(valid_cities)
                        city_name = chosen.get("name")
                        if isinstance(city_name, str):
                            strategies.append((
                                FieldCondition(key=f"{fname}.country.cities[].name", match=MatchValue(value=city_name)),
                                f'{fname}.country.cities[].name == "{city_name}"'
                            ))
                        population = chosen.get("population")
                        if isinstance(population, (int, float)) and not isinstance(population, bool):
                            population = float(population)
                            strategies.append((
                                FieldCondition(key=f"{fname}.country.cities[].population", range=Range(gte=population)),
                                f"{fname}.country.cities[].population >= {population}"
                            ))
                        sightseeing = chosen.get("sightseeing")
                        if isinstance(sightseeing, list):
                            valid_sights = [str(v) for v in sightseeing if v is not None]
                            if valid_sights:
                                sight = random.choice(valid_sights)
                                strategies.append((
                                    FieldCondition(key=f"{fname}.country.cities[].sightseeing", match=MatchValue(value=sight)),
                                    f'{fname}.country.cities[].sightseeing == "{sight}"'
                                ))
            if strategies and random.random() < 0.65:
                return random.choice(strategies)

        current = json_obj
        path_str = fname  # Qdrant 使用点号路径: fname.key1.key2

        max_depth = random.randint(1, 12)
        depth = 0

        while depth < max_depth:
            if isinstance(current, dict) and len(current) > 0:
                k = random.choice(list(current.keys()))
                path_str += f".{k}"
                current = current[k]
                depth += 1
                if self._has_valid_content(current) and random.random() < 0.2:
                    break
            elif isinstance(current, (list, np.ndarray)) and len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                path_str += f"[{idx}]"
                current = current[idx]
                depth += 1
            else:
                break

        # 转换 numpy 标量
        if hasattr(current, "item"): current = current.item()

        # Case 1: Null
        if current is None:
            return self._qdrant_is_null_filter(path_str), f"{path_str} is null"

        # Case 2: 复杂类型停在中间 → is not null
        if isinstance(current, (dict, list, np.ndarray)):
            return self._qdrant_not_empty_filter(path_str), f"{path_str} is not empty"

        # Case 3: 标量 → 生成边界条件
        if isinstance(current, bool):
            return FieldCondition(key=path_str, match=MatchValue(value=current)), f"{path_str} == {current}"

        elif isinstance(current, int):
            return self._gen_pqs_boundary_int(path_str, current)

        elif isinstance(current, float):
            return self._gen_pqs_boundary_float(path_str, current)

        elif isinstance(current, str):
            return self._gen_pqs_boundary_str(path_str, current)

        return self._qdrant_not_null_filter(fname), self._not_nullish_expr(fname)

    def _gen_pqs_boundary_int(self, path, val):
        """JSON 深层 INT 边界条件（10种策略）"""
        val = self._clip_int64(val)
        strategies = []
        # 1. 精确匹配
        strategies.append((FieldCondition(key=path, match=MatchValue(value=val)), f"{path} == {val}"))
        # 2. 闭区间
        strategies.append((FieldCondition(key=path, range=Range(gte=val, lte=val)), f"{path} >= {val} AND <= {val}"))
        # 3. 开区间 ±1
        low_1 = self._offset_int64(val, -1)
        high_1 = self._offset_int64(val, 1)
        strategies.append((FieldCondition(key=path, range=Range(gt=low_1, lt=high_1)), f"{path} > {low_1} AND < {high_1}"))
        # 4. NOT反面 (val < boundary → NOT gte boundary)
        boundary = self._offset_int64(val, random.randint(1, 100))
        strategies.append((Filter(must_not=[FieldCondition(key=path, range=Range(gte=boundary))]), f"NOT ({path} >= {boundary})"))
        # 5. NOT反面 (val > boundary → NOT lte boundary)
        boundary_low = self._offset_int64(val, -random.randint(1, 100))
        strategies.append((Filter(must_not=[FieldCondition(key=path, range=Range(lte=boundary_low))]), f"NOT ({path} <= {boundary_low})"))
        # 6. MatchAny 包含噪声
        dummies = [self._offset_int64(val, random.randint(100000, 200000)) for _ in range(3)]
        candidates = dummies + [val]
        random.shuffle(candidates)
        strategies.append((FieldCondition(key=path, match=MatchAny(any=candidates)), f"{path} in {candidates}"))
        # 7. 零值特殊: 如果 val >= 0，用 gte=0
        if val >= 0:
            strategies.append((FieldCondition(key=path, range=Range(gte=0)), f"{path} >= 0"))
        # 8. 符号逻辑: 如果 val > 0，NOT lte=-1
        if val > 0:
            strategies.append((Filter(must_not=[FieldCondition(key=path, range=Range(lte=-1))]), f"NOT ({path} <= -1)"))
        return random.choice(strategies)

    def _gen_pqs_boundary_float(self, path, val):
        """JSON 深层 FLOAT 边界条件（7种策略）"""
        val = float(val)
        eps = 1e-5
        strategies = []
        # 1. 窄 epsilon
        strategies.append((FieldCondition(key=path, range=Range(gt=val-eps, lt=val+eps)), f"{path} ~= {val}"))
        # 2. 宽 epsilon
        wide = random.uniform(0.01, 1.0)
        strategies.append((FieldCondition(key=path, range=Range(gte=val-wide, lte=val+wide)), f"{path} in [{val-wide:.4f}, {val+wide:.4f}]"))
        # 3. 单边 gte
        strategies.append((FieldCondition(key=path, range=Range(gte=val-eps)), f"{path} >= {val-eps}"))
        # 4. NOT反面
        boundary = val - random.uniform(1.0, 100.0)
        strategies.append((Filter(must_not=[FieldCondition(key=path, range=Range(lte=boundary))]), f"NOT ({path} <= {boundary:.4f})"))
        # 5. 符号逻辑
        if val > 0:
            strategies.append((FieldCondition(key=path, range=Range(gt=0)), f"{path} > 0"))
        return random.choice(strategies)

    def _gen_pqs_boundary_str(self, path, val):
        """JSON 深层 STRING 边界条件（4种策略）"""
        val = str(val)
        strategies = []
        # 1. 精确匹配
        strategies.append((FieldCondition(key=path, match=MatchValue(value=val)), f'{path} == "{val}"'))
        # 2. MatchAny
        dummies = [self._random_string(8) for _ in range(3)]
        candidates = dummies + [val]
        random.shuffle(candidates)
        strategies.append((FieldCondition(key=path, match=MatchAny(any=candidates)), f'{path} in {candidates}'))
        # 3. MatchExcept 排除假值
        fake = [self._random_string(12) for _ in range(3)]
        strategies.append((FieldCondition(key=path, match=MatchExcept(**{"except": fake})), f'{path} not in {fake}'))
        return random.choice(strategies)

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


# --- 6. Dynamic Data Operations & Schema Evolution Helpers ---

def _to_native_point_id(value):
    """Normalize pandas/numpy integer scalars to a plain Python int for Qdrant client models."""
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _to_native_point_ids(values):
    return [_to_native_point_id(v) for v in values]


def query_qdrant_rows(qm, id_list, limit=5):
    """查询 Qdrant 中指定 ID 的完整 payload 数据，用于调试对比"""
    ids = _to_native_point_ids(list(id_list)[:limit])
    if not ids:
        return {}
    try:
        result = qm.retrieve(
            collection_name=COLLECTION_NAME,
            ids=ids,
            with_payload=True,
            with_vectors=False
        )
        return {p.id: p.payload for p in result}
    except Exception:
        return {}


def format_row_for_display(row_dict):
    """清理行数据用于显示：移除向量、转换 numpy 类型、NaN 转 None"""
    def to_native(v):
        if v is pd.NA:
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, float) and (v != v):  # NaN check
            return None
        if isinstance(v, dict):
            return {kk: to_native(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [to_native(x) for x in v]
        if isinstance(v, np.ndarray):
            return [to_native(x) for x in v.tolist()]
        return v

    clean = {}
    for k, v in row_dict.items():
        if k == "vector":
            continue
        clean[k] = to_native(v)
    return clean


def dataframe_row_to_dict(df, row_idx):
    """
    按列读取单行，避免 `df.iloc[idx].to_dict()` 将 Int64 行值隐式提升为 float。
    """
    out = {}
    for col in df.columns:
        out[col] = df[col].iloc[row_idx]
    return out


def print_diff_evidence(dm, qm, diff_ids, label, expr_str, file_log):
    """
    详细比对 Pandas Oracle 与 Qdrant 实际数据，高亮 NULL 不一致。
    用于调试 mismatch 根因。
    """
    if not diff_ids:
        return

    evidence_ids = sample_ids(diff_ids, 5)
    file_log(f"\n  === {label} Evidence (up to {len(evidence_ids)} IDs) ===")
    file_log(f"  Collection: {COLLECTION_NAME}")

    # 获取 Qdrant 实际数据
    qdrant_data = query_qdrant_rows(qm, evidence_ids, limit=5)
    retrieved_ids = sample_ids(qdrant_data.keys(), 5)
    if retrieved_ids and retrieved_ids != evidence_ids[:len(retrieved_ids)]:
        file_log(f"  Retrieved IDs: {retrieved_ids}")

    for sid in evidence_ids:
        file_log(f"  --- ID: {sid} ---")

        # Pandas 侧数据
        pdf_row = dm.df[dm.df["id"] == sid]
        if pdf_row.empty:
            file_log(f"    [Pandas] NOT FOUND (deleted?)")
        else:
            row_data = format_row_for_display(dataframe_row_to_dict(pdf_row, 0))
            null_fields = [k for k, v in row_data.items() if v is None and k != "id"]
            file_log(f"    [Pandas] {json.dumps(row_data, default=str, ensure_ascii=False)}")
            if null_fields:
                file_log(f"    [Pandas NULL fields] {null_fields}")

        # Qdrant 侧数据
        if sid in qdrant_data:
            qd = qdrant_data[sid]
            q_null_fields = [k for k, v in qd.items() if v is None]
            file_log(f"    [Qdrant] {json.dumps(qd, default=str, ensure_ascii=False)}")
            if q_null_fields:
                file_log(f"    [Qdrant NULL fields] {q_null_fields}")

            # NULL 不一致检测
            if not pdf_row.empty:
                row_data = format_row_for_display(dataframe_row_to_dict(pdf_row, 0))
                for field in row_data:
                    if field == "id":
                        continue
                    pd_val = row_data.get(field)
                    qd_val = qd.get(field)
                    pd_is_null = pd_val is None
                    qd_is_null = qd_val is None
                    if pd_is_null != qd_is_null:
                        file_log(f"    ⚠️ NULL MISMATCH on '{field}': Pandas={pd_val} vs Qdrant={qd_val}")
        else:
            file_log(f"    [Qdrant] NOT FOUND")


def _do_schema_evolution(dm, qm, file_log):
    """
    Schema Evolution 操作：随机添加新字段并回填部分数据。
    可选：对未回填的行显式设置字段为 None（严格 IsNull 语义）。
    默认关闭，改用演进字段 IsEmpty 语义以提升动态模式稳定性。
    """
    try:
        field_config = dm.evolve_schema_add_field()
        if field_config is None:
            file_log("[SchemaEvolution] Max evolved fields reached, skipping.")
            return False

        field_name = field_config["name"]
        field_type = get_type_name(field_config["type"])
        file_log(f"[SchemaEvolution] Adding field '{field_name}' (type={field_type})...")
        print(f"\n🧩 Schema Evolution: Adding '{field_name}' ({field_type})")

        # 回填部分数据
        backfill_data = dm.backfill_evolved_field(field_config)
        backfilled_ids = set()
        if backfill_data:
            fill_count = qm.backfill_field_data(field_name, backfill_data)
            backfilled_ids = {row_id for _, row_id, _ in backfill_data}
            file_log(f"[SchemaEvolution] Backfilled {fill_count}/{len(backfill_data)} rows for '{field_name}'")
            print(f"   ✔️ Backfilled {fill_count}/{len(backfill_data)} rows")
        else:
            file_log(f"[SchemaEvolution] No rows to backfill for '{field_name}'")

        # 对未回填行默认不做显式 NULL 回写：
        # 演进字段在查询层统一按 is_empty 处理，不依赖“显式 null vs 字段缺失”差异。
        # 逐条 upsert 全量点会显著拖慢动态模式，并可能引入额外一致性噪音。
        if SCHEMA_EVOLUTION_EXPLICIT_NULL_SYNC:
            all_ids = set(int(x) for x in dm.df["id"].tolist())
            unfilled_ids = list(all_ids - backfilled_ids)
            if unfilled_ids:
                null_count = qm.set_null_for_points(field_name, unfilled_ids)
                unfilled_indices = dm.df.index[dm.df["id"].isin(unfilled_ids)].tolist()
                dm.mark_payload_fields_present(unfilled_indices, field_name)
                file_log(
                    f"[SchemaEvolution] Explicitly set NULL for {null_count}/{len(unfilled_ids)} rows on '{field_name}'"
                )
        else:
            file_log(
                f"[SchemaEvolution] Explicit NULL sync skipped for '{field_name}' "
                f"(using is_empty semantics for evolved fields)."
            )

        # 创建索引
        qm.create_evolved_field_index(field_config)
        file_log(f"[SchemaEvolution] Index created for '{field_name}'")
        dm.normalize_dataframe_types()

        return True
    except Exception as e:
        file_log(f"[SchemaEvolution] Failed: {e}")
        print(f"   ⚠️ Schema Evolution failed: {e}")
        return False


PAYLOAD_MUTATION_FIELD_TYPES = {
    FieldType.INT,
    FieldType.FLOAT,
    FieldType.BOOL,
    FieldType.STRING,
    FieldType.UUID,
    FieldType.DATETIME,
    FieldType.GEO,
    FieldType.JSON,
    FieldType.ARRAY_INT,
    FieldType.ARRAY_STR,
    FieldType.ARRAY_FLOAT,
    FieldType.ARRAY_OBJECT,
}


def _payload_json_preview(payload, max_len=500):
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = repr(payload)
    if len(text) > max_len:
        return text[:max_len] + "...<truncated>"
    return text


def _pick_payload_mutation_targets(dm, max_points=4):
    if dm.df is None or dm.df.empty:
        return [], []
    count = min(random.randint(1, max_points), len(dm.df))
    indices = random.sample(list(dm.df.index), count)
    ids = _to_native_point_ids(dm.df.loc[indices, "id"].tolist())
    return indices, ids


def _pick_payload_mutation_fields(dm, max_fields=3):
    candidates = [
        field for field in dm.schema_config
        if field["type"] in PAYLOAD_MUTATION_FIELD_TYPES
    ]
    if not candidates:
        return []
    count = min(random.randint(1, max_fields), len(candidates))
    return random.sample(candidates, count)


def _build_payload_mutation_payload(dm, fields):
    payload = {}
    for field in fields:
        field_name = field["name"]
        field_type = field["type"]
        value = dm.generate_field_value(field)
        value = dm._normalize_scalar_for_field(value, field_type)
        if value is None:
            # set_payload(None) does not reliably create explicit null in Qdrant;
            # explicit-null behavior remains covered by upsert/schema-evolution
            # paths and dedicated operator tests.
            continue
        payload[field_name] = value
    return payload


def _apply_payload_values_to_df(dm, indices, payload):
    if not payload:
        return
    for idx in indices:
        for field_name, value in payload.items():
            if field_name in dm.df.columns:
                dm.df.at[int(idx), field_name] = copy.deepcopy(value)
    dm.mark_payload_fields_present(indices, list(payload.keys()))
    dm.normalize_dataframe_types()


def _do_payload_mutation_op(dm, qm, file_log):
    """
    执行一次 payload mutation 子操作，并同步 pandas oracle 状态。

    该路径只在 --payload-mutations 下开启。它依赖 payload_presence
    区分“字段缺失”和“显式 null”，避免 delete/clear/overwrite 后把
    Qdrant 的 missing-key 语义误当成 null。
    """
    indices, point_ids = _pick_payload_mutation_targets(dm)
    if not point_ids:
        return False

    op = random.choices(
        ["set_payload", "delete_payload", "overwrite_payload", "clear_payload"],
        weights=[0.45, 0.25, 0.20, 0.10],
        k=1,
    )[0]

    fields = _pick_payload_mutation_fields(dm)
    if op != "clear_payload" and not fields:
        return False

    field_names = [field["name"] for field in fields]
    try:
        if op == "set_payload":
            payload = _build_payload_mutation_payload(dm, fields)
            if not payload:
                return False
            qm.set_payload(
                collection_name=COLLECTION_NAME,
                payload=payload,
                points=point_ids,
                wait=True,
            )
            _apply_payload_values_to_df(dm, indices, payload)
            file_log(
                f"[DynamicPayload] set_payload ids={point_ids} "
                f"fields={list(payload.keys())} payload={_payload_json_preview(payload)}"
            )
            return True

        if op == "delete_payload":
            qm.delete_payload(
                collection_name=COLLECTION_NAME,
                keys=field_names,
                points=point_ids,
                wait=True,
            )
            dm.mark_payload_fields_missing(indices, field_names)
            file_log(f"[DynamicPayload] delete_payload ids={point_ids} fields={field_names}")
            return True

        if op == "overwrite_payload":
            payload = _build_payload_mutation_payload(dm, fields)
            if not payload:
                return False
            qm.overwrite_payload(
                collection_name=COLLECTION_NAME,
                payload=payload,
                points=point_ids,
                wait=True,
            )
            dm.mark_payload_fields_missing(indices, dm.schema_field_names())
            _apply_payload_values_to_df(dm, indices, payload)
            file_log(
                f"[DynamicPayload] overwrite_payload ids={point_ids} "
                f"fields={list(payload.keys())} payload={_payload_json_preview(payload)}"
            )
            return True

        qm.clear_payload(
            collection_name=COLLECTION_NAME,
            points_selector=point_ids,
            wait=True,
        )
        dm.mark_payload_fields_missing(indices, dm.schema_field_names())
        file_log(f"[DynamicPayload] clear_payload ids={point_ids}")
        return True
    except Exception as e:
        file_log(f"[DynamicPayload] {op} failed: {e}")
        return False


def _do_dynamic_op(dm, qm, file_log, delete_min_rows=100):
    """
    执行一次随机动态数据操作（insert/delete/upsert）。
    同时同步 dm.df 和 dm.vectors。
    """
    if PAYLOAD_MUTATIONS_ENABLED and random.random() < 0.40:
        if _do_payload_mutation_op(dm, qm, file_log):
            return

    op = random.choices(["insert", "delete", "upsert"], weights=[0.4, 0.4, 0.2], k=1)[0]
    batch_count = random.randint(1, 5)

    def _sync_explicit_nulls(rows, row_ids):
        """对本批次写入行中的 None 字段显式 set_payload(None)，避免字段缺失语义漂移。"""
        if not rows or not row_ids:
            return

        null_field_to_ids = {}
        for row in rows:
            rid = row.get("id")
            if rid is None:
                continue
            for k, v in row.items():
                if k == "id":
                    continue
                if v is None:
                    null_field_to_ids.setdefault(k, []).append(int(rid))

        for field_name, ids in null_field_to_ids.items():
            synced = qm.set_null_for_points(field_name, ids)
            file_log(
                f"[Dynamic] Explicit NULL sync field='{field_name}': {synced}/{len(ids)} rows"
            )

    if op == "insert":
        new_rows = []
        new_vecs = []
        for _ in range(batch_count):
            row = dm.normalize_row_for_storage(dm.generate_single_row())
            vec = dm.generate_single_vector()
            new_rows.append(row)
            new_vecs.append(vec)
        try:
            points = []
            for row, vec in zip(new_rows, new_vecs):
                payload = {k: v for k, v in row.items() if k != "id"}
                points.append(PointStruct(id=row["id"], vector=vec, payload=payload))
            qm.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
            # 先以 object 保留 Python 原生整数，再按 schema 显式收敛 dtype
            new_df = pd.DataFrame(new_rows, dtype=object)
            new_df = dm.coerce_dataframe_types(new_df)
            new_df = new_df.dropna(axis=1, how='all')  # 避免全NA列的 FutureWarning
            dm.append_payload_presence_rows(len(new_rows), present=True)
            dm.df = pd.concat([dm.df, new_df], ignore_index=True)
            dm.vectors = np.vstack([dm.vectors, np.array(new_vecs)])
            dm.normalize_dataframe_types()
            inserted_ids = [r["id"] for r in new_rows]
            _sync_explicit_nulls(new_rows, inserted_ids)
            file_log(f"[Dynamic] Inserted {len(new_rows)} rows: ids={inserted_ids}")
        except Exception as e:
            file_log(f"[Dynamic] Insert failed: {e}")

    elif op == "delete":
        if len(dm.df) > delete_min_rows:
            del_count = min(batch_count, len(dm.df) - delete_min_rows)
            del_ids = _to_native_point_ids(random.sample(dm.df["id"].tolist(), del_count))
            try:
                qm.client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=del_ids),
                    wait=True
                )
                idx = dm.df[dm.df["id"].isin(del_ids)].index.to_numpy()
                dm.drop_payload_presence_indices(idx)
                dm.df = dm.df.drop(idx).reset_index(drop=True)
                dm.vectors = np.delete(dm.vectors, idx, axis=0)
                dm.normalize_dataframe_types()
                file_log(f"[Dynamic] Deleted {len(del_ids)} rows: ids={del_ids}")

                # 删除后验证：确认被删 ID 真的不再存在
                for did in del_ids:
                    try:
                        verify_res = qm.retrieve(
                            collection_name=COLLECTION_NAME,
                            ids=[did],
                            with_payload=False,
                            with_vectors=False
                        )
                        if verify_res:
                            file_log(f"[DELETE_WARN] ID {did} still exists in Qdrant after delete!")
                    except Exception as ve:
                        file_log(f"[DELETE_VERIFY] ID {did} verification error: {ve}")
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
            row = dm.normalize_row_for_storage(dm.generate_single_row(id_override=target_id))
            vec = dm.generate_single_vector()
            upsert_rows.append(row)
            upsert_vecs.append(vec)
        try:
            points = []
            for row, vec in zip(upsert_rows, upsert_vecs):
                payload = {k: v for k, v in row.items() if k != "id"}
                points.append(PointStruct(id=row["id"], vector=vec, payload=payload))
            qm.upsert(
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
                    dm.mark_payload_fields_present([idx], dm.schema_field_names())
                    dm.vectors[idx] = vec
                else:
                    new_rows.append(row)
                    new_vectors.append(vec)
            if new_rows:
                # 避免 INT+NULL 在构造临时 DataFrame 时先漂移到 float
                new_df = pd.DataFrame(new_rows, dtype=object)
                new_df = dm.coerce_dataframe_types(new_df)
                new_df = new_df.dropna(axis=1, how='all')  # 避免全NA列的 FutureWarning
                dm.append_payload_presence_rows(len(new_rows), present=True)
                dm.df = pd.concat([dm.df, new_df], ignore_index=True)
                dm.vectors = np.vstack([dm.vectors, np.array(new_vectors)])
            dm.normalize_dataframe_types()
            upsert_ids = [r["id"] for r in upsert_rows]
            _sync_explicit_nulls(upsert_rows, upsert_ids)
            file_log(f"[Dynamic] Upserted {len(upsert_rows)} rows: ids={upsert_ids}")

            # Upsert 后数据同步验证：回查 Qdrant 确认数据一致
            for row in upsert_rows:
                rid = _to_native_point_id(row["id"])
                try:
                    qdrant_res = qm.retrieve(
                        collection_name=COLLECTION_NAME,
                        ids=[rid],
                        with_payload=True,
                        with_vectors=False
                    )
                    if not qdrant_res:
                        file_log(f"[SYNC_WARN] ID {rid} not found in Qdrant after upsert")
                    else:
                        # 检查一个关键字段是否同步
                        payload = qdrant_res[0].payload
                        pandas_row = dm.df[dm.df["id"] == rid]
                        if not pandas_row.empty:
                            # 检查 meta_json.price 是否一致
                            if "meta_json" in row and isinstance(row["meta_json"], dict):
                                pd_price = row["meta_json"].get("price")
                                qd_json = payload.get("meta_json", {})
                                qd_price = qd_json.get("price") if isinstance(qd_json, dict) else None
                                if pd_price != qd_price:
                                    file_log(f"[SYNC_WARN] ID {rid} price mismatch: Pandas={pd_price} vs Qdrant={qd_price}")
                except Exception as ve:
                    file_log(f"[SYNC_VERIFY] ID {rid} verification error: {ve}")
        except Exception as e:
            file_log(f"[Dynamic] Upsert failed: {e}")


# --- 7. Main Execution Functions ---

def _format_transport_line(qm):
    info = qm.get_transport_info()
    return (
        f"Transport: host={info.get('host')} "
        f"rest={info.get('rest_port')} grpc={info.get('grpc_port')} "
        f"requested_prefer_grpc={info.get('requested_prefer_grpc')} "
        f"backend={info.get('backend_class')} "
        f"backend_prefer_grpc={info.get('backend_prefer_grpc')}"
    )


def _transport_warning_line():
    if not PREFER_GRPC:
        return (
            "TransportWarning: REST path has known inconsistency on range comparisons "
            "at +/-float32_max; use --prefer-grpc if this affects mismatch analysis."
        )
    return None


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

    collection_name = set_collection_name("oracle", current_seed)

    # 1. 初始化随机化配置（种子已设置，确保可复现）
    _init_vector_check_config()

    dm = DataManager(data_seed=current_seed)
    dm.generate_schema()
    dm.generate_data()

    qm = QdrantManager()
    qm.connect()
    qm.reset_collection(dm.schema_config)
    qm.insert(dm)
    if PREFER_GRPC:
        synced = qm.sync_float_fields_from_storage(dm)
        print(f"🔧 Synced FLOAT payloads from Qdrant for oracle alignment: {synced} rows.")

    # 2. 日志设置
    log_filename = make_mode_log_path("oracle", current_seed)
    reproduce_command = build_reproduce_command("oracle", current_seed, rounds, enable_dynamic_ops)
    manifest_path = write_run_manifest(
        log_filename,
        "oracle",
        current_seed,
        rounds,
        reproduce_command,
        enable_dynamic_ops,
        schema_config=dm.schema_config,
    )
    print(f"\n📝 详细日志将写入: {display_path(log_filename)}")
    print(f"   Manifest: {display_path(manifest_path)}")
    print(f"   Collection: {collection_name}")
    print(f"   🔑 如需复现此次测试，运行: {reproduce_command}")
    print(f"🚀 开始测试 (控制台仅显示失败案例)...")

    qg = OracleQueryGenerator(dm)
    failed_cases = []
    known_bug_cases = []
    total_test = rounds

    with open(log_filename, "w", encoding="utf-8") as f:
        def file_log(msg):
            f.write(msg + "\n")
            f.flush()

        def sample_rows(id_set, limit=5):
            if not id_set:
                return []
            subset = dm.df[dm.df["id"].isin(list(id_set))].sort_values("id", kind="stable")
            rows = []
            for ridx in range(min(limit, len(subset))):
                raw = dataframe_row_to_dict(subset, ridx)
                rows.append(format_row_for_display(raw))
            return rows

        file_log(f"Start Testing: {total_test} rounds | Seed: {current_seed}")
        file_log(f"Reproduce: {reproduce_command}")
        file_log(f"Manifest: {manifest_path}")
        file_log(f"Collection: {collection_name}")
        file_log(_format_transport_line(qm))
        tw = _transport_warning_line()
        if tw:
            file_log(tw)
        file_log("=" * 50)

        for i in range(total_test):
            print(f"\r⏳ Running Test {i+1}/{total_test}...", end="", flush=True)

            # --- Payload 索引随机重建 (每50轮) ---
            if i > 0 and i % 50 == 0:
                rebuilt = qm.rebuild_payload_indexes(dm.schema_config)
                if rebuilt:
                    file_log(f"[IndexRebuild] Rebuilt indexes for: {rebuilt}")

            # --- 动态插入/删除/Upsert ---
            if enable_dynamic_ops and i > 0 and i % 10 == 0:
                _do_dynamic_op(dm, qm, file_log)

            # --- Schema Evolution (每30轮) ---
            if enable_dynamic_ops and i > 0 and i % 30 == 0:
                _do_schema_evolution(dm, qm, file_log)

            # 生成查询
            depth = random.randint(1, 15)
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
                scroll_result, scroll_plan = execute_scroll_query(
                    qm,
                    COLLECTION_NAME,
                    filter_obj,
                    len(dm.df),
                    with_payload=False,
                    with_vectors=False,
                )

                actual_ids = set()
                for point in scroll_result:
                    actual_ids.add(point.id)

                cost = (time.time() - start_t) * 1000

                file_log(f"  Pandas: {len(expected_ids)} | Qdrant: {len(actual_ids)} | Time: {cost:.1f}ms")
                file_log(f"  Scroll: {format_scroll_plan(scroll_plan)}")
                scroll_matches_oracle = expected_ids == actual_ids

                if should_run_count_cross_check(scroll_plan):
                    try:
                        count_result = qm.count(
                            collection_name=COLLECTION_NAME,
                            count_filter=filter_obj,
                            exact=True,
                        )
                        count_hits = int(count_result.count)
                        file_log(
                            f"  CountCheck: count={count_hits} | scroll={len(actual_ids)} | pandas={len(expected_ids)}"
                        )
                        if count_hits != len(actual_ids):
                            detail = (
                                f"Count/scroll mismatch: count={count_hits}, scroll={len(actual_ids)}, "
                                f"pandas={len(expected_ids)}"
                            )
                            print(f"\n⚠️ [Test {i}] COUNT/SCROLL MISMATCH!")
                            print(f"   Expr: {expr_str}")
                            print(f"   Detail: {detail}")
                            print(f"   Scroll: {format_scroll_plan(scroll_plan)}")
                            file_log(f"  -> COUNT/SCROLL MISMATCH! {detail}")
                            failed_cases.append(
                                {
                                    "id": i,
                                    "expr": expr_str,
                                    "detail": detail,
                                    "seed": current_seed,
                                }
                            )
                        elif scroll_matches_oracle and count_hits != len(expected_ids):
                            detail = (
                                f"Count/oracle mismatch: count={count_hits}, scroll={len(actual_ids)}, "
                                f"pandas={len(expected_ids)}"
                            )
                            print(f"\n⚠️ [Test {i}] COUNT/ORACLE MISMATCH!")
                            print(f"   Expr: {expr_str}")
                            print(f"   Detail: {detail}")
                            file_log(f"  -> COUNT/ORACLE MISMATCH! {detail}")
                            failed_cases.append(
                                {
                                    "id": i,
                                    "expr": expr_str,
                                    "detail": detail,
                                    "seed": current_seed,
                                }
                            )
                        elif not scroll_matches_oracle and count_hits == len(actual_ids):
                            file_log("  CountCheck: aligns_with_scroll (not an extra independent failure)")
                        else:
                            file_log("  CountCheck: PASS")
                    except Exception as e:
                        file_log(f"  CountCheck: ERROR {e}")

                if scroll_matches_oracle:
                    file_log("  -> MATCH")
                else:
                    known_boundary_bug = expr_mentions_known_unstable_boundary(expr_str)
                    print(f"\n❌ [Test {i}] MISMATCH!")
                    print(f"   Expr: {expr_str}")
                    print(f"   Expected: {len(expected_ids)} vs Actual: {len(actual_ids)}")
                    print(f"   Scroll: {format_scroll_plan(scroll_plan)}")

                    missing = expected_ids - actual_ids
                    extra = actual_ids - expected_ids
                    diff_msg = ""
                    if missing:
                        diff_msg += f"Missing IDs: {sample_ids(missing, 10)} "
                    if extra:
                        diff_msg += f"Extra IDs: {sample_ids(extra, 10)}"

                    print(f"   Diff: {diff_msg}")
                    print(f"   🔑 复现此bug: {reproduce_command}\n")

                    file_log(f"  -> MISMATCH! {diff_msg}")
                    file_log(f"  -> REPRODUCTION SEED: {current_seed}")
                    if known_boundary_bug:
                        known_msg = (
                            "  -> KNOWN-BOUNDARY-BUG-CANDIDATE: expression mentions unstable int64 extreme "
                            "or float32-max literals; Qdrant v1.17.0 range comparisons near these limits are known unreliable."
                        )
                        print("   Note: known numeric extreme boundary bug candidate; do not classify this as logic-only evidence.")
                        file_log(known_msg)

                    if missing:
                        missing_rows = sample_rows(missing)
                        file_log(f"  Missing rows sample: {missing_rows}")
                        print("   Missing rows (sample):")
                        for r in missing_rows[:3]:
                            print(f"     {r}")
                        print_diff_evidence(dm, qm, missing, "MISSING (in Pandas but not Qdrant)", expr_str, file_log)
                    if extra:
                        extra_rows = sample_rows(extra)
                        file_log(f"  Extra rows sample: {extra_rows}")
                        print("   Extra rows (sample):")
                        for r in extra_rows[:3]:
                            print(f"     {r}")
                        print_diff_evidence(dm, qm, extra, "EXTRA (in Qdrant but not Pandas)", expr_str, file_log)

                    case_payload = {
                        "id": i,
                        "expr": expr_str,
                        "detail": f"Exp: {len(expected_ids)} vs Act: {len(actual_ids)}. {diff_msg}",
                        "seed": current_seed
                    }
                    if known_boundary_bug:
                        known_bug_cases.append(case_payload)
                    else:
                        failed_cases.append(case_payload)

                # 向量 + 标量联合校验
                if expected_ids and random.random() < VECTOR_CHECK_RATIO:
                    try:
                        q_idx = random.randint(0, len(dm.vectors) - 1)
                        q_vec = dm.vectors[q_idx].tolist()
                        search_k = min(VECTOR_TOPK, len(dm.df))

                        # search_params 随机化：测试不同 HNSW 参数和精确搜索模式
                        use_exact = random.random() < 0.1  # 10% 使用精确搜索
                        hnsw_ef = random.choice([64, 128, 256, 512])
                        sp = SearchParams(hnsw_ef=hnsw_ef, exact=use_exact)

                        search_res = qm.query_points(
                            collection_name=COLLECTION_NAME,
                            query=q_vec,
                            query_filter=filter_obj,
                            limit=search_k,
                            with_payload=False,
                            search_params=sp
                        )

                        returned_ids = set(p.id for p in search_res.points)

                        if not returned_ids:
                            file_log("  VectorCheck: PASS (no ANN hits)")
                        elif returned_ids.issubset(expected_ids):
                            file_log(f"  VectorCheck: PASS ({len(returned_ids)} hits subset of scalar filter)")
                        else:
                            extra_vec = returned_ids - expected_ids
                            extra_sample = sample_ids(extra_vec, 3)
                            if expr_mentions_known_unstable_boundary(expr_str):
                                file_log(
                                    "  VectorCheck: KNOWN-BOUNDARY-BUG-CANDIDATE "
                                    f"extra ids {extra_sample}..."
                                )
                                print(
                                    f"\n⚠️ [Test {i}] Vector subset extras {extra_sample} "
                                    "over known numeric boundary expression; downgraded to known-boundary candidate."
                                )
                            else:
                                file_log(f"  VectorCheck: FAIL extra ids {extra_sample}...")
                                print(f"\n⚠️ [Test {i}] Vector subset violated: extras {extra_sample}...")
                                failed_cases.append({
                                    "id": i,
                                    "expr": expr_str,
                                    "detail": f"VectorCheck extras: {extra_sample}..."
                                })
                    except Exception as e:
                        file_log(f"  VectorCheck: ERROR {e}")

            except Exception as e:
                if is_qdrant_format_error(e):
                    file_log(f"  -> SKIP INVALID FILTER (Qdrant 400 format): {e}")
                    continue
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
        if known_bug_cases:
            print(f"⚠️  另有 {len(known_bug_cases)} 个已知数值极值边界候选案例，未计入新的逻辑失败。")
        print(f"📄 详细记录请查看: {display_path(log_filename)}")
    else:
        print(f"🚫 发现 {len(failed_cases)} 个失败案例！(已保存至日志)")
        if known_bug_cases:
            print(f"⚠️  另有 {len(known_bug_cases)} 个已知数值极值边界候选案例，单独记录，不计入上述失败数。")
        print("-" * 60)
        for case in failed_cases:
            print(f"🔴 Case {case['id']}:")
            print(f"   Expr: {case['expr']}")
            print(f"   Issue: {case['detail']}")
            if 'seed' in case:
                print(f"   🔑 复现: {reproduce_command}")
            print("-" * 30)
        for case in known_bug_cases:
            print(f"🟡 KnownBoundary Case {case['id']}:")
            print(f"   Expr: {case['expr']}")
            print(f"   Issue: {case['detail']}")
            if 'seed' in case:
                print(f"   🔑 复现: {reproduce_command}")
            print("-" * 30)
        print(f"📄 请查看 {display_path(log_filename)} 获取完整上下文。")
        print(f"🔑 全局复现命令: {reproduce_command}")


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

    collection_name = set_collection_name("equiv", seed)

    # 种子已设置，初始化可复现的随机配置
    _init_vector_check_config()

    # 初始化数据后再写 manifest，确保 schema_config 可复现地记录下来。
    dm = DataManager(data_seed=seed)
    dm.generate_schema()
    dm.generate_data()

    log_filename = make_mode_log_path("equiv", seed)
    reproduce_command = build_reproduce_command("equiv", seed, rounds, enable_dynamic_ops)
    manifest_path = write_run_manifest(
        log_filename,
        "equiv",
        seed,
        rounds,
        reproduce_command,
        enable_dynamic_ops,
        schema_config=dm.schema_config,
    )
    print("\n" + "="*60)
    print(f"👯 启动 Equivalence Mode (等价性测试)")
    print(f"   原理: Query(A) 应该等于 Query(Transformation(A))")
    print(f"   Seed: {seed}")
    print(f"📄 日志: {display_path(log_filename)}")
    print(f"   Manifest: {display_path(manifest_path)}")
    print(f"   Collection: {collection_name}")
    print(f"   Reproduce: {reproduce_command}")
    print("="*60)

    qm = QdrantManager()
    qm.connect()
    qm.reset_collection(dm.schema_config)
    qm.insert(dm)
    if PREFER_GRPC:
        synced = qm.sync_float_fields_from_storage(dm)
        print(f"🔧 Synced FLOAT payloads from Qdrant for oracle alignment: {synced} rows.")

    qg = EquivalenceQueryGenerator(dm)
    failed_cases = []
    known_bug_cases = []

    with open(log_filename, "w", encoding="utf-8") as f:
        def file_log(msg):
            f.write(msg + "\n")
            f.flush()

        file_log(f"Equivalence Test Started | Seed: {seed}")
        file_log(f"Reproduce: {reproduce_command}")
        file_log(f"Manifest: {manifest_path}")
        file_log(f"Collection: {collection_name}")
        file_log(_format_transport_line(qm))
        tw = _transport_warning_line()
        if tw:
            file_log(tw)

        for i in range(rounds):
            print(f"\r⚖️  Test {i+1}/{rounds}...", end="", flush=True)

            # --- Payload 索引随机重建 (每50轮) ---
            if i > 0 and i % 50 == 0:
                rebuilt = qm.rebuild_payload_indexes(dm.schema_config)
                if rebuilt:
                    file_log(f"[IndexRebuild] Rebuilt indexes for: {rebuilt}")

            # --- 动态操作（20% 概率）---
            if enable_dynamic_ops and i > 0 and random.random() < 0.2:
                _do_dynamic_op(dm, qm, file_log)

            # --- Schema Evolution (每30轮) ---
            if enable_dynamic_ops and i > 0 and i % 30 == 0:
                _do_schema_evolution(dm, qm, file_log)

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
                base_res, base_scroll_plan = execute_scroll_query(
                    qm,
                    COLLECTION_NAME,
                    base_filter,
                    len(dm.df),
                    with_payload=False,
                    with_vectors=False,
                )
                base_ids = set(p.id for p in base_res)
            except Exception as e:
                if is_qdrant_format_error(e):
                    file_log(f"[Test {i}] Base Query Skipped (invalid filter): {e}")
                    continue
                file_log(f"[Test {i}] Base Query Failed: {e}")
                continue

            log_header = f"[Test {i}] Base: {base_expr} (Hits: {len(base_ids)})"
            file_log(f"\n{log_header}")
            file_log(f"  BaseScroll: {format_scroll_plan(base_scroll_plan)}")

            # 执行并对比所有变体
            for m in mutations:
                m_type = m["type"]
                m_filter = m["filter"]
                m_expr = m["expr"]

                try:
                    mut_res, mut_scroll_plan = execute_scroll_query(
                        qm,
                        COLLECTION_NAME,
                        m_filter,
                        len(dm.df),
                        with_payload=False,
                        with_vectors=False,
                    )
                    mut_ids = set(p.id for p in mut_res)

                    if base_ids == mut_ids:
                        file_log(f"  ✅ [{m_type}] Match | Scroll: {format_scroll_plan(mut_scroll_plan)}")
                    else:
                        known_boundary_bug = (
                            expr_mentions_known_unstable_boundary(base_expr)
                            or expr_mentions_known_unstable_boundary(m_expr)
                        )
                        print(f"\n\n❌ EQUIVALENCE FAILURE [Test {i}]")
                        print(f"   Type: {m_type}")
                        print(f"   Base Expr: {base_expr}")
                        print(f"   Mut  Expr: {m_expr}")
                        print(f"   Base Hits: {len(base_ids)} | Mut Hits: {len(mut_ids)}")
                        print(f"   Base Scroll: {format_scroll_plan(base_scroll_plan)}")
                        print(f"   Mut  Scroll: {format_scroll_plan(mut_scroll_plan)}")

                        missing = base_ids - mut_ids
                        extra = mut_ids - base_ids
                        diff_msg = ""
                        if missing:
                            diff_msg += f"Mutation Missing: {sample_ids(missing, 5)} "
                        if extra:
                            diff_msg += f"Mutation Extra: {sample_ids(extra, 5)} "
                        print(f"   Diff: {diff_msg}")
                        print("-" * 50)

                        file_log(f"  ❌ [{m_type}] FAIL! {diff_msg}")
                        file_log(f"     Mut Expr: {m_expr}")
                        if known_boundary_bug:
                            file_log(
                                "     KNOWN-BOUNDARY-BUG-CANDIDATE: base or mutation expression mentions unstable "
                                "int64 extreme or float32-max literals; do not classify this as a pure logic rewrite failure."
                            )

                        # 详细证据对比
                        if missing:
                            print_diff_evidence(dm, qm, missing, f"Mutation MISSING [{m_type}]", m_expr, file_log)
                        if extra:
                            print_diff_evidence(dm, qm, extra, f"Mutation EXTRA [{m_type}]", m_expr, file_log)

                        case_payload = {
                            "id": i,
                            "type": m_type,
                            "base": base_expr,
                            "mut": m_expr,
                            "diff": diff_msg
                        }
                        if known_boundary_bug:
                            known_bug_cases.append(case_payload)
                        else:
                            failed_cases.append(case_payload)

                except Exception as e:
                    if is_qdrant_format_error(e):
                        file_log(f"  ⚠️ [{m_type}] SKIP INVALID FILTER: {e}")
                        continue
                    print(f"\n\n⚠️ Mutation Crash [Test {i}]")
                    print(f"   Type: {m_type}")
                    print(f"   Expr: {m_expr}")
                    print(f"   Error: {e}")
                    file_log(f"  ⚠️ [{m_type}] CRASH: {e}")

    print("\n" + "="*60)
    if failed_cases:
        print(f"🚫 发现 {len(failed_cases)} 个等价性错误！请检查日志。")
        if known_bug_cases:
            print(f"⚠️  另有 {len(known_bug_cases)} 个已知数值极值边界候选案例，未计入新的等价性错误。")
    else:
        print(f"✅ 所有等价性测试通过。")
        if known_bug_cases:
            print(f"⚠️  另有 {len(known_bug_cases)} 个已知数值极值边界候选案例，未计入新的等价性错误。")
    print(f"📄 详细日志请查看: {display_path(log_filename)}")


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
    else:
        seed = random.randint(0, 2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        print(f"\n🎲 PQS模式随机生成种子: {seed}")

    collection_name = set_collection_name("pqs", seed)

    # 种子已设置，初始化可复现的随机配置
    _init_vector_check_config()

    # 初始化数据后再写 manifest，确保 schema_config 可复现地记录下来。
    dm = DataManager(data_seed=seed)
    dm.generate_schema()
    dm.generate_data()

    log_filename = make_mode_log_path("pqs", seed)
    reproduce_command = build_reproduce_command("pqs", seed, rounds, enable_dynamic_ops)
    manifest_path = write_run_manifest(
        log_filename,
        "pqs",
        seed,
        rounds,
        reproduce_command,
        enable_dynamic_ops,
        schema_config=dm.schema_config,
    )
    print("\n" + "="*60)
    print(f"🚀 启动 PQS (Pivot Query Synthesis) 模式测试")
    print(f"   Seed: {seed}")
    print(f"📄 详细日志将写入: {display_path(log_filename)}")
    print(f"   Manifest: {display_path(manifest_path)}")
    print(f"   Collection: {collection_name}")
    print(f"   Reproduce: {reproduce_command}")
    print("="*60)

    qm = QdrantManager()
    qm.connect()
    qm.reset_collection(dm.schema_config)
    qm.insert(dm)
    if PREFER_GRPC:
        synced = qm.sync_float_fields_from_storage(dm)
        print(f"🔧 Synced FLOAT payloads from Qdrant for oracle alignment: {synced} rows.")

    pqs_gen = PQSQueryGenerator(dm)
    errors = []
    successful_tests = 0
    skipped_tests = 0

    def safe_format_row(row_mapping):
        row_dict = dict(row_mapping)
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
        file_log(f"Reproduce: {reproduce_command}")
        file_log(f"Manifest: {manifest_path}")
        file_log(f"Collection: {collection_name}")
        file_log(_format_transport_line(qm))
        tw = _transport_warning_line()
        if tw:
            file_log(tw)
        file_log("=" * 80)

        for i in range(rounds):
            # --- Payload 索引随机重建 (每50轮) ---
            if i > 0 and i % 50 == 0:
                rebuilt = qm.rebuild_payload_indexes(dm.schema_config)
                if rebuilt:
                    file_log(f"[IndexRebuild] Rebuilt indexes for: {rebuilt}")

            # --- 动态操作（20% 概率）---
            if enable_dynamic_ops and i > 0 and random.random() < 0.2:
                _do_dynamic_op(dm, qm, file_log)

            # --- Schema Evolution (每30轮) ---
            if enable_dynamic_ops and i > 0 and i % 30 == 0:
                _do_schema_evolution(dm, qm, file_log)

            random_idx = random.randint(0, len(dm.df) - 1)
            pivot_row = dataframe_row_to_dict(dm.df, random_idx)
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
                res, scroll_plan = execute_scroll_query(
                    qm,
                    COLLECTION_NAME,
                    filter_obj,
                    len(dm.df),
                    with_payload=False,
                    with_vectors=False,
                )
                cost = (time.time() - start_t) * 1000
                found_ids = set(p.id for p in res)

                successful_tests += 1
                if pivot_id in found_ids:
                    file_log(
                        f"  -> PASS | Found: {len(found_ids)} hits | Time: {cost:.2f}ms | Scroll: {format_scroll_plan(scroll_plan)}"
                    )
                else:
                    safe_row = safe_format_row(pivot_row)
                    json_data = safe_row.get("meta_json", {})
                    scalar_data = {k: v for k, v in safe_row.items() if k != "meta_json"}

                    print(f"\n\n❌ PQS ERROR DETECTED [Round {i}]")
                    print(f"   Target ID: {pivot_id}")
                    print(f"   Expression: {expr}")
                    print(f"   Found Count: {len(found_ids)} (Target NOT found)")
                    print(f"   Scroll: {format_scroll_plan(scroll_plan)}")
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

                    # 获取 Qdrant 侧数据对比
                    print_diff_evidence(dm, qm, {pivot_id}, "PQS MISSING PIVOT", expr, file_log)

                    errors.append({"id": pivot_id, "expr": expr})

            except Exception as e:
                if is_qdrant_format_error(e):
                    skipped_tests += 1
                    file_log(f"  -> SKIP INVALID FILTER (Qdrant 400 format): {e}")
                    continue
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
        print(f"📄 详细数据已记录至日志: {display_path(log_filename)}")
    if not errors:
        print(f"📄 详细日志请查看: {display_path(log_filename)}")


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
    else:
        seed = random.randint(0, 2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        print(f"\n🎲 GroupBy模式随机生成种子: {seed}")

    collection_name = set_collection_name("group", seed)

    # 种子已设置，初始化可复现的随机配置
    _init_vector_check_config()

    # 初始化数据后再写 manifest，确保 schema_config 可复现地记录下来。
    dm = DataManager(data_seed=seed)
    dm.generate_schema()
    dm.generate_data()

    log_filename = make_mode_log_path("group", seed)
    reproduce_command = build_reproduce_command("group", seed, rounds, enable_dynamic_ops)
    manifest_path = write_run_manifest(
        log_filename,
        "group",
        seed,
        rounds,
        reproduce_command,
        enable_dynamic_ops,
        schema_config=dm.schema_config,
    )
    print("\n" + "="*60)
    print(f"📊 启动 GroupBy 逻辑专项测试")
    print(f"   Seed: {seed}")
    print(f"   日志: {display_path(log_filename)}")
    print(f"   Manifest: {display_path(manifest_path)}")
    print(f"   Collection: {collection_name}")
    print(f"   Reproduce: {reproduce_command}")
    print("="*60)

    qm = QdrantManager()
    qm.connect()
    qm.reset_collection(dm.schema_config)
    qm.insert(dm)
    if PREFER_GRPC:
        synced = qm.sync_float_fields_from_storage(dm)
        print(f"🔧 Synced FLOAT payloads from Qdrant for oracle alignment: {synced} rows.")

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

        file_log(f"GroupBy Test Started | Rounds: {rounds} | Seed: {seed}")
        file_log(f"Reproduce: {reproduce_command}")
        file_log(f"Manifest: {manifest_path}")
        file_log(f"Collection: {collection_name}")
        file_log(_format_transport_line(qm))
        tw = _transport_warning_line()
        if tw:
            file_log(tw)

        for i in range(rounds):
            print(f"\r📊 GroupBy Test {i+1}/{rounds}...", end="", flush=True)

            # --- Payload 索引随机重建 (每50轮) ---
            if i > 0 and i % 50 == 0:
                rebuilt = qm.rebuild_payload_indexes(dm.schema_config)
                if rebuilt:
                    file_log(f"[IndexRebuild] Rebuilt indexes for: {rebuilt}")

            # --- 动态操作（20% 概率）---
            if enable_dynamic_ops and i > 0 and random.random() < 0.2:
                _do_dynamic_op(dm, qm, file_log)

            # --- Schema Evolution (每30轮) ---
            if enable_dynamic_ops and i > 0 and i % 30 == 0:
                _do_schema_evolution(dm, qm, file_log)

            group_field = random.choice(potential_group_fields)
            group_size = random.randint(1, 5)
            limit_groups = random.randint(2, 20)

            q_idx = random.randint(0, len(dm.vectors) - 1)
            q_vec = dm.vectors[q_idx].tolist()

            try:
                # Qdrant 的 group_by 搜索 - 使用 search_groups
                search_res = qm.query_points_groups(
                    collection_name=COLLECTION_NAME,
                    query=q_vec,
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
    print(f"📄 详细日志请查看: {display_path(log_filename)}")


# ---------------------------------------------------------------------------
#  Main Entry Point
# ---------------------------------------------------------------------------

MODE_DISPATCH = {
    "oracle": lambda a: run(rounds=a.rounds, seed=a.seed,
                            enable_dynamic_ops=a.dynamic or a.payload_mutations),
    "equiv":  lambda a: run_equivalence_mode(rounds=a.rounds, seed=a.seed,
                                             enable_dynamic_ops=a.dynamic or a.payload_mutations),
    "pqs":    lambda a: run_pqs_mode(rounds=a.pqs_rounds, seed=a.seed,
                                     enable_dynamic_ops=a.dynamic or a.payload_mutations),
    "group":  lambda a: run_group_test(rounds=a.rounds, seed=a.seed,
                                       enable_dynamic_ops=a.dynamic or a.payload_mutations),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，返回 Namespace。"""
    p = argparse.ArgumentParser(
        prog="qdrant_fuzz_oracle",
        description="Qdrant Fuzz Oracle — 动态模糊测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python qdrant_fuzz_oracle.py                         # 默认 Oracle 模式
  python qdrant_fuzz_oracle.py --seed 12345            # 固定种子
  python qdrant_fuzz_oracle.py --equiv --rounds 500    # 等价性模式
  python qdrant_fuzz_oracle.py --pqs --pqs-rounds 200  # PQS 模式
  python qdrant_fuzz_oracle.py --dynamic --rounds 300  # 动态操作模式
  python qdrant_fuzz_oracle.py --dynamic --payload-mutations # 动态 payload 更新/删除/覆盖
  python qdrant_fuzz_oracle.py --dynamic --evo-null-sync # 演进字段严格 IsNull 语义
  python qdrant_fuzz_oracle.py --scroll-mode paged     # 强制使用分页 scroll 路径
  python qdrant_fuzz_oracle.py --prefer-grpc            # 使用 gRPC 查询路径
  python qdrant_fuzz_oracle.py --read-consistency all --write-ordering strong
        """,
    )

    # ---- 模式 (互斥) ----
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--oracle", action="store_const", dest="mode",
                     const="oracle", help="Oracle 差分测试模式 (默认)")
    grp.add_argument("--equiv",  action="store_const", dest="mode",
                     const="equiv",  help="等价性变换测试模式")
    grp.add_argument("--pqs",    action="store_const", dest="mode",
                     const="pqs",    help="PQS (必中查询) 测试模式")
    grp.add_argument("--group",  action="store_const", dest="mode",
                     const="group",  help="GroupBy 测试模式")
    p.set_defaults(mode="oracle")

    # ---- 轮次 ----
    p.add_argument("--seed",       type=int, default=None,
                   help="固定随机种子以复现测试 (默认: 随机)")
    p.add_argument("--rounds",     type=int, default=1000,
                   help="主测试轮数 (默认: 1000)")
    p.add_argument("--pqs-rounds", type=int, default=1000, dest="pqs_rounds",
                   help="PQS 测试轮数 (默认: 1000)")
    p.add_argument("-N", "--rows", type=int, default=N, dest="rows",
                   help=f"初始数据量 (默认: {N})")
    p.add_argument("--dim", type=int, default=DIM,
                   help=f"向量维度 (默认: {DIM})")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, dest="batch_size",
                   help=f"批量 upsert 大小 (默认: {BATCH_SIZE})")
    p.add_argument("--sleep-interval", type=float, default=SLEEP_INTERVAL, dest="sleep_interval",
                   help=f"批量 upsert 间隔秒数 (默认: {SLEEP_INTERVAL})")
    p.add_argument(
        "--scroll-mode",
        type=str,
        choices=["bulk", "paged", "mixed"],
        default=SCROLL_MODE,
        help="scroll 执行策略：bulk=单次全量，paged=分页拉取，mixed=两者混合 (默认: mixed)",
    )
    p.add_argument(
        "--distance",
        type=str,
        choices=["random", *DISTANCE_NAME_MAP.keys()],
        default="random",
        help="向量距离类型；默认 random，会在每次运行中随机选取一个",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_LOG_DIR,
        help=f"日志输出目录 (默认: {DEFAULT_LOG_DIR})",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="可复现实验 run id；用于稳定日志/manifest 文件名",
    )
    p.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="显式指定 collection 名；默认按 mode/seed/run-id 稳定生成",
    )

    # ---- 连接参数 ----
    p.add_argument("--host", type=str, default=HOST,
                   help=f"Qdrant 主机地址 (默认: {HOST})")
    p.add_argument("--port", type=int, default=PORT,
                   help=f"Qdrant REST 端口 (默认: {PORT})")
    p.add_argument("--grpc-port", type=int, default=GRPC_PORT, dest="grpc_port",
                   help=f"Qdrant gRPC 端口 (默认: {GRPC_PORT})")
    p.add_argument("--prefer-grpc", action="store_true",
                   help="优先使用 gRPC 传输（仍保留 REST 端口以兼容部分 API）")
    p.add_argument(
        "--read-consistency",
        type=str,
        choices=["all", "majority", "quorum", "1", "random"],
        default="all",
        help="Qdrant 读一致性策略 (默认: all；可选 random)",
    )
    p.add_argument(
        "--write-ordering",
        type=str,
        choices=["weak", "medium", "strong", "random"],
        default="strong",
        help="Qdrant 写 ordering 策略 (默认: strong；可选 random)",
    )

    # ---- 功能开关 ----
    p.add_argument("--dynamic", action="store_true",
                   help="开启动态数据操作 (insert/delete/upsert)")
    p.add_argument(
        "--payload-mutations",
        action="store_true",
        help="开启动态 payload mutation 子操作（set/delete/overwrite/clear，隐含 --dynamic）",
    )
    p.add_argument("--chaos",   action="store_true",
                   help="开启混淆模式 (默认概率 10%%)")
    p.add_argument("--chaos-rate", type=float, default=0.0, dest="chaos_rate",
                   metavar="RATE",
                   help="自定义混淆概率 0.0-1.0 (隐含 --chaos)")
    p.add_argument(
        "--evo-null-sync",
        action="store_true",
        help="Schema Evolution 时为未回填行显式写入 NULL（演进字段采用严格 IsNull 语义）",
    )
    p.add_argument(
        "--include-known-int64-boundaries",
        action="store_true",
        help="允许通用 fuzz 注入已知不稳定的 int64 极值边界（默认关闭，以减少误报）",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global CHAOS_RATE
    global SCHEMA_EVOLUTION_EXPLICIT_NULL_SYNC
    global PAYLOAD_MUTATIONS_ENABLED
    global HOST, PORT, GRPC_PORT, PREFER_GRPC
    global READ_CONSISTENCY, WRITE_ORDERING
    global INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES
    global N, DIM, BATCH_SIZE, SLEEP_INTERVAL, LOG_DIR
    global FORCED_DISTANCE_TYPE
    global _global_id_counter
    global RUN_ID, DISTANCE_ARG, SCROLL_MODE
    global FORCED_COLLECTION_NAME

    args = parse_args(argv)
    if args.payload_mutations:
        args.dynamic = True

    # 处理 chaos 相关逻辑
    if args.chaos_rate > 0:
        CHAOS_RATE = args.chaos_rate
    elif args.chaos:
        CHAOS_RATE = 0.1

    SCHEMA_EVOLUTION_EXPLICIT_NULL_SYNC = bool(args.evo_null_sync)
    PAYLOAD_MUTATIONS_ENABLED = bool(args.payload_mutations)
    INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES = bool(args.include_known_int64_boundaries)
    HOST = args.host
    PORT = int(args.port)
    GRPC_PORT = int(args.grpc_port)
    PREFER_GRPC = bool(args.prefer_grpc)
    READ_CONSISTENCY = 1 if args.read_consistency == "1" else args.read_consistency
    WRITE_ORDERING = args.write_ordering
    N = max(1, int(args.rows))
    DIM = max(1, int(args.dim))
    BATCH_SIZE = max(1, int(args.batch_size))
    SLEEP_INTERVAL = max(0.0, float(args.sleep_interval))
    SCROLL_MODE = args.scroll_mode
    LOG_DIR = os.path.abspath(os.path.expanduser(args.log_dir))
    os.makedirs(LOG_DIR, exist_ok=True)
    FORCED_DISTANCE_TYPE = None if args.distance == "random" else DISTANCE_NAME_MAP[args.distance]
    DISTANCE_ARG = args.distance
    RUN_ID = args.run_id
    FORCED_COLLECTION_NAME = args.collection_name
    _global_id_counter = max(N * 10, 50000)

    # Banner
    print("=" * 80)
    print("🚀 Qdrant Fuzz Oracle 启动")
    print(f"   模式:       {args.mode}")
    print(f"   主测试轮数: {args.rounds}")
    print(f"   PQS测试轮数: {args.pqs_rounds}")
    print(f"   随机种子:   {args.seed or '(随机)'}")
    print(f"   动态操作:   {'开启' if args.dynamic else '关闭'}")
    print(f"   Payload变异:{'开启' if PAYLOAD_MUTATIONS_ENABLED else '关闭'}")
    print(f"   混淆概率:   {CHAOS_RATE}")
    print(f"   连接:       {HOST}:{PORT} (grpc:{GRPC_PORT}, prefer_grpc={PREFER_GRPC})")
    print(f"   读一致性:   {READ_CONSISTENCY}")
    print(f"   写排序:     {WRITE_ORDERING}")
    print(f"   数据规模:   rows={N}, dim={DIM}, batch={BATCH_SIZE}, sleep={SLEEP_INTERVAL}")
    print(f"   Scroll策略: {SCROLL_MODE}")
    print(f"   距离度量:   {args.distance}")
    print(f"   日志目录:   {display_path(LOG_DIR)}")
    if RUN_ID:
        print(f"   Run ID:     {RUN_ID}")
    if FORCED_COLLECTION_NAME:
        print(f"   Collection: {FORCED_COLLECTION_NAME}")
    if not PREFER_GRPC:
        print("   ⚠️  说明: REST 路径对 ±float32_max 的范围比较存在已知不一致；mismatch 日志会标记边界候选")
    print(
        "   演进空值语义: "
        + ("严格 IsNull（显式NULL回填）" if SCHEMA_EVOLUTION_EXPLICIT_NULL_SYNC else "IsEmpty（缺失/null/[]统一）")
    )
    if not INCLUDE_KNOWN_UNSTABLE_INT64_BOUNDARIES:
        print("   已知边界保护: 通用 fuzz 默认跳过已知不稳定的 int64 极值附近字面量")
    print("   已知边界标记: ±float32 最大有限值保留参与 fuzz，相关 mismatch 会标记为边界候选")
    if args.dynamic:
        effective_rounds = args.pqs_rounds if args.mode == "pqs" else args.rounds
        if effective_rounds < 30:
            print("   ⚠️  提示: 当前 dynamic 轮次 < 30，Schema Evolution 不会触发，missing-field 语义覆盖有限")
        else:
            print("   Missing覆盖: 将在第 30 轮触发 Schema Evolution，覆盖真正 missing field 路径")
    print("=" * 80)

    # Dispatch
    MODE_DISPATCH[args.mode](args)


if __name__ == "__main__":
    main()
