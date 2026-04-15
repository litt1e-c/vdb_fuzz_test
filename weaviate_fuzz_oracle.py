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
10. Randomized vector index (HNSW/FLAT/Dynamic/HFresh) + distance metric
11. Consistency level cycling (ONE/QUORUM/ALL)
12. Dynamic ops: insert/delete/update mid-test
13. Tokenization.FIELD for TEXT/TEXT_ARRAY: case-sensitive, no stopword filtering
14. Dynamic row tracking: Weaviate inverted-index bug detection & classification
15. Schema evolution: add property mid-test + full backfill before enabling queries
16. Scalar index mutation: destructive property-index drop with query-space adaptation
17. Engine-first shadow sync: fetch/retry/resync after dynamic writes before mutating oracle
18. Aggregate oracle mode: count/min/max/sum/mean/median/mode/top-occurrences over scalar and array probe fields
19. REST filter oracle mode: validated `where` JSON via batch delete dry-run against the REST parser path

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
import shlex
import requests
from dataclasses import dataclass
from datetime import timedelta, timezone
import weaviate
from weaviate.classes.config import (
    Configure, Property, DataType, Reconfigure,
    VectorDistances, Tokenization,
)
from weaviate.classes.query import Filter, GeoCoordinate, GroupBy, Sort, MetadataQuery
from weaviate.classes.data import DataObject
from weaviate.classes.config import ConsistencyLevel
from weaviate.collections.classes.aggregate import Metrics, GroupByAggregate

# --- Null 过滤默认启用 ---
ENABLE_NULL_FILTER = True

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 8080
CLASS_NAME = "FuzzOracleV2"
N = 10000
DIM = 128
BATCH_SIZE = 200
SLEEP_INTERVAL = 0.02
DEFAULT_QUERY_MAXIMUM_RESULTS = 10000


def _env_positive_int(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _env_probability(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, min(1.0, value))


QUERY_MAXIMUM_RESULTS = _env_positive_int(
    "WEAVIATE_FUZZ_QUERY_MAXIMUM_RESULTS",
    _env_positive_int("QUERY_MAXIMUM_RESULTS", DEFAULT_QUERY_MAXIMUM_RESULTS),
)
QUERY_PAGE_SIZE = _env_positive_int("WEAVIATE_FUZZ_QUERY_PAGE_SIZE", 1000)
FILTER_PARTITION_SIZE = _env_positive_int("WEAVIATE_FUZZ_FILTER_PARTITION_SIZE", DEFAULT_QUERY_MAXIMUM_RESULTS)
FILTER_ID_BATCH_SIZE = _env_positive_int("WEAVIATE_FUZZ_FILTER_ID_BATCH_SIZE", 512)
AGGREGATE_FILTER_MIN_DEPTH = _env_positive_int("WEAVIATE_FUZZ_AGGREGATE_FILTER_MIN_DEPTH", 2)
AGGREGATE_FILTER_MAX_DEPTH = max(
    AGGREGATE_FILTER_MIN_DEPTH,
    _env_positive_int("WEAVIATE_FUZZ_AGGREGATE_FILTER_MAX_DEPTH", 4),
)
REST_FILTER_MIN_DEPTH = _env_positive_int("WEAVIATE_FUZZ_REST_FILTER_MIN_DEPTH", 3)
REST_FILTER_MAX_DEPTH = max(
    REST_FILTER_MIN_DEPTH,
    _env_positive_int("WEAVIATE_FUZZ_REST_FILTER_MAX_DEPTH", 6),
)
REST_FILTER_MAX_COMPARE_RESULTS = _env_positive_int("WEAVIATE_FUZZ_REST_FILTER_MAX_COMPARE_RESULTS", 20000)
ORACLE_REST_CROSSCHECK_RATE = _env_probability("WEAVIATE_FUZZ_ORACLE_REST_CROSSCHECK_RATE", 0.35)
QUERY_MAXIMUM_RESULTS_RE = re.compile(r"QUERY_MAXIMUM_RESULTS '(\d+)'")

FUZZ_PROFILE_DEFAULT = "default"
FUZZ_PROFILE_INVERTED = "inverted"
ALL_FUZZ_PROFILES = [FUZZ_PROFILE_DEFAULT, FUZZ_PROFILE_INVERTED]
FUZZ_PROFILE = FUZZ_PROFILE_DEFAULT


def set_fuzz_profile(profile):
    global FUZZ_PROFILE
    normalized = str(profile or FUZZ_PROFILE_DEFAULT).strip().lower()
    if normalized not in ALL_FUZZ_PROFILES:
        raise ValueError(f"Unsupported fuzz profile: {profile}")
    FUZZ_PROFILE = normalized


def get_fuzz_profile():
    return FUZZ_PROFILE


def is_inverted_profile():
    return FUZZ_PROFILE == FUZZ_PROFILE_INVERTED

# --- 延迟初始化的随机变量 (在 run() 中种子设置后初始化) ---
VECTOR_CHECK_RATIO = None
VECTOR_TOPK = None
BOUNDARY_INJECTION_RATE = None

# 标量优先：过滤表达式更偏向标量字段，减少因数组/OBJECT 语义导致的噪声
SCALAR_QUERY_PRIORITY = 0.85

# Runtime maintenance knobs.  These intentionally default to conservative
# intervals because schema/property index changes are heavier than row updates.
ENABLE_SCHEMA_EVOLUTION = True
SCHEMA_EVOLUTION_INTERVAL = 30
MAX_EVOLVED_PROPERTIES = 4

ENABLE_SCALAR_INDEX_MUTATION = True
SCALAR_INDEX_MUTATION_INTERVAL = 45
MAX_SCALAR_INDEX_DROPS = 4

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


def stable_unique_values(values):
    return list(dict.fromkeys(values))


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


@dataclass
class PagedFetchResult:
    objects: list
    exhausted: bool


def get_query_maximum_results():
    return max(1, int(QUERY_MAXIMUM_RESULTS or DEFAULT_QUERY_MAXIMUM_RESULTS))


def get_query_page_size():
    return max(1, min(int(QUERY_PAGE_SIZE or 1), get_query_maximum_results()))


def get_filter_partition_size():
    return max(1, min(int(FILTER_PARTITION_SIZE or get_query_maximum_results()), get_query_maximum_results()))


def get_filter_id_batch_size():
    return max(1, min(int(FILTER_ID_BATCH_SIZE or 1), get_query_maximum_results()))


def update_query_limits_from_error(error):
    global QUERY_MAXIMUM_RESULTS, QUERY_PAGE_SIZE
    match = QUERY_MAXIMUM_RESULTS_RE.search(str(error))
    if not match:
        return None
    detected_limit = max(1, int(match.group(1)))
    QUERY_MAXIMUM_RESULTS = detected_limit
    QUERY_PAGE_SIZE = max(1, min(int(QUERY_PAGE_SIZE or detected_limit), detected_limit))
    return detected_limit


def is_query_maximum_results_error(error):
    err = str(error)
    return ("QUERY_MAXIMUM_RESULTS" in err) or ("query maximum results exceeded" in err.lower())


def capped_result_count(count, exhausted):
    return str(count) if exhausted else f"{count}+"


def combine_filters(*filters):
    active = [flt for flt in filters if flt is not None]
    if not active:
        return None
    combined = active[0]
    for flt in active[1:]:
        combined = combined & flt
    return combined


def build_id_subset_filter(ids):
    ids = [str(object_id) for object_id in ids]
    if not ids:
        return None
    if len(ids) == 1:
        return Filter.by_id().equal(ids[0])
    return Filter.by_id().contains_any(ids)


def _fetch_objects_single_page(collection, *, limit, **query_kwargs):
    response = collection.query.fetch_objects(limit=limit, **query_kwargs)
    objects = list(getattr(response, "objects", None) or [])
    return PagedFetchResult(objects, len(objects) < limit)


def fetch_objects_cursor(collection, *, max_objects=None, page_size=None, **query_kwargs):
    query_kwargs = dict(query_kwargs)
    explicit_limit = query_kwargs.pop("limit", None)
    query_kwargs.pop("offset", None)
    query_kwargs.pop("after", None)
    if query_kwargs.get("filters") is not None:
        raise ValueError("Cursor pagination does not support filtered fetches; use partitioned fetching instead")

    if explicit_limit is not None:
        max_objects = explicit_limit if max_objects is None else min(max_objects, explicit_limit)
    if max_objects is not None and max_objects <= 0:
        return PagedFetchResult([], True)

    page_size = max(1, int(page_size or get_query_page_size()))
    objects = []
    after = None
    seen_after = set()

    while True:
        remaining = None if max_objects is None else max_objects - len(objects)
        if remaining is not None and remaining <= 0:
            return PagedFetchResult(objects, False)

        page_limit = page_size if remaining is None else min(page_size, remaining)
        if page_limit <= 0:
            return PagedFetchResult(objects, False)

        call_kwargs = dict(query_kwargs)
        call_kwargs["limit"] = page_limit
        if after is not None:
            call_kwargs["after"] = after

        try:
            response = collection.query.fetch_objects(**call_kwargs)
        except Exception as exc:
            detected_limit = update_query_limits_from_error(exc)
            if detected_limit is not None and page_limit > detected_limit:
                page_size = max(1, min(page_size, detected_limit))
                continue
            raise

        page_objects = list(getattr(response, "objects", None) or [])
        if not page_objects:
            return PagedFetchResult(objects, True)

        objects.extend(page_objects)
        if len(page_objects) < page_limit:
            return PagedFetchResult(objects, True)

        next_after = str(page_objects[-1].uuid)
        if next_after == after or next_after in seen_after:
            raise RuntimeError("Cursor pagination stalled while fetching Weaviate objects")
        seen_after.add(next_after)
        after = next_after


def fetch_objects_by_row_num_partitions(collection, dm, *, filters=None, partition_size=None, max_objects=None, **query_kwargs):
    query_kwargs = dict(query_kwargs)
    query_kwargs.pop("limit", None)
    query_kwargs.pop("offset", None)
    query_kwargs.pop("after", None)

    if dm is None or dm.df is None or dm.df.empty or "row_num" not in dm.df.columns:
        return PagedFetchResult([], True)

    partition_size = max(1, min(int(partition_size or get_filter_partition_size()), get_query_maximum_results()))
    row_nums = pd.to_numeric(dm.df["row_num"], errors="coerce").dropna()
    if row_nums.empty:
        return PagedFetchResult([], True)

    floor = int(row_nums.min())
    ceiling = int(row_nums.max())
    start = (floor // partition_size) * partition_size

    while True:
        objects = []
        seen_ids = set()
        try:
            for lower in range(start, ceiling + 1, partition_size):
                upper = lower + partition_size
                range_filter = (
                    Filter.by_property("row_num").greater_or_equal(int(lower))
                    & Filter.by_property("row_num").less_than(int(upper))
                )
                batch_filter = combine_filters(filters, range_filter)
                response = _fetch_objects_single_page(
                    collection,
                    filters=batch_filter,
                    limit=partition_size,
                    **query_kwargs,
                )
                for obj in response.objects:
                    object_id = str(obj.uuid)
                    if object_id in seen_ids:
                        continue
                    seen_ids.add(object_id)
                    objects.append(obj)
                if max_objects is not None and len(objects) >= max_objects:
                    return PagedFetchResult(objects[:max_objects], False)
            return PagedFetchResult(objects, True)
        except Exception as exc:
            detected_limit = update_query_limits_from_error(exc)
            if detected_limit is not None and partition_size > detected_limit:
                partition_size = detected_limit
                start = (floor // partition_size) * partition_size
                continue
            raise


def fetch_objects_by_id_batches(collection, dm, *, filters=None, batch_size=None, max_objects=None, **query_kwargs):
    query_kwargs = dict(query_kwargs)
    query_kwargs.pop("limit", None)
    query_kwargs.pop("offset", None)
    query_kwargs.pop("after", None)

    if dm is None or dm.df is None or dm.df.empty:
        return PagedFetchResult([], True)

    batch_size = max(1, min(int(batch_size or get_filter_id_batch_size()), get_query_maximum_results()))
    all_ids = [str(object_id) for object_id in dm.df["id"].tolist()]
    objects = []
    seen_ids = set()

    for start in range(0, len(all_ids), batch_size):
        id_batch = all_ids[start:start + batch_size]
        id_filter = build_id_subset_filter(id_batch)
        batch_filter = combine_filters(filters, id_filter)
        response = _fetch_objects_single_page(
            collection,
            filters=batch_filter,
            limit=len(id_batch),
            **query_kwargs,
        )
        for obj in response.objects:
            object_id = str(obj.uuid)
            if object_id in seen_ids:
                continue
            seen_ids.add(object_id)
            objects.append(obj)
        if max_objects is not None and len(objects) >= max_objects:
            return PagedFetchResult(objects[:max_objects], False)
    return PagedFetchResult(objects, True)


def fetch_objects_resilient(collection, *, dm=None, filters=None, max_objects=None, page_size=None, **query_kwargs):
    query_kwargs = dict(query_kwargs)
    if filters is None:
        return fetch_objects_cursor(
            collection,
            max_objects=max_objects,
            page_size=page_size,
            **query_kwargs,
        )

    if max_objects is None:
        max_objects = len(dm.df) + 1 if dm is not None and getattr(dm, "df", None) is not None else get_query_maximum_results()
    if max_objects <= 0:
        return PagedFetchResult([], True)

    single_limit = max(1, min(int(max_objects), get_query_maximum_results()))
    try:
        result = _fetch_objects_single_page(
            collection,
            filters=filters,
            limit=single_limit,
            **query_kwargs,
        )
        if result.exhausted or max_objects <= get_query_maximum_results():
            return result
    except Exception as exc:
        if not is_query_maximum_results_error(exc):
            raise
        update_query_limits_from_error(exc)

    try:
        return fetch_objects_by_row_num_partitions(
            collection,
            dm,
            filters=filters,
            max_objects=max_objects,
            **query_kwargs,
        )
    except Exception:
        return fetch_objects_by_id_batches(
            collection,
            dm,
            filters=filters,
            max_objects=max_objects,
            **query_kwargs,
        )


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
ALL_VECTOR_INDEX_TYPES = ["hnsw", "flat", "dynamic", "hfresh"]
ALL_DISTANCE_METRICS = [VectorDistances.COSINE, VectorDistances.L2_SQUARED, VectorDistances.DOT]
VECTOR_INDEX_TYPE = None
DISTANCE_METRIC = None

# 一致性等级
ALL_CONSISTENCY_LEVELS = [ConsistencyLevel.ONE, ConsistencyLevel.QUORUM, ConsistencyLevel.ALL]
DEFAULT_CONSISTENCY_LEVEL = ConsistencyLevel.ALL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.getenv("WEAVIATE_FUZZ_LOG_DIR", os.path.join(SCRIPT_DIR, "weaviate_log"))
LOG_RUN_SUFFIX = os.getenv("WEAVIATE_FUZZ_LOG_SUFFIX", "")


def ensure_log_dir():
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    return DEFAULT_LOG_DIR


def make_log_path(filename: str) -> str:
    return os.path.join(ensure_log_dir(), filename)


def set_log_dir(path: str | None):
    global DEFAULT_LOG_DIR
    if path:
        DEFAULT_LOG_DIR = os.path.abspath(os.path.expanduser(path))


def set_log_suffix(value: str | None):
    global LOG_RUN_SUFFIX
    LOG_RUN_SUFFIX = str(value or "")


def sanitize_log_token(value) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "x"


def stable_log_filename(prefix: str, seed, rounds, mode: str, *extra_tokens) -> str:
    tokens = [
        prefix,
        f"mode-{mode}",
        f"seed-{seed}",
        f"rounds-{rounds}",
        f"N-{N}",
        f"profile-{get_fuzz_profile()}",
    ]
    tokens.extend(str(token) for token in extra_tokens if token not in (None, ""))
    if LOG_RUN_SUFFIX:
        tokens.append(f"tag-{LOG_RUN_SUFFIX}")
    return "__".join(sanitize_log_token(token) for token in tokens) + ".log"


def display_path(path: str) -> str:
    try:
        return os.path.relpath(path, start=os.getcwd())
    except Exception:
        return path


@dataclass(frozen=True)
class GraphQLEnum:
    value: str


def graphql_with_enum_operators(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "operator" and isinstance(item, str):
                out[key] = GraphQLEnum(item)
            else:
                out[key] = graphql_with_enum_operators(item)
        return out
    if isinstance(value, list):
        return [graphql_with_enum_operators(item) for item in value]
    if isinstance(value, tuple):
        return [graphql_with_enum_operators(item) for item in value]
    return value


def graphql_input_literal(value):
    if isinstance(value, GraphQLEnum):
        return str(value.value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"Non-finite float not supported in GraphQL literal: {value!r}")
        text = repr(float(value))
        return text if "." in text or "e" in text.lower() else f"{text}.0"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(graphql_input_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        ordered_keys = [
            key for key in ("path", "operator", "operands") if key in value
        ] + sorted(key for key in value.keys() if key not in {"path", "operator", "operands"})
        parts = [f"{key}:{graphql_input_literal(value[key])}" for key in ordered_keys]
        return "{ " + " ".join(parts) + " }"
    raise TypeError(f"Unsupported GraphQL literal type: {type(value)!r}")


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


def format_repro_command(
    seed,
    consistency,
    randomize_consistency=False,
    mode="oracle",
    rounds=None,
    rows=None,
    dynamic_enabled=True,
    host=None,
    port=None,
    extra_args=None,
):
    parts = ["python", "weaviate_fuzz_oracle.py", "--mode", str(mode), "--seed", str(seed)]
    if rounds is not None:
        parts.extend(["--rounds", str(rounds)])
    if rows is not None:
        parts.extend(["-N", str(rows)])
    if host is not None:
        parts.extend(["--host", str(host)])
    if port is not None:
        parts.extend(["--port", str(port)])
    if get_fuzz_profile() != FUZZ_PROFILE_DEFAULT:
        parts.extend(["--profile", get_fuzz_profile()])
    if not dynamic_enabled:
        parts.append("--no-dynamic")
    if randomize_consistency:
        parts.append("--random-consistency")
    elif (consistency or DEFAULT_CONSISTENCY_LEVEL) != DEFAULT_CONSISTENCY_LEVEL:
        parts.extend(["--consistency", (consistency or DEFAULT_CONSISTENCY_LEVEL).name.lower()])
    for item in extra_args or []:
        parts.append(str(item))
    return shlex.join(parts)


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
    DATE_ARRAY = "DATE_ARRAY"
    OBJECT = "OBJECT"  # Nested object 

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
        FieldType.DATE_ARRAY: DataType.DATE_ARRAY,
        # OBJECT handled specially in reset_collection
    }
    return mapping.get(ftype)


# --- FuzzStats: 统计/度量收集 ---
class FuzzStats:
    """查询延迟、错误分类、通过率等统计信息收集 (日志汇总)"""
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
    if ftype == FieldType.DATE_ARRAY:
        return [canonicalize_date_string(v) for v in value] if isinstance(value, list) else None
    return value


def normalize_array_compare_item(ftype, value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if ftype == FieldType.INT:
        return int(value)
    if ftype == FieldType.NUMBER:
        return float(value)
    if ftype == FieldType.BOOL:
        return bool(value)
    if ftype == FieldType.TEXT:
        return str(value)
    if ftype == FieldType.DATE:
        return to_utc_timestamp(value)
    return convert_numpy_types(value)


def normalize_array_compare_items(ftype, value):
    if not isinstance(value, list):
        return None
    out = [normalize_array_compare_item(ftype, item) for item in value]
    return [item for item in out if item is not None]


def array_membership_mask(series, item_type, targets, mode):
    normalized_targets = [normalize_array_compare_item(item_type, target) for target in targets]
    normalized_targets = [target for target in normalized_targets if target is not None]
    normalized_target_set = set(normalized_targets)

    if mode == "contains_any":
        return series.apply(
            lambda value, vals=normalized_targets: any(target in (normalize_array_compare_items(item_type, value) or []) for target in vals)
            if isinstance(value, list)
            else False
        )
    if mode == "contains_all":
        return series.apply(
            lambda value, vals=normalized_targets: all(target in (normalize_array_compare_items(item_type, value) or []) for target in vals)
            if isinstance(value, list)
            else False
        )
    return series.apply(
        lambda value, vals=normalized_target_set: not any(item in vals for item in (normalize_array_compare_items(item_type, value) or []))
        if isinstance(value, list)
        else True
    )


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [convert_numpy_types(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(x) for x in obj]
    return obj


def convert_value_for_weaviate(key, value, field_type_map):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None

    ftype = field_type_map.get(key)
    if ftype == FieldType.INT:
        return int(value)
    if ftype == FieldType.BOOL:
        return bool(value)
    if ftype == FieldType.NUMBER:
        return float(value)
    if ftype in (FieldType.TEXT, FieldType.DATE):
        return str(value)
    if ftype == FieldType.GEO:
        geo = normalize_geo_value(value)
        if geo is None:
            return None
        return GeoCoordinate(latitude=geo["latitude"], longitude=geo["longitude"])
    return convert_numpy_types(value)


def build_field_type_map(schema_config):
    field_type_map = {field["name"]: field["type"] for field in schema_config}
    field_type_map["row_num"] = FieldType.INT
    return field_type_map


def build_weaviate_properties(row, field_type_map):
    props = {}
    for key, value in row.items():
        if key == "id" or key.startswith("_"):
            continue
        converted = convert_value_for_weaviate(key, value, field_type_map)
        if converted is None:
            continue
        if isinstance(converted, list) and len(converted) == 0:
            continue
        props[key] = converted
    return props


def normalize_weaviate_property_for_oracle(ftype, value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if ftype == FieldType.INT:
        return int(value)
    if ftype == FieldType.NUMBER:
        return float(value)
    if ftype == FieldType.BOOL:
        return bool(value)
    if ftype == FieldType.TEXT:
        return str(value)
    if ftype == FieldType.DATE:
        return canonicalize_date_string(value)
    if ftype == FieldType.GEO:
        return normalize_geo_value(value)
    if ftype == FieldType.INT_ARRAY:
        return [int(v) for v in value] if isinstance(value, list) else None
    if ftype == FieldType.NUMBER_ARRAY:
        return [float(v) for v in value] if isinstance(value, list) else None
    if ftype == FieldType.BOOL_ARRAY:
        return [bool(v) for v in value] if isinstance(value, list) else None
    if ftype == FieldType.TEXT_ARRAY:
        return [str(v) for v in value] if isinstance(value, list) else None
    if ftype == FieldType.DATE_ARRAY:
        return [canonicalize_date_string(v) for v in value] if isinstance(value, list) else None
    if ftype == FieldType.OBJECT:
        return convert_numpy_types(value) if isinstance(value, dict) else None
    return convert_numpy_types(value)


def canonical_row_from_weaviate_props(dm, object_id, properties, fallback_row=None):
    props = properties or {}
    row = {"id": str(object_id)}

    for field in dm.schema_config:
        fname = field["name"]
        ftype = field["type"]
        row[fname] = normalize_weaviate_property_for_oracle(ftype, props.get(fname))

    row["row_num"] = int(props.get("row_num", fallback_row.get("row_num", -1) if fallback_row else -1))
    if fallback_row:
        for meta_name in ("_creation_time", "_update_time"):
            if meta_name in fallback_row:
                row[meta_name] = fallback_row[meta_name]
    return row


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
        self.profile = get_fuzz_profile()
        self.df = None
        self.vectors = None
        self.schema_config = []
        self.filterable_fields = set()  # Populated by reset_collection
        self.searchable_text_fields = set()
        self.field_index_state = {}
        self.dropped_property_indexes = set()
        self.evolved_field_names = set()
        self.py_rng = random.Random(self.seed ^ 0x5EEDFACE)
        self.value_rng = np.random.default_rng(self.seed ^ 0xA17E)
        self.vector_rng = np.random.default_rng(self.seed ^ 0xC0FFEE)
        self.id_counter = 0
        self.row_num_counter = 0
        self.null_ratio = self.py_rng.uniform(0.05, 0.15)
        self.array_capacity = self.py_rng.randint(5, 20)
        self.int_range = self.py_rng.randint(5000, 100000)
        self.double_scale = self.py_rng.uniform(100, 10000)

    def _next_uuid(self):
        self.id_counter += 1
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-fuzz-{self.seed}-{self.id_counter}"))

    def _next_row_num(self):
        current = int(self.row_num_counter)
        self.row_num_counter += 1
        return current

    def allocate_row_num(self, existing_id=None):
        if existing_id is not None and self.df is not None and not self.df.empty:
            matches = self.df.loc[self.df["id"] == str(existing_id), "row_num"]
            if not matches.empty:
                try:
                    return int(matches.iloc[0])
                except Exception:
                    pass
        return self._next_row_num()

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
            elif element_type == FieldType.DATE:
                values.append(self._random_date(boundary=True))
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

    def generate_non_null_scalar_value(self, ftype):
        for _ in range(12):
            value = self._generate_scalar_value(ftype)
            if not is_null_like(value) and not (ftype == FieldType.TEXT and value == ""):
                return value

        if ftype == FieldType.INT:
            return 0
        if ftype == FieldType.NUMBER:
            return 0.0
        if ftype == FieldType.BOOL:
            return False
        if ftype == FieldType.TEXT:
            return "evolved_fallback"
        if ftype == FieldType.DATE:
            return "2024-01-01T00:00:00Z"
        return None

    def plan_evolved_property(self):
        evolved_count = sum(1 for field in self.schema_config if field["name"].startswith("evo_"))
        if evolved_count >= MAX_EVOLVED_PROPERTIES:
            return None

        type_choices = [FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT, FieldType.DATE]
        weights = [0.28, 0.26, 0.14, 0.22, 0.10]
        ftype = self.py_rng.choices(type_choices, weights=weights, k=1)[0]
        field_name = f"evo_{evolved_count}_{ftype.lower()}"

        return {
            "name": field_name,
            "type": ftype,
            "evolved": True,
            "index_filterable": True,
            "index_searchable": self.py_rng.choice([True, False]) if ftype == FieldType.TEXT else False,
            "index_range": self.py_rng.choice([True, False]) if ftype in (FieldType.INT, FieldType.NUMBER, FieldType.DATE) else False,
        }

    def register_evolved_property(self, field_config):
        field_name = field_config["name"]
        if any(field["name"] == field_name for field in self.schema_config):
            return
        self.schema_config.append(field_config)
        self.evolved_field_names.add(field_name)
        self.field_index_state[field_name] = {
            "filterable": bool(field_config.get("index_filterable", True)),
            "searchable": bool(field_config.get("index_searchable", False)),
            "range": bool(field_config.get("index_range", False)),
        }
        if self.df is not None and field_name not in self.df.columns:
            self.df[field_name] = None

    def build_evolved_backfill_data(self, field_config):
        if self.df is None or self.df.empty:
            return []
        field_name = field_config["name"]
        ftype = field_config["type"]
        return [
            (int(idx), str(self.df.at[idx, "id"]), self.generate_non_null_scalar_value(ftype))
            for idx in self.df.index
        ]

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

    def _generate_default_schema(self):
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

    def _generate_inverted_schema(self):
        self.schema_config.extend(
            [
                {"name": "c0", "type": FieldType.INT, "index_filterable": True, "index_range": True},
                {"name": "c1", "type": FieldType.INT, "index_filterable": True, "index_range": True},
                {"name": "c2", "type": FieldType.INT, "index_filterable": True, "index_range": True},
                {"name": "c3", "type": FieldType.NUMBER, "index_filterable": True, "index_range": True},
                {"name": "c4", "type": FieldType.NUMBER, "index_filterable": True, "index_range": True},
                {"name": "c5", "type": FieldType.NUMBER, "index_filterable": True, "index_range": True},
                {"name": "c6", "type": FieldType.BOOL, "index_filterable": True},
                {"name": "c7", "type": FieldType.BOOL, "index_filterable": True},
                {"name": "c8", "type": FieldType.TEXT, "index_filterable": True, "index_searchable": True},
                {"name": "c9", "type": FieldType.TEXT, "index_filterable": True, "index_searchable": False},
                {"name": "c10", "type": FieldType.TEXT, "index_filterable": True, "index_searchable": True},
                {"name": "c11", "type": FieldType.TEXT, "index_filterable": True, "index_searchable": False},
                {"name": "c12", "type": FieldType.DATE, "index_filterable": True, "index_range": True},
                {"name": "c13", "type": FieldType.DATE, "index_filterable": True, "index_range": True},
                {"name": "searchOnlyText0", "type": FieldType.TEXT, "index_filterable": False, "index_searchable": True},
                {"name": "searchOnlyText1", "type": FieldType.TEXT, "index_filterable": False, "index_searchable": True},
                {"name": "tagsArray", "type": FieldType.INT_ARRAY, "max_capacity": self.array_capacity, "index_filterable": True},
                {"name": "labelsArray", "type": FieldType.TEXT_ARRAY, "max_capacity": self.array_capacity, "index_filterable": True},
                {"name": "scoresArray", "type": FieldType.NUMBER_ARRAY, "max_capacity": self.array_capacity, "index_filterable": True},
                {"name": "flagsArray", "type": FieldType.BOOL_ARRAY, "max_capacity": self.array_capacity, "index_filterable": True},
                {"name": "datesArray", "type": FieldType.DATE_ARRAY, "max_capacity": self.array_capacity, "index_filterable": True},
            ]
        )

    def generate_schema(self):
        profile_name = self.profile
        if is_inverted_profile():
            print(f"🎲 1. Defining Inverted-Focused Schema... [Profile={profile_name}]")
        else:
            print(f"🎲 1. Defining Dynamic Schema... [Profile={profile_name}]")
        self.schema_config = []
        if is_inverted_profile():
            self._generate_inverted_schema()
        else:
            self._generate_default_schema()

        print(f"   -> Generated {len(self.schema_config)} fields (plus id & vector).")
        for f in self.schema_config:
            print(f"      - {f['name']:<20} : {f['type']}")

    def generate_single_row(self, id_override=None, row_num=None):
        if row_num is None:
            row_num = self.allocate_row_num(existing_id=id_override)
        row = {
            "id": str(id_override) if id_override is not None else self._next_uuid(),
            "row_num": int(row_num),
        }

        for field in self.schema_config:
            fname = field["name"]
            ftype = field["type"]
            null_ratio = self.null_ratio
            if is_inverted_profile() and fname.startswith("searchOnlyText"):
                null_ratio = 0.0

            if ftype in {FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT, FieldType.DATE}:
                row[fname] = None if self.py_rng.random() < null_ratio else self._generate_scalar_value(ftype)
            elif ftype == FieldType.GEO:
                row[fname] = None if self.py_rng.random() < null_ratio else self._random_geo(boundary=True)
            elif ftype == FieldType.INT_ARRAY:
                row[fname] = None if self.py_rng.random() < null_ratio else self._generate_array_value(FieldType.INT)
            elif ftype == FieldType.TEXT_ARRAY:
                row[fname] = None if self.py_rng.random() < null_ratio else self._generate_array_value(FieldType.TEXT)
            elif ftype == FieldType.NUMBER_ARRAY:
                row[fname] = None if self.py_rng.random() < null_ratio else self._generate_array_value(FieldType.NUMBER)
            elif ftype == FieldType.BOOL_ARRAY:
                row[fname] = None if self.py_rng.random() < null_ratio else self._generate_array_value(FieldType.BOOL)
            elif ftype == FieldType.DATE_ARRAY:
                row[fname] = None if self.py_rng.random() < null_ratio else self._generate_array_value(FieldType.DATE)
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

        rows = [self.generate_single_row(row_num=self._next_row_num()) for _ in range(N)]
        self.df = self.rows_to_dataframe(rows)

        for field in self.schema_config:
            if field["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.DATE_ARRAY]:
                self.df[field["name"]] = self.df[field["name"]].apply(
                    lambda x: None if isinstance(x, list) and len(x) == 0 else x
                )
        self.normalize_dataframe_types()

        print("✅ Data Generation Complete.")
        null_counts = {col: self.df[col].isna().sum() for col in self.df.columns}
        print(f"   -> Null counts (sample): {dict(list(null_counts.items())[:5])}")


AGGREGATE_PROBE_FIELDS = [
    {"name": "agg_filter_bucket", "type": FieldType.INT, "index_filterable": True, "index_range": True},
    {"name": "agg_int_metric", "type": FieldType.INT, "index_filterable": True, "index_range": True},
    {"name": "agg_num_metric", "type": FieldType.NUMBER, "index_filterable": True, "index_range": True},
    {"name": "agg_bool_metric", "type": FieldType.BOOL, "index_filterable": True},
    {"name": "agg_text_metric", "type": FieldType.TEXT, "index_filterable": True, "index_searchable": True},
    {"name": "agg_date_metric", "type": FieldType.DATE, "index_filterable": True, "index_range": True},
    {"name": "agg_date_array_metric", "type": FieldType.DATE_ARRAY, "index_filterable": True},
    {"name": "agg_int_array_metric", "type": FieldType.INT_ARRAY, "index_filterable": True},
    {"name": "agg_num_array_metric", "type": FieldType.NUMBER_ARRAY, "index_filterable": True},
    {"name": "agg_bool_array_metric", "type": FieldType.BOOL_ARRAY, "index_filterable": True},
    {"name": "agg_text_array_metric", "type": FieldType.TEXT_ARRAY, "index_filterable": True},
]


def _copy_probe_value(value):
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _aggregate_probe_value(field_name, row_num):
    row_num = int(row_num)

    if field_name == "agg_filter_bucket":
        return int(row_num % 7)

    if field_name == "agg_int_metric":
        if row_num % 17 == 0:
            return None
        if row_num % 101 == 0:
            boundary_values = [-1_000_000_000, -1024, 0, 1024, 1_000_000_000]
            return int(boundary_values[(row_num // 101) % len(boundary_values)])
        values = [-1024, -1, 0, 3, 3, 7, 7, 7, 1024]
        return int(values[row_num % len(values)])

    if field_name == "agg_num_metric":
        if row_num % 19 == 0:
            return None
        if row_num % 103 == 0:
            boundary_values = [
                -1e12,
                float(next_float(0.0, -np.inf) or -1e-12),
                0.0,
                float(next_float(0.0, np.inf) or 1e-12),
                1e12,
            ]
            return float(boundary_values[(row_num // 103) % len(boundary_values)])
        values = [-2.5, -0.5, 0.0, 1.25, 1.25, 4.75, 4.75, 4.75, 1e6]
        return float(values[row_num % len(values)])

    if field_name == "agg_bool_metric":
        if row_num % 13 == 0:
            return None
        values = [True, True, False, True, False, True]
        return bool(values[row_num % len(values)])

    if field_name == "agg_text_metric":
        if row_num % 23 == 0:
            return None
        values = ["alpha", "alpha", "beta", "gamma", "gamma", "gamma", "delta", "epsilon"]
        return str(values[row_num % len(values)])

    if field_name == "agg_date_metric":
        if row_num % 29 == 0:
            return None
        values = [
            "1969-12-31T23:59:59Z",
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-02T12:00:00Z",
            "2024-01-05T00:00:00.25Z",
            "2024-01-05T00:00:00.25Z",
            "2024-01-05T00:00:00.25Z",
            "2024-02-29T12:34:56.123456Z",
            "2024-03-01T00:00:00Z",
            "2026-12-31T23:59:59.999999Z",
        ]
        return str(values[row_num % len(values)])

    if field_name == "agg_int_array_metric":
        if row_num % 11 == 0:
            return None
        values = [
            [1, 2],
            [2],
            [3, 3, 3],
            [],
            [7, -1, 7],
            [0],
            [-1024, 1024],
        ]
        return _copy_probe_value(values[row_num % len(values)])

    if field_name == "agg_date_array_metric":
        if row_num % 15 == 0:
            return None
        values = [
            ["1969-12-31T23:59:59Z", "2024-01-01T00:00:00Z"],
            ["2024-01-05T00:00:00.25Z"],
            [],
            ["2024-01-05T00:00:00.25Z", "2026-12-31T23:59:59.999999Z"],
            ["2024-02-29T12:34:56.123456Z"],
        ]
        return _copy_probe_value(values[row_num % len(values)])

    if field_name == "agg_num_array_metric":
        if row_num % 12 == 0:
            return None
        values = [
            [1.5, 2.5],
            [2.5],
            [4.75, 4.75],
            [],
            [-1.25, 9.5],
            [0.0],
            [float(next_float(0.0, -np.inf) or -1e-12), float(next_float(0.0, np.inf) or 1e-12)],
        ]
        return _copy_probe_value(values[row_num % len(values)])

    if field_name == "agg_bool_array_metric":
        if row_num % 14 == 0:
            return None
        values = [
            [True, False],
            [True],
            [],
            [False, False],
            [True, True, False],
        ]
        return _copy_probe_value(values[row_num % len(values)])

    if field_name == "agg_text_array_metric":
        if row_num % 16 == 0:
            return None
        values = [
            ["alpha", "beta"],
            ["gamma"],
            [],
            ["gamma", "gamma"],
            ["delta", "alpha"],
            ["epsilon"],
        ]
        return _copy_probe_value(values[row_num % len(values)])

    raise KeyError(f"Unknown aggregate probe field: {field_name}")


def inject_aggregate_probe_fields(dm):
    if dm.df is None or dm.df.empty:
        return

    existing = {field["name"] for field in dm.schema_config}
    for spec in AGGREGATE_PROBE_FIELDS:
        if spec["name"] not in existing:
            dm.schema_config.append(dict(spec))
            existing.add(spec["name"])

    row_nums = pd.to_numeric(dm.df["row_num"], errors="coerce").fillna(-1).astype(int).tolist()
    for spec, row_num_values in [(spec, row_nums) for spec in AGGREGATE_PROBE_FIELDS]:
        dm.df[spec["name"]] = [
            _aggregate_probe_value(spec["name"], row_num)
            for row_num in row_num_values
        ]

    dm.normalize_dataframe_types()


# --- 2. Weaviate Manager ---

def select_rows_by_mask(df, mask):
    if df is None:
        return pd.DataFrame()
    if mask is None:
        return df.iloc[0:0].copy()

    if isinstance(mask, pd.Series):
        series = mask.copy()
    elif np.isscalar(mask):
        series = pd.Series([bool(mask)] * len(df), index=df.index)
    else:
        try:
            series = pd.Series(list(mask), index=df.index)
        except Exception:
            series = pd.Series([False] * len(df), index=df.index)

    if not series.index.equals(df.index):
        series = series.reindex(df.index)

    bool_indexer = series.fillna(False).astype(bool).to_numpy()
    return df.loc[bool_indexer].copy()


def format_rfc3339_nano(ts):
    value = to_utc_timestamp(ts)
    if value is None:
        return ""

    epoch_ns = int(value.value)
    sec, nanos = divmod(epoch_ns, 1_000_000_000)
    dt = pd.Timestamp(sec, unit="s", tz="UTC").to_pydatetime()
    out = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if nanos:
        out += f".{nanos:09d}".rstrip("0")
    return out + "Z"


def aggregate_family_for_type(ftype):
    if ftype in (FieldType.INT, FieldType.INT_ARRAY):
        return "integer"
    if ftype in (FieldType.NUMBER, FieldType.NUMBER_ARRAY):
        return "number"
    if ftype in (FieldType.BOOL, FieldType.BOOL_ARRAY):
        return "boolean"
    if ftype in (FieldType.TEXT, FieldType.TEXT_ARRAY):
        return "text"
    if ftype in (FieldType.DATE, FieldType.DATE_ARRAY):
        return "date"
    raise ValueError(f"Unsupported aggregate type: {ftype}")


def flatten_values_for_aggregate(frame, field_config):
    if frame is None or frame.empty:
        return []

    name = field_config["name"]
    ftype = field_config["type"]
    if name not in frame.columns:
        return []

    values = []
    for raw in frame[name].tolist():
        if ftype == FieldType.INT:
            if not is_null_like(raw):
                values.append(int(raw))
        elif ftype == FieldType.NUMBER:
            if not is_null_like(raw):
                values.append(float(raw))
        elif ftype == FieldType.BOOL:
            if not is_null_like(raw):
                values.append(bool(raw))
        elif ftype == FieldType.TEXT:
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                values.append(str(raw))
        elif ftype == FieldType.DATE:
            parsed = to_utc_timestamp(raw)
            if parsed is not None:
                values.append(parsed)
        elif ftype == FieldType.DATE_ARRAY:
            if isinstance(raw, list):
                for item in raw:
                    parsed = to_utc_timestamp(item)
                    if parsed is not None:
                        values.append(parsed)
        elif ftype == FieldType.INT_ARRAY:
            if isinstance(raw, list):
                values.extend(int(item) for item in raw if not is_null_like(item))
        elif ftype == FieldType.NUMBER_ARRAY:
            if isinstance(raw, list):
                values.extend(float(item) for item in raw if not is_null_like(item))
        elif ftype == FieldType.BOOL_ARRAY:
            if isinstance(raw, list):
                values.extend(bool(item) for item in raw if not is_null_like(item))
        elif ftype == FieldType.TEXT_ARRAY:
            if isinstance(raw, list):
                values.extend(str(item) for item in raw if item is not None)
    return values


def numeric_mode_with_tiebreak(values):
    if not values:
        return 0
    counter = {}
    for value in values:
        counter[value] = counter.get(value, 0) + 1
    max_count = max(counter.values())
    return min(value for value, count in counter.items() if count == max_count)


def numeric_median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def expected_numeric_aggregate(values, integer=False):
    if not values:
        return {
            "count": 0,
            "maximum": 0,
            "mean": 0.0,
            "median": 0.0,
            "minimum": 0,
            "mode": 0,
            "sum_": 0,
        }

    total = sum(values)
    mode_value = numeric_mode_with_tiebreak(values)
    result = {
        "count": int(len(values)),
        "maximum": max(values),
        "mean": float(total) / float(len(values)),
        "median": numeric_median(values),
        "minimum": min(values),
        "mode": mode_value,
        "sum_": total,
    }
    if integer:
        result["maximum"] = int(result["maximum"])
        result["minimum"] = int(result["minimum"])
        result["mode"] = int(result["mode"])
        result["sum_"] = int(result["sum_"])
    else:
        result["maximum"] = float(result["maximum"])
        result["minimum"] = float(result["minimum"])
        result["mode"] = float(result["mode"])
        result["sum_"] = float(result["sum_"])
    return result


def expected_boolean_aggregate(values):
    if not values:
        return {
            "count": 0,
            "percentage_false": float("nan"),
            "percentage_true": float("nan"),
            "total_false": 0,
            "total_true": 0,
        }

    total_true = int(sum(1 for value in values if bool(value)))
    total_false = int(len(values) - total_true)
    count = int(len(values))
    return {
        "count": count,
        "percentage_false": float(total_false) / float(count),
        "percentage_true": float(total_true) / float(count),
        "total_false": total_false,
        "total_true": total_true,
    }


def expected_date_aggregate(values):
    if not values:
        return {
            "count": 0,
            "maximum": "",
            "median": "",
            "minimum": "",
            "mode": "",
            "_mode_candidates": set(),
        }

    ordered = sorted(int(value.value) for value in values)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 1:
        median_ns = ordered[mid]
    else:
        median_ns = ordered[mid - 1] + (ordered[mid] - ordered[mid - 1]) // 2

    counter = {}
    for epoch_ns in ordered:
        counter[epoch_ns] = counter.get(epoch_ns, 0) + 1
    max_count = max(counter.values())
    mode_candidates = {format_rfc3339_nano(pd.Timestamp(epoch_ns, unit="ns", tz="UTC")) for epoch_ns, freq in counter.items() if freq == max_count}
    canonical_sorted = [format_rfc3339_nano(pd.Timestamp(epoch_ns, unit="ns", tz="UTC")) for epoch_ns in ordered]

    return {
        "count": int(count),
        "maximum": canonical_sorted[-1],
        "median": format_rfc3339_nano(pd.Timestamp(median_ns, unit="ns", tz="UTC")),
        "minimum": canonical_sorted[0],
        "mode": sorted(mode_candidates)[0],
        "_mode_candidates": set(mode_candidates),
    }


def expected_text_aggregate(values, limit=5):
    if not values:
        return {
            "count": 0,
            "top_occurrences": [],
            "_top_occurrences_stable": True,
        }

    counter = {}
    for value in values:
        counter[value] = counter.get(value, 0) + 1

    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    stable = True
    if limit is not None and len(ordered) > limit:
        cutoff = ordered[limit - 1][1]
        included = sum(1 for _, freq in ordered[:limit] if freq == cutoff)
        total_at_cutoff = sum(1 for _, freq in ordered if freq == cutoff)
        stable = included == total_at_cutoff

    top = ordered if limit is None else ordered[:limit]
    return {
        "count": int(len(values)),
        "top_occurrences": [{"value": value, "count": int(freq)} for value, freq in top],
        "_top_occurrences_stable": stable,
    }


def build_expected_aggregate(field_config, frame, *, text_limit=5):
    values = flatten_values_for_aggregate(frame, field_config)
    family = aggregate_family_for_type(field_config["type"])
    if family == "integer":
        return expected_numeric_aggregate(values, integer=True)
    if family == "number":
        return expected_numeric_aggregate(values, integer=False)
    if family == "boolean":
        return expected_boolean_aggregate(values)
    if family == "date":
        return expected_date_aggregate(values)
    if family == "text":
        return expected_text_aggregate(values, limit=text_limit)
    raise ValueError(f"Unsupported aggregate family: {family}")


def build_aggregate_metrics(field_config, *, text_limit=5):
    name = field_config["name"]
    family = aggregate_family_for_type(field_config["type"])
    if family == "integer":
        return Metrics(name).integer(count=True, maximum=True, mean=True, median=True, minimum=True, mode=True, sum_=True)
    if family == "number":
        return Metrics(name).number(count=True, maximum=True, mean=True, median=True, minimum=True, mode=True, sum_=True)
    if family == "boolean":
        return Metrics(name).boolean(
            count=True,
            percentage_false=True,
            percentage_true=True,
            total_false=True,
            total_true=True,
        )
    if family == "date":
        return Metrics(name).date_(count=True, maximum=True, median=True, minimum=True, mode=True)
    if family == "text":
        return Metrics(name).text(
            count=True,
            top_occurrences_count=True,
            top_occurrences_value=True,
            limit=text_limit,
        )
    raise ValueError(f"Unsupported aggregate family: {family}")


def _floats_match(actual, expected, tol=1e-9):
    try:
        actual_f = float(actual)
        expected_f = float(expected)
    except Exception:
        return False
    if np.isnan(actual_f) and np.isnan(expected_f):
        return True
    return bool(np.isclose(actual_f, expected_f, rtol=tol, atol=tol, equal_nan=True))


def compare_aggregate_property(actual_prop, expected, field_config):
    family = aggregate_family_for_type(field_config["type"])
    mismatches = []

    if family in {"integer", "number"}:
        for key in ["count", "maximum", "mean", "median", "minimum", "mode", "sum_"]:
            actual = getattr(actual_prop, key)
            target = expected[key]
            if key in {"count", "maximum", "minimum", "mode", "sum_"} and family == "integer":
                if int(actual) != int(target):
                    mismatches.append(f"{key}: expected {target}, got {actual}")
            else:
                if not _floats_match(actual, target):
                    mismatches.append(f"{key}: expected {target}, got {actual}")
        return mismatches

    if family == "boolean":
        for key in ["count", "total_false", "total_true"]:
            actual = getattr(actual_prop, key)
            target = expected[key]
            if int(actual) != int(target):
                mismatches.append(f"{key}: expected {target}, got {actual}")
        for key in ["percentage_false", "percentage_true"]:
            actual = getattr(actual_prop, key)
            target = expected[key]
            if not _floats_match(actual, target):
                mismatches.append(f"{key}: expected {target}, got {actual}")
        return mismatches

    if family == "date":
        for key in ["count", "maximum", "median", "minimum"]:
            actual = getattr(actual_prop, key)
            target = expected[key]
            if key == "count":
                if int(actual) != int(target):
                    mismatches.append(f"{key}: expected {target}, got {actual}")
            elif str(actual) != str(target):
                mismatches.append(f"{key}: expected {target}, got {actual}")

        actual_mode = str(getattr(actual_prop, "mode"))
        mode_candidates = expected.get("_mode_candidates") or {expected["mode"]}
        if actual_mode not in mode_candidates:
            mismatches.append(f"mode: expected one of {sorted(mode_candidates)}, got {actual_mode}")
        return mismatches

    if family == "text":
        if int(getattr(actual_prop, "count")) != int(expected["count"]):
            mismatches.append(f"count: expected {expected['count']}, got {getattr(actual_prop, 'count')}")
        if expected.get("_top_occurrences_stable", True):
            actual_pairs = sorted(
                [{"value": occ.value, "count": int(occ.count)} for occ in getattr(actual_prop, "top_occurrences", [])],
                key=lambda item: (-item["count"], item["value"]),
            )
            expected_pairs = sorted(expected["top_occurrences"], key=lambda item: (-item["count"], item["value"]))
            if actual_pairs != expected_pairs:
                mismatches.append(f"top_occurrences: expected {expected_pairs}, got {actual_pairs}")
        return mismatches

    mismatches.append(f"Unsupported aggregate family: {family}")
    return mismatches


def aggregate_probe_field_map(dm):
    wanted = {spec["name"] for spec in AGGREGATE_PROBE_FIELDS if spec["name"] != "agg_filter_bucket"}
    return {field["name"]: field for field in dm.schema_config if field["name"] in wanted}


def aggregate_null_mask(series, ftype):
    if ftype == FieldType.TEXT:
        return series.apply(lambda value: value is None or (isinstance(value, float) and np.isnan(value)) or value == "")
    if ftype in (FieldType.INT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.TEXT_ARRAY, FieldType.DATE_ARRAY):
        return series.apply(
            lambda value: value is None
            or (isinstance(value, float) and np.isnan(value))
            or (isinstance(value, list) and len(value) == 0)
        )
    if ftype == FieldType.DATE:
        return series.apply(lambda value: to_utc_timestamp(value) is None)
    return series.apply(is_null_like)


def _normalize_bool_mask(mask, index):
    if isinstance(mask, pd.Series):
        series = mask.copy()
    else:
        series = pd.Series(mask, index=index)
    if not series.index.equals(index):
        series = series.reindex(index)
    return series.fillna(False).astype(bool)


def _choose_present_scalar_value(series, caster=None):
    present = [value for value in series.tolist() if value is not None and not (isinstance(value, float) and np.isnan(value))]
    if not present:
        return None
    value = random.choice(present)
    return caster(value) if caster is not None else value


def _choose_array_targets(series, *, count=1):
    items = []
    for value in series.tolist():
        if isinstance(value, list):
            items.extend(value)
    unique = list(dict.fromkeys(items))
    if not unique:
        return None
    target_count = max(1, min(int(count), len(unique)))
    return random.sample(unique, target_count)


def rest_value_key_for_type(ftype, *, multi=False):
    mapping = {
        FieldType.INT: "valueIntArray" if multi else "valueInt",
        FieldType.NUMBER: "valueNumberArray" if multi else "valueNumber",
        FieldType.BOOL: "valueBooleanArray" if multi else "valueBoolean",
        FieldType.TEXT: "valueTextArray" if multi else "valueText",
        FieldType.DATE: "valueDateArray" if multi else "valueDate",
    }
    return mapping.get(ftype)


def normalize_rest_where_value(ftype, value, *, multi=False):
    if multi:
        if not isinstance(value, list):
            value = [value]
        out = []
        for item in value:
            if ftype == FieldType.INT:
                out.append(int(item))
            elif ftype == FieldType.NUMBER:
                out.append(float(item))
            elif ftype == FieldType.BOOL:
                out.append(bool(item))
            elif ftype == FieldType.TEXT:
                out.append(str(item))
            elif ftype == FieldType.DATE:
                out.append(canonicalize_date_string(item) or str(item))
            else:
                out.append(item)
        return out

    if ftype == FieldType.INT:
        return int(value)
    if ftype == FieldType.NUMBER:
        return float(value)
    if ftype == FieldType.BOOL:
        return bool(value)
    if ftype == FieldType.TEXT:
        return str(value)
    if ftype == FieldType.DATE:
        return canonicalize_date_string(value) or str(value)
    return value


def build_rest_where(path, operator, *, ftype=None, value=None, raw_key=None):
    where_filter = {"path": list(path), "operator": operator}
    if raw_key is not None:
        where_filter[raw_key] = value
        return where_filter
    if ftype is None:
        return where_filter
    multi = operator in {"ContainsAny", "ContainsAll", "ContainsNone"} and isinstance(value, list)
    value_key = rest_value_key_for_type(ftype, multi=multi)
    if value_key is None:
        raise ValueError(f"Unsupported REST where type: {ftype}")
    where_filter[value_key] = normalize_rest_where_value(ftype, value, multi=multi)
    return where_filter


def build_rest_compound(operator, operands):
    return {"operator": operator, "operands": operands}


def rest_batch_delete_dry_run(collection_name, where_filter, *, timeout=60):
    payload = {
        "match": {
            "class": collection_name,
            "where": where_filter,
        },
        "dryRun": True,
        "output": "verbose",
    }
    response = requests.delete(f"http://{HOST}:{PORT}/v1/batch/objects", json=payload, timeout=timeout)
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}

    if response.status_code != 200:
        raise RuntimeError(f"REST batch delete dryRun failed: status={response.status_code} body={body}")

    results = body.get("results") or {}
    matches = int(results.get("matches") or 0)
    objects = results.get("objects") or []
    ids = sorted(str(obj.get("id")) for obj in objects if obj.get("status") == "DRYRUN")
    if matches != len(ids):
        raise RuntimeError(f"REST dryRun returned matches={matches} but listed ids={len(ids)} body={body}")
    return ids


def extract_rest_value(where_filter):
    for key in (
        "valueInt",
        "valueIntArray",
        "valueNumber",
        "valueNumberArray",
        "valueBoolean",
        "valueBooleanArray",
        "valueText",
        "valueTextArray",
        "valueDate",
        "valueDateArray",
        "valueString",
        "valueStringArray",
        "valueGeoRange",
    ):
        if key in where_filter:
            return key, where_filter[key]
    return None, None


def rest_path_to_filter_target(path):
    if path == ["id"]:
        return Filter.by_id()
    if len(path) == 1:
        name = path[0]
        if isinstance(name, str) and name.startswith("len(") and name.endswith(")"):
            return Filter.by_property(name[4:-1], length=True)
        return Filter.by_property(name)
    raise ValueError(f"Unsupported REST path for filter builder conversion: {path}")


def rest_where_to_filter(where_filter):
    operator = str(where_filter.get("operator"))
    if operator in {"And", "Or"}:
        operands = [rest_where_to_filter(operand) for operand in (where_filter.get("operands") or [])]
        operands = [operand for operand in operands if operand is not None]
        if len(operands) < 2:
            raise ValueError(f"{operator} requires at least two operands")
        current = operands[0]
        for operand in operands[1:]:
            current = (current & operand) if operator == "And" else (current | operand)
        return current
    if operator == "Not":
        operands = where_filter.get("operands") or []
        if len(operands) != 1:
            raise ValueError("Not requires exactly one operand")
        return Filter.not_(rest_where_to_filter(operands[0]))

    target = rest_path_to_filter_target(where_filter.get("path") or [])
    _, value = extract_rest_value(where_filter)

    if operator == "IsNull":
        return target.is_none(bool(value))

    method_map = {
        "Equal": "equal",
        "NotEqual": "not_equal",
        "GreaterThan": "greater_than",
        "GreaterThanEqual": "greater_or_equal",
        "LessThan": "less_than",
        "LessThanEqual": "less_or_equal",
        "Like": "like",
        "ContainsAny": "contains_any",
        "ContainsAll": "contains_all",
        "ContainsNone": "contains_none",
    }
    method_name = method_map.get(operator)
    if method_name is None:
        raise ValueError(f"Unsupported REST operator for filter builder conversion: {operator}")
    return getattr(target, method_name)(value)


def rest_path_info(qg, path):
    if path == ["id"]:
        return {"name": "id", "type": FieldType.TEXT, "series": qg.df["id"], "is_length": False}
    if len(path) != 1:
        raise ValueError(f"Unsupported REST path: {path}")
    token = path[0]
    field_type_map = build_field_type_map(qg.dm.schema_config)
    if token.startswith("len(") and token.endswith(")"):
        name = token[4:-1]
        return {"name": name, "type": field_type_map.get(name), "series": qg.df[name], "is_length": True}
    return {"name": token, "type": field_type_map.get(token), "series": qg.df[token], "is_length": False}


def normalize_rest_compare_scalar(ftype, value):
    if ftype == FieldType.INT:
        return int(value)
    if ftype == FieldType.NUMBER:
        return float(value)
    if ftype == FieldType.BOOL:
        return bool(value)
    if ftype == FieldType.DATE:
        return to_utc_timestamp(value)
    return str(value)


def normalize_rest_compare_array_item(ftype, value):
    if ftype == FieldType.DATE:
        return to_utc_timestamp(value)
    return normalize_rest_compare_scalar(ftype, value)


def length_mask_for_rest(qg, series, base_type, operator, target):
    if base_type in (FieldType.INT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.TEXT_ARRAY, FieldType.DATE_ARRAY):
        lengths = series.apply(lambda value: 0 if qg._is_effectively_null_value(value, base_type) else len(value) if isinstance(value, list) else 0)
    else:
        lengths = series.apply(lambda value: 0 if qg._is_effectively_null_value(value, base_type) else len(str(value)))

    target = int(target)
    if operator == "Equal":
        return lengths == target
    if operator == "NotEqual":
        return lengths != target
    if operator == "GreaterThan":
        return lengths > target
    if operator == "GreaterThanEqual":
        return lengths >= target
    if operator == "LessThan":
        return lengths < target
    if operator == "LessThanEqual":
        return lengths <= target
    raise ValueError(f"Unsupported REST length operator: {operator}")


def evaluate_rest_where_mask(qg, where_filter):
    operator = str(where_filter.get("operator"))
    base_index = qg.df.index

    if operator == "And":
        operands = where_filter.get("operands") or []
        if not operands:
            return pd.Series(False, index=base_index, dtype=bool)
        current = _normalize_bool_mask(evaluate_rest_where_mask(qg, operands[0]), base_index)
        for operand in operands[1:]:
            current = current & _normalize_bool_mask(evaluate_rest_where_mask(qg, operand), base_index)
        return current

    if operator == "Or":
        operands = where_filter.get("operands") or []
        if not operands:
            return pd.Series(False, index=base_index, dtype=bool)
        current = _normalize_bool_mask(evaluate_rest_where_mask(qg, operands[0]), base_index)
        for operand in operands[1:]:
            current = current | _normalize_bool_mask(evaluate_rest_where_mask(qg, operand), base_index)
        return current

    if operator == "Not":
        operands = where_filter.get("operands") or []
        if len(operands) != 1:
            raise ValueError("REST Not expects exactly one operand")
        return qg._apply_not_mask(evaluate_rest_where_mask(qg, operands[0]))

    info = rest_path_info(qg, where_filter.get("path") or [])
    series = info["series"]
    ftype = info["type"]
    raw_key, raw_value = extract_rest_value(where_filter)

    if info["is_length"]:
        return length_mask_for_rest(qg, series, ftype, operator, raw_value)

    null_mask = qg._effective_null_mask(series, ftype)

    if operator == "IsNull":
        return null_mask if bool(raw_value) else ~null_mask

    if operator == "Like":
        return rest_like_mask(series, str(raw_value))

    if operator in {"ContainsAny", "ContainsAll", "ContainsNone"}:
        array_item_type = rest_array_item_type(ftype)
        if array_item_type is None:
            targets = list(raw_value if isinstance(raw_value, list) else [raw_value])
            normalized_targets = [normalize_rest_compare_scalar(ftype, target) for target in targets]
            target_set = set(normalized_targets)
            if operator == "ContainsAny":
                return series.apply(
                    lambda value, vals=target_set: normalize_rest_compare_scalar(ftype, value) in vals
                    if not qg._is_effectively_null_value(value, ftype)
                    else False
                )
            if operator == "ContainsAll":
                return series.apply(
                    lambda value, vals=normalized_targets: all(normalize_rest_compare_scalar(ftype, value) == target for target in vals)
                    if not qg._is_effectively_null_value(value, ftype)
                    else False
                )
            return series.apply(
                lambda value, vals=target_set: normalize_rest_compare_scalar(ftype, value) not in vals
                if not qg._is_effectively_null_value(value, ftype)
                else True
            )

        targets = list(raw_value if isinstance(raw_value, list) else [raw_value])
        normalized_targets = [normalize_rest_compare_array_item(array_item_type, target) for target in targets]
        normalized_targets = [target for target in normalized_targets if target is not None]

        def normalize_items(value):
            if not isinstance(value, list):
                return None
            out = [normalize_rest_compare_array_item(array_item_type, item) for item in value]
            return [item for item in out if item is not None]

        if operator == "ContainsAny":
            return series.apply(
                lambda value, vals=normalized_targets: any(target in normalize_items(value) for target in vals)
                if isinstance(value, list)
                else False
            )
        if operator == "ContainsAll":
            return series.apply(
                lambda value, vals=normalized_targets: all(target in normalize_items(value) for target in vals)
                if isinstance(value, list)
                else False
            )
        return series.apply(
            lambda value, vals=normalized_targets: not any(target in normalize_items(value) for target in vals)
            if isinstance(value, list)
            else True
        )

    if ftype == FieldType.DATE:
        op_map = {
            "Equal": "==",
            "NotEqual": "!=",
            "GreaterThan": ">",
            "GreaterThanEqual": ">=",
            "LessThan": "<",
            "LessThanEqual": "<=",
        }
        mask = qg._date_cmp_mask(series, op_map[operator], raw_value)
        if operator == "NotEqual":
            mask = mask | null_mask
        return mask

    target = normalize_rest_compare_scalar(ftype, raw_value)

    def compare(value):
        if qg._is_effectively_null_value(value, ftype):
            return operator == "NotEqual"
        probe = normalize_rest_compare_scalar(ftype, value)
        try:
            if operator == "Equal":
                return probe == target
            if operator == "NotEqual":
                return probe != target
            if operator == "GreaterThan":
                return probe > target
            if operator == "GreaterThanEqual":
                return probe >= target
            if operator == "LessThan":
                return probe < target
            if operator == "LessThanEqual":
                return probe <= target
        except Exception:
            return False
        raise ValueError(f"Unsupported REST leaf operator: {operator}")

    return series.apply(compare)


def generate_simple_aggregate_filter(dm):
    df = dm.df
    index = df.index
    choices = [
        "bucket_eq",
        "bucket_range",
        "int_equal",
        "num_range",
        "bool_equal",
        "text_equal",
        "date_range",
        "null_check",
        "date_array_contains",
        "int_array_contains",
        "text_array_contains",
        "bool_array_contains",
    ]
    choice = random.choice(choices)

    if choice == "bucket_eq":
        series = df["agg_filter_bucket"]
        value = int(random.choice(sorted(int(v) for v in series.dropna().unique().tolist())))
        mask = series.apply(lambda item, target=value: int(item) == target if not is_null_like(item) else False)
        return Filter.by_property("agg_filter_bucket").equal(value), mask, f"agg_filter_bucket == {value}"

    if choice == "bucket_range":
        lo = random.randint(0, 4)
        hi = random.randint(lo, 6)
        series = df["agg_filter_bucket"]
        mask = series.apply(lambda item, left=lo, right=hi: left <= int(item) <= right if not is_null_like(item) else False)
        flt = Filter.by_property("agg_filter_bucket").greater_or_equal(int(lo)) & Filter.by_property("agg_filter_bucket").less_or_equal(int(hi))
        return flt, mask, f"agg_filter_bucket in [{lo},{hi}]"

    if choice == "int_equal":
        series = df["agg_int_metric"]
        value = _choose_present_scalar_value(series, int)
        if value is None:
            return None, None, None
        mask = series.apply(lambda item, target=value: int(item) == target if not is_null_like(item) else False)
        return Filter.by_property("agg_int_metric").equal(int(value)), mask, f"agg_int_metric == {value}"

    if choice == "num_range":
        series = df["agg_num_metric"]
        values = sorted(float(item) for item in series.dropna().tolist())
        if not values:
            return None, None, None
        anchor = float(random.choice(values))
        lo = next_float(anchor, -np.inf)
        hi = next_float(anchor, np.inf)
        if lo is None or hi is None:
            lo = anchor - 1e-9
            hi = anchor + 1e-9
        mask = series.apply(lambda item, left=lo, right=hi: left <= float(item) <= right if not is_null_like(item) else False)
        flt = Filter.by_property("agg_num_metric").greater_or_equal(float(lo)) & Filter.by_property("agg_num_metric").less_or_equal(float(hi))
        return flt, mask, f"agg_num_metric in [{lo},{hi}]"

    if choice == "bool_equal":
        series = df["agg_bool_metric"]
        value = bool(random.choice([True, False]))
        mask = series.apply(lambda item, target=value: bool(item) == target if not is_null_like(item) else False)
        return Filter.by_property("agg_bool_metric").equal(value), mask, f"agg_bool_metric == {value}"

    if choice == "text_equal":
        series = df["agg_text_metric"]
        value = _choose_present_scalar_value(series, str)
        if value is None:
            return None, None, None
        mask = series.apply(lambda item, target=value: str(item) == target if item is not None and not (isinstance(item, float) and np.isnan(item)) else False)
        return Filter.by_property("agg_text_metric").equal(value), mask, f'agg_text_metric == "{value}"'

    if choice == "date_range":
        series = df["agg_date_metric"]
        present = [to_utc_timestamp(item) for item in series.tolist()]
        present = [item for item in present if item is not None]
        if not present:
            return None, None, None
        anchor = random.choice(present)
        delta = random.choice([pd.Timedelta(microseconds=0), pd.Timedelta(microseconds=1), pd.Timedelta(seconds=1)])
        lo = format_rfc3339_nano(anchor - delta)
        hi = format_rfc3339_nano(anchor + delta)
        mask = series.apply(
            lambda item, left=to_utc_timestamp(lo), right=to_utc_timestamp(hi): left <= to_utc_timestamp(item) <= right
            if to_utc_timestamp(item) is not None
            else False
        )
        flt = Filter.by_property("agg_date_metric").greater_or_equal(lo) & Filter.by_property("agg_date_metric").less_or_equal(hi)
        return flt, mask, f"agg_date_metric in [{lo},{hi}]"

    if choice == "null_check":
        field_name, ftype = random.choice(
            [
                ("agg_int_metric", FieldType.INT),
                ("agg_bool_metric", FieldType.BOOL),
                ("agg_text_metric", FieldType.TEXT),
                ("agg_date_metric", FieldType.DATE),
                ("agg_date_array_metric", FieldType.DATE_ARRAY),
                ("agg_int_array_metric", FieldType.INT_ARRAY),
                ("agg_text_array_metric", FieldType.TEXT_ARRAY),
            ]
        )
        series = df[field_name]
        null_mask = aggregate_null_mask(series, ftype)
        if random.random() < 0.5:
            return Filter.by_property(field_name).is_none(True), null_mask, f"{field_name} is null"
        return Filter.by_property(field_name).is_none(False), ~null_mask, f"{field_name} is not null"

    if choice == "date_array_contains":
        series = df["agg_date_array_metric"]
        targets = _choose_array_targets(series, count=1)
        if not targets:
            return None, None, None
        target = canonicalize_date_string(targets[0]) or str(targets[0])
        mask = array_membership_mask(series, FieldType.DATE, [target], "contains_any")
        return Filter.by_property("agg_date_array_metric").contains_any([target]), mask, f"agg_date_array_metric contains_any [{target}]"

    if choice == "int_array_contains":
        series = df["agg_int_array_metric"]
        targets = _choose_array_targets(series, count=random.choice([1, 2]))
        if not targets:
            return None, None, None
        mode = random.choice(["contains_any", "contains_all", "contains_none"])
        if mode == "contains_any":
            mask = series.apply(lambda item, vals=targets: any(val in item for val in vals) if isinstance(item, list) else False)
            return Filter.by_property("agg_int_array_metric").contains_any([int(value) for value in targets]), mask, f"agg_int_array_metric contains_any {targets}"
        if mode == "contains_all":
            mask = series.apply(lambda item, vals=targets: all(val in item for val in vals) if isinstance(item, list) else False)
            return Filter.by_property("agg_int_array_metric").contains_all([int(value) for value in targets]), mask, f"agg_int_array_metric contains_all {targets}"
        mask = series.apply(lambda item, vals=targets: not any(val in item for val in vals) if isinstance(item, list) else True)
        return Filter.by_property("agg_int_array_metric").contains_none([int(value) for value in targets]), mask, f"agg_int_array_metric contains_none {targets}"

    if choice == "text_array_contains":
        series = df["agg_text_array_metric"]
        targets = _choose_array_targets(series, count=random.choice([1, 2]))
        if not targets:
            return None, None, None
        mode = random.choice(["contains_any", "contains_all", "contains_none"])
        if mode == "contains_any":
            mask = series.apply(lambda item, vals=targets: any(val in item for val in vals) if isinstance(item, list) else False)
            return Filter.by_property("agg_text_array_metric").contains_any([str(value) for value in targets]), mask, f"agg_text_array_metric contains_any {targets}"
        if mode == "contains_all":
            mask = series.apply(lambda item, vals=targets: all(val in item for val in vals) if isinstance(item, list) else False)
            return Filter.by_property("agg_text_array_metric").contains_all([str(value) for value in targets]), mask, f"agg_text_array_metric contains_all {targets}"
        mask = series.apply(lambda item, vals=targets: not any(val in item for val in vals) if isinstance(item, list) else True)
        return Filter.by_property("agg_text_array_metric").contains_none([str(value) for value in targets]), mask, f"agg_text_array_metric contains_none {targets}"

    if choice == "bool_array_contains":
        series = df["agg_bool_array_metric"]
        targets = _choose_array_targets(series, count=random.choice([1, 2]))
        if not targets:
            return None, None, None
        mask = series.apply(lambda item, vals=targets: any(val in item for val in vals) if isinstance(item, list) else False)
        return Filter.by_property("agg_bool_array_metric").contains_any([bool(value) for value in targets]), mask, f"agg_bool_array_metric contains_any {targets}"

    return None, None, None


def generate_aggregate_filter(dm):
    base_index = dm.df.index
    if random.random() < 0.15:
        return None, pd.Series(True, index=base_index, dtype=bool), "ALL", {"depth": 0, "atoms": 0, "target_depth": 0}

    def leaf():
        flt, mask, expr = generate_simple_aggregate_filter(dm)
        if flt is None:
            return None, None, None, None
        return flt, _normalize_bool_mask(mask, base_index), expr, {"depth": 0, "atoms": 1}

    def rec(depth):
        if depth <= 0:
            return leaf()

        op = random.choices(["and", "or", "not"], weights=[0.45, 0.40, 0.15], k=1)[0]
        if op == "not":
            inner_filter, inner_mask, inner_expr, inner_meta = rec(depth - 1)
            if inner_filter is None:
                return leaf()
            return (
                Filter.not_(inner_filter),
                ~inner_mask,
                f"NOT({inner_expr})",
                {"depth": 1 + int(inner_meta["depth"]), "atoms": int(inner_meta["atoms"])},
            )

        left_filter, left_mask, left_expr, left_meta = rec(depth - 1)
        right_filter, right_mask, right_expr, right_meta = rec(depth - 1)
        if left_filter is None:
            return right_filter, right_mask, right_expr, right_meta
        if right_filter is None:
            return left_filter, left_mask, left_expr, left_meta

        if op == "and":
            return (
                left_filter & right_filter,
                left_mask & right_mask,
                f"({left_expr} AND {right_expr})",
                {"depth": 1 + max(int(left_meta["depth"]), int(right_meta["depth"])), "atoms": int(left_meta["atoms"]) + int(right_meta["atoms"])},
            )
        return (
            left_filter | right_filter,
            left_mask | right_mask,
            f"({left_expr} OR {right_expr})",
            {"depth": 1 + max(int(left_meta["depth"]), int(right_meta["depth"])), "atoms": int(left_meta["atoms"]) + int(right_meta["atoms"])},
        )

    depth = random.randint(AGGREGATE_FILTER_MIN_DEPTH, AGGREGATE_FILTER_MAX_DEPTH)
    flt, mask, expr, meta = rec(depth)
    if flt is None:
        return None, pd.Series(True, index=base_index, dtype=bool), "ALL", {"depth": 0, "atoms": 0, "target_depth": depth}
    meta = dict(meta or {})
    meta["target_depth"] = depth
    return flt, _normalize_bool_mask(mask, base_index), expr, meta

class WeaviateManager:
    def __init__(self):
        self.client = None
        self.field_index_state = {}
        self.searchable_text_fields = set()
        self.actual_vector_index_type = None
        self.actual_distance_metric = None

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
        deterministic_index_profile = is_inverted_profile()
        filterable_fields = set()  # Track which fields have index_filterable=True
        searchable_text_fields = set()
        field_index_state = {}
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
                    field_index_state[field["name"]] = {"filterable": True, "searchable": False, "range": False}
                continue
            wv_type = get_weaviate_datatype(field["type"])
            if wv_type:
                # Scalar index config fuzzing (标量索引随机创建)
                idx_filterable = field.get("index_filterable")
                if idx_filterable is None:
                    if deterministic_index_profile:
                        idx_filterable = field["type"] != FieldType.GEO
                    else:
                        idx_filterable = True if field["type"] == FieldType.GEO else random.choice([True, True, True, False])  # 75% on
                idx_searchable = field.get("index_searchable")
                if idx_searchable is None:
                    if deterministic_index_profile:
                        idx_searchable = field["type"] in (FieldType.TEXT, FieldType.TEXT_ARRAY) and not idx_filterable
                    else:
                        idx_searchable = random.choice([True, False]) if field["type"] == FieldType.TEXT else False
                idx_range = field.get("index_range")
                if idx_range is None:
                    if deterministic_index_profile:
                        idx_range = field["type"] in (FieldType.INT, FieldType.NUMBER, FieldType.DATE)
                    else:
                        idx_range = random.choice([True, False]) if field["type"] in (FieldType.INT, FieldType.NUMBER, FieldType.DATE) else False
                # TEXT/TEXT_ARRAY 使用 Tokenization.FIELD: 精确匹配, 区分大小写, 无停用词过滤
                tok = Tokenization.FIELD if field["type"] in (FieldType.TEXT, FieldType.TEXT_ARRAY) else None
                p_kwargs = dict(name=field["name"], data_type=wv_type,
                                index_filterable=idx_filterable,
                                index_searchable=idx_searchable,
                                index_range_filters=idx_range)
                if tok is not None:
                    p_kwargs["tokenization"] = tok
                p = Property(**p_kwargs)
                properties.append(p)
                field_index_state[field["name"]] = {
                    "filterable": bool(idx_filterable),
                    "searchable": bool(idx_searchable),
                    "range": bool(idx_range),
                }
                if idx_filterable:
                    filterable_fields.add(field["name"])
                if field["type"] == FieldType.TEXT and idx_searchable:
                    searchable_text_fields.add(field["name"])
                index_config_log.append(f"{field['name']}: filt={idx_filterable} search={idx_searchable} range={idx_range}")
        properties.append(
            Property(
                name="row_num",
                data_type=DataType.INT,
                index_filterable=True,
                index_range_filters=True,
            )
        )
        if index_config_log:
            descriptor = "deterministic" if deterministic_index_profile else "randomized"
            print(f"   -> Scalar index config: {len(index_config_log)} fields {descriptor}")
            for ic in index_config_log[:5]:
                print(f"      {ic}")

        requested_vi_type = VECTOR_INDEX_TYPE or "hnsw"
        dist = DISTANCE_METRIC or VectorDistances.COSINE
        vi_type = requested_vi_type

        if vi_type == "hfresh" and dist == VectorDistances.DOT:
            dist = random.choice([VectorDistances.COSINE, VectorDistances.L2_SQUARED])
            print(f"   -> Adjusted HFRESH distance to supported metric: {dist}")

        if vi_type == "hnsw":
            ef_c = random.choice([64, 128, 256])
            max_conn = random.choice([16, 32, 64])
            vi_config = Configure.VectorIndex.hnsw(distance_metric=dist, ef_construction=ef_c, max_connections=max_conn)
            print(f"   -> VectorIndex: HNSW (ef_c={ef_c}, max_conn={max_conn})")
        elif vi_type == "flat":
            cache_sz = random.choice([100000, 200000, 400000])
            vi_config = Configure.VectorIndex.flat(distance_metric=dist, vector_cache_max_objects=cache_sz)
            print(f"   -> VectorIndex: FLAT (cache={cache_sz})")
        elif vi_type == "hfresh":
            posting_kb = random.choice([128, 256, 512])
            probe = random.choice([4, 8, 16])
            vi_config = Configure.VectorIndex.hfresh(
                distance_metric=dist,
                max_posting_size_kb=posting_kb,
                search_probe=probe,
            )
            print(f"   -> VectorIndex: HFRESH (posting_kb={posting_kb}, probe={probe})")
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
                if "async indexing" in err_msg or ("dynamic" in err_msg.lower() and vi_type == "dynamic"):
                    vi_config = Configure.VectorIndex.hnsw(distance_metric=dist)
                    vi_type = "hnsw"
                    print(f"   -> Falling back to HNSW (dynamic requires ASYNC_INDEXING=true in server env)")
                elif vi_type == "hfresh" and ("422" in err_msg or "hfresh" in err_msg.lower() or "unsupported" in err_msg.lower()):
                    vi_config = Configure.VectorIndex.hnsw(distance_metric=dist)
                    vi_type = "hnsw"
                    print("   -> Falling back to HNSW (HFRESH unsupported in current server config)")
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
                        vi_type = "hnsw"
                else:
                    if attempt == 2:
                        raise
        print("🛠️ Collection Created.")
        self.searchable_text_fields = searchable_text_fields
        self.field_index_state = field_index_state
        self.actual_vector_index_type = vi_type
        self.actual_distance_metric = dist
        return filterable_fields

    def _build_property_from_field_config(self, field_config):
        ftype = field_config["type"]
        wv_type = get_weaviate_datatype(ftype)
        if wv_type is None:
            raise ValueError(f"Unsupported evolved field type: {ftype}")

        p_kwargs = {
            "name": field_config["name"],
            "data_type": wv_type,
            "index_filterable": bool(field_config.get("index_filterable", True)),
            "index_searchable": bool(field_config.get("index_searchable", False)),
            "index_range_filters": bool(field_config.get("index_range", False)),
        }
        if ftype == FieldType.TEXT:
            p_kwargs["tokenization"] = Tokenization.FIELD
        return Property(**p_kwargs)

    def add_evolved_property(self, field_config):
        collection = self.client.collections.get(CLASS_NAME)
        collection.config.add_property(self._build_property_from_field_config(field_config))
        self.field_index_state[field_config["name"]] = {
            "filterable": bool(field_config.get("index_filterable", True)),
            "searchable": bool(field_config.get("index_searchable", False)),
            "range": bool(field_config.get("index_range", False)),
        }
        if field_config.get("index_searchable"):
            self.searchable_text_fields.add(field_config["name"])

    def fetch_objects_by_ids(self, ids, retries=4, wait_seconds=None):
        if not ids:
            return {}

        collection = self.client.collections.get(CLASS_NAME)
        targets = [str(object_id) for object_id in ids]
        wait_seconds = SLEEP_INTERVAL * 2 if wait_seconds is None else wait_seconds
        objects = {}

        for attempt in range(max(1, retries)):
            objects = {}
            for object_id in targets:
                try:
                    obj = collection.query.fetch_object_by_id(object_id)
                except Exception:
                    obj = None
                if obj is not None:
                    objects[object_id] = obj
            if len(objects) == len(targets):
                break
            if attempt + 1 < retries:
                time.sleep(wait_seconds)
        return objects

    def apply_fetched_rows_to_shadow(self, dm, objects_by_id, vectors_by_id=None):
        if not objects_by_id:
            return [], []

        vectors_by_id = vectors_by_id or {}
        id_to_index = {str(value): idx for idx, value in enumerate(dm.df["id"].tolist())} if not dm.df.empty else {}
        new_rows = []
        new_vectors = []
        synced_ids = []

        for object_id in sorted(objects_by_id):
            obj = objects_by_id[object_id]
            row_idx = id_to_index.get(str(object_id))
            fallback_row = dm.df.iloc[row_idx].to_dict() if row_idx is not None else None
            canonical = canonical_row_from_weaviate_props(dm, object_id, getattr(obj, "properties", None), fallback_row=fallback_row)
            if row_idx is not None:
                for key, value in canonical.items():
                    if key == "id":
                        continue
                    dm.df.at[row_idx, key] = value
                if str(object_id) in vectors_by_id:
                    dm.vectors[row_idx] = np.asarray(vectors_by_id[str(object_id)], dtype=np.float32)
            else:
                new_rows.append(canonical)
                if str(object_id) in vectors_by_id:
                    new_vectors.append(np.asarray(vectors_by_id[str(object_id)], dtype=np.float32))
            synced_ids.append(str(object_id))

        if new_rows:
            dm.df = pd.concat([dm.df, dm.rows_to_dataframe(new_rows)], ignore_index=True)
            if new_vectors:
                new_arr = np.asarray(new_vectors, dtype=np.float32)
                dm.vectors = new_arr if dm.vectors is None or len(dm.vectors) == 0 else np.vstack([dm.vectors, new_arr])

        dm.normalize_dataframe_types()
        return synced_ids, [row["id"] for row in new_rows]

    def sync_rows_from_engine(self, dm, ids, vectors_by_id=None, retries=4):
        objects_by_id = self.fetch_objects_by_ids(ids, retries=retries)
        synced_ids, new_ids = self.apply_fetched_rows_to_shadow(dm, objects_by_id, vectors_by_id=vectors_by_id)
        missing_ids = [str(object_id) for object_id in ids if str(object_id) not in objects_by_id]
        return synced_ids, missing_ids, new_ids

    def verify_deleted_ids(self, dm, ids, retries=4):
        remaining = self.fetch_objects_by_ids(ids, retries=retries)
        deleted_ids = [str(object_id) for object_id in ids if str(object_id) not in remaining]
        still_present = [str(object_id) for object_id in ids if str(object_id) in remaining]

        if deleted_ids and not dm.df.empty:
            keep = ~dm.df["id"].isin(deleted_ids)
            kept_index = dm.df[keep].index.to_numpy()
            dm.df = dm.df[keep].reset_index(drop=True)
            if dm.vectors is not None and len(dm.vectors) == len(keep):
                dm.vectors = dm.vectors[kept_index]
            dm.normalize_dataframe_types()
        return deleted_ids, still_present

    def backfill_evolved_property(self, dm, field_config, backfill_data, retries=3):
        if not backfill_data:
            return [], []

        collection = self.client.collections.get(CLASS_NAME)
        field_name = field_config["name"]
        field_type_map = {field_name: field_config["type"]}
        attempted_ids = []

        for pandas_idx, row_id, value in backfill_data:
            props = build_weaviate_properties({field_name: value}, field_type_map)
            if not props:
                continue
            attempted_ids.append(str(row_id))
            for attempt in range(max(1, retries)):
                try:
                    collection.data.update(uuid=str(row_id), properties=props)
                    break
                except Exception:
                    if attempt + 1 >= retries:
                        break
                    time.sleep(SLEEP_INTERVAL * 2)

        objects_by_id = self.fetch_objects_by_ids(attempted_ids, retries=retries)
        success_ids = []
        missing_ids = []
        for pandas_idx, row_id, _ in backfill_data:
            object_id = str(row_id)
            obj = objects_by_id.get(object_id)
            if obj is None:
                missing_ids.append(object_id)
                continue
            value = getattr(obj, "properties", {}).get(field_name)
            if value is None:
                missing_ids.append(object_id)
                continue
            dm.df.at[pandas_idx, field_name] = normalize_weaviate_property_for_oracle(field_config["type"], value)
            success_ids.append(object_id)

        dm.normalize_dataframe_types()
        return success_ids, missing_ids

    def mutate_scalar_index(self, dm):
        if len(dm.dropped_property_indexes) >= MAX_SCALAR_INDEX_DROPS:
            return None

        candidates = []
        for field in dm.schema_config:
            name = field["name"]
            state = dm.field_index_state.get(name, {})
            if state.get("filterable") and name in dm.filterable_fields:
                candidates.append((name, "filterable"))
            if field["type"] == FieldType.TEXT and state.get("searchable") and name in dm.searchable_text_fields:
                candidates.append((name, "searchable"))
            if state.get("range"):
                candidates.append((name, "rangeFilters"))

        candidates = [item for item in candidates if item not in dm.dropped_property_indexes]
        if not candidates:
            return None

        property_name, index_name = random.choice(candidates)
        collection = self.client.collections.get(CLASS_NAME)
        dropped = collection.config.delete_property_index(property_name, index_name)
        if not dropped:
            return None

        dm.dropped_property_indexes.add((property_name, index_name))
        state = dm.field_index_state.setdefault(property_name, {"filterable": False, "searchable": False, "range": False})
        if index_name == "filterable":
            state["filterable"] = False
            dm.filterable_fields.discard(property_name)
        elif index_name == "searchable":
            state["searchable"] = False
            dm.searchable_text_fields.discard(property_name)
        elif index_name == "rangeFilters":
            state["range"] = False
        return property_name, index_name

    def reconfigure_vector_index(self):
        collection = self.client.collections.get(CLASS_NAME)
        vi_type = self.actual_vector_index_type or VECTOR_INDEX_TYPE or "hnsw"

        if vi_type == "hnsw":
            new_ef = random.choice([64, 128, 256, 512])
            new_dyn = random.choice([4, 8, 12])
            new_cutoff = random.choice([20000, 40000, 60000])
            collection.config.update(
                vector_config=Reconfigure.Vectors.update(
                    vector_index_config=Reconfigure.VectorIndex.hnsw(
                        ef=new_ef,
                        dynamic_ef_factor=new_dyn,
                        flat_search_cutoff=new_cutoff,
                    )
                )
            )
            return f"HNSW ef={new_ef} dyn_ef={new_dyn} cutoff={new_cutoff}"

        if vi_type == "flat":
            cache_sz = random.choice([100000, 200000, 400000, 800000])
            collection.config.update(
                vector_config=Reconfigure.Vectors.update(
                    vector_index_config=Reconfigure.VectorIndex.flat(
                        vector_cache_max_objects=cache_sz,
                    )
                )
            )
            return f"FLAT cache={cache_sz}"

        if vi_type == "hfresh":
            posting_kb = random.choice([128, 256, 512, 1024])
            probe = random.choice([4, 8, 16, 32])
            collection.config.update(
                vector_config=Reconfigure.Vectors.update(
                    vector_index_config=Reconfigure.VectorIndex.hfresh(
                        max_posting_size_kb=posting_kb,
                        search_probe=probe,
                    )
                )
            )
            return f"HFRESH posting_kb={posting_kb} probe={probe}"

        if vi_type == "dynamic":
            threshold = random.choice([5000, 10000, 20000, 40000])
            collection.config.update(
                vector_config=Reconfigure.Vectors.update(
                    vector_index_config=Reconfigure.VectorIndex.dynamic(
                        threshold=threshold,
                    )
                )
            )
            return f"DYNAMIC threshold={threshold}"

        raise ValueError(f"Unsupported vector index type: {vi_type}")

    def _sync_metadata_column(self, dm, ids, retries, column_name, metadata_query, metadata_attr):
        if self.client is None or dm.df.empty:
            return

        collection = self.client.collections.get(CLASS_NAME)
        target_ids = None if ids is None else list(dict.fromkeys(str(x) for x in ids))
        if target_ids is not None and not target_ids:
            return

        target_set = set(target_ids or [])
        use_id_filter = bool(target_ids) and len(target_ids) <= get_query_page_size()
        fetch_cap = len(target_ids) if use_id_filter else len(dm.df) + 1
        mapping = {}

        for attempt in range(retries):
            kwargs = {
                "return_metadata": metadata_query,
                "return_properties": False,
            }
            if use_id_filter:
                if len(target_ids) == 1:
                    kwargs["filters"] = Filter.by_id().equal(target_ids[0])
                else:
                    kwargs["filters"] = Filter.by_id().contains_any(target_ids)

            if use_id_filter:
                response = fetch_objects_resilient(collection, dm=dm, max_objects=fetch_cap, **kwargs)
            else:
                response = fetch_objects_cursor(collection, max_objects=fetch_cap, **kwargs)
            mapping = {
                str(obj.uuid): self._normalize_metadata_time(
                    getattr(getattr(obj, "metadata", None), metadata_attr, None)
                )
                for obj in response.objects
                if target_ids is None or str(obj.uuid) in target_set
            }

            if target_ids is None or all(target in mapping for target in target_ids):
                break
            time.sleep(SLEEP_INTERVAL * 2)

        if column_name not in dm.df.columns:
            dm.df[column_name] = pd.NaT

        if target_ids is None:
            dm.df[column_name] = pd.to_datetime(
                dm.df["id"].map(mapping), utc=True, errors="coerce"
            )
        else:
            mask = dm.df["id"].isin(target_ids)
            dm.df.loc[mask, column_name] = dm.df.loc[mask, "id"].map(mapping)
            dm.df[column_name] = pd.to_datetime(dm.df[column_name], utc=True, errors="coerce")

    def sync_creation_times(self, dm, ids=None, retries=3):
        self._sync_metadata_column(
            dm,
            ids,
            retries,
            "_creation_time",
            MetadataQuery(creation_time=True),
            "creation_time",
        )

    def sync_update_times(self, dm, ids=None, retries=3):
        self._sync_metadata_column(
            dm,
            ids,
            retries,
            "_update_time",
            MetadataQuery(last_update_time=True),
            "last_update_time",
        )

    def insert(self, dm):
        print(f"⚡ 3. Inserting Data (Batch={BATCH_SIZE})...")
        collection = self.client.collections.get(CLASS_NAME)
        records = dm.df.to_dict(orient="records")
        total = len(records)
        field_type_map = build_field_type_map(dm.schema_config)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch = records[start:end]
            data_objects = []
            for i, row in enumerate(batch):
                props = build_weaviate_properties(row, field_type_map)
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
        self.profile = getattr(dm, "profile", get_fuzz_profile())
        self._inverted_profile = self.profile == FUZZ_PROFILE_INVERTED
        self._absent_uuid_counter = 0
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
        self.search_only_text_schema = [
            field
            for field in dm.schema_config
            if field["type"] == FieldType.TEXT
            and bool((dm.field_index_state.get(field["name"], {}) or {}).get("searchable"))
            and not bool((dm.field_index_state.get(field["name"], {}) or {}).get("filterable"))
        ]
        self.geo_schema = [f for f in self.schema if f["type"] == FieldType.GEO]
        self.array_schema = [f for f in self.schema if f["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.DATE_ARRAY]]
        self.search_only_text_array_schema = [
            field
            for field in dm.schema_config
            if field["type"] == FieldType.TEXT_ARRAY
            and bool((dm.field_index_state.get(field["name"], {}) or {}).get("searchable"))
            and not bool((dm.field_index_state.get(field["name"], {}) or {}).get("filterable"))
        ]
        property_length_text_names = like_text_names
        property_length_array_names = {f["name"] for f in self.array_schema}
        self.property_length_schema = [
            f for f in dm.schema_config
            if (f["type"] == FieldType.TEXT and f["name"] in property_length_text_names)
            or (f["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.DATE_ARRAY] and f["name"] in property_length_array_names)
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

    def _current_df(self):
        current_df = getattr(self.dm, "df", None)
        if current_df is not None:
            return current_df
        return self.df

    def _normalize_mask(self, mask):
        current_df = self._current_df()
        current_index = current_df.index if current_df is not None else pd.RangeIndex(0)

        if mask is None:
            return pd.Series(False, index=current_index, dtype="boolean")

        if isinstance(mask, pd.Series):
            series = mask.copy()
        elif np.isscalar(mask):
            series = pd.Series([mask] * len(current_index), index=current_index)
        else:
            try:
                if len(mask) == len(current_index):
                    series = pd.Series(list(mask), index=current_index)
                else:
                    series = pd.Series(list(mask))
            except Exception:
                series = pd.Series([bool(mask)] * len(current_index), index=current_index)

        if not series.index.is_unique:
            series = series.groupby(level=0).last()
        if not series.index.equals(current_index):
            series = series.reindex(current_index)

        def to_nullable_bool(value):
            if pd.isna(value):
                return pd.NA
            return bool(value)

        return series.map(to_nullable_bool).astype("boolean")

    def _finalize_expr(self, fc, mask, expr):
        if fc is None:
            return None, None, expr
        return fc, self._normalize_mask(mask), expr

    def _is_effectively_null_value(self, value, ftype):
        if is_null_like(value):
            return True
        if ftype == FieldType.TEXT and value == "":
            return True
        if ftype in (FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.DATE_ARRAY):
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
            candidate = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"weaviate-fuzz-absent-{self.dm.seed}-{self._absent_uuid_counter}",
                )
            )
            self._absent_uuid_counter += 1
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

    def _scalar_cmp_query_value(self, ftype, value):
        if ftype == FieldType.INT:
            return int(value)
        if ftype == FieldType.NUMBER:
            return float(value)
        if ftype == FieldType.BOOL:
            return bool(value)
        if ftype == FieldType.TEXT:
            return str(value)
        if ftype == FieldType.DATE:
            ts = to_utc_timestamp(value)
            return ts.to_pydatetime() if ts is not None else None
        return value

    def _scalar_cmp_mask(self, series, ftype, op, target):
        if ftype == FieldType.DATE:
            mask = self._date_cmp_mask(series, op, target)
            if op == "!=":
                return self._normalize_mask(mask) | self._normalize_mask(self._effective_null_mask(series, ftype))
            return self._normalize_mask(mask)

        def compare(value):
            if self._is_effectively_null_value(value, ftype):
                return op == "!="
            try:
                left = int(value) if ftype == FieldType.INT else (
                    float(value) if ftype == FieldType.NUMBER else (
                        bool(value) if ftype == FieldType.BOOL else str(value)
                    )
                )
                right = int(target) if ftype == FieldType.INT else (
                    float(target) if ftype == FieldType.NUMBER else (
                        bool(target) if ftype == FieldType.BOOL else str(target)
                    )
                )
                if op == "==":
                    return left == right
                if op == "!=":
                    return left != right
                if op == ">":
                    return left > right
                if op == "<":
                    return left < right
                if op == ">=":
                    return left >= right
                if op == "<=":
                    return left <= right
            except Exception:
                return False
            return False

        return self._normalize_mask(series.apply(compare))

    def _build_scalar_cmp_filter_clause(self, name, ftype, op, target):
        query_value = self._scalar_cmp_query_value(ftype, target)
        if ftype == FieldType.DATE and query_value is None:
            return None, None, None
        method = {
            "==": "equal",
            "!=": "not_equal",
            ">": "greater_than",
            "<": "less_than",
            ">=": "greater_or_equal",
            "<=": "less_or_equal",
        }[op]
        fc = getattr(Filter.by_property(name), method)(query_value)
        mask = self._scalar_cmp_mask(self.df[name], ftype, op, target)
        display_value = canonicalize_date_string(target) if ftype == FieldType.DATE else target
        return fc, mask, f"{name} {op} {display_value}"

    def _scalar_contains_expr_value(self, ftype, value):
        if ftype == FieldType.INT:
            return int(value)
        if ftype == FieldType.NUMBER:
            return float(value)
        if ftype == FieldType.BOOL:
            return bool(value)
        if ftype == FieldType.DATE:
            return canonicalize_date_string(value) or str(value)
        return value

    def _scalar_contains_query_value(self, ftype, value):
        if ftype == FieldType.INT:
            return int(value)
        if ftype == FieldType.NUMBER:
            return float(value)
        if ftype == FieldType.BOOL:
            return bool(value)
        if ftype == FieldType.DATE:
            ts = to_utc_timestamp(value)
            return ts.to_pydatetime() if ts is not None else None
        return value

    def _scalar_contains_mask(self, series, ftype, targets, mode):
        expr_targets = tuple(self._scalar_contains_expr_value(ftype, target) for target in targets)
        if mode == "contains_all":
            expr_targets = expr_targets[:1]
        if ftype == FieldType.INT:
            if mode == "contains_any":
                return series.apply(lambda value, vals=expr_targets: int(value) in vals if not is_null_like(value) else False)
            if mode == "contains_none":
                return series.apply(lambda value, vals=expr_targets: int(value) not in vals if not is_null_like(value) else True)
            return series.apply(
                lambda value, vals=expr_targets: all(int(value) == target for target in vals)
                if not is_null_like(value)
                else False
            )
        if ftype == FieldType.NUMBER:
            if mode == "contains_any":
                return series.apply(lambda value, vals=expr_targets: float(value) in vals if not is_null_like(value) else False)
            if mode == "contains_none":
                return series.apply(lambda value, vals=expr_targets: float(value) not in vals if not is_null_like(value) else True)
            return series.apply(
                lambda value, vals=expr_targets: all(float(value) == target for target in vals)
                if not is_null_like(value)
                else False
            )
        if ftype == FieldType.BOOL:
            if mode == "contains_any":
                return series.apply(lambda value, vals=expr_targets: bool(value) in vals if not is_null_like(value) else False)
            if mode == "contains_none":
                return series.apply(lambda value, vals=expr_targets: bool(value) not in vals if not is_null_like(value) else True)
            return series.apply(
                lambda value, vals=expr_targets: all(bool(value) == target for target in vals)
                if not is_null_like(value)
                else False
            )
        if ftype == FieldType.DATE:
            if mode == "contains_any":
                return series.apply(
                    lambda value, vals=expr_targets: (canonicalize_date_string(value) or str(value)) in vals
                    if to_utc_timestamp(value) is not None
                    else False
                )
            if mode == "contains_none":
                return series.apply(
                    lambda value, vals=expr_targets: (canonicalize_date_string(value) or str(value)) not in vals
                    if to_utc_timestamp(value) is not None
                    else True
                )
            return series.apply(
                lambda value, vals=expr_targets: all((canonicalize_date_string(value) or str(value)) == target for target in vals)
                if to_utc_timestamp(value) is not None
                else False
            )
        return pd.Series(False, index=series.index, dtype=bool)

    def _build_scalar_contains_filter_clause(self, name, ftype, series, mode, targets):
        query_targets = []
        expr_targets = []
        for target in targets:
            query_value = self._scalar_contains_query_value(ftype, target)
            if ftype == FieldType.DATE and query_value is None:
                return None, None, None
            query_targets.append(query_value)
            expr_targets.append(self._scalar_contains_expr_value(ftype, target))
        if mode == "contains_all":
            query_targets = query_targets[:1]
            expr_targets = expr_targets[:1]
        filter_builder = getattr(Filter.by_property(name), mode)
        mask = self._scalar_contains_mask(series, ftype, targets, mode)
        return filter_builder(query_targets), self._normalize_mask(mask), f"{name} {mode} {expr_targets}"

    def _build_scalar_contains_rest_clause(self, name, ftype, series, mode, targets):
        rest_operator = {
            "contains_any": "ContainsAny",
            "contains_all": "ContainsAll",
            "contains_none": "ContainsNone",
        }[mode]
        rest_targets = [self._scalar_contains_expr_value(ftype, target) for target in targets]
        if mode == "contains_all":
            rest_targets = rest_targets[:1]
        mask = self._scalar_contains_mask(series, ftype, targets, mode)
        return build_rest_where([name], rest_operator, ftype=ftype, value=rest_targets), self._normalize_mask(mask), f"{name} {mode} {rest_targets}"

    def _scalar_contains_contrast_value(self, ftype, value):
        if ftype == FieldType.INT:
            primary = int(value)
            contrast = clamp_weaviate_int(primary + random.choice([1, 7, 13, 101, -1, -7, -13, -101]))
            if contrast == primary:
                contrast = clamp_weaviate_int(primary + 1)
            return contrast
        if ftype == FieldType.NUMBER:
            primary = float(value)
            contrast = float(primary + random.choice([-17.0, -0.25, 0.25, 17.0]))
            if contrast == primary:
                contrast += 1.0
            return contrast
        if ftype == FieldType.BOOL:
            return not bool(value)
        if ftype == FieldType.DATE:
            ts = to_utc_timestamp(value)
            if ts is None:
                return None
            return ts + random.choice(
                [pd.Timedelta(days=7), -pd.Timedelta(days=7), pd.Timedelta(seconds=1), -pd.Timedelta(seconds=1)]
            )
        return None

    def gen_scalar_core_compound_expr(self):
        candidates = [field for field in self.schema if field["type"] in (FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT, FieldType.DATE)]
        random.shuffle(candidates)
        for field in candidates:
            name, ftype = field["name"], field["type"]
            series = self.df[name]

            if ftype in (FieldType.INT, FieldType.NUMBER, FieldType.DATE):
                pivot = self.get_value_for_query(name, ftype)
                if pivot is None:
                    continue
                strategies = ["ge_and_not_lt", "le_and_not_gt", "closed_range"]
                strategy = random.choice(strategies)

                if strategy == "ge_and_not_lt":
                    ge_filter, ge_mask, ge_expr = self._build_scalar_cmp_filter_clause(name, ftype, ">=", pivot)
                    lt_filter, lt_mask, lt_expr = self._build_scalar_cmp_filter_clause(name, ftype, "<", pivot)
                    if ge_filter is None or lt_filter is None:
                        continue
                    return (
                        ge_filter & Filter.not_(lt_filter),
                        self._normalize_mask(ge_mask) & self._apply_not_mask(lt_mask),
                        f"({ge_expr} AND NOT({lt_expr}))",
                    )

                if strategy == "le_and_not_gt":
                    le_filter, le_mask, le_expr = self._build_scalar_cmp_filter_clause(name, ftype, "<=", pivot)
                    gt_filter, gt_mask, gt_expr = self._build_scalar_cmp_filter_clause(name, ftype, ">", pivot)
                    if le_filter is None or gt_filter is None:
                        continue
                    return (
                        le_filter & Filter.not_(gt_filter),
                        self._normalize_mask(le_mask) & self._apply_not_mask(gt_mask),
                        f"({le_expr} AND NOT({gt_expr}))",
                    )

                other = self.get_value_for_query(name, ftype)
                if other is None:
                    continue
                if ftype == FieldType.INT:
                    lo, hi = sorted([int(pivot), int(other)])
                elif ftype == FieldType.NUMBER:
                    lo, hi = sorted([float(pivot), float(other)])
                else:
                    lo_ts = to_utc_timestamp(pivot)
                    hi_ts = to_utc_timestamp(other)
                    if lo_ts is None or hi_ts is None:
                        continue
                    lo, hi = sorted([lo_ts, hi_ts])
                ge_filter, ge_mask, ge_expr = self._build_scalar_cmp_filter_clause(name, ftype, ">=", lo)
                le_filter, le_mask, le_expr = self._build_scalar_cmp_filter_clause(name, ftype, "<=", hi)
                if ge_filter is None or le_filter is None:
                    continue
                return (
                    ge_filter & le_filter,
                    self._normalize_mask(ge_mask) & self._normalize_mask(le_mask),
                    f"({ge_expr} AND {le_expr})",
                )

            if ftype == FieldType.BOOL:
                pivot = self.get_value_for_query(name, ftype)
                if pivot is None:
                    continue
                pivot = bool(pivot)
                eq_filter, eq_mask, eq_expr = self._build_scalar_cmp_filter_clause(name, ftype, "==", pivot)
                opp_filter, opp_mask, opp_expr = self._build_scalar_cmp_filter_clause(name, ftype, "==", not pivot)
                if eq_filter is None or opp_filter is None:
                    continue
                return (
                    eq_filter & Filter.not_(opp_filter),
                    self._normalize_mask(eq_mask) & self._apply_not_mask(opp_mask),
                    f"({eq_expr} AND NOT({opp_expr}))",
                )

            pivot = self.get_value_for_query(name, ftype)
            if pivot is None:
                continue
            eq_filter, eq_mask, eq_expr = self._build_scalar_cmp_filter_clause(name, ftype, "==", pivot)
            neq_filter, neq_mask, neq_expr = self._build_scalar_cmp_filter_clause(name, ftype, "!=", pivot)
            if eq_filter is None or neq_filter is None:
                continue
            return (
                eq_filter & Filter.not_(neq_filter),
                self._normalize_mask(eq_mask) & self._apply_not_mask(neq_mask),
                f"({eq_expr} AND NOT({neq_expr}))",
            )

        return None, None, None

    def gen_scalar_contains_compound_expr(self):
        candidates = [field for field in self.schema if field["type"] in (FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.DATE)]
        random.shuffle(candidates)
        for field in candidates:
            name, ftype = field["name"], field["type"]
            series = self.df[name]
            value = self.get_value_for_query(name, ftype)
            if value is None:
                continue
            contrast = self._scalar_contains_contrast_value(ftype, value)
            strategies = ["not_contains_any", "contains_any_or_null", "contains_all_and_not_none"]
            if ftype in (FieldType.INT, FieldType.NUMBER, FieldType.DATE):
                strategies.append("contains_none_and_range")
            strategy = random.choice(strategies)
            if strategy == "not_contains_any":
                targets = [value]
                if contrast is not None and random.random() < 0.55:
                    targets.append(contrast)
                contains_filter, contains_mask, contains_expr = self._build_scalar_contains_filter_clause(
                    name, ftype, series, "contains_any", targets
                )
                if contains_filter is None:
                    continue
                return Filter.not_(contains_filter), self._apply_not_mask(contains_mask), f"NOT({contains_expr})"
            if strategy == "contains_any_or_null":
                targets = [value]
                if contrast is not None:
                    targets.append(contrast)
                contains_filter, contains_mask, contains_expr = self._build_scalar_contains_filter_clause(
                    name, ftype, series, "contains_any", targets
                )
                if contains_filter is None:
                    continue
                null_filter = Filter.by_property(name).is_none(True)
                null_mask = self._effective_null_mask(series, ftype)
                return (
                    contains_filter | null_filter,
                    self._normalize_mask(contains_mask) | self._normalize_mask(null_mask),
                    f"({contains_expr} OR {name} is null)",
                )
            if strategy == "contains_all_and_not_none":
                contains_all_filter, contains_all_mask, contains_all_expr = self._build_scalar_contains_filter_clause(
                    name, ftype, series, "contains_all", [value]
                )
                contains_none_filter, contains_none_mask, contains_none_expr = self._build_scalar_contains_filter_clause(
                    name, ftype, series, "contains_none", [value]
                )
                if contains_all_filter is None or contains_none_filter is None:
                    continue
                return (
                    contains_all_filter & Filter.not_(contains_none_filter),
                    self._normalize_mask(contains_all_mask) & self._apply_not_mask(contains_none_mask),
                    f"({contains_all_expr} AND NOT({contains_none_expr}))",
                )
            if contrast is None:
                continue
            contains_none_filter, contains_none_mask, contains_none_expr = self._build_scalar_contains_filter_clause(
                name, ftype, series, "contains_none", [contrast]
            )
            if contains_none_filter is None:
                continue
            if ftype == FieldType.INT:
                pivot = int(value)
                range_filter = Filter.by_property(name).greater_or_equal(pivot)
                range_mask = series.apply(lambda item, target=pivot: int(item) >= target if not is_null_like(item) else False)
                return (
                    contains_none_filter & range_filter,
                    self._normalize_mask(contains_none_mask) & self._normalize_mask(range_mask),
                    f"({contains_none_expr} AND {name} >= {pivot})",
                )
            if ftype == FieldType.NUMBER:
                pivot = float(value)
                lo, hi = float_window(pivot)
                if lo is None or hi is None:
                    continue
                range_filter = Filter.by_property(name).greater_or_equal(lo) & Filter.by_property(name).less_or_equal(hi)
                range_mask = series.apply(
                    lambda item, left=lo, right=hi: left <= float(item) <= right if not is_null_like(item) else False
                )
                return (
                    contains_none_filter & range_filter,
                    self._normalize_mask(contains_none_mask) & self._normalize_mask(range_mask),
                    f"({contains_none_expr} AND {name} in [{lo}, {hi}])",
                )
            pivot = canonicalize_date_string(value) or str(value)
            pivot_ts = to_utc_timestamp(value)
            if pivot_ts is None:
                continue
            pivot_dt = pivot_ts.to_pydatetime()
            range_filter = Filter.by_property(name).greater_or_equal(pivot_dt) & Filter.by_property(name).less_or_equal(pivot_dt)
            range_mask = series.apply(
                lambda item, target=pivot: (canonicalize_date_string(item) or str(item)) == target
                if to_utc_timestamp(item) is not None
                else False
            )
            return (
                contains_none_filter & range_filter,
                self._normalize_mask(contains_none_mask) & self._normalize_mask(range_mask),
                f"({contains_none_expr} AND {name} in [{pivot}, {pivot}])",
            )
        return None, None, None

    def gen_scalar_contains_compound_rest_expr(self):
        candidates = [field for field in self.schema if field["type"] in (FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.DATE)]
        random.shuffle(candidates)
        for field in candidates:
            name, ftype = field["name"], field["type"]
            series = self.df[name]
            value = self.get_value_for_query(name, ftype)
            if value is None:
                continue
            contrast = self._scalar_contains_contrast_value(ftype, value)
            strategies = ["not_contains_any", "contains_any_or_null", "contains_all_and_not_none"]
            if ftype in (FieldType.INT, FieldType.NUMBER, FieldType.DATE):
                strategies.append("contains_none_and_range")
            strategy = random.choice(strategies)
            if strategy == "not_contains_any":
                targets = [value]
                if contrast is not None and random.random() < 0.55:
                    targets.append(contrast)
                contains_where, contains_mask, contains_expr = self._build_scalar_contains_rest_clause(
                    name, ftype, series, "contains_any", targets
                )
                if contains_where is None:
                    continue
                return build_rest_compound("Not", [contains_where]), self._apply_not_mask(contains_mask), f"NOT({contains_expr})"
            if strategy == "contains_any_or_null":
                targets = [value]
                if contrast is not None:
                    targets.append(contrast)
                contains_where, contains_mask, contains_expr = self._build_scalar_contains_rest_clause(
                    name, ftype, series, "contains_any", targets
                )
                if contains_where is None:
                    continue
                null_where = build_rest_where([name], "IsNull", raw_key="valueBoolean", value=True)
                null_mask = self._effective_null_mask(series, ftype)
                return (
                    build_rest_compound("Or", [contains_where, null_where]),
                    self._normalize_mask(contains_mask) | self._normalize_mask(null_mask),
                    f"({contains_expr} OR {name} is null)",
                )
            if strategy == "contains_all_and_not_none":
                contains_all_where, contains_all_mask, contains_all_expr = self._build_scalar_contains_rest_clause(
                    name, ftype, series, "contains_all", [value]
                )
                contains_none_where, contains_none_mask, contains_none_expr = self._build_scalar_contains_rest_clause(
                    name, ftype, series, "contains_none", [value]
                )
                if contains_all_where is None or contains_none_where is None:
                    continue
                return (
                    build_rest_compound("And", [contains_all_where, build_rest_compound("Not", [contains_none_where])]),
                    self._normalize_mask(contains_all_mask) & self._apply_not_mask(contains_none_mask),
                    f"({contains_all_expr} AND NOT({contains_none_expr}))",
                )
            if contrast is None:
                continue
            contains_none_where, contains_none_mask, contains_none_expr = self._build_scalar_contains_rest_clause(
                name, ftype, series, "contains_none", [contrast]
            )
            if contains_none_where is None:
                continue
            if ftype == FieldType.INT:
                pivot = int(value)
                range_where = build_rest_where([name], "GreaterThanEqual", ftype=FieldType.INT, value=pivot)
                range_mask = series.apply(lambda item, target=pivot: int(item) >= target if not is_null_like(item) else False)
                return (
                    build_rest_compound("And", [contains_none_where, range_where]),
                    self._normalize_mask(contains_none_mask) & self._normalize_mask(range_mask),
                    f"({contains_none_expr} AND {name} >= {pivot})",
                )
            if ftype == FieldType.NUMBER:
                pivot = float(value)
                lo, hi = float_window(pivot)
                if lo is None or hi is None:
                    continue
                range_where = build_rest_compound(
                    "And",
                    [
                        build_rest_where([name], "GreaterThanEqual", ftype=FieldType.NUMBER, value=lo),
                        build_rest_where([name], "LessThanEqual", ftype=FieldType.NUMBER, value=hi),
                    ],
                )
                range_mask = series.apply(
                    lambda item, left=lo, right=hi: left <= float(item) <= right if not is_null_like(item) else False
                )
                return (
                    build_rest_compound("And", [contains_none_where, range_where]),
                    self._normalize_mask(contains_none_mask) & self._normalize_mask(range_mask),
                    f"({contains_none_expr} AND {name} in [{lo}, {hi}])",
                )
            pivot = canonicalize_date_string(value) or str(value)
            pivot_ts = to_utc_timestamp(value)
            if pivot_ts is None:
                continue
            range_where = build_rest_compound(
                "And",
                [
                    build_rest_where([name], "GreaterThanEqual", ftype=FieldType.DATE, value=pivot),
                    build_rest_where([name], "LessThanEqual", ftype=FieldType.DATE, value=pivot),
                ],
            )
            range_mask = series.apply(
                lambda item, target=pivot: (canonicalize_date_string(item) or str(item)) == target
                if to_utc_timestamp(item) is not None
                else False
            )
            return (
                build_rest_compound("And", [contains_none_where, range_where]),
                self._normalize_mask(contains_none_mask) & self._normalize_mask(range_mask),
                f"({contains_none_expr} AND {name} in [{pivot}, {pivot}])",
            )
        return None, None, None

    def _text_contains_mask(self, series, targets, mode):
        expr_targets = tuple(stable_unique_values(str(target) for target in targets))
        if mode == "contains_all":
            expr_targets = expr_targets[:1]
        target_set = set(expr_targets)
        if mode == "contains_any":
            return series.apply(
                lambda value, vals=target_set: str(value) in vals
                if value is not None and not (isinstance(value, float) and np.isnan(value))
                else False
            )
        if mode == "contains_none":
            return series.apply(
                lambda value, vals=target_set: str(value) not in vals
                if value is not None and not (isinstance(value, float) and np.isnan(value))
                else True
            )
        return series.apply(
            lambda value, vals=expr_targets: all(str(value) == target for target in vals)
            if value is not None and not (isinstance(value, float) and np.isnan(value))
            else False
        )

    def _build_text_contains_filter_clause(self, name, series, mode, targets):
        expr_targets = stable_unique_values(str(target) for target in targets)
        if mode == "contains_all":
            expr_targets = expr_targets[:1]
        filter_builder = getattr(Filter.by_property(name), mode)
        mask = self._text_contains_mask(series, expr_targets, mode)
        return filter_builder(expr_targets), self._normalize_mask(mask), f'{name} {mode} {expr_targets}'

    def _build_text_contains_rest_clause(self, name, series, mode, targets):
        rest_operator = {
            "contains_any": "ContainsAny",
            "contains_all": "ContainsAll",
            "contains_none": "ContainsNone",
        }[mode]
        expr_targets = stable_unique_values(str(target) for target in targets)
        if mode == "contains_all":
            expr_targets = expr_targets[:1]
        mask = self._text_contains_mask(series, expr_targets, mode)
        return build_rest_where([name], rest_operator, ftype=FieldType.TEXT, value=expr_targets), self._normalize_mask(mask), f'{name} {mode} {expr_targets}'

    def _like_pattern_variants(self, value, *, allow_boundary=False):
        sv = str(value)
        variants = []
        if allow_boundary:
            variants.extend([("*", "all"), ("**", "all2")])
        if len(sv) == 3:
            variants.append(("???", "three_q"))
        if sv in {"%", "_"}:
            variants.append((sv, "literal_boundary"))
        if len(sv) >= 1:
            k = random.randint(1, min(4, len(sv)))
            prefix = "".join(ch for ch in sv[:k] if ch.isalnum()) or sv[:1] or "a"
            variants.append((f"{prefix}*", "prefix"))
        if len(sv) >= 2:
            k = random.randint(1, min(4, len(sv)))
            suffix = "".join(ch for ch in sv[-k:] if ch.isalnum()) or sv[-1:] or "z"
            variants.append((f"*{suffix}", "suffix"))
            pos = random.randint(0, len(sv) - 1)
            variants.append((sv[:pos] + "?" + sv[pos + 1 :], "single_q"))
        if len(sv) >= 3:
            i = random.randint(0, len(sv) - 2)
            j = random.randint(i + 1, min(i + 5, len(sv)))
            sub = "".join(ch for ch in sv[i:j] if ch.isalnum()) or sv[i:j] or "x"
            variants.append((f"*{sub}*", "contains"))
        if not variants:
            variants.append(("*", "fallback"))
        return stable_unique_values(variants)

    def gen_like_compound_expr(self):
        text_fields = self.like_text_schema[:]
        random.shuffle(text_fields)
        for field in text_fields:
            name = field["name"]
            series = self.df[name]
            value = self.get_value_for_query(name, FieldType.TEXT)
            if value is None:
                continue
            sv = str(value)
            pattern, _ = random.choice(self._like_pattern_variants(sv, allow_boundary=True))
            like_filter = Filter.by_property(name).like(pattern)
            like_mask = rest_like_mask(series, pattern)
            null_mask = self._effective_null_mask(series, FieldType.TEXT)
            candidates = stable_unique_values(str(item) for item in self._field_values.get(name, []) if str(item) != sv)
            contrast = random.choice(candidates) if candidates else self._random_string(6, 12)
            strategy = random.choice(["not_like", "like_or_null", "like_and_not_contains_none_self", "like_and_contains_none_other"])
            if strategy == "not_like":
                return Filter.not_(like_filter), self._apply_not_mask(like_mask), f'NOT({name} like "{pattern}")'
            if strategy == "like_or_null":
                return like_filter | Filter.by_property(name).is_none(True), self._normalize_mask(like_mask) | self._normalize_mask(null_mask), f'({name} like "{pattern}" OR {name} is null)'
            if strategy == "like_and_not_contains_none_self":
                contains_none_filter, contains_none_mask, contains_none_expr = self._build_text_contains_filter_clause(
                    name, series, "contains_none", [sv]
                )
                return (
                    like_filter & Filter.not_(contains_none_filter),
                    self._normalize_mask(like_mask) & self._apply_not_mask(contains_none_mask),
                    f'({name} like "{pattern}" AND NOT({contains_none_expr}))',
                )
            contains_none_filter, contains_none_mask, contains_none_expr = self._build_text_contains_filter_clause(
                name, series, "contains_none", [contrast]
            )
            return (
                like_filter & contains_none_filter,
                self._normalize_mask(like_mask) & self._normalize_mask(contains_none_mask),
                f'({name} like "{pattern}" AND {contains_none_expr})',
            )
        return None, None, None

    def gen_like_compound_rest_expr(self):
        text_fields = self.like_text_schema[:]
        random.shuffle(text_fields)
        for field in text_fields:
            name = field["name"]
            series = self.df[name]
            value = self.get_value_for_query(name, FieldType.TEXT)
            if value is None:
                continue
            sv = str(value)
            pattern, _ = random.choice(self._like_pattern_variants(sv, allow_boundary=True))
            like_where = build_rest_where([name], "Like", ftype=FieldType.TEXT, value=pattern)
            like_mask = rest_like_mask(series, pattern)
            null_mask = self._effective_null_mask(series, FieldType.TEXT)
            candidates = stable_unique_values(str(item) for item in self._field_values.get(name, []) if str(item) != sv)
            contrast = random.choice(candidates) if candidates else self._random_string(6, 12)
            strategy = random.choice(["not_like", "like_or_null", "like_and_not_contains_none_self", "like_and_contains_none_other"])
            if strategy == "not_like":
                return build_rest_compound("Not", [like_where]), self._apply_not_mask(like_mask), f'NOT({name} like "{pattern}")'
            if strategy == "like_or_null":
                return (
                    build_rest_compound("Or", [like_where, build_rest_where([name], "IsNull", raw_key="valueBoolean", value=True)]),
                    self._normalize_mask(like_mask) | self._normalize_mask(null_mask),
                    f'({name} like "{pattern}" OR {name} is null)',
                )
            if strategy == "like_and_not_contains_none_self":
                contains_none_where, contains_none_mask, contains_none_expr = self._build_text_contains_rest_clause(
                    name, series, "contains_none", [sv]
                )
                return (
                    build_rest_compound("And", [like_where, build_rest_compound("Not", [contains_none_where])]),
                    self._normalize_mask(like_mask) & self._apply_not_mask(contains_none_mask),
                    f'({name} like "{pattern}" AND NOT({contains_none_expr}))',
                )
            contains_none_where, contains_none_mask, contains_none_expr = self._build_text_contains_rest_clause(
                name, series, "contains_none", [contrast]
            )
            return (
                build_rest_compound("And", [like_where, contains_none_where]),
                self._normalize_mask(like_mask) & self._normalize_mask(contains_none_mask),
                f'({name} like "{pattern}" AND {contains_none_expr})',
            )
        return None, None, None

    def gen_true_like_compound(self, row):
        text_fields = self.like_text_schema[:]
        random.shuffle(text_fields)
        for field in text_fields:
            name = field["name"]
            value = row.get(name)
            if self._is_effectively_null_value(value, FieldType.TEXT):
                continue
            series = self.df[name]
            sv = str(value)
            pattern, _ = random.choice(self._like_pattern_variants(sv, allow_boundary=True))
            strategy = random.choice(["like_or_null", "like_and_not_contains_none_self"])
            if strategy == "like_or_null":
                return (
                    Filter.by_property(name).like(pattern) | Filter.by_property(name).is_none(True),
                    f'({name} like "{pattern}" OR {name} is null)',
                )
            contains_none_filter, _, contains_none_expr = self._build_text_contains_filter_clause(name, series, "contains_none", [sv])
            return (
                Filter.by_property(name).like(pattern) & Filter.not_(contains_none_filter),
                f'({name} like "{pattern}" AND NOT({contains_none_expr}))',
            )
        return None, None

    def gen_true_scalar_contains_compound(self, row):
        fields = [field for field in self.schema if field["type"] in (FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.DATE)]
        random.shuffle(fields)
        for field in fields:
            name, ftype = field["name"], field["type"]
            value = row.get(name)
            if self._is_effectively_null_value(value, ftype):
                continue
            series = self.df[name]
            contrast = self._scalar_contains_contrast_value(ftype, value)
            strategies = ["contains_all_and_not_none", "contains_any_or_null"]
            if contrast is not None:
                strategies.append("contains_none_guard")
            strategy = random.choice(strategies)
            if strategy == "contains_all_and_not_none":
                contains_all_filter, _, contains_all_expr = self._build_scalar_contains_filter_clause(
                    name, ftype, series, "contains_all", [value]
                )
                contains_none_filter, contains_none_mask, contains_none_expr = self._build_scalar_contains_filter_clause(
                    name, ftype, series, "contains_none", [value]
                )
                if contains_all_filter is None or contains_none_filter is None:
                    continue
                return (
                    contains_all_filter & Filter.not_(contains_none_filter),
                    f"({contains_all_expr} AND NOT({contains_none_expr}))",
                )
            if strategy == "contains_any_or_null":
                targets = [value]
                if contrast is not None:
                    targets.append(contrast)
                contains_any_filter, _, contains_any_expr = self._build_scalar_contains_filter_clause(
                    name, ftype, series, "contains_any", targets
                )
                if contains_any_filter is None:
                    continue
                return contains_any_filter | Filter.by_property(name).is_none(True), f"({contains_any_expr} OR {name} is null)"
            contains_none_filter, _, contains_none_expr = self._build_scalar_contains_filter_clause(
                name, ftype, series, "contains_none", [contrast]
            )
            if contains_none_filter is None:
                continue
            if ftype == FieldType.BOOL:
                guard = Filter.by_property(name).equal(bool(value))
                return contains_none_filter & guard, f"({contains_none_expr} AND {name} == {bool(value)})"
            if ftype == FieldType.INT:
                pivot = int(value)
                guard = Filter.by_property(name).greater_or_equal(pivot)
                return contains_none_filter & guard, f"({contains_none_expr} AND {name} >= {pivot})"
            if ftype == FieldType.NUMBER:
                pivot = float(value)
                lo, hi = float_window(pivot)
                if lo is None or hi is None:
                    continue
                guard = Filter.by_property(name).greater_or_equal(lo) & Filter.by_property(name).less_or_equal(hi)
                return contains_none_filter & guard, f"({contains_none_expr} AND {name} in [{lo}, {hi}])"
            pivot = canonicalize_date_string(value) or str(value)
            pivot_ts = to_utc_timestamp(value)
            if pivot_ts is None:
                continue
            pivot_dt = pivot_ts.to_pydatetime()
            guard = Filter.by_property(name).greater_or_equal(pivot_dt) & Filter.by_property(name).less_or_equal(pivot_dt)
            return contains_none_filter & guard, f"({contains_none_expr} AND {name} in [{pivot}, {pivot}])"
        return None, None

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
        arr_fields = [
            f for f in self.schema if f["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.DATE_ARRAY]
        ]
        for field in self.search_only_text_array_schema:
            if field["name"] not in {candidate["name"] for candidate in arr_fields}:
                arr_fields.append(field)
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
        unique_items = stable_unique_values(all_items)
        num = min(random.randint(2, 4), len(unique_items))
        if num < 1:
            return None, None, None
        targets = [self._convert_to_native(t) for t in random.sample(unique_items, num)]

        mode = random.choice(["contains_any", "contains_all", "contains_none"])
        # Tokenization.FIELD → 精确匹配, 不再需要 .lower() 变换
        if mode == "contains_any":
            fc = Filter.by_property(fname).contains_any(targets)
            item_type = rest_array_item_type(ftype)
            mask = array_membership_mask(series, item_type, targets, "contains_any")
            return fc, mask, f"{fname} contains_any {targets}"
        elif mode == "contains_all":
            fc = Filter.by_property(fname).contains_all(targets)
            item_type = rest_array_item_type(ftype)
            mask = array_membership_mask(series, item_type, targets, "contains_all")
            return fc, mask, f"{fname} contains_all {targets}"
        else:
            # contains_none: 不含 targets 中任何值; null 数组视为匹配
            fc = Filter.by_property(fname).contains_none(targets)
            item_type = rest_array_item_type(ftype)
            mask = array_membership_mask(series, item_type, targets, "contains_none")
            return fc, mask, f"{fname} contains_none {targets}"

    def _build_array_contains_filter_clause(self, name, ftype, series, mode, targets):
        item_type = rest_array_item_type(ftype)
        if item_type is None:
            return None, None, None

        expr_targets = [self._convert_to_native(target) for target in targets]
        if item_type == FieldType.DATE:
            expr_targets = [canonicalize_date_string(target) or str(target) for target in expr_targets]
            filter_targets = []
            for target in expr_targets:
                ts = pd.Timestamp(target)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                filter_targets.append(ts.to_pydatetime())
        else:
            filter_targets = expr_targets

        fc = getattr(Filter.by_property(name), mode)(filter_targets)
        mask = array_membership_mask(series, item_type, expr_targets, mode)
        return fc, self._normalize_mask(mask), f"{name} {mode} {expr_targets}"

    def gen_array_contains_compound_expr(self):
        arr_fields = [
            f for f in self.schema
            if f["type"] in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.DATE_ARRAY]
        ]
        if not arr_fields:
            return None, None, None

        field = random.choice(arr_fields)
        fname, ftype = field["name"], field["type"]
        series = self.df[fname]
        arrays = [arr for arr in series.dropna().tolist() if isinstance(arr, list) and len(arr) > 0]
        if not arrays:
            return None, None, None

        unique_items = stable_unique_values([item for arr in arrays for item in arr])
        if not unique_items:
            return None, None, None

        strategies = ["contains_any_and_not_none_self", "contains_any_or_null"]
        if len(unique_items) >= 2:
            strategies.append("contains_any_and_none_other")
        if any(len(stable_unique_values(arr)) >= 2 for arr in arrays):
            strategies.append("contains_all_and_not_none_self")

        strategy = random.choice(strategies)

        if strategy == "contains_any_or_null":
            sample_size = min(len(unique_items), random.choice([1, 1, 2]))
            targets = random.sample(unique_items, sample_size)
            contains_any_filter, contains_any_mask, contains_any_expr = self._build_array_contains_filter_clause(
                fname, ftype, series, "contains_any", targets
            )
            if contains_any_filter is None:
                return None, None, None
            null_filter = Filter.by_property(fname).is_none(True)
            null_mask = self._effective_null_mask(series, ftype)
            return (
                contains_any_filter | null_filter,
                self._normalize_mask(contains_any_mask) | self._normalize_mask(null_mask),
                f"({contains_any_expr} OR {fname} is null)",
            )

        if strategy == "contains_all_and_not_none_self":
            candidate_arrays = [
                arr for arr in arrays
                if len(stable_unique_values(arr)) >= 2
            ]
            if not candidate_arrays:
                return None, None, None
            picked = stable_unique_values(random.choice(candidate_arrays))
            targets = picked[: min(len(picked), random.choice([1, 2]))]
            contains_all_filter, contains_all_mask, contains_all_expr = self._build_array_contains_filter_clause(
                fname, ftype, series, "contains_all", targets
            )
            contains_none_filter, contains_none_mask, contains_none_expr = self._build_array_contains_filter_clause(
                fname, ftype, series, "contains_none", targets
            )
            if contains_all_filter is None or contains_none_filter is None:
                return None, None, None
            return (
                contains_all_filter & Filter.not_(contains_none_filter),
                self._normalize_mask(contains_all_mask) & self._apply_not_mask(contains_none_mask),
                f"({contains_all_expr} AND NOT({contains_none_expr}))",
            )

        sample_size = min(len(unique_items), random.choice([1, 1, 2]))
        hit_targets = random.sample(unique_items, sample_size)
        contains_any_filter, contains_any_mask, contains_any_expr = self._build_array_contains_filter_clause(
            fname, ftype, series, "contains_any", hit_targets
        )
        if contains_any_filter is None:
            return None, None, None

        if strategy == "contains_any_and_not_none_self":
            contains_none_filter, contains_none_mask, contains_none_expr = self._build_array_contains_filter_clause(
                fname, ftype, series, "contains_none", hit_targets
            )
            if contains_none_filter is None:
                return None, None, None
            return (
                contains_any_filter & Filter.not_(contains_none_filter),
                self._normalize_mask(contains_any_mask) & self._apply_not_mask(contains_none_mask),
                f"({contains_any_expr} AND NOT({contains_none_expr}))",
            )

        contrast_pool = [item for item in unique_items if item not in hit_targets]
        if not contrast_pool:
            return None, None, None
        contrast_targets = random.sample(contrast_pool, min(len(contrast_pool), random.choice([1, 1, 2])))
        contains_none_filter, contains_none_mask, contains_none_expr = self._build_array_contains_filter_clause(
            fname, ftype, series, "contains_none", contrast_targets
        )
        if contains_none_filter is None:
            return None, None, None
        return (
            contains_any_filter & contains_none_filter,
            self._normalize_mask(contains_any_mask) & self._normalize_mask(contains_none_mask),
            f"({contains_any_expr} AND {contains_none_expr})",
        )

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
            unique_ids = stable_unique_values(id_values)
            sample_size = min(len(unique_ids), random.randint(1, 3))
            targets = random.sample(unique_ids, sample_size)
            if random.random() < 0.5:
                targets.append(fake_id)
            targets = list(dict.fromkeys(targets))
            return Filter.by_id().contains_any(targets), id_series.isin(targets), f"id contains_any {targets}"

        if mode == "contains_any_absent":
            targets = [fake_id]
            return Filter.by_id().contains_any(targets), id_series.isin(targets), f"id contains_any {targets}"

        unique_ids = stable_unique_values(id_values)
        sample_size = min(len(unique_ids), random.randint(1, 3))
        targets = random.sample(unique_ids, sample_size)
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

    def gen_search_only_text_exact_expr(self):
        if not self.search_only_text_schema:
            return None, None, None

        field = random.choice(self.search_only_text_schema)
        name = field["name"]
        series = self.df[name]
        null_mask = self._effective_null_mask(series, FieldType.TEXT)
        target = self.get_value_for_query(name, FieldType.TEXT)
        if target is None:
            return None, None, None

        target = str(target)
        if random.random() < 0.5:
            return (
                Filter.by_property(name).equal(target),
                series.apply(
                    lambda value, tv=target: str(value) == tv
                    if value is not None and not (isinstance(value, float) and np.isnan(value))
                    else False
                ),
                f'{name} == "{target}" [search-only]',
            )

        return (
            Filter.by_property(name).not_equal(target),
            series.apply(
                lambda value, tv=target: str(value) != tv
                if value is not None and not (isinstance(value, float) and np.isnan(value))
                else True
            ) | null_mask,
            f'{name} != "{target}" [search-only]',
        )

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
        """Nested Object 查询 (gen_json_advanced_expr)
        NOTE: Local Weaviate v1.36.10 仅验证了 OBJECT 顶层 null-state。
        `operator_test/weaviate/nested_object_filter_operator.py` 显示
        `path:["metaObj","price"]` 这类 child path 会被 GraphQL/REST 解析器拒绝，
        报错 `missing an argument after 'price'`，因此当前主 fuzzer 不生成该类路径。
        顶层仅可靠使用 is_none(True) 以及 NOT(is_none(True))；direct is_none(False) 仍属于未实现/不支持边界。
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

        elif strat == "not_contains" and ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.DATE_ARRAY]:
            all_items = []
            for arr in self.df[name].dropna():
                if isinstance(arr, list): all_items.extend(arr)
            # Tokenization.FIELD: 不需要停用词过滤
            if all_items:
                t = self._convert_to_native(random.choice(all_items))
                fc = Filter.not_(Filter.by_property(name).contains_any([t]))
                item_type = rest_array_item_type(ftype)
                return fc, array_membership_mask(series, item_type, [t], "contains_none"), f"NOT({name} contains {t})"

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
        id_rate = 0.14 if self._inverted_profile else 0.10
        metadata_rate = 0.14 if self._inverted_profile else 0.10
        property_length_rate = 0.22 if self._inverted_profile else 0.12
        like_rate = 0.36 if self._inverted_profile else 0.24

        def boundary_like_expr(field_name, field_series):
            pattern = random.choice(["*", "**", "???", "%", "_"])
            return (
                Filter.by_property(field_name).like(pattern),
                rest_like_mask(field_series, pattern),
                f'{field_name} like "{pattern}"',
            )

        if random.random() < id_rate:
            res = self.gen_id_expr()
            if res[0] is not None:
                return res

        if self.creation_time_series is not None and random.random() < metadata_rate:
            res = self.gen_creation_time_expr()
            if res[0] is not None:
                return res

        if self.update_time_series is not None and random.random() < metadata_rate:
            res = self.gen_update_time_expr()
            if res[0] is not None:
                return res

        if self.property_length_schema and random.random() < property_length_rate:
            res = self.gen_property_length_expr()
            if res[0] is not None:
                return res

        search_only_exact_rate = 0.18 if self._inverted_profile else 0.0
        if self.search_only_text_schema and random.random() < search_only_exact_rate:
            res = self.gen_search_only_text_exact_expr()
            if res[0] is not None:
                return res

        if self.like_text_schema and random.random() < like_rate:
            f = random.choice(self.like_text_schema)
            name = f["name"]
            series = self.df[name]
            sv = self.get_value_for_query(name, FieldType.TEXT)
            if sv is not None:
                sv = str(sv)
                if random.random() < 0.12:
                    return boundary_like_expr(name, series)
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
        scalar_priority = 0.97 if self._inverted_profile else SCALAR_QUERY_PRIORITY
        if self.scalar_schema and random.random() < scalar_priority:
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
            mode = random.choices(
                ["equal", "not_equal", "contains_any", "contains_none", "contains_all"],
                weights=[0.34, 0.18, 0.18, 0.18, 0.12],
                k=1,
            )[0]
            if mode == "not_equal":
                fc = Filter.by_property(name).not_equal(vb)
                mask = series.apply(safe_cmp("!=", vb)) | null_mask
                es = f"{name} != {vb}"
            elif mode == "contains_any":
                fc = Filter.by_property(name).contains_any([vb])
                mask = series.apply(lambda x, target=vb: bool(x) == target if not is_null_like(x) else False)
                es = f"{name} contains_any [{vb}]"
            elif mode == "contains_none":
                fc = Filter.by_property(name).contains_none([vb])
                mask = series.apply(lambda x, target=vb: bool(x) != target if not is_null_like(x) else True)
                es = f"{name} contains_none [{vb}]"
            elif mode == "contains_all":
                fc = Filter.by_property(name).contains_all([vb])
                mask = series.apply(lambda x, target=vb: bool(x) == target if not is_null_like(x) else False)
                es = f"{name} contains_all [{vb}]"
            else:
                fc = Filter.by_property(name).equal(vb)
                mask = series.apply(safe_cmp("==", vb))
                es = f"{name} == {vb}"

        elif ftype == FieldType.INT:
            vi = int(val)
            op = random.choices(
                [">", "<", "==", ">=", "<=", "!=", "contains_any", "contains_none", "contains_all"],
                weights=[0.11, 0.11, 0.14, 0.10, 0.10, 0.10, 0.12, 0.12, 0.10],
                k=1,
            )[0]
            op_map = {"==": "equal", "!=": "not_equal", ">": "greater_than", "<": "less_than", ">=": "greater_or_equal", "<=": "less_or_equal"}
            if op == "contains_any":
                targets = [vi]
                if random.random() < 0.55:
                    targets.append(clamp_weaviate_int(vi + random.choice([7, 13, 101, -7, -13, -101])))
                target_set = set(targets)
                fc = Filter.by_property(name).contains_any(targets)
                mask = series.apply(lambda x, vals=target_set: int(x) in vals if not is_null_like(x) else False)
                es = f"{name} contains_any {targets}"
            elif op == "contains_none":
                targets = [vi]
                if random.random() < 0.55:
                    targets.append(clamp_weaviate_int(vi + random.choice([7, 13, 101, -7, -13, -101])))
                target_set = set(targets)
                fc = Filter.by_property(name).contains_none(targets)
                mask = series.apply(lambda x, vals=target_set: int(x) not in vals if not is_null_like(x) else True)
                es = f"{name} contains_none {targets}"
            elif op == "contains_all":
                fc = Filter.by_property(name).contains_all([vi])
                mask = series.apply(lambda x, target=vi: int(x) == target if not is_null_like(x) else False)
                es = f"{name} contains_all [{vi}]"
            else:
                fc = getattr(Filter.by_property(name), op_map[op])(vi)
                mask = series.apply(safe_cmp(op, vi))
                es = f"{name} {op} {vi}"
            if op == "!=":
                mask = mask | null_mask  # Weaviate not_equal includes null rows

        elif ftype == FieldType.NUMBER:
            vf = float(val)
            op = random.choices(
                [">", "<", ">=", "<=", "!=", "contains_any", "contains_none", "contains_all"],
                weights=[0.14, 0.14, 0.12, 0.12, 0.12, 0.14, 0.12, 0.10],
                k=1,
            )[0]
            op_map = {"!=": "not_equal", ">": "greater_than", "<": "less_than", ">=": "greater_or_equal", "<=": "less_or_equal"}
            if op == "contains_any":
                targets = [vf]
                if random.random() < 0.55:
                    noise = float(vf + random.choice([-17.0, -0.25, 0.25, 17.0]))
                    targets.append(noise)
                target_set = set(targets)
                fc = Filter.by_property(name).contains_any(targets)
                mask = series.apply(lambda x, vals=target_set: float(x) in vals if not is_null_like(x) else False)
                es = f"{name} contains_any {targets}"
            elif op == "contains_none":
                targets = [vf]
                if random.random() < 0.55:
                    noise = float(vf + random.choice([-17.0, -0.25, 0.25, 17.0]))
                    targets.append(noise)
                target_set = set(targets)
                fc = Filter.by_property(name).contains_none(targets)
                mask = series.apply(lambda x, vals=target_set: float(x) not in vals if not is_null_like(x) else True)
                es = f"{name} contains_none {targets}"
            elif op == "contains_all":
                fc = Filter.by_property(name).contains_all([vf])
                mask = series.apply(lambda x, target=vf: float(x) == target if not is_null_like(x) else False)
                es = f"{name} contains_all [{vf}]"
            else:
                fc = getattr(Filter.by_property(name), op_map[op])(vf)
                mask = series.apply(safe_cmp(op, vf))
                es = f"{name} {op} {vf}"
            if op == "!=":
                mask = mask | null_mask  # Weaviate not_equal includes null rows

        elif ftype == FieldType.TEXT:
            if random.random() < 0.12:
                return boundary_like_expr(name, series)
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
            target_ts = to_utc_timestamp(val)
            target_dt = target_ts.to_pydatetime() if target_ts is not None else val
            op = random.choices(
                [">", "<", ">=", "<=", "==", "!=", "contains_any", "contains_none", "contains_all"],
                weights=[0.11, 0.11, 0.11, 0.11, 0.12, 0.10, 0.12, 0.12, 0.10],
                k=1,
            )[0]
            op_map = {"==": "equal", "!=": "not_equal", ">": "greater_than", "<": "less_than", ">=": "greater_or_equal", "<=": "less_or_equal"}
            if op == "contains_any":
                targets = [target_dt]
                if target_ts is not None and random.random() < 0.55:
                    noise_ts = target_ts + random.choice([pd.Timedelta(days=7), pd.Timedelta(seconds=1), -pd.Timedelta(days=7), -pd.Timedelta(seconds=1)])
                    targets.append(noise_ts.to_pydatetime())
                target_values = {canonicalize_date_string(target) or str(target) for target in targets}
                fc = Filter.by_property(name).contains_any(targets)
                mask = series.apply(
                    lambda x, vals=target_values: (canonicalize_date_string(x) or str(x)) in vals
                    if to_utc_timestamp(x) is not None
                    else False
                )
                es = f"{name} contains_any {sorted(target_values)}"
            elif op == "contains_none":
                targets = [target_dt]
                if target_ts is not None and random.random() < 0.55:
                    noise_ts = target_ts + random.choice([pd.Timedelta(days=7), pd.Timedelta(seconds=1), -pd.Timedelta(days=7), -pd.Timedelta(seconds=1)])
                    targets.append(noise_ts.to_pydatetime())
                target_values = {canonicalize_date_string(target) or str(target) for target in targets}
                fc = Filter.by_property(name).contains_none(targets)
                mask = series.apply(
                    lambda x, vals=target_values: (canonicalize_date_string(x) or str(x)) not in vals
                    if to_utc_timestamp(x) is not None
                    else True
                )
                es = f"{name} contains_none {sorted(target_values)}"
            elif op == "contains_all":
                target_value = canonicalize_date_string(target_dt) or str(target_dt)
                fc = Filter.by_property(name).contains_all([target_dt])
                mask = series.apply(
                    lambda x, target=target_value: (canonicalize_date_string(x) or str(x)) == target
                    if to_utc_timestamp(x) is not None
                    else False
                )
                es = f"{name} contains_all [{target_value}]"
            else:
                fc = getattr(Filter.by_property(name), op_map[op])(val)
                mask = self._date_cmp_mask(series, op, val)
                es = f"{name} {op} {val}"
            if op == "!=":
                mask = mask | null_mask

        elif ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.DATE_ARRAY]:
            all_items = []
            for arr in self.df[name].dropna():
                if isinstance(arr, list): all_items.extend(arr)
            # Tokenization.FIELD: 不需要停用词过滤, 精确匹配
            if all_items:
                t = self._convert_to_native(random.choice(all_items))
                fc = Filter.by_property(name).contains_any([t])
                item_type = rest_array_item_type(ftype)
                mask = array_membership_mask(series, item_type, [t], "contains_any")
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
        normalized = self._normalize_mask(mask)
        return self._normalize_mask(
            normalized.apply(lambda x: True if pd.isna(x) else not bool(x))
        )

    def _complex_expr_atom_budget(self, depth):
        depth = max(1, int(depth))
        if self._inverted_profile:
            low = max(6, depth + 4)
            high = max(low, min(28, depth * 2 + 4))
        else:
            low = max(4, depth + 2)
            high = max(low, min(20, depth * 2 + 2))
        return random.randint(low, high)

    def _complex_expr_leaf(self):
        for _ in range(10):
            r = random.random()
            if r < 0.08:
                res = self.gen_like_compound_expr()
                if res[0] is not None:
                    return (*self._finalize_expr(*res), {"depth": 0, "atoms": 1})
            elif r < 0.22:
                res = self.gen_scalar_core_compound_expr()
                if res[0] is not None:
                    return (*self._finalize_expr(*res), {"depth": 0, "atoms": 1})
            elif r < 0.32:
                res = self.gen_scalar_contains_compound_expr()
                if res[0] is not None:
                    return (*self._finalize_expr(*res), {"depth": 0, "atoms": 1})
            elif r < 0.36:
                res = self.gen_array_contains_compound_expr()
                if res[0] is not None:
                    return (*self._finalize_expr(*res), {"depth": 0, "atoms": 1})
            elif r < 0.40:
                res = self.gen_constant_expr()
                if res[0]:
                    return (*self._finalize_expr(*res), {"depth": 0, "atoms": 1})
            elif r < 0.56:
                res = self.gen_boundary_expr()
                if res[0] is not None:
                    return (*self._finalize_expr(*res), {"depth": 0, "atoms": 1})
            elif r < 0.60 and self.array_schema:
                res = self.gen_multi_array_expr()
                if res[0] is not None:
                    return (*self._finalize_expr(*res), {"depth": 0, "atoms": 1})
            elif r < 0.74:
                res = self.gen_not_expr()
                if res[0] is not None:
                    return (*self._finalize_expr(*res), {"depth": 1, "atoms": 1})
            elif r < 0.78:
                res = self.gen_nested_object_expr()
                if res[0] is not None:
                    return (*self._finalize_expr(*res), {"depth": 0, "atoms": 1})
            fc, mask, expr = self.gen_atomic_expr()
            if fc is not None:
                return (*self._finalize_expr(fc, mask, expr), {"depth": 0, "atoms": 1})

        res = self.gen_constant_expr()
        if res[0]:
            return (*self._finalize_expr(*res), {"depth": 0, "atoms": 1})
        return None, None, "EMPTY", {"depth": 0, "atoms": 0}

    def _split_complex_expr_budget(self, depth, atom_budget):
        atom_budget = max(2, int(atom_budget))
        if depth > 4:
            side_cap = min((4 if self._inverted_profile else 3), atom_budget - 1)
            side_atoms = random.randint(1, max(1, side_cap))
            deep_atoms = max(1, atom_budget - side_atoms)
            if random.random() < 0.5:
                return deep_atoms, side_atoms
            return side_atoms, deep_atoms

        left_atoms = max(1, atom_budget // 2)
        right_atoms = max(1, atom_budget - left_atoms)
        if atom_budget >= 5 and random.random() < 0.45:
            shift_cap = min(atom_budget // 3, max(0, left_atoms - 1))
            if shift_cap > 0:
                shift = random.randint(0, shift_cap)
                left_atoms -= shift
                right_atoms += shift
        if random.random() < 0.5:
            return left_atoms, right_atoms
        return right_atoms, left_atoms

    def _split_complex_expr_depths(self, depth):
        depth = max(1, int(depth))
        if depth <= 1:
            return 0, 0
        if depth > 4:
            side_depth = random.randint(0, min(2, depth - 2))
            deep_depth = depth - 1
            if random.random() < 0.5:
                return deep_depth, side_depth
            return side_depth, deep_depth

        left_depth = depth - 1
        right_depth = max(0, depth - 1 - random.choice([0, 0, 1]))
        if random.random() < 0.5:
            return left_depth, right_depth
        return right_depth, left_depth

    def gen_complex_expr(self, depth, *, atom_budget=None, return_meta=False):
        target_depth = max(0, int(depth))
        if atom_budget is None:
            atom_budget = self._complex_expr_atom_budget(target_depth)
        atom_budget = max(1, int(atom_budget))

        def rec(remaining_depth, remaining_atoms):
            remaining_depth = max(0, int(remaining_depth))
            remaining_atoms = max(1, int(remaining_atoms))

            if remaining_depth >= 8:
                leaf_cutoff = 0.02 if self._inverted_profile else 0.04
            elif remaining_depth >= 5:
                leaf_cutoff = 0.05 if self._inverted_profile else 0.08
            else:
                leaf_cutoff = 0.14 if self._inverted_profile else 0.20
            if remaining_depth <= 2:
                leaf_cutoff += 0.10
            if remaining_atoms <= 3:
                leaf_cutoff += 0.16

            if remaining_depth <= 0 or remaining_atoms <= 1 or random.random() < leaf_cutoff:
                return self._complex_expr_leaf()

            if remaining_depth > 4 or remaining_atoms > 8:
                branch_weights = [0.34, 0.24, 0.42] if self._inverted_profile else [0.38, 0.28, 0.34]
            else:
                branch_weights = [0.44, 0.36, 0.20] if self._inverted_profile else [0.42, 0.38, 0.20]
            op = random.choices(["and", "or", "not"], weights=branch_weights, k=1)[0]

            if op == "not":
                if remaining_depth <= 3 and random.random() < 0.25:
                    not_res = self.gen_not_expr()
                    if not_res[0] is not None:
                        fc, mask, expr = self._finalize_expr(*not_res)
                        return fc, mask, expr, {"depth": 1, "atoms": 1}

                inner_fc, inner_mask, inner_expr, inner_meta = rec(remaining_depth - 1, max(1, remaining_atoms - 1))
                if not inner_fc:
                    return self._complex_expr_leaf()
                fc_not = Filter.not_(inner_fc)
                mask_not = self._apply_not_mask(inner_mask)
                depth_meta = 1 + int((inner_meta or {}).get("depth", 0))
                atom_meta = int((inner_meta or {}).get("atoms", 1))
                fc_not, mask_not, expr_not = self._finalize_expr(fc_not, mask_not, f"NOT({inner_expr})")
                return fc_not, mask_not, expr_not, {"depth": depth_meta, "atoms": atom_meta}

            left_depth, right_depth = self._split_complex_expr_depths(remaining_depth)
            left_atoms, right_atoms = self._split_complex_expr_budget(remaining_depth, remaining_atoms)
            fl, ml, el, left_meta = rec(left_depth, left_atoms)
            fr, mr, er, right_meta = rec(right_depth, right_atoms)
            if not fl:
                return fr, mr, er, right_meta
            if not fr:
                return fl, ml, el, left_meta

            ml = self._normalize_mask(ml)
            mr = self._normalize_mask(mr)
            depth_meta = 1 + max(
                int((left_meta or {}).get("depth", 0)),
                int((right_meta or {}).get("depth", 0)),
            )
            atom_meta = int((left_meta or {}).get("atoms", 1)) + int((right_meta or {}).get("atoms", 1))
            if op == "and":
                fc, mask, expr = self._finalize_expr(fl & fr, ml & mr, f"({el} AND {er})")
            else:
                fc, mask, expr = self._finalize_expr(fl | fr, ml | mr, f"({el} OR {er})")
            return fc, mask, expr, {"depth": depth_meta, "atoms": atom_meta}

        fc, mask, expr, meta = rec(target_depth, atom_budget)
        meta = dict(meta or {})
        meta.setdefault("depth", 0)
        meta.setdefault("atoms", 0 if fc is None else 1)
        meta["target_depth"] = target_depth
        meta["target_atoms"] = atom_budget

        if return_meta:
            return fc, mask, expr, meta
        return fc, mask, expr


def rest_array_item_type(ftype):
    mapping = {
        FieldType.INT_ARRAY: FieldType.INT,
        FieldType.NUMBER_ARRAY: FieldType.NUMBER,
        FieldType.BOOL_ARRAY: FieldType.BOOL,
        FieldType.TEXT_ARRAY: FieldType.TEXT,
        FieldType.DATE_ARRAY: FieldType.DATE,
    }
    return mapping.get(ftype)


def rest_like_mask(series, pattern):
    escaped = re.escape(pattern)
    regex = "^" + escaped.replace(r"\*", ".*").replace(r"\?", ".") + "$"
    matcher = re.compile(regex)
    return series.apply(
        lambda value, rx=matcher: bool(rx.match(str(value)))
        if value is not None and not (isinstance(value, float) and np.isnan(value))
        else False
    )


def rest_text_length_mask(series, op, target):
    lengths = series.apply(
        lambda value: 0
        if value is None or (isinstance(value, float) and np.isnan(value))
        else len(str(value))
    )
    if op == "Equal":
        return lengths == int(target)
    if op == "GreaterThan":
        return lengths > int(target)
    if op == "GreaterThanEqual":
        return lengths >= int(target)
    if op == "LessThan":
        return lengths < int(target)
    if op == "LessThanEqual":
        return lengths <= int(target)
    return pd.Series(False, index=series.index)


def generate_simple_rest_filter(qg):
    df = qg.df
    supported_scalar_schema = [
        field for field in qg.schema
        if field["type"] in [FieldType.INT, FieldType.NUMBER, FieldType.BOOL, FieldType.TEXT, FieldType.DATE]
    ]
    search_only_text_schema = list(getattr(qg, "search_only_text_schema", []) or [])
    supported_array_schema = [
        field for field in qg.schema
        if field["type"] in [FieldType.INT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.TEXT_ARRAY, FieldType.DATE_ARRAY]
    ]
    length_schema = [field for field in qg.property_length_schema if field["type"] == FieldType.TEXT]

    choices = []
    if qg._existing_ids():
        choices.extend(["id"] * 2)
    if length_schema:
        choices.extend(["length"] * 2)
    if supported_scalar_schema:
        choices.extend(["scalar"] * 9)
        choices.extend(["null"] * 2)
    if search_only_text_schema:
        choices.extend(["search_only_exact"] * 3)
    if supported_array_schema:
        choices.extend(["array"] * 4)
        choices.extend(["null"] * 2)
    if not choices:
        return None, None, None

    choice = random.choice(choices)

    if choice == "id":
        existing_ids = qg._existing_ids()
        if not existing_ids:
            return None, None, None
        if random.random() < 0.75:
            target_id = str(random.choice(existing_ids))
        else:
            target_id = qg._random_absent_uuid(existing_ids)
        mask = df["id"].apply(lambda value, target=target_id: str(value) == target)
        return build_rest_where(["id"], "Equal", ftype=FieldType.TEXT, value=target_id), mask, f'id == "{target_id}"'

    if choice == "length":
        field = random.choice(length_schema)
        name = field["name"]
        series = df[name]
        non_null_lengths = [
            len(str(value))
            for value in series.tolist()
            if value is not None and not (isinstance(value, float) and np.isnan(value))
        ]
        if non_null_lengths and random.random() < 0.75:
            target = int(random.choice(non_null_lengths + [0]))
        else:
            target = int(random.choice([0, 1, 2, 4, 8, 16, 32]))
        operator = random.choice(["Equal", "GreaterThan", "LessThanEqual"])
        mask = rest_text_length_mask(series, operator, target)
        return build_rest_where([f"len({name})"], operator, raw_key="valueInt", value=int(target)), mask, f"len({name}) {operator} {target}"

    if choice == "null":
        field = random.choice(supported_scalar_schema + supported_array_schema)
        name, ftype = field["name"], field["type"]
        series = df[name]
        null_mask = qg._effective_null_mask(series, ftype)
        is_null = random.random() < 0.5
        mask = null_mask if is_null else ~null_mask
        return build_rest_where([name], "IsNull", raw_key="valueBoolean", value=is_null), mask, f"{name} is {'null' if is_null else 'not null'}"

    if choice == "search_only_exact":
        field = random.choice(search_only_text_schema)
        name = field["name"]
        series = df[name]
        target = qg.get_value_for_query(name, FieldType.TEXT)
        if target is None:
            return None, None, None
        target = str(target)
        operator = random.choice(["Equal", "NotEqual"])
        if operator == "Equal":
            mask = series.apply(
                lambda value, tv=target: str(value) == tv
                if value is not None and not (isinstance(value, float) and np.isnan(value))
                else False
            )
        else:
            mask = series.apply(
                lambda value, tv=target: str(value) != tv
                if value is not None and not (isinstance(value, float) and np.isnan(value))
                else True
            )
        return build_rest_where([name], operator, ftype=FieldType.TEXT, value=target), mask, f"{name} {operator} {target} [search-only]"

    if choice == "array":
        field = random.choice(supported_array_schema)
        name, ftype = field["name"], field["type"]
        item_type = rest_array_item_type(ftype)
        series = df[name]
        targets = _choose_array_targets(series, count=random.choice([1, 1, 2]))
        if not targets:
            null_mask = qg._effective_null_mask(series, ftype)
            return build_rest_where([name], "IsNull", raw_key="valueBoolean", value=True), null_mask, f"{name} is null"
        if item_type == FieldType.DATE:
            targets = [canonicalize_date_string(target) or str(target) for target in targets]
        mode = random.choice(["ContainsAny", "ContainsAll", "ContainsNone"])
        if mode == "ContainsAny":
            mask = array_membership_mask(series, item_type, targets, "contains_any")
        elif mode == "ContainsAll":
            mask = array_membership_mask(series, item_type, targets, "contains_all")
        else:
            mask = array_membership_mask(series, item_type, targets, "contains_none")
        return build_rest_where([name], mode, ftype=item_type, value=targets), mask, f"{name} {mode.lower()} {targets}"

    field = random.choice(supported_scalar_schema)
    name, ftype = field["name"], field["type"]
    series = df[name]
    null_mask = qg._effective_null_mask(series, ftype)

    def scalar_mask(op_name, target_value):
        def compare(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return op_name == "NotEqual"
            try:
                if op_name == "Equal":
                    return value == target_value
                if op_name == "NotEqual":
                    return value != target_value
                if op_name == "GreaterThan":
                    return value > target_value
                if op_name == "GreaterThanEqual":
                    return value >= target_value
                if op_name == "LessThan":
                    return value < target_value
                if op_name == "LessThanEqual":
                    return value <= target_value
            except Exception:
                return False
            return False
        return series.apply(compare)

    if ftype == FieldType.INT:
        target = qg.get_value_for_query(name, FieldType.INT)
        if target is None:
            return None, None, None
        target = int(target)
        if random.random() < 0.30:
            mode = random.choice(["ContainsAny", "ContainsNone", "ContainsAll"])
            targets = [target]
            if mode != "ContainsAll" and random.random() < 0.50:
                targets.append(clamp_weaviate_int(target + random.choice([7, 13, 101, -7, -13, -101])))
            target_set = set(targets)
            if mode == "ContainsAny":
                mask = series.apply(lambda value, vals=target_set: int(value) in vals if not is_null_like(value) else False)
            elif mode == "ContainsNone":
                mask = series.apply(lambda value, vals=target_set: int(value) not in vals if not is_null_like(value) else True)
            else:
                mask = series.apply(lambda value, tv=target: int(value) == tv if not is_null_like(value) else False)
                targets = [target]
            return build_rest_where([name], mode, ftype=FieldType.INT, value=targets), mask, f"{name} {mode} {targets}"
        if random.random() < 0.30:
            other = qg.get_value_for_query(name, FieldType.INT)
            if other is not None:
                lo, hi = sorted([target, int(other)])
                mask = series.apply(lambda value, left=lo, right=hi: left <= int(value) <= right if not is_null_like(value) else False)
                where_filter = build_rest_compound(
                    "And",
                    [
                        build_rest_where([name], "GreaterThanEqual", ftype=FieldType.INT, value=lo),
                        build_rest_where([name], "LessThanEqual", ftype=FieldType.INT, value=hi),
                    ],
                )
                return where_filter, mask, f"{name} in [{lo},{hi}]"
        operator = random.choice(["Equal", "NotEqual", "GreaterThan", "GreaterThanEqual", "LessThan", "LessThanEqual"])
        mask = scalar_mask(operator, target)
        if operator == "NotEqual":
            mask = mask | null_mask
        return build_rest_where([name], operator, ftype=FieldType.INT, value=target), mask, f"{name} {operator} {target}"

    if ftype == FieldType.NUMBER:
        target = qg.get_value_for_query(name, FieldType.NUMBER)
        if target is None:
            return None, None, None
        target = float(target)
        if random.random() < 0.30:
            mode = random.choice(["ContainsAny", "ContainsNone", "ContainsAll"])
            targets = [target]
            if mode != "ContainsAll" and random.random() < 0.50:
                targets.append(float(target + random.choice([-17.0, -0.25, 0.25, 17.0])))
            target_set = set(targets)
            if mode == "ContainsAny":
                mask = series.apply(lambda value, vals=target_set: float(value) in vals if not is_null_like(value) else False)
            elif mode == "ContainsNone":
                mask = series.apply(lambda value, vals=target_set: float(value) not in vals if not is_null_like(value) else True)
            else:
                mask = series.apply(lambda value, tv=target: float(value) == tv if not is_null_like(value) else False)
                targets = [target]
            return build_rest_where([name], mode, ftype=FieldType.NUMBER, value=targets), mask, f"{name} {mode} {targets}"
        if random.random() < 0.45:
            lo, hi = float_window(target)
            if lo is not None and hi is not None:
                mask = series.apply(lambda value, left=lo, right=hi: left <= float(value) <= right if not is_null_like(value) else False)
                where_filter = build_rest_compound(
                    "And",
                    [
                        build_rest_where([name], "GreaterThanEqual", ftype=FieldType.NUMBER, value=lo),
                        build_rest_where([name], "LessThanEqual", ftype=FieldType.NUMBER, value=hi),
                    ],
                )
                return where_filter, mask, f"{name} ≈ {target}"
        operator = random.choice(["GreaterThan", "GreaterThanEqual", "LessThan", "LessThanEqual"])
        mask = scalar_mask(operator, target)
        return build_rest_where([name], operator, ftype=FieldType.NUMBER, value=target), mask, f"{name} {operator} {target}"

    if ftype == FieldType.BOOL:
        target = bool(qg.get_value_for_query(name, FieldType.BOOL))
        operator = random.choice(["Equal", "NotEqual", "ContainsAny", "ContainsNone", "ContainsAll"])
        if operator == "ContainsAny":
            mask = series.apply(lambda value, tv=target: bool(value) == tv if not is_null_like(value) else False)
            return build_rest_where([name], operator, ftype=FieldType.BOOL, value=[target]), mask, f"{name} {operator} [{target}]"
        if operator == "ContainsNone":
            mask = series.apply(lambda value, tv=target: bool(value) != tv if not is_null_like(value) else True)
            return build_rest_where([name], operator, ftype=FieldType.BOOL, value=[target]), mask, f"{name} {operator} [{target}]"
        if operator == "ContainsAll":
            mask = series.apply(lambda value, tv=target: bool(value) == tv if not is_null_like(value) else False)
            return build_rest_where([name], operator, ftype=FieldType.BOOL, value=[target]), mask, f"{name} {operator} [{target}]"
        mask = scalar_mask(operator, target)
        if operator == "NotEqual":
            mask = mask | null_mask
        return build_rest_where([name], operator, ftype=FieldType.BOOL, value=target), mask, f"{name} {operator} {target}"

    if ftype == FieldType.TEXT:
        target = qg.get_value_for_query(name, FieldType.TEXT)
        if target is None:
            return None, None, None
        target = str(target)
        operator = random.choice(
            ["Equal", "NotEqual", "LikePrefix", "LikeSuffix", "LikeContains", "LikeSingle", "LikeStar", "LikeDoubleStar", "LikeThreeQ", "LikePercent", "LikeUnderscore", "ContainsAny", "ContainsNone"]
        )
        if operator == "Equal":
            mask = series.apply(lambda value, tv=target: str(value) == tv if value is not None and not (isinstance(value, float) and np.isnan(value)) else False)
            return build_rest_where([name], "Equal", ftype=FieldType.TEXT, value=target), mask, f'{name} == "{target}"'
        if operator == "NotEqual":
            mask = series.apply(lambda value, tv=target: str(value) != tv if value is not None and not (isinstance(value, float) and np.isnan(value)) else True)
            return build_rest_where([name], "NotEqual", ftype=FieldType.TEXT, value=target), mask, f'{name} != "{target}"'
        if operator == "ContainsAny":
            extra = qg._random_string(6, 12) if random.random() < 0.30 else None
            targets = [target] + ([extra] if extra else [])
            mask = series.apply(
                lambda value, vals=set(targets): str(value) in vals
                if value is not None and not (isinstance(value, float) and np.isnan(value))
                else False
            )
            return build_rest_where([name], "ContainsAny", ftype=FieldType.TEXT, value=targets), mask, f"{name} contains_any {targets}"
        if operator == "ContainsNone":
            extra = qg._random_string(6, 12) if random.random() < 0.50 else None
            targets = [target] + ([extra] if extra else [])
            mask = series.apply(
                lambda value, vals=set(targets): str(value) not in vals
                if value is not None and not (isinstance(value, float) and np.isnan(value))
                else True
            )
            return build_rest_where([name], "ContainsNone", ftype=FieldType.TEXT, value=targets), mask, f"{name} contains_none {targets}"

        if operator == "LikeStar":
            pattern = "*"
        elif operator == "LikeDoubleStar":
            pattern = "**"
        elif operator == "LikeThreeQ":
            pattern = "???"
        elif operator == "LikePercent":
            pattern = "%"
        elif operator == "LikeUnderscore":
            pattern = "_"
        elif operator == "LikePrefix":
            prefix_len = random.randint(1, min(4, max(1, len(target))))
            pattern = f"{target[:prefix_len]}*"
        elif operator == "LikeSuffix":
            suffix_len = random.randint(1, min(4, max(1, len(target))))
            pattern = f"*{target[-suffix_len:]}"
        elif operator == "LikeContains":
            start = random.randint(0, max(0, len(target) - 2))
            end = random.randint(start + 1, min(start + 5, len(target))) if len(target) >= 2 else 1
            pattern = f"*{target[start:end]}*"
        else:
            if len(target) < 2:
                pattern = f"{target}*"
            else:
                pos = random.randint(0, len(target) - 1)
                pattern = target[:pos] + "?" + target[pos + 1:]
        mask = rest_like_mask(series, pattern)
        return build_rest_where([name], "Like", ftype=FieldType.TEXT, value=pattern), mask, f'{name} like "{pattern}"'

    if ftype == FieldType.DATE:
        target = qg.get_value_for_query(name, FieldType.DATE)
        if target is None:
            return None, None, None
        target = canonicalize_date_string(target) or str(target)
        if random.random() < 0.30:
            mode = random.choice(["ContainsAny", "ContainsNone", "ContainsAll"])
            targets = [target]
            anchor = to_utc_timestamp(target)
            if mode != "ContainsAll" and anchor is not None and random.random() < 0.50:
                noise = canonicalize_date_string(anchor + random.choice([pd.Timedelta(days=7), pd.Timedelta(seconds=1), -pd.Timedelta(days=7), -pd.Timedelta(seconds=1)]))
                if noise:
                    targets.append(noise)
            target_set = set(targets)
            if mode == "ContainsAny":
                mask = series.apply(
                    lambda value, vals=target_set: (canonicalize_date_string(value) or str(value)) in vals
                    if to_utc_timestamp(value) is not None
                    else False
                )
            elif mode == "ContainsNone":
                mask = series.apply(
                    lambda value, vals=target_set: (canonicalize_date_string(value) or str(value)) not in vals
                    if to_utc_timestamp(value) is not None
                    else True
                )
            else:
                mask = series.apply(
                    lambda value, tv=target: (canonicalize_date_string(value) or str(value)) == tv
                    if to_utc_timestamp(value) is not None
                    else False
                )
                targets = [target]
            return build_rest_where([name], mode, ftype=FieldType.DATE, value=targets), mask, f"{name} {mode} {targets}"
        if random.random() < 0.30:
            anchor = to_utc_timestamp(target)
            if anchor is not None:
                delta = random.choice([pd.Timedelta(microseconds=1), pd.Timedelta(seconds=1)])
                lo = canonicalize_date_string(anchor - delta)
                hi = canonicalize_date_string(anchor + delta)
                if lo and hi:
                    mask = series.apply(
                        lambda value, left=to_utc_timestamp(lo), right=to_utc_timestamp(hi): left <= to_utc_timestamp(value) <= right
                        if to_utc_timestamp(value) is not None
                        else False
                    )
                    where_filter = build_rest_compound(
                        "And",
                        [
                            build_rest_where([name], "GreaterThanEqual", ftype=FieldType.DATE, value=lo),
                            build_rest_where([name], "LessThanEqual", ftype=FieldType.DATE, value=hi),
                        ],
                    )
                    return where_filter, mask, f"{name} in [{lo},{hi}]"
        operator = random.choice(["Equal", "NotEqual", "GreaterThan", "GreaterThanEqual", "LessThan", "LessThanEqual"])
        mask = qg._date_cmp_mask(series, {"Equal": "==", "NotEqual": "!=", "GreaterThan": ">", "GreaterThanEqual": ">=", "LessThan": "<", "LessThanEqual": "<="}[operator], target)
        if operator == "NotEqual":
            mask = mask | null_mask
        return build_rest_where([name], operator, ftype=FieldType.DATE, value=target), mask, f"{name} {operator} {target}"

    return None, None, None


def generate_rest_filter(qg):
    base_index = qg.df.index

    def leaf():
        if random.random() < 0.12:
            where_filter, mask, expr = qg.gen_like_compound_rest_expr()
            if where_filter is not None:
                return where_filter, _normalize_bool_mask(mask, base_index), expr, {"depth": 1, "atoms": 2}
        if random.random() < 0.18:
            where_filter, mask, expr = qg.gen_scalar_contains_compound_rest_expr()
            if where_filter is not None:
                return where_filter, _normalize_bool_mask(mask, base_index), expr, {"depth": 1, "atoms": 2}
        where_filter, mask, expr = generate_simple_rest_filter(qg)
        if where_filter is None:
            return None, None, None, None
        return where_filter, _normalize_bool_mask(mask, base_index), expr, {"depth": 0, "atoms": 1}

    def rec(depth):
        if depth <= 0 or random.random() < 0.22:
            return leaf()

        op = random.choices(["and", "or", "not"], weights=[0.42, 0.38, 0.20], k=1)[0]
        if op == "not":
            inner_filter, inner_mask, inner_expr, inner_meta = rec(depth - 1)
            if inner_filter is None:
                return leaf()
            return (
                build_rest_compound("Not", [inner_filter]),
                qg._apply_not_mask(inner_mask),
                f"NOT({inner_expr})",
                {"depth": 1 + int(inner_meta["depth"]), "atoms": int(inner_meta["atoms"])},
            )

        left_filter, left_mask, left_expr, left_meta = rec(depth - 1)
        right_filter, right_mask, right_expr, right_meta = rec(depth - 1)
        if left_filter is None:
            return right_filter, right_mask, right_expr, right_meta
        if right_filter is None:
            return left_filter, left_mask, left_expr, left_meta

        if op == "and":
            return (
                build_rest_compound("And", [left_filter, right_filter]),
                _normalize_bool_mask(left_mask, base_index) & _normalize_bool_mask(right_mask, base_index),
                f"({left_expr} AND {right_expr})",
                {"depth": 1 + max(int(left_meta["depth"]), int(right_meta["depth"])), "atoms": int(left_meta["atoms"]) + int(right_meta["atoms"])},
            )
        return (
            build_rest_compound("Or", [left_filter, right_filter]),
            _normalize_bool_mask(left_mask, base_index) | _normalize_bool_mask(right_mask, base_index),
            f"({left_expr} OR {right_expr})",
            {"depth": 1 + max(int(left_meta["depth"]), int(right_meta["depth"])), "atoms": int(left_meta["atoms"]) + int(right_meta["atoms"])},
        )

    depth = random.randint(REST_FILTER_MIN_DEPTH, REST_FILTER_MAX_DEPTH)
    where_filter, mask, expr, meta = rec(depth)
    if where_filter is None:
        return None, pd.Series(False, index=base_index, dtype=bool), "EMPTY", {"depth": 0, "atoms": 0, "target_depth": depth}
    meta["target_depth"] = depth
    evaluated_mask = evaluate_rest_where_mask(qg, where_filter)
    return where_filter, _normalize_bool_mask(evaluated_mask, base_index), expr, meta


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
        gen_pqs_expr:
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
        gen_complex_noise: 包含多层 AND/OR/NOT 嵌套
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

        if random.random() < 0.16:
            compound_filter, compound_expr = self.gen_true_like_compound(row)
            if compound_filter is not None:
                return compound_filter, compound_expr

        if random.random() < 0.18:
            compound_filter, compound_expr = self.gen_true_scalar_contains_compound(row)
            if compound_filter is not None:
                return compound_filter, compound_expr

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
                strat = random.choice(["eq", "neq", "contains_any_self", "contains_none_noise", "contains_all_self"])
                if strat == "eq":
                    return Filter.by_property(fname).equal(bool(val)), f"{fname} == {bool(val)}"
                if strat == "contains_any_self":
                    return Filter.by_property(fname).contains_any([bool(val)]), f"{fname} contains_any [{bool(val)}]"
                if strat == "contains_none_noise":
                    return Filter.by_property(fname).contains_none([not bool(val)]), f"{fname} contains_none [{not bool(val)}]"
                if strat == "contains_all_self":
                    return Filter.by_property(fname).contains_all([bool(val)]), f"{fname} contains_all [{bool(val)}]"
                else:
                    return Filter.by_property(fname).not_equal(not bool(val)), f"{fname} != {not bool(val)}"
            elif ftype == FieldType.INT:
                vi = int(val)
                strat = random.choice(["eq", "ge_self", "le_self", "lt_above", "range_tight", "range_pm1", "not_lt", "neq_fake", "contains_any_self", "contains_none_noise", "contains_all_self"])
                if strat == "eq":
                    return Filter.by_property(fname).equal(vi), f"{fname} == {vi}"
                elif strat == "ge_self":
                    return Filter.by_property(fname).greater_or_equal(vi), f"{fname} >= {vi}"
                elif strat == "le_self":
                    return Filter.by_property(fname).less_or_equal(vi), f"{fname} <= {vi}"
                elif strat == "contains_any_self":
                    return Filter.by_property(fname).contains_any([vi]), f"{fname} contains_any [{vi}]"
                elif strat == "contains_none_noise":
                    fake = clamp_weaviate_int(vi + random.choice([7, 13, 101, -7, -13, -101]))
                    if fake == vi:
                        fake = clamp_weaviate_int(fake + 1)
                    return Filter.by_property(fname).contains_none([fake]), f"{fname} contains_none [{fake}]"
                elif strat == "contains_all_self":
                    return Filter.by_property(fname).contains_all([vi]), f"{fname} contains_all [{vi}]"
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
                strat = random.choice(["ge_self", "le_self", "lt_above", "gt_below", "range", "not_gt", "contains_any_self", "contains_none_noise", "contains_all_self"])
                lo, hi = float_window(vf)
                if lo is None:
                    continue
                if strat == "ge_self":
                    return Filter.by_property(fname).greater_or_equal(vf), f"{fname}>={vf}"
                elif strat == "le_self":
                    return Filter.by_property(fname).less_or_equal(vf), f"{fname}<={vf}"
                elif strat == "contains_any_self":
                    return Filter.by_property(fname).contains_any([vf]), f"{fname} contains_any [{vf}]"
                elif strat == "contains_none_noise":
                    fake = float(vf + random.choice([-17.0, -0.25, 0.25, 17.0]))
                    if fake == vf:
                        fake += 1.0
                    return Filter.by_property(fname).contains_none([fake]), f"{fname} contains_none [{fake}]"
                elif strat == "contains_all_self":
                    return Filter.by_property(fname).contains_all([vf]), f"{fname} contains_all [{vf}]"
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
                strat = random.choice(["eq", "ge_self", "le_self", "lt_above", "gt_below", "contains_any_self", "contains_none_noise", "contains_all_self"])
                if strat == "ge_self":
                    return Filter.by_property(fname).greater_or_equal(str(val)), f'{fname} >= "{val}"'
                if strat == "le_self":
                    return Filter.by_property(fname).less_or_equal(str(val)), f'{fname} <= "{val}"'
                if strat == "contains_any_self":
                    ts = to_utc_timestamp(val)
                    if ts is not None:
                        return Filter.by_property(fname).contains_any([ts.to_pydatetime()]), f'{fname} contains_any ["{canonicalize_date_string(val) or val}"]'
                if strat == "contains_none_noise":
                    ts = to_utc_timestamp(val)
                    if ts is not None:
                        fake = ts + pd.Timedelta(days=7)
                        return Filter.by_property(fname).contains_none([fake.to_pydatetime()]), f'{fname} contains_none ["{canonicalize_date_string(fake)}"]'
                if strat == "contains_all_self":
                    ts = to_utc_timestamp(val)
                    if ts is not None:
                        return Filter.by_property(fname).contains_all([ts.to_pydatetime()]), f'{fname} contains_all ["{canonicalize_date_string(val) or val}"]'
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
            elif ftype in [FieldType.INT_ARRAY, FieldType.TEXT_ARRAY, FieldType.NUMBER_ARRAY, FieldType.BOOL_ARRAY, FieldType.DATE_ARRAY]:
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


GRAPHQL_PROBE_CLASS_PREFIX = "GraphQLScalarProbe"
GRAPHQL_PROBE_PROPERTIES = [
    Property(name="tag", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
    Property(name="bucket", data_type=DataType.INT, index_filterable=True, index_range_filters=True),
    Property(name="flag", data_type=DataType.BOOL, index_filterable=True),
    Property(name="score", data_type=DataType.NUMBER, index_filterable=True, index_range_filters=True),
    Property(name="eventTs", data_type=DataType.DATE, index_filterable=True, index_range_filters=True),
    Property(
        name="body",
        data_type=DataType.TEXT,
        tokenization=Tokenization.WORD,
        index_filterable=True,
        index_searchable=True,
    ),
    Property(name="nullableText", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
    Property(name="textArr", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD, index_filterable=True),
]
GRAPHQL_PROBE_FIXTURE = [
    {
        "tag": "alpha",
        "bucket": 0,
        "flag": True,
        "score": 1.0,
        "eventTs": "2024-01-01T00:00:00Z",
        "body": "nebula anchor apple red",
        "nullableText": "red",
        "textArr": ["red", "alpha"],
        "vector": [1.0, 0.0, 0.0, 0.0],
    },
    {
        "tag": "beta",
        "bucket": 1,
        "flag": False,
        "score": 2.5,
        "eventTs": "2024-01-02T00:00:00Z",
        "body": "nebula orange blue",
        "nullableText": None,
        "textArr": ["blue", "beta"],
        "vector": [0.0, 1.0, 0.0, 0.0],
    },
    {
        "tag": "gamma",
        "bucket": 2,
        "flag": True,
        "score": 3.5,
        "eventTs": "2024-01-03T00:00:00Z",
        "body": "anchor grape green",
        "nullableText": "green",
        "textArr": ["green", "gamma"],
        "vector": [0.0, 0.0, 1.0, 0.0],
    },
    {
        "tag": "delta",
        "bucket": 1,
        "flag": True,
        "score": -1.0,
        "eventTs": "2024-01-04T00:00:00Z",
        "body": "delta red marker",
        "nullableText": "red",
        "textArr": ["red", "delta"],
        "vector": [0.0, 0.0, 0.0, 1.0],
    },
    {
        "tag": "epsilon",
        "bucket": 2,
        "flag": False,
        "score": 0.0,
        "eventTs": "2024-01-05T00:00:00Z",
        "body": "epsilon neutral yellow",
        "nullableText": "yellow",
        "textArr": ["yellow", "epsilon"],
        "vector": [0.70710677, 0.70710677, 0.0, 0.0],
    },
    {
        "tag": "zeta",
        "bucket": 3,
        "flag": True,
        "score": 10.0,
        "eventTs": "2024-01-06T00:00:00Z",
        "body": "zeta comet tail",
        "textArr": ["tail", "zeta"],
        "vector": [0.6, 0.0, 0.8, 0.0],
    },
]


def graphql_probe_collection_name(seed):
    return f"{GRAPHQL_PROBE_CLASS_PREFIX}{int(seed) % 1_000_000_000}"


def graphql_probe_uuid(seed, tag):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-graphql-probe-{seed}-{tag}"))


def graphql_probe_rows(seed):
    rows = []
    for item in GRAPHQL_PROBE_FIXTURE:
        props = {key: value for key, value in item.items() if key != "vector"}
        rows.append(
            {
                "id": graphql_probe_uuid(seed, props["tag"]),
                "properties": props,
                "vector": list(item["vector"]),
            }
        )
    return rows


def reset_graphql_probe_collection(client, collection_name, seed):
    try:
        client.collections.delete(collection_name)
    except Exception:
        pass
    client.collections.create(
        name=collection_name,
        properties=GRAPHQL_PROBE_PROPERTIES,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=64,
                max_connections=16,
            )
        ),
        inverted_index_config=Configure.inverted_index(
            index_null_state=True,
            index_property_length=True,
            index_timestamps=True,
        ),
    )
    collection = client.collections.get(collection_name)
    objects = [
        DataObject(uuid=row["id"], properties=row["properties"], vector=row["vector"])
        for row in graphql_probe_rows(seed)
    ]
    result = collection.data.insert_many(objects)
    errors = getattr(result, "errors", None)
    if errors:
        raise RuntimeError(f"GraphQL probe insert errors: {errors}")
    return collection


def graphql_raw_query_or_raise(client, query):
    result = client.graphql_raw_query(query)
    errors = getattr(result, "errors", None)
    if errors:
        raise RuntimeError(f"GraphQL errors: {errors} | query={query}")
    return result


def graphql_get_tags(result, collection_name):
    rows = getattr(result, "get", {}).get(collection_name, []) or []
    return [str(row.get("tag")) for row in rows]


def graphql_aggregate_groups(result, collection_name):
    return getattr(result, "aggregate", {}).get(collection_name, []) or []


def graphql_probe_expected_tags(seed, predicate):
    return sorted(
        row["properties"]["tag"]
        for row in graphql_probe_rows(seed)
        if predicate(row["properties"])
    )


def graphql_probe_expected_group_counts(seed, field_name):
    counts = {}
    for row in graphql_probe_rows(seed):
        value = row["properties"].get(field_name)
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def graphql_probe_case_order(seed, total_cases):
    offset = int(seed) % max(1, total_cases)
    return offset


# --- 6. Main Execution ---

def initialize_seeded_run(seed=None):
    global VECTOR_CHECK_RATIO, VECTOR_TOPK, VECTOR_INDEX_TYPE, DISTANCE_METRIC, BOUNDARY_INJECTION_RATE
    current_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)
    if is_inverted_profile():
        VECTOR_CHECK_RATIO = random.uniform(0.02, 0.10)
        VECTOR_TOPK = random.randint(20, 80)
        BOUNDARY_INJECTION_RATE = random.uniform(0.22, 0.40)
        VECTOR_INDEX_TYPE = random.choice(["hnsw", "flat", "dynamic"])
    else:
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
    print(f"   Profile: {get_fuzz_profile()}")
    print(f"   VecIndex: {VECTOR_INDEX_TYPE}, Dist: {DISTANCE_METRIC}, VecRatio: {VECTOR_CHECK_RATIO:.2f}, TopK: {VECTOR_TOPK}, BoundaryRate: {BOUNDARY_INJECTION_RATE:.2f}")
    print(
        f"   QueryPageSize: {get_query_page_size()} (CapHint={get_query_maximum_results()}, "
        f"Partition={get_filter_partition_size()}, IdBatch={get_filter_id_batch_size()})"
    )
    print(
        f"   OracleREST: rate={ORACLE_REST_CROSSCHECK_RATE:.2f}, "
        f"Depth={REST_FILTER_MIN_DEPTH}..{REST_FILTER_MAX_DEPTH}, "
        f"RestCompareCap={REST_FILTER_MAX_COMPARE_RESULTS}"
    )

    dm = DataManager(current_seed)
    dm.generate_schema()
    dm.generate_data()
    wm = WeaviateManager()
    wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config)
        dm.searchable_text_fields = set(getattr(wm, "searchable_text_fields", set()) or set())
        dm.field_index_state = dict(getattr(wm, "field_index_state", {}) or {})
        active_vector_index_type = getattr(wm, "actual_vector_index_type", VECTOR_INDEX_TYPE)
        active_distance_metric = getattr(wm, "actual_distance_metric", DISTANCE_METRIC)
        print(f"   ActualVecIndex: {active_vector_index_type}, ActualDist: {active_distance_metric}")
        wm.insert(dm)
        wm.sync_creation_times(dm)
        wm.sync_update_times(dm)
        logf = make_log_path(
            stable_log_filename(
                "weaviate_fuzz_test",
                current_seed,
                rounds,
                "oracle",
                f"dynamic-{'on' if enable_dynamic_ops else 'off'}",
                f"consistency-{consistency_label(resolved_consistency, randomize=randomize_consistency)}",
            )
        )
        repro_cmd = format_repro_command(
            current_seed,
            resolved_consistency,
            randomize_consistency=randomize_consistency,
            mode="oracle",
            rounds=rounds,
            rows=N,
            dynamic_enabled=enable_dynamic_ops,
            host=HOST,
            port=PORT,
        )
        print(f"\n📝 Log: {display_path(logf)}")
        print(f"    Reproduce: {repro_cmd}")
        print(" Testing...")

        qg = OracleQueryGenerator(dm)
        col = wm.client.collections.get(CLASS_NAME)
        fails = []
        stats = FuzzStats()
        # Track dynamically modified row IDs to distinguish Weaviate inverted index bugs
        # from fuzzer logic bugs. Weaviate has a known bug where replace() corrupts
        # the inverted index for BOOL/array fields on modified rows.
        dynamic_ids = set()
        rest_crosschecks = 0
        rest_crosscheck_skips = 0

        with open(logf, "w", encoding="utf-8") as f:
            def flog(msg):
                f.write(msg + "\n"); f.flush()

            def sample(ids, lim=5):
                if not ids: return []
                stable_ids = sorted(str(object_id) for object_id in ids)
                return dm.df[dm.df["id"].isin(stable_ids)].sort_values("id").to_dict("records")[:lim]

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
                f"Profile: {get_fuzz_profile()} | "
                f"Consistency: {consistency_label(resolved_consistency, randomize=randomize_consistency)} | "
                f"VecIdx: {active_vector_index_type} | Dist: {active_distance_metric}"
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
                            field_type_map = build_field_type_map(dm.schema_config)
                            objs = []
                            for r, v in zip(rows, vecs):
                                p = build_weaviate_properties(r, field_type_map)
                                objs.append(DataObject(uuid=r["id"], properties=p, vector=v.tolist()))
                            dyn_result = col.data.insert_many(objs)
                            dyn_errors = getattr(dyn_result, "errors", None)
                            if not dyn_errors and isinstance(dyn_result, dict):
                                dyn_errors = dyn_result.get("errors")
                            if dyn_errors:
                                raise RuntimeError(f"dynamic batch insert partially failed: {dyn_errors}")
                            vectors_by_id = {str(row["id"]): vec for row, vec in zip(rows, vecs)}
                            synced_ids, missing_ids, new_ids = wm.sync_rows_from_engine(
                                dm,
                                [row["id"] for row in rows],
                                vectors_by_id=vectors_by_id,
                                retries=4,
                            )
                            dynamic_ids.update(synced_ids)
                            flog(f"[Dyn] Inserted req={bc} visible={len(synced_ids)} new={len(new_ids)}")
                            if missing_ids:
                                flog(f"[Dyn] ❌ Insert visibility miss: {missing_ids}")
                                fails.extend({"id": i, "detail": f"Insert not visible: {mid}"} for mid in missing_ids)
                            if synced_ids:
                                wm.sync_creation_times(dm, ids=synced_ids)
                                wm.sync_update_times(dm, ids=synced_ids)
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
                                deleted_ids, still_present = wm.verify_deleted_ids(dm, dids, retries=4)
                                dynamic_ids.difference_update(deleted_ids)
                                flog(f"[Dyn] Deleted {len(deleted_ids)}/{len(dids)}")
                                if still_present:
                                    for object_id in still_present:
                                        flog(f"[Dyn] ❌ Deleted ID still exists: {object_id}")
                                        fails.append({"id": i, "detail": f"Ghost after delete: {object_id}"})
                                if deleted_ids:
                                    qg = OracleQueryGenerator(dm)
                            except Exception as e:
                                flog(f"[Dyn] Delete fail: {e}")
                    elif op == "update":
                        if not dm.df.empty:
                            try:
                                ui = random.randint(0, len(dm.df) - 1)
                                uid = dm.df.iloc[ui]["id"]
                                nr = dm.generate_single_row(id_override=uid)
                                nv = dm.generate_single_vector()
                                field_type_map = build_field_type_map(dm.schema_config)
                                p = build_weaviate_properties(nr, field_type_map)
                                col.data.replace(uuid=uid, properties=p, vector=nv.tolist())
                                synced_ids, missing_ids, _ = wm.sync_rows_from_engine(
                                    dm,
                                    [uid],
                                    vectors_by_id={str(uid): nv},
                                    retries=4,
                                )
                                if synced_ids:
                                    dynamic_ids.update(synced_ids)
                                    wm.sync_update_times(dm, ids=synced_ids)
                                    qg = OracleQueryGenerator(dm)
                                    flog(f"[Dyn] Updated {uid}")
                                if missing_ids:
                                    flog(f"[Dyn] ❌ Update visibility miss: {missing_ids}")
                                    fails.extend({"id": i, "detail": f"Update not visible: {mid}"} for mid in missing_ids)
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
                            field_type_map = build_field_type_map(dm.schema_config)
                            for row, vec in zip(upsert_rows, upsert_vecs):
                                rid = row["id"]
                                p = build_weaviate_properties(row, field_type_map)
                                match_idx = dm.df.index[dm.df["id"] == rid].tolist()
                                if match_idx:
                                    # Update existing
                                    col.data.replace(uuid=rid, properties=p, vector=vec.tolist())
                                else:
                                    # Insert new
                                    col.data.insert(properties=p, uuid=rid, vector=vec.tolist())
                            vectors_by_id = {str(row["id"]): vec for row, vec in zip(upsert_rows, upsert_vecs)}
                            synced_ids, missing_ids, new_ids = wm.sync_rows_from_engine(
                                dm,
                                [row["id"] for row in upsert_rows],
                                vectors_by_id=vectors_by_id,
                                retries=4,
                            )
                            dynamic_ids.update(synced_ids)
                            flog(f"[Dyn] Upserted req={len(upsert_rows)} visible={len(synced_ids)} new={len(new_ids)}")
                            if missing_ids:
                                flog(f"[Dyn] ❌ Upsert visibility miss: {missing_ids}")
                                fails.extend({"id": i, "detail": f"Upsert not visible: {mid}"} for mid in missing_ids)
                            if synced_ids:
                                wm.sync_creation_times(dm, ids=synced_ids)
                                wm.sync_update_times(dm, ids=synced_ids)
                                qg = OracleQueryGenerator(dm)
                        except Exception as e:
                            flog(f"[Dyn] Upsert fail: {e}")

                if enable_dynamic_ops and ENABLE_SCHEMA_EVOLUTION and i > 0 and i % SCHEMA_EVOLUTION_INTERVAL == 0:
                    try:
                        field_config = dm.plan_evolved_property()
                        if field_config:
                            field_name = field_config["name"]
                            wm.add_evolved_property(field_config)
                            dm.register_evolved_property(field_config)
                            backfill_data = dm.build_evolved_backfill_data(field_config)
                            success_ids, missing_ids = wm.backfill_evolved_property(dm, field_config, backfill_data, retries=4)
                            flog(
                                f"[Schema] Added property {field_name} type={field_config['type']} "
                                f"backfill={len(success_ids)}/{len(backfill_data)}"
                            )
                            if missing_ids:
                                flog(f"[Schema] ⚠️ Backfill incomplete for {field_name}: sample={missing_ids[:5]}")
                            if len(success_ids) == len(backfill_data):
                                if field_config.get("index_filterable", True):
                                    dm.filterable_fields.add(field_name)
                                if field_config.get("index_searchable", False):
                                    dm.searchable_text_fields.add(field_name)
                                wm.sync_update_times(dm, ids=success_ids)
                                qg = OracleQueryGenerator(dm)
                            else:
                                fails.append({"id": i, "detail": f"Schema evolution partial backfill: {field_name}"})
                    except Exception as e:
                        flog(f"[Schema] Failed: {e}")

                if ENABLE_SCALAR_INDEX_MUTATION and i > 0 and i % SCALAR_INDEX_MUTATION_INTERVAL == 0:
                    try:
                        mutated = wm.mutate_scalar_index(dm)
                        if mutated:
                            property_name, index_name = mutated
                            flog(f"[IndexMut] Dropped {index_name} index on {property_name}")
                            qg = OracleQueryGenerator(dm)
                    except Exception as e:
                        flog(f"[IndexMut] Failed: {e}")

                # Vector index maintenance pressure every 40 rounds
                if i > 0 and i % 40 == 0 and active_vector_index_type in {"hnsw", "flat", "hfresh", "dynamic"}:
                    try:
                        desc = wm.reconfigure_vector_index()
                        flog(f"[Reconfig] {desc}")
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

                depth_lo, depth_hi = (6, 14) if is_inverted_profile() else (2, 10)
                depth = random.randint(depth_lo, depth_hi)
                fo = None
                rest_where = None
                filter_source = "grpc"
                filter_meta = None
                build_ms = 0.0
                for _ in range(10):
                    if random.random() < ORACLE_REST_CROSSCHECK_RATE:
                        try:
                            build_t0 = time.time()
                            rest_candidate, pm_candidate, es_candidate, rest_meta = generate_rest_filter(qg)
                            build_ms = (time.time() - build_t0) * 1000.0
                            if rest_candidate is not None:
                                fo_candidate = rest_where_to_filter(rest_candidate)
                                fo, pm, es = fo_candidate, pm_candidate, es_candidate
                                rest_where = rest_candidate
                                filter_source = "grpc+rest"
                                depth = max(depth, int((rest_meta or {}).get("target_depth", depth)))
                                filter_meta = dict(rest_meta or {})
                                filter_meta.setdefault("target_atoms", filter_meta.get("atoms", 0))
                                break
                        except Exception:
                            pass
                    build_t0 = time.time()
                    fo, pm, es, filter_meta = qg.gen_complex_expr(depth, return_meta=True)
                    build_ms = (time.time() - build_t0) * 1000.0
                    if fo:
                        break
                if not fo:
                    stats.record_skip()
                    continue

                expr_depth = int((filter_meta or {}).get("depth", 0))
                expr_atoms = int((filter_meta or {}).get("atoms", 0))
                target_depth = int((filter_meta or {}).get("target_depth", depth))
                target_atoms = int((filter_meta or {}).get("target_atoms", expr_atoms))
                flog(f"\n[T{i}] {es}")
                flog(f"  Source={filter_source}")
                flog(f"  CL={cl}")
                flog(
                    f"  FilterMeta: depth={expr_depth} target={target_depth} "
                    f"atoms={expr_atoms}/{target_atoms} build_ms={build_ms:.1f}"
                )
                if rest_where is not None:
                    flog(f"  RestWhere={json.dumps(rest_where, ensure_ascii=False, sort_keys=True)}")
                pm = qg._normalize_mask(pm)
                exp = set(dm.df.loc[pm.fillna(False), "id"].tolist())

                try:
                    t0 = time.time()
                    col_cl = col.with_consistency_level(cl)
                    query_cap = max(1, min(len(dm.df) + 1, len(exp) + 1))
                    res = fetch_objects_resilient(
                        col_cl,
                        dm=dm,
                        filters=fo,
                        max_objects=query_cap,
                        return_properties=False,
                    )
                    act = set(str(o.uuid) for o in res.objects)
                    act_exhausted = res.exhausted
                    if not act_exhausted and query_cap < len(dm.df) + 1:
                        res = fetch_objects_resilient(
                            col_cl,
                            dm=dm,
                            filters=fo,
                            max_objects=len(dm.df) + 1,
                            return_properties=False,
                        )
                        act = set(str(o.uuid) for o in res.objects)
                        act_exhausted = res.exhausted
                    ms = (time.time() - t0) * 1000
                    flog(f"  Pandas:{len(exp)} | Weaviate:{capped_result_count(len(act), act_exhausted)} | {ms:.1f}ms")
                    round_failed = False

                    if act_exhausted and exp == act:
                        flog("  -> MATCH")
                    else:
                        mi, ex = exp - act, act - exp
                        dm_ = f"Missing:{len(mi)} Extra:{len(ex)}"
                        if not act_exhausted:
                            dm_ += f" TruncatedAt:{len(act)}"
                        # Check if mismatch involves ONLY dynamically modified rows
                        # Weaviate has a known bug: replace() corrupts inverted index
                        # for BOOL/BOOL_ARRAY/INT_ARRAY fields on modified rows.
                        mi_dyn = mi & dynamic_ids
                        ex_dyn = ex & dynamic_ids
                        is_weaviate_bug = act_exhausted and (mi == mi_dyn) and (ex == ex_dyn) and dynamic_ids
                        if cl != ConsistencyLevel.ALL:
                            flog(f"  -> WARN (CL={cl}): {dm_}")
                        elif is_weaviate_bug:
                            flog(f"  -> WEAVIATE_BUG (dynamic rows only): {dm_}")
                            flog(f"    Dynamic IDs in Extra: {len(ex_dyn)}, Missing: {len(mi_dyn)}")
                            stats.weaviate_bugs = getattr(stats, 'weaviate_bugs', 0) + 1
                        else:
                            print(f"\n❌ [T{i}] MISMATCH! {dm_}")
                            print(f"   Expr: {es}")
                            print(f"   🔑 {repro_cmd}")
                            flog(f"  -> MISMATCH! {dm_}")
                            if mi: flog(f"  Missing: {sample(mi)}")
                            if ex: flog(f"  Extra: {sample(ex)}")
                            for eid in sorted(ex)[:2]:
                                try:
                                    obj = col.query.fetch_object_by_id(eid)
                                    exists = obj is not None
                                    verdict, diffs = diagnose_id_mismatch(eid, es)
                                    flog(f"    ExtraID {eid}: exists={exists} verdict={verdict}")
                                    for d in diffs[:4]:
                                        flog(f"      {d}")
                                except Exception as ve:
                                    flog(f"    ExtraID verify err: {ve}")
                            for mid in sorted(mi)[:2]:
                                try:
                                    verdict, diffs = diagnose_id_mismatch(mid, es)
                                    flog(f"    MissingID {mid}: verdict={verdict}")
                                    for d in diffs[:4]:
                                        flog(f"      {d}")
                                except Exception as ve:
                                    flog(f"    MissingID verify err: {ve}")
                            fails.append({"id": i, "expr": es, "detail": dm_, "seed": current_seed})
                            round_failed = True

                    if rest_where is not None:
                        if len(exp) > REST_FILTER_MAX_COMPARE_RESULTS:
                            rest_crosscheck_skips += 1
                            flog(
                                f"  REST: SKIP expected rows {len(exp)} exceed RestCompareCap={REST_FILTER_MAX_COMPARE_RESULTS}"
                            )
                        else:
                            rest_crosschecks += 1
                            try:
                                rest_t0 = time.time()
                                rest_act = set(rest_batch_delete_dry_run(CLASS_NAME, rest_where, timeout=60))
                                rest_ms = (time.time() - rest_t0) * 1000.0
                                flog(f"  REST: Pandas:{len(exp)} | Rest:{len(rest_act)} | {rest_ms:.1f}ms")
                                if rest_act != exp:
                                    rest_missing, rest_extra = exp - rest_act, rest_act - exp
                                    rest_detail = f"REST Missing:{len(rest_missing)} Extra:{len(rest_extra)}"
                                    print(f"\n❌ [T{i}] REST MISMATCH! {rest_detail}")
                                    print(f"   Expr: {es}")
                                    print(f"   🔑 {repro_cmd}")
                                    flog(f"  REST: FAIL {rest_detail}")
                                    if rest_missing:
                                        flog(f"  REST Missing: {sample(rest_missing)}")
                                    if rest_extra:
                                        flog(f"  REST Extra: {sample(rest_extra)}")
                                    fails.append({"id": i, "expr": es, "detail": rest_detail, "seed": current_seed})
                                    round_failed = True
                                else:
                                    flog("  REST: MATCH")
                            except Exception as rest_exc:
                                rest_err = str(rest_exc)
                                if is_query_maximum_results_error(rest_err):
                                    rest_crosscheck_skips += 1
                                    update_query_limits_from_error(rest_err)
                                    flog(f"  REST: SKIP query maximum results ({get_query_maximum_results()}): {rest_err}")
                                else:
                                    raise

                    if round_failed:
                        stats.record(False, ms, expr_depth, "mismatch")
                    else:
                        stats.record(True, ms, expr_depth)

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
                            offset_window_pages = max(1, get_query_maximum_results() // ps)
                            if max_pages > offset_window_pages:
                                flog(
                                    f"  Page: capped offset scan {max_pages}->{offset_window_pages} pages "
                                    f"(QUERY_MAXIMUM_RESULTS={get_query_maximum_results()})"
                                )
                                max_pages = offset_window_pages

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
                            if is_query_maximum_results_error(e):
                                update_query_limits_from_error(e)
                                flog(f"  Page: SKIP offset window cap ({get_query_maximum_results()}): {e}")
                            else:
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
                    stats.record(False, 0, expr_depth, cat)
                    fails.append({"id": i, "expr": es, "detail": str(e)})

        print("\n" + "="*60)
        print(f"📊 Stats: {stats.summary()}")
        print(f"🔁 OracleREST: checks={rest_crosschecks} skips={rest_crosscheck_skips}")
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
        print(f" Reproduce: {repro_cmd}")
    finally:
        wm.close()


def run_graphql_probe_mode(rounds=12, seed=None):
    current_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)

    logf = make_log_path(stable_log_filename("weaviate_graphql_probe", current_seed, rounds, "graphql-probe"))
    repro_cmd = format_repro_command(
        current_seed,
        DEFAULT_CONSISTENCY_LEVEL,
        mode="graphql-probe",
        rounds=rounds,
        rows=N,
        dynamic_enabled=False,
        host=HOST,
        port=PORT,
    )
    collection_name = graphql_probe_collection_name(current_seed)

    where_range = graphql_with_enum_operators({
        "operator": "And",
        "operands": [
            {"path": ["bucket"], "operator": "GreaterThanEqual", "valueInt": 1},
            {"path": ["bucket"], "operator": "LessThanEqual", "valueInt": 2},
        ],
    })
    where_int_ge_not_lt = graphql_with_enum_operators({
        "operator": "And",
        "operands": [
            {"path": ["bucket"], "operator": "GreaterThanEqual", "valueInt": 1},
            {
                "operator": "Not",
                "operands": [
                    {"path": ["bucket"], "operator": "LessThan", "valueInt": 1},
                ],
            },
        ],
    })
    where_score_le_not_gt = graphql_with_enum_operators({
        "operator": "And",
        "operands": [
            {"path": ["score"], "operator": "LessThanEqual", "valueNumber": 2.5},
            {
                "operator": "Not",
                "operands": [
                    {"path": ["score"], "operator": "GreaterThan", "valueNumber": 2.5},
                ],
            },
        ],
    })
    where_bool_true_not_false = graphql_with_enum_operators({
        "operator": "And",
        "operands": [
            {"path": ["flag"], "operator": "Equal", "valueBoolean": True},
            {
                "operator": "Not",
                "operands": [
                    {"path": ["flag"], "operator": "Equal", "valueBoolean": False},
                ],
            },
        ],
    })
    where_date_ge_not_lt = graphql_with_enum_operators({
        "operator": "And",
        "operands": [
            {"path": ["eventTs"], "operator": "GreaterThanEqual", "valueDate": "2024-01-02T00:00:00Z"},
            {
                "operator": "Not",
                "operands": [
                    {"path": ["eventTs"], "operator": "LessThan", "valueDate": "2024-01-02T00:00:00Z"},
                ],
            },
        ],
    })
    where_text_eq_not_ne = graphql_with_enum_operators({
        "operator": "And",
        "operands": [
            {"path": ["nullableText"], "operator": "Equal", "valueText": "red"},
            {
                "operator": "Not",
                "operands": [
                    {"path": ["nullableText"], "operator": "NotEqual", "valueText": "red"},
                ],
            },
        ],
    })
    where_compound = graphql_with_enum_operators({
        "operator": "And",
        "operands": [
            {"path": ["flag"], "operator": "Equal", "valueBoolean": True},
            {
                "operator": "Or",
                "operands": [
                    {"path": ["tag"], "operator": "Like", "valueText": "a*"},
                    {"path": ["textArr"], "operator": "ContainsAny", "valueText": ["delta"]},
                ],
            },
        ],
    })
    where_nested_common = graphql_with_enum_operators({
        "operator": "And",
        "operands": [
            {"path": ["bucket"], "operator": "GreaterThanEqual", "valueInt": 1},
            {"path": ["bucket"], "operator": "LessThanEqual", "valueInt": 2},
            {
                "operator": "Or",
                "operands": [
                    {"path": ["flag"], "operator": "Equal", "valueBoolean": True},
                    {"path": ["nullableText"], "operator": "IsNull", "valueBoolean": True},
                ],
            },
        ],
    })
    agg_where = graphql_with_enum_operators({
        "operator": "And",
        "operands": [
            {"path": ["bucket"], "operator": "GreaterThanEqual", "valueInt": 1},
            {"path": ["flag"], "operator": "Equal", "valueBoolean": True},
        ],
    })

    expected_where_range = graphql_probe_expected_tags(
        current_seed,
        lambda props: 1 <= int(props["bucket"]) <= 2,
    )
    expected_where_int_ge_not_lt = graphql_probe_expected_tags(
        current_seed,
        lambda props: int(props["bucket"]) >= 1,
    )
    expected_where_score_le_not_gt = graphql_probe_expected_tags(
        current_seed,
        lambda props: float(props["score"]) <= 2.5,
    )
    expected_where_bool_true_not_false = graphql_probe_expected_tags(
        current_seed,
        lambda props: bool(props["flag"]),
    )
    expected_where_date_ge_not_lt = graphql_probe_expected_tags(
        current_seed,
        lambda props: str(props["eventTs"]) >= "2024-01-02T00:00:00Z",
    )
    expected_where_text_eq_not_ne = graphql_probe_expected_tags(
        current_seed,
        lambda props: props.get("nullableText") == "red",
    )
    expected_where_compound = graphql_probe_expected_tags(
        current_seed,
        lambda props: bool(props["flag"]) and (
            str(props["tag"]).startswith("a") or "delta" in (props.get("textArr") or [])
        ),
    )
    expected_where_nested_common = graphql_probe_expected_tags(
        current_seed,
        lambda props: 1 <= int(props["bucket"]) <= 2 and (bool(props["flag"]) or props.get("nullableText") is None),
    )
    expected_agg_where = [
        row["properties"]
        for row in graphql_probe_rows(current_seed)
        if int(row["properties"]["bucket"]) >= 1 and bool(row["properties"]["flag"])
    ]
    expected_group_counts = graphql_probe_expected_group_counts(current_seed, "bucket")

    cases = [
        (
            "get-where-range",
            (
                "{ Get { "
                f"{collection_name}(where:{graphql_input_literal(where_range)}, limit:20) "
                "{ tag _additional { id } } } }"
            ),
            lambda result: (
                sorted(graphql_get_tags(result, collection_name)) == expected_where_range,
                f"expected={expected_where_range} observed={sorted(graphql_get_tags(result, collection_name))}",
            ),
        ),
        (
            "get-where-int-ge-not-lt",
            (
                "{ Get { "
                f"{collection_name}(where:{graphql_input_literal(where_int_ge_not_lt)}, limit:20) "
                "{ tag } } }"
            ),
            lambda result: (
                sorted(graphql_get_tags(result, collection_name)) == expected_where_int_ge_not_lt,
                f"expected={expected_where_int_ge_not_lt} observed={sorted(graphql_get_tags(result, collection_name))}",
            ),
        ),
        (
            "get-where-score-le-not-gt",
            (
                "{ Get { "
                f"{collection_name}(where:{graphql_input_literal(where_score_le_not_gt)}, limit:20) "
                "{ tag } } }"
            ),
            lambda result: (
                sorted(graphql_get_tags(result, collection_name)) == expected_where_score_le_not_gt,
                f"expected={expected_where_score_le_not_gt} observed={sorted(graphql_get_tags(result, collection_name))}",
            ),
        ),
        (
            "get-where-bool-true-not-false",
            (
                "{ Get { "
                f"{collection_name}(where:{graphql_input_literal(where_bool_true_not_false)}, limit:20) "
                "{ tag } } }"
            ),
            lambda result: (
                sorted(graphql_get_tags(result, collection_name)) == expected_where_bool_true_not_false,
                f"expected={expected_where_bool_true_not_false} observed={sorted(graphql_get_tags(result, collection_name))}",
            ),
        ),
        (
            "get-where-date-ge-not-lt",
            (
                "{ Get { "
                f"{collection_name}(where:{graphql_input_literal(where_date_ge_not_lt)}, limit:20) "
                "{ tag } } }"
            ),
            lambda result: (
                sorted(graphql_get_tags(result, collection_name)) == expected_where_date_ge_not_lt,
                f"expected={expected_where_date_ge_not_lt} observed={sorted(graphql_get_tags(result, collection_name))}",
            ),
        ),
        (
            "get-where-text-eq-not-ne",
            (
                "{ Get { "
                f"{collection_name}(where:{graphql_input_literal(where_text_eq_not_ne)}, limit:20) "
                "{ tag } } }"
            ),
            lambda result: (
                sorted(graphql_get_tags(result, collection_name)) == expected_where_text_eq_not_ne,
                f"expected={expected_where_text_eq_not_ne} observed={sorted(graphql_get_tags(result, collection_name))}",
            ),
        ),
        (
            "get-where-compound",
            (
                "{ Get { "
                f"{collection_name}(where:{graphql_input_literal(where_compound)}, limit:20) "
                "{ tag } } }"
            ),
            lambda result: (
                sorted(graphql_get_tags(result, collection_name)) == expected_where_compound,
                f"expected={expected_where_compound} observed={sorted(graphql_get_tags(result, collection_name))}",
            ),
        ),
        (
            "get-where-range-or-null",
            (
                "{ Get { "
                f"{collection_name}(where:{graphql_input_literal(where_nested_common)}, limit:20) "
                "{ tag } } }"
            ),
            lambda result: (
                sorted(graphql_get_tags(result, collection_name)) == expected_where_nested_common,
                f"expected={expected_where_nested_common} observed={sorted(graphql_get_tags(result, collection_name))}",
            ),
        ),
        (
            "get-bm25-and",
            (
                "{ Get { "
                f"{collection_name}(bm25:{graphql_input_literal({'query': 'nebula anchor', 'properties': ['body'], 'searchOperator': {'operator': GraphQLEnum('And')}})}, limit:5) "
                "{ tag } } }"
            ),
            lambda result: (
                sorted(graphql_get_tags(result, collection_name)) == ["alpha"],
                f"expected=['alpha'] observed={sorted(graphql_get_tags(result, collection_name))}",
            ),
        ),
        (
            "get-hybrid-top1",
            (
                "{ Get { "
                f"{collection_name}(hybrid:{graphql_input_literal({'query': 'nebula anchor', 'alpha': 0.2, 'vector': [1.0, 0.0, 0.0, 0.0], 'properties': ['body'], 'bm25SearchOperator': {'operator': GraphQLEnum('And')}})}, limit:3) "
                "{ tag } } }"
            ),
            lambda result: (
                bool(graphql_get_tags(result, collection_name))
                and graphql_get_tags(result, collection_name)[0] == "alpha",
                f"top_tags={graphql_get_tags(result, collection_name)}",
            ),
        ),
        (
            "aggregate-where-score",
            (
                "{ Aggregate { "
                f"{collection_name}(where:{graphql_input_literal(agg_where)}) "
                "{ meta { count } score { count minimum maximum sum mean } } } }"
            ),
            lambda result: (
                (lambda groups: (
                    len(groups) == 1
                    and int((groups[0].get("meta") or {}).get("count") or 0) == len(expected_agg_where)
                    and int((groups[0].get("score") or {}).get("count") or 0) == len(expected_agg_where)
                    and _floats_match(float((groups[0].get("score") or {}).get("minimum") or 0.0), min(float(row["score"]) for row in expected_agg_where))
                    and _floats_match(float((groups[0].get("score") or {}).get("maximum") or 0.0), max(float(row["score"]) for row in expected_agg_where))
                    and _floats_match(float((groups[0].get("score") or {}).get("sum") or 0.0), sum(float(row["score"]) for row in expected_agg_where))
                    and _floats_match(float((groups[0].get("score") or {}).get("mean") or 0.0), sum(float(row["score"]) for row in expected_agg_where) / len(expected_agg_where))
                ))(graphql_aggregate_groups(result, collection_name)),
                f"observed={graphql_aggregate_groups(result, collection_name)}",
            ),
        ),
        (
            "aggregate-groupby-bucket",
            (
                "{ Aggregate { "
                f"{collection_name}(groupBy:[\"bucket\"]) "
                "{ meta { count } groupedBy { path value } } } }"
            ),
            lambda result: (
                (lambda groups: (
                    {str((group.get('groupedBy') or {}).get('value')): int((group.get('meta') or {}).get('count') or 0) for group in groups} == expected_group_counts
                    and all((group.get("groupedBy") or {}).get("path") == ["bucket"] for group in groups)
                ))(graphql_aggregate_groups(result, collection_name)),
                f"expected={expected_group_counts} observed={graphql_aggregate_groups(result, collection_name)}",
            ),
        ),
        (
            "aggregate-near-vector-local-boundary",
            (
                "{ Aggregate { "
                f"{collection_name}(objectLimit:1 nearVector:{graphql_input_literal({'vector': [1.0, 0.0, 0.0, 0.0], 'distance': 0.0002})}) "
                "{ meta { count } score { count minimum maximum } } } }"
            ),
            lambda result: (
                (lambda groups: (
                    len(groups) == 1
                    and int((groups[0].get("meta") or {}).get("count") or 0) == 0
                    and int((groups[0].get("score") or {}).get("count") or 0) == 0
                ))(graphql_aggregate_groups(result, collection_name)),
                f"local-boundary observed={graphql_aggregate_groups(result, collection_name)}",
            ),
        ),
        (
            "aggregate-hybrid-object-limit",
            (
                "{ Aggregate { "
                f"{collection_name}(objectLimit:1 hybrid:{graphql_input_literal({'query': 'nebula anchor', 'alpha': 0.0, 'properties': ['body'], 'bm25SearchOperator': {'operator': GraphQLEnum('And')}})}) "
                "{ meta { count } score { count minimum maximum } } } }"
            ),
            lambda result: (
                (lambda groups: (
                    len(groups) == 1
                    and int((groups[0].get("meta") or {}).get("count") or 0) == 1
                    and int((groups[0].get("score") or {}).get("count") or 0) == 1
                    and _floats_match(float((groups[0].get("score") or {}).get("minimum") or 0.0), 1.0)
                    and _floats_match(float((groups[0].get("score") or {}).get("maximum") or 0.0), 1.0)
                ))(graphql_aggregate_groups(result, collection_name)),
                f"observed={graphql_aggregate_groups(result, collection_name)}",
            ),
        ),
    ]

    print(
        f"\n GraphQL Probe Mode | Seed: {current_seed} | Class: {collection_name} | "
        f"Cases/Loop: {len(cases)}"
    )
    print(f"📄 Log: {display_path(logf)}")
    print(f"🔁 Reproduce: {repro_cmd}")

    wm = WeaviateManager()
    wm.connect()
    try:
        reset_graphql_probe_collection(wm.client, collection_name, current_seed)
        stats = FuzzStats()
        failures = []

        offset = graphql_probe_case_order(current_seed, len(cases))
        with open(logf, "w", encoding="utf-8") as f:
            def flog(message):
                f.write(message + "\n")
                f.flush()

            flog(f"GraphQL Probe | Seed:{current_seed} | Class:{collection_name}")
            flog(f"Reproduce: {repro_cmd}")
            flog("=" * 80)

            for i in range(rounds):
                case_name, query, checker = cases[(offset + i) % len(cases)]
                print(f"\r⏳ GraphQL {i+1}/{rounds} [{case_name}]   ", end="", flush=True)
                t0 = time.time()
                try:
                    result = graphql_raw_query_or_raise(wm.client, query)
                    ok, detail = checker(result)
                    error_cat = None if ok else case_name
                except Exception as exc:
                    ok, detail, error_cat = False, f"ERR: {exc}", f"{case_name}_error"
                latency_ms = (time.time() - t0) * 1000.0
                stats.record(ok, latency_ms=latency_ms, error_cat=error_cat)
                flog(f"\n[T{i}] {case_name}")
                flog(f"  Query: {query}")
                flog(f"  Result: {'PASS' if ok else 'FAIL'} {latency_ms:.1f}ms")
                flog(f"  Detail: {detail}")
                if not ok:
                    failures.append({"id": i, "case": case_name, "detail": detail, "query": query})

        print()
        print(f"📊 GraphQL Probe: {stats.summary()}")
        print(f"{'✅ All passed' if not failures else f'🚫 {len(failures)} failures'}. Log: {display_path(logf)}")
    finally:
        try:
            wm.client.collections.delete(collection_name)
        except Exception:
            pass
        wm.close()


def run_equivalence_mode(rounds=100, seed=None):
    seed = initialize_seeded_run(seed)

    logf = make_log_path(stable_log_filename("weaviate_equiv_test", seed, rounds, "equiv"))
    print(f"\n Equivalence Mode | Seed: {seed} | Profile: {get_fuzz_profile()} | VecIdx: {VECTOR_INDEX_TYPE} | BoundaryRate: {BOUNDARY_INJECTION_RATE:.2f}")
    print(
        f"   QueryPageSize: {get_query_page_size()} (CapHint={get_query_maximum_results()}, "
        f"Partition={get_filter_partition_size()}, IdBatch={get_filter_id_batch_size()})"
    )
    print(f"   AggregateFilterDepth: {AGGREGATE_FILTER_MIN_DEPTH}..{AGGREGATE_FILTER_MAX_DEPTH}")
    print(f"📄 Log: {display_path(logf)}")

    dm = DataManager(seed); dm.generate_schema(); dm.generate_data()
    wm = WeaviateManager(); wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config); dm.searchable_text_fields = set(getattr(wm, "searchable_text_fields", set()) or set()); dm.field_index_state = dict(getattr(wm, "field_index_state", {}) or {}); wm.insert(dm); wm.sync_creation_times(dm); wm.sync_update_times(dm)
        qg = EquivalenceQueryGenerator(dm)
        col = wm.client.collections.get(CLASS_NAME)
        fails = []
        fetch_cap = len(dm.df) + 1
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
                    br = fetch_objects_resilient(col, dm=dm, filters=bf, max_objects=fetch_cap, return_properties=False)
                    bi = set(str(o.uuid) for o in br.objects)
                    flog(f"  Base: {capped_result_count(len(bi), br.exhausted)}")
                    for mut in qg.mutate_filter(bf, be):
                        try:
                            mr = fetch_objects_resilient(col, dm=dm, filters=mut["filter"], max_objects=fetch_cap, return_properties=False)
                            mi = set(str(o.uuid) for o in mr.objects)
                            if br.exhausted and mr.exhausted and bi == mi:
                                flog(f"  {mut['type']}: PASS")
                            else:
                                diff = bi.symmetric_difference(mi)
                                detail = f"diff:{len(diff)} base={capped_result_count(len(bi), br.exhausted)} mut={capped_result_count(len(mi), mr.exhausted)}"
                                flog(f"  {mut['type']}: FAIL ({detail})")
                                print(f"\n❌ [T{i}] {mut['type']} FAIL ({detail})")
                                # Deep evidence: print details of diff IDs
                                for did in sorted(diff)[:3]:
                                    row = dm.df[dm.df['id'] == did]
                                    if not row.empty:
                                        null_cols = [c for c in row.columns if row[c].isna().any()]
                                        flog(f"    DiffID {did}: null_cols={null_cols}")
                                        for ff in dm.schema_config[:5]:
                                            v = row.iloc[0].get(ff['name'])
                                            flog(f"      {ff['name']}={v}")
                                fails.append({"id": i, "type": mut["type"], "detail": detail})
                        except Exception as e:
                            flog(f"  {mut['type']}: ERR {e}")
                except Exception as e:
                    flog(f"  ERR: {e}")
        print(f"\n{'✅ All passed' if not fails else f'🚫 {len(fails)} failures'}. Log: {display_path(logf)}")
    finally:
        wm.close()


def run_pqs_mode(rounds=100, seed=None):
    seed = initialize_seeded_run(seed)

    logf = make_log_path(stable_log_filename("weaviate_pqs_test", seed, rounds, "pqs"))
    print(f"\n PQS Mode | Seed: {seed} | Profile: {get_fuzz_profile()} | VecIdx: {VECTOR_INDEX_TYPE} | BoundaryRate: {BOUNDARY_INJECTION_RATE:.2f}")
    print(
        f"   QueryPageSize: {get_query_page_size()} (CapHint={get_query_maximum_results()}, "
        f"Partition={get_filter_partition_size()}, IdBatch={get_filter_id_batch_size()})"
    )
    print(f"📄 Log: {display_path(logf)}")

    dm = DataManager(seed); dm.generate_schema(); dm.generate_data()
    wm = WeaviateManager(); wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config); dm.searchable_text_fields = set(getattr(wm, "searchable_text_fields", set()) or set()); dm.field_index_state = dict(getattr(wm, "field_index_state", {}) or {}); wm.insert(dm); wm.sync_creation_times(dm); wm.sync_update_times(dm)
        pqs = PQSQueryGenerator(dm)
        col = wm.client.collections.get(CLASS_NAME)
        errs, ok, skip = [], 0, 0
        fetch_cap = len(dm.df) + 1
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
                    res = fetch_objects_resilient(col, dm=dm, filters=pf, max_objects=fetch_cap, return_properties=False)
                    ri = set(str(o.uuid) for o in res.objects)
                    if pid in ri:
                        flog(f"  PASS ({capped_result_count(len(ri), res.exhausted)})"); ok += 1
                    else:
                        flog(f"  FAIL! Missing pivot ({capped_result_count(len(ri), res.exhausted)})")
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

    logf = make_log_path(stable_log_filename("weaviate_groupby_test", seed, rounds, "groupby"))
    print(f"\n GroupBy Mode | Seed: {seed} | Profile: {get_fuzz_profile()} | VecIdx: {VECTOR_INDEX_TYPE} | BoundaryRate: {BOUNDARY_INJECTION_RATE:.2f}")
    print(
        f"   QueryPageSize: {get_query_page_size()} (CapHint={get_query_maximum_results()}, "
        f"Partition={get_filter_partition_size()}, IdBatch={get_filter_id_batch_size()})"
    )
    print(f"📄 Log: {display_path(logf)}")

    dm = DataManager(seed); dm.generate_schema(); dm.generate_data()
    wm = WeaviateManager(); wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config); dm.searchable_text_fields = set(getattr(wm, "searchable_text_fields", set()) or set()); dm.field_index_state = dict(getattr(wm, "field_index_state", {}) or {}); wm.insert(dm)
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

                    response_groups = getattr(res, "groups", None)
                    groups = {}
                    group_source = "flattened objects"
                    if isinstance(response_groups, dict):
                        group_source = "response.groups"
                        for group_name in sorted(response_groups, key=lambda item: str(item)):
                            group_data = response_groups[group_name]
                            groups[str(group_name)] = [str(o.uuid) for o in (getattr(group_data, "objects", None) or [])]
                    else:
                        group_source = "belongs_to_group"
                        for o in objs:
                            gkey = getattr(o, "belongs_to_group", None)
                            if gkey is None:
                                gval = o.properties.get(gf["name"])
                                gkey = "" if gval is None else str(gval)
                            groups.setdefault(str(gkey), []).append(str(o.uuid))

                    actual_groups = len(groups)
                    max_per = max(len(v) for v in groups.values()) if groups else 0
                    flattened_total = len(objs)
                    grouped_total = sum(len(v) for v in groups.values())
                    flog(f"  groups={actual_groups}/{n_groups} max_per={max_per}/{per_group} total={flattened_total} grouped_total={grouped_total} via={group_source}")
                    if actual_groups > n_groups:
                        flog(f"  ❌ Too many groups!")
                        errs.append({"id": i, "detail": f"groups {actual_groups}>{n_groups}"})
                    elif max_per > per_group:
                        flog(f"  ❌ Too many per group!")
                        errs.append({"id": i, "detail": f"per_group {max_per}>{per_group}"})
                    elif grouped_total != flattened_total:
                        flog(f"  ❌ Group/object total mismatch!")
                        errs.append({"id": i, "detail": f"group_total {grouped_total}!={flattened_total}"})
                    else:
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


def run_aggregate_mode(rounds=100, seed=None, consistency=DEFAULT_CONSISTENCY_LEVEL, randomize_consistency=False):
    current_seed = initialize_seeded_run(seed)
    resolved_consistency = consistency or DEFAULT_CONSISTENCY_LEVEL

    logf = make_log_path(
        stable_log_filename(
            "weaviate_aggregate_test",
            current_seed,
            rounds,
            "aggregate",
            f"consistency-{consistency_label(resolved_consistency, randomize=randomize_consistency)}",
        )
    )
    print(
        f"\n Aggregate Mode | Seed: {current_seed} | Profile: {get_fuzz_profile()} | VecIdx: {VECTOR_INDEX_TYPE} | "
        f"Consistency: aggregate-default"
    )
    print(
        f"   QueryPageSize: {get_query_page_size()} (CapHint={get_query_maximum_results()}, "
        f"Partition={get_filter_partition_size()}, IdBatch={get_filter_id_batch_size()})"
    )
    print(f"📄 Log: {display_path(logf)}")

    dm = DataManager(current_seed)
    dm.generate_schema()
    dm.generate_data()
    inject_aggregate_probe_fields(dm)

    wm = WeaviateManager()
    wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config)
        dm.searchable_text_fields = set(getattr(wm, "searchable_text_fields", set()) or set())
        dm.field_index_state = dict(getattr(wm, "field_index_state", {}) or {})
        print(f"   ActualVecIndex: {getattr(wm, 'actual_vector_index_type', VECTOR_INDEX_TYPE)}, ActualDist: {getattr(wm, 'actual_distance_metric', DISTANCE_METRIC)}")
        wm.insert(dm)

        col = wm.client.collections.get(CLASS_NAME)
        probe_fields = aggregate_probe_field_map(dm)
        all_probe_fields = [probe_fields[name] for name in sorted(probe_fields)]
        if not all_probe_fields:
            print("  No aggregate probe fields available, skip.")
            return

        stats = FuzzStats()
        failures = []
        semantic_skips = 0

        with open(logf, "w", encoding="utf-8") as f:
            def flog(msg):
                f.write(msg + "\n")
                f.flush()

            flog(
                f"Aggregate | Seed:{current_seed} | Profile:{get_fuzz_profile()} | Consistency:{consistency_label(resolved_consistency, randomize=randomize_consistency)}"
            )

            for i in range(rounds):
                print(f"\r⏳ Aggregate {i+1}/{rounds}", end="", flush=True)
                cl = "aggregate-default"
                text_limit = random.choice([3, 5, 8])
                filters, filter_mask, filter_expr, filter_meta = generate_aggregate_filter(dm)
                filter_mask = _normalize_bool_mask(filter_mask, dm.df.index)
                subset = select_rows_by_mask(dm.df, filter_mask)
                expr_depth = int((filter_meta or {}).get("depth", 0))
                expr_atoms = int((filter_meta or {}).get("atoms", 0))
                target_depth = int((filter_meta or {}).get("target_depth", 0))

                if i % 6 == 0:
                    target_fields = list(all_probe_fields)
                    case_label = "all-probes"
                else:
                    primary = random.choice(all_probe_fields)
                    target_fields = [primary]
                    if random.random() < 0.30:
                        siblings = [
                            candidate
                            for candidate in all_probe_fields
                            if candidate["name"] != primary["name"]
                            and aggregate_family_for_type(candidate["type"]) == aggregate_family_for_type(primary["type"])
                        ]
                        if siblings:
                            target_fields.append(random.choice(siblings))
                    case_label = ",".join(field["name"] for field in target_fields)

                expected = {
                    field["name"]: build_expected_aggregate(field, subset, text_limit=text_limit)
                    for field in target_fields
                }
                metrics = [
                    build_aggregate_metrics(field, text_limit=text_limit)
                    for field in target_fields
                ]

                flog(f"\n[T{i}] {case_label}")
                flog(f"  CL={cl}")
                flog(f"  Filter: {filter_expr}")
                flog(f"  FilterMeta: depth={expr_depth} target={target_depth} atoms={expr_atoms}")
                flog(f"  MatchedRows: {len(subset)}")
                try:
                    t0 = time.time()
                    result = col.aggregate.over_all(
                        filters=filters,
                        total_count=True,
                        return_metrics=metrics,
                    )
                    latency_ms = (time.time() - t0) * 1000.0
                    mismatches = []

                    actual_total_count = int(getattr(result, "total_count", 0) or 0)
                    expected_total_count = int(len(subset))
                    if actual_total_count != expected_total_count:
                        mismatches.append(
                            f"total_count: expected {expected_total_count}, got {actual_total_count}"
                        )

                    actual_properties = dict(getattr(result, "properties", {}) or {})
                    for field in target_fields:
                        field_name = field["name"]
                        if field_name not in actual_properties:
                            mismatches.append(f"{field_name}: missing in aggregate response")
                            continue

                        expected_prop = expected[field_name]
                        prop_mismatches = compare_aggregate_property(
                            actual_properties[field_name],
                            expected_prop,
                            field,
                        )
                        if not expected_prop.get("_top_occurrences_stable", True):
                            semantic_skips += 1
                            flog(f"  Note: skipped strict top_occurrences ordering for {field_name} due to cutoff tie")
                        if prop_mismatches:
                            mismatches.extend([f"{field_name}.{entry}" for entry in prop_mismatches])

                    if mismatches:
                        stats.record(False, latency_ms=latency_ms, depth=expr_depth)
                        failures.append(
                            {
                                "id": i,
                                "filter": filter_expr,
                                "fields": [field["name"] for field in target_fields],
                                "detail": "; ".join(mismatches),
                            }
                        )
                        flog(f"  FAIL: {' | '.join(mismatches)}")
                    else:
                        stats.record(True, latency_ms=latency_ms, depth=expr_depth)
                        flog(f"  PASS {latency_ms:.1f}ms")
                except Exception as exc:
                    stats.record(False, depth=expr_depth, error_cat="query_error")
                    failures.append(
                        {
                            "id": i,
                            "filter": filter_expr,
                            "fields": [field["name"] for field in target_fields],
                            "detail": str(exc),
                        }
                    )
                    flog(f"  ERR: {exc}")

        print()
        print(f"📊 Aggregate: {stats.summary()} | SemanticSkips:{semantic_skips}")
        print(f"{'✅ All passed' if not failures else f'🚫 {len(failures)} failures'}. Log: {display_path(logf)}")
    finally:
        wm.close()


def run_rest_filter_mode(rounds=100, seed=None, consistency=DEFAULT_CONSISTENCY_LEVEL, randomize_consistency=False):
    current_seed = initialize_seeded_run(seed)
    resolved_consistency = consistency or DEFAULT_CONSISTENCY_LEVEL

    logf = make_log_path(
        stable_log_filename(
            "weaviate_rest_filter_test",
            current_seed,
            rounds,
            "rest-filter",
            f"consistency-{consistency_label(resolved_consistency, randomize=randomize_consistency)}",
        )
    )
    print(
        f"\n REST Filter Mode | Seed: {current_seed} | Profile: {get_fuzz_profile()} | VecIdx: {VECTOR_INDEX_TYPE} | "
        f"Consistency: rest-default"
    )
    print(
        f"   QueryPageSize: {get_query_page_size()} (CapHint={get_query_maximum_results()}, "
        f"RestCompareCap={REST_FILTER_MAX_COMPARE_RESULTS})"
    )
    print(f"   RestFilterDepth: {REST_FILTER_MIN_DEPTH}..{REST_FILTER_MAX_DEPTH}")
    print(f"📄 Log: {display_path(logf)}")

    dm = DataManager(current_seed)
    dm.generate_schema()
    dm.generate_data()
    inject_aggregate_probe_fields(dm)

    wm = WeaviateManager()
    wm.connect()
    try:
        dm.filterable_fields = wm.reset_collection(dm.schema_config)
        dm.searchable_text_fields = set(getattr(wm, "searchable_text_fields", set()) or set())
        dm.field_index_state = dict(getattr(wm, "field_index_state", {}) or {})
        print(f"   ActualVecIndex: {getattr(wm, 'actual_vector_index_type', VECTOR_INDEX_TYPE)}, ActualDist: {getattr(wm, 'actual_distance_metric', DISTANCE_METRIC)}")
        wm.insert(dm)

        col = wm.client.collections.get(CLASS_NAME)
        qg = OracleQueryGenerator(dm)
        failures = []
        stats = FuzzStats()

        with open(logf, "w", encoding="utf-8") as f:
            def flog(msg):
                f.write(msg + "\n")
                f.flush()

            flog(
                f"REST Filter | Seed:{current_seed} | Profile:{get_fuzz_profile()} | Consistency:{consistency_label(resolved_consistency, randomize=randomize_consistency)}"
            )

            for i in range(rounds):
                print(f"\r⏳ REST {i+1}/{rounds}", end="", flush=True)
                where_filter, mask, expr, meta = generate_rest_filter(qg)
                grpc_filter = rest_where_to_filter(where_filter)
                mask = _normalize_bool_mask(mask, dm.df.index)
                expected_ids = sorted(str(object_id) for object_id in dm.df.loc[mask.fillna(False), "id"].tolist())
                expected_set = set(expected_ids)
                expr_depth = int((meta or {}).get("depth", 0))
                expr_atoms = int((meta or {}).get("atoms", 0))
                target_depth = int((meta or {}).get("target_depth", 0))

                flog(f"\n[T{i}] {expr}")
                flog(f"  FilterMeta: depth={expr_depth} target={target_depth} atoms={expr_atoms}")
                flog(f"  ExpectedRows: {len(expected_ids)}")
                flog(f"  WhereJSON: {json.dumps(where_filter, ensure_ascii=False, sort_keys=True)}")

                if len(expected_ids) > REST_FILTER_MAX_COMPARE_RESULTS:
                    stats.record_skip()
                    flog(
                        f"  SKIP expected rows {len(expected_ids)} exceed RestCompareCap={REST_FILTER_MAX_COMPARE_RESULTS}"
                    )
                    continue

                try:
                    grpc_t0 = time.time()
                    query_cap = max(1, min(len(dm.df) + 1, len(expected_set) + 1))
                    grpc_res = fetch_objects_resilient(
                        col,
                        dm=dm,
                        filters=grpc_filter,
                        max_objects=query_cap,
                        return_properties=False,
                    )
                    grpc_ids = set(str(obj.uuid) for obj in grpc_res.objects)
                    grpc_exhausted = grpc_res.exhausted
                    if not grpc_exhausted and query_cap < len(dm.df) + 1:
                        grpc_res = fetch_objects_resilient(
                            col,
                            dm=dm,
                            filters=grpc_filter,
                            max_objects=len(dm.df) + 1,
                            return_properties=False,
                        )
                        grpc_ids = set(str(obj.uuid) for obj in grpc_res.objects)
                        grpc_exhausted = grpc_res.exhausted
                    grpc_ms = (time.time() - grpc_t0) * 1000.0

                    rest_t0 = time.time()
                    rest_ids = set(rest_batch_delete_dry_run(CLASS_NAME, where_filter, timeout=60))
                    rest_ms = (time.time() - rest_t0) * 1000.0
                    latency_ms = grpc_ms + rest_ms

                    grpc_match = grpc_exhausted and grpc_ids == expected_set
                    rest_match = rest_ids == expected_set

                    flog(
                        f"  Pandas:{len(expected_set)} | gRPC:{capped_result_count(len(grpc_ids), grpc_exhausted)} "
                        f"| REST:{len(rest_ids)} | {latency_ms:.1f}ms"
                    )

                    if grpc_match and rest_match:
                        stats.record(True, latency_ms=latency_ms, depth=expr_depth)
                        flog("  PASS")
                    else:
                        if grpc_match and not rest_match:
                            category = "rest_only_mismatch"
                        elif not grpc_match and rest_match:
                            category = "grpc_only_mismatch"
                        else:
                            category = "grpc_and_rest_mismatch"

                        grpc_missing = sorted(expected_set - grpc_ids)
                        grpc_extra = sorted(grpc_ids - expected_set)
                        rest_missing = sorted(expected_set - rest_ids)
                        rest_extra = sorted(rest_ids - expected_set)
                        detail = (
                            f"{category}: "
                            f"gRPC(M:{len(grpc_missing)} E:{len(grpc_extra)}"
                            f"{'' if grpc_exhausted else ' truncated'}) "
                            f"REST(M:{len(rest_missing)} E:{len(rest_extra)})"
                        )
                        failures.append({"id": i, "expr": expr, "detail": detail})
                        stats.record(False, latency_ms=latency_ms, depth=expr_depth, error_cat=category)
                        flog(f"  FAIL: {detail}")
                        if grpc_missing:
                            flog(f"  gRPC MissingSample: {grpc_missing[:5]}")
                        if grpc_extra:
                            flog(f"  gRPC ExtraSample: {grpc_extra[:5]}")
                        if rest_missing:
                            flog(f"  REST MissingSample: {rest_missing[:5]}")
                        if rest_extra:
                            flog(f"  REST ExtraSample: {rest_extra[:5]}")
                except Exception as exc:
                    err = str(exc)
                    if is_query_maximum_results_error(err):
                        update_query_limits_from_error(err)
                        stats.record_skip()
                        flog(f"  SKIP query maximum results: {err}")
                        continue
                    failures.append({"id": i, "expr": expr, "detail": err})
                    stats.record(False, depth=expr_depth, error_cat="query_error")
                    flog(f"  ERR: {err}")

        print()
        print(f"📊 REST Filter: {stats.summary()}")
        print(f"{'✅ All passed' if not failures else f'🚫 {len(failures)} failures'}. Log: {display_path(logf)}")
    finally:
        wm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weaviate Fuzz Oracle V2")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--rounds", type=int, default=500, help="Test rounds (default: 500)")
    parser.add_argument(
        "--mode",
        choices=["oracle", "equiv", "pqs", "groupby", "aggregate", "rest-filter", "graphql-probe"],
        default="oracle",
        help="Test mode",
    )
    parser.add_argument("--profile", choices=ALL_FUZZ_PROFILES, default=FUZZ_PROFILE_DEFAULT, help="Schema/query generation profile")
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
    parser.add_argument("--query-page-size", type=int, default=None, help="Cursor fetch page size for large-result queries")
    parser.add_argument("--aggregate-filter-min-depth", type=int, default=None, help="Minimum recursive depth for aggregate-mode filters")
    parser.add_argument("--aggregate-filter-max-depth", type=int, default=None, help="Maximum recursive depth for aggregate-mode filters")
    parser.add_argument("--rest-filter-min-depth", type=int, default=None, help="Minimum recursive depth for REST filter-mode filters")
    parser.add_argument("--rest-filter-max-depth", type=int, default=None, help="Maximum recursive depth for REST filter-mode filters")
    parser.add_argument("--rest-filter-max-compare-results", type=int, default=None, help="Skip REST dry-run comparisons above this matched-row count")
    parser.add_argument("--oracle-rest-crosscheck-rate", type=float, default=None, help="Probability of using a REST-crosschecked stable filter inside oracle mode")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for deterministic fuzzer-internal logs")
    parser.add_argument("--log-suffix", type=str, default=None, help="Stable suffix appended to fuzzer-internal log filenames")
    args = parser.parse_args()
    set_fuzz_profile(args.profile)
    set_log_dir(args.log_dir)
    set_log_suffix(args.log_suffix)

    if args.host: HOST = args.host
    if args.port: PORT = args.port
    if args.N: N = args.N
    if args.query_page_size: QUERY_PAGE_SIZE = max(1, args.query_page_size)
    if args.aggregate_filter_min_depth:
        AGGREGATE_FILTER_MIN_DEPTH = max(1, int(args.aggregate_filter_min_depth))
    if args.aggregate_filter_max_depth:
        AGGREGATE_FILTER_MAX_DEPTH = max(AGGREGATE_FILTER_MIN_DEPTH, int(args.aggregate_filter_max_depth))
    if args.rest_filter_min_depth:
        REST_FILTER_MIN_DEPTH = max(1, int(args.rest_filter_min_depth))
    if args.rest_filter_max_depth:
        REST_FILTER_MAX_DEPTH = max(REST_FILTER_MIN_DEPTH, int(args.rest_filter_max_depth))
    if args.rest_filter_max_compare_results:
        REST_FILTER_MAX_COMPARE_RESULTS = max(1, int(args.rest_filter_max_compare_results))
    if args.oracle_rest_crosscheck_rate is not None:
        ORACLE_REST_CROSSCHECK_RATE = max(0.0, min(1.0, float(args.oracle_rest_crosscheck_rate)))

    resolved_consistency = parse_consistency_arg(args.consistency)

    print("=" * 80)
    print(
        f" Weaviate Fuzz Oracle V2 | Mode: {args.mode} | Rounds: {args.rounds} | "
        f"Seed: {args.seed or '(random)'} | Profile: {get_fuzz_profile()} | Dynamic: {'OFF' if args.no_dynamic else 'ON'} | "
        f"Consistency: {consistency_label(resolved_consistency, randomize=args.random_consistency)}"
    )
    print("=" * 80)

    if args.mode == "equiv":
        run_equivalence_mode(rounds=args.rounds, seed=args.seed)
    elif args.mode == "pqs":
        run_pqs_mode(rounds=args.rounds, seed=args.seed)
    elif args.mode == "groupby":
        run_groupby_mode(rounds=args.rounds, seed=args.seed)
    elif args.mode == "aggregate":
        run_aggregate_mode(
            rounds=args.rounds,
            seed=args.seed,
            consistency=resolved_consistency,
            randomize_consistency=args.random_consistency,
        )
    elif args.mode == "rest-filter":
        run_rest_filter_mode(
            rounds=args.rounds,
            seed=args.seed,
            consistency=resolved_consistency,
            randomize_consistency=args.random_consistency,
        )
    elif args.mode == "graphql-probe":
        run_graphql_probe_mode(rounds=args.rounds, seed=args.seed)
    else:
        run(
            rounds=args.rounds,
            seed=args.seed,
            enable_dynamic_ops=not args.no_dynamic,
            consistency=resolved_consistency,
            randomize_consistency=args.random_consistency,
        )
