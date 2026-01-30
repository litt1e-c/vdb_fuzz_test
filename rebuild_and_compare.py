import argparse
import json
from typing import List, Dict, Any, Optional
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

HOST = "127.0.0.1"
PORT = "19531"
BASE_COLLECTION = "full_data_from_json"
MIN_COLLECTION = "min_from_json"
JSON_PATH = "milvus_full_data.json"
DEFAULT_INDEX = "AUTOINDEX"  # AUTOINDEX | FLAT | IVF_FLAT | IVF_SQ8 | IVF_PQ | HNSW

# ----------------------------------------------------------------------
# Constant row used if ID=83 is not found in loaded JSON
# ----------------------------------------------------------------------

TARGET_ROW = {
    'id': 83,
    'vector': [0.1] * 128,
    'c0': 2874.711365180692, 'c1': False, 'c2': None, 'c3': 1447.2535675523943,
    'c4': 3679.853387658721, 'c5': None, 'c6': -14186, 'c7': False,
    'c8': 'pORFBdCTbX6SBPmfXIS2fxj1UcScXHIsXC', 'c9': 3013.0714161832525,
    'c10': None, 'c11': -39617, 'c12': True, 'c13': -64474, 'c14': None,
    'c15': 2163.7449525940506, 'c16': 4025.8571008038884, 'c17': True,
    'meta_json': {'price': 116, 'color': 'Green', 'active': False, 'config': {'version': 7}, 'history': [39, 36, 60], 'user': {'k_9': 18840, 'k_10': {'test.key': 'QB1', 'k_9': '3', 'k_16': 'eF3cUV'}}, 'data': False, 'test.key': {'k_3': {'k_19': {'a b': ['bc', False, 'S9'], 'data': {'k_10': 69.67486626096031, 'test.key': 301, 'k_19': 980.2751004435628, 'k_14': 359.8521951479872, 'k_1': 304.6355619956549}}, 'k_2': {'k_15': {'k_5': 132.88691894945293, 'k_9': True}, 'k_17': {'log': 412.9507285077123, 'a b': False, 'k_12': True, 'k_4': ''}, 'k_12': {'k_4': None, 'k_6': True}, 'k_9': {'k_6': '1nkyR'}, 'k_18': False}}, 'k_4': {'k_15': {'k_19': None, 'k_8': None, 'k_14': {'k_16': '4FGEnKc', 'k_5': 'oQ17pJj'}, 'a b': {'user': None}, 'k_18': [False]}}}, 'k_9': None},
    'tags_array': [50, 53, 66]
}

EXPR = """
((tags_array is not null and ((((((meta_json["price"] > 204 and meta_json["price"] < 336) or (((c9 < -99996.796875 and meta_json["history"][0] > 29) and c11 == 47963) or ((c17 == false and meta_json["history"][0] > 33) and (c1 == true or c7 is null)))) and (c10 > "xmWs" or c10 != "Wxrr1")) and (meta_json["config"]["version"] == 1 and (((c1 is not null and (c8 like "7%" or meta_json["config"]["version"] == 5)) and (((meta_json["active"] == true and meta_json["color"] == "Red") and meta_json["history"][0] > 67) or (c2 == -17409 or c12 == false))) and ((meta_json["history"][0] > 63 and (c12 == false and meta_json["config"]["version"] == 4)) or (c4 <= 569.1141562900084 or (c3 < 1665.5894309423982 or c1 == false)))))) and ((((((c0 <= 2338.556268822096 and c6 < 18027) or ((meta_json["active"] == true and meta_json["color"] == "Red") or c9 >= -99996.796875)) and (meta_json["config"]["version"] < 6 or (c9 > 4551.084580109659 or c16 >= 4630.872072108272))) and (((c17 == false and c14 > -99999.984375) and (c12 == false and null is not null)) and (((meta_json["active"] == true and meta_json["color"] == "Red") and c8 != "6HTLwWvQkMahwNB") and (c15 is not null or meta_json["history"][0] > 79)))) and ((((c0 > 105381.9375 or c16 < 3787.6236053480816) and ((meta_json["active"] == true and meta_json["color"] == "Red") and meta_json["config"]["version"] == 9)) and (((meta_json["price"] > 272 and meta_json["price"] < 463) or (meta_json["active"] == true and meta_json["color"] == "Red")) and (c3 < 1216.1410473351923 or c17 == false))) or (meta_json["history"][0] > 28 or ((c3 >= 3107.3980916877663 and c17 == true) or tags_array is not null)))) and ((((((meta_json["active"] == true and meta_json["color"] == "Blue") and c9 <= 105382.9375) and (c11 >= 14233 or (meta_json["price"] > 307 and meta_json["price"] < 425))) or ((c8 like "Y%" or c12 == false) and (c12 == true and c11 >= -62503))) or (c9 < -99996.796875 and ((c0 > 5106.319216140297 and (meta_json["price"] > 378 and meta_json["price"] < 444)) or (c10 is not null and c17 == false)))) and c15 > 1284.0313107498841))) or ((meta_json["history"][0] > 36 and (((((c17 == true and c3 > 811.7130218876304) or (c16 < 2885.352818944058 or c2 <= 39657)) and (((meta_json["active"] == true and meta_json["color"] == "Blue") and c11 <= 180200) or ((meta_json["active"] == true and meta_json["color"] == "Red") and c9 >= 2708.415597613756))) and ((c5 == false and (c0 < 1244.7152217254431 or c11 <= -30538)) and ((c16 is not null or meta_json["history"][0] > 65) and (meta_json["config"]["version"] == 8 and c4 is null)))) and (c9 < 5050.670390985309 or ((((meta_json["price"] > 118 and meta_json["price"] < 279) and c2 < 35889) and (c11 <= 34733 and c2 != 16987)) and (c9 < 3210.6042287060172 or (c15 > 248.83421203997813 or c6 < -70676)))))) and ((((((c12 == true and c1 == true) and (c2 >= 54648 and c15 is not null)) and ((meta_json["config"]["version"] == 9 and c7 == true) or meta_json["config"]["version"] == 7)) and ((meta_json["config"]["version"] == 8 and (c4 > 2723.622008895618 or c12 == false)) and ((meta_json["history"][0] > 21 or c5 == false) or (meta_json["price"] >= 24 or c7 == true)))) or (c12 == true or (((meta_json["history"][0] > 58 or (meta_json["active"] == true and meta_json["color"] == "Red")) or meta_json["config"]["version"] == 1) and (meta_json["history"][0] > 32 or (c7 is null or meta_json["config"]["version"] == 9))))) or ((((c7 == true or (c6 > -11206 or null is not null)) and ((c4 is null and meta_json["config"]["version"] == 5) or c15 >= 1492.30450968316)) or meta_json["config"]["version"] == 4) or ((((c13 != 55845 and (meta_json["price"] > 290 and meta_json["price"] < 404)) or ((meta_json["active"] == true and meta_json["color"] == "Red") or (meta_json["price"] > 352 and meta_json["price"] < 480))) or ((c4 < 1042.2365004319415 or meta_json["config"]["version"] == 6) and (c17 is not null and c6 == 180150))) or (((c6 <= 6431 and null is not null) or null is null) and (((meta_json["price"] > 174 and meta_json["price"] < 288) and meta_json["active"] != false) and (c7 is null and c12 == true))))))))) or (((((((c10 is null and ((c6 == -79895 or c11 < 17025) or (c12 == true or meta_json["_non_exist_key"] < "PdhLl3V0zh"))) and (((meta_json["config"]["version"] == 7 or (meta_json["active"] == true and meta_json["color"] == "Blue")) or c6 is not null) or (c13 > 11389 and (meta_json["active"] == true and meta_json["color"] == "Red")))) and ((c17 == true and ((c5 == true or c16 is not null) and ((meta_json["active"] == true and meta_json["color"] == "Blue") and c0 <= 966.1100468054391))) and (((meta_json["config"]["version"] == 2 or c3 < 105383.3671875) or c7 == true) or ((c15 < 2194.5656510584968 or (meta_json["price"] > 479 and meta_json["price"] < 576)) and (meta_json["active"] == true and meta_json["color"] == "Blue"))))) or (((c3 >= 227.36232723846618 or ((c0 > 105381.9375 and meta_json["config"]["version"] == 9) and ((meta_json["active"] == true and meta_json["color"] == "Blue") and c6 is not null))) or (((c11 is null or meta_json["config"]["version"] == 5) or (c2 is null and c3 > 2526.467618787217)) or c7 == false)) and ((((c17 == true or c16 > -99999.9453125) or (meta_json["active"] != false or c8 is null)) or ((meta_json["active"] == true and meta_json["color"] == "Red") or (c3 < 110.94107653368546 and c2 == -51867))) or (((c1 == true or c14 <= 105382.3359375) or (c3 <= 2264.8937709420393 or c6 is not null)) or c9 is null)))) and meta_json["history"][0] > 51) or (((((((c13 is null or meta_json["config"]["version"] == 6) or (meta_json["history"][0] > 22 and c6 != 62392)) or ((meta_json["history"][0] > 59 and c15 < 1241.4003565530531) or (c3 <= 2600.706371546737 or c13 < 45088))) or (c0 < 4932.317201216683 and ((c9 <= 3433.645912140729 or c6 is not null) and (c2 is null or c16 <= 4061.710531104789)))) and (((((meta_json["price"] > 268 and meta_json["price"] < 408) and c9 <= 2742.260003390715) and (c3 is null and c7 == true)) or c7 == false) and ((c7 == true or (c7 == false and meta_json["history"][0] > 39)) or (c14 >= 902.9233764499173 or (c4 >= 2036.3710884122213 or c0 <= 105381.9375))))) or (((((meta_json["price"] > 383 and meta_json["price"] < 581) or (meta_json["history"][0] > 39 or c5 == false)) or ((c1 == false and c1 == true) or (c12 == false or c13 > -75095))) and (meta_json["active"] == true and meta_json["color"] == "Red")) and (c10 < "gMaX8K" or (((meta_json["config"]["version"] == 6 or meta_json is null) and (meta_json["config"]["version"] == 3 or c17 == true)) and ((c9 > 1956.9423181354957 or meta_json["history"][0] > 80) and (c7 is not null or (meta_json["price"] > 461 and meta_json["price"] < 555))))))) and (((((((meta_json["price"] > 476 and meta_json["price"] < 570) or meta_json["config"]["version"] == 6) and (c11 >= -13014 and (meta_json["price"] > 109 and meta_json["price"] < 248))) and ((c1 == false or c15 > 2947.97111300559) or tags_array is null)) or (((c11 <= -180073 or c4 >= 105376.953125) or (meta_json["config"]["version"] == 5 or meta_json["history"][0] > 66)) and ((c4 < 5360.62088697727 and c6 == -52148) or c3 is null))) or ((((c5 == true and (meta_json["active"] == true and meta_json["color"] == "Blue")) or ((meta_json["active"] == true and meta_json["color"] == "Blue") and (meta_json["active"] == true and meta_json["color"] == "Blue"))) or (((meta_json["price"] > 453 and meta_json["price"] < 612) and (meta_json["active"] == true and meta_json["color"] == "Blue")) and (c16 >= 2040.5062559106018 and c1 == true))) and c14 < 1624.9775499311854)) or (((meta_json["active"] == true and meta_json["color"] == "Blue") and (((null is not null or meta_json["config"]["version"] == 9) and c5 == true) or ((c13 == -52702 or c2 is not null) or (c8 != "#ftM5&E96@jwO&&bU#YejFeP%YA0q" or c13 < -10041)))) or ((((meta_json["active"] == true and meta_json["color"] == "Red") or (c15 >= 5056.03351006638 and meta_json["color"] != "Green")) or ((c10 != "" and c10 != "dwXq1b") and c5 is null)) and (c2 > -44823 or ((meta_json["history"][0] > 20 and c15 < 4732.944103361606) or (c16 > 2468.7947720395873 or c12 == true)))))))) and ((c6 < -744 or meta_json["history"][0] > 41) and meta_json is not null)))
"""


# ----------------------------------------------------------------------
# Build collection schema
# ----------------------------------------------------------------------

def get_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="c0", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c1", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c2", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c3", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c4", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c5", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c6", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c7", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c8", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="c9", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c10", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="c11", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c12", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c13", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c14", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c15", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c16", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c17", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="meta_json", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="tags_array", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=50, nullable=True),
    ]
    return CollectionSchema(fields, enable_dynamic_field=True)


# ----------------------------------------------------------------------
# Convert row-wise JSON into column-wise insertion format
# ----------------------------------------------------------------------

def rows_to_columns(schema: CollectionSchema, rows: List[Dict[str, Any]]):
    columns = []
    for f in schema.fields:
        name = f.name
        column = []
        for row in rows:
            column.append(row.get(name))
        columns.append(column)
    return columns


# ----------------------------------------------------------------------
# Normalize row values to ensure type consistency before comparison
# ----------------------------------------------------------------------

def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    bool_fields = {"c1", "c5", "c7", "c12", "c17"}
    int_fields = {"id", "c2", "c6", "c11", "c13"}
    float_fields = {"c0", "c3", "c4", "c9", "c14", "c15", "c16"}
    str_fields = {"c8", "c10"}
    def to_bool(val):
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(int(val))
        if isinstance(val, str):
            lv = val.lower()
            if lv in {"true", "1", "yes", "y", "t"}:
                return True
            if lv in {"false", "0", "no", "n", "f", ""}:
                return False
        return None

    def to_int(val):
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def to_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    for name in bool_fields:
        if name in out:
            out[name] = to_bool(out[name])
    for name in int_fields:
        if name in out:
            out[name] = to_int(out[name])
    for name in float_fields:
        if name in out:
            out[name] = to_float(out[name])
    for name in str_fields:
        if name in out and out[name] is not None and not isinstance(out[name], str):
            out[name] = str(out[name])
    if "tags_array" in out and out["tags_array"] is not None:
        out["tags_array"] = [to_int(v) for v in out["tags_array"]]
    if "vector" in out and out["vector"] is not None:
        out["vector"] = [to_float(v) for v in out["vector"]]
    return out


# ----------------------------------------------------------------------
# Create vector index based on selected index type
# ----------------------------------------------------------------------

def create_vector_index(col: Collection, name: str, index_type: str, index_params: Optional[Dict[str, Any]] = None):
    itype = index_type.upper()
    params = index_params.copy() if index_params else {}
    try:
        if itype == "AUTOINDEX":
            col.create_index("vector", {"index_type": "AUTOINDEX", "metric_type": "L2"})
            print(f" Created AUTOINDEX on {name}.")
        elif itype == "FLAT":
            try:
                col.create_index("vector", {"index_type": "FLAT", "metric_type": "L2"})
                print(f" Created FLAT index on {name}.")
            except Exception as e:
                print(f" FLAT not supported, fallback to IVF_FLAT(nlist=1): {e}")
                base_params = {"nlist": 1}
                base_params.update(params)
                col.create_index("vector", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": base_params})
                print(f" Created IVF_FLAT on {name} with params={base_params}.")
        elif itype == "IVF_FLAT":
            base_params = {"nlist": 64}
            base_params.update(params)
            col.create_index("vector", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": base_params})
            print(f" Created IVF_FLAT on {name} with params={base_params}.")
        elif itype == "IVF_SQ8":
            base_params = {"nlist": 64}
            base_params.update(params)
            col.create_index("vector", {"index_type": "IVF_SQ8", "metric_type": "L2", "params": base_params})
            print(f" Created IVF_SQ8 on {name} with params={base_params}.")
        elif itype == "IVF_PQ":
            base_params = {"nlist": 64, "m": 16}
            base_params.update(params)
            col.create_index("vector", {"index_type": "IVF_PQ", "metric_type": "L2", "params": base_params})
            print(f" Created IVF_PQ on {name} with params={base_params}.")
        elif itype == "HNSW":
            base_params = {"M": 16, "efConstruction": 200}
            base_params.update(params)
            col.create_index("vector", {"index_type": "HNSW", "metric_type": "L2", "params": base_params})
            print(f" Created HNSW on {name} with params={base_params}.")
        else:
            col.create_index("vector", {"index_type": "AUTOINDEX", "metric_type": "L2"})
            print(f" Unknown index '{index_type}', default AUTOINDEX applied on {name}.")
    except Exception as ie:
        print(f" Index creation failed on {name}: {ie}")


# ----------------------------------------------------------------------
# Drop collection if exists, recreate, insert data, build index
# ----------------------------------------------------------------------

def drop_create_and_insert(name: str, schema: CollectionSchema, rows: List[Dict[str, Any]], index_type: str = DEFAULT_INDEX, index_params: Optional[Dict[str, Any]] = None):
    if utility.has_collection(name):
        print(f" Dropping existing collection {name}...")
        utility.drop_collection(name)
    col = Collection(name=name, schema=schema)
    if not rows:
        print(f" No rows to insert for {name}.")
        return col
    normalized = [normalize_row(r) for r in rows]
    data = rows_to_columns(schema, normalized)
    col.insert(data)
    col.flush()
    create_vector_index(col, name, index_type, index_params=index_params)
    return col


# ----------------------------------------------------------------------
# Execute query and return matched IDs
# ----------------------------------------------------------------------

def run_query(name: str, expr: str, limit: int):
    col = Collection(name)
    col.load()
    capped_limit = max(1, min(limit, 16384))
    res = col.query(expr, output_fields=["id"], limit=capped_limit)
    ids = [r["id"] for r in res]
    return ids


def project_row(row: Dict[str, Any], schema: CollectionSchema) -> Dict[str, Any]:
    names = [f.name for f in schema.fields]
    return {k: row.get(k) for k in names}



# ----------------------------------------------------------------------
# Fetch a single row by ID for verification
# ----------------------------------------------------------------------

def fetch_row_by_id(name: str, id_value: int, schema: CollectionSchema) -> Optional[Dict[str, Any]]:
    col = Collection(name)
    col.load()
    res = col.query(expr=f"id == {id_value}", output_fields=["*"], limit=1)
    if not res:
        return None
    return project_row(res[0], schema)


def load_json_rows(path: str, limit: int = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def parse_index_params(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        print(f" Failed to parse --index-params, ignoring: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Rebuild collections from JSON and compare query results.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--json", dest="json_path", default=JSON_PATH)
    parser.add_argument("--base", dest="base_name", default=BASE_COLLECTION)
    parser.add_argument("--min", dest="min_name", default=MIN_COLLECTION)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit when loading JSON")
    parser.add_argument("--query-limit", type=int, default=20000, help="Limit for query results")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Vector index type: AUTOINDEX | FLAT | IVF_FLAT | IVF_SQ8 | IVF_PQ | HNSW")
    parser.add_argument("--index-params", default=None, help='JSON string for index params, e.g. "{\"nlist\":128}" or "{\"M\":16,\"efConstruction\":200}"')
    parser.add_argument("--min-source", default="base", help="Source for minimal row: base | constant")
    args = parser.parse_args()

    print(f" Connecting to {args.host}:{args.port}...")
    connections.connect("default", host=args.host, port=args.port)

    print(f" Loading JSON from {args.json_path}...")
    base_rows = load_json_rows(args.json_path, args.limit)
    print(f" Loaded {len(base_rows)} rows from JSON.")

    schema = get_schema()

    print(f" Rebuilding base collection {args.base_name}...")
    idx_params = parse_index_params(args.index_params)
    drop_create_and_insert(args.base_name, schema, base_rows, index_type=args.index, index_params=idx_params)

    print(f" Rebuilding minimal collection {args.min_name} (1 row)...")
    min_row = None
    if args.min_source.lower() == "base":
        # Try to locate id=83 from loaded JSON rows
        for r in base_rows:
            if r.get("id") == 83:
                min_row = r
                break
    if min_row is None:
        print(" Using constant TARGET_ROW for minimal collection (base row not found or min-source!=base).")
        min_row = TARGET_ROW
    drop_create_and_insert(args.min_name, schema, [min_row], index_type=args.index, index_params=idx_params)

    query_limit = args.query_limit if args.query_limit is not None else len(base_rows)

    print(f" Running query on {args.base_name}...")
    base_ids = run_query(args.base_name, EXPR, query_limit)
    print(f" Base collection hits: {len(base_ids)}")

    print(f" Running query on {args.min_name}...")
    min_ids = run_query(args.min_name, EXPR, query_limit)
    print(f" Minimal collection hits: {len(min_ids)}")

    base_set = set(base_ids)
    min_set = set(min_ids)

    only_in_base = sorted(base_set - min_set)
    only_in_min = sorted(min_set - base_set)

    print("\n" + "=" * 60)
    print(f"Only in base ({len(only_in_base)}): {only_in_base[:20]}")
    if len(only_in_base) > 20:
        print(f"... truncated, total {len(only_in_base)}")
    print(f"Only in minimal ({len(only_in_min)}): {only_in_min}")
    print("=" * 60)

    if 83 in base_set and 83 not in min_set:
        print("!!! Reproduced discrepancy: ID 83 present in base, absent in minimal.Use the same expression to query")
    elif 83 in min_set and 83 not in base_set:
        print(" Unexpected: ID 83 only in minimal.")
    else:
        print(" ID 83 consistency: present in both or absent in both.")

    print("\n Verifying ID 83 presence and equality...")
    base_row_83 = fetch_row_by_id(args.base_name, 83, schema)
    min_row_83 = fetch_row_by_id(args.min_name, 83, schema)
    print(f"Base has 83: {'✅' if base_row_83 is not None else '❌'}; Minimal has 83: {'✅' if min_row_83 is not None else '❌'}")
    if base_row_83 and min_row_83:
        n_base = normalize_row(base_row_83)
        n_min = normalize_row(min_row_83)
        if n_base == n_min:
            print("🟢 Rows equal after normalization.")
        else:
            # Show concise diff
            diffs = []
            for k in n_base.keys():
                if n_base.get(k) != n_min.get(k):
                    diffs.append((k, n_base.get(k), n_min.get(k)))
            print(f"⚠️ Rows differ in {len(diffs)} fields (show up to 10):")
            for k, a, b in diffs[:10]:
                print(f" - {k}: base={a} | min={b}")


if __name__ == "__main__":
    main()
