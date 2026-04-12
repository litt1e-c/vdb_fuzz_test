import os
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
PORT = os.getenv("MILVUS_PORT", "19532")
NAME = "repro_json_array_null_semantics"

connections.connect(host=HOST, port=PORT)

if utility.has_collection(NAME):
    utility.drop_collection(NAME)

schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(
            name="arr_int",
            dtype=DataType.ARRAY,
            element_type=DataType.INT64,
            max_capacity=8,
            nullable=True,
        ),
        FieldSchema(
            name="meta",
            dtype=DataType.JSON,
            nullable=True,
        ),
    ],
    description="test JSON/ARRAY null semantics"
)

col = Collection(name=NAME, schema=schema)

# 最基础索引，避免后续 query/search 受别的问题影响
col.create_index("vec", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
col.load()

def try_insert(tag, rows):
    print(f"\n=== INSERT TEST: {tag} ===")
    try:
        mr = col.insert(rows)
        print("insert ok, insert_count =", mr.insert_count)
        col.flush()
    except Exception as e:
        print("insert failed:")
        print(repr(e))

# 1) ARRAY 整列为 null：按文档，nullable=True 时应允许
try_insert(
    "array field = None",
    [
        {"id": 1, "vec": [0.1, 0.2], "arr_int": None, "meta": {"a": 1}},
        {"id": 2, "vec": [0.2, 0.3], "meta": {"a": 2}},  # arr_int omitted -> also null
    ],
)

# 2) ARRAY 某个元素为 null：按文档，不应支持
try_insert(
    "array element contains None",
    [
        {"id": 3, "vec": [0.3, 0.4], "arr_int": [1, None, 3], "meta": {"a": 3}},
    ],
)

# 3) JSON 内部 key = null：按文档，这是允许的，且 meta 本身不算 null
try_insert(
    "json inner key = None",
    [
        {"id": 4, "vec": [0.4, 0.5], "arr_int": [4, 5], "meta": {"a": None, "b": 99}},
        {"id": 5, "vec": [0.5, 0.6], "arr_int": [7, 8], "meta": None},  # 整个 JSON 列 null
        {"id": 6, "vec": [0.6, 0.7], "arr_int": [9], },                  # meta omitted -> null
    ],
)

print("\n=== QUERY TESTS ===")

tests = [
    'arr_int IS NULL',
    'arr_int IS NOT NULL',
    'meta IS NULL',
    'meta IS NOT NULL',
    # 你 issue 里的关键表达式
    'arr_int[0] IS NULL',
    'arr_int[0] IS NOT NULL',
]

for expr in tests:
    print(f"\nexpr: {expr}")
    try:
        rows = col.query(expr=expr, output_fields=["id", "arr_int", "meta"])
        print("rows =", rows)
    except Exception as e:
        print("query failed:")
        print(repr(e))

# 4) JSON 内部 key 为 null 时，观察整列语义
print("\n=== JSON FIELD CONTENT CHECK ===")
try:
    rows = col.query(expr='id in [4,5,6]', output_fields=["id", "meta"])
    print(rows)
except Exception as e:
    print(repr(e))