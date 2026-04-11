import os
import random

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "empty_array_compaction_repro"
ROW_COUNT = 1200
DELETE_IDS = [10003, 10004, 10005, 10006, 10007]

TARGET_IDS = [4080, 4327]
REDUCED_EXPR = (
    '(((meta_json is not null and ((meta_json["price"] > 164 and meta_json["price"] < 324))) '
    'and ((c3 is not null and (c3 == 57818)) or '
    '(arr_int32_0 is not null and (array_length(arr_int32_0) <= 2)))) '
    'and not ((c1 is null or (c3 is not null and (c3 < 13935)))))'
)

CHECKS = [
    "arr_int32_0 is null",
    "arr_int32_0 is not null",
    "array_length(arr_int32_0) == 0",
    REDUCED_EXPR,
]


def make_schema():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="c0", dtype=DataType.INT16, nullable=True),
        FieldSchema(name="c1", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c2", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c3", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c4", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="meta_json", dtype=DataType.JSON, nullable=True),
        FieldSchema(
            name="arr_int32_0",
            dtype=DataType.ARRAY,
            element_type=DataType.INT32,
            max_capacity=8,
            nullable=True,
        ),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=4),
    ]
    return CollectionSchema(fields)


def build_rows():
    random.seed(0)
    rows = [
        {
            "id": 4080,
            "c0": 28329,
            "c1": -17079,
            "c2": -2.2250738585072014e-308,
            "c3": 43879,
            "c4": False,
            "meta_json": {
                "price": 195,
                "color": "Blue",
                "active": True,
                "config": {"version": 6},
                "history": [24, 0, 7],
            },
            "arr_int32_0": [],
            "vector": [0.1, 0.2, 0.3, 0.4],
        },
        {
            "id": 4327,
            "c0": 5404,
            "c1": 57780,
            "c2": 112.3542539655108,
            "c3": 49036,
            "c4": False,
            "meta_json": {
                "price": 167,
                "color": "Blue",
                "active": True,
                "history": [59, 56, 71],
            },
            "arr_int32_0": [],
            "vector": [0.4, 0.3, 0.2, 0.1],
        },
    ]

    base_id = 10003
    for rid in range(base_id, base_id + ROW_COUNT - 2):
        arr_val = random.choice([[], [-18], [19], [91, -98], None])
        rows.append(
            {
                "id": rid,
                "c0": random.randint(-30000, 30000),
                "c1": random.randint(-100000, 100000),
                "c2": random.uniform(-1000, 1000),
                "c3": random.randint(-60000, 60000),
                "c4": random.choice([True, False, None]),
                "meta_json": {
                    "price": random.randint(0, 600),
                    "color": random.choice(["Blue", "Red", "Green"]),
                    "active": random.choice([True, False]),
                    "config": {"version": random.randint(1, 9)},
                    "history": [random.randint(0, 99) for _ in range(3)],
                },
                "arr_int32_0": arr_val,
                "vector": [random.random() for _ in range(4)],
            }
        )
    return rows


def query_ids(col, expr):
    rows = col.query(
        f"id in {TARGET_IDS} and ({expr})",
        output_fields=["id"],
        consistency_level="Strong",
        timeout=20,
    )
    return sorted(row["id"] for row in rows)


def show_stage(col, stage):
    print(f"\n[{stage}]")
    for expr in CHECKS:
        print(f"{expr} -> {query_ids(col, expr)}")


def main():
    print(f"Connecting to Milvus at {HOST}:{PORT}")
    connections.connect(host=HOST, port=PORT, timeout=10)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    col = Collection(COLLECTION_NAME, make_schema())
    rows = build_rows()
    print(f"Inserting {len(rows)} rows...")
    col.insert(rows)

    print("Building vector and array indexes...")
    col.create_index(
        "vector",
        {"index_type": "IVF_SQ8", "metric_type": "COSINE", "params": {"nlist": 32}},
        index_name="vector_idx",
        timeout=60,
    )
    col.create_index("arr_int32_0", {"index_type": "BITMAP"}, timeout=60)
    col.load(timeout=60)

    show_stage(col, "BEFORE_DELETE")

    print(f"\nDeleting unrelated rows: {DELETE_IDS}")
    col.delete(f"id in {DELETE_IDS}", timeout=30)
    col.flush(timeout=30)
    show_stage(col, "AFTER_DELETE")

    print("\nRunning compaction...")
    col.compact(timeout=60)
    col.wait_for_compaction_completed(timeout=60)
    col.release()
    col.load(timeout=60)
    show_stage(col, "AFTER_COMPACT")

    is_null_ids = query_ids(col, "arr_int32_0 is null")
    is_not_null_ids = query_ids(col, "arr_int32_0 is not null")
    len_zero_ids = query_ids(col, "array_length(arr_int32_0) == 0")
    reduced_ids = query_ids(col, REDUCED_EXPR)

    bug_reproduced = (
        is_null_ids == TARGET_IDS
        and is_not_null_ids == []
        and len_zero_ids == TARGET_IDS
        and reduced_ids == []
    )

    print("\n[RESULT]")
    if bug_reproduced:
        print("BUG_REPRODUCED: empty arrays become IS NULL after delete+compact, while array_length(...) == 0 stays true.")
    else:
        print("NO_REPRO: the expected issue state did not trigger in this run.")

    utility.drop_collection(COLLECTION_NAME)
    connections.disconnect("default")


if __name__ == "__main__":
    main()
