import os
import random
import time

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
PORT = os.getenv("MILVUS_PORT", "19531")
COLLECTION_NAME = "array_length_operator_validation"
COMPACTION_COLLECTION_NAME = "array_length_compaction_validation"


def query_ids(collection, expr, ids=None):
    base_expr = expr
    if ids is not None:
        base_expr = f"id in {ids} and ({expr})"
    rows = collection.query(
        base_expr,
        output_fields=["id"],
        consistency_level="Strong",
        timeout=20,
    )
    return sorted(row["id"] for row in rows)


def build_rows():
    return [
        {"id": 1, "int_arr": [1, 2, 3], "vec": [0.0, 0.0]},
        {"id": 2, "int_arr": [2], "vec": [1.0, 1.0]},
        {"id": 3, "int_arr": [], "vec": [2.0, 2.0]},
        {"id": 4, "int_arr": None, "vec": [3.0, 3.0]},
        {"id": 5, "vec": [4.0, 4.0]},
        {"id": 6, "int_arr": [1, 2], "vec": [5.0, 5.0]},
    ]


def build_compaction_rows():
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
            "vec": [0.1, 0.2],
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
            "vec": [0.2, 0.1],
        },
    ]

    for rid in range(10003, 10003 + 1200 - 2):
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
                "arr_int32_0": random.choice([[], [-18], [19], [91, -98], None]),
                "vec": [random.random(), random.random()],
            }
        )
    return rows


def run_basic_validation():
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME, timeout=10)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(
            name="int_arr",
            dtype=DataType.ARRAY,
            element_type=DataType.INT32,
            max_capacity=8,
            nullable=True,
        ),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    col = Collection(COLLECTION_NAME, CollectionSchema(fields))
    col.insert(build_rows())
    col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=10)
    col.load(timeout=10)
    time.sleep(1)

    tests = [
        ("length_eq_0", "array_length(int_arr) == 0", [3]),
        ("length_lt_1", "array_length(int_arr) < 1", [3]),
        ("length_eq_1", "array_length(int_arr) == 1", [2]),
        ("length_ge_2", "array_length(int_arr) >= 2", [1, 6]),
        ("length_le_2", "array_length(int_arr) <= 2", [2, 3, 6]),
        ("raw_not_length_lt_1", "not (array_length(int_arr) < 1)", [1, 2, 6]),
    ]

    print("--- ARRAY_LENGTH operator validation ---")
    for name, expr, expected_ids in tests:
        try:
            actual_ids = query_ids(col, expr)
            status = "PASS" if actual_ids == expected_ids else "FAIL"
            print(
                f"{name}: {status} | expr={expr} | "
                f"expected={expected_ids} | actual={actual_ids}"
            )
        except Exception as exc:
            print(
                f"{name}: ERROR | expr={expr} | "
                f"error={type(exc).__name__}: {exc}"
            )

    utility.drop_collection(COLLECTION_NAME, timeout=10)


def run_compaction_probe():
    target_ids = [4080, 4327]
    delete_ids = [10003, 10004, 10005, 10006, 10007]
    reduced_expr = (
        '(((meta_json is not null and ((meta_json["price"] > 164 and meta_json["price"] < 324))) '
        'and ((c3 is not null and (c3 == 57818)) or '
        '(arr_int32_0 is not null and (array_length(arr_int32_0) <= 2)))) '
        'and not ((c1 is null or (c3 is not null and (c3 < 13935)))))'
    )

    if utility.has_collection(COMPACTION_COLLECTION_NAME):
        utility.drop_collection(COMPACTION_COLLECTION_NAME, timeout=10)

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
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    col = Collection(COMPACTION_COLLECTION_NAME, CollectionSchema(fields))
    col.insert(build_compaction_rows())
    col.create_index("vec", {"metric_type": "COSINE", "index_type": "IVF_SQ8", "params": {"nlist": 32}}, timeout=60)
    col.create_index("arr_int32_0", {"index_type": "BITMAP"}, timeout=60)
    col.load(timeout=60)

    before_ids = query_ids(col, reduced_expr, ids=target_ids)
    col.delete(f"id in {delete_ids}", timeout=30)
    col.flush(timeout=30)
    col.compact(timeout=60)
    col.wait_for_compaction_completed(timeout=60)
    col.release()
    col.load(timeout=60)

    null_ids = query_ids(col, "arr_int32_0 is null", ids=target_ids)
    not_null_ids = query_ids(col, "arr_int32_0 is not null", ids=target_ids)
    len_zero_ids = query_ids(col, "array_length(arr_int32_0) == 0", ids=target_ids)
    after_ids = query_ids(col, reduced_expr, ids=target_ids)

    bug_reproduced = (
        before_ids == target_ids
        and null_ids == target_ids
        and not_null_ids == []
        and len_zero_ids == target_ids
        and after_ids == []
    )

    status = "BUG_REPRODUCED" if bug_reproduced else "NO_REPRO"
    print(
        f"compaction_empty_array_probe: {status} | "
        f"before={before_ids} | is_null={null_ids} | "
        f"is_not_null={not_null_ids} | len_zero={len_zero_ids} | after={after_ids}"
    )

    utility.drop_collection(COMPACTION_COLLECTION_NAME, timeout=10)


def main():
    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as exc:
        print(f"connection_failed: {exc}")
        return

    try:
        run_basic_validation()
        run_compaction_probe()
    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)
        if utility.has_collection(COMPACTION_COLLECTION_NAME):
            utility.drop_collection(COMPACTION_COLLECTION_NAME, timeout=10)
        connections.disconnect("default")


if __name__ == "__main__":
    main()
