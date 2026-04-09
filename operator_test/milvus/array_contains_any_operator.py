import os
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
COLLECTION_NAME = "array_contains_any_operator_validation"


def query_ids(collection, expr):
    rows = collection.query(
        expr,
        output_fields=["id"],
        consistency_level="Strong",
        timeout=10,
    )
    return sorted(row["id"] for row in rows)


def build_rows():
    return [
        {
            "id": 1,
            "int_arr": [1, 2, 3],
            "bool_arr": [True, False],
            "str_arr": ["red", "blue"],
            "vec": [0.0, 0.0],
        },
        {
            "id": 2,
            "int_arr": [2],
            "bool_arr": [False],
            "str_arr": ["blue"],
            "vec": [1.0, 1.0],
        },
        {
            "id": 3,
            "int_arr": [],
            "bool_arr": [],
            "str_arr": [],
            "vec": [2.0, 2.0],
        },
        {
            "id": 4,
            "int_arr": None,
            "bool_arr": None,
            "str_arr": None,
            "vec": [3.0, 3.0],
        },
        {
            "id": 5,
            "vec": [4.0, 4.0],
        },
        {
            "id": 6,
            "int_arr": [1, 2],
            "bool_arr": [True],
            "str_arr": ["red", "green"],
            "vec": [5.0, 5.0],
        },
    ]


def main():
    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as exc:
        print(f"connection_failed: {exc}")
        return

    try:
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
            FieldSchema(
                name="bool_arr",
                dtype=DataType.ARRAY,
                element_type=DataType.BOOL,
                max_capacity=8,
                nullable=True,
            ),
            FieldSchema(
                name="str_arr",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=8,
                max_length=32,
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
            ("int_contains_any_hit", "array_contains_any(int_arr, [1, 9])", [1, 6]),
            ("int_contains_any_hit_2", "array_contains_any(int_arr, [2, 9])", [1, 2, 6]),
            ("bool_contains_any_true", "array_contains_any(bool_arr, [true])", [1, 6]),
            ('str_contains_any', 'array_contains_any(str_arr, ["red", "green"])', [1, 6]),
            ("raw_not_contains_any", "not (array_contains_any(int_arr, [1, 9]))", [2, 3]),
        ]

        print("--- ARRAY_CONTAINS_ANY operator validation ---")
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

    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)
        connections.disconnect("default")


if __name__ == "__main__":
    main()
