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

from operator_case_validator import run_operator_cases


HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
PORT = os.getenv("MILVUS_PORT", "19532")
COLLECTION_NAME = "array_subscript_access_validation"


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
            "arr_int": [5, 1],
            "arr_str": ["admin", "root"],
            "arr_bool": [True, False],
            "vec": [0.0, 0.0],
        },
        {
            "id": 2,
            "arr_int": [2],
            "arr_str": ["user"],
            "arr_bool": [False],
            "vec": [1.0, 1.0],
        },
        {
            "id": 3,
            "arr_int": [-1, 4],
            "arr_str": ["guest", "ops"],
            "arr_bool": [True],
            "vec": [2.0, 2.0],
        },
        {
            "id": 4,
            "arr_int": [],
            "arr_str": [],
            "arr_bool": [],
            "vec": [3.0, 3.0],
        },
        {
            "id": 5,
            "arr_int": None,
            "arr_str": None,
            "arr_bool": None,
            "vec": [4.0, 4.0],
        },
    ]


def main():
    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as exc:
        print(f"connection_failed: {exc}")
        return 2

    try:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(
                name="arr_int",
                dtype=DataType.ARRAY,
                element_type=DataType.INT64,
                max_capacity=4,
                nullable=True,
            ),
            FieldSchema(
                name="arr_str",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=4,
                max_length=32,
                nullable=True,
            ),
            FieldSchema(
                name="arr_bool",
                dtype=DataType.ARRAY,
                element_type=DataType.BOOL,
                max_capacity=4,
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
            ("normal", "int_index_gt", "arr_int[0] > 3", [1]),
            ("normal", "int_second_index_eq", "arr_int[1] == 4", [3]),
            ("normal", "int_out_of_range_filtered", "arr_int[2] > 0", []),
            ("normal", "str_index_eq", 'arr_str[0] == "user"', [2]),
            ("normal", "str_second_index_eq", 'arr_str[1] == "ops"', [3]),
            ("normal", "bool_index_eq_true", "arr_bool[0] == true", [1, 3]),
            ("expected_error", "int_subscript_is_null_unsupported", "arr_int[0] is null", None, "unsupported"),
            (
                "expected_error",
                "int_subscript_is_not_null_unsupported",
                "arr_int[0] is not null",
                None,
                "unsupported",
            ),
        ]

        failed = run_operator_cases(
            collection=col,
            tests=tests,
            query_fn=query_ids,
            title="ARRAY subscript access validation",
        )
        return 0 if failed == 0 else 1

    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)
        connections.disconnect("default")


if __name__ == "__main__":
    raise SystemExit(main())
