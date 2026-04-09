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
COLLECTION_NAME = "is_not_null_operator_validation"


def query_ids(collection, expr):
    rows = collection.query(
        expr,
        output_fields=["id"],
        consistency_level="Strong",
        timeout=10,
    )
    return sorted(row["id"] for row in rows)


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
            FieldSchema(name="scalar_txt", dtype=DataType.VARCHAR, max_length=32, nullable=True),
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=4, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(COLLECTION_NAME, CollectionSchema(fields))

        col.insert(
            [
                {
                    "id": 1,
                    "scalar_txt": "x",
                    "meta": {"role": "admin"},
                    "tags": [1, 2],
                    "vec": [0.0, 0.0],
                },
                {
                    "id": 2,
                    "scalar_txt": None,
                    "meta": None,
                    "tags": None,
                    "vec": [1.0, 1.0],
                },
                {
                    "id": 3,
                    "vec": [2.0, 2.0],
                },
                {
                    "id": 4,
                    "scalar_txt": "",
                    "meta": {"role": None},
                    "tags": [3],
                    "vec": [3.0, 3.0],
                },
                {
                    "id": 5,
                    "scalar_txt": "y",
                    "meta": {},
                    "tags": [],
                    "vec": [4.0, 4.0],
                },
            ]
        )
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=10)
        col.load(timeout=10)
        time.sleep(1)

        tests = [
            (
                "normal",
                "scalar_is_not_null",
                "scalar_txt is not null",
                [1, 4, 5],
            ),
            (
                "normal",
                "json_is_not_null",
                "meta is not null",
                [1, 4, 5],
            ),
            (
                "normal",
                "array_is_not_null",
                "tags is not null",
                [1, 4, 5],
            ),
            (
                "normal",
                "json_path_is_not_null_observed",
                'meta["role"] is not null',
                [1],
            ),
            (
                "expected_error",
                "array_subscript_is_not_null_unsupported",
                "tags[0] is not null",
                None,
            ),
        ]

        print("--- IS NOT NULL operator validation ---")
        for mode, name, expr, expected_ids in tests:
            try:
                actual_ids = query_ids(col, expr)
                if mode == "expected_error":
                    print(
                        f"{name}: FAIL | expr={expr} | "
                        f"expected_error=unsupported | actual={actual_ids}"
                    )
                else:
                    status = "PASS" if actual_ids == expected_ids else "FAIL"
                    print(
                        f"{name}: {status} | expr={expr} | "
                        f"expected={expected_ids} | actual={actual_ids}"
                    )
            except Exception as exc:
                if mode == "expected_error":
                    print(
                        f"{name}: PASS | expr={expr} | "
                        f"expected_error=unsupported | "
                        f"actual={type(exc).__name__}: {exc}"
                    )
                else:
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
