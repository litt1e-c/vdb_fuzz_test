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
PORT = os.getenv("MILVUS_PORT", "19531")
COLLECTION_NAME = "and_operator_validation"


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
        return 2

    try:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="color", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="nullable_score", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="c_true", dtype=DataType.BOOL),
            FieldSchema(name="c_false", dtype=DataType.BOOL),
            FieldSchema(name="c_null", dtype=DataType.BOOL, nullable=True),
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(COLLECTION_NAME, CollectionSchema(fields))

        col.insert(
            [
                {
                    "id": 1,
                    "color": "red",
                    "nullable_score": 10,
                    "c_true": True,
                    "c_false": False,
                    "c_null": None,
                    "meta": {"role": "admin", "value": 100},
                    "vec": [0.0, 0.0],
                },
                {
                    "id": 2,
                    "color": "blue",
                    "nullable_score": None,
                    "c_true": True,
                    "c_false": False,
                    "c_null": None,
                    "meta": {"role": "user", "value": 40},
                    "vec": [1.0, 1.0],
                },
                {
                    "id": 3,
                    "color": "green",
                    "nullable_score": 5,
                    "c_true": True,
                    "c_false": False,
                    "c_null": None,
                    "meta": {"team": "ops"},
                    "vec": [2.0, 2.0],
                },
                {
                    "id": 4,
                    "color": "yellow",
                    "nullable_score": None,
                    "c_true": True,
                    "c_false": False,
                    "c_null": None,
                    "meta": None,
                    "vec": [3.0, 3.0],
                },
            ]
        )
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=10)
        col.load(timeout=10)
        time.sleep(1)

        tests = [
            (
                "normal",
                "scalar_true_true",
                '(id == 1) and (color == "red")',
                [1],
            ),
            (
                "normal",
                "scalar_true_false",
                '(id == 1) and (color == "blue")',
                [],
            ),
            (
                "normal",
                "scalar_guarded_null_check",
                '(nullable_score is not null) and (nullable_score > 8)',
                [1],
            ),
            (
                "normal",
                "json_guarded_and",
                '(meta is not null and (meta["value"] > 50)) and (id == 1)',
                [1],
            ),
            (
                "normal",
                "json_missing_key_and",
                '(meta is not null and (meta["role"] == "admin")) and (id == 3)',
                [],
            ),
            (
                "normal",
                "json_null_field_and",
                '(meta is not null and (meta["role"] == "admin")) and (id == 4)',
                [],
            ),
            (
                "normal",
                "not_null_and_false_3vl_probe",
                'not ((c_null == true) and (c_false == true))',
                [1, 2, 3, 4],
            ),
            (
                "expected_error",
                "constant_json_and_unsupported",
                '(null is null) and (meta["value"] > 50)',
                None,
                "parse_rejection",
            ),
        ]

        failed = run_operator_cases(
            collection=col,
            tests=tests,
            query_fn=query_ids,
            title="AND operator validation",
        )
        return 0 if failed == 0 else 1

    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)
        connections.disconnect("default")


if __name__ == "__main__":
    raise SystemExit(main())
