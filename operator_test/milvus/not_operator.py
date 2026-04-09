import time
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "not_operator_validation"


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
                    "nullable_score": 10,
                    "c_true": True,
                    "c_false": False,
                    "c_null": None,
                    "meta": {"value": 100},
                    "vec": [0.0, 0.0],
                },
                {
                    "id": 2,
                    "nullable_score": None,
                    "c_true": True,
                    "c_false": False,
                    "c_null": None,
                    "meta": {"value": 40},
                    "vec": [1.0, 1.0],
                },
                {
                    "id": 3,
                    "nullable_score": 5,
                    "c_true": True,
                    "c_false": False,
                    "c_null": False,
                    "meta": {"team": "ops"},
                    "vec": [2.0, 2.0],
                },
                {
                    "id": 4,
                    "nullable_score": 8,
                    "c_true": True,
                    "c_false": False,
                    "c_null": True,
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
                "scalar_not_eq",
                "not (id == 1)",
                [2, 3, 4],
            ),
            (
                "scalar_not_compare_null_filtered",
                "not (nullable_score > 8)",
                [3, 4],
            ),
            (
                "not_is_null",
                "not (nullable_score is null)",
                [1, 3, 4],
            ),
            (
                "not_is_not_null",
                "not (nullable_score is not null)",
                [2],
            ),
            (
                "double_not_is_null",
                "not (not (c_null is null))",
                [1, 2],
            ),
            (
                "not_and_range",
                "not ((id > 1) and (id < 4))",
                [1, 4],
            ),
            (
                "not_or_eq",
                "not ((id == 1) or (id == 2))",
                [3, 4],
            ),
            (
                "not_null_and_false_3vl_probe",
                "not ((c_null == true) and (c_false == true))",
                [1, 2, 3, 4],
            ),
            (
                "json_guarded_not_compare",
                'not ((meta is not null and (meta["value"] > 50)))',
                [2, 3, 4],
            ),
            (
                "json_raw_not_compare_observed",
                'not (meta["value"] > 50)',
                [2, 3],
            ),
            (
                "json_raw_not_missing_observed",
                'not (meta["missing"] > 50)',
                [1, 2, 3],
            ),
        ]

        print("--- NOT operator validation ---")
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
