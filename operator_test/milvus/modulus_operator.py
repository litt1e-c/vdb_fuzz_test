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
PORT = os.getenv("MILVUS_PORT", "19532")
COLLECTION_NAME = "modulus_operator_validation"


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
            "c_int": 5,
            "c_float": 5.5,
            "c_double": 5.5,
            "vec": [0.0, 0.0],
        },
        {
            "id": 2,
            "c_int": 4,
            "c_float": 4.25,
            "c_double": 4.25,
            "vec": [1.0, 1.0],
        },
        {
            "id": 3,
            "c_int": -5,
            "c_float": -5.5,
            "c_double": -5.5,
            "vec": [2.0, 2.0],
        },
        {
            "id": 4,
            "c_int": 0,
            "c_float": 0.0,
            "c_double": 0.0,
            "vec": [3.0, 3.0],
        },
        {
            "id": 5,
            "c_int": None,
            "c_float": None,
            "c_double": None,
            "vec": [4.0, 4.0],
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
            FieldSchema(name="c_int", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="c_float", dtype=DataType.FLOAT, nullable=True),
            FieldSchema(name="c_double", dtype=DataType.DOUBLE, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(COLLECTION_NAME, CollectionSchema(fields))
        col.insert(build_rows())
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=10)
        col.load(timeout=10)
        time.sleep(1)

        print("--- MODULUS operator validation ---")
        tests = [
            ("normal", "int_even_zero", '(c_int is not null and (c_int % 2 == 0))', [2, 4]),
            ("normal", "int_positive_remainder", '(c_int is not null and (c_int % 2 == 1))', [1]),
            ("normal", "int_negative_remainder", '(c_int is not null and (c_int % 2 == -1))', [3]),
            ("normal", "int_positive_gt_zero", '(c_int is not null and (c_int % 4 > 0))', [1]),
            (
                "expected_error",
                "float_positive_remainder_unsupported",
                '(c_float is not null and (c_float % 2 > 1.0))',
                None,
            ),
            (
                "expected_error",
                "float_negative_remainder_unsupported",
                '(c_float is not null and (c_float % 2 < -1.0))',
                None,
            ),
            (
                "expected_error",
                "double_small_nonnegative_unsupported",
                '(c_double is not null and (c_double % 2 >= 0 and c_double % 2 <= 0.3))',
                None,
            ),
        ]

        failed = 0
        for mode, name, expr, expected_ids in tests:
            try:
                actual_ids = query_ids(col, expr)
                if mode == "expected_error":
                    failed += 1
                    print(
                        f"{name}: FAIL | expr={expr} | "
                        f"expected_error=unsupported | actual={actual_ids}"
                    )
                else:
                    status = "PASS" if actual_ids == expected_ids else "FAIL"
                    if status == "FAIL":
                        failed += 1
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
                    failed += 1
                    print(
                        f"{name}: ERROR | expr={expr} | "
                        f"error={type(exc).__name__}: {exc}"
                    )

        print(
            "mod_zero_probe: SKIPPED_RISKY | "
            "reason=zero-divisor arithmetic is kept out of shared-server validation"
        )

        summary = "PASS" if failed == 0 else "FAIL"
        print(f"summary: {summary} | failed={failed} | total={len(tests)}")

    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)
        connections.disconnect("default")


if __name__ == "__main__":
    main()
