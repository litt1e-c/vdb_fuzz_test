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
COLLECTION_NAME = "subtract_operator_validation"
INT64_MIN = -(2**63)


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
            "c_int": INT64_MIN,
            "c_float": 3.5,
            "c_double": 1.25,
            "vec": [0.0, 0.0],
        },
        {
            "id": 2,
            "c_int": 100,
            "c_float": 1000.0,
            "c_double": 1000.0,
            "vec": [1.0, 1.0],
        },
        {
            "id": 3,
            "c_int": INT64_MIN + 10,
            "c_float": None,
            "c_double": None,
            "vec": [2.0, 2.0],
        },
        {
            "id": 4,
            "c_int": None,
            "vec": [3.0, 3.0],
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

        print("--- SUBTRACT operator validation ---")
        safe_tests = [
            ("safe_int_eq", '(c_int is not null and (c_int - 33 == 67))', [2]),
            ("safe_float_gt", '(c_float is not null and (c_float - 2 > 1))', [1, 2]),
            ("safe_double_le", '(c_double is not null and (c_double - 2 <= -0.7))', [1]),
        ]
        overflow_probes = [
            ("underflow_false_positive", 'id in [1, 2] and (c_int - 1 >= 0)', [2]),
            ("underflow_false_negative", 'id in [1, 3] and (c_int - 10 < 0)', [1, 3]),
        ]

        safe_failures = 0
        for name, expr, expected_ids in safe_tests:
            try:
                actual_ids = query_ids(col, expr)
                status = "PASS" if actual_ids == expected_ids else "FAIL"
                if status == "FAIL":
                    safe_failures += 1
                print(
                    f"{name}: {status} | expr={expr} | "
                    f"expected={expected_ids} | actual={actual_ids}"
                )
            except Exception as exc:
                safe_failures += 1
                print(
                    f"{name}: ERROR | expr={expr} | "
                    f"error={type(exc).__name__}: {exc}"
                )

        reproduced = 0
        for name, expr, expected_ids in overflow_probes:
            try:
                actual_ids = query_ids(col, expr)
                status = "BUG_REPRODUCED" if actual_ids != expected_ids else "NO_REPRO"
                if status == "BUG_REPRODUCED":
                    reproduced += 1
                print(
                    f"{name}: {status} | expr={expr} | "
                    f"expected_math={expected_ids} | actual={actual_ids}"
                )
            except Exception as exc:
                print(
                    f"{name}: ERROR | expr={expr} | "
                    f"error={type(exc).__name__}: {exc}"
                )

        summary = "PASS" if safe_failures == 0 else "FAIL"
        print(
            f"summary: {summary} | safe_failed={safe_failures} | "
            f"safe_total={len(safe_tests)} | overflow_reproduced={reproduced}/{len(overflow_probes)}"
        )

    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)
        connections.disconnect("default")


if __name__ == "__main__":
    main()
