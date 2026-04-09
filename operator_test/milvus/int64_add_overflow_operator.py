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
COLLECTION_NAME = "int64_add_overflow_operator_validation"
MAX_INT64 = 9223372036854775807


def query_ids(collection, expr):
    rows = collection.query(
        expr,
        output_fields=["id", "c3"],
        consistency_level="Strong",
        timeout=10,
    )
    return [(row["id"], row.get("c3")) for row in rows]


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
            FieldSchema(name="c3", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(COLLECTION_NAME, CollectionSchema(fields))

        col.insert(
            [
                {"id": 1, "c3": MAX_INT64 - 1, "vec": [0.0, 0.0]},
                {"id": 2, "c3": MAX_INT64 - 2, "vec": [1.0, 1.0]},
                {"id": 3, "c3": 100, "vec": [2.0, 2.0]},
                {"id": 4, "c3": None, "vec": [3.0, 3.0]},
            ]
        )
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=10)
        col.load(timeout=10)
        time.sleep(1)

        tests = [
            (
                "overflow_probe_add_10_le",
                "(c3 is not null and (c3 + 10 <= 27881))",
                [3],
                "BUG_REPRODUCED if ids other than 3 appear",
            ),
            (
                "overflow_probe_add_10_lt_zero",
                "(c3 is not null and (c3 + 10 < 0))",
                [],
                "BUG_REPRODUCED if non-empty",
            ),
            (
                "boundary_control_add_1_lt_zero",
                "(c3 is not null and (c3 + 1 < 0))",
                [],
                "should stay empty without overflow",
            ),
            (
                "sanity_positive_compare",
                "(c3 is not null and (c3 > 0))",
                [1, 2, 3],
                "sanity check",
            ),
        ]

        print("--- INT64 add overflow probe ---")
        print(f"host={HOST} port={PORT}")
        for name, expr, expected_ids, note in tests:
            actual = query_ids(col, expr)
            actual_ids = [row_id for row_id, _ in actual]
            if "BUG_REPRODUCED" in note:
                status = "BUG_REPRODUCED" if actual_ids != expected_ids else "NO_REPRO"
            else:
                status = "PASS" if actual_ids == expected_ids else "FAIL"
            print(
                f"{name}: {status} | expr={expr} | "
                f"expected={expected_ids} | actual={actual_ids} | note={note}"
            )
            if actual:
                print(f"  rows={actual}")

    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)
        connections.disconnect("default")


if __name__ == "__main__":
    main()
