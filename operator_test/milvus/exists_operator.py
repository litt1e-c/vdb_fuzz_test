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
COLLECTION_NAME = "exists_operator_validation"


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
            "meta": {
                "price": 10,
                "role": "admin",
                "nested": {"x": 1},
                "history": [1, 2],
                "obj": {"y": 2, "z": True},
            },
            "vec": [0.0, 0.0],
        },
        {
            "id": 2,
            "meta": {
                "price": None,
                "role": None,
                "nested": {"x": None},
                "history": [],
                "obj": {},
            },
            "vec": [1.0, 1.0],
        },
        {
            "id": 3,
            "meta": {},
            "vec": [2.0, 2.0],
        },
        {
            "id": 4,
            "meta": None,
            "vec": [3.0, 3.0],
        },
        {
            "id": 5,
            "vec": [4.0, 4.0],
        },
        {
            "id": 6,
            "meta": {
                "nested": {"x": 1},
                "history": [None],
                "obj": {"z": None},
            },
            "vec": [5.0, 5.0],
        },
        {
            "id": 7,
            "meta": {
                "zero": 0,
                "flag": False,
                "empty_str": "",
                "nested": {"x": 0},
                "history": [0],
                "obj": {"z": False},
            },
            "vec": [6.0, 6.0],
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
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        col = Collection(COLLECTION_NAME, CollectionSchema(fields))
        col.insert(build_rows())
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=10)
        col.load(timeout=10)
        time.sleep(1)

        tests = [
            ("function_syntax_price", 'exists(meta["price"])', [1]),
            ("legacy_syntax_price", 'exists meta["price"]', [1]),
            ("exists_role", 'exists(meta["role"])', [1]),
            ("exists_nested_container", 'exists(meta["nested"])', [1, 6, 7]),
            ("exists_nested_scalar", 'exists(meta["nested"]["x"])', [1, 6, 7]),
            ("exists_history_container", 'exists(meta["history"])', [1, 7]),
            ("exists_history_index", 'exists(meta["history"][0])', [1, 7]),
            ("exists_obj_container", 'exists(meta["obj"])', [1, 7]),
            ("exists_obj_nested_false", 'exists(meta["obj"]["z"])', [1, 7]),
            ("exists_zero", 'exists(meta["zero"])', [7]),
            ("exists_flag_false", 'exists(meta["flag"])', [7]),
            ("exists_empty_string", 'exists(meta["empty_str"])', [7]),
            ("exists_missing", 'exists(meta["missing"])', []),
            ("raw_not_exists_price", 'not (exists(meta["price"]))', [2, 3, 4, 5, 6, 7]),
        ]

        print("--- EXISTS operator validation ---")
        failed = 0
        for name, expr, expected_ids in tests:
            try:
                actual_ids = query_ids(col, expr)
                status = "PASS" if actual_ids == expected_ids else "FAIL"
                if status == "FAIL":
                    failed += 1
                print(
                    f"{name}: {status} | expr={expr} | "
                    f"expected={expected_ids} | actual={actual_ids}"
                )
            except Exception as exc:
                failed += 1
                print(
                    f"{name}: ERROR | expr={expr} | "
                    f"error={type(exc).__name__}: {exc}"
                )

        summary = "PASS" if failed == 0 else "FAIL"
        print(f"summary: {summary} | failed={failed} | total={len(tests)}")

    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)
        connections.disconnect("default")


if __name__ == "__main__":
    main()
