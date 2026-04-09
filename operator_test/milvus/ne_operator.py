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
COLLECTION_NAME = "ne_operator_validation"


def query_ids(collection, expr):
    rows = collection.query(expr, output_fields=["id"], consistency_level="Strong")
    return sorted(row["id"] for row in rows)


def main():
    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as exc:
        print(f"connection_failed: {exc}")
        return

    try:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="age", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=32, nullable=True),
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        schema = CollectionSchema(fields)
        col = Collection(COLLECTION_NAME, schema)

        col.insert(
            [
                {
                    "id": 1,
                    "age": 30,
                    "role": "admin",
                    "meta": {"role": "admin"},
                    "vec": [0.0, 0.0],
                },
                {
                    "id": 2,
                    "age": 31,
                    "role": "user",
                    "meta": {"role": "user"},
                    "vec": [1.0, 1.0],
                },
                {
                    "id": 3,
                    "age": None,
                    "role": None,
                    "meta": {"team": "ops"},
                    "vec": [2.0, 2.0],
                },
                {
                    "id": 4,
                    "age": 29,
                    "role": "guest",
                    "meta": None,
                    "vec": [3.0, 3.0],
                },
            ]
        )
        col.flush()
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"})
        col.load()
        time.sleep(1)

        tests = [
            ("int_ne_guarded", "(age is not null and (age != 30))", [2, 4]),
            ('string_ne_guarded', '(role is not null and (role != "admin"))', [2, 4]),
            ('json_ne_guarded_bugcheck', '(meta is not null and (meta["role"] != "admin"))', [2]),
            ('json_ne_direct_bugcheck', 'meta["role"] != "admin"', [2]),
        ]

        print("--- NE operator validation ---")
        for name, expr, expected_ids in tests:
            actual_ids = query_ids(col, expr)
            status = "PASS" if actual_ids == expected_ids else "FAIL"
            print(
                f"{name}: {status} | expr={expr} | "
                f"expected={expected_ids} | actual={actual_ids}"
            )
    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)
        connections.disconnect("default")


if __name__ == "__main__":
    main()
