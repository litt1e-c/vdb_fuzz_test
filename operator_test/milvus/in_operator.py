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
COLLECTION_NAME = "in_operator_validation"


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
            FieldSchema(name="active", dtype=DataType.BOOL, nullable=True),
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        schema = CollectionSchema(fields)
        col = Collection(COLLECTION_NAME, schema)

        col.insert(
            [
                {
                    "id": 1,
                    "age": 20,
                    "role": "admin",
                    "active": True,
                    "meta": {"role": "admin", "num": 10},
                    "vec": [0.0, 0.0],
                },
                {
                    "id": 2,
                    "age": 30,
                    "role": "user",
                    "active": False,
                    "meta": {"role": "user", "num": 9223372036854775800},
                    "vec": [1.0, 1.0],
                },
                {
                    "id": 3,
                    "age": 40,
                    "role": "guest",
                    "active": True,
                    "meta": {"team": "ops"},
                    "vec": [2.0, 2.0],
                },
                {
                    "id": 4,
                    "age": None,
                    "role": None,
                    "active": None,
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
            (
                "scalar_varchar_in_guarded",
                '(role is not null and (role in ["admin", "guest"]))',
                [1, 3],
            ),
            (
                "scalar_int_in_guarded",
                "(age is not null and (age in [20, 30]))",
                [1, 2],
            ),
            (
                "scalar_bool_in_guarded",
                "(active is not null and (active in [true]))",
                [1, 3],
            ),
            (
                "json_role_in_guarded",
                '(meta is not null and (meta["role"] in ["admin"]))',
                [1],
            ),
            (
                "json_role_in_direct",
                'meta["role"] in ["admin"]',
                [1],
            ),
            (
                "json_missing_key_filtered",
                '(meta is not null and (meta["role"] in ["user"]))',
                [2],
            ),
            (
                "json_nonmatch_control",
                'meta["role"] in ["guest"]',
                [],
            ),
            (
                "json_numeric_in_exact",
                'meta["num"] in [9223372036854775800]',
                [2],
            ),
            (
                "json_numeric_in_nonmatch_control",
                'meta["num"] in [9223372036854775807]',
                [],
            ),
            (
                "json_numeric_in_mixed_type_bugcheck",
                'meta["num"] in [9223372036854775807, 1.5]',
                [],
            ),
        ]

        print("--- IN operator validation ---")
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
