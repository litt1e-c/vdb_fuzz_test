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
COLLECTION_NAME = "like_operator_validation"


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
            utility.drop_collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=32, nullable=True),
            FieldSchema(name="meta", dtype=DataType.JSON, nullable=True),
            FieldSchema(
                name="tags",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=4,
                max_length=32,
                nullable=True,
            ),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        schema = CollectionSchema(fields)
        col = Collection(COLLECTION_NAME, schema)

        col.insert(
            [
                {
                    "id": 1,
                    "role": "admin",
                    "meta": {"role": "admin"},
                    "tags": ["admin", "ops"],
                    "vec": [0.0, 0.0],
                },
                {
                    "id": 2,
                    "role": "user",
                    "meta": {"role": "user"},
                    "tags": ["user", "eng"],
                    "vec": [1.0, 1.0],
                },
                {
                    "id": 3,
                    "role": "guest",
                    "meta": {"role": "guest"},
                    "tags": ["guest", "sales"],
                    "vec": [2.0, 2.0],
                },
                {
                    "id": 4,
                    "role": None,
                    "meta": {"team": "ops"},
                    "tags": ["ops"],
                    "vec": [3.0, 3.0],
                },
                {
                    "id": 5,
                    "role": None,
                    "meta": None,
                    "tags": None,
                    "vec": [4.0, 4.0],
                },
                {
                    "id": 6,
                    "role": "alpha",
                    "meta": {"role": None},
                    "tags": ["alpha"],
                    "vec": [5.0, 5.0],
                },
            ]
        )
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"}, timeout=10)
        col.load(timeout=10)
        time.sleep(1)

        tests = [
            (
                "scalar_prefix_like",
                '(role is not null and (role like "ad%"))',
                [1],
            ),
            (
                "scalar_suffix_like",
                '(role is not null and (role like "%er"))',
                [2],
            ),
            (
                "scalar_infix_like",
                '(role is not null and (role like "%ues%"))',
                [3],
            ),
            (
                "scalar_underscore_like",
                '(role is not null and (role like "u_er"))',
                [2],
            ),
            (
                "json_prefix_like_direct",
                'meta["role"] like "ad%"',
                [1],
            ),
            (
                "json_suffix_like_direct",
                'meta["role"] like "%er"',
                [2],
            ),
            (
                "json_infix_like_direct",
                'meta["role"] like "%ues%"',
                [3],
            ),
            (
                "json_underscore_like_direct",
                'meta["role"] like "u_er"',
                [2],
            ),
            (
                "json_missing_key_filtered",
                'meta["role"] like "%ops%"',
                [],
            ),
            (
                "json_null_or_null_field_filtered",
                'meta["role"] like "alph%"',
                [],
            ),
            (
                "json_guarded_not_like",
                'meta is not null and not (meta["role"] like "ad%")',
                [2, 3, 4, 6],
            ),
            (
                "json_guarded_or_like",
                '(meta is not null and (meta["role"] like "ad%")) or id == 2',
                [1, 2],
            ),
            (
                "array_prefix_like",
                'tags[0] like "ad%"',
                [1],
            ),
            (
                "array_suffix_like",
                'tags[0] like "%min"',
                [1],
            ),
            (
                "array_infix_like",
                'tags[0] like "%dmi%"',
                [1],
            ),
            (
                "array_underscore_like",
                'tags[0] like "ad_in"',
                [1],
            ),
        ]

        print("--- LIKE operator validation ---")
        for name, expr, expected_ids in tests:
            actual_ids = query_ids(col, expr)
            status = "PASS" if actual_ids == expected_ids else "FAIL"
            print(
                f"{name}: {status} | expr={expr} | "
                f"expected={expected_ids} | actual={actual_ids}"
            )

    finally:
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME, timeout=10)
        connections.disconnect("default")


if __name__ == "__main__":
    main()
