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
COLLECTION_NAME = "or_operator_validation"


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
            FieldSchema(name="color", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="nullable_score", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        schema = CollectionSchema(fields)
        col = Collection(COLLECTION_NAME, schema)

        col.insert(
            [
                {
                    "id": 1,
                    "color": "red",
                    "nullable_score": None,
                    "vec": [0.0, 0.0],
                },
                {
                    "id": 2,
                    "color": "green",
                    "nullable_score": None,
                    "vec": [1.0, 1.0],
                },
            ]
        )
        col.flush()
        col.create_index("vec", {"metric_type": "L2", "index_type": "FLAT"})
        col.load()
        time.sleep(1)

        tests = [
            ("baseline_left", 'color == "red"', [1]),
            ("baseline_right", 'color == "blue"', []),
            ("baseline_or", 'color == "red" or color == "blue"', [1]),
            ("edge_left", "id == 1", [1]),
            ("edge_right", "nullable_score > 0", []),
            ("edge_or", "(id == 1) or (nullable_score > 0)", [1]),
        ]

        print("--- OR operator validation ---")
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
