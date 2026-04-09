import sys

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
CLASS_NAME = "BugSignedZeroNumberFilters"

ROWS = [
    DataObject(
        uuid="00000000-0000-0000-0000-000000000001",
        properties={"tag": "neg0", "c1": -0.0},
    ),
    DataObject(
        uuid="00000000-0000-0000-0000-000000000002",
        properties={"tag": "pos0", "c1": 0.0},
    ),
    DataObject(
        uuid="00000000-0000-0000-0000-000000000003",
        properties={"tag": "neg_big", "c1": -1000000.0},
    ),
]


def fetch_tags(collection, flt):
    res = collection.query.fetch_objects(filters=flt, limit=10)
    return sorted(obj.properties["tag"] for obj in res.objects)


def print_check(name, expected, actual):
    print(f"{name}:")
    print(f"  expected: {expected}")
    print(f"  actual:   {actual}")
    return actual == expected


def main():
    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    try:
        if client.collections.exists(CLASS_NAME):
            client.collections.delete(CLASS_NAME)

        col = client.collections.create(
            name=CLASS_NAME,
            properties=[
                Property(name="tag", data_type=DataType.TEXT, index_filterable=True),
                Property(
                    name="c1",
                    data_type=DataType.NUMBER,
                    index_filterable=True,
                    index_range_filterable=False,
                ),
            ],
            inverted_index_config=Configure.inverted_index(index_null_state=True),
        )
        col.data.insert_many(ROWS)

        print("rows:", [{"tag": obj.properties["tag"], "c1": obj.properties["c1"]} for obj in ROWS])

        ok_eq = print_check(
            "equal(0.0)",
            ["neg0", "pos0"],
            fetch_tags(col, Filter.by_property("c1").equal(0.0)),
        )
        ok_approx = print_check(
            "approx_zero_range[-2e-323,2e-323]",
            ["neg0", "pos0"],
            fetch_tags(
                col,
                Filter.by_property("c1").greater_or_equal(-2e-323)
                & Filter.by_property("c1").less_or_equal(2e-323),
            ),
        )
        ok_le = print_check(
            "c1 <= -1000000.0",
            ["neg_big"],
            fetch_tags(col, Filter.by_property("c1").less_or_equal(-1000000.0)),
        )
        ok_ge = print_check(
            "c1 >= 0.0",
            ["neg0", "pos0"],
            fetch_tags(col, Filter.by_property("c1").greater_or_equal(0.0)),
        )

        if ok_eq and ok_approx and ok_le and ok_ge:
            print("NO BUG: NUMBER filters treat -0.0 consistently")
            return 0

        print("BUG CONFIRMED: Weaviate mishandles -0.0 in NUMBER equality/range/order filters")
        return 2
    finally:
        try:
            if client.collections.exists(CLASS_NAME):
                client.collections.delete(CLASS_NAME)
        finally:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
