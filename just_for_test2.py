import sys
import traceback

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter

HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
CLASS_NAME = "BugLengthFilterableArrayParity"

ROWS = [
    DataObject(
        uuid="00000000-0000-0000-0000-000000000001",
        properties={
            "tag": "r1",
            "text_search_only": "abc",      # len = 3
            "ints_non_filterable": [1, 2],  # len = 2
            "ints_filterable": [1, 2],      # len = 2
        },
        vector=[0.0, 0.0, 0.0],
    ),
    DataObject(
        uuid="00000000-0000-0000-0000-000000000002",
        properties={
            "tag": "r2",
            "text_search_only": "abcd",         # len = 4
            "ints_non_filterable": [1, 2, 3],   # len = 3
            "ints_filterable": [1, 2, 3],       # len = 3
        },
        vector=[0.0, 0.0, 0.0],
    ),
    DataObject(
        uuid="00000000-0000-0000-0000-000000000003",
        properties={
            "tag": "r3",
            "text_search_only": "",   # len = 0
            "ints_non_filterable": [],# len = 0
            "ints_filterable": [],    # len = 0
        },
        vector=[0.0, 0.0, 0.0],
    ),
]


def fetch_tags(collection, flt):
    res = collection.query.fetch_objects(filters=flt, limit=20)
    return sorted(obj.properties["tag"] for obj in res.objects)


def run_case(collection, name, flt, expected=None):
    print(f"\n[{name}]")
    try:
        actual = fetch_tags(collection, flt)
        print("actual:  ", actual)
        if expected is not None:
            print("expected:", expected)
            print("match:   ", actual == expected)
        return ("ok", actual)
    except Exception as e:
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR MSG :", str(e))
        tb = traceback.format_exc()
        print(tb)
        return ("err", f"{type(e).__name__}: {e}")


def main():
    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    try:
        if client.collections.exists(CLASS_NAME):
            client.collections.delete(CLASS_NAME)

        col = client.collections.create(
            name=CLASS_NAME,
            vector_config=Configure.Vectors.self_provided(),
            inverted_index_config=Configure.inverted_index(
                index_property_length=True,
                index_null_state=True,
            ),
            properties=[
                Property(name="tag", data_type=DataType.TEXT, index_filterable=True),
                Property(
                    name="text_search_only",
                    data_type=DataType.TEXT,
                    index_filterable=False,
                    index_searchable=True,
                ),
                Property(
                    name="ints_non_filterable",
                    data_type=DataType.INT_ARRAY,
                    index_filterable=False,
                ),
                Property(
                    name="ints_filterable",
                    data_type=DataType.INT_ARRAY,
                    index_filterable=True,
                ),
            ],
        )

        col.data.insert_many(ROWS)

        print("[rows]")
        for row in ROWS:
            print(row.uuid, row.properties)

        # 1) text_search_only 长度过滤：search-only text
        run_case(
            col,
            'text_search_only len == 3',
            Filter.by_property("text_search_only", length=True).equal(3),
            expected=["r1"],
        )

        # 2) non-filterable INT_ARRAY 长度过滤：这是你想盯的点
        run_case(
            col,
            'ints_non_filterable len == 2',
            Filter.by_property("ints_non_filterable", length=True).equal(2),
            expected=["r1"],
        )

        # 3) filterable INT_ARRAY 长度过滤：对照组
        run_case(
            col,
            'ints_filterable len == 2',
            Filter.by_property("ints_filterable", length=True).equal(2),
            expected=["r1"],
        )

        # 再补几组，避免只测一个长度
        run_case(
            col,
            'text_search_only len == 0',
            Filter.by_property("text_search_only", length=True).equal(0),
            expected=["r3"],
        )

        run_case(
            col,
            'ints_non_filterable len == 0',
            Filter.by_property("ints_non_filterable", length=True).equal(0),
            expected=["r3"],
        )

        run_case(
            col,
            'ints_filterable len == 0',
            Filter.by_property("ints_filterable", length=True).equal(0),
            expected=["r3"],
        )

        print("\n[diagnosis guide]")
        print("- 如果三个都成功：说明这个最小 case 里没有复现。")
        print("- 如果 text_search_only 成功，ints_filterable 成功，但 ints_non_filterable 报错：强烈怀疑 non-filterable INT_ARRAY length 实现不一致。")
        print("- 如果 text_search_only 成功，而两个 INT_ARRAY 都报错：怀疑 array-length 整体链路有问题。")
        print("- 如果三个都报错：先检查 index_property_length 是否真的生效。")

        return 0

    finally:
        try:
            if client.collections.exists(CLASS_NAME):
                client.collections.delete(CLASS_NAME)
        finally:
            client.close()


if __name__ == "__main__":
    sys.exit(main())