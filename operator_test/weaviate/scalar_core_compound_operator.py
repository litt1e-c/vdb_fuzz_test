import datetime as dt
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_NAME = "ScalarCoreCompoundValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-scalar-core-compound-{label}"))


def utc(year, month, day):
    return dt.datetime(year, month, day, tzinfo=dt.timezone.utc)


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={"tag": "a", "intVal": 1, "numVal": 1.5, "flag": True, "ts": "2024-01-01T00:00:00Z", "textVal": "Alpha"},
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={"tag": "b", "intVal": 2, "numVal": 2.5, "flag": False, "ts": "2024-01-02T00:00:00Z", "textVal": "Beta"},
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={"tag": "c", "intVal": 3, "numVal": 2.5, "flag": True, "ts": "2024-01-03T00:00:00Z", "textVal": "Alpha"},
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={"tag": "d"},
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("e"),
        properties={"tag": "e", "intVal": 4, "numVal": 4.5, "flag": True, "ts": "2024-01-04T00:00:00Z", "textVal": "Gamma"},
        vector=[0.5, 0.0, 0.5],
    ),
]


def create_collection(client, name):
    return client.collections.create(
        name=name,
        properties=[
            Property(name="tag", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="intVal", data_type=DataType.INT, index_filterable=True, index_range_filters=True),
            Property(name="numVal", data_type=DataType.NUMBER, index_filterable=True, index_range_filters=True),
            Property(name="flag", data_type=DataType.BOOL, index_filterable=True),
            Property(name="ts", data_type=DataType.DATE, index_filterable=True, index_range_filters=True),
            Property(name="textVal", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def fetch_tags(collection, flt):
    response = collection.query.fetch_objects(filters=flt, limit=20)
    return sorted(obj.properties["tag"] for obj in response.objects)


def print_check(name, doc_expectation, oracle_expectation, expected, actual):
    print(name)
    print(f"  docs:    {doc_expectation}")
    print(f"  oracle:  {oracle_expectation}")
    print(f"  expected:{expected}")
    print(f"  observed:{actual}")
    ok = actual == expected
    print(f"  result:  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    collection_name = COLLECTION_NAME
    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    created = False
    try:
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        collection = create_collection(client, collection_name)
        created = True
        result = collection.data.insert_many(ROWS)
        errors = getattr(result, "errors", None)
        if errors:
            print(f"Insert errors: {errors}")
            return 2

        checks = []

        checks.append(print_check(
            "int_ge_and_not_lt_same_pivot",
            "Closed lower-bound semantics stay stable when the same pivot is expressed as `>= pivot` and `NOT(< pivot)` on an INT field.",
            "Main oracle can safely use `INT >= pivot AND NOT(INT < pivot)` as a deeper common-scalar self-consistency pattern.",
            ["b", "c", "e"],
            fetch_tags(
                collection,
                Filter.by_property("intVal").greater_or_equal(2) & Filter.not_(Filter.by_property("intVal").less_than(2)),
            ),
        ))
        checks.append(print_check(
            "int_closed_range",
            "Two-sided INT range filters keep ordinary inclusive range semantics under conjunction.",
            "Main oracle can safely emit `INT >= lo AND INT <= hi` as a common scalar range pattern.",
            ["b", "c"],
            fetch_tags(
                collection,
                Filter.by_property("intVal").greater_or_equal(2) & Filter.by_property("intVal").less_or_equal(3),
            ),
        ))
        checks.append(print_check(
            "number_le_and_not_gt_same_pivot",
            "Closed upper-bound semantics stay stable when the same pivot is expressed as `<= pivot` and `NOT(> pivot)` on a NUMBER field.",
            "Main oracle can safely use `NUMBER <= pivot AND NOT(NUMBER > pivot)` as a deeper common-scalar self-consistency pattern.",
            ["a", "b", "c"],
            fetch_tags(
                collection,
                Filter.by_property("numVal").less_or_equal(2.5) & Filter.not_(Filter.by_property("numVal").greater_than(2.5)),
            ),
        ))
        checks.append(print_check(
            "date_ge_and_not_lt_same_pivot",
            "DATE lower-bound filters remain stable under the paired negated strict-lower predicate.",
            "Main oracle can safely use `DATE >= pivot AND NOT(DATE < pivot)` as a deeper common-scalar date pattern.",
            ["b", "c", "e"],
            fetch_tags(
                collection,
                Filter.by_property("ts").greater_or_equal(utc(2024, 1, 2)) & Filter.not_(Filter.by_property("ts").less_than(utc(2024, 1, 2))),
            ),
        ))
        checks.append(print_check(
            "bool_true_and_not_false",
            "A BOOL equality predicate can be conjuncted with the negated opposite equality without changing the positive result set.",
            "Main oracle can safely use `flag == true AND NOT(flag == false)` as a deeper common-scalar BOOL pattern.",
            ["a", "c", "e"],
            fetch_tags(
                collection,
                Filter.by_property("flag").equal(True) & Filter.not_(Filter.by_property("flag").equal(False)),
            ),
        ))
        checks.append(print_check(
            "text_equal_and_not_not_equal_same_value",
            "FIELD-tokenized exact TEXT equality remains stable under conjunction with `NOT(not_equal(same_value))`.",
            "Main oracle can safely use `TEXT == value AND NOT(TEXT != value)` as a deeper exact-text self-consistency pattern.",
            ["a", "c"],
            fetch_tags(
                collection,
                Filter.by_property("textVal").equal("Alpha") & Filter.not_(Filter.by_property("textVal").not_equal("Alpha")),
            ),
        ))

        return 0 if all(checks) else 1
    finally:
        if created:
            try:
                client.collections.delete(collection_name)
            except Exception:
                pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
