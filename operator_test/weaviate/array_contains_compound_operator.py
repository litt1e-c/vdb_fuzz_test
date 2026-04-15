import datetime as dt
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_NAME = "ArrayContainsCompoundValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-array-contains-compound-{label}"))


def utc(year, month, day):
    return dt.datetime(year, month, day, tzinfo=dt.timezone.utc)


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "tagsArray": [1, 2],
            "labelsArray": ["Alpha", "Beta"],
            "scoresArray": [1.0, 2.0],
            "flagsArray": [True],
            "datesArray": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={
            "tag": "b",
            "tagsArray": [3],
            "labelsArray": ["alpha"],
            "scoresArray": [2.0],
            "flagsArray": [False],
            "datesArray": ["2024-01-02T00:00:00Z"],
        },
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={
            "tag": "c",
            "tagsArray": [2, 2],
            "labelsArray": ["Gamma"],
            "scoresArray": [3.0, 4.0],
            "flagsArray": [True, False],
            "datesArray": ["2024-01-03T00:00:00Z"],
        },
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={"tag": "d"},
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("e"),
        properties={
            "tag": "e",
            "tagsArray": [7],
            "labelsArray": [""],
            "scoresArray": [-1.0],
            "flagsArray": [False],
            "datesArray": ["2024-01-04T00:00:00Z"],
        },
        vector=[0.5, 0.0, 0.5],
    ),
]


def create_collection(client, name):
    return client.collections.create(
        name=name,
        properties=[
            Property(name="tag", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="tagsArray", data_type=DataType.INT_ARRAY, index_filterable=True),
            Property(name="labelsArray", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="scoresArray", data_type=DataType.NUMBER_ARRAY, index_filterable=True),
            Property(name="flagsArray", data_type=DataType.BOOL_ARRAY, index_filterable=True),
            Property(name="datesArray", data_type=DataType.DATE_ARRAY, index_filterable=True),
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
            "int_array_contains_any_and_not_none_self",
            "Boolean composition keeps array-membership semantics intact on INT_ARRAY when the same singleton target is used on both sides.",
            "Main oracle can safely model `contains_any([v]) AND NOT(contains_none([v]))` as a deeper self-consistency pattern for array membership.",
            ["a", "c"],
            fetch_tags(
                collection,
                Filter.by_property("tagsArray").contains_any([2]) & Filter.not_(Filter.by_property("tagsArray").contains_none([2])),
            ),
        ))
        checks.append(print_check(
            "text_array_contains_any_or_null",
            "ContainsAny on FIELD-tokenized TEXT_ARRAY can be disjoined with `is_null(true)` to retain missing arrays.",
            "Main oracle can safely generate `TEXT_ARRAY contains_any(...) OR is_null(true)` for compound array coverage.",
            ["a", "d"],
            fetch_tags(
                collection,
                Filter.by_property("labelsArray").contains_any(["Alpha"]) | Filter.by_property("labelsArray").is_none(True),
            ),
        ))
        checks.append(print_check(
            "bool_array_contains_all_or_null",
            "ContainsAll on BOOL_ARRAY can be composed with `is_null(true)` without changing the positive membership meaning.",
            "Main oracle can safely model `BOOL_ARRAY contains_all([True, False]) OR is_null(true)` using the positive membership mask plus the null mask.",
            ["c", "d"],
            fetch_tags(
                collection,
                Filter.by_property("flagsArray").contains_all([True, False]) | Filter.by_property("flagsArray").is_none(True),
            ),
        ))
        checks.append(print_check(
            "number_array_contains_any_and_none_other",
            "ContainsAny and ContainsNone on NUMBER_ARRAY can be conjuncted on disjoint target sets to isolate a narrower subset of rows.",
            "Main oracle can safely combine positive and exclusion membership masks on the same NUMBER_ARRAY field when the target sets are disjoint.",
            ["b", "c"],
            fetch_tags(
                collection,
                Filter.by_property("scoresArray").contains_any([2.0, 4.0]) & Filter.by_property("scoresArray").contains_none([1.0]),
            ),
        ))
        checks.append(print_check(
            "date_array_contains_any_and_not_none_self",
            "DATE_ARRAY supports the same self-consistency compound used for scalar DATE membership: positive hit AND NOT(contains_none(self)).",
            "Main oracle can safely reuse the validated self-target compound on DATE_ARRAY fields with UTC datetime inputs.",
            ["a", "b"],
            fetch_tags(
                collection,
                Filter.by_property("datesArray").contains_any([utc(2024, 1, 2)])
                & Filter.not_(Filter.by_property("datesArray").contains_none([utc(2024, 1, 2)])),
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
