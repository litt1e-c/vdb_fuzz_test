import datetime as dt
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_NAME = "ScalarContainsCompoundValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-scalar-contains-compound-{label}"))


def utc(year, month, day):
    return dt.datetime(year, month, day, tzinfo=dt.timezone.utc)


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={"tag": "a", "intVal": 1, "numVal": 1.5, "flag": True, "ts": "2024-01-01T00:00:00Z"},
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={"tag": "b", "intVal": 2, "numVal": 2.5, "flag": False, "ts": "2024-01-02T00:00:00Z"},
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={"tag": "c", "intVal": 1, "numVal": 2.5, "flag": True, "ts": "2024-01-03T00:00:00Z"},
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={"tag": "d", "flag": False},
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("e"),
        properties={"tag": "e", "intVal": 3, "numVal": 3.5, "flag": True, "ts": "2024-01-02T00:00:00Z"},
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
            "int_not_contains_none_singleton",
            "Local compound filters allow NOT to wrap scalar INT contains_none([v]) and the resulting filter excludes null/missing rows that positive contains_none would include.",
            "Main oracle can safely model NOT(int contains_none [v]) with explicit NOT-mask semantics rather than SQL three-valued logic.",
            ["a", "c"],
            fetch_tags(collection, Filter.not_(Filter.by_property("intVal").contains_none([1]))),
        ))
        checks.append(print_check(
            "int_contains_none_and_ge",
            "Local scalar INT contains_none can be conjuncted with a range predicate; missing rows still satisfy contains_none but are removed by the range side.",
            "Main oracle can combine scalar contains_none with INT range predicates using boolean-mask conjunction.",
            ["b", "e"],
            fetch_tags(
                collection,
                Filter.by_property("intVal").contains_none([1]) & Filter.by_property("intVal").greater_or_equal(2),
            ),
        ))
        checks.append(print_check(
            "bool_not_contains_any_true",
            "Local NOT(bool contains_any [true]) returns false rows plus missing rows under the installed stack semantics.",
            "Main oracle can treat NOT(contains_any(...)) as NOT over the positive contains mask with null/missing rows included on the negated side.",
            ["b", "d"],
            fetch_tags(collection, Filter.not_(Filter.by_property("flag").contains_any([True]))),
        ))
        checks.append(print_check(
            "number_contains_all_and_not_none",
            "A singleton scalar NUMBER contains_all([v]) remains stable under conjunction with NOT(contains_none([v])).",
            "Main oracle can safely use contains_all([v]) AND NOT(contains_none([v])) as a deeper self-consistency pattern for scalar NUMBER fields.",
            ["b", "c"],
            fetch_tags(
                collection,
                Filter.by_property("numVal").contains_all([2.5]) & Filter.not_(Filter.by_property("numVal").contains_none([2.5])),
            ),
        ))
        checks.append(print_check(
            "date_contains_any_or_null",
            "Local scalar DATE contains_any can be disjoined with is_null to keep missing rows in the result set.",
            "Main oracle can safely generate DATE contains_any(... ) OR is_null(true) as a deep scalar compound filter.",
            ["b", "d", "e"],
            fetch_tags(
                collection,
                Filter.by_property("ts").contains_any([utc(2024, 1, 2)]) | Filter.by_property("ts").is_none(True),
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
