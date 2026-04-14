import datetime as dt
import time
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_PREFIX = "ScalarContainsValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-scalar-contains-{label}"))


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


def print_observation(name, note, observed):
    print(name)
    print(f"  note:    {note}")
    print(f"  observed:{observed}")
    print("  result:  OBSERVED")


def main():
    suffix = f"{int(time.time())}{uuid.uuid4().hex[:8]}"
    collection_name = f"{COLLECTION_PREFIX}{suffix}"
    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    created = False
    try:
        if client.collections.exists(collection_name):
            print(f"Refusing to reuse existing collection: {collection_name}")
            return 2

        collection = create_collection(client, collection_name)
        created = True
        result = collection.data.insert_many(ROWS)
        errors = getattr(result, "errors", None)
        if errors:
            print(f"Insert errors: {errors}")
            return 2

        checks = []

        checks.append(print_check(
            "int_scalar_contains_any",
            "Local Python-client filter builder accepts contains_any on scalar INT values.",
            "Main oracle can conservatively treat scalar INT contains_any([v,...]) as equality membership over present scalar values.",
            ["a", "c"],
            fetch_tags(collection, Filter.by_property("intVal").contains_any([1, 9])),
        ))
        checks.append(print_check(
            "int_scalar_contains_none",
            "Local scalar contains_none excludes matching scalar values and includes missing rows.",
            "Main oracle can model scalar INT contains_none as value-not-in-targets with null/missing rows treated as matches.",
            ["b", "d", "e"],
            fetch_tags(collection, Filter.by_property("intVal").contains_none([1, 9])),
        ))
        checks.append(print_check(
            "int_scalar_contains_all_singleton",
            "A singleton contains_all([v]) on local scalar INT behaves like equality.",
            "Main oracle can safely generate scalar contains_all only in the singleton case.",
            ["a", "c"],
            fetch_tags(collection, Filter.by_property("intVal").contains_all([1])),
        ))

        checks.append(print_check(
            "number_scalar_contains_any",
            "Local scalar NUMBER contains_any behaves as membership over present numeric scalars.",
            "Main oracle can model scalar NUMBER contains_any with exact float membership on the generated subset.",
            ["b", "c"],
            fetch_tags(collection, Filter.by_property("numVal").contains_any([2.5, 7.5])),
        ))
        checks.append(print_check(
            "number_scalar_contains_none",
            "Local scalar NUMBER contains_none excludes the targeted numeric values and includes missing rows.",
            "Main oracle can model scalar NUMBER contains_none as numeric non-membership with null/missing rows included.",
            ["a", "d", "e"],
            fetch_tags(collection, Filter.by_property("numVal").contains_none([2.5, 7.5])),
        ))

        checks.append(print_check(
            "bool_scalar_contains_any",
            "Local scalar BOOL contains_any([true]) behaves like equality to true.",
            "Main oracle can model scalar BOOL contains_any/contains_all(singleton) as equality and contains_none as boolean exclusion.",
            ["a", "c", "e"],
            fetch_tags(collection, Filter.by_property("flag").contains_any([True])),
        ))
        checks.append(print_check(
            "bool_scalar_contains_none",
            "Local scalar BOOL contains_none([true]) returns false rows plus missing rows.",
            "Main oracle can model scalar BOOL contains_none with null/missing rows treated as matches.",
            ["b", "d"],
            fetch_tags(collection, Filter.by_property("flag").contains_none([True])),
        ))
        checks.append(print_check(
            "bool_scalar_contains_all_singleton",
            "A singleton contains_all([true]) on local scalar BOOL behaves like equality to true.",
            "Main oracle can safely generate singleton scalar BOOL contains_all.",
            ["a", "c", "e"],
            fetch_tags(collection, Filter.by_property("flag").contains_all([True])),
        ))

        checks.append(print_check(
            "date_scalar_contains_any",
            "Local scalar DATE contains_any behaves as timestamp membership over present scalar values.",
            "Main oracle can model scalar DATE contains_any/contains_none/contains_all(singleton) on canonical timestamps.",
            ["b", "e"],
            fetch_tags(collection, Filter.by_property("ts").contains_any([utc(2024, 1, 2)])),
        ))
        checks.append(print_check(
            "date_scalar_contains_none",
            "Local scalar DATE contains_none excludes targeted timestamps and includes missing rows.",
            "Main oracle can model scalar DATE contains_none as timestamp non-membership with null/missing rows included.",
            ["a", "c", "d"],
            fetch_tags(collection, Filter.by_property("ts").contains_none([utc(2024, 1, 2)])),
        ))
        checks.append(print_check(
            "date_scalar_contains_all_singleton",
            "A singleton contains_all([timestamp]) on local scalar DATE behaves like equality to that timestamp.",
            "Main oracle can safely generate singleton scalar DATE contains_all.",
            ["b", "e"],
            fetch_tags(collection, Filter.by_property("ts").contains_all([utc(2024, 1, 2)])),
        ))

        print_observation(
            "scalar_contains_all_multi_target_observation",
            "On local scalar fields, a multi-target contains_all([v1, v2]) only matches if one scalar value can satisfy the whole target set. The tested INT case returned no rows for distinct targets [1, 2], so the main oracle keeps scalar contains_all to the singleton subset.",
            fetch_tags(collection, Filter.by_property("intVal").contains_all([1, 2])),
        )

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
