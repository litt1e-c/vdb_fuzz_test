import sys
import time
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051

COLLECTION_PREFIX = "IsNullOperatorValidation"
OBJECT_COLLECTION_PREFIX = "IsNullObjectOnlyValidation"


def deterministic_uuid(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-is-null-operator-{label}"))


MIXED_ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "nullableInt": 1,
            "nullableBool": True,
            "nullableDate": "2024-01-01T00:00:00Z",
            "optionalText": "alpha",
            "tagsArray": [1],
            "labelsArray": ["x"],
            "metaObj": {"price": 100},
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={
            "tag": "b",
            "nullableInt": None,
            "nullableBool": None,
            "nullableDate": None,
            "optionalText": None,
            "tagsArray": None,
            "labelsArray": None,
            "metaObj": None,
        },
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={
            "tag": "c",
        },
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={
            "tag": "d",
            "nullableInt": 0,
            "nullableBool": False,
            "nullableDate": "2024-01-02T00:00:00Z",
            "optionalText": "",
            "tagsArray": [],
            "labelsArray": [],
            "metaObj": {},
        },
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("e"),
        properties={
            "tag": "e",
            "nullableInt": 5,
            "nullableBool": True,
            "nullableDate": "2024-01-03T00:00:00Z",
            "optionalText": "beta",
            "tagsArray": [2, 3],
            "labelsArray": ["y"],
            "metaObj": {"price": 200},
        },
        vector=[0.5, 0.0, 0.5],
    ),
]


OBJECT_ONLY_ROWS = [
    DataObject(
        uuid=deterministic_uuid("obj_present"),
        properties={
            "tag": "present",
            "metaObj": {"price": 100},
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("obj_null"),
        properties={
            "tag": "null",
            "metaObj": None,
        },
        vector=[0.0, 1.0, 0.0],
    ),
]


def create_mixed_collection(client, name: str, object_property_null_state: bool):
    object_kwargs = dict(
        name="metaObj",
        data_type=DataType.OBJECT,
        index_filterable=True,
        nested_properties=[Property(name="price", data_type=DataType.INT)],
    )
    if object_property_null_state:
        object_kwargs["index_null_state"] = True

    return client.collections.create(
        name=name,
        properties=[
            Property(
                name="tag",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="nullableInt",
                data_type=DataType.INT,
                index_filterable=True,
                index_range_filterable=True,
            ),
            Property(
                name="nullableBool",
                data_type=DataType.BOOL,
                index_filterable=True,
            ),
            Property(
                name="nullableDate",
                data_type=DataType.DATE,
                index_filterable=True,
                index_range_filterable=True,
            ),
            Property(
                name="optionalText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="tagsArray",
                data_type=DataType.INT_ARRAY,
                index_filterable=True,
            ),
            Property(
                name="labelsArray",
                data_type=DataType.TEXT_ARRAY,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(**object_kwargs),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def create_object_only_collection(client, name: str, object_property_null_state: bool):
    object_kwargs = dict(
        name="metaObj",
        data_type=DataType.OBJECT,
        index_filterable=True,
        nested_properties=[Property(name="price", data_type=DataType.INT)],
    )
    if object_property_null_state:
        object_kwargs["index_null_state"] = True

    return client.collections.create(
        name=name,
        properties=[
            Property(
                name="tag",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(**object_kwargs),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def insert_rows(collection, rows):
    result = collection.data.insert_many(rows)
    errors = getattr(result, "errors", None)
    if errors:
        raise RuntimeError(f"insert_many errors: {errors}")


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


def print_observation(name, note, actual):
    print(name)
    print(f"  note:    {note}")
    print(f"  observed:{actual}")
    print("  result:  OBSERVED")


def build_name(prefix: str) -> str:
    return f"{prefix}{int(time.time())}{uuid.uuid4().hex[:8]}"


def cleanup_collection(client, name: str):
    try:
        if client.collections.exists(name):
            client.collections.delete(name)
            print(f"Cleaned up collection: {name}")
    except Exception as exc:
        print(f"Cleanup warning for {name}: {exc}")


def main():
    mixed_name = build_name(COLLECTION_PREFIX)
    mixed_prop_name = build_name(f"{COLLECTION_PREFIX}PropNullState")
    object_pair_name = build_name(OBJECT_COLLECTION_PREFIX)
    created_names = []

    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    try:
        mixed_collection = create_mixed_collection(
            client, mixed_name, object_property_null_state=False
        )
        created_names.append(mixed_name)
        insert_rows(mixed_collection, MIXED_ROWS)

        checks = []
        checks.append(
            print_check(
                "nullable_int_is_none_true_missing_matches_null_assumption",
                "IsNull filters for null state are documented; the docs do not give a separate missing-property truth table.",
                "Current fuzzer omits None-valued properties at write time, so it assumes omitted rows satisfy is_none(True).",
                ["b", "c"],
                fetch_tags(mixed_collection, Filter.by_property("nullableInt").is_none(True)),
            )
        )
        checks.append(
            print_check(
                "nullable_int_is_none_false_present_rows_only",
                "IsNull(false) should keep non-null scalar values.",
                "Oracle maps is_none(False) to present scalar rows in the current fuzzer subset.",
                ["a", "d", "e"],
                fetch_tags(mixed_collection, Filter.by_property("nullableInt").is_none(False)),
            )
        )
        checks.append(
            print_check(
                "nullable_bool_is_none_true_missing_matches_null_assumption",
                "IsNull filters for null state are documented; the docs do not give a separate missing-property truth table.",
                "Current fuzzer treats omitted BOOL properties as null-equivalent for is_none(True).",
                ["b", "c"],
                fetch_tags(mixed_collection, Filter.by_property("nullableBool").is_none(True)),
            )
        )
        checks.append(
            print_check(
                "nullable_date_is_none_true_missing_matches_null_assumption",
                "IsNull filters for null state are documented; the docs do not give a separate missing-property truth table.",
                "Current fuzzer treats omitted DATE properties as null-equivalent for is_none(True).",
                ["b", "c"],
                fetch_tags(mixed_collection, Filter.by_property("nullableDate").is_none(True)),
            )
        )
        checks.append(
            print_check(
                "optional_text_is_none_true_includes_empty_string",
                "Official null-state docs say empty strings are equivalent to null values.",
                "Oracle maps explicit null, omitted text, and empty strings to the effective-null text subset.",
                ["b", "c", "d"],
                fetch_tags(mixed_collection, Filter.by_property("optionalText").is_none(True)),
            )
        )
        checks.append(
            print_check(
                "optional_text_is_none_false_excludes_empty_string",
                "Official null-state docs say empty strings are equivalent to null values.",
                "Oracle maps is_none(False) to present non-empty text rows.",
                ["a", "e"],
                fetch_tags(mixed_collection, Filter.by_property("optionalText").is_none(False)),
            )
        )
        checks.append(
            print_check(
                "int_array_is_none_true_includes_empty_array",
                "Official null-state docs say zero-length arrays are equivalent to null values.",
                "Oracle maps explicit null, omitted arrays, and empty arrays to the effective-null array subset.",
                ["b", "c", "d"],
                fetch_tags(mixed_collection, Filter.by_property("tagsArray").is_none(True)),
            )
        )
        checks.append(
            print_check(
                "int_array_is_none_false_excludes_empty_array",
                "Official null-state docs say zero-length arrays are equivalent to null values.",
                "Oracle maps is_none(False) to arrays with at least one element.",
                ["a", "e"],
                fetch_tags(mixed_collection, Filter.by_property("tagsArray").is_none(False)),
            )
        )
        checks.append(
            print_check(
                "text_array_is_none_true_includes_empty_array",
                "Official null-state docs say zero-length arrays are equivalent to null values.",
                "Oracle maps explicit null, omitted arrays, and empty arrays to the effective-null array subset.",
                ["b", "c", "d"],
                fetch_tags(mixed_collection, Filter.by_property("labelsArray").is_none(True)),
            )
        )
        checks.append(
            print_check(
                "text_array_is_none_false_excludes_empty_array",
                "Official null-state docs say zero-length arrays are equivalent to null values.",
                "Oracle maps is_none(False) to arrays with at least one element.",
                ["a", "e"],
                fetch_tags(mixed_collection, Filter.by_property("labelsArray").is_none(False)),
            )
        )

        print_observation(
            "object_is_none_true_fuzzer_schema_observation",
            "This matches the current fuzzer schema shape: collection-level index_null_state=True, but no property-level object index_null_state override.",
            fetch_tags(mixed_collection, Filter.by_property("metaObj").is_none(True)),
        )
        print_observation(
            "object_is_none_false_fuzzer_schema_observation",
            "GitHub issue10642 reports OBJECT is_none(False) returning 0 results; this records current behavior for the fuzzer-like schema shape.",
            fetch_tags(mixed_collection, Filter.by_property("metaObj").is_none(False)),
        )

        mixed_prop_collection = create_mixed_collection(
            client, mixed_prop_name, object_property_null_state=True
        )
        created_names.append(mixed_prop_name)
        insert_rows(mixed_prop_collection, MIXED_ROWS)

        print_observation(
            "object_is_none_true_property_null_state_observation",
            "This adds property-level index_null_state=True on the OBJECT field to compare with the current fuzzer schema.",
            fetch_tags(mixed_prop_collection, Filter.by_property("metaObj").is_none(True)),
        )
        print_observation(
            "object_is_none_false_property_null_state_observation",
            "This adds property-level index_null_state=True on the OBJECT field to compare with the current fuzzer schema and issue10642.",
            fetch_tags(mixed_prop_collection, Filter.by_property("metaObj").is_none(False)),
        )

        object_pair_collection = create_object_only_collection(
            client, object_pair_name, object_property_null_state=True
        )
        created_names.append(object_pair_name)
        insert_rows(object_pair_collection, OBJECT_ONLY_ROWS)

        print_observation(
            "object_pair_is_none_true_explicit_null_only_observation",
            "This mirrors the local historical POC shape without a missing OBJECT row.",
            fetch_tags(object_pair_collection, Filter.by_property("metaObj").is_none(True)),
        )
        print_observation(
            "object_pair_is_none_false_explicit_null_only_observation",
            "This mirrors the local historical POC shape without a missing OBJECT row.",
            fetch_tags(object_pair_collection, Filter.by_property("metaObj").is_none(False)),
        )

        if all(checks):
            print("Summary: all documented/validated is_null subset checks passed on the local Weaviate service.")
            return 0

        print("Summary: one or more documented/validated is_null subset checks failed.")
        return 1
    finally:
        for name in reversed(created_names):
            cleanup_collection(client, name)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
