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
COLLECTION_PREFIX = "PropertyLengthOperatorValidation"


def deterministic_uuid(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-property-length-{label}"))


ROWS = [
    DataObject(
        uuid=deterministic_uuid("text_three_array_two"),
        properties={
            "tag": "text_three_array_two",
            "textField": "abc",
            "searchOnlyText": "abcd",
            "textArray": ["red", "blue"],
            "intArray": [1, 2],
            "noFilterIntArray": [10, 11],
        },
    ),
    DataObject(
        uuid=deterministic_uuid("text_empty_array_empty"),
        properties={
            "tag": "text_empty_array_empty",
            "textField": "",
            "searchOnlyText": "",
            "textArray": [],
            "intArray": [],
            "noFilterIntArray": [],
        },
    ),
    DataObject(
        uuid=deterministic_uuid("text_null_array_null"),
        properties={
            "tag": "text_null_array_null",
            "textField": None,
            "searchOnlyText": None,
            "textArray": None,
            "intArray": None,
            "noFilterIntArray": None,
        },
    ),
    DataObject(
        uuid=deterministic_uuid("text_missing_array_missing"),
        properties={
            "tag": "text_missing_array_missing",
        },
    ),
    DataObject(
        uuid=deterministic_uuid("text_one_array_one"),
        properties={
            "tag": "text_one_array_one",
            "textField": "x",
            "searchOnlyText": "z",
            "textArray": ["solo"],
            "intArray": [7],
            "noFilterIntArray": [5],
        },
    ),
]


def create_collection(client, name: str):
    return client.collections.create(
        name=name,
        properties=[
            Property(name="tag", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="textField", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(
                name="searchOnlyText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=False,
                index_searchable=True,
            ),
            Property(name="textArray", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="intArray", data_type=DataType.INT_ARRAY, index_filterable=True),
            Property(name="noFilterIntArray", data_type=DataType.INT_ARRAY, index_filterable=False),
        ],
        inverted_index_config=Configure.inverted_index(index_null_state=True, index_property_length=True),
    )


def insert_rows(collection):
    result = collection.data.insert_many(ROWS)
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


def print_observation(name, note, outcome):
    print(name)
    print(f"  note:    {note}")
    print(f"  observed:{outcome}")
    print("  result:  OBSERVED")
    return True


def cleanup_collection(client, name: str):
    try:
        if client.collections.exists(name):
            client.collections.delete(name)
            print(f"Cleaned up collection: {name}")
    except Exception as exc:
        print(f"Cleanup warning for {name}: {exc}")


def main():
    collection_name = f"{COLLECTION_PREFIX}{int(time.time())}{uuid.uuid4().hex[:8]}"
    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    created = False
    try:
        collection = create_collection(client, collection_name)
        created = True
        insert_rows(collection)

        checks = []
        checks.append(
            print_check(
                "text_length_equal_three",
                "The checked docs describe property length filtering through a `len(property)` / `length=True` metadata path.",
                "Current oracle maps TEXT property length to Python string length for present non-null strings.",
                ["text_three_array_two"],
                fetch_tags(collection, Filter.by_property("textField", length=True).equal(3)),
            )
        )
        checks.append(
            print_check(
                "text_length_greater_than_zero",
                "The checked docs show comparison operators over property length.",
                "Current oracle treats non-empty TEXT values as length greater than zero and excludes null/missing/empty values.",
                ["text_one_array_one", "text_three_array_two"],
                fetch_tags(collection, Filter.by_property("textField", length=True).greater_than(0)),
            )
        )
        checks.append(
            print_check(
                "text_length_equal_zero",
                "The checked docs do not provide a null/missing/empty-string truth table for property length.",
                "Current oracle treats explicit empty strings, explicit nulls, and missing properties as length zero for TEXT property length.",
                ["text_empty_array_empty", "text_missing_array_missing", "text_null_array_null"],
                fetch_tags(collection, Filter.by_property("textField", length=True).equal(0)),
            )
        )
        checks.append(
            print_check(
                "text_array_length_equal_two",
                "The Python-client error message for empty-list filtering points to property-length filtering for list length.",
                "Current oracle maps array property length to element count for present arrays.",
                ["text_three_array_two"],
                fetch_tags(collection, Filter.by_property("textArray", length=True).equal(2)),
            )
        )
        checks.append(
            print_check(
                "text_array_length_equal_zero",
                "The checked docs do not provide a null/missing/empty-array truth table for property length.",
                "Current oracle treats explicit empty arrays, explicit nulls, and missing properties as length zero for array property length.",
                ["text_empty_array_empty", "text_missing_array_missing", "text_null_array_null"],
                fetch_tags(collection, Filter.by_property("textArray", length=True).equal(0)),
            )
        )
        checks.append(
            print_check(
                "int_array_length_less_or_equal_one",
                "The checked docs describe property length generically rather than as a text-only operator.",
                "Current oracle maps INT_ARRAY property length to element count and treats null/missing as length zero.",
                ["text_empty_array_empty", "text_missing_array_missing", "text_null_array_null", "text_one_array_one"],
                fetch_tags(collection, Filter.by_property("intArray", length=True).less_or_equal(1)),
            )
        )
        checks.append(
            print_check(
                "text_length_and_property_filter_composition",
                "The checked filter docs allow boolean composition around filters.",
                "Current oracle intersects property-length masks with ordinary property masks.",
                ["text_three_array_two"],
                fetch_tags(
                    collection,
                    Filter.by_property("textField", length=True).greater_or_equal(3)
                    & Filter.by_property("tag").equal("text_three_array_two"),
                ),
            )
        )
        checks.append(
            print_check(
                "search_only_text_length_greater_than_zero",
                "The checked docs say property-length filtering requires the property length index, but do not state a per-property filterable requirement.",
                "Current oracle treats locally queryable search-only TEXT length exactly like normal TEXT length.",
                ["text_one_array_one", "text_three_array_two"],
                fetch_tags(collection, Filter.by_property("searchOnlyText", length=True).greater_than(0)),
            )
        )
        checks.append(
            print_check(
                "search_only_text_length_equal_zero",
                "The checked docs do not provide a null/missing/empty truth table for search-only TEXT property length.",
                "Current oracle treats empty, null, and missing search-only TEXT values as length zero when the collection indexes property length.",
                ["text_empty_array_empty", "text_missing_array_missing", "text_null_array_null"],
                fetch_tags(collection, Filter.by_property("searchOnlyText", length=True).equal(0)),
            )
        )
        try:
            observed = fetch_tags(collection, Filter.by_property("noFilterIntArray", length=True).greater_than(0))
            checks.append(
                print_observation(
                    "no_filter_int_array_length_greater_than_zero_observation",
                    "The checked docs do not state whether non-filterable arrays are length-queryable when `index_property_length=True`.",
                    observed,
                )
            )
        except Exception as exc:
            checks.append(
                print_observation(
                    "no_filter_int_array_length_greater_than_zero_observation",
                    "Local Weaviate returned a query error for non-filterable INT_ARRAY property length, so this broader subset is not claimed.",
                    str(exc),
                )
            )
        checks.append(
            print_check(
                "not_text_array_length_equal_two",
                "The checked filter docs allow NOT composition around filters.",
                "Current oracle complements the property-length equality mask.",
                ["text_empty_array_empty", "text_missing_array_missing", "text_null_array_null", "text_one_array_one"],
                fetch_tags(collection, Filter.not_(Filter.by_property("textArray", length=True).equal(2))),
            )
        )

        if all(checks):
            print("Summary: all validated property_length checks passed on the local Weaviate service.")
            return 0

        print("Summary: one or more validated property_length checks failed.")
        return 1
    finally:
        if created:
            cleanup_collection(client, collection_name)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
