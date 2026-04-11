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
COLLECTION_PREFIX = "NotEqualOperatorValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-not-equal-operator-{label}"))


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "textField": "Alpha Beta",
            "wordText": "Alpha Beta",
            "intVal": 10,
            "boolVal": True,
            "numberVal": 0.0,
            "dateVal": "2024-02-29T23:59:59+08:00",
            "nullableText": "red",
            "nullableInt": 1,
            "nullableBool": True,
            "nullableNumber": 1.5,
            "nullableDate": "2024-02-29T23:59:59+08:00",
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={
            "tag": "b",
            "textField": "alpha beta",
            "wordText": "alpha beta",
            "intVal": 20,
            "boolVal": False,
            "numberVal": -0.0,
            "dateVal": "2024-02-29T15:59:59Z",
            "nullableText": None,
            "nullableInt": None,
            "nullableBool": None,
            "nullableNumber": None,
            "nullableDate": None,
        },
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={
            "tag": "c",
            "textField": "Alpha",
            "wordText": "Alpha",
            "intVal": 10,
            "boolVal": True,
            "numberVal": 1.5,
            "dateVal": "2024-03-01T00:00:00Z",
            "nullableText": "",
            "nullableInt": 3,
            "nullableBool": False,
            "nullableNumber": 2.5,
            "nullableDate": "2024-03-01T00:00:00Z",
        },
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={
            "tag": "d",
            "textField": "Alpha-Beta",
            "wordText": "Alpha-Beta",
            "intVal": -10,
            "boolVal": False,
            "numberVal": -1000000.0,
            "dateVal": "2024-02-28T15:59:59Z",
        },
        vector=[0.5, 0.5, 0.0],
    ),
]


def create_collection(client, name):
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
                name="textField",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="wordText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD,
                index_filterable=True,
            ),
            Property(
                name="intVal",
                data_type=DataType.INT,
                index_filterable=True,
                index_range_filterable=True,
            ),
            Property(name="boolVal", data_type=DataType.BOOL, index_filterable=True),
            Property(
                name="numberVal",
                data_type=DataType.NUMBER,
                index_filterable=True,
                index_range_filterable=True,
            ),
            Property(
                name="dateVal",
                data_type=DataType.DATE,
                index_filterable=True,
                index_range_filterable=True,
            ),
            Property(
                name="nullableText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(name="nullableInt", data_type=DataType.INT, index_filterable=True),
            Property(name="nullableBool", data_type=DataType.BOOL, index_filterable=True),
            Property(
                name="nullableNumber",
                data_type=DataType.NUMBER,
                index_filterable=True,
                index_range_filterable=True,
            ),
            Property(
                name="nullableDate",
                data_type=DataType.DATE,
                index_filterable=True,
                index_range_filterable=True,
            ),
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


def print_observation(name, why_observed, actual):
    print(name)
    print(f"  note:    {why_observed}")
    print(f"  observed:{actual}")
    print("  result:  OBSERVED")


def print_mismatch_probe(name, expected_if_consistent, actual, interpretation):
    print(name)
    print(f"  expected_if_consistent:{expected_if_consistent}")
    print(f"  observed:              {actual}")
    ok = actual == expected_if_consistent
    print(f"  interpretation:        {interpretation}")
    print(f"  result:                {'PASS' if ok else 'MISMATCH_OBSERVED'}")
    return ok


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
            "int_not_equal",
            "NotEqual is listed as a where-filter operator with valueInt for int properties.",
            "Current fuzzer maps INT not_equal to pandas != over present values and includes null or omitted rows only when present.",
            ["b", "d"],
            fetch_tags(collection, Filter.by_property("intVal").not_equal(10)),
        ))

        checks.append(print_check(
            "bool_not_equal",
            "NotEqual is listed as a where-filter operator with valueBoolean for boolean properties.",
            "Current fuzzer maps BOOL not_equal to pandas != over present values and includes null or omitted rows only when present.",
            ["b", "d"],
            fetch_tags(collection, Filter.by_property("boolVal").not_equal(True)),
        ))

        checks.append(print_check(
            "number_not_equal_regular_value",
            "NotEqual is listed as a where-filter operator with valueNumber for number properties.",
            "Current fuzzer maps regular finite NUMBER not_equal to pandas !=, while excluding signed -0.0 generation.",
            ["a", "b", "d"],
            fetch_tags(collection, Filter.by_property("numberVal").not_equal(1.5)),
        ))

        checks.append(print_check(
            "field_tokenized_text_not_equal",
            "Docs say Equal behavior on multi-word text depends on property tokenization; NotEqual is listed but not given separate text-tokenization rules.",
            "Current fuzzer uses Tokenization.FIELD for TEXT, so non-empty text NotEqual is modeled as exact case-sensitive string inequality plus null or omitted inclusion.",
            ["b", "c", "d"],
            fetch_tags(collection, Filter.by_property("textField").not_equal("Alpha Beta")),
        ))

        checks.append(print_check(
            "date_not_equal_utc_normalized_instant",
            "Docs list valueDate as an RFC3339 typed value and say date filters can be used similarly to numbers.",
            "A conservative DATE not_equal oracle compares UTC-normalized instants and includes only different present instants for non-null fields.",
            ["c", "d"],
            fetch_tags(collection, Filter.by_property("dateVal").not_equal("2024-02-29T15:59:59Z")),
        ))

        checks.append(print_check(
            "nullable_int_not_equal_includes_null_missing",
            "Docs do not provide a NotEqual-specific null or missing truth table.",
            "Local oracle assumption for direct NotEqual includes null and omitted scalar rows, matching the current fuzzer mask.",
            ["b", "c", "d"],
            fetch_tags(collection, Filter.by_property("nullableInt").not_equal(1)),
        ))

        checks.append(print_check(
            "nullable_bool_not_equal_includes_null_missing",
            "Docs do not provide a NotEqual-specific null or missing truth table.",
            "Local oracle assumption for direct NotEqual includes null and omitted boolean rows, matching the current fuzzer mask.",
            ["b", "c", "d"],
            fetch_tags(collection, Filter.by_property("nullableBool").not_equal(True)),
        ))

        checks.append(print_check(
            "nullable_number_not_equal_includes_null_missing",
            "Docs do not provide a NotEqual-specific null or missing truth table.",
            "Local oracle assumption for direct NotEqual includes null and omitted number rows, matching the current fuzzer mask.",
            ["b", "c", "d"],
            fetch_tags(collection, Filter.by_property("nullableNumber").not_equal(1.5)),
        ))

        checks.append(print_check(
            "nullable_text_not_equal_includes_null_missing_empty_string",
            "Official null-state docs say empty strings are null-equivalent for IsNull, but they do not define NotEqual text behavior.",
            "Current fuzzer excludes empty strings as generated target values; for a non-empty target it treats null, omitted, and empty-string rows as included by NotEqual.",
            ["b", "c", "d"],
            fetch_tags(collection, Filter.by_property("nullableText").not_equal("red")),
        ))

        checks.append(print_check(
            "nullable_date_not_equal_includes_null_missing",
            "Docs do not provide a NotEqual-specific date null or missing truth table.",
            "A conservative DATE not_equal extension includes null and omitted date rows if local behavior matches other scalar NotEqual filters.",
            ["b", "c", "d"],
            fetch_tags(collection, Filter.by_property("nullableDate").not_equal("2024-02-29T15:59:59Z")),
        ))

        direct_text = fetch_tags(collection, Filter.by_property("nullableText").not_equal("red"))
        negated_equal_text = fetch_tags(
            collection,
            Filter.not_(Filter.by_property("nullableText").equal("red")),
        )
        checks.append(print_check(
            "direct_not_equal_matches_not_equal_negation_for_tested_text_subset",
            "NotEqual and Not are separate documented operators; docs do not state a full equivalence law under nulls.",
            "For this tested FIELD-tokenized text subset, direct NotEqual and NOT(Equal) are expected to agree.",
            direct_text,
            negated_equal_text,
        ))

        signed_zero_mismatches = []
        for name, expected, actual in [
            (
                "number_not_equal_positive_zero_signed_zero_probe",
                ["c", "d"],
                fetch_tags(collection, Filter.by_property("numberVal").not_equal(0.0)),
            ),
            (
                "number_not_equal_negative_zero_signed_zero_probe",
                ["c", "d"],
                fetch_tags(collection, Filter.by_property("numberVal").not_equal(-0.0)),
            ),
        ]:
            ok = print_mismatch_probe(
                name,
                expected,
                actual,
                "This follows local historical issue10917's signed-zero consistency expectation; mismatches are recorded as engine-side observations outside the current generated NUMBER subset.",
            )
            if not ok:
                signed_zero_mismatches.append(name)

        print_observation(
            "word_tokenized_text_not_equal_multi_word_observation",
            "Official docs say Equal on multi-word text depends on tokenization; the main fuzzer uses Tokenization.FIELD, so WORD-tokenization NotEqual behavior is observed but not included in the oracle subset.",
            fetch_tags(collection, Filter.by_property("wordText").not_equal("Alpha Beta")),
        )

        print_observation(
            "empty_string_not_equal_observation",
            "The main fuzzer excludes empty strings from generated text target values because null-state docs treat empty strings as null-equivalent; direct NotEqual('') is observation-only.",
            fetch_tags(collection, Filter.by_property("nullableText").not_equal("")),
        )

        if signed_zero_mismatches:
            print(f"Signed-zero mismatch probes: {signed_zero_mismatches}")

        if all(checks):
            print("Summary: all NotEqual operator checks passed on the local Weaviate service.")
            if signed_zero_mismatches:
                print("Summary note: signed-zero NUMBER NotEqual mismatches were observed outside the current generated NUMBER subset.")
            return 0

        print("Summary: one or more NotEqual operator checks failed.")
        return 1
    finally:
        try:
            if created and collection_name.startswith(COLLECTION_PREFIX):
                client.collections.delete(collection_name)
                print(f"Cleaned up collection: {collection_name}")
        finally:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
