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
COLLECTION_PREFIX = "EqualOperatorValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-equal-operator-{label}"))


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
            Property(name="intVal", data_type=DataType.INT, index_filterable=True),
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
            "int_equal",
            "Equal is listed as a where-filter operator with valueInt for int properties.",
            "Current fuzzer maps INT equal to exact pandas equality on present values.",
            ["a", "c"],
            fetch_tags(collection, Filter.by_property("intVal").equal(10)),
        ))

        checks.append(print_check(
            "bool_equal",
            "Equal is listed as a where-filter operator with valueBoolean for boolean properties.",
            "Current fuzzer maps BOOL equal to exact pandas equality on present values.",
            ["a", "c"],
            fetch_tags(collection, Filter.by_property("boolVal").equal(True)),
        ))

        checks.append(print_check(
            "field_tokenized_text_equal_case_sensitive",
            "Docs say Equal behavior on multi-word text depends on property tokenization.",
            "Current fuzzer sets Tokenization.FIELD, so text equality is modeled as exact case-sensitive string equality.",
            ["a"],
            fetch_tags(collection, Filter.by_property("textField").equal("Alpha Beta")),
        ))

        checks.append(print_check(
            "field_tokenized_text_equal_case_mismatch",
            "Docs say text Equal depends on tokenization; case behavior is therefore not generic official equality semantics.",
            "For the fuzzer's Tokenization.FIELD subset, case-mismatched strings should not match.",
            ["b"],
            fetch_tags(collection, Filter.by_property("textField").equal("alpha beta")),
        ))

        checks.append(print_check(
            "nullable_int_equal_excludes_null_missing",
            "Docs list Equal as a property predicate but do not give an equality-specific null/missing truth table.",
            "Current fuzzer treats null or omitted scalar values as non-matches for positive equality.",
            ["a"],
            fetch_tags(collection, Filter.by_property("nullableInt").equal(1)),
        ))

        checks.append(print_check(
            "nullable_text_equal_excludes_null_missing_empty_string",
            "Official null-state docs say empty strings are null-equivalent for IsNull, but they do not define Equal('') as a generated fuzzer case.",
            "Current fuzzer excludes empty strings from candidate text equality values and treats only exact present non-empty strings as positive equality matches.",
            ["a"],
            fetch_tags(collection, Filter.by_property("nullableText").equal("red")),
        ))

        checks.append(print_check(
            "date_equal_utc_normalized_instant",
            "Docs state date filters use RFC3339 date strings and datetimes can be filtered similarly to numbers.",
            "Current fuzzer normalizes DATE values to UTC timestamps for equality comparison.",
            ["a", "b"],
            fetch_tags(collection, Filter.by_property("dateVal").equal("2024-02-29T15:59:59Z")),
        ))

        checks.append(print_check(
            "number_equal_regular_value",
            "Equal is listed with valueNumber for number properties.",
            "Regular finite NUMBER equality is locally validated but intentionally outside the main positive atomic generator to avoid broad float-exactness claims.",
            ["c"],
            fetch_tags(collection, Filter.by_property("numberVal").equal(1.5)),
        ))

        signed_zero_mismatches = []
        for name, expected, actual in [
            (
                "number_equal_positive_zero_signed_zero_probe",
                ["a", "b"],
                fetch_tags(collection, Filter.by_property("numberVal").equal(0.0)),
            ),
            (
                "number_equal_negative_zero_signed_zero_probe",
                ["a", "b"],
                fetch_tags(collection, Filter.by_property("numberVal").equal(-0.0)),
            ),
            (
                "number_approx_zero_range_signed_zero_probe",
                ["a", "b"],
                fetch_tags(
                    collection,
                    Filter.by_property("numberVal").greater_or_equal(-2e-323)
                    & Filter.by_property("numberVal").less_or_equal(2e-323),
                ),
            ),
            (
                "number_greater_or_equal_zero_signed_zero_probe",
                ["a", "b", "c"],
                fetch_tags(collection, Filter.by_property("numberVal").greater_or_equal(0.0)),
            ),
            (
                "number_less_or_equal_zero_signed_zero_probe",
                ["a", "b", "d"],
                fetch_tags(collection, Filter.by_property("numberVal").less_or_equal(0.0)),
            ),
        ]:
            ok = print_mismatch_probe(
                name,
                expected,
                actual,
                "This follows local historical issue10917's signed-zero consistency expectation; mismatches are recorded as engine-side observations, not as failures of the current main fuzzer subset.",
            )
            if not ok:
                signed_zero_mismatches.append(name)

        print_observation(
            "word_tokenized_text_equal_multi_word_observation",
            "Official docs say Equal on multi-word text depends on tokenization; the main fuzzer uses Tokenization.FIELD, so WORD-tokenization behavior is observed here but not included in the current oracle subset.",
            fetch_tags(collection, Filter.by_property("wordText").equal("Alpha Beta")),
        )

        print_observation(
            "empty_string_equal_observation",
            "The main fuzzer excludes empty strings from generated text equality values because null-state docs treat empty strings as null-equivalent; this direct Equal('') probe is observation-only.",
            fetch_tags(collection, Filter.by_property("nullableText").equal("")),
        )

        if signed_zero_mismatches:
            print(f"Signed-zero mismatch probes: {signed_zero_mismatches}")

        if all(checks):
            print("Summary: all Equal operator checks passed on the local Weaviate service.")
            if signed_zero_mismatches:
                print("Summary note: signed-zero NUMBER equality/range mismatches were observed outside the current main fuzzer subset.")
            return 0

        print("Summary: one or more Equal operator checks failed.")
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
