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
COLLECTION_PREFIX = "NotOperatorValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-not-operator-{label}"))


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "round": "Double",
            "answer": "Mexico",
            "points": 100,
            "nullableScore": 10,
            "optionalText": "red",
            "tagsArray": [1, 2],
            "metaObj": {"price": 100},
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={
            "tag": "b",
            "round": "Double",
            "answer": "Yucatan",
            "points": 400,
            "nullableScore": None,
            "optionalText": None,
            "tagsArray": None,
            "metaObj": None,
        },
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={
            "tag": "c",
            "round": "Jeopardy",
            "answer": "Bird",
            "points": 800,
            "nullableScore": 30,
            "optionalText": "blue",
            "tagsArray": [3],
            "metaObj": {"price": 300},
        },
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={
            "tag": "d",
            "round": "Double",
            "answer": "Bird",
            "points": 200,
        },
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("e"),
        properties={
            "tag": "e",
            "round": "Double",
            "answer": "Fish",
            "points": 500,
            "nullableScore": 5,
            "optionalText": "green",
            "tagsArray": [],
            "metaObj": {"price": 500},
        },
        vector=[0.5, 0.0, 0.5],
    ),
    DataObject(
        uuid=deterministic_uuid("f"),
        properties={
            "tag": "f",
            "round": "Double",
            "answer": "EmptyText",
            "points": 600,
            "nullableScore": 0,
            "optionalText": "",
            "tagsArray": [4],
            "metaObj": {},
        },
        vector=[0.2, 0.2, 0.2],
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
                name="round",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="answer",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="points",
                data_type=DataType.INT,
                index_filterable=True,
                index_range_filterable=True,
            ),
            Property(
                name="nullableScore",
                data_type=DataType.INT,
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
                name="metaObj",
                data_type=DataType.OBJECT,
                index_filterable=True,
                index_null_state=True,
                nested_properties=[
                    Property(name="price", data_type=DataType.INT),
                ],
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

        scalar_not_equal = Filter.not_(Filter.by_property("round").equal("Double"))
        checks.append(print_check(
            "not_scalar_text_equal",
            "NOT negates one filter condition.",
            "Oracle maps NOT(round == 'Double') to the complement of the generated equality mask.",
            ["c"],
            fetch_tags(collection, scalar_not_equal),
        ))

        numeric_not_cmp = Filter.not_(Filter.by_property("nullableScore").greater_than(25))
        checks.append(print_check(
            "not_numeric_comparison_with_null_or_missing",
            "Docs document NOT composition but do not fully specify scalar null/missing truth tables.",
            "Current fuzzer oracle treats null or omitted scalar values as included by NOT over an ordinary positive comparison.",
            ["a", "b", "d", "e", "f"],
            fetch_tags(collection, numeric_not_cmp),
        ))

        text_not_equal = Filter.not_(Filter.by_property("optionalText").equal("red"))
        checks.append(print_check(
            "not_text_equal_with_null_or_missing",
            "Docs document NOT composition but do not fully specify text null/missing truth tables.",
            "Current fuzzer oracle treats null or omitted text values as included by NOT over equality.",
            ["b", "c", "d", "e", "f"],
            fetch_tags(collection, text_not_equal),
        ))

        range_not = Filter.not_(
            Filter.by_property("points").greater_or_equal(200)
            & Filter.by_property("points").less_or_equal(500)
        )
        checks.append(print_check(
            "not_range_compound",
            "NOT may wrap a nested filter expression.",
            "Oracle negates the range conjunction over present numeric values.",
            ["a", "c", "f"],
            fetch_tags(collection, range_not),
        ))

        contains_not = Filter.not_(Filter.by_property("tagsArray").contains_any([2]))
        checks.append(print_check(
            "not_array_contains_any_with_null_or_missing",
            "Docs document NOT composition; array null/missing behavior is operand-level behavior.",
            "Current fuzzer oracle treats null, omitted, and empty arrays as included by NOT over contains_any.",
            ["b", "c", "d", "e", "f"],
            fetch_tags(collection, contains_not),
        ))

        text_is_null = Filter.by_property("optionalText").is_none(True)
        checks.append(print_check(
            "text_is_none_true_null_missing_empty_string",
            "Official null-state docs say empty strings are equivalent to null values.",
            "Fuzzer null masks treat None, omitted text, and empty string as effective null.",
            ["b", "d", "f"],
            fetch_tags(collection, text_is_null),
        ))

        text_is_not_null = Filter.by_property("optionalText").is_none(False)
        checks.append(print_check(
            "text_is_none_false_excludes_empty_string",
            "Official null-state docs say empty strings are equivalent to null values.",
            "Fuzzer null masks therefore exclude empty strings from is_none(False).",
            ["a", "c", "e"],
            fetch_tags(collection, text_is_not_null),
        ))

        text_not_is_null = Filter.not_(Filter.by_property("optionalText").is_none(True))
        checks.append(print_check(
            "not_text_is_none_true_excludes_empty_string",
            "NOT over text null-state should preserve the documented empty-string-as-null boundary in the tested subset.",
            "Oracle maps NOT(is_none(True)) to the non-null/non-empty text rows.",
            ["a", "c", "e"],
            fetch_tags(collection, text_not_is_null),
        ))

        text_not_is_not_null = Filter.not_(Filter.by_property("optionalText").is_none(False))
        checks.append(print_check(
            "not_text_is_none_false_includes_empty_string",
            "NOT over text non-null state should return the documented null-equivalent rows in the tested subset.",
            "Oracle maps NOT(is_none(False)) to null, omitted, and empty-string text rows.",
            ["b", "d", "f"],
            fetch_tags(collection, text_not_is_not_null),
        ))

        array_is_null = Filter.by_property("tagsArray").is_none(True)
        checks.append(print_check(
            "array_is_none_true_null_missing_empty_array",
            "Official null-state docs say zero-length arrays are equivalent to null values.",
            "Fuzzer data generation normalizes empty arrays to None, so the oracle treats null, omitted, and empty arrays as effective null.",
            ["b", "d", "e"],
            fetch_tags(collection, array_is_null),
        ))

        array_is_not_null = Filter.by_property("tagsArray").is_none(False)
        checks.append(print_check(
            "array_is_none_false_excludes_empty_array",
            "Official null-state docs say zero-length arrays are equivalent to null values.",
            "Oracle maps is_none(False) to arrays with at least one element.",
            ["a", "c", "f"],
            fetch_tags(collection, array_is_not_null),
        ))

        array_not_is_null = Filter.not_(Filter.by_property("tagsArray").is_none(True))
        checks.append(print_check(
            "not_array_is_none_true_excludes_empty_array",
            "NOT over array null-state should preserve the documented empty-array-as-null boundary in the tested subset.",
            "Oracle maps NOT(is_none(True)) to arrays with at least one element.",
            ["a", "c", "f"],
            fetch_tags(collection, array_not_is_null),
        ))

        array_not_is_not_null = Filter.not_(Filter.by_property("tagsArray").is_none(False))
        checks.append(print_check(
            "not_array_is_none_false_includes_empty_array",
            "NOT over array non-null state should return the documented null-equivalent rows in the tested subset.",
            "Oracle maps NOT(is_none(False)) to null, omitted, and empty-array rows.",
            ["b", "d", "e"],
            fetch_tags(collection, array_not_is_not_null),
        ))

        not_is_null = Filter.not_(Filter.by_property("nullableScore").is_none(True))
        checks.append(print_check(
            "not_is_none_true_scalar",
            "NOT negates the scalar null-state filter.",
            "Oracle maps NOT(is_none(True)) to the non-null/non-missing scalar rows.",
            ["a", "c", "e", "f"],
            fetch_tags(collection, not_is_null),
        ))

        object_not_is_null = Filter.not_(Filter.by_property("metaObj").is_none(True))
        checks.append(print_check(
            "not_is_none_true_object",
            "OBJECT null filtering is a separate operand-level behavior; issue10642 treats direct OBJECT is_none(False) as an unimplemented or unsupported boundary.",
            "Current oracle treats NOT(OBJECT is_none(True)) as non-null/non-missing OBJECT rows for the validated subset.",
            ["a", "c", "e", "f"],
            fetch_tags(collection, object_not_is_null),
        ))

        double_not_null = Filter.not_(Filter.not_(Filter.by_property("nullableScore").is_none(True)))
        checks.append(print_check(
            "double_not_is_none_true_scalar",
            "Double NOT should preserve the wrapped filter over the tested subset.",
            "PQS and equivalence-style checks rely on double NOT preserving the selected generated predicate.",
            ["b", "d"],
            fetch_tags(collection, double_not_null),
        ))

        not_or_filter = Filter.not_(
            Filter.by_property("answer").equal("Bird")
            | Filter.by_property("points").less_than(150)
        )
        checks.append(print_check(
            "not_over_or_compound",
            "NOT may wrap a nested OR expression.",
            "Oracle applies complement after the OR mask is computed.",
            ["b", "e", "f"],
            fetch_tags(collection, not_or_filter),
        ))

        object_is_null = Filter.by_property("metaObj").is_none(True)
        checks.append(print_check(
            "object_is_none_true_null_missing_not_empty_object",
            "Docs cover null-state generally; OBJECT empty-object behavior is not specified by the checked pages.",
            "The fuzzer currently treats None or omitted OBJECT as null and a Python dict, including an empty dict, as non-null.",
            ["b", "d"],
            fetch_tags(collection, object_is_null),
        ))

        print_observation(
            "direct_object_is_none_false_observation",
            "Direct OBJECT is_none(False) is tracked as an unimplemented or unsupported feature boundary by issue10642, so this observation is not used as a passing oracle assertion.",
            fetch_tags(collection, Filter.by_property("metaObj").is_none(False)),
        )

        if all(checks):
            print("Summary: all NOT operator checks passed on the local Weaviate service.")
            return 0

        print("Summary: one or more NOT operator checks failed.")
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
