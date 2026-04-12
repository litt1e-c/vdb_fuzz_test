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
COLLECTION_PREFIX = "ContainsAllOperatorValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-contains-all-operator-{label}"))


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "tagsArray": [1, 2],
            "scoresArray": [-1.0, 1.5],
            "flagsArray": [True],
            "labelsArray": ["Alpha", "Beta"],
            "fieldText": "Alpha Beta",
            "nullableFieldText": "Alpha Beta",
            "wordText": "Alpha Beta",
            "searchOnlyText": "Alpha Beta",
            "nullableTextArray": ["red", "blue"],
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={
            "tag": "b",
            "tagsArray": [3],
            "scoresArray": [0.0],
            "flagsArray": [False],
            "labelsArray": ["alpha"],
            "fieldText": "alpha beta",
            "nullableFieldText": None,
            "wordText": "alpha beta",
            "searchOnlyText": "alpha beta",
            "nullableTextArray": None,
        },
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={
            "tag": "c",
            "tagsArray": [2, 2],
            "scoresArray": [2.0, 3.0],
            "flagsArray": [True, False],
            "labelsArray": ["Alpha-Beta"],
            "fieldText": "Alpha-Beta",
            "wordText": "Alpha-Beta",
            "searchOnlyText": "Alpha-Beta",
        },
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={
            "tag": "d",
            "tagsArray": [9],
            "scoresArray": [-4.0],
            "flagsArray": [False],
            "labelsArray": ["Gamma"],
            "fieldText": "Alpha",
            "nullableFieldText": "Alpha",
            "wordText": "Alpha",
            "searchOnlyText": "Alpha",
            "nullableTextArray": ["blue"],
        },
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("e"),
        properties={
            "tag": "e",
            "tagsArray": [7],
            "scoresArray": [10.0],
            "flagsArray": [False],
            "labelsArray": [""],
            "fieldText": "",
            "nullableFieldText": "",
            "wordText": "",
            "searchOnlyText": "",
            "nullableTextArray": ["green"],
        },
        vector=[0.5, 0.0, 0.5],
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
                name="tagsArray",
                data_type=DataType.INT_ARRAY,
                index_filterable=True,
            ),
            Property(
                name="scoresArray",
                data_type=DataType.NUMBER_ARRAY,
                index_filterable=True,
            ),
            Property(
                name="flagsArray",
                data_type=DataType.BOOL_ARRAY,
                index_filterable=True,
            ),
            Property(
                name="labelsArray",
                data_type=DataType.TEXT_ARRAY,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="fieldText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="nullableFieldText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="wordText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="searchOnlyText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=False,
                index_searchable=True,
            ),
            Property(
                name="nullableTextArray",
                data_type=DataType.TEXT_ARRAY,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def fetch_tags(collection, flt):
    response = collection.query.fetch_objects(filters=flt, limit=30)
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
            "int_array_contains_all",
            "ContainsAll is documented for array and text properties and returns rows containing all supplied values.",
            "Current fuzzer models INT_ARRAY contains_all as Python all-membership over present lists.",
            ["a"],
            fetch_tags(collection, Filter.by_property("tagsArray").contains_all([1, 2])),
        ))

        checks.append(print_check(
            "number_array_contains_all",
            "ContainsAll is documented for array and text properties and returns rows containing all supplied values.",
            "Current fuzzer models NUMBER_ARRAY contains_all as exact float all-membership over present lists.",
            ["c"],
            fetch_tags(collection, Filter.by_property("scoresArray").contains_all([2.0, 3.0])),
        ))

        checks.append(print_check(
            "bool_array_contains_all",
            "ContainsAll is documented for array and text properties and returns rows containing all supplied values.",
            "Current fuzzer models BOOL_ARRAY contains_all as boolean all-membership over present lists.",
            ["c"],
            fetch_tags(collection, Filter.by_property("flagsArray").contains_all([True, False])),
        ))

        checks.append(print_check(
            "text_array_contains_all_field_exact_case_sensitive",
            "The docs say text filtering is tokenization-dependent; FIELD tokenization keeps exact values as tokens.",
            "Current TEXT_ARRAY oracle uses FIELD tokenization and therefore exact case-sensitive element membership for every requested target.",
            ["a"],
            fetch_tags(collection, Filter.by_property("labelsArray").contains_all(["Alpha", "Beta"])),
        ))

        checks.append(print_check(
            "text_array_contains_all_case_sensitive_negative",
            "ContainsAll requires that all listed values be present under the property's tokenization behavior.",
            "Current FIELD-tokenized TEXT_ARRAY oracle is exact and case-sensitive, so no row contains both `Alpha` and `alpha`.",
            [],
            fetch_tags(collection, Filter.by_property("labelsArray").contains_all(["Alpha", "alpha"])),
        ))

        checks.append(print_check(
            "field_text_contains_all_exact_full_value",
            "The docs say ContainsAll works on text by treating the property as an array of tokens according to tokenization.",
            "For the fuzzer's FIELD-tokenized scalar TEXT subset, the full string is treated as one case-sensitive token, so a single-target contains_all matches exact whole-field values.",
            ["a"],
            fetch_tags(collection, Filter.by_property("fieldText").contains_all(["Alpha Beta"])),
        ))

        checks.append(print_check(
            "field_text_contains_all_field_token_not_split",
            "For FIELD tokenization, the whole text value is one token rather than separate word tokens.",
            "Current conservative scalar-TEXT oracle therefore treats `contains_all([\"Alpha\", \"Beta\"])` as a non-match for FIELD-tokenized text.",
            [],
            fetch_tags(collection, Filter.by_property("fieldText").contains_all(["Alpha", "Beta"])),
        ))

        checks.append(print_check(
            "nullable_text_array_contains_all_excludes_null_missing",
            "The checked docs define ContainsAll for text and array properties but do not provide a null or missing truth table.",
            "Current fuzzer treats null and omitted arrays as non-matches for positive contains_all.",
            ["a"],
            fetch_tags(collection, Filter.by_property("nullableTextArray").contains_all(["red"])),
        ))

        checks.append(print_check(
            "nullable_field_text_contains_all_excludes_null_missing",
            "The checked docs define ContainsAll for text properties but do not provide a null or missing truth table.",
            "Current fuzzer treats null and omitted scalar FIELD text as non-matches for positive contains_all.",
            ["a"],
            fetch_tags(collection, Filter.by_property("nullableFieldText").contains_all(["Alpha Beta"])),
        ))

        search_only_cfg = next(
            (prop for prop in client.collections.get(collection_name).config.get().properties if prop.name == "searchOnlyText"),
            None,
        )
        try:
            search_only_result = fetch_tags(
                collection,
                Filter.by_property("searchOnlyText").contains_all(["Alpha Beta"]),
            )
            print_observation(
                "search_only_text_contains_all_observation",
                "This probes whether scalar TEXT ContainsAll can run when `index_filterable` is disabled but `index_searchable` is enabled. The checked docs do not state this operator-specific requirement precisely.",
                {
                    "property_config": str(search_only_cfg),
                    "query_result": search_only_result,
                },
            )
        except Exception as exc:
            print_observation(
                "search_only_text_contains_all_observation",
                "This probes whether scalar TEXT ContainsAll can run when `index_filterable` is disabled but `index_searchable` is enabled. The checked docs do not state this operator-specific requirement precisely.",
                {
                    "property_config": str(search_only_cfg),
                    "query_error": str(exc),
                },
            )

        word_text_result = fetch_tags(collection, Filter.by_property("wordText").contains_all(["alpha", "beta"]))
        print_observation(
            "word_text_contains_all_observation",
            "The docs say text is treated as tokens according to the tokenization scheme. This records local WORD-tokenized behavior without elevating it to the current main oracle subset.",
            {
                "query": 'Filter.by_property("wordText").contains_all(["alpha", "beta"])',
                "query_result": word_text_result,
            },
        )

        try:
            empty_result = collection.data.insert(
                uuid=deterministic_uuid("empty_probe"),
                properties={
                    "tag": "empty_probe",
                    "nullableTextArray": [],
                },
                vector=[0.2, 0.2, 0.2],
            )
            empty_probe_fetch = collection.query.fetch_objects(
                filters=Filter.by_property("tag").equal("empty_probe"),
                limit=5,
            )
            empty_probe_query = fetch_tags(
                collection,
                Filter.by_property("nullableTextArray").contains_all(["red"]),
            )
            print_observation(
                "empty_array_insert_observation",
                "This probes how the local client/server path handles an explicit empty text array. The current fuzzer normalizes empty arrays to null and does not rely on direct empty-array semantics in the main oracle subset.",
                {
                    "insert_result": str(empty_result),
                    "fetched_properties": [obj.properties for obj in empty_probe_fetch.objects],
                    "contains_all_after_insert": empty_probe_query,
                },
            )
        except Exception as exc:
            print_observation(
                "empty_array_insert_observation",
                "This probes how the local client/server path handles an explicit empty text array. The current fuzzer normalizes empty arrays to null and does not rely on direct empty-array semantics in the main oracle subset.",
                {
                    "insert_error": str(exc),
                },
            )

        if all(checks):
            print("Summary: all contains_all operator checks passed on the local Weaviate service.")
            print("Summary note: WORD tokenization, searchable-only scalar text, and explicit empty-array insertion were recorded as local observations only.")
            return 0

        print("Summary: one or more contains_all operator checks failed.")
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
