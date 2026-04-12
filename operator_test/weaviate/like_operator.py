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
COLLECTION_PREFIX = "LikeOperatorValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-like-operator-{label}"))


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "fieldText": "Alpha Beta",
            "wordText": "Alpha Beta",
            "filterOnlyText": "Alpha Beta",
            "searchOnlyText": "Alpha Beta",
            "nullableText": "red",
            "literalText": "car*",
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={
            "tag": "b",
            "fieldText": "alpha beta",
            "wordText": "alpha beta",
            "filterOnlyText": "alpha beta",
            "searchOnlyText": "alpha beta",
            "nullableText": None,
            "literalText": "car",
        },
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={
            "tag": "c",
            "fieldText": "Alpha-Beta",
            "wordText": "Alpha-Beta",
            "filterOnlyText": "Alpha-Beta",
            "searchOnlyText": "Alpha-Beta",
            "nullableText": "",
            "literalText": "care",
        },
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={
            "tag": "d",
            "fieldText": "Alpha",
            "wordText": "Alpha",
            "filterOnlyText": "Alpha",
            "searchOnlyText": "Alpha",
            "literalText": "car?",
        },
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("e"),
        properties={
            "tag": "e",
            "fieldText": "BetaAlpha",
            "wordText": "BetaAlpha",
            "filterOnlyText": "BetaAlpha",
            "searchOnlyText": "BetaAlpha",
            "nullableText": "rose",
            "literalText": "scar",
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
                name="fieldText",
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
                name="filterOnlyText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
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
                name="nullableText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
            Property(
                name="literalText",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
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
            "field_like_prefix_case_sensitive",
            "Docs say Like supports `*` wildcards for partial text matches, and field tokenization treats the whole property value as one token.",
            "Current fuzzer models Tokenization.FIELD like as case-sensitive wildcard matching against the full stored string.",
            ["a", "c", "d"],
            fetch_tags(collection, Filter.by_property("fieldText").like("Alpha*")),
        ))

        checks.append(print_check(
            "field_like_suffix_case_sensitive",
            "Docs say Like supports suffix-style matching with `*`.",
            "Current fuzzer models FIELD suffix patterns as case-sensitive full-string wildcard matches.",
            ["a", "c"],
            fetch_tags(collection, Filter.by_property("fieldText").like("*Beta")),
        ))

        checks.append(print_check(
            "field_like_contains_case_sensitive",
            "Docs say Like supports infix matching such as `*car*`.",
            "Current fuzzer models FIELD infix patterns as case-sensitive substring matches on the full stored string.",
            ["a"],
            fetch_tags(collection, Filter.by_property("fieldText").like("*ha B*")),
        ))

        checks.append(print_check(
            "field_like_single_character",
            "Docs say `?` matches exactly one unknown character.",
            "Current fuzzer models single-character wildcards against the full FIELD-tokenized string.",
            ["d"],
            fetch_tags(collection, Filter.by_property("fieldText").like("Alph?")),
        ))

        checks.append(print_check(
            "filter_only_like_without_searchable",
            "Official docs describe Like as a text filter and do not say that BM25-style searchable indexing is required.",
            "Current fuzzer uses FIELD text properties with index_filterable enabled; this check validates the locally used subset where index_searchable is disabled.",
            ["a", "c", "d"],
            fetch_tags(collection, Filter.by_property("filterOnlyText").like("Alpha*")),
        ))

        checks.append(print_check(
            "nullable_text_like_excludes_null_missing_empty_for_nonempty_prefix",
            "The docs define Like wildcard semantics but do not provide a Like-specific null or empty-string truth table.",
            "Current fuzzer treats null, missing, and empty text values as outside the positive FIELD-like subset for non-empty patterns.",
            ["a", "e"],
            fetch_tags(collection, Filter.by_property("nullableText").like("r*")),
        ))

        checks.append(print_check(
            "literal_wildcard_limitation_prefix",
            "Official docs explicitly say Like cannot match `*` or `?` as literal wildcard characters.",
            "Current oracle therefore does not attempt literal-wildcard exactness claims; `car*` is treated as a wildcard pattern, not a literal string.",
            ["a", "b", "c", "d"],
            fetch_tags(collection, Filter.by_property("literalText").like("car*")),
        ))

        checks.append(print_check(
            "literal_wildcard_limitation_single_char",
            "Official docs explicitly say Like cannot match `*` or `?` as literal wildcard characters.",
            "Current oracle treats `?` as a wildcard, so `car?` matches any four-character `carX` string including literal wildcard characters stored in the data.",
            ["a", "c", "d"],
            fetch_tags(collection, Filter.by_property("literalText").like("car?")),
        ))

        cfg = client.collections.get(collection_name).config.get()
        search_only_cfg = next((prop for prop in cfg.properties if prop.name == "searchOnlyText"), None)
        try:
            search_only_result = fetch_tags(collection, Filter.by_property("searchOnlyText").like("Alpha*"))
            print_observation(
                "search_only_like_observation",
                "This probes whether Like can run on a property with index_filterable disabled but index_searchable enabled. It is observation-only because the official docs do not define this requirement precisely for Like.",
                {
                    "property_config": str(search_only_cfg),
                    "query_result": search_only_result,
                },
            )
        except Exception as exc:
            print_observation(
                "search_only_like_observation",
                "This probes whether Like can run on a property with index_filterable disabled but index_searchable enabled. It is observation-only because the official docs do not define this requirement precisely for Like.",
                {
                    "property_config": str(search_only_cfg),
                    "query_error": str(exc),
                },
            )

        word_result = fetch_tags(collection, Filter.by_property("wordText").like("alpha*"))
        print_observation(
            "word_tokenization_like_observation",
            "The official docs say tokenization influences where filters and that `word` tokenization lowercases and splits on non-alphanumeric characters. This result is recorded as local behavior rather than a hard oracle claim because the docs do not give a Like-specific token-by-token truth table.",
            {
                "query": 'Filter.by_property("wordText").like("alpha*")',
                "query_result": word_result,
            },
        )

        nullable_star_result = fetch_tags(collection, Filter.by_property("nullableText").like("*"))
        print_observation(
            "empty_string_like_star_observation",
            "The docs say `*` matches zero, one, or more characters, but they do not explain how this interacts with explicit empty strings, nulls, and omitted properties in text filters.",
            {
                "query": 'Filter.by_property("nullableText").like(\"*\")',
                "query_result": nullable_star_result,
            },
        )

        if all(checks):
            print("Summary: all like operator checks passed on the local Weaviate service.")
            print("Summary note: tokenization, searchable-only indexing, and empty-string wildcard behavior were recorded as local observations only.")
            return 0

        print("Summary: one or more like operator checks failed.")
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
