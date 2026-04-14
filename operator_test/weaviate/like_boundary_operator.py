import time
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_PREFIX = "LikeBoundaryValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-like-boundary-{label}"))


ROWS = [
    DataObject(uuid=deterministic_uuid("a"), properties={"tag": "a", "textVal": "Alpha"}, vector=[1.0, 0.0, 0.0]),
    DataObject(uuid=deterministic_uuid("b"), properties={"tag": "b", "textVal": "car*"}, vector=[0.0, 1.0, 0.0]),
    DataObject(uuid=deterministic_uuid("c"), properties={"tag": "c", "textVal": "car?"}, vector=[0.0, 0.0, 1.0]),
    DataObject(uuid=deterministic_uuid("d"), properties={"tag": "d", "textVal": "%_"}, vector=[0.5, 0.5, 0.0]),
    DataObject(uuid=deterministic_uuid("e"), properties={"tag": "e", "textVal": ""}, vector=[0.5, 0.0, 0.5]),
    DataObject(uuid=deterministic_uuid("f"), properties={"tag": "f", "textVal": "abc"}, vector=[0.4, 0.3, 0.2]),
    DataObject(uuid=deterministic_uuid("g"), properties={"tag": "g", "textVal": "_"}, vector=[0.2, 0.4, 0.3]),
    DataObject(uuid=deterministic_uuid("h"), properties={"tag": "h", "textVal": "%"}, vector=[0.3, 0.2, 0.4]),
]


def create_collection(client, name):
    return client.collections.create(
        name=name,
        properties=[
            Property(name="tag", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="textVal", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def fetch_tags(collection, pattern):
    response = collection.query.fetch_objects(filters=Filter.by_property("textVal").like(pattern), limit=20)
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
            "like_star_matches_all_present",
            "The documented `*` wildcard matches zero or more characters.",
            "Main oracle can safely treat `like(\"*\")` as matching all present FIELD-text rows, including empty strings.",
            ["a", "b", "c", "d", "e", "f", "g", "h"],
            fetch_tags(collection, "*"),
        ))
        checks.append(print_check(
            "like_double_star_matches_all_present",
            "Repeated `*` wildcards are accepted locally and still behave as a zero-or-more wildcard pattern.",
            "Main oracle can conservatively fold `**` into the same semantics as `*` on the validated FIELD subset.",
            ["a", "b", "c", "d", "e", "f", "g", "h"],
            fetch_tags(collection, "**"),
        ))
        checks.append(print_check(
            "like_three_question_exact_length",
            "The documented `?` wildcard matches exactly one character.",
            "Main oracle can safely model `???` as exact-length-three matching on FIELD text.",
            ["f"],
            fetch_tags(collection, "???"),
        ))
        checks.append(print_check(
            "like_percent_is_literal",
            "The official docs define `*` and `?` as wildcards; `%` is not documented as a wildcard.",
            "Main oracle can treat `%` as a literal character rather than SQL-like wildcard syntax.",
            ["h"],
            fetch_tags(collection, "%"),
        ))
        checks.append(print_check(
            "like_underscore_is_literal",
            "The official docs define `*` and `?` as wildcards; `_` is not documented as a wildcard.",
            "Main oracle can treat `_` as a literal character rather than SQL-like wildcard syntax.",
            ["g"],
            fetch_tags(collection, "_"),
        ))

        print_observation(
            "like_empty_pattern_observation",
            "On the local FIELD-text subset, an empty pattern matched the explicit empty-string row only. This is kept observation-only because the main oracle currently normalizes empty strings into the null-like boundary elsewhere.",
            fetch_tags(collection, ""),
        )
        print_observation(
            "like_long_star_pattern_observation",
            "A very long all-`*` pattern behaved like a universal wildcard locally. This is useful as a stress probe but not necessary as a high-rate mainline generator case.",
            fetch_tags(collection, "*" * 300),
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
