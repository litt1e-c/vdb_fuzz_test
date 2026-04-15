import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_NAME = "LikeCompoundValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-like-compound-{label}"))


ROWS = [
    DataObject(uuid=deterministic_uuid("a"), properties={"tag": "a", "textVal": "Alpha"}, vector=[1.0, 0.0, 0.0]),
    DataObject(uuid=deterministic_uuid("b"), properties={"tag": "b", "textVal": "Alps"}, vector=[0.0, 1.0, 0.0]),
    DataObject(uuid=deterministic_uuid("c"), properties={"tag": "c", "textVal": "Beta"}, vector=[0.0, 0.0, 1.0]),
    DataObject(uuid=deterministic_uuid("d"), properties={"tag": "d", "textVal": "abc"}, vector=[0.5, 0.5, 0.0]),
    DataObject(uuid=deterministic_uuid("e"), properties={"tag": "e", "textVal": "%"}, vector=[0.5, 0.0, 0.5]),
    DataObject(uuid=deterministic_uuid("f"), properties={"tag": "f", "textVal": "_"}, vector=[0.4, 0.3, 0.2]),
    DataObject(uuid=deterministic_uuid("g"), properties={"tag": "g", "textVal": ""}, vector=[0.2, 0.4, 0.3]),
    DataObject(uuid=deterministic_uuid("h"), properties={"tag": "h"}, vector=[0.3, 0.2, 0.4]),
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
            "not_like_three_question_includes_missing",
            "LIKE on FIELD text matches present rows only; NOT should therefore include non-matching present rows plus the missing row under the local denylist semantics.",
            "Main oracle can safely model NOT(text like \"???\") with explicit NOT-mask semantics that include the missing row.",
            ["a", "b", "c", "e", "f", "g", "h"],
            fetch_tags(collection, Filter.not_(Filter.by_property("textVal").like("???"))),
        ))
        checks.append(print_check(
            "like_prefix_and_contains_none_other",
            "An allowlist LIKE predicate can be intersected with a denylist-style scalar contains_none predicate on the same FIELD-text property.",
            "Main oracle can model `like(\"Al*\") AND contains_none([\"Alpha\"])` as allowlist/denylist intersection, leaving only the non-Alpha `Al*` row.",
            ["b"],
            fetch_tags(
                collection,
                Filter.by_property("textVal").like("Al*") & Filter.by_property("textVal").contains_none(["Alpha"]),
            ),
        ))
        checks.append(print_check(
            "like_prefix_and_not_contains_none_self",
            "A LIKE predicate can be conjoined with NOT(contains_none([self])) to isolate the rows equal to the targeted string while still exercising mixed allow/deny merging.",
            "Main oracle can safely use `like(\"Al*\") AND NOT(contains_none([\"Alpha\"]))` as a deeper text compound pattern.",
            ["a"],
            fetch_tags(
                collection,
                Filter.by_property("textVal").like("Al*") & Filter.not_(Filter.by_property("textVal").contains_none(["Alpha"])),
            ),
        ))
        checks.append(print_check(
            "like_star_or_null",
            "LIKE `*` matches all present rows; disjoining with is_null keeps the missing row too.",
            "Main oracle can safely generate `like(\"*\") OR is_null(true)` as a deep but explainable LIKE/null compound.",
            ["a", "b", "c", "d", "e", "f", "g", "h"],
            fetch_tags(collection, Filter.by_property("textVal").like("*") | Filter.by_property("textVal").is_none(True)),
        ))
        checks.append(print_check(
            "not_like_percent_includes_missing",
            "The locally validated literal `%` pattern can be negated; the missing row remains included on the NOT side.",
            "Main oracle can treat `NOT(like(\"%\"))` as denylist negation over the literal-`%` row plus the missing row.",
            ["a", "b", "c", "d", "f", "g", "h"],
            fetch_tags(collection, Filter.not_(Filter.by_property("textVal").like("%"))),
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
