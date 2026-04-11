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
COLLECTION_PREFIX = "AndOperatorValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-and-operator-{label}"))


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "round": "Double",
            "answer": "Mexico",
            "points": 100,
            "nullableScore": 10,
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
            "metaObj": {"price": 500},
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

        amp_filter = (
            Filter.by_property("round").equal("Double")
            & Filter.by_property("points").less_than(500)
        )
        checks.append(print_check(
            "ampersand_pair_and",
            "AND combines multiple conditions; the result should contain only objects satisfying both operands.",
            "Pandas oracle maps this as mask(round == 'Double') & mask(points < 500).",
            ["a", "b", "d"],
            fetch_tags(collection, amp_filter),
        ))

        all_of_filter = Filter.all_of([
            Filter.by_property("points").greater_than(100),
            Filter.by_property("points").less_than(700),
            Filter.by_property("round").equal("Double"),
        ])
        checks.append(print_check(
            "all_of_three_operand_and",
            "Python client docs state Filter.all_of combines a list of filters with AND.",
            "Pandas oracle folds the three boolean masks with conjunction.",
            ["b", "d", "e"],
            fetch_tags(collection, all_of_filter),
        ))

        nested_filter = (
            Filter.by_property("answer").equal("Bird")
            & (
                Filter.by_property("points").greater_than(700)
                | Filter.by_property("points").less_than(300)
            )
        )
        checks.append(print_check(
            "nested_and_or",
            "Docs allow nested filters with outer And/Or operands.",
            "Oracle applies the same explicit parenthesized boolean expression.",
            ["c", "d"],
            fetch_tags(collection, nested_filter),
        ))

        not_filter = (
            Filter.by_property("round").equal("Double")
            & Filter.not_(Filter.by_property("answer").equal("Yucatan"))
        )
        checks.append(print_check(
            "and_with_not_operand",
            "Official examples show AND operands may include a NOT subfilter.",
            "Oracle treats this as the intersection of the round predicate and the generated NOT predicate.",
            ["a", "d", "e"],
            fetch_tags(collection, not_filter),
        ))

        null_filter = (
            Filter.by_property("points").greater_than(150)
            & Filter.by_property("nullableScore").is_none(True)
        )
        checks.append(print_check(
            "and_with_null_or_missing_operand",
            "Docs define AND composition, but do not fully specify scalar missing/null truth tables here.",
            "Current fuzzer oracle assumes omitted or null inserted scalar values satisfy is_none(True), then intersects with the points predicate.",
            ["b", "d"],
            fetch_tags(collection, null_filter),
        ))

        object_null_filter = (
            Filter.by_property("points").greater_than(150)
            & Filter.by_property("metaObj").is_none(True)
        )
        checks.append(print_check(
            "and_with_object_null_or_missing_operand",
            "Docs define AND composition; OBJECT null filtering is a separate operand-level behavior.",
            "Current fuzzer may compose OBJECT is_none(True) under AND, while OBJECT is_none(False) remains outside this validated subset.",
            ["b", "d"],
            fetch_tags(collection, object_null_filter),
        ))

        if all(checks):
            print("Summary: all AND operator checks passed on the local Weaviate service.")
            return 0

        print("Summary: one or more AND operator checks failed.")
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
