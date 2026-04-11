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
COLLECTION_PREFIX = "GreaterOrEqualOperatorValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-greater-or-equal-operator-{label}"))


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "intFilterOnly": 5,
            "intRangeOnlyObservation": 5,
            "numberVal": -1.0,
            "zeroNumberVal": -1.0,
            "dateVal": "2024-02-29T23:59:58Z",
            "nullableInt": 1,
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={
            "tag": "b",
            "intFilterOnly": 10,
            "intRangeOnlyObservation": 10,
            "numberVal": -0.0,
            "zeroNumberVal": -0.0,
            "dateVal": "2024-02-29T23:59:59+08:00",
            "nullableInt": None,
        },
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={
            "tag": "c",
            "intFilterOnly": 11,
            "intRangeOnlyObservation": 11,
            "numberVal": 0.0,
            "zeroNumberVal": 0.0,
            "dateVal": "2024-02-29T15:59:59Z",
            "nullableInt": 5,
        },
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={
            "tag": "d",
            "intFilterOnly": 20,
            "intRangeOnlyObservation": 20,
            "numberVal": 0.5,
            "zeroNumberVal": 0.5,
            "dateVal": "2024-03-01T00:00:00Z",
        },
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("e"),
        properties={
            "tag": "e",
            "intFilterOnly": 30,
            "intRangeOnlyObservation": 30,
            "numberVal": 2.0,
            "zeroNumberVal": 2.0,
            "dateVal": "2024-03-01T00:00:01Z",
            "nullableInt": 9,
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
                name="intFilterOnly",
                data_type=DataType.INT,
                index_filterable=True,
                index_range_filterable=False,
            ),
            Property(
                name="intRangeOnlyObservation",
                data_type=DataType.INT,
                index_filterable=False,
                index_range_filterable=True,
            ),
            Property(
                name="numberVal",
                data_type=DataType.NUMBER,
                index_filterable=True,
                index_range_filterable=True,
            ),
            Property(
                name="zeroNumberVal",
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
                name="nullableInt",
                data_type=DataType.INT,
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


def print_observation(name, why_observed, observed):
    print(name)
    print(f"  note:    {why_observed}")
    print(f"  observed:{observed}")
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
        observations = []

        checks.append(print_check(
            "int_greater_or_equal_filter_only",
            "GreaterThanEqual is a documented comparison operator for typed property values.",
            "Current oracle maps INT greater_or_equal to strict pandas >= on present values.",
            ["b", "c", "d", "e"],
            fetch_tags(collection, Filter.by_property("intFilterOnly").greater_or_equal(10)),
        ))

        cfg = client.collections.get(collection_name).config.get()
        range_only_cfg = next(
            (prop for prop in cfg.properties if prop.name == "intRangeOnlyObservation"),
            None,
        )
        try:
            range_only_result = fetch_tags(
                collection,
                Filter.by_property("intRangeOnlyObservation").greater_or_equal(10),
            )
            observations.append(("range_only_query_result", range_only_result))
            print_observation(
                "int_greater_or_equal_range_only_observation",
                "Official indexing docs suggest comparison filters can rely on range indexing, but this path is observation-only because it is outside the current fuzzer subset.",
                {
                    "property_config": str(range_only_cfg),
                    "query_result": range_only_result,
                },
            )
        except Exception as exc:
            observations.append(("range_only_query_error", str(exc)))
            print_observation(
                "int_greater_or_equal_range_only_observation",
                "The local client/server combination did not preserve or use the requested range-only property configuration; this keeps range-only greater_or_equal outside the validated oracle subset.",
                {
                    "property_config": str(range_only_cfg),
                    "query_error": str(exc),
                },
            )

        checks.append(print_check(
            "number_greater_or_equal_regular_value",
            "GreaterThanEqual is documented for numeric values.",
            "Current oracle maps NUMBER greater_or_equal to strict float >= for regular finite values.",
            ["d", "e"],
            fetch_tags(collection, Filter.by_property("numberVal").greater_or_equal(0.5)),
        ))

        checks.append(print_check(
            "date_greater_or_equal_utc_normalized",
            "Docs state RFC3339 dates can be filtered similarly to numbers.",
            "Current oracle normalizes DATE values to UTC instants and applies strict >=.",
            ["a", "b", "c", "d", "e"],
            fetch_tags(collection, Filter.by_property("dateVal").greater_or_equal("2024-02-29T15:59:59Z")),
        ))

        checks.append(print_check(
            "nullable_int_greater_or_equal_excludes_null_missing",
            "The docs define comparison filtering but do not document a greater_or_equal-specific null or missing truth table.",
            "Current oracle treats null or omitted scalar values as non-matches for strict greater_or_equal.",
            ["c", "e"],
            fetch_tags(collection, Filter.by_property("nullableInt").greater_or_equal(5)),
        ))

        signed_zero_mismatches = []
        for name, expected, actual in [
            (
                "number_greater_or_equal_positive_zero_signed_zero_probe",
                ["b", "c", "d", "e"],
                fetch_tags(collection, Filter.by_property("zeroNumberVal").greater_or_equal(0.0)),
            ),
            (
                "number_greater_or_equal_negative_zero_signed_zero_probe",
                ["b", "c", "d", "e"],
                fetch_tags(collection, Filter.by_property("zeroNumberVal").greater_or_equal(-0.0)),
            ),
        ]:
            ok = print_mismatch_probe(
                name,
                expected,
                actual,
                "This follows the local signed-zero consistency expectation from the issue10917 family; mismatches are recorded as engine-side observations outside the current main fuzzer subset.",
            )
            if not ok:
                signed_zero_mismatches.append(name)

        if observations:
            print(f"Observation summary: {observations}")
        if signed_zero_mismatches:
            print(f"Signed-zero mismatch probes: {signed_zero_mismatches}")

        if all(checks):
            print("Summary: all greater_or_equal operator checks passed on the local Weaviate service.")
            if observations:
                print("Summary note: range-only comparison indexing remained observation-only outside the current validated subset.")
            if signed_zero_mismatches:
                print("Summary note: signed-zero NUMBER greater_or_equal mismatches were observed outside the current validated subset.")
            return 0

        print("Summary: one or more greater_or_equal operator checks failed.")
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
