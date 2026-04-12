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
COLLECTION_PREFIX = "ByIdOperatorValidation"


def deterministic_uuid(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-by-id-{label}"))


ABSENT_UUID = deterministic_uuid("absent")
ROWS = [
    DataObject(
        uuid=deterministic_uuid("alpha"),
        properties={"tag": "alpha", "bucket": "left"},
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("beta"),
        properties={"tag": "beta", "bucket": "right"},
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("gamma"),
        properties={"tag": "gamma", "bucket": "right"},
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("delta"),
        properties={"tag": "delta", "bucket": "left"},
        vector=[0.5, 0.5, 0.0],
    ),
]


def create_collection(client, name: str):
    return client.collections.create(
        name=name,
        properties=[
            Property(
                name="tag",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=False,
                index_searchable=True,
            ),
            Property(
                name="bucket",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
            ),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
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

        alpha_id = deterministic_uuid("alpha")
        beta_id = deterministic_uuid("beta")
        gamma_id = deterministic_uuid("gamma")
        delta_id = deterministic_uuid("delta")

        checks = []
        checks.append(
            print_check(
                "by_id_equal_single_match",
                "The checked docs show filtering by object id through a metadata id path / by_id entry point.",
                "Current oracle maps by_id.equal(uuid) to exact dataframe id equality.",
                ["alpha"],
                fetch_tags(collection, Filter.by_id().equal(alpha_id)),
            )
        )
        checks.append(
            print_check(
                "by_id_equal_independent_of_property_filter_index",
                "The docs present object-id filtering as metadata-based rather than property-token based.",
                "The current fuzzer treats by_id as always queryable metadata, even when ordinary properties are not filterable.",
                ["beta"],
                fetch_tags(collection, Filter.by_id().equal(beta_id)),
            )
        )
        checks.append(
            print_check(
                "by_id_equal_absent_uuid_returns_empty",
                "The checked docs show equality filtering by object id, but do not spell out an absent-UUID truth table.",
                "The current oracle assumes by_id.equal(valid_absent_uuid) matches no present row.",
                [],
                fetch_tags(collection, Filter.by_id().equal(ABSENT_UUID)),
            )
        )
        checks.append(
            print_check(
                "by_id_contains_any_multiple_ids",
                "The local Python client exposes by_id.contains_any([...]) for metadata id filtering.",
                "Current oracle maps by_id.contains_any(ids) to dataframe id membership.",
                ["alpha", "gamma"],
                fetch_tags(collection, Filter.by_id().contains_any([alpha_id, gamma_id])),
            )
        )
        checks.append(
            print_check(
                "by_id_contains_any_absent_uuid_returns_empty",
                "The local Python client exposes by_id.contains_any([...]) for metadata id filtering.",
                "The current oracle assumes contains_any with only valid absent UUIDs matches no present row.",
                [],
                fetch_tags(collection, Filter.by_id().contains_any([ABSENT_UUID])),
            )
        )
        checks.append(
            print_check(
                "not_by_id_contains_any_existing_uuid",
                "The checked filter docs allow boolean composition around metadata filters.",
                "The current oracle complements dataframe id membership for NOT(by_id.contains_any([existing_uuid])).",
                ["alpha", "delta", "gamma"],
                fetch_tags(collection, Filter.not_(Filter.by_id().contains_any([beta_id]))),
            )
        )
        checks.append(
            print_check(
                "not_by_id_contains_any_absent_uuid_returns_all_rows",
                "The checked docs do not give an id-specific absent-UUID truth table for NOT(...), but boolean composition is documented.",
                "The current oracle assumes NOT(by_id.contains_any([absent_uuid])) matches all present rows.",
                ["alpha", "beta", "delta", "gamma"],
                fetch_tags(collection, Filter.not_(Filter.by_id().contains_any([ABSENT_UUID]))),
            )
        )
        checks.append(
            print_check(
                "by_id_not_equal_existing_id",
                "The local Python client exposes by_id.not_equal(uuid) for metadata id filtering.",
                "Current oracle maps by_id.not_equal(uuid) to dataframe id inequality.",
                ["alpha", "delta", "gamma"],
                fetch_tags(collection, Filter.by_id().not_equal(beta_id)),
            )
        )
        checks.append(
            print_check(
                "by_id_not_equal_absent_id_returns_all_rows",
                "The checked docs do not give an id-specific truth table for absent UUID values.",
                "The current oracle assumes by_id.not_equal(absent_uuid) matches all present rows.",
                ["alpha", "beta", "delta", "gamma"],
                fetch_tags(collection, Filter.by_id().not_equal(ABSENT_UUID)),
            )
        )
        checks.append(
            print_check(
                "by_id_contains_none_existing_ids",
                "The local Python client exposes by_id.contains_none([...]) for metadata id filtering.",
                "Current oracle maps by_id.contains_none(ids) to dataframe id non-membership.",
                ["beta", "delta"],
                fetch_tags(collection, Filter.by_id().contains_none([alpha_id, gamma_id])),
            )
        )
        checks.append(
            print_check(
                "by_id_contains_none_absent_id_returns_all_rows",
                "The checked docs do not give an id-specific contains_none truth table for absent UUID values.",
                "The current oracle assumes by_id.contains_none([absent_uuid]) matches all present rows.",
                ["alpha", "beta", "delta", "gamma"],
                fetch_tags(collection, Filter.by_id().contains_none([ABSENT_UUID])),
            )
        )
        checks.append(
            print_check(
                "by_id_and_property_filter_composition",
                "Metadata id filters can be combined with ordinary filters in the checked filter docs.",
                "Current oracle intersects dataframe id membership with the other predicate mask.",
                ["gamma"],
                fetch_tags(
                    collection,
                    Filter.by_id().contains_any([alpha_id, gamma_id]) & Filter.by_property("bucket").equal("right"),
                ),
            )
        )
        checks.append(
            print_check(
                "by_id_contains_any_with_absent_uuid_noise",
                "The checked docs do not document an id-specific noise-value example, but valid UUID inputs are accepted by the client.",
                "Current oracle treats absent UUIDs in contains_any as harmless non-matching noise.",
                ["delta"],
                fetch_tags(collection, Filter.by_id().contains_any([ABSENT_UUID, delta_id])),
            )
        )

        if all(checks):
            print("Summary: all validated by_id checks passed on the local Weaviate service.")
            return 0

        print("Summary: one or more validated by_id checks failed.")
        return 1
    finally:
        if created:
            cleanup_collection(client, collection_name)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
