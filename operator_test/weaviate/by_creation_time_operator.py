import sys
import time
import uuid
from datetime import timedelta

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.query import Filter, MetadataQuery


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_PREFIX = "ByCreationTimeOperatorValidation"


def deterministic_uuid(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-by-creation-time-{label}"))


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
        inverted_index_config=Configure.inverted_index(index_null_state=True, index_timestamps=True),
    )


def insert_row(collection, label: str, bucket: str, vector):
    collection.data.insert(
        uuid=deterministic_uuid(label),
        properties={"tag": label, "bucket": bucket},
        vector=vector,
    )


def fetch_tags(collection, flt):
    response = collection.query.fetch_objects(filters=flt, limit=20)
    return sorted(obj.properties["tag"] for obj in response.objects)


def fetch_creation_times(collection):
    response = collection.query.fetch_objects(
        limit=20,
        return_metadata=MetadataQuery(creation_time=True),
    )
    mapping = {}
    for obj in response.objects:
        mapping[obj.properties["tag"]] = obj.metadata.creation_time
    return mapping


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

        insert_row(collection, "alpha", "left", [1.0, 0.0, 0.0])
        time.sleep(1.1)
        insert_row(collection, "beta", "right", [0.0, 1.0, 0.0])
        time.sleep(1.1)
        insert_row(collection, "gamma", "left", [0.0, 0.0, 1.0])

        creation_times = {}
        for _ in range(5):
            creation_times = fetch_creation_times(collection)
            if {"alpha", "beta", "gamma"} <= set(creation_times):
                break
            time.sleep(0.2)
        if {"alpha", "beta", "gamma"} - set(creation_times):
            raise RuntimeError(f"missing creation times: {creation_times}")

        alpha_time = creation_times["alpha"]
        beta_time = creation_times["beta"]
        gamma_time = creation_times["gamma"]

        if not (alpha_time < beta_time < gamma_time):
            raise RuntimeError(
                f"unexpected creation-time order: alpha={alpha_time}, beta={beta_time}, gamma={gamma_time}"
            )

        mid_alpha_beta = alpha_time + (beta_time - alpha_time) / 2
        mid_beta_gamma = beta_time + (gamma_time - beta_time) / 2
        future_time = gamma_time + timedelta(days=365)

        checks = []
        checks.append(
            print_check(
                "by_creation_time_equal_single_match",
                "The checked docs show creation-time filtering as a metadata filter entry point.",
                "Current oracle maps by_creation_time.equal(ts) to exact metadata timestamp equality.",
                ["beta"],
                fetch_tags(collection, Filter.by_creation_time().equal(beta_time)),
            )
        )
        checks.append(
            print_check(
                "by_creation_time_independent_of_property_filter_index",
                "The checked docs present creation-time filtering as metadata-based rather than property-token based.",
                "The current fuzzer treats by_creation_time as metadata, independent of ordinary property filterability.",
                ["alpha"],
                fetch_tags(collection, Filter.by_creation_time().equal(alpha_time)),
            )
        )
        checks.append(
            print_check(
                "by_creation_time_greater_than_midpoint",
                "The checked docs show comparison operators over creation time metadata.",
                "Current oracle compares metadata datetimes in UTC.",
                ["beta", "gamma"],
                fetch_tags(collection, Filter.by_creation_time().greater_than(mid_alpha_beta)),
            )
        )
        checks.append(
            print_check(
                "by_creation_time_less_or_equal_beta",
                "The checked docs show comparison operators over creation time metadata.",
                "Current oracle compares metadata datetimes in UTC.",
                ["alpha", "beta"],
                fetch_tags(collection, Filter.by_creation_time().less_or_equal(beta_time)),
            )
        )
        checks.append(
            print_check(
                "by_creation_time_not_equal_beta",
                "The local Python client exposes by_creation_time.not_equal(ts) for metadata filtering.",
                "Current oracle maps by_creation_time.not_equal(ts) to metadata timestamp inequality.",
                ["alpha", "gamma"],
                fetch_tags(collection, Filter.by_creation_time().not_equal(beta_time)),
            )
        )
        checks.append(
            print_check(
                "by_creation_time_contains_any_alpha_gamma",
                "The local Python client exposes by_creation_time.contains_any([...]) for metadata filtering.",
                "Current oracle maps by_creation_time.contains_any(times) to metadata timestamp membership.",
                ["alpha", "gamma"],
                fetch_tags(collection, Filter.by_creation_time().contains_any([alpha_time, gamma_time])),
            )
        )
        checks.append(
            print_check(
                "by_creation_time_contains_none_beta",
                "The local Python client exposes by_creation_time.contains_none([...]) for metadata filtering.",
                "Current oracle maps by_creation_time.contains_none(times) to metadata timestamp non-membership.",
                ["alpha", "gamma"],
                fetch_tags(collection, Filter.by_creation_time().contains_none([beta_time])),
            )
        )
        checks.append(
            print_check(
                "by_creation_time_less_than_alpha_empty",
                "The checked docs show less-than comparison over creation time metadata.",
                "Current oracle expects timestamps earlier than the first inserted object's creation time to match no row here.",
                [],
                fetch_tags(collection, Filter.by_creation_time().less_than(alpha_time)),
            )
        )
        checks.append(
            print_check(
                "by_creation_time_greater_than_future_empty",
                "The checked docs show greater-than comparison over creation time metadata.",
                "Current oracle expects a far-future timestamp threshold to match no current row.",
                [],
                fetch_tags(collection, Filter.by_creation_time().greater_than(future_time)),
            )
        )
        checks.append(
            print_check(
                "by_creation_time_and_property_filter_composition",
                "The checked filter docs allow boolean composition around metadata filters.",
                "Current oracle intersects metadata timestamp masks with ordinary property predicates.",
                ["beta"],
                fetch_tags(
                    collection,
                    Filter.by_creation_time().greater_or_equal(mid_alpha_beta)
                    & Filter.by_creation_time().less_than(mid_beta_gamma)
                    & Filter.by_property("bucket").equal("right"),
                ),
            )
        )

        if all(checks):
            print("Summary: all validated by_creation_time checks passed on the local Weaviate service.")
            return 0

        print("Summary: one or more validated by_creation_time checks failed.")
        return 1
    finally:
        if created:
            cleanup_collection(client, collection_name)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
