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
COLLECTION_PREFIX = "ByUpdateTimeOperatorValidation"


def deterministic_uuid(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-by-update-time-{label}"))


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
            Property(
                name="rev",
                data_type=DataType.INT,
                index_filterable=True,
            ),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True, index_timestamps=True),
    )


def insert_row(collection, label: str, bucket: str, rev: int, vector):
    collection.data.insert(
        uuid=deterministic_uuid(label),
        properties={"tag": label, "bucket": bucket, "rev": rev},
        vector=vector,
    )


def fetch_tags(collection, flt):
    response = collection.query.fetch_objects(filters=flt, limit=20)
    return sorted(obj.properties["tag"] for obj in response.objects)


def fetch_metadata(collection):
    response = collection.query.fetch_objects(
        limit=20,
        return_metadata=MetadataQuery(creation_time=True, last_update_time=True),
    )
    mapping = {}
    for obj in response.objects:
        mapping[obj.properties["tag"]] = {
            "creation_time": obj.metadata.creation_time,
            "update_time": obj.metadata.last_update_time,
            "rev": obj.properties["rev"],
            "bucket": obj.properties["bucket"],
        }
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


def print_observation(name, summary, actual):
    print(name)
    print(f"  observation:{summary}")
    print(f"  observed:   {actual}")


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

        insert_row(collection, "alpha", "left", 0, [1.0, 0.0, 0.0, 0.0])
        time.sleep(1.1)
        insert_row(collection, "beta", "right", 0, [0.0, 1.0, 0.0, 0.0])
        time.sleep(1.1)
        insert_row(collection, "gamma", "left", 0, [0.0, 0.0, 1.0, 0.0])

        initial_metadata = {}
        for _ in range(5):
            initial_metadata = fetch_metadata(collection)
            if {"alpha", "beta", "gamma"} <= set(initial_metadata):
                break
            time.sleep(0.2)
        if {"alpha", "beta", "gamma"} - set(initial_metadata):
            raise RuntimeError(f"missing initial metadata: {initial_metadata}")

        time.sleep(1.1)
        collection.data.replace(
            uuid=deterministic_uuid("beta"),
            properties={"tag": "beta", "bucket": "right", "rev": 1},
            vector=[0.0, 1.0, 1.0, 0.0],
        )

        time.sleep(1.1)
        collection.data.update(
            uuid=deterministic_uuid("gamma"),
            properties={"rev": 1},
        )

        metadata = {}
        for _ in range(10):
            metadata = fetch_metadata(collection)
            if {"alpha", "beta", "gamma"} <= set(metadata):
                beta_advanced = (
                    metadata["beta"]["update_time"] > initial_metadata["beta"]["update_time"]
                )
                gamma_advanced = (
                    metadata["gamma"]["update_time"] > initial_metadata["gamma"]["update_time"]
                )
                if beta_advanced and gamma_advanced:
                    break
            time.sleep(0.2)
        if {"alpha", "beta", "gamma"} - set(metadata):
            raise RuntimeError(f"missing final metadata: {metadata}")

        alpha_creation = initial_metadata["alpha"]["creation_time"]
        alpha_update = metadata["alpha"]["update_time"]
        beta_creation = initial_metadata["beta"]["creation_time"]
        beta_update = metadata["beta"]["update_time"]
        gamma_creation = initial_metadata["gamma"]["creation_time"]
        gamma_update = metadata["gamma"]["update_time"]

        if not (alpha_update == initial_metadata["alpha"]["update_time"]):
            raise RuntimeError(
                f"untouched alpha update_time changed unexpectedly: initial={initial_metadata['alpha']['update_time']} final={alpha_update}"
            )
        if not (beta_update > initial_metadata["beta"]["update_time"]):
            raise RuntimeError(
                f"beta update_time did not advance after replace: initial={initial_metadata['beta']['update_time']} final={beta_update}"
            )
        if not (gamma_update > initial_metadata["gamma"]["update_time"]):
            raise RuntimeError(
                f"gamma update_time did not advance after update: initial={initial_metadata['gamma']['update_time']} final={gamma_update}"
            )
        if not (alpha_update < beta_update < gamma_update):
            raise RuntimeError(
                f"unexpected update-time order: alpha={alpha_update}, beta={beta_update}, gamma={gamma_update}"
            )

        print_observation(
            "initial_update_time_equals_creation_time_observation",
            "Locally, freshly inserted objects returned identical creation_time and last_update_time before any later update.",
            {
                "alpha": alpha_creation == initial_metadata["alpha"]["update_time"],
                "beta": beta_creation == initial_metadata["beta"]["update_time"],
                "gamma": gamma_creation == initial_metadata["gamma"]["update_time"],
            },
        )

        mid_beta_gamma = beta_update + (gamma_update - beta_update) / 2
        future_time = gamma_update + timedelta(days=365)

        checks = []
        checks.append(
            print_check(
                "by_update_time_equal_single_match",
                "The checked docs describe update-time filtering as metadata timestamp filtering.",
                "Current oracle maps by_update_time.equal(ts) to exact server-returned last_update_time equality.",
                ["beta"],
                fetch_tags(collection, Filter.by_update_time().equal(beta_update)),
            )
        )
        checks.append(
            print_check(
                "by_update_time_independent_of_property_filter_index",
                "The checked docs present object timestamp filtering as metadata-based rather than property-token based.",
                "The current fuzzer treats by_update_time as metadata, independent of ordinary property filterability.",
                ["alpha"],
                fetch_tags(collection, Filter.by_update_time().equal(alpha_update)),
            )
        )
        checks.append(
            print_check(
                "by_update_time_greater_than_midpoint",
                "The checked docs show comparison filtering over object timestamps.",
                "Current oracle compares UTC-normalized last_update_time values from object metadata.",
                ["gamma"],
                fetch_tags(collection, Filter.by_update_time().greater_than(mid_beta_gamma)),
            )
        )
        checks.append(
            print_check(
                "by_update_time_less_or_equal_beta",
                "The checked docs show comparison filtering over object timestamps.",
                "Current oracle compares UTC-normalized last_update_time values from object metadata.",
                ["alpha", "beta"],
                fetch_tags(collection, Filter.by_update_time().less_or_equal(beta_update)),
            )
        )
        checks.append(
            print_check(
                "by_update_time_not_equal_beta",
                "The local Python client exposes by_update_time.not_equal(ts) for metadata filtering.",
                "Current oracle maps by_update_time.not_equal(ts) to last_update_time inequality.",
                ["alpha", "gamma"],
                fetch_tags(collection, Filter.by_update_time().not_equal(beta_update)),
            )
        )
        checks.append(
            print_check(
                "by_update_time_contains_any_alpha_gamma",
                "The local Python client exposes by_update_time.contains_any([...]) for metadata filtering.",
                "Current oracle maps by_update_time.contains_any(times) to last_update_time membership.",
                ["alpha", "gamma"],
                fetch_tags(collection, Filter.by_update_time().contains_any([alpha_update, gamma_update])),
            )
        )
        checks.append(
            print_check(
                "by_update_time_contains_none_beta",
                "The local Python client exposes by_update_time.contains_none([...]) for metadata filtering.",
                "Current oracle maps by_update_time.contains_none(times) to last_update_time non-membership.",
                ["alpha", "gamma"],
                fetch_tags(collection, Filter.by_update_time().contains_none([beta_update])),
            )
        )
        checks.append(
            print_check(
                "by_update_time_less_than_alpha_empty",
                "The checked docs show less-than comparison over object timestamps.",
                "Current oracle expects timestamps earlier than the earliest observed last_update_time to match no row here.",
                [],
                fetch_tags(collection, Filter.by_update_time().less_than(alpha_update)),
            )
        )
        checks.append(
            print_check(
                "by_update_time_greater_than_future_empty",
                "The checked docs show greater-than comparison over object timestamps.",
                "Current oracle expects a far-future timestamp threshold to match no current row.",
                [],
                fetch_tags(collection, Filter.by_update_time().greater_than(future_time)),
            )
        )
        checks.append(
            print_check(
                "by_update_time_and_property_filter_composition",
                "The checked filter docs allow boolean composition around metadata filters.",
                "Current oracle intersects last_update_time masks with ordinary property predicates.",
                ["beta"],
                fetch_tags(
                    collection,
                    Filter.by_update_time().greater_or_equal(beta_update)
                    & Filter.by_update_time().less_than(gamma_update)
                    & Filter.by_property("bucket").equal("right"),
                ),
            )
        )

        if all(checks):
            print("Summary: all validated by_update_time checks passed on the local Weaviate service.")
            return 0

        print("Summary: one or more validated by_update_time checks failed.")
        return 1
    finally:
        if created:
            cleanup_collection(client, collection_name)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
