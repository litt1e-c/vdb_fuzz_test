import json
import time
import uuid

import requests
import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_PREFIX = "NestedObjectFilterValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-nested-object-filter-{label}"))


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "metaObj": {"price": 120, "color": "red", "active": True, "score": 1.5},
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={
            "tag": "b",
            "metaObj": {"price": 80, "color": "blue", "active": False, "score": 2.5},
        },
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("c"),
        properties={
            "tag": "c",
            "metaObj": {"price": 150, "color": "red", "active": False},
        },
        vector=[0.0, 0.0, 1.0],
    ),
    DataObject(
        uuid=deterministic_uuid("d"),
        properties={"tag": "d", "metaObj": None},
        vector=[0.5, 0.5, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("e"),
        properties={"tag": "e"},
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
                name="metaObj",
                data_type=DataType.OBJECT,
                index_filterable=True,
                nested_properties=[
                    Property(name="price", data_type=DataType.INT, index_filterable=True, index_range_filters=True),
                    Property(name="color", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
                    Property(name="active", data_type=DataType.BOOL, index_filterable=True),
                    Property(name="score", data_type=DataType.NUMBER, index_filterable=True, index_range_filters=True),
                ],
            ),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def print_check(name, note, expected, observed):
    print(name)
    print(f"  note:    {note}")
    print(f"  expected:{expected}")
    print(f"  observed:{observed}")
    ok = observed == expected
    print(f"  result:  {'PASS' if ok else 'FAIL'}")
    return ok


def graphql_raw(query):
    response = requests.post(
        f"http://{HOST}:{PORT}/v1/graphql",
        json={"query": query},
        timeout=20,
    )
    return response.status_code, response.json()


def rest_batch_delete_dry_run(payload):
    response = requests.delete(
        f"http://{HOST}:{PORT}/v1/batch/objects",
        json=payload,
        timeout=20,
    )
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    return response.status_code, body


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

        gql_status, gql_body = graphql_raw(
            f'{{ Get {{ {collection_name}(where: {{path:["metaObj","price"], operator: GreaterThan, valueInt: 100}}) {{ tag }} }} }}'
        )
        gql_message = ((gql_body.get("errors") or [{}])[0]).get("message", "")
        checks.append(print_check(
            "graphql_nested_object_price_rejected",
            "Local Weaviate v1.36.10 rejects OBJECT child paths in GraphQL where-path parsing; this boundary should stay out of the main oracle until the engine/client path semantics are genuinely supported.",
            {"status": 200, "message_substring": "missing an argument after 'price'"},
            {"status": gql_status, "message_substring": "missing an argument after 'price'" if "missing an argument after 'price'" in gql_message else gql_message},
        ))

        rest_status, rest_body = rest_batch_delete_dry_run(
            {
                "match": {
                    "class": collection_name,
                    "where": {"path": ["metaObj", "price"], "operator": "GreaterThan", "valueInt": 100},
                },
                "dryRun": True,
                "output": "verbose",
            }
        )
        rest_message = ((rest_body.get("error") or [{}])[0]).get("message", "")
        checks.append(print_check(
            "rest_nested_object_price_rejected",
            "The REST batch-delete dry-run path rejects the same OBJECT child path with the same parser boundary, so REST filter mode should not generate nested OBJECT child paths on this local version.",
            {"status": 500, "message_substring": "missing an argument after 'price'"},
            {"status": rest_status, "message_substring": "missing an argument after 'price'" if "missing an argument after 'price'" in rest_message else rest_message},
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
