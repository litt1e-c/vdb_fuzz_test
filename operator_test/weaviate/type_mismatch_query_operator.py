import json
import uuid

import requests
import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter

from operator_case_validator import OperatorCase, run_operator_cases


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
BASE_URL = f"http://{HOST}:{PORT}"
COLLECTION_NAME = "TypeMismatchQueryValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-type-mismatch-query-{label}"))


class GraphQLEnum:
    def __init__(self, value):
        self.value = str(value)

    def __str__(self):
        return self.value


ROWS = [
    DataObject(
        uuid=deterministic_uuid("a"),
        properties={
            "tag": "a",
            "intVal": 1,
            "numVal": 1.5,
            "boolVal": True,
            "textVal": "alpha",
            "dateVal": "2024-01-01T00:00:00Z",
            "intArr": [1, 2],
        },
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid("b"),
        properties={
            "tag": "b",
            "intVal": 2,
            "numVal": 2.5,
            "boolVal": False,
            "textVal": "beta",
            "dateVal": "2024-01-02T00:00:00Z",
            "intArr": [2],
        },
        vector=[0.0, 1.0, 0.0],
    ),
]


def create_collection(client, name):
    return client.collections.create(
        name=name,
        properties=[
            Property(name="tag", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="intVal", data_type=DataType.INT, index_filterable=True, index_range_filters=True),
            Property(name="numVal", data_type=DataType.NUMBER, index_filterable=True, index_range_filters=True),
            Property(name="boolVal", data_type=DataType.BOOL, index_filterable=True),
            Property(name="textVal", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True, index_searchable=True),
            Property(name="dateVal", data_type=DataType.DATE, index_filterable=True, index_range_filters=True),
            Property(name="intArr", data_type=DataType.INT_ARRAY, index_filterable=True),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True, index_property_length=True),
    )


def graphql_with_enum_operators(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "operator":
                out[key] = GraphQLEnum(item)
            else:
                out[key] = graphql_with_enum_operators(item)
        return out
    if isinstance(value, list):
        return [graphql_with_enum_operators(item) for item in value]
    return value


def graphql_input_literal(value):
    if isinstance(value, GraphQLEnum):
        return str(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(graphql_input_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        ordered_keys = sorted(value.keys())
        return "{" + ", ".join(f"{key}:{graphql_input_literal(value[key])}" for key in ordered_keys) + "}"
    raise TypeError(f"Unsupported GraphQL literal value: {type(value)!r}")


def extract_graphql_rows(result, collection_name):
    if isinstance(result, dict):
        data = result.get("data") or {}
        get_payload = data.get("Get") or data.get("get") or {}
        if isinstance(get_payload, dict):
            return get_payload.get(collection_name, []) or []

    data = getattr(result, "data", None)
    if isinstance(data, dict):
        get_payload = data.get("Get") or data.get("get") or {}
        if isinstance(get_payload, dict):
            return get_payload.get(collection_name, []) or []

    get_payload = getattr(result, "get", None)
    if isinstance(get_payload, dict):
        return get_payload.get(collection_name, []) or []

    return []


def fetch_python_tags(collection, flt):
    response = collection.query.fetch_objects(filters=flt, limit=20)
    return sorted(obj.properties["tag"] for obj in response.objects)


def fetch_rest_tags(collection_name, where_filter):
    payload = {
        "match": {
            "class": collection_name,
            "where": where_filter,
        },
        "dryRun": True,
        "output": "verbose",
    }
    response = requests.delete(f"{BASE_URL}/v1/batch/objects", json=payload, timeout=30)
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    if response.status_code != 200:
        raise RuntimeError(f"REST status={response.status_code} body={json.dumps(body, sort_keys=True)}")

    results = body.get("results") or {}
    objects = results.get("objects") or []
    return sorted(str(obj.get("id")) for obj in objects if obj.get("status") == "DRYRUN")


def fetch_graphql_tags(client, collection_name, where_filter):
    gql_where = graphql_with_enum_operators(where_filter)
    query = (
        "{ Get { "
        f"{collection_name}(where:{graphql_input_literal(gql_where)}, limit:64) "
        "{ tag _additional { id } } } }"
    )
    result = client.graphql_raw_query(query)
    errors = None
    if isinstance(result, dict):
        errors = result.get("errors")
    if errors is None:
        errors = getattr(result, "errors", None)
    if errors:
        raise RuntimeError(f"GraphQL errors: {errors}")

    rows = extract_graphql_rows(result, collection_name)
    return sorted(str((row.get("_additional") or {}).get("id")) for row in rows if (row.get("_additional") or {}).get("id"))


def cleanup_collection(client, name):
    try:
        if client.collections.exists(name):
            client.collections.delete(name)
    except Exception as exc:
        print(f"Cleanup warning for {name}: {exc}")


def main():
    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    created = False
    try:
        cleanup_collection(client, COLLECTION_NAME)
        collection = create_collection(client, COLLECTION_NAME)
        created = True
        result = collection.data.insert_many(ROWS)
        errors = getattr(result, "errors", None)
        if errors:
            print(f"Insert errors: {errors}")
            return 2

        python_cases = [
            OperatorCase("empty_or_error", "python_int_equal_text", Filter.by_property("intVal").equal("oops"), error_category="type_rejection"),
            OperatorCase("empty_or_error", "python_int_gt_text", Filter.by_property("intVal").greater_than("oops"), error_category="type_rejection"),
            OperatorCase("empty_or_error", "python_bool_equal_text", Filter.by_property("boolVal").equal("true"), error_category="type_rejection"),
            OperatorCase("empty_or_error", "python_date_lt_int", Filter.by_property("dateVal").less_than(7), error_category="type_rejection"),
            OperatorCase("empty_or_error", "python_int_contains_any_text_array", Filter.by_property("intVal").contains_any(["1", "2"]), error_category="type_rejection"),
        ]
        rest_cases = [
            OperatorCase("empty_or_error", "rest_int_equal_valueText", {"path": ["intVal"], "operator": "Equal", "valueText": "oops"}, error_category="type_rejection"),
            OperatorCase("empty_or_error", "rest_int_gt_valueText", {"path": ["intVal"], "operator": "GreaterThan", "valueText": "oops"}, error_category="type_rejection"),
            OperatorCase("empty_or_error", "rest_bool_equal_valueText", {"path": ["boolVal"], "operator": "Equal", "valueText": "true"}, error_category="type_rejection"),
            OperatorCase("empty_or_error", "rest_date_lt_valueInt", {"path": ["dateVal"], "operator": "LessThan", "valueInt": 7}, error_category="type_rejection"),
            OperatorCase("empty_or_error", "rest_int_contains_any_valueTextArray", {"path": ["intVal"], "operator": "ContainsAny", "valueTextArray": ["1", "2"]}, error_category="type_rejection"),
        ]
        graphql_cases = [
            OperatorCase("empty_or_error", "graphql_int_equal_valueText", {"path": ["intVal"], "operator": "Equal", "valueText": "oops"}, error_category="type_rejection"),
            OperatorCase("empty_or_error", "graphql_int_gt_valueText", {"path": ["intVal"], "operator": "GreaterThan", "valueText": "oops"}, error_category="type_rejection"),
            OperatorCase("empty_or_error", "graphql_bool_equal_valueText", {"path": ["boolVal"], "operator": "Equal", "valueText": "true"}, error_category="type_rejection"),
            OperatorCase("empty_or_error", "graphql_date_lt_valueInt", {"path": ["dateVal"], "operator": "LessThan", "valueInt": 7}, error_category="type_rejection"),
            OperatorCase("empty_or_error", "graphql_int_contains_any_valueTextArray", {"path": ["intVal"], "operator": "ContainsAny", "valueTextArray": ["1", "2"]}, error_category="validation_error"),
        ]

        failed = 0
        failed += run_operator_cases(
            subject=collection,
            tests=python_cases,
            query_fn=fetch_python_tags,
            title="Weaviate scalar type-mismatch: Python filter builder / gRPC path",
            default_expected_error="type_rejection",
        )
        failed += run_operator_cases(
            subject=COLLECTION_NAME,
            tests=rest_cases,
            query_fn=fetch_rest_tags,
            title="Weaviate scalar type-mismatch: REST where JSON path",
            default_expected_error="type_rejection",
        )
        failed += run_operator_cases(
            subject=(client, COLLECTION_NAME),
            tests=graphql_cases,
            query_fn=lambda subject, expr: fetch_graphql_tags(subject[0], subject[1], expr),
            title="Weaviate scalar type-mismatch: GraphQL where JSON path",
            default_expected_error="type_rejection",
        )

        if failed == 0:
            print("Summary: all type-mismatch scalar robustness checks passed.")
            return 0

        print(f"Summary: type-mismatch scalar robustness failures observed ({failed}).")
        return 1
    finally:
        if created:
            cleanup_collection(client, COLLECTION_NAME)
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
