import sys
import time
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_PREFIX = "GraphqlWhereJsonOperatorValidation"


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-graphql-where-json-{label}"))


class GraphQLEnum:
    def __init__(self, value):
        self.value = str(value)

    def __str__(self):
        return self.value


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


ROWS = [
    {
        "tag": "a",
        "intVal": -10,
        "numVal": -1.5,
        "boolVal": True,
        "textVal": "alpha",
        "dateVal": "2024-01-01T00:00:00Z",
        "intArr": [-10, 0],
        "numArr": [-1.5, 0.5],
        "boolArr": [True, False],
        "textArr": ["alpha", "beta"],
        "dateArr": ["2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z"],
    },
    {
        "tag": "b",
        "intVal": 0,
        "numVal": 0.5,
        "boolVal": False,
        "textVal": "beta",
        "dateVal": "2024-01-02T00:00:00Z",
        "intArr": [0],
        "numArr": [0.5],
        "boolArr": [False],
        "textArr": ["beta"],
        "dateArr": ["2024-01-02T00:00:00Z"],
    },
    {
        "tag": "c",
        "intVal": 7,
        "numVal": 4.75,
        "boolVal": True,
        "textVal": "gamma",
        "dateVal": "2024-01-03T00:00:00Z",
        "intArr": [7, 7],
        "numArr": [4.75, 4.75],
        "boolArr": [True, True],
        "textArr": ["gamma", "alpha"],
        "dateArr": ["2024-01-03T00:00:00Z"],
    },
    {
        "tag": "d",
        "intVal": 42,
        "numVal": 42.5,
        "boolVal": False,
        "textVal": "alphabet",
        "dateVal": "2024-01-04T00:00:00Z",
        "intArr": [],
        "numArr": [],
        "boolArr": [],
        "textArr": [],
        "dateArr": [],
    },
    {
        "tag": "e",
        "intVal": None,
        "numVal": None,
        "boolVal": None,
        "textVal": None,
        "dateVal": None,
        "intArr": None,
        "numArr": None,
        "boolArr": None,
        "textArr": None,
        "dateArr": None,
    },
    {
        "tag": "f",
    },
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
            Property(name="numArr", data_type=DataType.NUMBER_ARRAY, index_filterable=True),
            Property(name="boolArr", data_type=DataType.BOOL_ARRAY, index_filterable=True),
            Property(name="textArr", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="dateArr", data_type=DataType.DATE_ARRAY, index_filterable=True),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True, index_property_length=True),
    )


def insert_rows(collection):
    objects = []
    for idx, props in enumerate(ROWS):
        clean_props = {key: value for key, value in props.items() if value is not None}
        objects.append(
            DataObject(
                uuid=deterministic_uuid(props["tag"]),
                properties=clean_props,
                vector=[1.0, float(idx + 1) / 10.0, float((idx % 3) + 1) / 10.0],
            )
        )
    result = collection.data.insert_many(objects)
    errors = getattr(result, "errors", None)
    if errors:
        raise RuntimeError(f"insert_many errors: {errors}")


def tags_for_ids(ids):
    id_to_tag = {deterministic_uuid(row["tag"]): row["tag"] for row in ROWS}
    return sorted(id_to_tag[object_id] for object_id in ids)


def wf(path, operator, **value):
    out = {"path": path, "operator": operator}
    normalized = {}
    for key, raw_value in value.items():
        mapped_key = key
        if isinstance(raw_value, list):
            if key == "valueInt":
                mapped_key = "valueIntArray"
            elif key == "valueNumber":
                mapped_key = "valueNumberArray"
            elif key == "valueText":
                mapped_key = "valueTextArray"
            elif key == "valueBoolean":
                mapped_key = "valueBooleanArray"
            elif key == "valueDate":
                mapped_key = "valueDateArray"
            elif key == "valueString":
                mapped_key = "valueStringArray"
        normalized[mapped_key] = raw_value
    out.update(normalized)
    return out


def compound(operator, operands):
    return {"operator": operator, "operands": operands}


def graphql_query_or_raise(client, query):
    result = client.graphql_raw_query(query)
    errors = getattr(result, "errors", None)
    if errors:
        raise RuntimeError(f"GraphQL errors: {errors} | query={query}")
    return result


def graphql_ids_for_where(client, collection_name, where_filter):
    gql_where = graphql_with_enum_operators(where_filter)
    query = (
        "{ Get { "
        f"{collection_name}(where:{graphql_input_literal(gql_where)}, limit:64) "
        "{ tag _additional { id } } } }"
    )
    result = graphql_query_or_raise(client, query)
    rows = getattr(result, "get", {}).get(collection_name, []) or []
    ids = sorted(str((row.get("_additional") or {}).get("id")) for row in rows if (row.get("_additional") or {}).get("id"))
    return ids, query


def check(client, collection_name, name, where_filter, expected_tags):
    ids, query = graphql_ids_for_where(client, collection_name, where_filter)
    actual_tags = tags_for_ids(ids)
    ok = actual_tags == sorted(expected_tags)
    print(name)
    print(f"  where:   {where_filter}")
    print(f"  query:   {query}")
    print(f"  expected:{sorted(expected_tags)}")
    print(f"  observed:{actual_tags}")
    print(f"  result:  {'PASS' if ok else 'FAIL'}")
    return ok


def cleanup_collection(client, name):
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

        checks = [
            check(client, collection_name, "int_equal", wf(["intVal"], "Equal", valueInt=7), ["c"]),
            check(client, collection_name, "int_not_equal", wf(["intVal"], "NotEqual", valueInt=7), ["a", "b", "d", "e", "f"]),
            check(client, collection_name, "int_range", compound("And", [
                wf(["intVal"], "GreaterThanEqual", valueInt=0),
                wf(["intVal"], "LessThanEqual", valueInt=42),
            ]), ["b", "c", "d"]),
            check(client, collection_name, "number_greater_than", wf(["numVal"], "GreaterThan", valueNumber=1.0), ["c", "d"]),
            check(client, collection_name, "number_less_or_equal", wf(["numVal"], "LessThanEqual", valueNumber=0.5), ["a", "b"]),
            check(client, collection_name, "bool_equal", wf(["boolVal"], "Equal", valueBoolean=True), ["a", "c"]),
            check(client, collection_name, "text_equal_field", wf(["textVal"], "Equal", valueText="alpha"), ["a"]),
            check(client, collection_name, "text_not_equal_field", wf(["textVal"], "NotEqual", valueText="alpha"), ["b", "c", "d", "e", "f"]),
            check(client, collection_name, "text_like_prefix", wf(["textVal"], "Like", valueText="alpha*"), ["a", "d"]),
            check(client, collection_name, "text_like_suffix", wf(["textVal"], "Like", valueText="*bet"), ["d"]),
            check(client, collection_name, "text_like_contains", wf(["textVal"], "Like", valueText="*pha*"), ["a", "d"]),
            check(client, collection_name, "date_less_or_equal", wf(["dateVal"], "LessThanEqual", valueDate="2024-01-02T00:00:00Z"), ["a", "b"]),
            check(client, collection_name, "null_text", wf(["textVal"], "IsNull", valueBoolean=True), ["e", "f"]),
            check(client, collection_name, "not_null_text", wf(["textVal"], "IsNull", valueBoolean=False), ["a", "b", "c", "d"]),
            check(client, collection_name, "and_bool_text", compound("And", [
                wf(["boolVal"], "Equal", valueBoolean=True),
                wf(["textVal"], "Like", valueText="*a*"),
            ]), ["a", "c"]),
            check(client, collection_name, "or_int_bool", compound("Or", [
                wf(["intVal"], "Equal", valueInt=-10),
                wf(["boolVal"], "Equal", valueBoolean=False),
            ]), ["a", "b", "d"]),
            check(client, collection_name, "not_text_equal", compound("Not", [
                wf(["textVal"], "Equal", valueText="alpha"),
            ]), ["b", "c", "d", "e", "f"]),
        ]

        if all(checks):
            print("Summary: all GraphQL where JSON checks passed on the local Weaviate service.")
            return 0
        print("Summary: GraphQL where JSON mismatches observed.")
        return 1
    finally:
        if created:
            cleanup_collection(client, collection_name)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
