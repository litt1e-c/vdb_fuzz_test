import sys
import time
import uuid

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter

HOST = '127.0.0.1'
PORT = 8080
GRPC_PORT = 50051
COLLECTION_PREFIX = 'ArithmeticExpressionFilterProbe'


def deterministic_uuid(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'weaviate-arithmetic-expression-probe-{label}'))


ROWS = [
    DataObject(
        uuid=deterministic_uuid('a'),
        properties={'tag': 'a', 'score': 100, 'score_plus_100': 200},
        vector=[1.0, 0.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid('b'),
        properties={'tag': 'b', 'score': 950, 'score_plus_100': 1050},
        vector=[0.0, 1.0, 0.0],
    ),
    DataObject(
        uuid=deterministic_uuid('c'),
        properties={'tag': 'c', 'score': 1200, 'score_plus_100': 1300},
        vector=[0.0, 0.0, 1.0],
    ),
]


def create_collection(client, name):
    return client.collections.create(
        name=name,
        properties=[
            Property(name='tag', data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name='score', data_type=DataType.INT, index_filterable=True, index_range_filterable=True),
            Property(name='score_plus_100', data_type=DataType.INT, index_filterable=True, index_range_filterable=True),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def fetch_tags(collection, flt):
    response = collection.query.fetch_objects(filters=flt, limit=20)
    return sorted(obj.properties['tag'] for obj in response.objects)


def run_raw_query(client, query):
    return client.graphql_raw_query(query)


def response_errors(result):
    if result is None:
        return None
    if isinstance(result, dict):
        return result.get('errors')
    try:
        return result['errors']
    except Exception:
        pass
    return getattr(result, 'errors', None)


def extract_tags_from_raw(result, collection_name):
    try:
        rows = result.get[collection_name]
        return sorted(row['tag'] for row in rows)
    except Exception:
        pass
    try:
        rows = result['data']['Get'][collection_name]
        return sorted(row['tag'] for row in rows)
    except Exception:
        return None


def print_section(name, payload):
    print(name)
    print(payload)


def main():
    suffix = f"{int(time.time())}{uuid.uuid4().hex[:8]}"
    collection_name = f"{COLLECTION_PREFIX}{suffix}"
    client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
    created = False
    exit_code = 0
    try:
        collection = create_collection(client, collection_name)
        created = True
        result = collection.data.insert_many(ROWS)
        errors = getattr(result, 'errors', None)
        if errors:
            print(f'Insert errors: {errors}')
            return 2

        control_client = fetch_tags(collection, Filter.by_property('score').less_than(1000))
        print_section(
            'control_client_filter',
            {
                'query': 'Filter.by_property("score").less_than(1000)',
                'expected': ['a', 'b'],
                'observed': control_client,
                'result': 'PASS' if control_client == ['a', 'b'] else 'FAIL',
            },
        )
        if control_client != ['a', 'b']:
            exit_code = 1

        control_raw_query = f'''{{
  Get {{
    {collection_name}(
      where: {{
        path: ["score"]
        operator: LessThan
        valueInt: 1000
      }}
    ) {{
      tag
      score
    }}
  }}
}}'''
        control_raw = run_raw_query(client, control_raw_query)
        control_raw_tags = extract_tags_from_raw(control_raw, collection_name)
        print_section(
            'control_raw_graphql',
            {
                'query': control_raw_query,
                'errors': response_errors(control_raw),
                'observed_tags': control_raw_tags,
                'result': 'PASS' if control_raw_tags == ['a', 'b'] and not response_errors(control_raw) else 'FAIL',
            },
        )
        if control_raw_tags != ['a', 'b'] or response_errors(control_raw):
            exit_code = 1

        try:
            arithmetic_filter = Filter.by_property('score+100').less_than(1000)
            arithmetic_client_tags = fetch_tags(collection, arithmetic_filter)
            print_section(
                'client_filter_arithmetic_like_path',
                {
                    'query': 'Filter.by_property("score+100").less_than(1000)',
                    'observed_tags': arithmetic_client_tags,
                    'interpretation': 'Client accepts the string syntactically, but the server treats it as a property path token, not an arithmetic expression.',
                    'result': 'OBSERVED_NO_ERROR',
                },
            )
        except Exception as exc:
            print_section(
                'client_filter_arithmetic_like_path',
                {
                    'query': 'Filter.by_property("score+100").less_than(1000)',
                    'error': str(exc),
                    'interpretation': 'Arithmetic-like property strings are not accepted as executable expressions.',
                    'result': 'REJECTED',
                },
            )

        try:
            arithmetic_eq_filter = Filter.by_property('score+100').equal(1000)
            arithmetic_eq_tags = fetch_tags(collection, arithmetic_eq_filter)
            print_section(
                'client_filter_arithmetic_like_path_equal',
                {
                    'query': 'Filter.by_property("score+100").equal(1000)',
                    'observed_tags': arithmetic_eq_tags,
                    'interpretation': 'Client accepts the string syntactically, but the server treats it as a property path token, not an arithmetic equality expression.',
                    'result': 'OBSERVED_NO_ERROR',
                },
            )
        except Exception as exc:
            print_section(
                'client_filter_arithmetic_like_path_equal',
                {
                    'query': 'Filter.by_property("score+100").equal(1000)',
                    'error': str(exc),
                    'interpretation': 'Arithmetic-like equality expressions are not accepted as executable filter expressions.',
                    'result': 'REJECTED',
                },
            )

        constant_no_path_query = f'''{{
  Get {{
    {collection_name}(
      where: {{
        operator: Equal
        valueInt: 1
      }}
    ) {{
      tag
      score
    }}
  }}
}}'''
        constant_no_path = run_raw_query(client, constant_no_path_query)
        print_section(
            'raw_graphql_constant_no_path_equal',
            {
                'query': constant_no_path_query,
                'errors': response_errors(constant_no_path),
                'observed_tags': extract_tags_from_raw(constant_no_path, collection_name),
                'interpretation': 'If rejected, constant predicates such as 1=1 are not part of the where grammar because where expects a path-based predicate.',
            },
        )

        constant_empty_path_query = f'''{{
  Get {{
    {collection_name}(
      where: {{
        path: []
        operator: Equal
        valueInt: 1
      }}
    ) {{
      tag
      score
    }}
  }}
}}'''
        constant_empty_path = run_raw_query(client, constant_empty_path_query)
        print_section(
            'raw_graphql_constant_empty_path_equal',
            {
                'query': constant_empty_path_query,
                'errors': response_errors(constant_empty_path),
                'observed_tags': extract_tags_from_raw(constant_empty_path, collection_name),
                'interpretation': 'If rejected, even a syntactic placeholder path cannot turn where into a constant-expression language.',
            },
        )

        raw_arithmetic_path_query = f'''{{
  Get {{
    {collection_name}(
      where: {{
        path: ["score+100"]
        operator: LessThan
        valueInt: 1000
      }}
    ) {{
      tag
      score
    }}
  }}
}}'''
        raw_arithmetic_path = run_raw_query(client, raw_arithmetic_path_query)
        print_section(
            'raw_graphql_path_score_plus_100',
            {
                'query': raw_arithmetic_path_query,
                'errors': response_errors(raw_arithmetic_path),
                'observed_tags': extract_tags_from_raw(raw_arithmetic_path, collection_name),
                'interpretation': 'If rejected, the server does not support computed arithmetic expressions in where.path.',
            },
        )

        raw_arithmetic_equal_query = f'''{{
  Get {{
    {collection_name}(
      where: {{
        path: ["score+100"]
        operator: Equal
        valueInt: 1000
      }}
    ) {{
      tag
      score
    }}
  }}
}}'''
        raw_arithmetic_equal = run_raw_query(client, raw_arithmetic_equal_query)
        print_section(
            'raw_graphql_path_score_plus_100_equal',
            {
                'query': raw_arithmetic_equal_query,
                'errors': response_errors(raw_arithmetic_equal),
                'observed_tags': extract_tags_from_raw(raw_arithmetic_equal, collection_name),
                'interpretation': 'If rejected, arithmetic equalities such as score+100=1000 are not part of the where grammar either.',
            },
        )

        raw_split_path_query = f'''{{
  Get {{
    {collection_name}(
      where: {{
        path: ["score", "+", "100"]
        operator: LessThan
        valueInt: 1000
      }}
    ) {{
      tag
      score
    }}
  }}
}}'''
        raw_split_path = run_raw_query(client, raw_split_path_query)
        print_section(
            'raw_graphql_split_arithmetic_tokens',
            {
                'query': raw_split_path_query,
                'errors': response_errors(raw_split_path),
                'observed_tags': extract_tags_from_raw(raw_split_path, collection_name),
                'interpretation': 'If rejected, the server only accepts property/reference path elements, not infix operators.',
            },
        )

        raw_unknown_operator_query = f'''{{
  Get {{
    {collection_name}(
      where: {{
        path: ["score"]
        operator: Add
        valueInt: 100
      }}
    ) {{
      tag
      score
    }}
  }}
}}'''
        raw_unknown_operator = run_raw_query(client, raw_unknown_operator_query)
        print_section(
            'raw_graphql_unknown_operator_add',
            {
                'query': raw_unknown_operator_query,
                'errors': response_errors(raw_unknown_operator),
                'observed_tags': extract_tags_from_raw(raw_unknown_operator, collection_name),
                'interpretation': 'If rejected, arithmetic operators such as Add are not part of the GraphQL where operator set.',
            },
        )

        control_materialized_query = f'''{{
  Get {{
    {collection_name}(
      where: {{
        path: ["score_plus_100"]
        operator: LessThan
        valueInt: 1000
      }}
    ) {{
      tag
      score_plus_100
    }}
  }}
}}'''
        control_materialized = run_raw_query(client, control_materialized_query)
        control_materialized_tags = extract_tags_from_raw(control_materialized, collection_name)
        print_section(
            'control_materialized_property',
            {
                'query': control_materialized_query,
                'expected': ['a'],
                'errors': response_errors(control_materialized),
                'observed_tags': control_materialized_tags,
                'interpretation': 'Materializing the derived value as a real property works; this is the supported workaround if arithmetic semantics are needed.',
                'result': 'PASS' if control_materialized_tags == ['a'] and not response_errors(control_materialized) else 'FAIL',
            },
        )
        if control_materialized_tags != ['a'] or response_errors(control_materialized):
            exit_code = 1

        tautology_property_query = f'''{{
  Get {{
    {collection_name}(
      where: {{
        path: ["score"]
        operator: GreaterThan
        valueInt: 0
      }}
    ) {{
      tag
      score
    }}
  }}
}}'''
        tautology_property = run_raw_query(client, tautology_property_query)
        tautology_property_tags = extract_tags_from_raw(tautology_property, collection_name)
        print_section(
            'property_based_tautology_workaround',
            {
                'query': tautology_property_query,
                'expected': ['a', 'b', 'c'],
                'errors': response_errors(tautology_property),
                'observed_tags': tautology_property_tags,
                'interpretation': 'Always-true style filters are supported only when expressed as normal property predicates, not as literal constant expressions like 1=1.',
                'result': 'PASS' if tautology_property_tags == ['a', 'b', 'c'] and not response_errors(tautology_property) else 'FAIL',
            },
        )
        if tautology_property_tags != ['a', 'b', 'c'] or response_errors(tautology_property):
            exit_code = 1

        print('summary')
        print({
            'collection': collection_name,
            'conclusion': 'where filters operate on stored property/reference/metadata paths plus documented operators; constant expressions such as 1=1 and arithmetic expressions such as score+100=1000 are not supported as executable filter syntax in this local setup.'
        })
        return exit_code
    finally:
        try:
            if created and collection_name.startswith(COLLECTION_PREFIX):
                client.collections.delete(collection_name)
                print(f'Cleaned up collection: {collection_name}')
        finally:
            client.close()


if __name__ == '__main__':
    sys.exit(main())
