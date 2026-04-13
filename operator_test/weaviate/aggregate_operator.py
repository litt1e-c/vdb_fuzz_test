import math
import sys
import time
import uuid

import pandas as pd
import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.collections.classes.aggregate import Metrics


HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_PREFIX = "AggregateOperatorValidation"


FIELDS = {
    "intVal": "integer",
    "numVal": "number",
    "boolVal": "boolean",
    "textVal": "text",
    "dateVal": "date",
    "dateArr": "date",
    "intArr": "integer",
    "numArr": "number",
    "boolArr": "boolean",
    "textArr": "text",
}

ARRAY_FIELDS = {"dateArr", "intArr", "numArr", "boolArr", "textArr"}

ROW_PROPS = [
    {
        "tag": "a",
        "bucket": 0,
        "intVal": -1024,
        "numVal": -1e-12,
        "boolVal": True,
        "textVal": "alpha",
        "dateVal": "1969-12-31T23:59:59Z",
        "dateArr": ["1969-12-31T23:59:59Z", "2024-01-01T00:00:00Z"],
        "intArr": [-1024, 0],
        "numArr": [-1e-12, 0.0],
        "boolArr": [True, False],
        "textArr": ["alpha", "beta"],
    },
    {
        "tag": "b",
        "bucket": 0,
        "intVal": 0,
        "numVal": 0.0,
        "boolVal": True,
        "textVal": "alpha",
        "dateVal": "2024-01-01T00:00:00Z",
        "dateArr": ["2024-01-01T00:00:00Z"],
        "intArr": [0],
        "numArr": [0.0],
        "boolArr": [True],
        "textArr": ["alpha"],
    },
    {
        "tag": "c",
        "bucket": 1,
        "intVal": 7,
        "numVal": 4.75,
        "boolVal": False,
        "textVal": "gamma",
        "dateVal": "2024-01-05T00:00:00.25Z",
        "dateArr": ["2024-01-05T00:00:00.25Z"],
        "intArr": [7, 7],
        "numArr": [4.75, 4.75],
        "boolArr": [False, False],
        "textArr": ["gamma", "gamma"],
    },
    {
        "tag": "d",
        "bucket": 1,
        "intVal": 7,
        "numVal": 4.75,
        "boolVal": True,
        "textVal": "gamma",
        "dateVal": "2024-01-05T00:00:00.25Z",
        "dateArr": [],
        "intArr": [],
        "numArr": [],
        "boolArr": [],
        "textArr": [],
    },
    {
        "tag": "e",
        "bucket": 2,
        "intVal": 1024,
        "numVal": 1e6,
        "boolVal": None,
        "textVal": "gamma",
        "dateVal": "2026-12-31T23:59:59.999999Z",
        "dateArr": ["2024-01-05T00:00:00.25Z", "2026-12-31T23:59:59.999999Z"],
        "intArr": [1024],
        "numArr": [1e6],
        "boolArr": None,
        "textArr": ["gamma"],
    },
    {
        "tag": "f",
        "bucket": 2,
    },
]


def deterministic_uuid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"weaviate-aggregate-operator-{label}"))


def create_collection(client, name):
    return client.collections.create(
        name=name,
        properties=[
            Property(name="tag", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="bucket", data_type=DataType.INT, index_filterable=True, index_range_filters=True),
            Property(name="intVal", data_type=DataType.INT, index_filterable=True, index_range_filters=True),
            Property(name="numVal", data_type=DataType.NUMBER, index_filterable=True, index_range_filters=True),
            Property(name="boolVal", data_type=DataType.BOOL, index_filterable=True),
            Property(name="textVal", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_filterable=True),
            Property(name="dateVal", data_type=DataType.DATE, index_filterable=True, index_range_filters=True),
            Property(name="dateArr", data_type=DataType.DATE_ARRAY, index_filterable=True),
            Property(name="intArr", data_type=DataType.INT_ARRAY, index_filterable=True),
            Property(name="numArr", data_type=DataType.NUMBER_ARRAY, index_filterable=True),
            Property(name="boolArr", data_type=DataType.BOOL_ARRAY, index_filterable=True),
            Property(name="textArr", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD, index_filterable=True),
        ],
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_null_state=True),
    )


def insert_rows(collection):
    objects = [
        DataObject(
            uuid=deterministic_uuid(row["tag"]),
            properties=dict(row),
            vector=[1.0, float(idx + 1) / 10.0, float((idx % 3) + 1) / 10.0],
        )
        for idx, row in enumerate(ROW_PROPS)
    ]
    result = collection.data.insert_many(objects)
    errors = getattr(result, "errors", None)
    if errors:
        raise RuntimeError(f"insert_many errors: {errors}")


def format_rfc3339_nano(value):
    ts = pd.to_datetime(value, utc=True)
    epoch_ns = int(ts.value)
    sec, nanos = divmod(epoch_ns, 1_000_000_000)
    dt = pd.Timestamp(sec, unit="s", tz="UTC").to_pydatetime()
    out = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if nanos:
        out += f".{nanos:09d}".rstrip("0")
    return out + "Z"


def flatten_values(rows, field):
    values = []
    for row in rows:
        value = row.get(field)
        if field in ARRAY_FIELDS:
            if isinstance(value, list):
                values.extend(value)
        elif value is not None:
            values.append(value)
    return values


def numeric_expected(values, integer=False):
    if not values:
        return {"count": 0, "maximum": 0, "mean": 0.0, "median": 0.0, "minimum": 0, "mode": 0, "sum_": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = float(ordered[mid]) if len(ordered) % 2 else float(ordered[mid - 1] + ordered[mid]) / 2.0
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    max_count = max(counts.values())
    mode = min(value for value, count in counts.items() if count == max_count)
    total = sum(values)
    result = {
        "count": len(values),
        "maximum": max(values),
        "mean": float(total) / float(len(values)),
        "median": median,
        "minimum": min(values),
        "mode": mode,
        "sum_": total,
    }
    if integer:
        for key in ["maximum", "minimum", "mode", "sum_"]:
            result[key] = int(result[key])
    return result


def boolean_expected(values):
    if not values:
        return {"count": 0, "percentage_false": math.nan, "percentage_true": math.nan, "total_false": 0, "total_true": 0}
    total_true = sum(1 for value in values if bool(value))
    total_false = len(values) - total_true
    return {
        "count": len(values),
        "percentage_false": total_false / len(values),
        "percentage_true": total_true / len(values),
        "total_false": total_false,
        "total_true": total_true,
    }


def date_expected(values):
    if not values:
        return {"count": 0, "maximum": "", "median": "", "minimum": "", "mode": "", "_mode_candidates": {""}}
    ordered = sorted(int(pd.to_datetime(value, utc=True).value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median_ns = ordered[mid]
    else:
        median_ns = ordered[mid - 1] + (ordered[mid] - ordered[mid - 1]) // 2
    counts = {}
    for epoch_ns in ordered:
        counts[epoch_ns] = counts.get(epoch_ns, 0) + 1
    max_count = max(counts.values())
    mode_candidates = {
        format_rfc3339_nano(pd.Timestamp(epoch_ns, unit="ns", tz="UTC"))
        for epoch_ns, count in counts.items()
        if count == max_count
    }
    return {
        "count": len(ordered),
        "maximum": format_rfc3339_nano(pd.Timestamp(ordered[-1], unit="ns", tz="UTC")),
        "median": format_rfc3339_nano(pd.Timestamp(median_ns, unit="ns", tz="UTC")),
        "minimum": format_rfc3339_nano(pd.Timestamp(ordered[0], unit="ns", tz="UTC")),
        "mode": sorted(mode_candidates)[0],
        "_mode_candidates": mode_candidates,
    }


def text_expected(values, limit=5):
    if not values:
        return {"count": 0, "top_occurrences": []}
    counts = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return {"count": len(values), "top_occurrences": [{"value": value, "count": count} for value, count in ordered]}


def expected_for(rows, field):
    family = FIELDS[field]
    values = flatten_values(rows, field)
    if family == "integer":
        return numeric_expected([int(value) for value in values], integer=True)
    if family == "number":
        return numeric_expected([float(value) for value in values], integer=False)
    if family == "boolean":
        return boolean_expected([bool(value) for value in values])
    if family == "date":
        return date_expected(values)
    if family == "text":
        return text_expected(values)
    raise AssertionError(f"unknown family: {family}")


def metrics_for(field):
    family = FIELDS[field]
    if family == "integer":
        return Metrics(field).integer(count=True, maximum=True, mean=True, median=True, minimum=True, mode=True, sum_=True)
    if family == "number":
        return Metrics(field).number(count=True, maximum=True, mean=True, median=True, minimum=True, mode=True, sum_=True)
    if family == "boolean":
        return Metrics(field).boolean(count=True, percentage_false=True, percentage_true=True, total_false=True, total_true=True)
    if family == "date":
        return Metrics(field).date_(count=True, maximum=True, median=True, minimum=True, mode=True)
    if family == "text":
        return Metrics(field).text(count=True, top_occurrences_count=True, top_occurrences_value=True, limit=5)
    raise AssertionError(f"unknown family: {family}")


def values_match(actual, expected):
    if isinstance(expected, float):
        if math.isnan(expected):
            return isinstance(actual, float) and math.isnan(actual)
        return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def compare_property(field, actual, expected):
    mismatches = []
    family = FIELDS[field]
    if family in {"integer", "number"}:
        keys = ["count", "maximum", "mean", "median", "minimum", "mode", "sum_"]
    elif family == "boolean":
        keys = ["count", "percentage_false", "percentage_true", "total_false", "total_true"]
    elif family == "date":
        keys = ["count", "maximum", "median", "minimum"]
    else:
        keys = ["count"]
    for key in keys:
        observed = getattr(actual, key)
        if not values_match(observed, expected[key]):
            mismatches.append(f"{key}: expected {expected[key]!r}, got {observed!r}")
    if family == "date":
        observed_mode = getattr(actual, "mode")
        if observed_mode not in expected.get("_mode_candidates", {expected["mode"]}):
            mismatches.append(
                f"mode: expected one of {sorted(expected.get('_mode_candidates', {expected['mode']}))!r}, got {observed_mode!r}"
            )
    if family == "text":
        observed_top = sorted(
            [{"value": occurrence.value, "count": int(occurrence.count)} for occurrence in actual.top_occurrences],
            key=lambda item: (-item["count"], item["value"]),
        )
        expected_top = sorted(expected["top_occurrences"], key=lambda item: (-item["count"], item["value"]))
        if observed_top != expected_top:
            mismatches.append(f"top_occurrences: expected {expected_top!r}, got {observed_top!r}")
    return mismatches


def print_case(name, expected_rows, response):
    print(name)
    ok = True
    expected_total = len(expected_rows)
    print(f"  total_count expected={expected_total} observed={response.total_count}")
    if response.total_count != expected_total:
        ok = False
    for field in FIELDS:
        expected = expected_for(expected_rows, field)
        actual = response.properties[field]
        mismatches = compare_property(field, actual, expected)
        print(f"  {field}: {'PASS' if not mismatches else 'FAIL'}")
        for mismatch in mismatches:
            print(f"    {mismatch}")
        ok = ok and not mismatches
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def run_case(collection, name, filters, expected_rows):
    response = collection.aggregate.over_all(
        filters=filters,
        total_count=True,
        return_metrics=[metrics_for(field) for field in FIELDS],
    )
    return print_case(name, expected_rows, response)


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
            run_case(collection, "aggregate_all_rows", None, ROW_PROPS),
            run_case(
                collection,
                "aggregate_where_bucket_zero",
                Filter.by_property("bucket").equal(0),
                [row for row in ROW_PROPS if row.get("bucket") == 0],
            ),
            run_case(
                collection,
                "aggregate_where_array_contains",
                Filter.by_property("intArr").contains_any([7]),
                [row for row in ROW_PROPS if 7 in row.get("intArr", [])],
            ),
            run_case(
                collection,
                "aggregate_where_text_null",
                Filter.by_property("textVal").is_none(True),
                [row for row in ROW_PROPS if row.get("textVal") is None],
            ),
            run_case(
                collection,
                "aggregate_empty_filter",
                Filter.by_property("bucket").equal(99),
                [],
            ),
        ]

        if all(checks):
            print("Summary: all aggregate operator checks passed on the local Weaviate service.")
            return 0
        print("Summary: aggregate operator mismatches observed.")
        return 1
    finally:
        if created:
            cleanup_collection(client, collection_name)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
