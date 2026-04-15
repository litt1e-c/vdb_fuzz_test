"""
Validate Qdrant datetime filter semantics across explicit datetime index profiles.

This operator is deterministic and focused on a narrow high-value question:
1. `datetime_range` semantics should agree across default / on-disk / principal profiles.
2. Closed-bound equality and inclusive inequalities should preserve UTC-normalized matches.
3. `must_not(datetime_range)` should keep rows where the inner datetime condition is false,
   including missing / null / tested type-mismatch rows.
4. `scroll`, `count`, and exact `query_points` should agree on the same fixture.
5. REST and gRPC should behave the same on this conservative scalar subset.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    DatetimeIndexParams,
    DatetimeIndexType,
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    SearchParams,
    VectorParams,
)


QUERY_VECTOR = [1.0, 0.0]


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def unique_collection_name(run_id: str, prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"datetime_profile_{slugify(run_id, max_len=32)}_{transport}"


def fetch_server_info(host: str, port: int) -> dict[str, object]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{host}:{port}/", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def build_client(args: argparse.Namespace, prefer_grpc: bool) -> QdrantClient:
    return QdrantClient(
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port,
        prefer_grpc=prefer_grpc,
        timeout=args.timeout,
    )


def build_points() -> list[PointStruct]:
    payload_rows = [
        (
            1,
            {
                "dt_default": "1970-01-01T00:00:00Z",
                "dt_on_disk": "1970-01-01T00:00:00Z",
                "dt_principal": "1970-01-01T00:00:00Z",
                "dt_principal_on_disk": "1970-01-01T00:00:00Z",
            },
        ),
        (
            2,
            {
                "dt_default": "2020-01-01T00:00:00Z",
                "dt_on_disk": "2020-01-01T00:00:00Z",
                "dt_principal": "2020-01-01T00:00:00Z",
                "dt_principal_on_disk": "2020-01-01T00:00:00Z",
            },
        ),
        (
            3,
            {
                "dt_default": "2021-01-10T06:40:10Z",
                "dt_on_disk": "2021-01-10T06:40:10Z",
                "dt_principal": "2021-01-10T06:40:10Z",
                "dt_principal_on_disk": "2021-01-10T06:40:10Z",
            },
        ),
        (
            4,
            {
                "dt_default": "2021-01-10T06:40:11Z",
                "dt_on_disk": "2021-01-10T06:40:11Z",
                "dt_principal": "2021-01-10T06:40:11Z",
                "dt_principal_on_disk": "2021-01-10T06:40:11Z",
            },
        ),
        (
            5,
            {
                "dt_default": "2024-02-29T12:34:56Z",
                "dt_on_disk": "2024-02-29T12:34:56Z",
                "dt_principal": "2024-02-29T12:34:56Z",
                "dt_principal_on_disk": "2024-02-29T12:34:56Z",
            },
        ),
        (
            6,
            {
                "dt_default": None,
                "dt_on_disk": None,
                "dt_principal": None,
                "dt_principal_on_disk": None,
            },
        ),
        (
            7,
            {},
        ),
        (
            8,
            {
                "dt_default": 1704067200,
                "dt_on_disk": 1704067200,
                "dt_principal": 1704067200,
                "dt_principal_on_disk": 1704067200,
            },
        ),
        (
            9,
            {
                "dt_default": "2021-01-10T07:40:10+01:00",
                "dt_on_disk": "2021-01-10T07:40:10+01:00",
                "dt_principal": "2021-01-10T07:40:10+01:00",
                "dt_principal_on_disk": "2021-01-10T07:40:10+01:00",
            },
        ),
        (
            10,
            {
                "dt_default": "2021-01-10 06:40:10+0000",
                "dt_on_disk": "2021-01-10 06:40:10+0000",
                "dt_principal": "2021-01-10 06:40:10+0000",
                "dt_principal_on_disk": "2021-01-10 06:40:10+0000",
            },
        ),
    ]
    return [
        PointStruct(
            id=point_id,
            vector=[1.0, 0.0] if point_id % 2 else [0.0, 1.0],
            payload=payload,
        )
        for point_id, payload in payload_rows
    ]


def index_specs() -> list[tuple[str, object]]:
    return [
        ("dt_default", DatetimeIndexParams(type=DatetimeIndexType.DATETIME)),
        ("dt_on_disk", DatetimeIndexParams(type=DatetimeIndexType.DATETIME, on_disk=True)),
        ("dt_principal", DatetimeIndexParams(type=DatetimeIndexType.DATETIME, is_principal=True)),
        (
            "dt_principal_on_disk",
            DatetimeIndexParams(type=DatetimeIndexType.DATETIME, is_principal=True, on_disk=True),
        ),
    ]


def scroll_ids(client: QdrantClient, collection_name: str, scroll_filter: Filter) -> list[int]:
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=64,
        with_payload=False,
        with_vectors=False,
    )
    return sorted(int(point.id) for point in points)


def count_hits(client: QdrantClient, collection_name: str, scroll_filter: Filter) -> int:
    return int(
        client.count(
            collection_name=collection_name,
            count_filter=scroll_filter,
            exact=True,
        ).count
    )


def query_ids(client: QdrantClient, collection_name: str, query_filter: Filter) -> list[int]:
    result = client.query_points(
        collection_name=collection_name,
        query=QUERY_VECTOR,
        query_filter=query_filter,
        limit=64,
        with_payload=False,
        with_vectors=False,
        search_params=SearchParams(exact=True),
    )
    return sorted(int(point.id) for point in result.points)


def run_transport(args: argparse.Namespace, prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(args.run_id, prefer_grpc)
    client = build_client(args, prefer_grpc)
    all_passed = True

    field_cases = [
        (
            "dt_default",
            "default",
            [
                (
                    "lte_exact_boundary",
                    Filter(
                        must=[
                            FieldCondition(
                                key="dt_default",
                                range=DatetimeRange(lte="2021-01-10T06:40:10Z"),
                            )
                        ]
                    ),
                    [1, 2, 3, 9, 10],
                    "Inclusive upper bound keeps all UTC-equal boundary rows.",
                ),
                (
                    "exact_closed_boundary",
                    Filter(
                        must=[
                            FieldCondition(
                                key="dt_default",
                                range=DatetimeRange(
                                    gte="2021-01-10T06:40:10Z",
                                    lte="2021-01-10T06:40:10Z",
                                ),
                            )
                        ]
                    ),
                    [3, 9, 10],
                    "Closed bounds preserve UTC-normalized equality on tested input formats.",
                ),
                (
                    "not_gte_2020",
                    Filter(
                        must_not=[
                            FieldCondition(
                                key="dt_default",
                                range=DatetimeRange(gte="2020-01-01T00:00:00Z"),
                            )
                        ]
                    ),
                    [1, 6, 7, 8],
                    "must_not keeps older rows plus null / missing / tested integer type mismatch rows.",
                ),
            ],
        ),
        (
            "dt_on_disk",
            "on_disk",
            [
                (
                    "lte_exact_boundary",
                    Filter(
                        must=[
                            FieldCondition(
                                key="dt_on_disk",
                                range=DatetimeRange(lte="2021-01-10T06:40:10Z"),
                            )
                        ]
                    ),
                    [1, 2, 3, 9, 10],
                    "Inclusive upper bound matches the same UTC-normalized set on on-disk datetime index.",
                ),
                (
                    "exact_closed_boundary",
                    Filter(
                        must=[
                            FieldCondition(
                                key="dt_on_disk",
                                range=DatetimeRange(
                                    gte="2021-01-10T06:40:10Z",
                                    lte="2021-01-10T06:40:10Z",
                                ),
                            )
                        ]
                    ),
                    [3, 9, 10],
                    "Closed bounds preserve UTC-normalized equality on on-disk datetime index.",
                ),
                (
                    "not_gte_2020",
                    Filter(
                        must_not=[
                            FieldCondition(
                                key="dt_on_disk",
                                range=DatetimeRange(gte="2020-01-01T00:00:00Z"),
                            )
                        ]
                    ),
                    [1, 6, 7, 8],
                    "must_not keeps older rows plus null / missing / tested integer type mismatch rows on on-disk index.",
                ),
            ],
        ),
        (
            "dt_principal",
            "principal",
            [
                (
                    "lte_exact_boundary",
                    Filter(
                        must=[
                            FieldCondition(
                                key="dt_principal",
                                range=DatetimeRange(lte="2021-01-10T06:40:10Z"),
                            )
                        ]
                    ),
                    [1, 2, 3, 9, 10],
                    "Inclusive upper bound matches the same UTC-normalized set on principal datetime index.",
                ),
                (
                    "exact_closed_boundary",
                    Filter(
                        must=[
                            FieldCondition(
                                key="dt_principal",
                                range=DatetimeRange(
                                    gte="2021-01-10T06:40:10Z",
                                    lte="2021-01-10T06:40:10Z",
                                ),
                            )
                        ]
                    ),
                    [3, 9, 10],
                    "Closed bounds preserve UTC-normalized equality on principal datetime index.",
                ),
                (
                    "not_gte_2020",
                    Filter(
                        must_not=[
                            FieldCondition(
                                key="dt_principal",
                                range=DatetimeRange(gte="2020-01-01T00:00:00Z"),
                            )
                        ]
                    ),
                    [1, 6, 7, 8],
                    "must_not keeps older rows plus null / missing / tested integer type mismatch rows on principal index.",
                ),
            ],
        ),
        (
            "dt_principal_on_disk",
            "principal_on_disk",
            [
                (
                    "lte_exact_boundary",
                    Filter(
                        must=[
                            FieldCondition(
                                key="dt_principal_on_disk",
                                range=DatetimeRange(lte="2021-01-10T06:40:10Z"),
                            )
                        ]
                    ),
                    [1, 2, 3, 9, 10],
                    "Inclusive upper bound matches the same UTC-normalized set on principal+on-disk datetime index.",
                ),
                (
                    "exact_closed_boundary",
                    Filter(
                        must=[
                            FieldCondition(
                                key="dt_principal_on_disk",
                                range=DatetimeRange(
                                    gte="2021-01-10T06:40:10Z",
                                    lte="2021-01-10T06:40:10Z",
                                ),
                            )
                        ]
                    ),
                    [3, 9, 10],
                    "Closed bounds preserve UTC-normalized equality on principal+on-disk datetime index.",
                ),
                (
                    "not_gte_2020",
                    Filter(
                        must_not=[
                            FieldCondition(
                                key="dt_principal_on_disk",
                                range=DatetimeRange(gte="2020-01-01T00:00:00Z"),
                            )
                        ]
                    ),
                    [1, 6, 7, 8],
                    "must_not keeps older rows plus null / missing / tested integer type mismatch rows on principal+on-disk index.",
                ),
            ],
        ),
    ]

    try:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        for field_name, field_schema in index_specs():
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

        print(f"\n--- datetime profile validation ({transport}) ---")
        print(f"collection={collection_name}")
        for _, profile_name, cases in field_cases:
            for test_name, query_filter, expected_ids, note in cases:
                scroll_result = scroll_ids(client, collection_name, query_filter)
                count_result = count_hits(client, collection_name, query_filter)
                query_result = query_ids(client, collection_name, query_filter)
                passed = (
                    scroll_result == expected_ids
                    and query_result == expected_ids
                    and count_result == len(expected_ids)
                )
                status = "PASS" if passed else "FAIL"
                print(
                    f"{status} profile={profile_name} test={test_name} "
                    f"scroll={scroll_result} count={count_result} query={query_result}"
                )
                if passed:
                    print(f"  note={note}")
                else:
                    all_passed = False
                    print(f"  expected={expected_ids}")
                    print(f"  note={note}")
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")

    return all_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qdrant datetime range semantics across index profiles.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", dest="grpc_port", type=int, default=6334)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--run-id", default="datetime-profile")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        info = fetch_server_info(args.host, args.port)
    except Exception:
        info = {}

    print("Qdrant datetime profile operator validation")
    print(f"server_version={info.get('version', 'unknown')}")
    print(f"server_commit={info.get('commit', 'unknown')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(args, prefer_grpc=False)
    grpc_ok = run_transport(args, prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all datetime profile checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one datetime profile check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
