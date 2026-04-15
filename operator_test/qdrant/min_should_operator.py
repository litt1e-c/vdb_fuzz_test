"""
Validate Qdrant `min_should` filter semantics.

This operator focuses on the source-backed semantics that at least `min_count`
conditions from a list must match. It keeps the fixture small and deterministic
so the result is suitable as an oracle root before adding any random fuzzer
generation for `min_should`.
"""

from __future__ import annotations

import argparse
import random
import time
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    MinShould,
    PointStruct,
    Range,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334
RUN_ID: str | None = None
_COLLECTION_COUNTER = 0


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def unique_collection_name(prefer_grpc: bool) -> str:
    global _COLLECTION_COUNTER
    transport = "grpc" if prefer_grpc else "rest"
    if RUN_ID:
        _COLLECTION_COUNTER += 1
        return f"min_should_operator_{slugify(RUN_ID, max_len=36)}_{transport}_{_COLLECTION_COUNTER:02d}"
    return f"min_should_operator_{transport}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


def fetch_server_info() -> dict[str, object]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{HOST}:{PORT}/", timeout=5) as resp:
        import json

        return json.loads(resp.read().decode("utf-8"))


def build_client(prefer_grpc: bool) -> QdrantClient:
    return QdrantClient(
        host=HOST,
        port=PORT,
        grpc_port=GRPC_PORT,
        prefer_grpc=prefer_grpc,
        timeout=30,
    )


def build_points() -> list[PointStruct]:
    return [
        PointStruct(id=1, vector=[1.0, 0.0], payload={"city": "London", "color": "red", "score": 10}),
        PointStruct(id=2, vector=[0.0, 1.0], payload={"city": "London", "color": "green", "score": 5}),
        PointStruct(id=3, vector=[1.0, 1.0], payload={"city": "Berlin", "color": "red", "score": -3}),
        PointStruct(id=4, vector=[0.5, 0.5], payload={"city": "Berlin", "color": "blue"}),
        PointStruct(id=5, vector=[0.2, 0.2], payload={"score": None}),
        PointStruct(id=6, vector=[0.3, 0.3], payload={}),
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


def count_total(client: QdrantClient, collection_name: str, count_filter: Filter) -> int:
    return int(
        client.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=True,
        ).count
    )


def city(value: str) -> FieldCondition:
    return FieldCondition(key="city", match=MatchValue(value=value))


def color(value: str) -> FieldCondition:
    return FieldCondition(key="color", match=MatchValue(value=value))


def score_gt(value: int) -> FieldCondition:
    return FieldCondition(key="score", range=Range(gt=value))


def score_lt(value: int) -> FieldCondition:
    return FieldCondition(key="score", range=Range(lt=value))


def run_transport(prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(prefer_grpc)
    client = build_client(prefer_grpc)
    all_passed = True

    tests = [
        (
            "min_one_matches_should_like_or",
            Filter(
                min_should=MinShould(
                    conditions=[city("London"), color("red"), score_lt(0)],
                    min_count=1,
                )
            ),
            [1, 2, 3],
            "min_count=1 behaves like OR over the listed conditions in this ordinary subset.",
        ),
        (
            "min_zero_of_three_vacuously_true",
            Filter(
                min_should=MinShould(
                    conditions=[city("London"), color("red"), score_lt(0)],
                    min_count=0,
                )
            ),
            [1, 2, 3, 4, 5, 6],
            "On the validated local build, min_count=0 is vacuously true even when no child condition matches.",
        ),
        (
            "min_zero_of_empty_vacuously_true",
            Filter(min_should=MinShould(conditions=[], min_count=0)),
            [1, 2, 3, 4, 5, 6],
            "On the validated local build, an empty min_should list with min_count=0 is also vacuously true.",
        ),
        (
            "min_positive_of_empty_treated_as_noop",
            Filter(min_should=MinShould(conditions=[], min_count=1)),
            [1, 2, 3, 4, 5, 6],
            "The validated REST and gRPC API behavior treats an empty min_should condition list as a no-op even with positive min_count.",
        ),
        (
            "min_two_of_three",
            Filter(
                min_should=MinShould(
                    conditions=[city("London"), color("red"), score_lt(0)],
                    min_count=2,
                )
            ),
            [1, 3],
            "A point must satisfy at least two of the three conditions.",
        ),
        (
            "min_count_greater_than_condition_count_false",
            Filter(
                min_should=MinShould(
                    conditions=[city("London"), color("red"), score_lt(0)],
                    min_count=4,
                )
            ),
            [],
            "When min_count exceeds the number of listed conditions, the validated local behavior is a total miss.",
        ),
        (
            "min_three_of_three_no_match",
            Filter(
                min_should=MinShould(
                    conditions=[city("London"), color("red"), score_lt(0)],
                    min_count=3,
                )
            ),
            [],
            "No fixture point satisfies all three ordinary conditions.",
        ),
        (
            "min_all_of_three_positive",
            Filter(
                min_should=MinShould(
                    conditions=[city("London"), color("red"), score_gt(7)],
                    min_count=3,
                )
            ),
            [1],
            "When min_count equals the number of conditions, min_should behaves like MUST over the same list.",
        ),
        (
            "min_should_combines_with_must",
            Filter(
                must=[city("London")],
                min_should=MinShould(
                    conditions=[color("red"), score_gt(7)],
                    min_count=2,
                ),
            ),
            [1],
            "Top-level must and min_should are conjunctive, matching the source checker path.",
        ),
        (
            "min_should_with_must_not_sibling",
            Filter(
                must_not=[city("Berlin")],
                min_should=MinShould(
                    conditions=[color("red"), score_gt(7)],
                    min_count=1,
                ),
            ),
            [1],
            "Top-level must_not is conjunctive with min_should and removes rows even when they satisfy one min_should child.",
        ),
        (
            "min_should_with_should_sibling_is_conjunctive",
            Filter(
                should=[city("London")],
                min_should=MinShould(
                    conditions=[color("red"), score_gt(7)],
                    min_count=2,
                ),
            ),
            [1],
            "Top-level should and min_should are both checked, so the result is their conjunction rather than a union.",
        ),
        (
            "min_should_nested_filter_conditions",
            Filter(
                min_should=MinShould(
                    conditions=[
                        Filter(must=[city("Berlin"), color("red")]),
                        score_gt(0),
                    ],
                    min_count=1,
                )
            ),
            [1, 2, 3],
            "Nested Filter conditions participate as ordinary min_should conditions.",
        ),
        (
            "min_should_nested_filter_all_of_two",
            Filter(
                min_should=MinShould(
                    conditions=[
                        Filter(must=[city("London"), color("red")]),
                        score_gt(7),
                    ],
                    min_count=2,
                )
            ),
            [1],
            "Nested Filter children still count as one satisfied min_should child, so min_count=2 requires both the inner filter and the sibling condition.",
        ),
        (
            "min_should_excludes_missing_null_for_ordinary_predicates",
            Filter(
                min_should=MinShould(
                    conditions=[score_gt(0), score_lt(0)],
                    min_count=1,
                )
            ),
            [1, 2, 3],
            "Missing/null payloads do not satisfy ordinary range predicates inside min_should.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- min_should operator validation ({transport}) ---")
        print(f"collection={collection_name}")
        for name, scroll_filter, expected_ids, note in tests:
            actual_ids = scroll_ids(client, collection_name, scroll_filter)
            actual_count = count_total(client, collection_name, scroll_filter)
            passed = actual_ids == expected_ids and actual_count == len(expected_ids)
            if not passed:
                all_passed = False
            print(
                f"{name}: {'PASS' if passed else 'FAIL'} | "
                f"expected={expected_ids} | actual={actual_ids} | "
                f"expected_count={len(expected_ids)} | actual_count={actual_count} | note={note}"
            )
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")

    return all_passed


def main(argv: list[str] | None = None) -> int:
    global HOST, PORT, GRPC_PORT, RUN_ID

    parser = argparse.ArgumentParser(description="Validate Qdrant min_should filter semantics")
    parser.add_argument("--host", default=HOST, help="Qdrant REST host")
    parser.add_argument("--port", type=int, default=PORT, help="Qdrant REST port")
    parser.add_argument("--grpc-port", type=int, default=GRPC_PORT, dest="grpc_port", help="Qdrant gRPC port")
    parser.add_argument("--run-id", default=None, help="Optional deterministic run id for collection naming")
    args = parser.parse_args(argv)

    HOST = args.host
    PORT = int(args.port)
    GRPC_PORT = int(args.grpc_port)
    RUN_ID = args.run_id

    try:
        server_info = fetch_server_info()
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant min_should operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all min_should operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one min_should operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
