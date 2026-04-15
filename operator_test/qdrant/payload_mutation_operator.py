"""
Validate payload mutation semantics that are especially relevant to scalar fuzzing.

This operator focuses on deterministic, documented payload update paths:
1. `set_payload` merges selected keys and preserves untouched keys.
2. `overwrite_payload` replaces the whole payload of selected points.
3. `delete_payload` removes selected keys, including when points are selected by filter.
4. `clear_payload` removes the whole payload and changes subsequent filter behavior.
5. `scroll`, exact `count`, and exact `query_points` should agree after each
   mutation phase.

The oracle only checks post-mutation filter semantics that are already validated
elsewhere in this repository (`match`, `range`, `is_empty`, `is_null`).
"""

from __future__ import annotations

import argparse
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    IsNullCondition,
    MatchValue,
    PayloadField,
    PayloadSchemaType,
    PointStruct,
    Range,
    SearchParams,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334
QUERY_VECTOR = [1.0, 0.0]
RUN_ID = "payload-mutation"


INDEX_SCHEMAS = {
    "status": PayloadSchemaType.KEYWORD,
    "score": PayloadSchemaType.INTEGER,
    "city": PayloadSchemaType.KEYWORD,
    "flag": PayloadSchemaType.BOOL,
}


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"payload_mutation_{slugify(RUN_ID, max_len=36)}_{transport}"


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
        PointStruct(
            id=1,
            vector=[1.0, 0.0],
            payload={"status": "draft", "score": 10, "city": "rome", "flag": True},
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={"status": "draft", "score": 20, "city": "rome", "flag": False},
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={"status": "published", "score": 30, "city": "paris", "flag": True},
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={"status": "archived", "score": 40, "city": "berlin", "flag": False},
        ),
        PointStruct(id=5, vector=[0.2, 0.8], payload={}),
        PointStruct(
            id=6,
            vector=[0.8, 0.2],
            payload={"status": None, "score": 60, "city": "madrid", "flag": True},
        ),
    ]


def create_indexes(client: QdrantClient, collection_name: str) -> None:
    for field_name, field_schema in INDEX_SCHEMAS.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


def rebuild_index(client: QdrantClient, collection_name: str, field_name: str) -> None:
    client.delete_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        wait=True,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        field_schema=INDEX_SCHEMAS[field_name],
        wait=True,
    )


def scroll_ids(client: QdrantClient, collection_name: str, scroll_filter: Filter) -> list[int]:
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=64,
        with_payload=False,
        with_vectors=False,
    )
    return sorted(int(point.id) for point in points)


def count_hits(client: QdrantClient, collection_name: str, count_filter: Filter) -> int:
    return int(
        client.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=True,
        ).count
    )


def query_ids_exact(client: QdrantClient, collection_name: str, query_filter: Filter) -> list[int]:
    response = client.query_points(
        collection_name=collection_name,
        query=QUERY_VECTOR,
        query_filter=query_filter,
        limit=64,
        with_payload=False,
        with_vectors=False,
        search_params=SearchParams(exact=True),
    )
    return sorted(int(point.id) for point in response.points)


def capture_filter_result(client: QdrantClient, collection_name: str, scroll_filter: Filter) -> dict[str, object]:
    scroll = scroll_ids(client, collection_name, scroll_filter)
    count = count_hits(client, collection_name, scroll_filter)
    query = query_ids_exact(client, collection_name, scroll_filter)
    return {
        "scroll_ids": scroll,
        "count": count,
        "query_ids": query,
    }


def match_filter(key: str, value) -> Filter:
    return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])


def score_range_filter(gte: int) -> Filter:
    return Filter(must=[FieldCondition(key="score", range=Range(gte=gte))])


def is_empty_filter(key: str) -> Filter:
    return Filter(must=[IsEmptyCondition(is_empty=PayloadField(key=key))])


def is_null_filter(key: str) -> Filter:
    return Filter(must=[IsNullCondition(is_null=PayloadField(key=key))])


def check_case(
    client: QdrantClient,
    collection_name: str,
    name: str,
    scroll_filter: Filter,
    expected_ids: list[int],
    note: str,
) -> bool:
    actual = capture_filter_result(client, collection_name, scroll_filter)
    expected = {
        "scroll_ids": expected_ids,
        "count": len(expected_ids),
        "query_ids": expected_ids,
    }
    ok = actual == expected
    print(f"{name}: {'PASS' if ok else 'FAIL'} | expected={expected} | actual={actual} | note={note}")
    return ok


def run_transport(prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(prefer_grpc)
    client = build_client(prefer_grpc)
    all_passed = True

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        create_indexes(client, collection_name)
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- payload mutation operator validation ({transport}) ---")
        print(f"collection={collection_name}")

        for name, scroll_filter, expected_ids, note in [
            (
                "baseline_draft_match",
                match_filter("status", "draft"),
                [1, 2],
                "The initial exact-match baseline is stable before payload mutations.",
            ),
            (
                "baseline_score_ge_30",
                score_range_filter(30),
                [3, 4, 6],
                "The initial numeric baseline captures later overwrite/delete effects.",
            ),
            (
                "baseline_status_is_empty",
                is_empty_filter("status"),
                [5, 6],
                "Missing and explicit null are empty in the validated subset.",
            ),
        ]:
            all_passed &= check_case(client, collection_name, name, scroll_filter, expected_ids, note)

        client.set_payload(
            collection_name=collection_name,
            payload={"status": "published", "score": 35},
            points=[2],
            wait=True,
        )
        for name, scroll_filter, expected_ids, note in [
            (
                "after_set_status_published",
                match_filter("status", "published"),
                [2, 3],
                "set_payload merges selected fields and keeps the pre-existing city for later filter-based updates.",
            ),
            (
                "after_set_score_ge_30",
                score_range_filter(30),
                [2, 3, 4, 6],
                "Merged numeric updates immediately affect ordinary range filters.",
            ),
        ]:
            all_passed &= check_case(client, collection_name, name, scroll_filter, expected_ids, note)

        client.set_payload(
            collection_name=collection_name,
            payload={"flag": True},
            points=match_filter("city", "rome"),
            wait=True,
        )
        for name, scroll_filter, expected_ids, note in [
            (
                "after_filter_set_flag_true",
                match_filter("flag", True),
                [1, 2, 3, 6],
                "set_payload selected by filter updates all matching points and preserves later non-matching rows.",
            ),
        ]:
            all_passed &= check_case(client, collection_name, name, scroll_filter, expected_ids, note)

        client.overwrite_payload(
            collection_name=collection_name,
            payload={"status": "review", "city": "lisbon"},
            points=[3],
            wait=True,
        )
        for name, scroll_filter, expected_ids, note in [
            (
                "after_overwrite_status_review",
                match_filter("status", "review"),
                [3],
                "overwrite_payload replaces the full payload rather than merging it.",
            ),
            (
                "after_overwrite_score_ge_30",
                score_range_filter(30),
                [2, 4, 6],
                "After overwrite, point 3 no longer matches numeric score filters because `score` was removed.",
            ),
            (
                "after_overwrite_city_lisbon",
                match_filter("city", "lisbon"),
                [3],
                "The overwritten payload becomes the new authoritative scalar state.",
            ),
            (
                "after_overwrite_flag_true",
                match_filter("flag", True),
                [1, 2, 6],
                "overwrite_payload removes previously indexed keys that are omitted from the new payload.",
            ),
        ]:
            all_passed &= check_case(client, collection_name, name, scroll_filter, expected_ids, note)

        rebuild_index(client, collection_name, "flag")
        for name, scroll_filter, expected_ids, note in [
            (
                "after_flag_rebuild_flag_true",
                match_filter("flag", True),
                [1, 2, 6],
                "Rebuilding the bool payload index after merge+overwrite keeps the same post-mutation truth set.",
            ),
        ]:
            all_passed &= check_case(client, collection_name, name, scroll_filter, expected_ids, note)

        client.delete_payload(
            collection_name=collection_name,
            keys=["status"],
            points=match_filter("city", "rome"),
            wait=True,
        )
        for name, scroll_filter, expected_ids, note in [
            (
                "after_delete_status_published",
                match_filter("status", "published"),
                [],
                "delete_payload selected by filter removes the exact-match key from all selected points.",
            ),
            (
                "after_delete_status_is_empty",
                is_empty_filter("status"),
                [1, 2, 5, 6],
                "Deleted keys become missing and therefore satisfy the validated `is_empty` semantics.",
            ),
        ]:
            all_passed &= check_case(client, collection_name, name, scroll_filter, expected_ids, note)

        rebuild_index(client, collection_name, "status")
        for name, scroll_filter, expected_ids, note in [
            (
                "after_status_rebuild_status_review",
                match_filter("status", "review"),
                [3],
                "Rebuilding the keyword payload index after delete/overwrite preserves remaining exact matches.",
            ),
            (
                "after_status_rebuild_status_is_empty",
                is_empty_filter("status"),
                [1, 2, 5, 6],
                "Rebuilding the keyword payload index after delete_payload still treats deleted keys as empty in this validated subset.",
            ),
        ]:
            all_passed &= check_case(client, collection_name, name, scroll_filter, expected_ids, note)

        client.clear_payload(
            collection_name=collection_name,
            points_selector=[6],
            wait=True,
        )
        for name, scroll_filter, expected_ids, note in [
            (
                "after_clear_status_is_null",
                is_null_filter("status"),
                [],
                "clear_payload removes the explicit null field entirely, so `is_null` no longer matches.",
            ),
            (
                "after_clear_status_is_empty",
                is_empty_filter("status"),
                [1, 2, 5, 6],
                "A fully cleared payload behaves like a missing field under the validated `is_empty` subset.",
            ),
            (
                "after_clear_score_ge_30",
                score_range_filter(30),
                [2, 4],
                "clear_payload also removes numeric fields, so later range queries see only the remaining indexed scores.",
            ),
        ]:
            all_passed &= check_case(client, collection_name, name, scroll_filter, expected_ids, note)
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")

    return all_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qdrant payload mutation semantics")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--grpc-port", dest="grpc_port", type=int, default=GRPC_PORT)
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args()


def main() -> int:
    global HOST, PORT, GRPC_PORT, RUN_ID
    args = parse_args()
    HOST = args.host
    PORT = int(args.port)
    GRPC_PORT = int(args.grpc_port)
    RUN_ID = args.run_id

    try:
        server_info = fetch_server_info()
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant payload mutation operator validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all payload mutation operator checks passed on REST and gRPC.")
        return 0

    print("\nSummary: at least one payload mutation operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
