"""
Validate payload-index profile persistence across a Qdrant restart.

This validator is intentionally split into two externally orchestrated phases:
1. `prepare`: create a deterministic collection, insert points, create payload
   indexes with explicit profile params, and run a pre-restart smoke check.
2. `verify`: after the server restarts, rerun the same scalar queries/facets to
   confirm that payload index state was recovered correctly.

The goal is not to fuzz undocumented behavior. It validates a conservative,
paper-friendly subset of documented payload index configurations that are worth
carrying into the main scalar fuzzer later.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    BoolIndexParams,
    BoolIndexType,
    DatetimeIndexParams,
    DatetimeIndexType,
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    FloatIndexParams,
    FloatIndexType,
    IntegerIndexParams,
    IntegerIndexType,
    KeywordIndexParams,
    KeywordIndexType,
    MatchPhrase,
    MatchText,
    MatchValue,
    PointStruct,
    Range,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    UuidIndexParams,
    UuidIndexType,
    VectorParams,
)


RUN_ID = "persistent-index-operator"
STATE_VERSION = 1


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def collection_name(run_id: str) -> str:
    return f"persistent_scalar_index_{slugify(run_id, max_len=40)}"


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
    return [
        PointStruct(
            id=1,
            vector=[1.0, 0.0],
            payload={
                "tenant_kw": "alpha",
                "num_lookup": 10,
                "num_range": 5,
                "score_float": 1.0,
                "flag": True,
                "event_time": "2024-01-01T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000001",
                "title_text": "quick brown fox",
                "arr_num": [1, 2],
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0],
            payload={
                "tenant_kw": "beta",
                "num_lookup": 20,
                "num_range": 25,
                "score_float": 1.75,
                "flag": False,
                "event_time": "2024-01-02T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000002",
                "title_text": "quick blue hare",
                "arr_num": [7, 9],
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0, 1.0],
            payload={
                "tenant_kw": "alpha",
                "num_lookup": 30,
                "num_range": 15,
                "score_float": 0.25,
                "flag": True,
                "event_time": "2024-01-03T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000003",
                "title_text": "slow brown fox",
                "arr_num": [],
            },
        ),
        PointStruct(
            id=4,
            vector=[0.5, 0.5],
            payload={
                "tenant_kw": "gamma",
                "num_lookup": 10,
                "num_range": 30,
                "score_float": 2.5,
                "flag": True,
                "event_time": "2024-01-04T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000004",
                "title_text": "tenant gamma item",
                "arr_num": [5, 8],
            },
        ),
        PointStruct(
            id=5,
            vector=[0.2, 0.8],
            payload={
                "tenant_kw": "beta",
                "num_lookup": 40,
                "num_range": 12,
                "score_float": 1.2,
                "flag": False,
                "event_time": "2024-01-05T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000005",
                "title_text": None,
                "arr_num": None,
            },
        ),
        PointStruct(
            id=6,
            vector=[0.8, 0.2],
            payload={
                "num_lookup": 50,
                "num_range": 18,
                "score_float": 0.9,
                "flag": False,
                "event_time": "2024-01-06T00:00:00Z",
                "uuid_tag": "00000000-0000-0000-0000-000000000006",
            },
        ),
    ]


def index_specs() -> list[tuple[str, object, str]]:
    return [
        (
            "tenant_kw",
            KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True, on_disk=True),
            "keyword_tenant_on_disk",
        ),
        (
            "num_lookup",
            IntegerIndexParams(type=IntegerIndexType.INTEGER, lookup=True, range=False, on_disk=True),
            "integer_lookup_only_on_disk",
        ),
        (
            "num_range",
            IntegerIndexParams(
                type=IntegerIndexType.INTEGER,
                lookup=False,
                range=True,
                is_principal=True,
                on_disk=True,
            ),
            "integer_range_only_principal_on_disk",
        ),
        (
            "score_float",
            FloatIndexParams(type=FloatIndexType.FLOAT, is_principal=True, on_disk=True),
            "float_principal_on_disk",
        ),
        (
            "flag",
            BoolIndexParams(type=BoolIndexType.BOOL, on_disk=True),
            "bool_on_disk",
        ),
        (
            "event_time",
            DatetimeIndexParams(type=DatetimeIndexType.DATETIME, is_principal=True, on_disk=True),
            "datetime_principal_on_disk",
        ),
        (
            "uuid_tag",
            UuidIndexParams(type=UuidIndexType.UUID, is_tenant=True, on_disk=True),
            "uuid_tenant_on_disk",
        ),
        (
            "title_text",
            TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                lowercase=True,
                phrase_matching=True,
                on_disk=True,
            ),
            "text_word_phrase_on_disk",
        ),
        (
            "arr_num",
            IntegerIndexParams(type=IntegerIndexType.INTEGER, lookup=True, range=True, on_disk=True),
            "integer_array_on_disk",
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


def normalize_facet_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def facet_counts(response: Any) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for hit in response.hits:
        counts[normalize_facet_value(hit.value)] = int(hit.count)
    return counts


def query_cases() -> list[tuple[str, Filter, list[int], str]]:
    return [
        (
            "keyword_tenant_match",
            Filter(must=[FieldCondition(key="tenant_kw", match=MatchValue(value="alpha"))]),
            [1, 3],
            "Keyword tenant index should keep exact-match behavior after restart.",
        ),
        (
            "integer_lookup_eq",
            Filter(must=[FieldCondition(key="num_lookup", match=MatchValue(value=10))]),
            [1, 4],
            "Lookup-only integer index should preserve equality filtering.",
        ),
        (
            "integer_range_gt",
            Filter(must=[FieldCondition(key="num_range", range=Range(gt=20))]),
            [2, 4],
            "Range-only integer index should preserve range filtering.",
        ),
        (
            "float_range_gte",
            Filter(must=[FieldCondition(key="score_float", range=Range(gte=1.5))]),
            [2, 4],
            "Float payload index should preserve numeric range filtering.",
        ),
        (
            "bool_match_true",
            Filter(must=[FieldCondition(key="flag", match=MatchValue(value=True))]),
            [1, 3, 4],
            "Bool payload index should preserve exact true/false matching.",
        ),
        (
            "datetime_range_gte",
            Filter(must=[FieldCondition(key="event_time", range=DatetimeRange(gte="2024-01-03T00:00:00Z"))]),
            [3, 4, 5, 6],
            "Datetime payload index should preserve ordered timestamp filtering.",
        ),
        (
            "uuid_exact_match",
            Filter(
                must=[
                    FieldCondition(
                        key="uuid_tag",
                        match=MatchValue(value="00000000-0000-0000-0000-000000000002"),
                    )
                ]
            ),
            [2],
            "UUID tenant index should preserve exact UUID filtering.",
        ),
        (
            "text_index_single_token",
            Filter(must=[FieldCondition(key="title_text", match=MatchText(text="brown"))]),
            [1, 3],
            "Indexed WORD-token text matching should survive restart.",
        ),
        (
            "text_index_phrase",
            Filter(must=[FieldCondition(key="title_text", match=MatchPhrase(phrase="quick brown"))]),
            [1],
            "Indexed phrase matching should survive restart.",
        ),
        (
            "integer_array_range",
            Filter(must=[FieldCondition(key="arr_num", range=Range(gte=8))]),
            [2, 4],
            "Integer array index should preserve any-element range filtering.",
        ),
    ]


def facet_cases() -> list[tuple[str, dict[str, object], dict[Any, int], str]]:
    return [
        (
            "facet_tenant_keyword_all",
            {"key": "tenant_kw", "limit": 10, "exact": True},
            {"alpha": 2, "beta": 2, "gamma": 1},
            "Facet should still count persisted keyword payload values after restart.",
        ),
        (
            "facet_tenant_keyword_filtered",
            {
                "key": "tenant_kw",
                "limit": 10,
                "exact": True,
                "facet_filter": Filter(must=[FieldCondition(key="flag", match=MatchValue(value=True))]),
            },
            {"alpha": 2, "gamma": 1},
            "Facet filtering should still work on recovered payload indexes.",
        ),
        (
            "facet_bool_all",
            {"key": "flag", "limit": 10, "exact": True},
            {True: 3, False: 3},
            "Bool facet counts should remain stable after restart.",
        ),
    ]


def run_validation_round(client: QdrantClient, collection_name_value: str, transport: str, phase: str) -> bool:
    all_passed = True
    print(f"\n--- persistent index validation ({phase}, {transport}) ---")
    print(f"collection={collection_name_value}")

    for name, scroll_filter, expected_ids, note in query_cases():
        actual_ids = scroll_ids(client, collection_name_value, scroll_filter)
        passed = actual_ids == expected_ids
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status} | expected={expected_ids} | actual={actual_ids} | note={note}")

    for name, kwargs, expected_counts, note in facet_cases():
        response = client.facet(collection_name=collection_name_value, **kwargs)
        actual_counts = facet_counts(response)
        passed = actual_counts == expected_counts
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status} | expected={expected_counts} | actual={actual_counts} | note={note}")

    return all_passed


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_state(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qdrant payload-index persistence across restart")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", type=int, default=6334)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--run-id", default="persistent-index-operator")
    parser.add_argument("--state-path", default="persistent_index_state.json")
    parser.add_argument("--phase", choices=["prepare", "verify"], required=True)
    parser.add_argument("--rest-only", action="store_true")
    parser.add_argument("--grpc-only", action="store_true")
    return parser.parse_args()


def run_prepare(args: argparse.Namespace) -> bool:
    client = build_client(args, prefer_grpc=bool(args.grpc_only and not args.rest_only))
    name = collection_name(args.run_id)
    state_path = Path(args.state_path).expanduser().resolve()

    try:
        client.delete_collection(name)
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=2, distance=Distance.DOT),
    )
    client.upsert(collection_name=name, points=build_points(), wait=True)
    for field_name, field_schema, profile_name in index_specs():
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )
        print(f"prepared_index: field={field_name} profile={profile_name}")

    prepare_ok = run_validation_round(client, name, "prepare-client", "prepare")
    state_payload = {
        "state_version": STATE_VERSION,
        "run_id": args.run_id,
        "collection_name": name,
        "point_count": len(build_points()),
        "profiles": [{"field_name": field_name, "profile": profile_name} for field_name, _, profile_name in index_specs()],
    }
    write_state(state_path, state_payload)
    print(f"state_path={state_path}")

    if prepare_ok:
        print("\nSummary: all persistent index prepare checks passed.")
    else:
        print("\nSummary: at least one persistent index prepare check failed.")
    return prepare_ok


def run_verify(args: argparse.Namespace) -> bool:
    state_path = Path(args.state_path).expanduser().resolve()
    if state_path.exists():
        state = load_state(state_path)
        name = str(state.get("collection_name") or collection_name(args.run_id))
    else:
        name = collection_name(args.run_id)
        print(f"state_warning: missing state file, falling back to collection={name}")

    all_passed = True
    if not args.grpc_only:
        rest_client = build_client(args, prefer_grpc=False)
        all_passed = run_validation_round(rest_client, name, "REST", "verify") and all_passed
    if not args.rest_only:
        grpc_client = build_client(args, prefer_grpc=True)
        all_passed = run_validation_round(grpc_client, name, "gRPC", "verify") and all_passed

    cleanup_client = build_client(args, prefer_grpc=False)
    try:
        cleanup_client.delete_collection(name)
    except Exception as exc:
        print(f"cleanup_warning: {exc}")

    if all_passed:
        print("\nSummary: all persistent index verify checks passed.")
    else:
        print("\nSummary: at least one persistent index verify check failed.")
    return all_passed


def main() -> int:
    global RUN_ID
    args = parse_args()
    RUN_ID = args.run_id

    try:
        server_info = fetch_server_info(args.host, args.port)
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant persistent payload-index validation")
    print(f"phase={args.phase}")
    print(f"target={args.host}:{args.port} grpc:{args.grpc_port}")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")
    print(f"run_id={args.run_id}")
    print(f"state_path={Path(args.state_path).expanduser().resolve()}")

    ok = run_prepare(args) if args.phase == "prepare" else run_verify(args)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
