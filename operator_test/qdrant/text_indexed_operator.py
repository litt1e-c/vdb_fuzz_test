"""
Validation for Qdrant text/phrase matching with an explicit full-text index.

This is intentionally separate from text_operator.py and phrase_operator.py:
those files document the current fuzzer's no-full-text-index substring oracle,
while this file validates a conservative indexed-token subset for coverage
boosting and future oracle expansion.
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
    MatchPhrase,
    MatchText,
    PointStruct,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

RUN_ID = "text-indexed-operator"


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def unique_collection_name(prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"text_indexed_operator_{slugify(RUN_ID, max_len=40)}_{transport}"


def fetch_server_info(host: str, port: int) -> dict[str, object]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{host}:{port}/", timeout=5) as response:
        import json

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
        PointStruct(id=1, vector=[1.0, 0.0], payload={"title": "quick brown fox"}),
        PointStruct(id=2, vector=[0.0, 1.0], payload={"title": "quick blue hare"}),
        PointStruct(id=3, vector=[1.0, 1.0], payload={"title": "slow brown fox"}),
        PointStruct(id=4, vector=[0.5, 0.5], payload={"title": "hardware cheap good"}),
        PointStruct(id=5, vector=[0.2, 0.8], payload={"title": "QUICK BROWN uppercase"}),
        PointStruct(id=6, vector=[0.8, 0.2], payload={"title": "prefixbrown suffix"}),
        PointStruct(id=7, vector=[0.3, 0.7], payload={}),
        PointStruct(id=8, vector=[0.7, 0.3], payload={"title": None}),
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


def create_text_index(client: QdrantClient, collection_name: str) -> None:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="title",
        field_schema=TextIndexParams(
            type=TextIndexType.TEXT,
            tokenizer=TokenizerType.WORD,
            lowercase=True,
            phrase_matching=True,
        ),
        wait=True,
    )


def run_transport(args: argparse.Namespace, prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(prefer_grpc)
    client = build_client(args, prefer_grpc)
    all_passed = True

    tests = [
        (
            "indexed_text_single_token",
            Filter(must=[FieldCondition(key="title", match=MatchText(text="brown"))]),
            [1, 3, 5],
            "WORD tokenizer should match the full token 'brown' and lowercase indexed text.",
        ),
        (
            "indexed_text_no_partial_token",
            Filter(must=[FieldCondition(key="title", match=MatchText(text="prefix"))]),
            [],
            "WORD tokenizer should not treat 'prefixbrown' as the token 'prefix'.",
        ),
        (
            "indexed_text_exact_custom_token",
            Filter(must=[FieldCondition(key="title", match=MatchText(text="prefixbrown"))]),
            [6],
            "A full WORD token should match even when it contains the substring 'brown'.",
        ),
        (
            "indexed_phrase_contiguous",
            Filter(must=[FieldCondition(key="title", match=MatchPhrase(phrase="quick brown"))]),
            [1, 5],
            "Phrase matching is enabled, so contiguous indexed token phrases should match.",
        ),
        (
            "indexed_phrase_reversed_negative",
            Filter(must=[FieldCondition(key="title", match=MatchPhrase(phrase="brown quick"))]),
            [],
            "The reversed phrase should not match when that token order is absent.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)
        create_text_index(client, collection_name)

        print(f"\n--- indexed text operator validation ({transport}) ---")
        print(f"collection={collection_name}")
        for name, scroll_filter, expected_ids, note in tests:
            actual_ids = scroll_ids(client, collection_name, scroll_filter)
            passed = actual_ids == expected_ids
            if not passed:
                all_passed = False
            status = "PASS" if passed else "FAIL"
            print(
                f"{name}: {status} | expected={expected_ids} | actual={actual_ids} | note={note}"
            )
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")

    return all_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qdrant indexed text/phrase operator semantics")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", type=int, default=6334)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--rest-only", action="store_true")
    parser.add_argument("--grpc-only", action="store_true")
    parser.add_argument("--run-id", default="text-indexed-operator")
    return parser.parse_args()


def main() -> int:
    global RUN_ID
    args = parse_args()
    RUN_ID = args.run_id
    try:
        server_info = fetch_server_info(args.host, args.port)
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant indexed text operator validation")
    print(f"target={args.host}:{args.port} grpc:{args.grpc_port}")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = True if args.grpc_only else run_transport(args, prefer_grpc=False)
    grpc_ok = True if args.rest_only else run_transport(args, prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all indexed text operator checks passed.")
        return 0

    print("\nSummary: at least one indexed text operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
