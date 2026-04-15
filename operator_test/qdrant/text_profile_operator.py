"""
Validate conservative Qdrant full-text index profile semantics that are useful
for scalar/filter coverage experiments.

This file deliberately stays separate from the main random fuzzer until each
profile has a deterministic, explainable oracle:
1. WORD tokenizer with ascii_folding on/off.
2. PREFIX tokenizer query behavior with max-ngram truncation.
3. PREFIX tokenizer querying of stopword-like prefixes (`the` -> `theory`).
4. WORD tokenizer with language/custom stopwords plus Snowball English stemming.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    Language,
    MatchText,
    PointStruct,
    Snowball,
    SnowballLanguage,
    SnowballParams,
    StopwordsSet,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)


RUN_ID = "text-profile-operator"


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def unique_collection_name(profile: str, prefer_grpc: bool) -> str:
    transport = "grpc" if prefer_grpc else "rest"
    return f"text_profile_{slugify(RUN_ID, max_len=24)}_{slugify(profile, max_len=24)}_{transport}"


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


def scroll_ids(client: QdrantClient, collection_name: str, scroll_filter: Filter) -> list[int]:
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=64,
        with_payload=False,
        with_vectors=False,
    )
    return sorted(int(point.id) for point in points)


def profile_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "word_ascii_disabled",
            "index": TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                lowercase=True,
                ascii_folding=False,
            ),
            "points": [
                PointStruct(id=1, vector=[1.0, 0.0], payload={"title": "ação no coração"}),
                PointStruct(id=2, vector=[0.0, 1.0], payload={"title": "café com leite"}),
                PointStruct(id=3, vector=[0.5, 0.5], payload={"title": "plain ascii"}),
            ],
            "tests": [
                (
                    "accented_query_matches_without_folding",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="ação"))]),
                    [1],
                    "Non-folded queries must still match the original accented token.",
                ),
                (
                    "ascii_query_does_not_match_without_folding",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="acao"))]),
                    [],
                    "ASCII-only query should not match accented tokens when ascii_folding=false.",
                ),
            ],
        },
        {
            "name": "word_ascii_enabled",
            "index": TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                lowercase=True,
                ascii_folding=True,
            ),
            "points": [
                PointStruct(id=1, vector=[1.0, 0.0], payload={"title": "ação no coração"}),
                PointStruct(id=2, vector=[0.0, 1.0], payload={"title": "café com leite"}),
                PointStruct(id=3, vector=[0.5, 0.5], payload={"title": "jalapeño über"}),
            ],
            "tests": [
                (
                    "ascii_query_matches_with_folding",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="acao"))]),
                    [1],
                    "ASCII folding should normalize ação -> acao at query/index time.",
                ),
                (
                    "non_ascii_query_still_matches_with_folding",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="ação"))]),
                    [1],
                    "The original accented query should still match when folding is enabled.",
                ),
                (
                    "mixed_ascii_folding_examples",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="uber"))]),
                    [3],
                    "ASCII folding should also normalize Über -> uber.",
                ),
            ],
        },
        {
            "name": "prefix_query_and_stopword_semantics",
            "index": TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.PREFIX,
                lowercase=True,
                min_token_len=1,
                max_token_len=4,
                stopwords=StopwordsSet(custom=["the"]),
            ),
            "points": [
                PointStruct(id=1, vector=[1.0, 0.0], payload={"title": "robotics future"}),
                PointStruct(id=2, vector=[0.0, 1.0], payload={"title": "robot arms"}),
                PointStruct(id=3, vector=[1.0, 1.0], payload={"title": "robust shield"}),
                PointStruct(id=4, vector=[0.5, 0.5], payload={"title": "robo helper"}),
                PointStruct(id=5, vector=[0.2, 0.8], payload={"title": "theory lesson"}),
            ],
            "tests": [
                (
                    "prefix_uppercase_query_truncates_to_max_ngram",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="ROBO"))]),
                    [1, 2, 4],
                    "Prefix query tokens use the maximal ngram and lowercase normalization.",
                ),
                (
                    "prefix_distinguishes_other_prefixes",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="robu"))]),
                    [3],
                    "A different four-character prefix should only match robust* rows.",
                ),
                (
                    "prefix_query_keeps_stopword_like_prefixes",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="the"))]),
                    [5],
                    "Query-side prefix tokenization should not discard stopword-like partial words.",
                ),
            ],
        },
        {
            "name": "word_language_stopwords_and_stemming",
            "index": TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                lowercase=True,
                stopwords=StopwordsSet(languages=[Language.ENGLISH], custom=["I'd"]),
                stemmer=SnowballParams(type=Snowball.SNOWBALL, language=SnowballLanguage.ENGLISH),
            ),
            "points": [
                PointStruct(id=1, vector=[1.0, 0.0], payload={"title": "The interestingly proceeding living fox"}),
                PointStruct(id=2, vector=[0.0, 1.0], payload={"title": "I'd like the walker"}),
                PointStruct(id=3, vector=[0.5, 0.5], payload={"title": "brown dogs only"}),
            ],
            "tests": [
                (
                    "stemming_matches_interestingly_via_interest",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="interest"))]),
                    [1],
                    "English Snowball stemming should normalize interestingly -> interest.",
                ),
                (
                    "stemming_matches_proceeding_via_proceed",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="proceed"))]),
                    [1],
                    "English Snowball stemming should normalize proceeding -> proceed.",
                ),
                (
                    "stemming_matches_living_via_live",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="live"))]),
                    [1],
                    "English Snowball stemming should normalize living -> live.",
                ),
                (
                    "language_stopwords_removed_from_query",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="the"))]),
                    [],
                    "English stopwords should be removed from indexed/query token streams.",
                ),
                (
                    "custom_stopword_removed_from_query",
                    Filter(must=[FieldCondition(key="title", match=MatchText(text="I'd"))]),
                    [],
                    "Custom stopwords merged with language stopwords should also be removed.",
                ),
            ],
        },
    ]


def run_profile(
    client: QdrantClient,
    prefer_grpc: bool,
    profile_name: str,
    index_params: TextIndexParams,
    points: list[PointStruct],
    tests: list[tuple[str, Filter, list[int], str]],
) -> bool:
    collection_name = unique_collection_name(profile_name, prefer_grpc)
    all_passed = True

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=points, wait=True)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
            field_schema=index_params,
            wait=True,
        )

        print(f"\nprofile={profile_name} collection={collection_name}")
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
            print(f"cleanup_warning ({profile_name}): {exc}")

    return all_passed


def run_transport(args: argparse.Namespace, prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    client = build_client(args, prefer_grpc)
    all_passed = True

    print(f"\n--- text profile operator validation ({transport}) ---")
    for spec in profile_specs():
        ok = run_profile(
            client=client,
            prefer_grpc=prefer_grpc,
            profile_name=str(spec["name"]),
            index_params=spec["index"],
            points=list(spec["points"]),
            tests=list(spec["tests"]),
        )
        if not ok:
            all_passed = False
    return all_passed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qdrant text profile operator semantics")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", type=int, default=6334)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--rest-only", action="store_true")
    parser.add_argument("--grpc-only", action="store_true")
    parser.add_argument("--run-id", default="text-profile-operator")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global RUN_ID
    args = parse_args(argv)
    RUN_ID = args.run_id
    try:
        server_info = fetch_server_info(args.host, args.port)
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant text profile operator validation")
    print(f"target={args.host}:{args.port} grpc:{args.grpc_port}")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = True if args.grpc_only else run_transport(args, prefer_grpc=False)
    grpc_ok = True if args.rest_only else run_transport(args, prefer_grpc=True)

    if rest_ok and grpc_ok:
        print("\nSummary: all text profile operator checks passed.")
        return 0

    print("\nSummary: at least one text profile operator check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
