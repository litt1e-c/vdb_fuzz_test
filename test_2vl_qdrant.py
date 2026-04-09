"""
Qdrant logic semantics probe.

This script is intentionally opinionated:
- it distinguishes missing / explicit null / empty / scalar / nested cases
- it includes null-aware predicates such as is_null / is_empty / values_count
- it includes logical forms such as A OR NOT A, which distinguish 2VL from SQL-like 3VL

Run example:
  python test_2vl_qdrant.py --prefer-grpc
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass

from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    IsNullCondition,
    MatchAny,
    MatchExcept,
    MatchValue,
    Nested,
    NestedCondition,
    PayloadField,
    PointStruct,
    Range,
    VectorParams,
)


COLLECTION = "qdrant_logic_probe"


@dataclass(frozen=True)
class Case:
    name: str
    scroll_filter: Filter
    expected_ids: list[int]
    note: str


def payload_field(key: str) -> PayloadField:
    return PayloadField(key=key)


def is_null(key: str) -> IsNullCondition:
    return IsNullCondition(is_null=payload_field(key))


def is_empty(key: str) -> IsEmptyCondition:
    return IsEmptyCondition(is_empty=payload_field(key))


def fetch_server_info(host: str, port: int) -> dict[str, object]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{host}:{port}/", timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_client(args: argparse.Namespace) -> QdrantClient:
    return QdrantClient(
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port,
        prefer_grpc=args.prefer_grpc,
        timeout=30,
    )


def build_dataset() -> list[PointStruct]:
    return [
        PointStruct(
            id=1,
            vector=[1.0, 0.0, 0.0, 0.0],
            payload={
                "score": 10,
                "tags": [1, 2],
                "misc_values": [1, 2],
                "items": [{"a": 1, "active": True}, {"a": 2, "active": False}],
            },
        ),
        PointStruct(
            id=2,
            vector=[0.0, 1.0, 0.0, 0.0],
            payload={},
        ),
        PointStruct(
            id=3,
            vector=[0.0, 0.0, 1.0, 0.0],
            payload={
                "score": None,
                "tags": None,
                "misc_values": None,
                "items": None,
            },
        ),
        PointStruct(
            id=4,
            vector=[0.0, 0.0, 0.0, 1.0],
            payload={
                "score": 5,
                "tags": [],
                "misc_values": [],
                "items": [],
            },
        ),
        PointStruct(
            id=5,
            vector=[1.0, 1.0, 0.0, 0.0],
            payload={
                "score": 20,
                "tags": [4, 5],
                "misc_values": 99,
                "items": [{"a": 3, "active": True}, {"a": 1, "active": False}],
            },
        ),
        PointStruct(
            id=6,
            vector=[1.0, 0.0, 1.0, 0.0],
            payload={
                "score": -1,
                "tags": [1],
                "misc_values": [7],
                "items": [{"a": 0, "active": False}],
            },
        ),
    ]


def build_cases() -> list[Case]:
    atom_a = FieldCondition(key="score", range=Range(gt=0))
    score_eq_10 = FieldCondition(key="score", match=MatchValue(value=10))
    nested_a_eq_1 = NestedCondition(
        nested=Nested(
            key="items",
            filter=Filter(must=[FieldCondition(key="a", match=MatchValue(value=1))]),
        )
    )

    return [
        Case(
            name="ATOM__must__score_eq_10",
            scroll_filter=Filter(must=[score_eq_10]),
            expected_ids=[1],
            note="Ordinary comparisons do not match missing/null rows.",
        ),
        Case(
            name="ATOM__must_not__score_eq_10",
            scroll_filter=Filter(must_not=[score_eq_10]),
            expected_ids=[2, 3, 4, 5, 6],
            note="must_not keeps rows where the inner predicate does not match, including missing/null.",
        ),
        Case(
            name="ATOM__must__score_gt_0",
            scroll_filter=Filter(must=[atom_a]),
            expected_ids=[1, 4, 5],
            note="Range predicates also ignore missing/null.",
        ),
        Case(
            name="ATOM__must_not__score_gt_0",
            scroll_filter=Filter(must_not=[atom_a]),
            expected_ids=[2, 3, 6],
            note="NOT(score > 0) keeps missing/null rows, which is the key 2VL discriminator.",
        ),
        Case(
            name="ATOM__must__score_not_eq_10",
            scroll_filter=Filter(must=[FieldCondition(key="score", match=MatchExcept(**{"except": [10]}))]),
            expected_ids=[4, 5, 6],
            note="MatchExcept does not match missing/null rows.",
        ),
        Case(
            name="ARRAY__must__tags_contains_1",
            scroll_filter=Filter(must=[FieldCondition(key="tags", match=MatchAny(any=[1]))]),
            expected_ids=[1, 6],
            note="Array membership should only match rows that actually contain 1.",
        ),
        Case(
            name="NULL__must__score_is_null",
            scroll_filter=Filter(must=[is_null("score")]),
            expected_ids=[3],
            note="is_null matches explicit null only, not missing.",
        ),
        Case(
            name="NULL__must__score_is_empty",
            scroll_filter=Filter(must=[is_empty("score")]),
            expected_ids=[2, 3],
            note="is_empty matches missing and explicit null on scalar fields.",
        ),
        Case(
            name="NULL__must_not__score_is_empty",
            scroll_filter=Filter(must_not=[is_empty("score")]),
            expected_ids=[1, 4, 5, 6],
            note="must_not is_empty keeps all non-empty rows.",
        ),
        Case(
            name="NULL__must__tags_is_null",
            scroll_filter=Filter(must=[is_null("tags")]),
            expected_ids=[3],
            note="Explicit null remains distinguishable from missing and [].",
        ),
        Case(
            name="NULL__must__tags_is_empty",
            scroll_filter=Filter(must=[is_empty("tags")]),
            expected_ids=[2, 3, 4],
            note="For arrays, is_empty matches missing, null, and [].",
        ),
        Case(
            name="COUNT__must__misc_values_lt_1",
            scroll_filter=Filter(must=[FieldCondition(key="misc_values", values_count=models.ValuesCount(lt=1))]),
            expected_ids=[3, 4],
            note="Observed behavior on v1.17.0: explicit null and [] count as 0, but missing does not participate.",
        ),
        Case(
            name="COUNT__must__misc_values_gte_0",
            scroll_filter=Filter(must=[FieldCondition(key="misc_values", values_count=models.ValuesCount(gte=0))]),
            expected_ids=[1, 3, 4, 5, 6],
            note="All present values participate; missing stays excluded.",
        ),
        Case(
            name="NESTED__must__items_a_eq_1",
            scroll_filter=Filter(must=[nested_a_eq_1]),
            expected_ids=[1, 5],
            note="Nested requires at least one array element to satisfy the inner filter.",
        ),
        Case(
            name="NESTED__must_not__items_a_eq_1",
            scroll_filter=Filter(must_not=[nested_a_eq_1]),
            expected_ids=[2, 3, 4, 6],
            note="Outer must_not keeps missing/null/[] nested fields.",
        ),
        Case(
            name="NESTED__must__items_not_a_eq_1",
            scroll_filter=Filter(
                must=[
                    NestedCondition(
                        nested=Nested(
                            key="items",
                            filter=Filter(must_not=[FieldCondition(key="a", match=MatchValue(value=1))]),
                        )
                    )
                ]
            ),
            expected_ids=[1, 5, 6],
            note="Inner must_not works per nested element; rows need at least one surviving element.",
        ),
        Case(
            name="LOGIC__must__score_gt_0_or_not_score_gt_0",
            scroll_filter=Filter(
                should=[
                    Filter(must=[atom_a]),
                    Filter(must_not=[atom_a]),
                ]
            ),
            expected_ids=[1, 2, 3, 4, 5, 6],
            note="A OR NOT A returns every row, including missing/null: strong evidence for 2VL set logic rather than SQL 3VL.",
        ),
        Case(
            name="LOGIC__must__score_gt_0_and_not_score_gt_0",
            scroll_filter=Filter(
                must=[
                    Filter(must=[atom_a]),
                    Filter(must_not=[atom_a]),
                ]
            ),
            expected_ids=[],
            note="A AND NOT A is unsatisfiable under the observed semantics.",
        ),
        Case(
            name="LOGIC__must__not_not_score_gt_0",
            scroll_filter=Filter(must_not=[Filter(must_not=[atom_a])]),
            expected_ids=[1, 4, 5],
            note="Double negation collapses back to the original match set.",
        ),
    ]


def scroll_ids(client: QdrantClient, scroll_filter: Filter) -> list[int]:
    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=scroll_filter,
        limit=128,
        with_payload=False,
        with_vectors=False,
    )
    return sorted(int(point.id) for point in points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Qdrant logic semantics.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--grpc-port", dest="grpc_port", type=int, default=6334)
    parser.add_argument("--prefer-grpc", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = build_client(args)

    try:
        server_info = fetch_server_info(args.host, args.port)
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    client.upsert(collection_name=COLLECTION, points=build_dataset(), wait=True)

    print("=" * 88)
    print("Qdrant Logic Semantics Probe")
    print("=" * 88)
    print(f"Server version:  {server_info.get('version')}")
    print(f"Server commit:   {server_info.get('commit')}")
    print(f"Client version:  {pkg_version('qdrant-client')}")
    print(f"Transport:       {'gRPC' if args.prefer_grpc else 'REST'}")
    print("Collection:      qdrant_logic_probe")
    print("Row roles:")
    print("  1 -> regular scalar/array/nested values")
    print("  2 -> missing fields")
    print("  3 -> explicit null fields")
    print("  4 -> empty array fields")
    print("  5 -> alternate regular values + scalar misc_values")
    print("  6 -> negative scalar + array contains 1 + nested non-matching element")
    print("-" * 88)

    failures = 0
    for case in build_cases():
        actual_ids = scroll_ids(client, case.scroll_filter)
        ok = actual_ids == case.expected_ids
        marker = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"{marker:<4} {case.name}")
        print(f"  actual:   {actual_ids}")
        print(f"  expected: {case.expected_ids}")
        print(f"  note:     {case.note}")
        print()

    client.delete_collection(COLLECTION)

    print("-" * 88)
    if failures:
        print(f"Summary: {failures} failing case(s).")
        return 1

    print("Summary: all logic cases matched the expected semantics.")
    print("Classification: Qdrant core filter composition behaves as 2VL set logic,")
    print("while is_null/is_empty/values_count add dedicated null-aware predicates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
