"""
Reproduce a REST/gRPC transport mismatch for Qdrant text indexing when
`lowercase=false` is combined with custom case-sensitive stopwords.

Observed on local Qdrant v1.17.0:
- REST behaves like the Rust tokenizer tests suggest:
  - lowercase `lazy` survives when stopwords contain only `LAZY`
  - exact stopwords `The` and `LAZY` do not match
- gRPC returns the opposite answers for this fixture.

This script is intentionally kept outside the main operator/oracle path until
the behavior is understood and stabilized.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    PointStruct,
    StopwordsSet,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)


COLLECTION = "bug_text_case_sensitive_stopwords_transport"


def build_client(prefer_grpc: bool) -> QdrantClient:
    return QdrantClient(
        host="127.0.0.1",
        port=6333,
        grpc_port=6334,
        prefer_grpc=prefer_grpc,
        timeout=30,
    )


def build_points() -> list[PointStruct]:
    return [
        PointStruct(id=1, vector=[1.0, 0.0], payload={"title": "The quick brown fox jumps over the lazy dog"}),
        PointStruct(id=2, vector=[0.0, 1.0], payload={"title": "LAZY sentinel"}),
    ]


def scroll_ids(client: QdrantClient, scroll_filter: Filter) -> list[int]:
    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=scroll_filter,
        limit=32,
        with_payload=False,
        with_vectors=False,
    )
    return sorted(int(point.id) for point in points)


def run_transport(prefer_grpc: bool) -> dict[str, list[int]]:
    transport = "gRPC" if prefer_grpc else "REST"
    client = build_client(prefer_grpc)
    try:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=COLLECTION, points=build_points(), wait=True)
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="title",
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                lowercase=False,
                stopwords=StopwordsSet(custom=["the", "The", "LAZY"]),
            ),
            wait=True,
        )

        queries = {
            "lazy": Filter(must=[FieldCondition(key="title", match=MatchText(text="lazy"))]),
            "The": Filter(must=[FieldCondition(key="title", match=MatchText(text="The"))]),
            "LAZY": Filter(must=[FieldCondition(key="title", match=MatchText(text="LAZY"))]),
        }
        results = {name: scroll_ids(client, flt) for name, flt in queries.items()}

        print(f"\n--- {transport} ---")
        for name, ids in results.items():
            print(f"{name}: {ids}")
        return results
    finally:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass


def main() -> int:
    rest = run_transport(prefer_grpc=False)
    grpc = run_transport(prefer_grpc=True)

    print("\nExpected from tokenizer tests / REST on this setup:")
    print("lazy  -> [1]")
    print("The   -> []")
    print("LAZY  -> []")

    print("\nComparison:")
    for key in ("lazy", "The", "LAZY"):
        print(f"{key}: REST={rest[key]} | gRPC={grpc[key]}")

    if rest != grpc:
        print("\n[MISMATCH] REST and gRPC disagree for case-sensitive custom stopwords.")
        return 1

    print("\nNo transport mismatch observed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
