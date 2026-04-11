"""
Minimal validation for the Qdrant FormulaQuery `sum` expression.

This script validates a conservative research subset:
1. Ordinary summation rescales scores as expected.
2. Negative terms can produce negative and zero totals in the tested subset.
3. Formula defaults can supply a missing term in the tested subset.
4. Missing terms without defaults are rejected.
5. A single overflowing candidate rejects the whole tested query.
6. Summation order matters in the tested floating-point subset.
7. REST, gRPC, and Python `:memory:` agree on the tested subset.
"""

from __future__ import annotations

import math
import random
import time
import urllib.request
from importlib.metadata import version as pkg_version

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FormulaQuery,
    PointStruct,
    Prefetch,
    SumExpression,
    VectorParams,
)


HOST = "127.0.0.1"
PORT = 6333
GRPC_PORT = 6334
BASE_QUERY_VECTOR = [1.0, 0.0]


def unique_collection_name(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


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


def build_safe_points() -> list[PointStruct]:
    return [
        PointStruct(id=1, vector=[1.0, 0.0], payload={"a": 2.0, "b": 3.0, "neg": -2.0}),
        PointStruct(id=2, vector=[0.9, 0.1], payload={"a": 4.0, "b": -5.0, "neg": -3.0}),
    ]


def build_invalid_points() -> list[PointStruct]:
    return [
        PointStruct(id=10, vector=[1.0, 0.0], payload={"huge": 2e38, "mixed": 1.0}),
        PointStruct(id=11, vector=[0.9, 0.1], payload={"huge": 2e38, "mixed": 2e38}),
    ]


def query_scores(client: QdrantClient, collection_name: str, formula: FormulaQuery) -> dict[int, float]:
    response = client.query_points(
        collection_name=collection_name,
        prefetch=Prefetch(query=BASE_QUERY_VECTOR, limit=16),
        query=formula,
        limit=16,
        with_payload=False,
        with_vectors=False,
    )
    return {int(point.id): float(point.score) for point in response.points}


def scores_match(actual: dict[int, float], expected: dict[int, float], tolerance: float = 1e-5) -> bool:
    if set(actual) != set(expected):
        return False
    for point_id, expected_score in expected.items():
        actual_score = actual[point_id]
        if not math.isclose(actual_score, expected_score, rel_tol=tolerance, abs_tol=tolerance):
            return False
    return True


def run_success_case(
    client: QdrantClient,
    collection_name: str,
    name: str,
    formula: FormulaQuery,
    expected_scores: dict[int, float],
    note: str,
) -> bool:
    try:
        actual_scores = query_scores(client, collection_name, formula)
        passed = scores_match(actual_scores, expected_scores)
        status = "PASS" if passed else "FAIL"
        print(
            f"{name}: {status} | expected_scores={expected_scores} | actual_scores={actual_scores} | note={note}"
        )
        return passed
    except Exception as exc:
        print(f"{name}: FAIL | unexpected_exception={type(exc).__name__}: {exc} | note={note}")
        return False


def run_error_case(
    client: QdrantClient,
    collection_name: str,
    name: str,
    formula: FormulaQuery,
    expected_any_fragments: tuple[str, ...],
    note: str,
) -> bool:
    try:
        actual_scores = query_scores(client, collection_name, formula)
        print(
            f"{name}: FAIL | expected_error_fragments={expected_any_fragments} | actual_scores={actual_scores} | note={note}"
        )
        return False
    except Exception as exc:
        message = str(exc)
        passed = any(fragment in message for fragment in expected_any_fragments)
        status = "PASS" if passed else "FAIL"
        print(
            f"{name}: {status} | error_type={type(exc).__name__} | error={message!r} | note={note}"
        )
        return passed


def run_transport(prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    safe_collection = unique_collection_name(
        "sum_formula_safe_grpc" if prefer_grpc else "sum_formula_safe_rest"
    )
    invalid_collection = unique_collection_name(
        "sum_formula_invalid_grpc" if prefer_grpc else "sum_formula_invalid_rest"
    )
    client = build_client(prefer_grpc)
    all_passed = True

    success_cases = [
        (
            "safe_sum",
            FormulaQuery(formula=SumExpression(sum=["a", "b"])),
            {1: 5.0, 2: -1.0},
            "Finite terms are added normally in the tested subset.",
        ),
        (
            "sum_with_negative_term",
            FormulaQuery(formula=SumExpression(sum=["a", "neg"])),
            {1: 0.0, 2: 1.0},
            "Negative terms can produce zero and positive totals in the tested subset.",
        ),
        (
            "missing_term_with_default",
            FormulaQuery(
                formula=SumExpression(sum=["missing_term", 2.0]),
                defaults={"missing_term": 3.0},
            ),
            {1: 5.0, 2: 5.0},
            "Formula defaults can supply a missing term in the tested subset.",
        ),
        (
            "order_sensitive_big_plus_small_minus_big",
            FormulaQuery(formula=SumExpression(sum=[1e20, 1.0, -1e20])),
            {1: 0.0, 2: 0.0},
            "The tested floating-point order `[1e20, 1.0, -1e20]` loses the small term and evaluates to 0 on all validated backends.",
        ),
        (
            "order_sensitive_big_minus_big_plus_small",
            FormulaQuery(formula=SumExpression(sum=[1e20, -1e20, 1.0])),
            {1: 1.0, 2: 1.0},
            "The tested floating-point order `[1e20, -1e20, 1.0]` preserves the final small term and evaluates to 1 on all validated backends.",
        ),
    ]

    error_cases = [
        (
            safe_collection,
            "missing_term_without_default",
            FormulaQuery(formula=SumExpression(sum=["missing_term", 2.0])),
            ("missing_term", "No value found"),
            "A missing term without defaults is rejected when it must be evaluated.",
        ),
        (
            invalid_collection,
            "overflow_sum_aborts_query",
            FormulaQuery(formula=SumExpression(sum=["huge", "huge"])),
            ("non-finite", "as f32 = inf"),
            "A single overflowing sum rejects the whole tested query instead of returning only the finite rows.",
        ),
        (
            invalid_collection,
            "mixed_query_aborts_on_single_overflow_candidate",
            FormulaQuery(formula=SumExpression(sum=["mixed", "huge"])),
            ("non-finite", "as f32 = inf"),
            "A single overflowing candidate in the rescored set rejects the whole tested query instead of returning only the finite rows.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=safe_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=safe_collection, points=build_safe_points(), wait=True)

        client.create_collection(
            collection_name=invalid_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=invalid_collection, points=build_invalid_points(), wait=True)

        print(f"\n--- sum formula validation ({transport}) ---")
        print(f"safe_collection={safe_collection}")
        print(f"invalid_collection={invalid_collection}")

        for case in success_cases:
            if not run_success_case(client, safe_collection, *case):
                all_passed = False

        for collection_name, name, formula, expected_any_fragments, note in error_cases:
            if not run_error_case(
                client,
                collection_name,
                name,
                formula,
                expected_any_fragments,
                note,
            ):
                all_passed = False

        return all_passed
    finally:
        for collection_name in [safe_collection, invalid_collection]:
            try:
                client.delete_collection(collection_name)
            except Exception as exc:
                print(f"cleanup_warning ({transport}, {collection_name}): {exc}")


def run_memory_probe() -> bool:
    safe_collection = unique_collection_name("sum_formula_safe_memory")
    invalid_collection = unique_collection_name("sum_formula_invalid_memory")
    client = QdrantClient(":memory:")
    all_passed = True

    success_cases = [
        (
            "safe_sum",
            FormulaQuery(formula=SumExpression(sum=["a", "b"])),
            {1: 5.0, 2: -1.0},
            "Finite terms are added normally in the tested subset.",
        ),
        (
            "sum_with_negative_term",
            FormulaQuery(formula=SumExpression(sum=["a", "neg"])),
            {1: 0.0, 2: 1.0},
            "Negative terms can produce zero and positive totals in the tested subset.",
        ),
        (
            "missing_term_with_default",
            FormulaQuery(
                formula=SumExpression(sum=["missing_term", 2.0]),
                defaults={"missing_term": 3.0},
            ),
            {1: 5.0, 2: 5.0},
            "Formula defaults can supply a missing term in the tested subset.",
        ),
        (
            "order_sensitive_big_plus_small_minus_big",
            FormulaQuery(formula=SumExpression(sum=[1e20, 1.0, -1e20])),
            {1: 0.0, 2: 0.0},
            "The tested floating-point order `[1e20, 1.0, -1e20]` loses the small term and evaluates to 0 on all validated backends.",
        ),
        (
            "order_sensitive_big_minus_big_plus_small",
            FormulaQuery(formula=SumExpression(sum=[1e20, -1e20, 1.0])),
            {1: 1.0, 2: 1.0},
            "The tested floating-point order `[1e20, -1e20, 1.0]` preserves the final small term and evaluates to 1 on all validated backends.",
        ),
    ]

    error_cases = [
        (
            safe_collection,
            "missing_term_without_default",
            FormulaQuery(formula=SumExpression(sum=["missing_term", 2.0])),
            ("missing_term", "No value found"),
            "A missing term without defaults is rejected when it must be evaluated.",
        ),
        (
            invalid_collection,
            "overflow_sum_aborts_query",
            FormulaQuery(formula=SumExpression(sum=["huge", "huge"])),
            ("non-finite", "as f32 = inf"),
            "A single overflowing sum rejects the whole tested query instead of returning only the finite rows.",
        ),
        (
            invalid_collection,
            "mixed_query_aborts_on_single_overflow_candidate",
            FormulaQuery(formula=SumExpression(sum=["mixed", "huge"])),
            ("non-finite", "as f32 = inf"),
            "A single overflowing candidate in the rescored set rejects the whole tested query instead of returning only the finite rows.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=safe_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=safe_collection, points=build_safe_points(), wait=True)

        client.create_collection(
            collection_name=invalid_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=invalid_collection, points=build_invalid_points(), wait=True)

        print("\n--- sum formula validation (memory) ---")
        print(f"safe_collection={safe_collection}")
        print(f"invalid_collection={invalid_collection}")

        for case in success_cases:
            if not run_success_case(client, safe_collection, *case):
                all_passed = False

        for collection_name, name, formula, expected_any_fragments, note in error_cases:
            if not run_error_case(
                client,
                collection_name,
                name,
                formula,
                expected_any_fragments,
                note,
            ):
                all_passed = False

        return all_passed
    finally:
        for collection_name in [safe_collection, invalid_collection]:
            try:
                client.delete_collection(collection_name)
            except Exception as exc:
                print(f"cleanup_warning (memory, {collection_name}): {exc}")


def main() -> int:
    try:
        server_info = fetch_server_info()
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant FormulaQuery sum validation")
    print(f"server_version={server_info.get('version')}")
    print(f"server_commit={server_info.get('commit')}")
    print(f"client_version={pkg_version('qdrant-client')}")

    rest_ok = run_transport(prefer_grpc=False)
    grpc_ok = run_transport(prefer_grpc=True)
    memory_ok = run_memory_probe()

    if rest_ok and grpc_ok and memory_ok:
        print("\nRESULT: PASS")
        return 0

    print("\nRESULT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
