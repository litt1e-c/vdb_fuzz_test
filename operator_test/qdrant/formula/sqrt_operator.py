"""
Minimal validation for the Qdrant FormulaQuery `sqrt` expression.

This script validates a conservative research subset:
1. Ordinary square roots rescales scores as expected.
2. `sqrt(0)` returns 0 in the tested subset.
3. Formula defaults can supply a missing input in the tested subset.
4. Negative inputs are rejected.
5. A single overflowing square-root score rejects the whole tested query.
6. REST, gRPC, and Python `:memory:` agree on the tested subset.
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
    SqrtExpression,
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
        PointStruct(id=1, vector=[1.0, 0.0], payload={"pos": 9.0, "zero": 0.0, "neg": -4.0}),
        PointStruct(id=2, vector=[0.9, 0.1], payload={"pos": 16.0, "zero": 0.0, "neg": -1.0}),
    ]


def build_overflow_points() -> list[PointStruct]:
    return [
        PointStruct(id=10, vector=[1.0, 0.0], payload={"huge": 1e80}),
        PointStruct(id=11, vector=[0.9, 0.1], payload={"huge": 4.0}),
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
        "sqrt_formula_safe_grpc" if prefer_grpc else "sqrt_formula_safe_rest"
    )
    overflow_collection = unique_collection_name(
        "sqrt_formula_overflow_grpc" if prefer_grpc else "sqrt_formula_overflow_rest"
    )
    client = build_client(prefer_grpc)
    all_passed = True

    success_cases = [
        (
            "safe_sqrt",
            FormulaQuery(formula=SqrtExpression(sqrt="pos")),
            {1: 3.0, 2: 4.0},
            "Finite non-negative inputs are rescored as ordinary square roots in the tested subset.",
        ),
        (
            "zero_sqrt",
            FormulaQuery(formula=SqrtExpression(sqrt="zero")),
            {1: 0.0, 2: 0.0},
            "The tested `sqrt(0)` case evaluates to 0 on all validated backends.",
        ),
        (
            "missing_input_with_default",
            FormulaQuery(formula=SqrtExpression(sqrt="missing_input"), defaults={"missing_input": 25.0}),
            {1: 5.0, 2: 5.0},
            "Formula defaults can supply a missing square-root input in the tested subset.",
        ),
    ]

    error_cases = [
        (
            safe_collection,
            "negative_sqrt",
            FormulaQuery(formula=SqrtExpression(sqrt="neg")),
            ("non-finite", "√"),
            "Negative square-root inputs are rejected in the tested subset.",
        ),
        (
            safe_collection,
            "missing_input_without_default",
            FormulaQuery(formula=SqrtExpression(sqrt="missing_input")),
            ("missing_input", "No value found"),
            "A missing square-root input without defaults is rejected when it must be evaluated.",
        ),
        (
            overflow_collection,
            "overflowing_sqrt_aborts_query",
            FormulaQuery(formula=SqrtExpression(sqrt="huge")),
            ("non-finite", "as f32 = inf"),
            "A single overflowing square-root score rejects the whole tested query instead of silently dropping only the overflowing row.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=safe_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=safe_collection, points=build_safe_points(), wait=True)

        client.create_collection(
            collection_name=overflow_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=overflow_collection, points=build_overflow_points(), wait=True)

        print(f"\n--- sqrt formula validation ({transport}) ---")
        print(f"safe_collection={safe_collection}")
        print(f"overflow_collection={overflow_collection}")

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
        for collection_name in [safe_collection, overflow_collection]:
            try:
                client.delete_collection(collection_name)
            except Exception as exc:
                print(f"cleanup_warning ({transport}, {collection_name}): {exc}")


def run_memory_probe() -> bool:
    safe_collection = unique_collection_name("sqrt_formula_safe_memory")
    overflow_collection = unique_collection_name("sqrt_formula_overflow_memory")
    client = QdrantClient(":memory:")
    all_passed = True

    success_cases = [
        (
            "safe_sqrt",
            FormulaQuery(formula=SqrtExpression(sqrt="pos")),
            {1: 3.0, 2: 4.0},
            "Finite non-negative inputs are rescored as ordinary square roots in the tested subset.",
        ),
        (
            "zero_sqrt",
            FormulaQuery(formula=SqrtExpression(sqrt="zero")),
            {1: 0.0, 2: 0.0},
            "The tested `sqrt(0)` case evaluates to 0 on all validated backends.",
        ),
        (
            "missing_input_with_default",
            FormulaQuery(formula=SqrtExpression(sqrt="missing_input"), defaults={"missing_input": 25.0}),
            {1: 5.0, 2: 5.0},
            "Formula defaults can supply a missing square-root input in the tested subset.",
        ),
    ]

    error_cases = [
        (
            safe_collection,
            "negative_sqrt",
            FormulaQuery(formula=SqrtExpression(sqrt="neg")),
            ("non-finite", "√"),
            "Negative square-root inputs are rejected in the tested subset.",
        ),
        (
            safe_collection,
            "missing_input_without_default",
            FormulaQuery(formula=SqrtExpression(sqrt="missing_input")),
            ("missing_input", "No value found"),
            "A missing square-root input without defaults is rejected when it must be evaluated.",
        ),
        (
            overflow_collection,
            "overflowing_sqrt_aborts_query",
            FormulaQuery(formula=SqrtExpression(sqrt="huge")),
            ("non-finite", "as f32 = inf"),
            "A single overflowing square-root score rejects the whole tested query instead of silently dropping only the overflowing row.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=safe_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=safe_collection, points=build_safe_points(), wait=True)

        client.create_collection(
            collection_name=overflow_collection,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=overflow_collection, points=build_overflow_points(), wait=True)

        print("\n--- sqrt formula validation (memory) ---")
        print(f"safe_collection={safe_collection}")
        print(f"overflow_collection={overflow_collection}")

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
        for collection_name in [safe_collection, overflow_collection]:
            try:
                client.delete_collection(collection_name)
            except Exception as exc:
                print(f"cleanup_warning (memory, {collection_name}): {exc}")


def main() -> int:
    try:
        server_info = fetch_server_info()
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant FormulaQuery sqrt validation")
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
