"""
Minimal validation for the Qdrant FormulaQuery `ln` expression.

This script validates a conservative research subset:
1. Ordinary natural logarithms rescales scores as expected.
2. `ln(1)` returns 0 in the tested subset.
3. Positive inputs smaller than 1 produce finite negative scores in the tested subset.
4. Formula defaults can supply a missing input in the tested subset.
5. Zero and negative inputs are rejected.
6. A single invalid candidate rejects the whole tested query.
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
    LnExpression,
    PointStruct,
    Prefetch,
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
        PointStruct(id=1, vector=[1.0, 0.0], payload={"pos": math.e**2, "one": 1.0, "frac": 0.5}),
        PointStruct(id=2, vector=[0.9, 0.1], payload={"pos": math.e, "one": 1.0, "frac": 0.25}),
    ]


def build_invalid_points() -> list[PointStruct]:
    return [
        PointStruct(id=10, vector=[1.0, 0.0], payload={"mixed": math.e, "zero": 0.0, "neg": -1.0}),
        PointStruct(id=11, vector=[0.9, 0.1], payload={"mixed": 0.0, "zero": 0.0, "neg": -4.0}),
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
        "ln_formula_safe_grpc" if prefer_grpc else "ln_formula_safe_rest"
    )
    invalid_collection = unique_collection_name(
        "ln_formula_invalid_grpc" if prefer_grpc else "ln_formula_invalid_rest"
    )
    client = build_client(prefer_grpc)
    all_passed = True

    success_cases = [
        (
            "safe_ln_positive",
            FormulaQuery(formula=LnExpression(ln="pos")),
            {1: 2.0, 2: 1.0},
            "Finite positive inputs greater than 1 are rescored as ordinary natural logarithms in the tested subset.",
        ),
        (
            "ln_one",
            FormulaQuery(formula=LnExpression(ln="one")),
            {1: 0.0, 2: 0.0},
            "The tested `ln(1)` case evaluates to 0 on all validated backends.",
        ),
        (
            "ln_fractional_positive",
            FormulaQuery(formula=LnExpression(ln="frac")),
            {1: math.log(0.5), 2: math.log(0.25)},
            "Positive inputs smaller than 1 produce finite negative scores in the tested subset.",
        ),
        (
            "missing_input_with_default",
            FormulaQuery(formula=LnExpression(ln="missing_input"), defaults={"missing_input": math.e}),
            {1: 1.0, 2: 1.0},
            "Formula defaults can supply a missing logarithm input in the tested subset.",
        ),
    ]

    error_cases = [
        (
            invalid_collection,
            "ln_zero",
            FormulaQuery(formula=LnExpression(ln="zero")),
            ("non-finite", "ln(0"),
            "Zero logarithm inputs are rejected in the tested subset.",
        ),
        (
            invalid_collection,
            "ln_negative",
            FormulaQuery(formula=LnExpression(ln="neg")),
            ("non-finite", "ln(-"),
            "Negative logarithm inputs are rejected in the tested subset.",
        ),
        (
            safe_collection,
            "missing_input_without_default",
            FormulaQuery(formula=LnExpression(ln="missing_input")),
            ("missing_input", "No value found"),
            "A missing logarithm input without defaults is rejected when it must be evaluated.",
        ),
        (
            invalid_collection,
            "mixed_query_aborts_on_single_invalid_candidate",
            FormulaQuery(formula=LnExpression(ln="mixed")),
            ("non-finite", "ln(0"),
            "A single invalid candidate in the rescored set rejects the whole tested query instead of returning only the valid rows.",
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

        print(f"\n--- ln formula validation ({transport}) ---")
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
    safe_collection = unique_collection_name("ln_formula_safe_memory")
    invalid_collection = unique_collection_name("ln_formula_invalid_memory")
    client = QdrantClient(":memory:")
    all_passed = True

    success_cases = [
        (
            "safe_ln_positive",
            FormulaQuery(formula=LnExpression(ln="pos")),
            {1: 2.0, 2: 1.0},
            "Finite positive inputs greater than 1 are rescored as ordinary natural logarithms in the tested subset.",
        ),
        (
            "ln_one",
            FormulaQuery(formula=LnExpression(ln="one")),
            {1: 0.0, 2: 0.0},
            "The tested `ln(1)` case evaluates to 0 on all validated backends.",
        ),
        (
            "ln_fractional_positive",
            FormulaQuery(formula=LnExpression(ln="frac")),
            {1: math.log(0.5), 2: math.log(0.25)},
            "Positive inputs smaller than 1 produce finite negative scores in the tested subset.",
        ),
        (
            "missing_input_with_default",
            FormulaQuery(formula=LnExpression(ln="missing_input"), defaults={"missing_input": math.e}),
            {1: 1.0, 2: 1.0},
            "Formula defaults can supply a missing logarithm input in the tested subset.",
        ),
    ]

    error_cases = [
        (
            invalid_collection,
            "ln_zero",
            FormulaQuery(formula=LnExpression(ln="zero")),
            ("non-finite", "ln(0"),
            "Zero logarithm inputs are rejected in the tested subset.",
        ),
        (
            invalid_collection,
            "ln_negative",
            FormulaQuery(formula=LnExpression(ln="neg")),
            ("non-finite", "ln(-"),
            "Negative logarithm inputs are rejected in the tested subset.",
        ),
        (
            safe_collection,
            "missing_input_without_default",
            FormulaQuery(formula=LnExpression(ln="missing_input")),
            ("missing_input", "No value found"),
            "A missing logarithm input without defaults is rejected when it must be evaluated.",
        ),
        (
            invalid_collection,
            "mixed_query_aborts_on_single_invalid_candidate",
            FormulaQuery(formula=LnExpression(ln="mixed")),
            ("non-finite", "ln(0"),
            "A single invalid candidate in the rescored set rejects the whole tested query instead of returning only the valid rows.",
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

        print("\n--- ln formula validation (memory) ---")
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

    print("Qdrant FormulaQuery ln validation")
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
