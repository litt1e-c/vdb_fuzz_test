"""
Minimal validation for the Qdrant FormulaQuery `div` expression.

This script validates a conservative research subset:
1. Ordinary numeric division rescales scores as expected.
2. Division is lazily evaluated when the left operand is exactly zero.
3. Division by zero without an explicit fallback is rejected.
4. `by_zero_default` is honored for non-zero numerators in the tested subset.
5. Extremely large quotients that produce non-finite tested scores are rejected.
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
    DivExpression,
    DivParams,
    FormulaQuery,
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


def build_points() -> list[PointStruct]:
    return [
        PointStruct(id=1, vector=[1.0, 0.0], payload={"num": 10.0}),
        PointStruct(id=2, vector=[0.9, 0.1], payload={"num": 4.0}),
        PointStruct(id=3, vector=[0.8, 0.2], payload={"num": 0.0}),
    ]


def query_scores(
    client: QdrantClient,
    collection_name: str,
    formula: FormulaQuery,
) -> dict[int, float]:
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
    expected_fragments: tuple[str, ...],
    note: str,
) -> bool:
    try:
        actual_scores = query_scores(client, collection_name, formula)
        print(
            f"{name}: FAIL | expected_error_fragments={expected_fragments} | actual_scores={actual_scores} | note={note}"
        )
        return False
    except Exception as exc:
        message = str(exc)
        passed = all(fragment in message for fragment in expected_fragments)
        status = "PASS" if passed else "FAIL"
        print(
            f"{name}: {status} | error_type={type(exc).__name__} | error={message!r} | note={note}"
        )
        return passed


def run_transport(prefer_grpc: bool) -> bool:
    transport = "gRPC" if prefer_grpc else "REST"
    collection_name = unique_collection_name(
        "div_formula_operator_grpc" if prefer_grpc else "div_formula_operator_rest"
    )
    client = build_client(prefer_grpc)
    all_passed = True

    success_cases = [
        (
            "safe_division",
            FormulaQuery(formula=DivExpression(div=DivParams(left="num", right=2.0))),
            {1: 5.0, 2: 2.0, 3: 0.0},
            "Ordinary numeric division rescales each prefetched point score according to `num / 2.0`.",
        ),
        (
            "left_zero_short_circuit_missing_right",
            FormulaQuery(formula=DivExpression(div=DivParams(left=0.0, right="missing"))),
            {1: 0.0, 2: 0.0, 3: 0.0},
            "A constant zero left operand short-circuits division and does not evaluate the missing right operand in the tested subset.",
        ),
        (
            "right_zero_with_default",
            FormulaQuery(
                formula=DivExpression(div=DivParams(left="num", right=0.0, by_zero_default=7.0))
            ),
            {1: 7.0, 2: 7.0, 3: 0.0},
            "The tested zero-divisor fallback is returned for non-zero numerators, while a zero numerator still short-circuits to 0.",
        ),
    ]

    error_cases = [
        (
            "right_zero_without_default",
            FormulaQuery(formula=DivExpression(div=DivParams(left="num", right=0.0))),
            ("non-finite",),
            "Division by zero without an explicit fallback is rejected in the tested subset.",
        ),
        (
            "missing_right_without_default",
            FormulaQuery(formula=DivExpression(div=DivParams(left="num", right="missing"))),
            ("missing",),
            "A missing right operand without defaults is rejected when it must be evaluated.",
        ),
        (
            "overflowing_quotient",
            FormulaQuery(formula=DivExpression(div=DivParams(left="num", right=1e-38))),
            ("non-finite",),
            "A quotient that exceeds the tested finite score range is rejected instead of returning a non-finite score.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print(f"\n--- div formula validation ({transport}) ---")
        print(f"collection={collection_name}")

        for case in success_cases:
            if not run_success_case(client, collection_name, *case):
                all_passed = False

        for name, formula, expected_fragments, note in error_cases:
            if not run_error_case(client, collection_name, name, formula, expected_fragments, note):
                all_passed = False

        return all_passed
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning ({transport}): {exc}")


def run_memory_probe() -> bool:
    collection_name = unique_collection_name("div_formula_operator_memory")
    client = QdrantClient(":memory:")
    all_passed = True

    success_cases = [
        (
            "safe_division",
            FormulaQuery(formula=DivExpression(div=DivParams(left="num", right=2.0))),
            {1: 5.0, 2: 2.0, 3: 0.0},
            "Ordinary numeric division rescales each prefetched point score according to `num / 2.0`.",
        ),
        (
            "left_zero_short_circuit_missing_right",
            FormulaQuery(formula=DivExpression(div=DivParams(left=0.0, right="missing"))),
            {1: 0.0, 2: 0.0, 3: 0.0},
            "A constant zero left operand short-circuits division and does not evaluate the missing right operand in the tested subset.",
        ),
        (
            "right_zero_with_default",
            FormulaQuery(
                formula=DivExpression(div=DivParams(left="num", right=0.0, by_zero_default=7.0))
            ),
            {1: 7.0, 2: 7.0, 3: 0.0},
            "The tested zero-divisor fallback is returned for non-zero numerators, while a zero numerator still short-circuits to 0.",
        ),
    ]

    error_cases = [
        (
            "right_zero_without_default",
            FormulaQuery(formula=DivExpression(div=DivParams(left="num", right=0.0))),
            ("non-finite",),
            "Division by zero without an explicit fallback is rejected in the tested subset.",
        ),
        (
            "missing_right_without_default",
            FormulaQuery(formula=DivExpression(div=DivParams(left="num", right="missing"))),
            ("missing",),
            "A missing right operand without defaults is rejected when it must be evaluated.",
        ),
        (
            "overflowing_quotient",
            FormulaQuery(formula=DivExpression(div=DivParams(left="num", right=1e-38))),
            ("non-finite",),
            "A quotient that exceeds the tested finite score range is rejected instead of returning a non-finite score.",
        ),
    ]

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )
        client.upsert(collection_name=collection_name, points=build_points(), wait=True)

        print("\n--- div formula validation (memory) ---")
        print(f"collection={collection_name}")

        for case in success_cases:
            if not run_success_case(client, collection_name, *case):
                all_passed = False

        for name, formula, expected_fragments, note in error_cases:
            if not run_error_case(client, collection_name, name, formula, expected_fragments, note):
                all_passed = False

        return all_passed
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:
            print(f"cleanup_warning (memory): {exc}")


def main() -> int:
    try:
        server_info = fetch_server_info()
    except Exception as exc:
        server_info = {"version": "unknown", "commit": "unknown", "error": repr(exc)}

    print("Qdrant FormulaQuery div validation")
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
