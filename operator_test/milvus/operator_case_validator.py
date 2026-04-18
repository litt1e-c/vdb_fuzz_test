from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Iterable, Sequence


ERROR_CATEGORIES = {
    "parse_rejection",
    "unsupported",
    "type_rejection",
    "domain_error",
}

ERROR_CATEGORY_MODES = {"ignore", "warn", "strict"}


@dataclass(frozen=True)
class OperatorCase:
    mode: str
    name: str
    expr: str
    expected_ids: list[int] | None = None
    error_category: str | None = None


def classify_expected_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".lower()

    if any(token in message for token in ("divide by zero", "division by zero", "modulo by zero", "domain error")):
        return "domain_error"
    if "overflow" in message and "stack overflow" not in message:
        return "domain_error"

    if "unsupported data type" in message:
        return "unsupported"
    if any(token in message for token in ("not supported", "unsupported", "not implement", "not support")):
        return "unsupported"
    if "can only apply on constants" in message:
        return "unsupported"

    if any(
        token in message
        for token in (
            "type mismatch",
            "wrong type",
            "data type",
            "integer type",
            "float type",
            "double type",
            "array type",
            "json type",
            "varchar type",
            "string type",
            "bool type",
            "expected type",
            "invalid parameter",
            "val_not_set",
        )
    ):
        return "type_rejection"

    if any(
        token in message
        for token in (
            "parse",
            "parser",
            "syntax",
            "invalid expression",
            "mismatched input",
            "no viable alternative",
            "cannot parse",
        )
    ):
        return "parse_rejection"

    return "unsupported"


def expected_error_category_mode() -> str:
    mode = os.getenv("MILVUS_EXPECTED_ERROR_CATEGORY_MODE", "warn").strip().lower() or "warn"
    if mode not in ERROR_CATEGORY_MODES:
        return "warn"
    return mode


def _coerce_case(raw_case: OperatorCase | Sequence[object], default_expected_error: str) -> OperatorCase:
    if isinstance(raw_case, OperatorCase):
        case = raw_case
    else:
        if len(raw_case) not in (4, 5):
            raise ValueError(f"operator case must have 4 or 5 fields, got {len(raw_case)}: {raw_case!r}")
        mode = str(raw_case[0])
        name = str(raw_case[1])
        expr = str(raw_case[2])
        expected_ids = raw_case[3]
        error_category = raw_case[4] if len(raw_case) == 5 else None
        case = OperatorCase(
            mode=mode,
            name=name,
            expr=expr,
            expected_ids=list(expected_ids) if isinstance(expected_ids, (list, tuple)) else None,
            error_category=str(error_category) if error_category is not None else None,
        )

    if case.mode not in {"normal", "expected_error"}:
        raise ValueError(f"unsupported operator case mode: {case.mode}")

    if case.mode == "expected_error":
        category = case.error_category or default_expected_error
        if category not in ERROR_CATEGORIES:
            raise ValueError(f"unsupported expected_error category: {category}")
        return OperatorCase(
            mode=case.mode,
            name=case.name,
            expr=case.expr,
            expected_ids=None,
            error_category=category,
        )

    return case


def run_operator_cases(
    *,
    collection,
    tests: Iterable[OperatorCase | Sequence[object]],
    query_fn: Callable[[object, str], list[int]],
    title: str,
    default_expected_error: str = "unsupported",
) -> int:
    failed = 0
    total = 0
    normal_cases = 0
    expected_error_cases = 0
    category_mode = expected_error_category_mode()

    print(f"--- {title} ---")
    print(f"expected_error_category_mode={category_mode}")
    for raw_case in tests:
        case = _coerce_case(raw_case, default_expected_error)
        total += 1

        try:
            actual_ids = query_fn(collection, case.expr)
            if case.mode == "expected_error":
                expected_error_cases += 1
                failed += 1
                print(
                    f"{case.name}: FAIL | expr={case.expr} | "
                    f"expected_error={case.error_category} | actual={actual_ids}"
                )
                continue

            normal_cases += 1
            status = "PASS" if actual_ids == case.expected_ids else "FAIL"
            if status == "FAIL":
                failed += 1
            print(
                f"{case.name}: {status} | expr={case.expr} | "
                f"expected={case.expected_ids} | actual={actual_ids}"
            )
        except Exception as exc:
            if case.mode == "expected_error":
                expected_error_cases += 1
                actual_category = classify_expected_error(exc)
                category_matches = actual_category == case.error_category
                if not category_matches and category_mode == "strict":
                    failed += 1
                    print(
                        f"{case.name}: FAIL | expr={case.expr} | "
                        f"expected_error={case.error_category} | "
                        f"actual_error_category={actual_category} | "
                        f"reason=category_mismatch | actual={type(exc).__name__}: {exc}"
                    )
                    continue

                status = "PASS"
                if not category_matches and category_mode == "warn":
                    status = "PASS_WARN"
                print(
                    f"{case.name}: {status} | expr={case.expr} | "
                    f"expected_error={case.error_category} | "
                    f"actual_error_category={actual_category} | "
                    f"actual={type(exc).__name__}: {exc}"
                )
                continue

            normal_cases += 1
            failed += 1
            print(
                f"{case.name}: ERROR | expr={case.expr} | "
                f"error={type(exc).__name__}: {exc}"
            )

    summary = "PASS" if failed == 0 else "FAIL"
    print(
        f"summary: {summary} | failed={failed} | total={total} | "
        f"normal={normal_cases} | expected_error={expected_error_cases}"
    )
    return failed
