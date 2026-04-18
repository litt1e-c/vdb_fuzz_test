from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Iterable, Sequence


ERROR_CATEGORIES = {
    "parse_rejection",
    "unsupported",
    "type_rejection",
    "validation_error",
}

ERROR_CATEGORY_MODES = {"ignore", "warn", "strict"}


@dataclass(frozen=True)
class OperatorCase:
    mode: str
    name: str
    expr: object
    expected_ids: list[str] | None = None
    error_category: str | None = None


def classify_expected_error(exc: Exception | str) -> str:
    message = str(exc).lower()

    if any(token in message for token in ("not supported", "unsupported", "not implement", "not support")):
        return "unsupported"

    if any(
        token in message
        for token in (
            'cannot use "value',
            "expected an integer value",
            "expected value to be bool",
            "expected value to be time.time",
            "expected value to be",
            "type should be []int",
            "type should be []float",
            "type should be []bool",
            "type should be []string",
            "got string",
            "got int",
            "could not parse string",
            "parseable string",
            "wrong type",
            "expected type",
            "data type filter cannot use",
        )
    ):
        return "type_rejection"

    if any(
        token in message
        for token in (
            'argument "where" has invalid value',
            "unknown field",
            "invalid where filter",
            "invalid 'where' filter",
            "validation",
        )
    ):
        return "validation_error"

    if any(token in message for token in ("syntax", "parser", "cannot parse", "parse error")):
        return "parse_rejection"

    return "validation_error"


def expected_error_category_mode() -> str:
    mode = os.getenv("WEAVIATE_EXPECTED_ERROR_CATEGORY_MODE", "warn").strip().lower() or "warn"
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
        expr = raw_case[2]
        expected_ids = raw_case[3]
        error_category = raw_case[4] if len(raw_case) == 5 else None
        case = OperatorCase(
            mode=mode,
            name=name,
            expr=expr,
            expected_ids=list(expected_ids) if isinstance(expected_ids, (list, tuple)) else None,
            error_category=str(error_category) if error_category is not None else None,
        )

    if case.mode not in {"normal", "expected_error", "empty_or_error"}:
        raise ValueError(f"unsupported operator case mode: {case.mode}")

    if case.mode in {"expected_error", "empty_or_error"}:
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
    subject,
    tests: Iterable[OperatorCase | Sequence[object]],
    query_fn: Callable[[object, object], list[str]],
    title: str,
    default_expected_error: str = "validation_error",
) -> int:
    failed = 0
    total = 0
    normal_cases = 0
    expected_error_cases = 0
    empty_or_error_cases = 0
    category_mode = expected_error_category_mode()

    print(f"--- {title} ---")
    print(f"expected_error_category_mode={category_mode}")
    for raw_case in tests:
        case = _coerce_case(raw_case, default_expected_error)
        total += 1

        try:
            actual_ids = query_fn(subject, case.expr)
            if case.mode == "expected_error":
                expected_error_cases += 1
                failed += 1
                print(
                    f"{case.name}: FAIL | expr={case.expr} | "
                    f"expected_error={case.error_category} | actual={actual_ids}"
                )
                continue
            if case.mode == "empty_or_error":
                empty_or_error_cases += 1
                if actual_ids:
                    failed += 1
                    print(
                        f"{case.name}: FAIL | expr={case.expr} | "
                        f"expected_empty_or_error={case.error_category} | actual={actual_ids}"
                    )
                else:
                    print(
                        f"{case.name}: PASS | expr={case.expr} | "
                        f"expected_empty_or_error={case.error_category} | actual=[]"
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
            if case.mode in {"expected_error", "empty_or_error"}:
                if case.mode == "expected_error":
                    expected_error_cases += 1
                    label = "expected_error"
                else:
                    empty_or_error_cases += 1
                    label = "expected_empty_or_error"
                actual_category = classify_expected_error(exc)
                category_matches = actual_category == case.error_category
                if not category_matches and category_mode == "strict":
                    failed += 1
                    print(
                        f"{case.name}: FAIL | expr={case.expr} | "
                        f"{label}={case.error_category} | "
                        f"actual_error_category={actual_category} | "
                        f"reason=category_mismatch | actual={type(exc).__name__}: {exc}"
                    )
                    continue

                status = "PASS"
                if not category_matches and category_mode == "warn":
                    status = "PASS_WARN"
                print(
                    f"{case.name}: {status} | expr={case.expr} | "
                    f"{label}={case.error_category} | "
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
        f"normal={normal_cases} | expected_error={expected_error_cases} | "
        f"empty_or_error={empty_or_error_cases}"
    )
    return failed
