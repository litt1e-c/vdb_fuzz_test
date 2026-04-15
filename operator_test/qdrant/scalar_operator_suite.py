#!/usr/bin/env python3
"""
Run the validated Qdrant scalar/filter operator checks as one deterministic suite.

This wrapper does not invent new semantics. It only orchestrates existing
operator validators that already have dedicated YAML oracle records.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


OPERATOR_DIR = Path(__file__).resolve().parent
SCALAR_OPERATORS = {
    "must": OPERATOR_DIR / "must_operator.py",
    "must_not": OPERATOR_DIR / "must_not_operator.py",
    "should": OPERATOR_DIR / "should_operator.py",
    "match_value": OPERATOR_DIR / "match_value_operator.py",
    "match_any": OPERATOR_DIR / "match_any_operator.py",
    "match_except": OPERATOR_DIR / "match_except_operator.py",
    "range": OPERATOR_DIR / "range_operator.py",
    "min_should": OPERATOR_DIR / "min_should_operator.py",
    "datetime_range": OPERATOR_DIR / "datetime_range_operator.py",
    "datetime_profile": OPERATOR_DIR / "datetime_profile_operator.py",
    "is_null": OPERATOR_DIR / "is_null_operator.py",
    "is_empty": OPERATOR_DIR / "is_empty_operator.py",
    "values_count": OPERATOR_DIR / "values_count_operator.py",
    "text": OPERATOR_DIR / "text_operator.py",
    "text_any": OPERATOR_DIR / "text_any_operator.py",
    "phrase": OPERATOR_DIR / "phrase_operator.py",
    "nested": OPERATOR_DIR / "nested_operator.py",
    "nested_key": OPERATOR_DIR / "nested_key_operator.py",
    "geo_bounding_box": OPERATOR_DIR / "geo_bounding_box_operator.py",
    "geo_radius": OPERATOR_DIR / "geo_radius_operator.py",
    "uuid_match": OPERATOR_DIR / "uuid_match_operator.py",
    "has_id": OPERATOR_DIR / "has_id_operator.py",
    "scroll_pagination": OPERATOR_DIR / "scroll_pagination_operator.py",
    "payload_mutation": OPERATOR_DIR / "payload_mutation_operator.py",
    "filter_index_equivalence": OPERATOR_DIR / "filter_index_equivalence_operator.py",
    "heterogeneous_payload": OPERATOR_DIR / "heterogeneous_payload_operator.py",
}

DEFAULT_OPERATOR_SET = [
    "must",
    "must_not",
    "should",
    "match_value",
    "match_any",
    "match_except",
    "range",
    "min_should",
    "datetime_range",
    "datetime_profile",
    "is_null",
    "is_empty",
    "values_count",
    "text",
    "text_any",
    "phrase",
    "nested",
    "nested_key",
    "geo_bounding_box",
    "geo_radius",
    "uuid_match",
    "has_id",
    "scroll_pagination",
    "payload_mutation",
    "filter_index_equivalence",
    "heterogeneous_payload",
]


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def parse_operator_list(raw: str) -> list[str]:
    token = raw.strip().lower()
    if token == "all":
        return list(DEFAULT_OPERATOR_SET)

    selected = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(SCALAR_OPERATORS))
    if unknown:
        raise ValueError(f"unknown scalar operator(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("empty scalar operator list")
    return selected


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"qdrant_scalar_{name}_operator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scalar operator module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_deterministic_collection_namer(module, name: str, run_id: str) -> None:
    if not hasattr(module, "unique_collection_name"):
        return

    counter = {"value": 0}

    def deterministic_name(*args) -> str:
        counter["value"] += 1
        arg_tokens = [slugify(arg, max_len=24) for arg in args]
        parts = [slugify(name, max_len=32), slugify(run_id, max_len=40), *arg_tokens, f"{counter['value']:02d}"]
        return "_".join(part for part in parts if part)

    module.unique_collection_name = deterministic_name


def run_operator(
    name: str,
    path: Path,
    host: str,
    port: int,
    grpc_port: int,
    run_id: str,
) -> bool:
    print(f"\n{'=' * 80}")
    print(f"Scalar operator: {name}")
    print(f"Runner: {path}")
    print(f"Target: {host}:{port} (grpc:{grpc_port})")
    print(f"Run ID: {run_id}")
    print(f"{'=' * 80}")

    module = load_module(name, path)
    install_deterministic_collection_namer(module, name, run_id)
    if hasattr(module, "HOST"):
        module.HOST = host
    if hasattr(module, "PORT"):
        module.PORT = port
    if hasattr(module, "GRPC_PORT"):
        module.GRPC_PORT = grpc_port
    child_argv = [
        str(path),
        "--host",
        host,
        "--port",
        str(port),
        "--grpc-port",
        str(grpc_port),
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = child_argv
        exit_code = int(module.main())
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception as exc:
        print(f"{name}: FAIL | unexpected_exception={type(exc).__name__}: {exc}")
        return False
    finally:
        sys.argv = old_argv

    ok = exit_code == 0
    print(f"Scalar operator {name}: {'PASS' if ok else 'FAIL'} (exit={exit_code})")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Qdrant scalar/filter operator validators")
    parser.add_argument("--host", default="127.0.0.1", help="Qdrant REST host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant REST port")
    parser.add_argument("--grpc-port", type=int, default=6334, dest="grpc_port", help="Qdrant gRPC port")
    parser.add_argument(
        "--operators",
        default="all",
        help="Comma-separated scalar operators to run, or 'all' (default: all)",
    )
    parser.add_argument("--run-id", default="scalar-operator-suite", help="Deterministic run id for collection naming")
    parser.add_argument("--list-operators", action="store_true", help="List available scalar operators and exit")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed operator")
    args = parser.parse_args(argv)

    if args.list_operators:
        print("\n".join(SCALAR_OPERATORS))
        return 0

    try:
        selected = parse_operator_list(args.operators)
    except ValueError as exc:
        print(f"scalar_operator_suite: FAIL | {exc}")
        return 2

    print("Qdrant scalar operator validation suite")
    print(f"Target: {args.host}:{args.port} (grpc:{args.grpc_port})")
    print(f"Run ID: {args.run_id}")
    print(f"Operators: {', '.join(selected)}")

    failed: list[str] = []
    for name in selected:
        path = SCALAR_OPERATORS[name]
        if not path.exists():
            print(f"{name}: FAIL | runner missing: {path}")
            failed.append(name)
            if args.stop_on_failure:
                break
            continue

        if not run_operator(name, path, args.host, args.port, args.grpc_port, args.run_id):
            failed.append(name)
            if args.stop_on_failure:
                break

    if failed:
        print(f"\nSummary: at least one scalar operator check failed: {', '.join(failed)}")
        return 1

    print("\nSummary: all scalar operator checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
