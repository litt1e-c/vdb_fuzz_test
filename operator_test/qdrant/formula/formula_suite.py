#!/usr/bin/env python3
"""
Run the existing Qdrant FormulaQuery operator validators as one coverage job.

The individual formula validators already contain the oracle logic and YAML
records. This wrapper only injects the target server address so the coverage
scheduler can run them on non-default ports.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


FORMULA_DIR = Path(__file__).resolve().parent
FORMULA_OPERATORS = {
    "abs": FORMULA_DIR / "abs_operator.py",
    "div": FORMULA_DIR / "div_operator.py",
    "exp": FORMULA_DIR / "exp_operator.py",
    "ln": FORMULA_DIR / "ln_operator.py",
    "log10": FORMULA_DIR / "log10_operator.py",
    "mult": FORMULA_DIR / "mult_operator.py",
    "neg": FORMULA_DIR / "neg_operator.py",
    "pow": FORMULA_DIR / "pow_operator.py",
    "sqrt": FORMULA_DIR / "sqrt_operator.py",
    "sum": FORMULA_DIR / "sum_operator.py",
}


def slugify(value: object, max_len: int = 48) -> str:
    text = str(value).strip().lower()
    pieces = [ch if ch.isalnum() else "-" for ch in text]
    collapsed = "".join(pieces).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return (collapsed or "x")[:max_len]


def parse_operator_list(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(FORMULA_OPERATORS)

    selected = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(FORMULA_OPERATORS))
    if unknown:
        raise ValueError(f"unknown formula operator(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("empty formula operator list")
    return selected


def load_operator_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"qdrant_formula_{name}_operator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load formula operator module: {path}")

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


def run_operator(name: str, path: Path, host: str, port: int, grpc_port: int, run_id: str) -> bool:
    print(f"\n{'=' * 80}")
    print(f"Formula operator: {name}")
    print(f"Runner: {path}")
    print(f"Target: {host}:{port} (grpc:{grpc_port})")
    print(f"Run ID: {run_id}")
    print(f"{'=' * 80}")

    module = load_operator_module(name, path)
    module.HOST = host
    module.PORT = port
    module.GRPC_PORT = grpc_port
    install_deterministic_collection_namer(module, name, run_id)

    if not hasattr(module, "main"):
        print(f"{name}: FAIL | missing main() in {path}")
        return False

    try:
        exit_code = int(module.main())
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception as exc:
        print(f"{name}: FAIL | unexpected_exception={type(exc).__name__}: {exc}")
        return False

    ok = exit_code == 0
    print(f"Formula operator {name}: {'PASS' if ok else 'FAIL'} (exit={exit_code})")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Qdrant FormulaQuery operator validators")
    parser.add_argument("--host", default="127.0.0.1", help="Qdrant REST host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant REST port")
    parser.add_argument("--grpc-port", type=int, default=6334, dest="grpc_port", help="Qdrant gRPC port")
    parser.add_argument(
        "--operators",
        default="all",
        help="Comma-separated formula operators to run, or 'all' (default: all)",
    )
    parser.add_argument("--run-id", default="formula-operator-suite", help="Deterministic run id for collection naming")
    parser.add_argument("--list-operators", action="store_true", help="List available formula operators and exit")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed operator")
    args = parser.parse_args(argv)

    if args.list_operators:
        print("\n".join(FORMULA_OPERATORS))
        return 0

    try:
        selected = parse_operator_list(args.operators)
    except ValueError as exc:
        print(f"formula_suite: FAIL | {exc}")
        return 2

    print("Qdrant formula operator validation suite")
    print(f"Target: {args.host}:{args.port} (grpc:{args.grpc_port})")
    print(f"Run ID: {args.run_id}")
    print(f"Operators: {', '.join(selected)}")

    failed: list[str] = []
    for name in selected:
        path = FORMULA_OPERATORS[name]
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
        print(f"\nSummary: at least one formula operator check failed: {', '.join(failed)}")
        return 1

    print("\nSummary: all formula operator checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
