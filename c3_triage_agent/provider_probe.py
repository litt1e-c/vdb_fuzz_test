#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import requests
except Exception as exc:  # pragma: no cover
    print(f"requests import failed: {exc}", file=sys.stderr)
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = ROOT / ".env"
RUNS_DIR = ROOT / "runs"


def load_dotenv_file(path: Path, override: bool = True) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def normalize_endpoint_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if not base:
        raise ValueError("empty base_url")
    for suffix in ("/chat/completions", "/responses"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.endswith("/v1"):
        root = base
    else:
        root = base + "/v1"
    if endpoint == "responses":
        return root + "/responses"
    if endpoint == "chat":
        return root + "/chat/completions"
    raise ValueError(f"unsupported endpoint: {endpoint}")


def build_responses_input(system_prompt: str, user_prompt: str, input_mode: str) -> str | list[dict]:
    mode = (input_mode or "simple").strip().lower()
    if mode == "structured":
        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ]
    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"


def build_payload(endpoint: str, payload_mode: str, model: str, input_mode: str, reasoning_effort: str, verbosity: str, store: str) -> dict:
    if payload_mode == "minimal":
        system_prompt = ""
        user_prompt = "Hello, this is a test!"
    elif payload_mode == "autonomous":
        system_prompt = (
            "You are an autonomous triage agent for vector-database filter failures. "
            "Return JSON only."
        )
        user_prompt = (
            "Seed: 1171171189\n"
            "Transport: host=127.0.0.1 rest=6333 grpc=6334 prefer_grpc=False\n"
            "[Test 71] Expr: ((NOT (scores_array is null) OR meta_json.config.version == 5) "
            "AND c15 <= -3.4028234663852886e+38)\n"
            "Pandas: 43 | Qdrant: 30\n"
            "MISMATCH missing IDs: [1573, 4230, 1800]\n"
            "Evidence: Qdrant row stores c15 as -3.402823466385289e+38."
        )
    else:
        raise ValueError(f"unsupported payload_mode: {payload_mode}")

    if endpoint == "responses":
        payload = {
            "model": model,
            "input": build_responses_input(system_prompt, user_prompt, input_mode),
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if verbosity:
            payload["text"] = {"verbosity": verbosity}
        if store in {"true", "false"}:
            payload["store"] = store == "true"
        return payload

    payload = {
        "model": model,
        "messages": (
            [{"role": "user", "content": user_prompt}]
            if not system_prompt
            else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        ),
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Packy/OpenAI-compatible provider behavior with minimal and autonomous payloads.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env file")
    parser.add_argument("--model", required=True, help="Model name to test")
    parser.add_argument("--base-url", default="", help="Base URL; defaults to C3_BASE_URL from .env")
    parser.add_argument("--api-key", default="", help="API key; defaults to C3_API_KEY from .env")
    parser.add_argument("--endpoint", choices=["responses", "chat", "both"], default="both", help="Which endpoint(s) to test")
    parser.add_argument("--payload-mode", choices=["minimal", "autonomous", "both"], default="both", help="Which payload style(s) to test")
    parser.add_argument("--responses-input-mode", choices=["simple", "structured"], default=os.getenv("C3_RESPONSES_INPUT_MODE", "simple") or "simple")
    parser.add_argument("--reasoning-effort", default=os.getenv("C3_REASONING_EFFORT", ""))
    parser.add_argument("--verbosity", default=os.getenv("C3_VERBOSITY", ""))
    parser.add_argument(
        "--store",
        choices=["omit", "true", "false"],
        default=(
            "false"
            if os.getenv("C3_DISABLE_RESPONSE_STORAGE", "").strip().lower() in {"1", "true", "yes", "on"}
            else (os.getenv("C3_STORE", "").strip().lower() if os.getenv("C3_STORE", "").strip().lower() in {"true", "false"} else "omit")
        ),
        help="Set responses.store explicitly",
    )
    parser.add_argument("--timeout-sec", type=int, default=120)
    args = parser.parse_args()

    load_dotenv_file(Path(args.env_file).expanduser().resolve(), override=True)
    api_key = args.api_key or os.getenv("C3_API_KEY", "")
    base_url = args.base_url or os.getenv("C3_BASE_URL", "")
    if not api_key or not base_url:
        raise SystemExit("Missing API key or base URL. Set them in .env or pass --api-key/--base-url.")

    endpoint_modes = ["responses", "chat"] if args.endpoint == "both" else [args.endpoint]
    payload_modes = ["minimal", "autonomous"] if args.payload_mode == "both" else [args.payload_mode]

    run_dir = RUNS_DIR / ("provider_probe_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for endpoint in endpoint_modes:
        for payload_mode in payload_modes:
            url = normalize_endpoint_url(base_url, endpoint)
            payload = build_payload(
                endpoint=endpoint,
                payload_mode=payload_mode,
                model=args.model,
                input_mode=args.responses_input_mode,
                reasoning_effort=args.reasoning_effort,
                verbosity=args.verbosity,
                store=args.store,
            )
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            label = f"{endpoint}_{payload_mode}"
            request_record = {
                "label": label,
                "url": url,
                "payload": payload,
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer ***REDACTED***",
                },
            }
            write_json(run_dir / f"{label}_request.json", request_record)

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=args.timeout_sec)
                body = resp.text
                result = {
                    "label": label,
                    "status_code": resp.status_code,
                    "ok": resp.ok,
                    "body_preview": body[:4000],
                }
                try:
                    result["json"] = resp.json()
                except Exception:
                    result["json"] = None
            except Exception as exc:
                result = {
                    "label": label,
                    "status_code": None,
                    "ok": False,
                    "error": repr(exc),
                }

            write_json(run_dir / f"{label}_response.json", result)
            summary.append(result)

    print(json.dumps({"run_dir": str(run_dir), "results": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
