#!/usr/bin/env python3
"""Run the autonomous C3 triage loop from one raw mismatch report.

Typical usage:
    python c3_triage_agent/autonomous_triage_runner.py \
      --report-file c3_triage_agent/incidents/qdrant_test71_mismatch.txt \
      --provider openai \
      --model qwen3-coder-plus \
      --max-steps 6

What this script does:
1. load `.env` and provider settings
2. send the raw report plus workspace hints to the model
3. let the model request `run` / `write_file` actions
4. execute allowed local commands
5. feed command outputs back to the model
6. save the full trace under `c3_triage_agent/runs/<timestamp>/`
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from providers import ProviderConfig, ProviderError, build_provider


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKDIR = ROOT.parent
PROMPT_PATH = ROOT / "AUTO_AGENT_PROMPT.md"
HISTORY_CATALOG_PATH = ROOT / "history_issue_catalog.json"
RUNS_DIR = ROOT / "runs"
DEFAULT_ENV_FILE = ROOT / ".env"
REPROS_ROOT = ROOT / "repros"
AUTO_CASES_ROOT = ROOT / "auto_cases"
HISTORY_ROOT = DEFAULT_WORKDIR / "history_find_bug"

DEFAULT_ALLOWED_PREFIXES = [
    "python ",
    "python3 ",
    str((DEFAULT_WORKDIR / "venv" / "bin" / "python").resolve()) + " ",
    "rg ",
    "sed ",
    "cat ",
    "head ",
    "tail ",
    "ls ",
    "find ",
]

BANNED_PATTERNS = [
    r"\brm\b",
    r"\bsudo\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+checkout\b",
    r"\bdocker\b",
    r"\bcurl\b",
    r"\bwget\b",
    r"\bssh\b",
    r"\bscp\b",
    r">\s*/",
]


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
        if not key:
            continue
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def bootstrap_env(argv: list[str]) -> Path:
    env_file = DEFAULT_ENV_FILE
    for idx, arg in enumerate(argv):
        if arg == "--env-file" and idx + 1 < len(argv):
            env_file = Path(argv[idx + 1]).expanduser().resolve()
            break
        if arg.startswith("--env-file="):
            env_file = Path(arg.split("=", 1)[1]).expanduser().resolve()
            break
    load_dotenv_file(env_file, override=True)
    return env_file


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def env_bool_or_none(*names: str) -> bool | None:
    for name in names:
        raw = os.getenv(name, "").strip().lower()
        if not raw:
            continue
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous C3 triage loop from a raw report.")
    parser.add_argument("--report-file", required=True, help="Path to a raw mismatch report text file")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env file for API config")
    parser.add_argument("--provider", required=True, choices=["openai", "openai_compat", "anthropic", "codex", "claude", "packycode"])
    parser.add_argument("--model", required=True, help="Model name for the provider")
    parser.add_argument("--api-key", default=os.getenv("C3_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("C3_BASE_URL", ""))
    parser.add_argument("--wire-api", default=os.getenv("C3_WIRE_API", "chat_completions"), choices=["chat_completions", "responses"])
    parser.add_argument("--responses-input-mode", default=os.getenv("C3_RESPONSES_INPUT_MODE", ""), choices=["", "structured", "simple"])
    parser.add_argument("--reasoning-effort", default=os.getenv("C3_REASONING_EFFORT", ""))
    parser.add_argument("--verbosity", default=os.getenv("C3_VERBOSITY", ""))
    parser.add_argument("--store", default="", choices=["", "true", "false"], help="Set responses.store explicitly")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true", help="Skip the LLM call and only save the initial prompt context")
    parser.add_argument("--exclude-history", action="append", default=[], help="Exclude matching history issues from the prompt summary (repeatable)")
    parser.add_argument(
        "--history-prompt-mode",
        default="summary",
        choices=["summary", "hidden"],
        help="Whether to include history issue summaries directly in the initial prompt",
    )
    return parser.parse_args()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_history_catalog(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_run_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "commands").mkdir(exist_ok=True)
    return run_dir


def is_command_allowed(command: str, allowed_prefixes: list[str]) -> tuple[bool, str]:
    stripped = command.strip()
    if not stripped:
        return False, "empty command"
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, stripped):
            return False, f"banned pattern matched: {pattern}"
    if any(stripped.startswith(prefix) for prefix in allowed_prefixes):
        return True, ""
    return False, "command prefix not allowed"


def truncate_text(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...<truncated>...\n" + text[-half:]


def run_shell_command(command: str, workdir: Path, timeout_sec: int) -> dict:
    # Each model-requested command is executed locally and the combined
    # stdout/stderr is returned to the next model step for analysis.
    started = time.time()
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    duration = time.time() - started
    combined = proc.stdout
    if proc.stderr:
        combined += "\n[stderr]\n" + proc.stderr
    return {
        "command": command,
        "returncode": proc.returncode,
        "duration_sec": round(duration, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "combined_output": combined,
    }


def extract_json_block(text: str) -> dict | None:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates.extend(fenced)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_trace(path: Path, step: int, status: str, summary: str, analysis: str, actions: list[dict] | None = None) -> None:
    lines = [f"## Step {step}", f"- status: {status}"]
    if summary:
        lines.append(f"- summary: {summary}")
    if analysis:
        lines.append(f"- analysis: {analysis}")
    if actions:
        action_lines = []
        for action in actions:
            if action.get("type") == "run":
                action_lines.append(f"  - run `{action.get('command', '')}`")
            elif action.get("type") == "write_file":
                action_lines.append(f"  - write_file `{action.get('path', '')}`")
            else:
                action_lines.append(f"  - {action}")
        if action_lines:
            lines.append("- actions:")
            lines.extend(action_lines)
    path.write_text(
        (path.read_text(encoding="utf-8") + "\n\n" if path.exists() else "") + "\n".join(lines),
        encoding="utf-8",
    )


def write_provider_request_debug(path: Path, provider) -> None:
    request_debug = provider.get_last_request_debug() if hasattr(provider, "get_last_request_debug") else None
    if request_debug:
        write_json(path, request_debug)


def resolve_write_path(raw_path: str, workdir: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (workdir / path).resolve()
    else:
        path = path.resolve()
    allowed_roots = [REPROS_ROOT.resolve(), AUTO_CASES_ROOT.resolve()]
    for root in allowed_roots:
        try:
            path.relative_to(root)
            return path
        except ValueError:
            continue
    raise ValueError(f"path must stay under one of: {', '.join(str(root) for root in allowed_roots)}")
    return path


def write_file_action(path: Path, content: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    preview = "\n".join(content.splitlines()[:20])
    return {
        "path": str(path),
        "bytes_written": len(content.encode("utf-8")),
        "line_count": len(content.splitlines()),
        "preview": preview,
    }


def filter_history_items(items: list[dict], exclude_patterns: list[str]) -> list[dict]:
    if not exclude_patterns:
        return items
    filtered = []
    lowered_patterns = [p.lower() for p in exclude_patterns if p]
    for item in items:
        haystack = " ".join(
            str(item.get(key, ""))
            for key in ("engine", "issue_id", "title", "summary", "file")
        ).lower()
        if any(pattern in haystack for pattern in lowered_patterns):
            continue
        filtered.append(item)
    return filtered


def format_history_catalog(items: list[dict]) -> str:
    lines = []
    for item in items:
        lines.append(
            f"- [{item['engine']} issue{item['issue_id']}] {item['title']}: {item['summary']} ({item['file']})"
        )
    return "\n".join(lines)


def build_initial_prompt(report_text: str, run_dir: Path, history_items: list[dict], history_prompt_mode: str) -> str:
    report = truncate_text(report_text, 6000)
    if history_prompt_mode == "hidden":
        history_section = (
            "History issue summaries:\n"
            "- hidden for this evaluation run; use history_bug_root to search the workspace yourself\n\n"
        )
    else:
        history_text = format_history_catalog(history_items)
        history_section = "History issue summaries:\n" f"{history_text}\n\n"
    return (
        "Raw report:\n"
        f"{report}\n\n"
        "Workspace:\n"
        f"- root: {DEFAULT_WORKDIR}\n"
        f"- repros_root: {REPROS_ROOT}\n"
        f"- auto_cases_root: {AUTO_CASES_ROOT}\n"
        f"- history_bug_root: {HISTORY_ROOT}\n"
        f"- run_dir: {run_dir}\n\n"
        "Default targets:\n"
        "- milvus host=127.0.0.1 port=19531\n"
        "- qdrant host=127.0.0.1 rest=6333 grpc=6334\n"
        "- weaviate host=127.0.0.1 rest=8080 grpc=50051\n\n"
        f"{history_section}"
        "Response rules:\n"
        "- return JSON only\n"
        "- status is need_more_data or final\n"
        "- max 4 actions per step\n"
        "- action types: run, write_file\n"
    )


def build_followup_prompt(step: int, action_results: list[dict]) -> str:
    payload = {
        "step": step,
        "action_results": action_results,
    }
    return "Continue autonomous triage. Return JSON only.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> int:
    bootstrapped_env = bootstrap_env(sys.argv[1:])
    args = parse_args()
    report_path = Path(args.report_file).resolve()
    report_text = load_text(report_path)
    history_catalog = load_history_catalog(HISTORY_CATALOG_PATH)
    prompt_history_items = filter_history_items(history_catalog, args.exclude_history)
    run_dir = ensure_run_dir()
    write_json(
        run_dir / "runner_config.json",
        {
            "mode": "autonomous_raw_report",
            "env_file": str(bootstrapped_env),
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
            "wire_api": args.wire_api,
            "responses_input_mode": args.responses_input_mode,
            "reasoning_effort": args.reasoning_effort,
            "verbosity": args.verbosity,
            "api_key_masked": mask_secret(args.api_key),
            "api_key_length": len(args.api_key or ""),
            "report_file": str(report_path),
            "exclude_history": args.exclude_history,
            "history_prompt_count": len(prompt_history_items),
            "history_prompt_mode": args.history_prompt_mode,
        },
    )
    (run_dir / "report_snapshot.txt").write_text(report_text, encoding="utf-8")
    write_json(run_dir / "history_prompt_items.json", {"items": prompt_history_items})
    initial_prompt = build_initial_prompt(report_text, run_dir, prompt_history_items, args.history_prompt_mode)
    (run_dir / "system_prompt.md").write_text(load_text(PROMPT_PATH), encoding="utf-8")
    (run_dir / "prompt_step_0_user.md").write_text(initial_prompt, encoding="utf-8")
    if args.dry_run:
        payload = {
            "run_dir": str(run_dir),
            "report_file": str(report_path),
            "note": "dry-run only previews the prompt context; it does not call the model, so no analysis steps are generated",
            "initial_prompt_preview": truncate_text(initial_prompt, 4000),
        }
        write_json(run_dir / "dry_run_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.api_key or not args.base_url:
        raise SystemExit("Missing --api-key or --base-url. You can also export C3_API_KEY and C3_BASE_URL.")

    system_prompt = load_text(PROMPT_PATH)
    provider = build_provider(
        ProviderConfig(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            wire_api=args.wire_api,
            responses_input_mode=(
                args.responses_input_mode
                or ("simple" if args.provider == "packycode" and args.wire_api == "responses" else "structured")
            ),
            reasoning_effort=(args.reasoning_effort or None),
            verbosity=(args.verbosity or None),
            store=(
                (args.store == "true")
                if args.store
                else (
                    False
                    if env_bool_or_none("C3_DISABLE_RESPONSE_STORAGE") is True
                    else env_bool_or_none("C3_STORE")
                )
            ),
        )
    )

    messages = [{"role": "user", "content": initial_prompt}]
    transcript_path = run_dir / "transcript.jsonl"
    trace_path = run_dir / "analysis_trace.md"
    transcript_path.write_text(
        json.dumps({"step": -1, "role": "system", "content": system_prompt}, ensure_ascii=False) + "\n"
        + json.dumps({"step": 0, "role": "user", "content": initial_prompt}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    for step in range(args.max_steps):
        # One loop iteration = one model turn. The model may ask for shell
        # actions, we execute them, then send the results back as the next
        # user message.
        try:
            response_text = provider.generate(system_prompt, messages)
        except ProviderError as exc:
            write_provider_request_debug(run_dir / f"provider_request_step_{step}.json", provider)
            error_payload = {"step": step, "provider_error": str(exc)}
            write_json(run_dir / "provider_error.json", error_payload)
            print(json.dumps(error_payload, ensure_ascii=False, indent=2))
            return 1
        write_provider_request_debug(run_dir / f"provider_request_step_{step}.json", provider)

        transcript_path.write_text(
            transcript_path.read_text(encoding="utf-8") + json.dumps({"step": step, "role": "assistant", "content": response_text}, ensure_ascii=False) + "\n"
            if transcript_path.exists()
            else json.dumps({"step": step, "role": "assistant", "content": response_text}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        parsed = extract_json_block(response_text)
        if not parsed:
            raw_path = run_dir / f"raw_response_step_{step}.txt"
            raw_path.write_text(response_text, encoding="utf-8")
            print(f"Model response was not valid JSON. Saved to {raw_path}")
            return 1

        write_json(run_dir / f"response_step_{step}.json", parsed)
        status = parsed.get("status")
        summary = str(parsed.get("summary", "") or "")
        analysis = str(parsed.get("analysis", "") or "")
        if status == "final":
            final_payload = parsed.get("final", {})
            append_trace(trace_path, step, str(status), summary, analysis)
            write_json(run_dir / "final_verdict.json", final_payload)
            print(json.dumps({"step": step, "status": status, "summary": summary, "analysis": analysis}, ensure_ascii=False, indent=2))
            print(json.dumps(final_payload, ensure_ascii=False, indent=2))
            return 0

        if status != "need_more_data":
            print(f"Unexpected status at step {step}: {status}")
            return 1

        actions = parsed.get("actions", [])[:4]
        append_trace(trace_path, step, str(status), summary, analysis, actions)
        print(json.dumps({"step": step, "status": status, "summary": summary, "analysis": analysis, "actions": actions}, ensure_ascii=False, indent=2))
        action_results = []
        for idx, action in enumerate(actions):
            action_type = action.get("type")
            label = action.get("label", f"step_{step}_{idx}")
            if action_type == "run":
                command = action.get("command", "")
                ok, reason = is_command_allowed(command, DEFAULT_ALLOWED_PREFIXES)
                if not ok:
                    result = {
                        "type": "run",
                        "label": label,
                        "command": command,
                        "returncode": -1,
                        "duration_sec": 0.0,
                        "combined_output": f"rejected by runner: {reason}",
                    }
                else:
                    result = run_shell_command(command, workdir=DEFAULT_WORKDIR, timeout_sec=args.timeout_sec)
                    result["type"] = "run"
                    result["label"] = label
                    write_json(run_dir / "commands" / f"step_{step}_{idx}_{label}.json", result)
                action_results.append(
                    {
                        "type": "run",
                        "label": label,
                        "command": result["command"],
                        "returncode": result["returncode"],
                        "duration_sec": result["duration_sec"],
                        "combined_output": truncate_text(result["combined_output"], 5000),
                    }
                )
                continue

            if action_type == "write_file":
                raw_path = action.get("path", "")
                content = action.get("content", "")
                try:
                    path = resolve_write_path(raw_path, DEFAULT_WORKDIR)
                    write_result = write_file_action(path, content)
                    result = {
                        "type": "write_file",
                        "label": label,
                        "path": str(path),
                        "status": "ok",
                        "bytes_written": write_result["bytes_written"],
                        "line_count": write_result["line_count"],
                        "preview": truncate_text(write_result["preview"], 2000),
                    }
                except Exception as exc:
                    result = {
                        "type": "write_file",
                        "label": label,
                        "path": raw_path,
                        "status": "error",
                        "error": str(exc),
                    }
                write_json(run_dir / "commands" / f"step_{step}_{idx}_{label}.json", result)
                action_results.append(result)
                continue

            result = {
                "type": action_type or "",
                "label": label,
                "status": "error",
                "error": f"unsupported action type: {action_type}",
            }
            write_json(run_dir / "commands" / f"step_{step}_{idx}_{label}.json", result)
            action_results.append(result)

        messages.append({"role": "assistant", "content": response_text})
        followup_prompt = build_followup_prompt(step, action_results)
        messages.append({"role": "user", "content": followup_prompt})
        (run_dir / f"prompt_step_{step + 1}_user.md").write_text(followup_prompt, encoding="utf-8")
        transcript_path.write_text(
            transcript_path.read_text(encoding="utf-8")
            + json.dumps({"step": step, "role": "user_action_results", "content": action_results}, ensure_ascii=False)
            + "\n"
            + json.dumps({"step": step + 1, "role": "user", "content": followup_prompt}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    timeout_payload = {"status": "incomplete", "reason": "max steps reached", "run_dir": str(run_dir)}
    write_json(run_dir / "timeout.json", timeout_payload)
    print(json.dumps(timeout_payload, ensure_ascii=False, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
