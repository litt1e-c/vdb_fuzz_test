from __future__ import annotations

import json
import time
import warnings
import urllib.error
import urllib.request
from dataclasses import dataclass

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import requests
except Exception:
    requests = None


class ProviderError(RuntimeError):
    pass


@dataclass
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    wire_api: str = "chat_completions"
    responses_input_mode: str = "structured"
    reasoning_effort: str | None = None
    verbosity: str | None = None
    store: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class BaseProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.last_request_debug: dict | None = None

    def generate(self, system_prompt: str, messages: list[dict]) -> str:
        raise NotImplementedError

    def smoke_test(self) -> str:
        raise NotImplementedError

    def get_last_request_debug(self) -> dict | None:
        return self.last_request_debug

    def _record_request_debug(self, url: str, payload: dict, headers: dict) -> None:
        safe_headers = dict(headers)
        auth = safe_headers.get("Authorization")
        if auth:
            safe_headers["Authorization"] = "Bearer ***REDACTED***"
        api_key = safe_headers.get("x-api-key")
        if api_key:
            safe_headers["x-api-key"] = "***REDACTED***"
        self.last_request_debug = {
            "url": url,
            "headers": safe_headers,
            "payload": payload,
        }

    def _post_json(self, url: str, payload: dict, headers: dict) -> dict:
        if requests is not None:
            last_exc = None
            for attempt in range(3):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=120)
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    raise ProviderError(f"Network error calling {url}: {exc}") from exc
                raw = resp.text
                if resp.status_code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    raise ProviderError(f"HTTP {resp.status_code} calling {url}: {raw}")
                break
            else:  # pragma: no cover
                raise ProviderError(f"Network error calling {url}: {last_exc}")
        else:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise ProviderError(f"HTTP {exc.code} calling {url}: {body}") from exc
            except urllib.error.URLError as exc:
                raise ProviderError(f"Network error calling {url}: {exc}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Non-JSON response from {url}: {raw[:500]}") from exc


def _normalize_openai_compat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base:
        raise ProviderError("Empty base_url for OpenAI-compatible provider")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _normalize_openai_responses_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base:
        raise ProviderError("Empty base_url for OpenAI responses provider")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return base + "/responses"
    return base + "/v1/responses"


def _normalize_anthropic_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base:
        raise ProviderError("Empty base_url for Anthropic provider")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


class OpenAICompatProvider(BaseProvider):
    def _openai_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

    def _build_chat_payload(self, system_prompt: str, messages: list[dict]) -> dict:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        return payload

    def _generate_chat_completions(self, system_prompt: str, messages: list[dict]) -> str:
        url = _normalize_openai_compat_url(self.config.base_url)
        payload = self._build_chat_payload(system_prompt, messages)
        headers = self._openai_headers()
        self._record_request_debug(url, payload, headers)
        resp = self._post_json(url, payload, headers)
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ProviderError(f"Unexpected OpenAI-compatible response: {resp}") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return str(content)

    def _to_responses_input(self, system_prompt: str, messages: list[dict]) -> list[dict]:
        items = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            }
        ]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            items.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        return items

    def _to_simple_responses_input(self, system_prompt: str, messages: list[dict]) -> str:
        chunks = []
        if system_prompt.strip():
            chunks.append("System:\n" + system_prompt.strip())
        for msg in messages:
            role = (msg.get("role") or "user").strip()
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            chunks.append(f"{role.capitalize()}:\n{content.strip()}")
        return "\n\n".join(chunks).strip()

    def _extract_responses_text(self, resp: dict) -> str:
        if isinstance(resp.get("output_text"), str) and resp.get("output_text"):
            return resp["output_text"]
        outputs = resp.get("output", [])
        parts = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for block in item.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"} and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        if parts:
            return "\n".join(parts)
        raise ProviderError(f"Unexpected OpenAI responses payload: {resp}")

    def _build_responses_payload(self, system_prompt: str, messages: list[dict]) -> dict:
        input_mode = (self.config.responses_input_mode or "structured").strip().lower()
        payload = {
            "model": self.config.model,
            "input": (
                self._to_simple_responses_input(system_prompt, messages)
                if input_mode == "simple"
                else self._to_responses_input(system_prompt, messages)
            ),
        }
        if self.config.max_tokens is not None:
            payload["max_output_tokens"] = self.config.max_tokens
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.config.verbosity:
            payload["text"] = {"verbosity": self.config.verbosity}
        if self.config.store is not None:
            payload["store"] = self.config.store
        return payload

    def _generate_responses(self, system_prompt: str, messages: list[dict]) -> str:
        url = _normalize_openai_responses_url(self.config.base_url)
        payload = self._build_responses_payload(system_prompt, messages)
        headers = self._openai_headers()
        self._record_request_debug(url, payload, headers)
        resp = self._post_json(url, payload, headers)
        return self._extract_responses_text(resp)

    def generate(self, system_prompt: str, messages: list[dict]) -> str:
        wire_api = (self.config.wire_api or "chat_completions").strip().lower()
        if wire_api == "responses":
            return self._generate_responses(system_prompt, messages)
        return self._generate_chat_completions(system_prompt, messages)

    def smoke_test(self) -> str:
        wire_api = (self.config.wire_api or "chat_completions").strip().lower()
        headers = self._openai_headers()
        if wire_api == "responses":
            url = _normalize_openai_responses_url(self.config.base_url)
            payload = {
                "model": self.config.model,
                "input": "Hello, this is a test!",
            }
            self._record_request_debug(url, payload, headers)
            resp = self._post_json(url, payload, headers)
            return self._extract_responses_text(resp)

        url = _normalize_openai_compat_url(self.config.base_url)
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": "Hello, this is a test!"}],
        }
        self._record_request_debug(url, payload, headers)
        resp = self._post_json(url, payload, headers)
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ProviderError(f"Unexpected OpenAI-compatible response: {resp}") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return str(content)


class AnthropicProvider(BaseProvider):
    def generate(self, system_prompt: str, messages: list[dict]) -> str:
        url = _normalize_anthropic_url(self.config.base_url)
        payload = {
            "model": self.config.model,
            "system": system_prompt,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }
        self._record_request_debug(url, payload, headers)
        resp = self._post_json(url, payload, headers)
        try:
            content = resp["content"]
        except Exception as exc:
            raise ProviderError(f"Unexpected Anthropic response: {resp}") from exc
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return str(content)

    def smoke_test(self) -> str:
        return self.generate("", [{"role": "user", "content": "Hello, this is a test!"}])


def build_provider(config: ProviderConfig) -> BaseProvider:
    provider = config.provider.lower()
    if provider == "packycode" and not config.responses_input_mode:
        config.responses_input_mode = "simple"
    if provider in {"openai", "openai_compat", "codex", "packycode"}:
        return OpenAICompatProvider(config)
    if provider in {"anthropic", "claude"}:
        return AnthropicProvider(config)
    raise ProviderError(f"Unsupported provider: {config.provider}")
