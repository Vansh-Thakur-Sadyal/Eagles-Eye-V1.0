"""Provider-agnostic LLM client.

Eagles Eye never hardcodes a vendor.  SENTINEL_LLM_PROVIDER selects one of:

  none              deterministic template generation only (fully offline)
  openai_compatible any /v1/chat/completions endpoint (vLLM, LM Studio,
                    OpenRouter, Together, Groq, Mistral's API, ...)
  ollama            a local Ollama server - the natural pairing with the
                    operator's own GPU
  anthropic         the Anthropic Messages API
  mistral           Mistral's chat completions API

Every caller must work when the provider is ``none``: the reasoning agents fall
back to deterministic templates so the platform is never silently degraded into
"the LLM was down, so there is no incident summary".
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings

log = logging.getLogger("sentinel.llm")


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        self.provider = (s.llm_provider or "none").lower()
        self.base_url = s.llm_base_url.rstrip("/")
        self.api_key = s.llm_api_key
        self.model = s.llm_model
        self.timeout = s.llm_timeout

    @property
    def enabled(self) -> bool:
        if self.provider in ("none", ""):
            return False
        if self.provider == "ollama":
            return bool(self.model)
        if self.provider in ("anthropic", "mistral"):
            # Both have a fixed public endpoint, so no base_url is needed.
            return bool(self.api_key and self.model)
        return bool(self.base_url and self.model)

    def info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model or None,
            "base_url": self.base_url or None,
            "enabled": self.enabled,
            "api_key_configured": bool(self.api_key),
        }

    # ------------------------------------------------------------------ call
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 900,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not self.enabled:
            raise LLMUnavailable(f"LLM provider '{self.provider}' is not configured")
        try:
            if self.provider == "ollama":
                return self._ollama(messages, temperature, max_tokens)
            if self.provider == "anthropic":
                return self._anthropic(messages, temperature, max_tokens)
            return self._openai_compatible(messages, temperature, max_tokens, tools)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"LLM request failed: {exc}") from exc

    def _openai_compatible(self, messages, temperature, max_tokens, tools) -> str:
        url = f"{self.base_url}/chat/completions"
        if self.provider == "mistral" and not self.base_url:
            url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"].get("content") or ""

    def _ollama(self, messages, temperature, max_tokens) -> str:
        base = self.base_url or "http://localhost:11434"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                f"{base}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            r.raise_for_status()
            data = r.json()
        return data.get("message", {}).get("content", "")

    def _anthropic(self, messages, temperature, max_tokens) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        base = self.base_url or "https://api.anthropic.com"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                f"{base}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "system": system or None,
                    "messages": convo,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return "".join(parts)

    def complete_json(self, messages, *, temperature: float = 0.0, max_tokens: int = 900) -> Any:
        """Ask for JSON and parse defensively - models add prose around it."""
        text = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        return extract_json(text)

    def health(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "not configured", **self.info()}
        try:
            reply = self.complete(
                [{"role": "user", "content": "Reply with the single word: ready"}],
                max_tokens=12,
            )
            return {"ok": True, "reply": reply.strip()[:40], **self.info()}
        except Exception as exc:
            return {"ok": False, "reason": str(exc), **self.info()}


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of a model response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("model response contained no parsable JSON")


_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm() -> None:
    global _client
    _client = None
