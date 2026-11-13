"""Generators for the free-tier chain: Groq (OpenAI-compatible) and Gemini, plus a local model for development.

Model ids come from the environment only. Any failure (HTTP error, 429, timeout, an empty or malformed
body) raises ``ProviderUnavailable`` so the chain can move on; nothing here can start billing, because a
free-tier key without a payment method has no billing to start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import AskSettings
from .limits import Limiter


class ProviderUnavailable(RuntimeError):
    """One provider could not answer; ``kind`` is rate_limit, error or empty."""

    def __init__(self, message: str, kind: str = "error") -> None:
        super().__init__(message)
        self.kind = kind


class Unavailable(RuntimeError):
    """Every provider in the chain failed or is out of budget (the API answers 503, fallback limit)."""


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class GroqProvider:
    name = "groq"

    def __init__(
        self,
        key: str,
        model: str,
        base_url: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model, self.base_url, self.timeout = model, base_url, timeout
        self._headers = {"Authorization": f"Bearer {key}"}
        self._client = httpx.Client(transport=transport, timeout=timeout)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions", json=body, headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"groq: {type(exc).__name__}") from exc
        if r.status_code == 429:
            raise ProviderUnavailable("groq: 429", "rate_limit")
        if r.status_code >= 400:
            raise ProviderUnavailable(f"groq: HTTP {r.status_code}")
        try:
            text = _strip_thinking(r.json()["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderUnavailable("groq: malformed response") from exc
        if not text:
            raise ProviderUnavailable("groq: empty", "empty")
        return text


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        key: str,
        model: str,
        base_url: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.base_url, self.timeout = base_url, timeout
        self._headers = {"x-goog-api-key": key}
        self._client = httpx.Client(transport=transport, timeout=timeout)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
        }
        try:
            r = self._client.post(
                f"{self.base_url}/{self.model}:generateContent", json=body, headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"gemini: {type(exc).__name__}") from exc
        if r.status_code == 429:
            raise ProviderUnavailable("gemini: 429", "rate_limit")
        if r.status_code >= 400:
            raise ProviderUnavailable(f"gemini: HTTP {r.status_code}")
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = _strip_thinking("".join(p.get("text", "") for p in parts))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderUnavailable("gemini: no candidates (blocked or empty)", "empty") from exc
        if not text:
            raise ProviderUnavailable("gemini: empty", "empty")
        return text


class LocalProvider:
    """A local Hugging Face model through this repository's own generator, for development and tests."""

    name = "local"

    def __init__(self, model: str, device: str = "auto") -> None:
        from rag_chatbot.llm import LocalLLM

        self._llm = LocalLLM(model, device)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        from rag_chatbot.llm import Prompt

        text = self._llm.generate(Prompt(system=system, user=user), max_tokens, 0.0)
        if not text.strip():
            raise ProviderUnavailable("local: empty", "empty")
        return text.strip()


@dataclass
class Generation:
    text: str
    provider: str
    attempts: list[str]


class ProviderChain:
    """Try each provider in order. A provider is skipped when its own daily budget or the global daily cap
    is spent, so the service stops asking before a free tier would answer 429."""

    def __init__(self, providers: list[Provider], limiter: Limiter, settings: AskSettings) -> None:
        self.providers = providers
        self.limiter = limiter
        self.settings = settings

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.providers]

    def generate(self, system: str, user: str, force_fail: bool = False) -> Generation:
        attempts: list[str] = []
        if force_fail or not self.providers:
            raise Unavailable("no provider available")
        if not self.limiter.take_global(self.settings.global_daily_cap):
            raise Unavailable("global daily cap reached")
        for p in self.providers:
            cap = self.settings.provider_daily_caps.get(p.name)
            if cap is not None and not self.limiter.take_provider(p.name, cap):
                attempts.append(f"{p.name}: daily budget spent")
                continue
            try:
                return Generation(
                    p.complete(system, user, self.settings.max_new_tokens), p.name, attempts
                )
            except ProviderUnavailable as exc:
                attempts.append(str(exc))
        raise Unavailable("; ".join(attempts) or "no provider")


def build_chain(
    settings: AskSettings, limiter: Limiter, transport: httpx.BaseTransport | None = None
) -> ProviderChain:
    providers: list[Provider] = []
    for name in settings.provider_order:
        if name == "groq" and settings.groq_api_key and settings.groq_model:
            providers.append(
                GroqProvider(
                    settings.groq_api_key,
                    settings.groq_model,
                    settings.groq_base_url,
                    settings.provider_timeout_s,
                    transport,
                )
            )
        elif name == "gemini" and settings.gemini_api_key and settings.gemini_model:
            providers.append(
                GeminiProvider(
                    settings.gemini_api_key,
                    settings.gemini_model,
                    settings.gemini_base_url,
                    settings.provider_timeout_s,
                    transport,
                )
            )
        elif name == "local" and settings.local_model:
            providers.append(LocalProvider(settings.local_model))
    return ProviderChain(providers, limiter, settings)
