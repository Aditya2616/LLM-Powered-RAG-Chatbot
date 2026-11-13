"""Generators for the chain: two Groq models, then Gemini on its free tier, plus a local model for development.

Model ids come from the environment only. Any failure (HTTP error, 429, timeout, an empty or malformed
body) raises ``ProviderUnavailable`` so the chain can move on. Nothing here can start billing: a free-tier key
without a payment method has no billing to start, and the chain also keeps its own request and token budgets
so it stops asking before a provider would answer 429.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    """Every provider in the chain failed or is out of budget (the API answers 503, fallback limit).

    ``attempts`` lists one reason code per provider that was tried or skipped, and ``retry_after`` is the number
    of seconds until a per-minute budget frees up when that is the only thing in the way (else 0)."""

    def __init__(
        self, message: str, attempts: list[str] | None = None, retry_after: int = 0
    ) -> None:
        super().__init__(message)
        self.attempts = attempts or []
        self.retry_after = retry_after


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def estimate_tokens(system: str, user: str, max_tokens: int) -> int:
    """A deliberately high estimate of what one call costs against a tokens-per-minute budget: about one
    token per 3.5 characters of prompt, plus the whole completion allowance."""
    return int((len(system) + len(user)) / 3.5) + max_tokens


class GroqProvider:
    """An OpenAI-compatible chat endpoint. ``name`` is the budget key (groq, groq-backup)."""

    def __init__(
        self,
        key: str,
        model: str,
        base_url: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
        name: str = "groq",
        reasoning_effort: str = "",
    ) -> None:
        self.name, self.model, self.base_url, self.timeout = name, model, base_url, timeout
        self.reasoning_effort = reasoning_effort
        self.last_usage: dict = {}
        self._headers = {"Authorization": f"Bearer {key}"}
        self._client = httpx.Client(transport=transport, timeout=timeout)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions", json=body, headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{self.name}: {type(exc).__name__}") from exc
        if r.status_code == 429:
            raise ProviderUnavailable(f"{self.name}: 429", "rate_limit")
        if r.status_code >= 400:
            raise ProviderUnavailable(f"{self.name}: HTTP {r.status_code}")
        try:
            data = r.json()
            text = _strip_thinking(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderUnavailable(f"{self.name}: malformed response") from exc
        self.last_usage = data.get("usage") or {}
        if not text:
            raise ProviderUnavailable(f"{self.name}: empty", "empty")
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
        self.last_usage: dict = {}
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
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = _strip_thinking("".join(p.get("text", "") for p in parts))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderUnavailable("gemini: no candidates (blocked or empty)", "empty") from exc
        self.last_usage = data.get("usageMetadata") or {}
        if not text:
            raise ProviderUnavailable("gemini: empty", "empty")
        return text


class LocalProvider:
    """A local Hugging Face model through this repository's own generator, for development and tests."""

    name = "local"

    def __init__(self, model: str, device: str = "auto") -> None:
        from rag_chatbot.llm import LocalLLM

        self._llm = LocalLLM(model, device)
        self.last_usage: dict = {}

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
    attempts: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class ProviderChain:
    """Try each provider in order. A provider is skipped when its own daily request budget, its per-minute or
    per-day token budget, or the global daily cap is spent, so the service stops asking before a free tier
    would answer 429. Token budgets use ``estimate_tokens`` and are corrected with the provider's reported
    usage after a call."""

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
            raise Unavailable(
                "no provider available", ["chain: forced" if force_fail else "chain: no_providers"]
            )
        if not self.limiter.take_global(self.settings.global_daily_cap):
            raise Unavailable("global daily cap reached", ["global: cap_day"])
        cfg = self.settings
        est = estimate_tokens(system, user, cfg.max_new_tokens)
        minute_blocked: list[bool] = []
        for p in self.providers:
            cap = cfg.provider_daily_caps.get(p.name)
            if cap is not None and not self.limiter.take_provider(p.name, cap):
                attempts.append(f"{p.name}: budget_requests_day")
                continue
            tmin, tday = cfg.token_caps_minute.get(p.name), cfg.token_caps_day.get(p.name)
            if (tmin is not None or tday is not None) and not self.limiter.take_tokens(
                p.name, est, tmin, tday
            ):
                block = getattr(self.limiter, "last_block", "") or "day"
                attempts.append(f"{p.name}: budget_tokens_{block}")
                minute_blocked.append(block == "minute")
                continue
            try:
                text = p.complete(system, user, cfg.max_new_tokens)
            except ProviderUnavailable as exc:
                attempts.append(str(exc))
                if tmin is not None or tday is not None:
                    self.limiter.adjust_tokens(p.name, -est)  # nothing was generated
                continue
            usage = dict(getattr(p, "last_usage", {}) or {})
            real = int(usage.get("total_tokens") or usage.get("totalTokenCount") or 0)
            if real and (tmin is not None or tday is not None):
                self.limiter.adjust_tokens(p.name, real - est)
            return Generation(text, p.name, attempts, usage)
        only_minute = (
            bool(minute_blocked)
            and all(minute_blocked)
            and len(minute_blocked) == len(self.providers)
        )
        retry = int(60 - self.limiter.clock() % 60) + 1 if only_minute else 0
        raise Unavailable("; ".join(attempts) or "no provider", attempts, retry)


def build_chain(
    settings: AskSettings, limiter: Limiter, transport: httpx.BaseTransport | None = None
) -> ProviderChain:
    providers: list[Provider] = []
    s = settings
    for name in s.provider_order:
        if name == "groq" and s.groq_api_key and s.groq_model:
            providers.append(
                GroqProvider(
                    s.groq_api_key,
                    s.groq_model,
                    s.groq_base_url,
                    s.provider_timeout_s,
                    transport,
                    "groq",
                    s.groq_reasoning_effort,
                )
            )
        elif name == "groq-backup" and s.groq_api_key and s.groq_model_backup:
            providers.append(
                GroqProvider(
                    s.groq_api_key,
                    s.groq_model_backup,
                    s.groq_base_url,
                    s.provider_timeout_s,
                    transport,
                    "groq-backup",
                    s.groq_backup_reasoning_effort,
                )
            )
        elif name == "gemini" and s.gemini_enabled and s.gemini_api_key and s.gemini_model:
            providers.append(
                GeminiProvider(
                    s.gemini_api_key,
                    s.gemini_model,
                    s.gemini_base_url,
                    s.provider_timeout_s,
                    transport,
                )
            )
        elif name == "local" and s.local_model:
            providers.append(LocalProvider(s.local_model))
    return ProviderChain(providers, limiter, s)
