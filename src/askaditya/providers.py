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

    def __init__(self, message: str, kind: str = "error", retry_after: int = 0) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after  # seconds, from the provider's Retry-After header on a 429


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


_DURATION = re.compile(
    r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$"
)


def _duration_s(text: str) -> float | None:
    """Seconds from a plain number or a Groq-style duration such as ``7.66s``, ``1m5.2s`` or ``250ms``."""
    text = (text or "").strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = _DURATION.match(text)
    if not text or not m or not any(m.groups()):
        return None
    h, mi, sec, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + sec + ms / 1000


def _retry_after(response: httpx.Response, default: int) -> int:
    """How long to leave a rate-limited model alone: the ``Retry-After`` header, else the time Groq says its
    token (then request) window needs to reset (``x-ratelimit-reset-tokens``), else ``default``."""
    for name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        seconds = _duration_s(response.headers.get(name, ""))
        if seconds is not None:
            return max(1, min(int(seconds) + 1, 3600))
    return default


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


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
            raise ProviderUnavailable(f"{self.name}: 429", "rate_limit", _retry_after(r, 20))
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
            raise ProviderUnavailable("gemini: 429", "rate_limit", _retry_after(r, 30))
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
    """Try each provider in order, and move on before a free tier would answer 429.

    Accounting is by what the providers actually report, not by worst-case guesses:

    * a model's tokens in the last 60 seconds (a sliding window of ten-second buckets) and in the current day
      are the sum of the real ``usage`` of finished calls plus a reservation for each call in flight;
    * the reservation is the recent average real usage of that model (``DEFAULT_RESERVE`` until there is any),
      and it is replaced by the real figure when the call returns, or removed if the call failed;
    * a 429 from a provider takes that model out of the chain until the time its ``Retry-After`` names.

    When nothing can answer, ``Unavailable.retry_after`` is the number of seconds until the earliest budget or
    cooldown frees up (never less than one).
    """

    DEFAULT_RESERVE = 1000

    def __init__(self, providers: list[Provider], limiter: Limiter, settings: AskSettings) -> None:
        self.providers = providers
        self.limiter = limiter
        self.settings = settings
        self._recent: dict[str, list[int]] = {}
        self._cooldown: dict[str, float] = {}

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.providers]

    def _reserve(self, name: str) -> int:
        recent = self._recent.get(name)
        return int(sum(recent) / len(recent)) if recent else self.DEFAULT_RESERVE

    def _token_wait(self, name: str, reserve: int, now: float) -> tuple[str, int] | None:
        """None when the call fits; otherwise (reason code, seconds until it would fit)."""
        cfg, lim = self.settings, self.limiter
        tday = cfg.token_caps_day.get(name)
        if tday is not None and lim.tokens_today(name, now) + reserve > tday:
            return "budget_tokens_day", int(86400 - now % 86400) + 1
        tmin = cfg.token_caps_minute.get(name)
        if tmin is not None:
            buckets = lim.tokens_window(name, now)
            total = sum(c for _, c in buckets)
            if total + reserve > tmin:
                need, freed = total + reserve - tmin, 0
                wait = 60
                for (
                    start,
                    count,
                ) in buckets:  # oldest first: the time at which enough of them has aged out
                    freed += count
                    if freed >= need:
                        wait = int(start + 60 - now) + 1
                        break
                return "budget_tokens_minute", max(1, wait)
        return None

    def generate(
        self,
        system: str,
        user: str,
        force_fail: bool = False,
        exclude: set[str] | None = None,
        only: set[str] | None = None,
    ) -> Generation:
        attempts: list[str] = []
        waits: list[int] = []
        if force_fail or not self.providers:
            raise Unavailable(
                "no provider available",
                ["chain: forced" if force_fail else "chain: no_providers"],
                60,
            )
        lim, cfg = self.limiter, self.settings
        if not lim.take_global(cfg.global_daily_cap):
            raise Unavailable(
                "global daily cap reached",
                ["global: cap_day"],
                int(86400 - lim.clock() % 86400) + 1,
            )
        for p in self.providers:
            if (exclude and p.name in exclude) or (only and p.name not in only):
                continue
            now = lim.clock()
            cool = self._cooldown.get(p.name, 0.0)
            if cool > now:
                attempts.append(f"{p.name}: cooldown")
                waits.append(int(cool - now) + 1)
                continue
            cap = cfg.provider_daily_caps.get(p.name)
            if cap is not None and not lim.take_provider(p.name, cap):
                attempts.append(f"{p.name}: budget_requests_day")
                waits.append(int(86400 - now % 86400) + 1)
                continue
            tracked = p.name in cfg.token_caps_minute or p.name in cfg.token_caps_day
            reserve = self._reserve(p.name)
            if tracked:
                blocked = self._token_wait(p.name, reserve, now)
                if blocked:
                    attempts.append(f"{p.name}: {blocked[0]}")
                    waits.append(blocked[1])
                    continue
                bucket = lim.add_tokens(p.name, reserve, now)
            try:
                text = p.complete(system, user, cfg.max_new_tokens)
            except ProviderUnavailable as exc:
                attempts.append(str(exc))
                if tracked:
                    lim.correct_tokens(
                        p.name, -reserve, bucket, lim.clock()
                    )  # nothing was generated
                if exc.kind == "rate_limit":
                    ra = exc.retry_after or 20
                    self._cooldown[p.name] = lim.clock() + ra
                    attempts.append(f"{p.name}: cooldown_set {ra}s")
                    waits.append(ra)
                else:
                    waits.append(5)
                continue
            usage = dict(getattr(p, "last_usage", {}) or {})
            real = int(usage.get("total_tokens") or usage.get("totalTokenCount") or 0)
            if real:
                self._recent.setdefault(p.name, []).append(real)
                del self._recent[p.name][:-20]
            if tracked and real:
                lim.correct_tokens(p.name, real - reserve, bucket, lim.clock())
            return Generation(text, p.name, attempts, usage)
        raise Unavailable(
            "; ".join(attempts) or "no provider", attempts, max(1, min(waits) if waits else 60)
        )


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
