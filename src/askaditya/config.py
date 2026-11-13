"""Settings for the Ask-Aditya service, read from environment variables.

Nothing here names a hosted model: ``GROQ_MODEL`` and ``GEMINI_MODEL`` must be set in the environment.
A provider without a key and a model is simply not part of the chain.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_ORIGINS = ("https://adityajain16.vercel.app", "http://127.0.0.1:8765")
CONTACT_EMAIL = "itsaditya1610@gmail.com"


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _caps(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = int(v)
    return out


@dataclass(frozen=True)
class AskSettings:
    index_dir: Path = Path("private/index")
    allowed_origins: tuple[str, ...] = ALLOWED_ORIGINS
    provider_order: tuple[str, ...] = ("groq", "gemini")
    groq_api_key: str = ""
    groq_model: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_api_key: str = ""
    gemini_model: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    local_model: str = ""
    upstash_url: str = ""
    upstash_token: str = ""
    ip_per_minute: int = 10
    ip_per_day: int = 60
    global_daily_cap: int = 120
    provider_daily_caps: dict[str, int] = field(
        default_factory=lambda: {"groq": 100, "gemini": 100}
    )
    top_k: int = 5
    max_new_tokens: int = 220
    provider_timeout_s: float = 8.0
    test_hooks: bool = False
    contact_email: str = CONTACT_EMAIL
    ip_salt: str = "ask-aditya"
    eval_token: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AskSettings:
        e: Mapping[str, str] = os.environ if env is None else env
        order = tuple(
            p.strip().lower()
            for p in e.get("ASK_PROVIDER_ORDER", "groq,gemini").split(",")
            if p.strip()
        )
        caps = {"groq": 100, "gemini": 100}
        caps.update(_caps(e.get("ASK_PROVIDER_DAILY_CAPS", "")))
        return cls(
            index_dir=Path(e.get("ASK_INDEX_DIR", "private/index")),
            provider_order=order,
            groq_api_key=e.get("GROQ_API_KEY", ""),
            groq_model=e.get("GROQ_MODEL", ""),
            groq_base_url=e.get("GROQ_BASE_URL", cls.groq_base_url).rstrip("/"),
            gemini_api_key=e.get("GEMINI_API_KEY", ""),
            gemini_model=e.get("GEMINI_MODEL", ""),
            gemini_base_url=e.get("GEMINI_BASE_URL", cls.gemini_base_url).rstrip("/"),
            local_model=e.get("ASK_LOCAL_MODEL", ""),
            upstash_url=(e.get("UPSTASH_REDIS_REST_URL") or e.get("KV_REST_API_URL") or "").rstrip(
                "/"
            ),
            upstash_token=e.get("UPSTASH_REDIS_REST_TOKEN") or e.get("KV_REST_API_TOKEN") or "",
            ip_per_minute=_int(e, "ASK_IP_PER_MINUTE", 10),
            ip_per_day=_int(e, "ASK_IP_PER_DAY", 60),
            global_daily_cap=_int(e, "ASK_GLOBAL_DAILY_CAP", 120),
            provider_daily_caps=caps,
            top_k=_int(e, "ASK_TOP_K", 5),
            max_new_tokens=_int(e, "ASK_MAX_NEW_TOKENS", 220),
            provider_timeout_s=float(e.get("ASK_PROVIDER_TIMEOUT_S", "8") or 8),
            test_hooks=e.get("ASK_TEST_HOOKS", "") == "1",
            contact_email=e.get("ASK_CONTACT_EMAIL", CONTACT_EMAIL),
            ip_salt=e.get("ASK_IP_SALT", "ask-aditya"),
            eval_token=e.get("ASK_EVAL_TOKEN", ""),
        )
