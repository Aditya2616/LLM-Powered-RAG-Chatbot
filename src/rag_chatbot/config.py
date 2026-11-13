"""Settings, read from environment variables (and an optional ``.env`` file).

Nothing is read at import time. ``Settings.from_env()`` is called by the CLI and the API factory, and
tests build ``Settings`` directly. Hosted model ids are never defaulted: choosing a hosted provider
without naming its model in the environment is an error with a clear message.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

EMBEDDING_PROVIDERS = ("huggingface", "openai", "gemini", "hashing")
LLM_PROVIDERS = ("local", "openai", "gemini", "extractive")
RETRIEVERS = ("dense", "bm25", "hybrid")


class ConfigError(ValueError):
    """Raised for a missing or invalid setting. The message says which variable to set."""


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _opt_float(env: Mapping[str, str], name: str, default: float | None) -> float | None:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"off", "none", "false"}:
        return None
    return _float(env, name, default or 0.0)


@dataclass(frozen=True)
class Settings:
    # storage
    data_dir: Path = Path("data")
    # providers
    embeddings_provider: str = "huggingface"
    llm_provider: str = "local"
    device: str = "auto"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = ""
    openai_embedding_model: str = ""
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = ""
    gemini_embedding_model: str = ""
    reranker_model: str = ""
    # retrieval
    chunk_size: int = 500
    chunk_overlap: int = 75
    top_k: int = 4
    retriever: str = "hybrid"
    min_dense_score: float | None = None
    # generation
    max_new_tokens: int = 160
    temperature: float = 0.0
    few_shot: bool = False
    # chat state, cache, uploads
    max_history_turns: int = 5
    session_ttl_seconds: int = 3600
    cache_maxsize: int = 256
    cache_ttl_seconds: int = 3600
    snippet_length: int = 220
    max_upload_bytes: int = 10 * 1024 * 1024
    max_upload_files: int = 10
    log_level: str = "INFO"

    @property
    def raw_docs_dir(self) -> Path:
        return self.data_dir / "raw_docs"

    @property
    def vector_store_dir(self) -> Path:
        return self.data_dir / "vector_store"

    @property
    def query_log_path(self) -> Path:
        return self.data_dir / "query_log.jsonl"

    def with_(self, **changes) -> Settings:
        return replace(self, **changes)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e: Mapping[str, str] = os.environ if env is None else env
        s = cls(
            data_dir=Path(e.get("RAG_DATA_DIR", "data")),
            embeddings_provider=e.get("EMBEDDINGS_PROVIDER", "huggingface").strip().lower(),
            llm_provider=e.get("LLM_PROVIDER", "local").strip().lower(),
            device=e.get("RAG_DEVICE", "auto").strip().lower(),
            hf_embedding_model=e.get("HF_EMBEDDING_MODEL", cls.hf_embedding_model),
            local_llm_model=e.get("LOCAL_LLM_MODEL", cls.local_llm_model),
            openai_api_key=e.get("OPENAI_API_KEY", ""),
            openai_base_url=e.get("OPENAI_BASE_URL", cls.openai_base_url).rstrip("/"),
            openai_model=e.get("OPENAI_MODEL", ""),
            openai_embedding_model=e.get("OPENAI_EMBEDDING_MODEL", ""),
            gemini_api_key=e.get("GEMINI_API_KEY", ""),
            gemini_base_url=e.get("GEMINI_BASE_URL", cls.gemini_base_url).rstrip("/"),
            gemini_model=e.get("GEMINI_MODEL", ""),
            gemini_embedding_model=e.get("GEMINI_EMBEDDING_MODEL", ""),
            reranker_model=e.get("RERANKER_MODEL", ""),
            chunk_size=_int(e, "CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int(e, "CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_int(e, "TOP_K", cls.top_k),
            retriever=e.get("RETRIEVER", cls.retriever).strip().lower(),
            min_dense_score=_opt_float(e, "MIN_DENSE_SCORE", cls.min_dense_score),
            max_new_tokens=_int(e, "MAX_NEW_TOKENS", cls.max_new_tokens),
            temperature=_float(e, "TEMPERATURE", cls.temperature),
            few_shot=e.get("FEW_SHOT", "").strip().lower() in {"1", "true", "yes", "on"},
            max_history_turns=_int(e, "MAX_HISTORY_TURNS", cls.max_history_turns),
            session_ttl_seconds=_int(e, "SESSION_TTL_SECONDS", cls.session_ttl_seconds),
            cache_maxsize=_int(e, "CACHE_MAXSIZE", cls.cache_maxsize),
            cache_ttl_seconds=_int(e, "CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            snippet_length=_int(e, "SNIPPET_LENGTH", cls.snippet_length),
            max_upload_bytes=_int(e, "MAX_UPLOAD_BYTES", cls.max_upload_bytes),
            max_upload_files=_int(e, "MAX_UPLOAD_FILES", cls.max_upload_files),
            log_level=e.get("LOG_LEVEL", cls.log_level).upper(),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if self.embeddings_provider not in EMBEDDING_PROVIDERS:
            raise ConfigError(
                f"EMBEDDINGS_PROVIDER must be one of {', '.join(EMBEDDING_PROVIDERS)}, "
                f"got {self.embeddings_provider!r}"
            )
        if self.llm_provider not in LLM_PROVIDERS:
            raise ConfigError(
                f"LLM_PROVIDER must be one of {', '.join(LLM_PROVIDERS)}, got {self.llm_provider!r}"
            )
        if self.retriever not in RETRIEVERS:
            raise ConfigError(
                f"RETRIEVER must be one of {', '.join(RETRIEVERS)}, got {self.retriever!r}"
            )
        if self.chunk_size < 50:
            raise ConfigError("CHUNK_SIZE must be at least 50 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be at least 0 and smaller than CHUNK_SIZE")
        if self.top_k < 1:
            raise ConfigError("TOP_K must be at least 1")

    def require_hosted(self) -> None:
        """Check that the hosted providers that are selected have everything they need."""
        if "openai" in (self.embeddings_provider, self.llm_provider) and not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required when a provider is set to openai")
        if self.llm_provider == "openai" and not self.openai_model:
            raise ConfigError(
                "OPENAI_MODEL must be set (the model id comes from the environment only)"
            )
        if self.embeddings_provider == "openai" and not self.openai_embedding_model:
            raise ConfigError("OPENAI_EMBEDDING_MODEL must be set when EMBEDDINGS_PROVIDER=openai")
        if "gemini" in (self.embeddings_provider, self.llm_provider) and not self.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is required when a provider is set to gemini")
        if self.llm_provider == "gemini" and not self.gemini_model:
            raise ConfigError(
                "GEMINI_MODEL must be set (the model id comes from the environment only)"
            )
        if self.embeddings_provider == "gemini" and not self.gemini_embedding_model:
            raise ConfigError("GEMINI_EMBEDDING_MODEL must be set when EMBEDDINGS_PROVIDER=gemini")


def load_dotenv_if_present(path: str | os.PathLike = ".env") -> None:
    """Load ``.env`` into the process environment if the file exists (existing variables win)."""
    from dotenv import load_dotenv

    load_dotenv(path, override=False)


def pick_device(requested: str = "auto") -> str:
    """``cuda`` when available and requested or automatic, else ``cpu``."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ConfigError("RAG_DEVICE must be auto, cpu or cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # torch missing or broken: stay on the CPU
        pass
    if requested == "cuda":
        raise ConfigError("RAG_DEVICE=cuda but no CUDA device is available")
    return "cpu"
