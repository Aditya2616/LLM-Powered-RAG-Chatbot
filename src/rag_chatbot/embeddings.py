"""Embedding providers behind LangChain's ``Embeddings`` interface.

``huggingface`` (default, local sentence-transformers), ``openai`` (any OpenAI-compatible
``/embeddings`` endpoint), ``gemini`` (the Gemini REST API) and ``hashing`` (a dependency-free
feature-hashing bag of words, used by the offline tests and the zero-download quickstart).
Every provider returns L2-normalised vectors, so an inner product is a cosine similarity.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import httpx
import numpy as np
from langchain_core.embeddings import Embeddings

from .config import ConfigError, Settings, pick_device
from .text import content_tokens


class ProviderError(RuntimeError):
    """A hosted provider returned an error or an unusable response."""


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict,
    headers: dict[str, str],
    retries: int = 2,
    backoff: float = 0.5,
) -> dict:
    """POST JSON, retrying 429 and 5xx responses a couple of times, and return the parsed body."""
    last = ""
    for attempt in range(retries + 1):
        try:
            resp = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code < 400:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise ProviderError("provider returned a body that is not JSON") from exc
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
        if attempt < retries:
            time.sleep(backoff * (2**attempt))
    raise ProviderError(f"request to {url.split('?')[0]} failed ({last})")


class HashingEmbeddings(Embeddings):
    """Feature-hashed bag of content words. Lexical, deterministic, no model, no network."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.fingerprint = f"hashing:{dim}"

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in content_tokens(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return l2_normalise(np.stack([self._vector(t) for t in texts])).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbeddings(Embeddings):
    """Local sentence-transformers model, loaded on first use."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 64) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.fingerprint = f"huggingface:{model_name}"
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ConfigError(
                    "the huggingface embedder needs the local extra: pip install -e '.[local]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=pick_device(self.device))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out = self._load().encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return l2_normalise(out).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbeddings(Embeddings):
    """``POST {base_url}/embeddings``. Works with any server that speaks the same protocol."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        transport: httpx.BaseTransport | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.fingerprint = f"openai:{model}"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(transport=transport, timeout=60.0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            body = post_json(
                self._client,
                f"{self.base_url}/embeddings",
                {"model": self.model, "input": batch},
                self._headers,
            )
            try:
                rows = sorted(body["data"], key=lambda r: r["index"])
                out.extend(r["embedding"] for r in rows)
            except (KeyError, TypeError) as exc:
                raise ProviderError("unexpected response shape from /embeddings") from exc
        if len(out) != len(texts):
            raise ProviderError("provider returned a different number of embeddings than inputs")
        return l2_normalise(np.array(out)).tolist() if out else []

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class GeminiEmbeddings(Embeddings):
    """``models/{model}:batchEmbedContents`` on the Gemini REST API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        transport: httpx.BaseTransport | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.fingerprint = f"gemini:{self.model}"
        self._headers = {"x-goog-api-key": api_key}
        self._client = httpx.Client(transport=transport, timeout=60.0)

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            reqs = [
                {"model": self.model, "taskType": task, "content": {"parts": [{"text": t}]}}
                for t in batch
            ]
            body = post_json(
                self._client,
                f"{self.base_url}/{self.model}:batchEmbedContents",
                {"requests": reqs},
                self._headers,
            )
            try:
                out.extend(e["values"] for e in body["embeddings"])
            except (KeyError, TypeError) as exc:
                raise ProviderError("unexpected response shape from batchEmbedContents") from exc
        if len(out) != len(texts):
            raise ProviderError("provider returned a different number of embeddings than inputs")
        return l2_normalise(np.array(out)).tolist() if out else []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def get_embeddings(settings: Settings, transport: httpx.BaseTransport | None = None) -> Embeddings:
    """Build the configured embedder. ``transport`` lets tests swap in a stubbed HTTP transport."""
    provider = settings.embeddings_provider
    if provider == "hashing":
        return HashingEmbeddings()
    if provider == "huggingface":
        return SentenceTransformerEmbeddings(settings.hf_embedding_model, settings.device)
    settings.require_hosted()
    if provider == "openai":
        return OpenAIEmbeddings(
            settings.openai_api_key,
            settings.openai_embedding_model,
            settings.openai_base_url,
            transport,
        )
    return GeminiEmbeddings(
        settings.gemini_api_key,
        settings.gemini_embedding_model,
        settings.gemini_base_url,
        transport,
    )
