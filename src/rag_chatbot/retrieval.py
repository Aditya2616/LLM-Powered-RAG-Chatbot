"""Retrievers: dense (FAISS), BM25, and a hybrid that fuses the two with reciprocal rank fusion.

Every hit carries the dense cosine, the BM25 score and the final score the ranking used, so the
interface and the abstention gate can show or use them regardless of which retriever ran.
An optional cross-encoder reranker can be configured; whether it helps is measured, not assumed.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langchain_core.documents import Document

from .config import ConfigError, pick_device
from .store import VectorStore
from .text import STOPWORDS, stem, tokens

RRF_K = 60


def bm25_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75) over a list of chunks. Small, exact, dependency-free."""

    def __init__(self, chunks: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(bm25_tokens(c.page_content)) for c in chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg = float(self.lengths.mean()) if len(self.docs) else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        if not self.docs:
            return out
        for term in set(bm25_tokens(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (
                        1 - self.b + self.b * self.lengths[i] / (self.avg or 1.0)
                    )
                    out[i] += idf * tf * (self.k1 + 1) / norm
        return out


@dataclass
class Hit:
    index: int
    chunk: Document
    score: float  # the value the ranking used (cosine, BM25, RRF or reranker logit)
    dense: float  # cosine similarity to the query
    bm25: float
    rank: int = 0

    @property
    def chunk_id(self) -> str:
        return str(self.chunk.metadata.get("chunk_id", self.index))


@dataclass
class Retrieval:
    hits: list[Hit]
    query_vec: np.ndarray | None
    ms: float
    mode: str
    extra: dict[str, Any] = field(default_factory=dict)


class CrossEncoderReranker:
    """Optional second stage: scores (query, chunk) pairs jointly with a cross-encoder."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover
                raise ConfigError("the reranker needs: pip install -e '.[local]'") from exc
            self._model = CrossEncoder(self.model_name, device=pick_device(self.device))
        return [
            float(s)
            for s in self._model.predict([(query, t) for t in texts], show_progress_bar=False)
        ]


class Retriever:
    def __init__(
        self, store: VectorStore, mode: str = "hybrid", reranker: CrossEncoderReranker | None = None
    ) -> None:
        self.store = store
        self.mode = mode
        self.reranker = reranker
        self._bm25: BM25Index | None = None
        self._bm25_version = -1

    def _index(self) -> BM25Index:
        if self._bm25 is None or self._bm25_version != self.store.version:
            self._bm25 = BM25Index(self.store.chunks)
            self._bm25_version = self.store.version
        return self._bm25

    def retrieve(self, query: str, k: int, mode: str | None = None) -> Retrieval:
        mode = mode or self.mode
        t0 = time.perf_counter()
        store = self.store
        if not store.ready:
            return Retrieval([], None, 0.0, mode)
        n = len(store)
        qv = store.embed_query(query)
        cosine = store.cosine_all(qv)
        bm25 = self._index().scores(query)
        pool = min(n, max(20, 5 * k)) if (mode == "hybrid" or self.reranker) else min(n, k)

        dense_rank = [i for i, _ in store.dense_search(qv, pool)]
        bm25_rank = [int(i) for i in np.argsort(-bm25, kind="stable")[:pool] if bm25[i] > 0]
        if mode == "dense":
            order = [(i, float(cosine[i])) for i in dense_rank]
        elif mode == "bm25":
            order = [(i, float(bm25[i])) for i in bm25_rank]
        else:
            fused: dict[int, float] = {}
            for ranks in (dense_rank, bm25_rank):
                for r, i in enumerate(ranks):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
            order = sorted(fused.items(), key=lambda kv: -kv[1])
        if self.reranker and order:
            cand = order[: max(pool, k)]
            scores = self.reranker.score(query, [store.chunks[i].page_content for i, _ in cand])
            order = sorted(
                ((i, s) for (i, _), s in zip(cand, scores, strict=True)), key=lambda kv: -kv[1]
            )
        hits = [
            Hit(i, store.chunks[i], score, float(cosine[i]), float(bm25[i]), rank=r + 1)
            for r, (i, score) in enumerate(order[:k])
        ]
        return Retrieval(hits, qv, (time.perf_counter() - t0) * 1000, mode)
