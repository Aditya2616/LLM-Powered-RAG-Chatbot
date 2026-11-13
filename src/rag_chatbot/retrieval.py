"""Retrievers: dense (FAISS), BM25, and a hybrid that fuses the two with reciprocal rank fusion.

Every hit carries the dense cosine, the BM25 score and the final score the ranking used, so the
interface and the abstention gate can show or use them regardless of which retriever ran.
An optional cross-encoder reranker can be configured; whether it helps is measured, not assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langchain_core.documents import Document

from .bm25 import BM25Index, rrf_fuse
from .config import ConfigError, pick_device
from .store import VectorStore


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
            self._bm25 = BM25Index([c.page_content for c in self.store.chunks])
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
            order = rrf_fuse([dense_rank, bm25_rank])
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
