"""Okapi BM25 and reciprocal rank fusion, with no heavy imports (NumPy only).

Used by the hybrid retriever in ``retrieval.py`` and, unchanged, by the light "Ask Aditya" service,
which has to run in a serverless function without FAISS, LangChain or PyTorch.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from .text import STOPWORDS, stem, tokens

RRF_K = 60


def bm25_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75) over a list of texts. Small, exact, dependency-free."""

    def __init__(
        self,
        texts: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = bm25_tokens,
    ) -> None:
        self.k1, self.b = k1, b
        self.tokenize = tokenizer
        self.docs = [Counter(self.tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg = float(self.lengths.mean()) if len(self.docs) else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        # inverted index: term -> [(doc, tf)], so a query touches only the documents that contain it
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(self.docs):
            for term, tf in d.items():
                self.postings.setdefault(term, []).append((i, tf))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        if not self.docs:
            return out
        for term in set(self.tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg or 1.0))
                out[i] += idf * tf * (self.k1 + 1) / norm
        return out


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: each ranking contributes 1 / (k + rank) to every item it lists."""
    fused: dict[int, float] = {}
    for ranks in rankings:
        for r, i in enumerate(ranks):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])
