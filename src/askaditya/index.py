"""The runtime index: chunks plus BM25, loaded from ``index.json``. NumPy only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_chatbot.bm25 import BM25Index, rrf_fuse

from .build import DEFAULT_PLATFORM_NAME  # noqa: F401  (re-exported for callers that build indexes)


@dataclass
class Hit:
    chunk: dict
    score: float
    rank: int


class AskIndex:
    def __init__(
        self,
        chunks: list[dict],
        version: str,
        traps: list[dict] | None = None,
        dense: object | None = None,
    ) -> None:
        self.chunks = chunks
        self.version = version
        self.traps = traps or []
        self._bm25 = BM25Index([f"{c['title']} {c['section']} {c['text']}" for c in chunks])
        self.dense = dense  # optional (n, d) array of unit vectors, only used offline

    @classmethod
    def load(cls, directory: Path) -> AskIndex:
        data = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        dense = None
        npy = directory / "dense.npy"
        if npy.is_file():  # present only after an offline evaluation build; never needed to serve
            import numpy as np

            dense = np.load(npy)
        return cls(data["chunks"], data["version"], data.get("traps", []), dense)

    def search(self, query: str, k: int = 5, query_vec: object | None = None) -> list[Hit]:
        """BM25, or BM25 fused with dense cosine by reciprocal rank when a query vector is given."""
        scores = self._bm25.scores(query)
        order = [int(i) for i in scores.argsort()[::-1] if scores[i] > 0][: max(20, 4 * k)]
        if query_vec is not None and self.dense is not None:
            cos = self.dense @ query_vec
            dense_order = [int(i) for i in cos.argsort()[::-1][: max(20, 4 * k)]]
            fused = rrf_fuse([dense_order, order])[:k]
            return [Hit(self.chunks[i], float(s), r + 1) for r, (i, s) in enumerate(fused)]
        return [Hit(self.chunks[i], float(scores[i]), r + 1) for r, i in enumerate(order[:k])]
