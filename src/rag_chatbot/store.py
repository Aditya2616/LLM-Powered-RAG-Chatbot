"""The vector store: LangChain's FAISS wrapper over an exact inner-product index, with persistence.

Vectors are L2-normalised by the embedders, so the inner product is cosine similarity. The index is
exact (``IndexFlatIP``), which is the right choice at this scale and keeps results reproducible.

On disk (``data/vector_store/``): ``index.faiss`` (written with ``faiss.write_index``), ``chunks.jsonl``
(text and metadata in index order) and ``meta.json`` (embedder fingerprint, dimension, chunk count).
Nothing is unpickled, so loading a store never executes code from the files. A store built with one
embedder is refused when another is configured, because the vectors would not be comparable.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class IndexMismatch(RuntimeError):
    """The stored index was built with a different embedder than the one configured."""


@dataclass(frozen=True)
class SourceInfo:
    source: str
    chunks: int
    characters: int


class VectorStore:
    def __init__(
        self, embeddings: Embeddings, fingerprint: str, directory: Path | None = None
    ) -> None:
        self.embeddings = embeddings
        self.fingerprint = fingerprint
        self.directory = directory
        self.chunks: list[Document] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.version = 0
        self._vs: FAISS | None = None
        self._pca: tuple[np.ndarray, np.ndarray] | None = None
        self.lock = threading.RLock()

    # ---- state -------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def ready(self) -> bool:
        return len(self.chunks) > 0

    @property
    def vs(self) -> FAISS | None:
        return self._vs

    def sources(self) -> list[SourceInfo]:
        info: dict[str, list[int]] = {}
        for c in self.chunks:
            s = str(c.metadata.get("source", "unknown"))
            entry = info.setdefault(s, [0, 0])
            entry[0] += 1
            entry[1] += len(c.page_content)
        return [SourceInfo(s, n, ch) for s, (n, ch) in sorted(info.items())]

    # ---- building ----------------------------------------------------------------------------
    def _rebuild(self) -> None:
        self._pca = None
        if not self.chunks:
            self._vs = None
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            return
        index = faiss.IndexFlatIP(self.matrix.shape[1])
        index.add(self.matrix)
        docstore = InMemoryDocstore({str(i): c for i, c in enumerate(self.chunks)})
        self._vs = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id={i: str(i) for i in range(len(self.chunks))},
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        self.version += 1

    def add(self, chunks: list[Document], replace_sources: bool = True) -> int:
        """Embed and add chunks. Chunks from a source that is already stored replace the old ones."""
        if not chunks:
            return 0
        vectors = np.asarray(
            self.embeddings.embed_documents([c.page_content for c in chunks]), dtype=np.float32
        )
        with self.lock:
            if replace_sources:
                names = {str(c.metadata.get("source")) for c in chunks}
                keep = [
                    i
                    for i, c in enumerate(self.chunks)
                    if str(c.metadata.get("source")) not in names
                ]
                self.chunks = [self.chunks[i] for i in keep]
                self.matrix = (
                    self.matrix[keep] if len(keep) else np.zeros((0, vectors.shape[1]), np.float32)
                )
            if self.matrix.size and self.matrix.shape[1] != vectors.shape[1]:
                raise IndexMismatch("embedding dimension differs from the stored index")
            self.matrix = np.vstack([self.matrix, vectors]) if self.matrix.size else vectors
            self.chunks = [*self.chunks, *chunks]
            self._rebuild()
        return len(chunks)

    def remove_source(self, source: str) -> int:
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c.metadata.get("source") != source]
            removed = len(self.chunks) - len(keep)
            if removed:
                self.chunks = [self.chunks[i] for i in keep]
                self.matrix = self.matrix[keep] if keep else np.zeros((0, 0), np.float32)
                self._rebuild()
        return removed

    def clear(self) -> None:
        with self.lock:
            self.chunks = []
            self._rebuild()

    # ---- searching ---------------------------------------------------------------------------
    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.embeddings.embed_query(text), dtype=np.float32)

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k (chunk index, cosine) through the LangChain FAISS wrapper."""
        if self._vs is None:
            return []
        found = self._vs.similarity_search_with_score_by_vector(
            query_vec.tolist(), k=min(k, len(self))
        )
        ids = {id(c): i for i, c in enumerate(self.chunks)}
        return [(ids[id(doc)], float(score)) for doc, score in found]

    def cosine_all(self, query_vec: np.ndarray) -> np.ndarray:
        return self.matrix @ query_vec if self.ready else np.zeros(0, dtype=np.float32)

    # ---- projection for the interface ---------------------------------------------------------
    def _basis(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pca is None:
            mean = self.matrix.mean(axis=0)
            _, _, vt = np.linalg.svd(self.matrix - mean, full_matrices=False)
            self._pca = (mean, vt[:2])
        return self._pca

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """2-D principal-component coordinates for chunk or query vectors (for display only)."""
        mean, comps = self._basis()
        return (np.atleast_2d(vectors) - mean) @ comps.T

    # ---- persistence -------------------------------------------------------------------------
    def save(self) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if not self.chunks:
                for name in ("index.faiss", "chunks.jsonl", "meta.json"):
                    (self.directory / name).unlink(missing_ok=True)
                return
            faiss.write_index(self._vs.index, str(self.directory / "index.faiss"))
            with (self.directory / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                for c in self.chunks:
                    f.write(json.dumps({"text": c.page_content, "metadata": c.metadata}) + "\n")
            meta = {
                "fingerprint": self.fingerprint,
                "dim": int(self.matrix.shape[1]),
                "chunks": len(self),
            }
            (self.directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self) -> bool:
        """Load a saved store. Returns False when there is none; raises ``IndexMismatch`` if the
        stored embedder differs from the configured one."""
        if self.directory is None:
            return False
        files = [self.directory / n for n in ("index.faiss", "chunks.jsonl", "meta.json")]
        if not all(p.is_file() and not p.is_symlink() for p in files):
            return False
        meta = json.loads(files[2].read_text(encoding="utf-8"))
        if meta.get("fingerprint") != self.fingerprint:
            raise IndexMismatch(
                f"the stored index was built with {meta.get('fingerprint')!r} but {self.fingerprint!r} "
                "is configured; re-ingest the documents (rag-chatbot reindex)"
            )
        index = faiss.read_index(str(files[0]))
        rows = [
            json.loads(line) for line in files[1].read_text(encoding="utf-8").splitlines() if line
        ]
        if index.ntotal != len(rows):
            raise IndexMismatch("the stored index and chunk list disagree; re-ingest the documents")
        with self.lock:
            self.chunks = [Document(page_content=r["text"], metadata=r["metadata"]) for r in rows]
            self.matrix = index.reconstruct_n(0, index.ntotal).astype(np.float32)
            self._rebuild()
        logger.info("Loaded %d chunks from %s", len(self), self.directory)
        return True
