"""The RAG pipeline: ingest, retrieve, (optionally) gate, generate, cite, cache and log."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .embeddings import get_embeddings
from .llm import LLM, Prompt, get_llm
from .loaders import load_documents, split_documents
from .retrieval import CrossEncoderReranker, Hit, Retrieval, Retriever
from .store import VectorStore
from .text import ABSTAIN_TEXT, is_abstention, snippet, token_support

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a domain-specific assistant. Use ONLY the numbered context passages to answer. "
    "Answer in one or two short sentences. "
    f'If the answer is not in the context, reply with exactly: "{ABSTAIN_TEXT}". '
    "Do not use outside knowledge."
)

# Two worked turns, used when FEW_SHOT is on. They are about a different (invented) topic so they
# demonstrate the format and the refusal without leaking into any real corpus.
FEW_SHOT_TURNS = [
    {
        "role": "user",
        "content": "Context:\n[1] (source: example.txt) The library opens at 8:00 and closes at 20:00 "
        "on weekdays. Members may borrow up to 6 books for 21 days.\n\nQuestion: How many books may a "
        "member borrow?\nAnswer:",
    },
    {"role": "assistant", "content": "A member may borrow up to 6 books."},
    {
        "role": "user",
        "content": "Context:\n[1] (source: example.txt) The library opens at 8:00 and closes at 20:00 "
        "on weekdays. Members may borrow up to 6 books for 21 days.\n\nQuestion: Is there a cafe in "
        "the library?\nAnswer:",
    },
    {"role": "assistant", "content": ABSTAIN_TEXT},
]


class QueryCache:
    """In-memory LRU with a time-to-live."""

    def __init__(self, maxsize: int, ttl_seconds: int) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[tuple, tuple[dict, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> dict | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, ts = item
            if time.time() - ts > self.ttl_seconds:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: tuple, value: dict) -> None:
        with self._lock:
            self._store[key] = (value, time.time())
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class QueryLog:
    """Append-only JSON lines: one record per question."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        if self.path is None:
            return
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")


class RAGPipeline:
    def __init__(
        self, settings: Settings, store: VectorStore, retriever: Retriever, llm: LLM
    ) -> None:
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.llm = llm
        self.cache = QueryCache(settings.cache_maxsize, settings.cache_ttl_seconds)
        self.log = QueryLog(settings.query_log_path)
        self.history: dict[str, list[tuple[str, str]]] = {}
        self.session_seen: dict[str, float] = {}
        self._gen_lock = (
            threading.Lock()
        )  # one generation at a time; local models are not re-entrant

    # ---- ingest ------------------------------------------------------------------------------
    def ingest(self, paths: Iterable[Path]) -> int:
        """Load, chunk, embed and persist files. Re-ingesting a file name replaces its chunks."""
        docs = load_documents(list(paths))
        chunks = split_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        added = self.store.add(chunks)
        if added:
            self.store.save()
            self.cache.clear()
        return added

    def reindex_raw_docs(self) -> int:
        """Rebuild the index from ``data/raw_docs`` (for example after changing the embedder)."""
        self.store.clear()
        files = sorted(p for p in self.settings.raw_docs_dir.glob("*") if p.is_file())
        added = self.ingest(files) if files else 0
        if not files:
            self.store.save()
        return added

    # ---- prompt ------------------------------------------------------------------------------
    @staticmethod
    def _context(hits: list[Hit]) -> str:
        return "\n\n".join(
            f"[{i}] (source: {h.chunk.metadata.get('source')}) {h.chunk.page_content}"
            for i, h in enumerate(hits, 1)
        )

    def build_prompt(self, question: str, hits: list[Hit], history_text: str = "") -> Prompt:
        user = f"Context:\n{self._context(hits)}\n\n"
        if history_text:
            user += f"Conversation history:\n{history_text}\n\n"
        user += f"Question: {question}\nAnswer:"
        return Prompt(
            system=SYSTEM_PROMPT,
            user=user,
            question=question,
            passages=[h.chunk.page_content for h in hits],
            messages=FEW_SHOT_TURNS if self.settings.few_shot else None,
        )

    # ---- history -----------------------------------------------------------------------------
    def _prune_sessions(self) -> None:
        now = time.time()
        for sid in [
            s for s, t in self.session_seen.items() if now - t > self.settings.session_ttl_seconds
        ]:
            self.history.pop(sid, None)
            self.session_seen.pop(sid, None)

    def _history_text(self, session_id: str | None, use_history: bool) -> str:
        if not (session_id and use_history):
            return ""
        self._prune_sessions()
        return "\n\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in self.history.get(session_id, [])
        )

    def _remember(self, session_id: str | None, use_history: bool, q: str, a: str) -> None:
        if not (session_id and use_history):
            return
        turns = [*self.history.get(session_id, []), (q, a)][-self.settings.max_history_turns :]
        self.history[session_id] = turns
        self.session_seen[session_id] = time.time()

    # ---- citations ---------------------------------------------------------------------------
    def source_dicts(self, hits: list[Hit], answer: str | None = None) -> list[dict[str, Any]]:
        out = []
        for h in hits:
            md = h.chunk.metadata
            d: dict[str, Any] = {
                "rank": h.rank,
                "source": md.get("source"),
                "chunk": md.get("chunk"),
                "chunk_id": md.get("chunk_id"),
                "page": md.get("page"),
                "start": md.get("start"),
                "end": md.get("end"),
                "score": round(h.score, 4),
                "dense": round(h.dense, 4),
                "bm25": round(h.bm25, 4),
                "snippet": snippet(h.chunk.page_content, self.settings.snippet_length),
                "text": h.chunk.page_content,
            }
            if answer is not None:
                d["support"] = round(token_support(answer, h.chunk.page_content), 3)
            out.append(d)
        return out

    # ---- the question path -------------------------------------------------------------------
    def _gate(self, hits: list[Hit]) -> bool:
        thr = self.settings.min_dense_score
        return thr is not None and bool(hits) and max(h.dense for h in hits) < thr

    def _retrieve(self, question: str, k: int, mode: str | None) -> Retrieval:
        with self.store.lock:
            return self.retriever.retrieve(question, k, mode)

    def _finish(self, base: dict, question: str, session_id, use_history, t0: float) -> dict:
        base["timings"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self._remember(session_id, use_history, question, base["answer"])
        self.log.write(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "question": question,
                "answer": base["answer"],
                "abstained": base["abstained"],
                "abstain_reason": base["abstain_reason"],
                "cached": base["cached"],
                "retriever": base["retriever"],
                "sources": [s["chunk_id"] for s in base["sources"]],
                "total_ms": base["timings"]["total_ms"],
            }
        )
        logger.info("Q: %s | abstained=%s cached=%s", question, base["abstained"], base["cached"])
        return base

    def _payload(
        self, answer, abstain_reason, retrieval, hits, retrieve_ms, generate_ms, cached
    ) -> dict:
        abstained = abstain_reason is not None
        return {
            "answer": ABSTAIN_TEXT if abstained else answer,
            "abstained": abstained,
            "abstain_reason": abstain_reason,
            "sources": self.source_dicts(hits, None if abstained else answer),
            "cached": cached,
            "retriever": retrieval.mode,
            "model": {"embeddings": self.store.fingerprint, "llm": self.llm.name},
            "timings": {"retrieve_ms": round(retrieve_ms, 1), "generate_ms": round(generate_ms, 1)},
        }

    def stream(
        self,
        question: str,
        top_k: int | None = None,
        session_id: str | None = None,
        use_history: bool = True,
        retriever: str | None = None,
    ) -> Iterator[dict]:
        """Yield ``retrieval``, then ``token`` events, then ``done`` (the same payload as ``ask``)."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        t0 = time.perf_counter()
        k = top_k or self.settings.top_k
        history_text = self._history_text(session_id, use_history)
        mode = retriever or self.settings.retriever
        key = (self.store.version, question, k, mode, history_text)

        cached = self.cache.get(key)
        if cached is not None:
            payload = json.loads(json.dumps(cached))
            payload["cached"] = True
            yield {
                "event": "retrieval",
                "sources": payload["sources"],
                "retrieve_ms": 0.0,
                "cached": True,
            }
            if not payload["abstained"]:
                yield {"event": "token", "text": payload["answer"]}
            yield {"event": "done", **self._finish(payload, question, session_id, use_history, t0)}
            return

        retrieval = self._retrieve(question, k, mode)
        hits = retrieval.hits
        xy = None
        if retrieval.query_vec is not None and len(self.store) >= 2:
            xy = [round(float(v), 4) for v in self.store.project(retrieval.query_vec)[0]]
        yield {
            "event": "retrieval",
            "sources": self.source_dicts(hits),
            "retrieve_ms": round(retrieval.ms, 1),
            "query_xy": xy,
            "cached": False,
        }

        if not hits:
            reason = "empty_index"
            answer, gen_ms = ABSTAIN_TEXT, 0.0
        elif self._gate(hits):
            reason = "low_retrieval_score"
            answer, gen_ms = ABSTAIN_TEXT, 0.0
        else:
            reason = None
            prompt = self.build_prompt(question, hits, history_text)
            g0 = time.perf_counter()
            parts: list[str] = []
            with self._gen_lock:
                for piece in self.llm.stream(
                    prompt, self.settings.max_new_tokens, self.settings.temperature
                ):
                    parts.append(piece)
                    yield {"event": "token", "text": piece}
            gen_ms = (time.perf_counter() - g0) * 1000
            answer = "".join(parts).strip()
            if is_abstention(answer) or not answer:
                reason = "model_declined"
        payload = self._payload(answer, reason, retrieval, hits, retrieval.ms, gen_ms, False)
        payload["timings"]["retrieve_ms"] = round(retrieval.ms, 1)
        self.cache.set(key, payload)
        yield {
            "event": "done",
            **self._finish(json.loads(json.dumps(payload)), question, session_id, use_history, t0),
        }

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        session_id: str | None = None,
        use_history: bool = True,
        retriever: str | None = None,
    ) -> dict:
        """Answer a question. Returns the payload described in the README (answer, sources, timings)."""
        final: dict | None = None
        for ev in self.stream(question, top_k, session_id, use_history, retriever):
            if ev["event"] == "done":
                final = {k: v for k, v in ev.items() if k != "event"}
        assert final is not None
        return final


def build_pipeline(
    settings: Settings, transport: httpx.BaseTransport | None = None, load_index: bool = True
) -> RAGPipeline:
    """Wire the configured providers together and load the saved index if there is one."""
    embeddings = get_embeddings(settings, transport)
    store = VectorStore(
        embeddings,
        getattr(embeddings, "fingerprint", settings.embeddings_provider),
        settings.vector_store_dir,
    )
    if load_index:
        store.load()
    reranker = (
        CrossEncoderReranker(settings.reranker_model, settings.device)
        if settings.reranker_model
        else None
    )
    return RAGPipeline(
        settings,
        store,
        Retriever(store, settings.retriever, reranker),
        get_llm(settings, transport),
    )
