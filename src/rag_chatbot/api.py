"""FastAPI service: /query, /query/stream, /upload, /health, plus document management and the web UI."""

from __future__ import annotations

import functools
import json
import logging
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import ConfigError, Settings
from .embeddings import ProviderError
from .loaders import UploadRejected, validate_upload
from .pipeline import RAGPipeline, build_pipeline
from .store import IndexMismatch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_DIR = REPO_ROOT / "docs"
DEFAULT_SAMPLE_DIR = REPO_ROOT / "corpus" / "docs"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = Field(default=None, max_length=100)
    use_history: bool = True
    return_sources: bool = True
    retriever: Literal["dense", "bm25", "hybrid"] | None = None


class Source(BaseModel):
    rank: int
    source: str | None
    chunk: int | None
    chunk_id: str | None
    page: int | None = None
    start: int | None
    end: int | None
    score: float
    dense: float
    bm25: float
    snippet: str
    text: str
    support: float | None = None


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: Literal["model_declined", "low_retrieval_score", "empty_index"] | None
    sources: list[Source]
    cached: bool
    retriever: str
    model: dict[str, str]
    timings: dict[str, float]


class Rejected(BaseModel):
    file: str
    reason: str


class UploadResponse(BaseModel):
    message: str
    files: list[str]
    rejected: list[Rejected]
    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_ready: bool
    documents: int
    chunks: int
    embeddings: str
    llm: str
    retriever: str
    index_error: str | None = None


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    characters: int


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    settings: Settings | None = None,
    pipeline: RAGPipeline | None = None,
    web_dir: Path | None = None,
    sample_dir: Path | None = None,
) -> FastAPI:
    """Build the app. Models are loaded lazily on first use, so start-up is fast and ``/health``
    works before any model has been downloaded."""
    cfg = settings or Settings.from_env()
    state: dict = {"pipeline": pipeline, "index_error": None}
    web = web_dir if web_dir is not None else Path(os.environ.get("RAG_WEB_DIR", DEFAULT_WEB_DIR))
    sample = (
        sample_dir
        if sample_dir is not None
        else Path(os.environ.get("RAG_SAMPLE_DIR", DEFAULT_SAMPLE_DIR))
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logging.basicConfig(
            level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
        )
        if state["pipeline"] is None:
            try:
                state["pipeline"] = build_pipeline(cfg)
            except IndexMismatch as exc:
                state["index_error"] = str(exc)
                state["pipeline"] = build_pipeline(cfg, load_index=False)
        yield

    app = FastAPI(title="RAG Chatbot API", version=__version__, lifespan=lifespan)

    def pipe() -> RAGPipeline:
        p = state["pipeline"]
        if p is None:  # app used without the lifespan (e.g. a bare TestClient)
            p = state["pipeline"] = build_pipeline(cfg)
        if state["index_error"]:
            raise HTTPException(status_code=409, detail=state["index_error"])
        return p

    def provider_errors(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ConfigError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ProviderError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        return wrapper

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        p = state["pipeline"]
        if p is None:
            p = state["pipeline"] = build_pipeline(cfg, load_index=False)
        st = p.store
        return HealthResponse(
            status="ok" if not state["index_error"] else "degraded",
            version=__version__,
            vector_store_ready=st.ready,
            documents=len(st.sources()),
            chunks=len(st),
            embeddings=st.fingerprint,
            llm=p.llm.name,
            retriever=p.settings.retriever,
            index_error=state["index_error"],
        )

    @app.post("/query", response_model=QueryResponse)
    @provider_errors
    def query(payload: QueryRequest) -> QueryResponse:
        result = pipe().ask(
            payload.question,
            payload.top_k,
            payload.session_id,
            payload.use_history,
            payload.retriever,
        )
        if not payload.return_sources:
            result["sources"] = []
        return QueryResponse(**result)

    @app.post("/query/stream")
    def query_stream(payload: QueryRequest) -> StreamingResponse:
        p = pipe()
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        def events() -> Iterator[str]:
            try:
                for ev in p.stream(
                    payload.question,
                    payload.top_k,
                    payload.session_id,
                    payload.use_history,
                    payload.retriever,
                ):
                    name = ev.pop("event")
                    if name == "done" and not payload.return_sources:
                        ev["sources"] = []
                    yield _sse(name, ev)
            except (ProviderError, ConfigError, ValueError) as exc:
                yield _sse("error", {"detail": str(exc)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/upload", response_model=UploadResponse)
    async def upload(files: Annotated[list[UploadFile], File()]) -> UploadResponse:
        p = pipe()
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        if len(files) > cfg.max_upload_files:
            raise HTTPException(
                status_code=413, detail=f"At most {cfg.max_upload_files} files per upload"
            )
        accepted: list[tuple[str, bytes]] = []
        rejected: list[Rejected] = []
        for f in files:
            raw = await f.read(cfg.max_upload_bytes + 1)
            try:
                accepted.append((validate_upload(f.filename or "", raw, cfg.max_upload_bytes), raw))
            except UploadRejected as exc:
                rejected.append(Rejected(file=(f.filename or "(unnamed)")[:120], reason=str(exc)))
        if not accepted:
            status = 413 if any("larger" in r.reason for r in rejected) else 400
            raise HTTPException(status_code=status, detail=[r.model_dump() for r in rejected])

        def work() -> tuple[int, list[str]]:
            cfg.raw_docs_dir.mkdir(parents=True, exist_ok=True)
            paths: list[Path] = []
            for name, data in accepted:
                dest = cfg.raw_docs_dir / name
                dest.write_bytes(data)
                paths.append(dest)
            try:
                return p.ingest(paths), [x.name for x in paths]
            except Exception:
                for x in paths:  # do not leave an unreadable file behind
                    x.unlink(missing_ok=True)
                raise

        try:
            added, names = await run_in_threadpool(work)
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ConfigError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("ingest failed")
            raise HTTPException(
                status_code=422, detail=f"Could not read the document: {exc}"
            ) from exc
        if added == 0:
            raise HTTPException(
                status_code=422, detail="No text could be extracted from the upload"
            )
        return UploadResponse(
            message="Documents ingested",
            files=names,
            rejected=rejected,
            chunks_added=added,
            total_chunks=len(p.store),
        )

    @app.get("/documents", response_model=list[DocumentInfo])
    def documents() -> list[DocumentInfo]:
        return [
            DocumentInfo(source=s.source, chunks=s.chunks, characters=s.characters)
            for s in pipe().store.sources()
        ]

    @app.delete("/documents/{name}")
    def delete_document(name: str) -> dict:
        p = pipe()
        removed = p.store.remove_source(name)
        if not removed:
            raise HTTPException(status_code=404, detail="No such document")
        (cfg.raw_docs_dir / Path(name).name).unlink(missing_ok=True)
        p.store.save()
        p.cache.clear()
        return {"removed_chunks": removed}

    @app.post("/documents/sample", response_model=UploadResponse)
    @provider_errors
    def load_sample() -> UploadResponse:
        p = pipe()
        files = sorted(sample.glob("*.txt")) if sample.is_dir() else []
        if not files:
            raise HTTPException(status_code=404, detail="No bundled sample corpus found")
        added = p.ingest(files)
        return UploadResponse(
            message="Sample corpus ingested",
            files=[f.name for f in files],
            rejected=[],
            chunks_added=added,
            total_chunks=len(p.store),
        )

    @app.get("/index/map")
    def index_map() -> dict:
        st = pipe().store
        if len(st) < 2:
            return {"chunks": [], "version": st.version}
        xy = st.project(st.matrix)
        return {
            "version": st.version,
            "chunks": [
                {
                    "chunk_id": c.metadata.get("chunk_id"),
                    "source": c.metadata.get("source"),
                    "start": c.metadata.get("start"),
                    "end": c.metadata.get("end"),
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                }
                for c, (x, y) in zip(st.chunks, xy, strict=True)
            ],
        }

    if web.is_dir():  # registered last so that it never shadows an API route
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


app = create_app()
