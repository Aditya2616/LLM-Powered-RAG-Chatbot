"""HTTP API of the Ask-Aditya service: ``POST /api/ask`` (server-sent events) and ``GET /api/health``.

The answer is generated and verified completely before the first byte is sent, then streamed to the client
as ``token`` events followed by ``done``. That costs time to first token (it equals the generation time) but
means nothing unverified is ever shown.
"""

from __future__ import annotations

import hmac
import json
import re
import time
from collections.abc import Iterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import AskSettings
from .index import AskIndex
from .limits import Limiter, make_limiter
from .pipeline import AskPipeline
from .providers import ProviderChain, Unavailable, build_chain
from .safety import BadRequest, validate_request

LIMIT_TEXT = "The assistant is busy right now. Please try again later, or email Aditya directly."


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.headers.get("x-real-ip") or (
        request.client.host if request.client else "unknown"
    )


def create_app(
    settings: AskSettings | None = None,
    index: AskIndex | None = None,
    chain: ProviderChain | None = None,
    limiter: Limiter | None = None,
    transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    cfg = settings or AskSettings.from_env()
    state: dict = {"pipeline": None, "index": index, "t_start": time.time()}
    lim = limiter or make_limiter(cfg, transport=transport)

    def pipeline() -> AskPipeline:
        if state["pipeline"] is None:
            idx = state["index"] or AskIndex.load(cfg.index_dir)
            state["index"] = idx
            state["pipeline"] = AskPipeline(
                idx, chain or build_chain(cfg, lim, transport), cfg.contact_email, cfg.top_k
            )
        return state["pipeline"]

    app = FastAPI(title="Ask Aditya", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.allowed_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["content-type"],
        max_age=600,
    )

    @app.get("/api/health")
    def health() -> dict:
        idx = state["index"]
        if idx is None:
            try:
                idx = state["index"] = AskIndex.load(cfg.index_dir)
            except (OSError, ValueError, KeyError):
                return JSONResponse({"ok": False, "corpus_version": ""}, status_code=503)
        return {"ok": True, "corpus_version": idx.version}

    @app.post("/api/ask")
    async def ask(request: Request):
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            return JSONResponse({"error": "the body must be valid JSON"}, status_code=400)
        try:
            question, history = validate_request(payload)
        except BadRequest as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        hook = request.headers.get("x-ask-test", "") if cfg.test_hooks else ""
        exempt = bool(cfg.eval_token) and hmac.compare_digest(
            request.headers.get("x-ask-eval", ""), cfg.eval_token
        )
        allowed, retry = (
            (True, 0)
            if exempt
            else lim.check_ip(client_ip(request), cfg.ip_per_minute, cfg.ip_per_day)
        )
        if not allowed:
            return JSONResponse(
                {"error": "rate_limited", "retry_after": retry},
                status_code=429,
                headers={"Retry-After": str(retry)},
            )
        t0 = time.perf_counter()
        try:
            result = pipeline().answer(question, history, force_fail=hook == "limit")
        except Unavailable:
            body = {
                "error": "unavailable",
                "answer": LIMIT_TEXT,
                "sources": [],
                "grounded": False,
                "fallback": "limit",
            }
            return JSONResponse(body, status_code=503, headers={"Retry-After": "300"})
        except (OSError, ValueError, KeyError):
            return JSONResponse(
                {
                    "error": "unavailable",
                    "fallback": "limit",
                    "answer": LIMIT_TEXT,
                    "sources": [],
                    "grounded": False,
                },
                status_code=503,
            )
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        def events() -> Iterator[str]:
            for piece in re.findall(r"\S+\s*", result.answer):
                yield _sse("token", {"t": piece})
            yield _sse(
                "done",
                {
                    "answer": result.answer,
                    "sources": result.sources,
                    "grounded": result.grounded,
                    "fallback": result.fallback,
                },
            )

        headers = {
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "X-Ask-Ms": str(total_ms),
        }
        if cfg.test_hooks:
            headers["X-Ask-Stage"] = result.stage
            headers["X-Ask-Reason"] = re.sub(r"[^ -~]", "?", result.reason)[:120]
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    return app
