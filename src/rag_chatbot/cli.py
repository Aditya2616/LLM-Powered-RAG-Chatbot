"""Command line: ``rag-chatbot ingest | ask | reindex | info | serve``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, Settings, load_dotenv_if_present
from .embeddings import ProviderError
from .loaders import SUPPORTED_EXTENSIONS
from .pipeline import RAGPipeline, build_pipeline
from .store import IndexMismatch


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"no such file or folder: {raw}")
    return out


def _pipeline() -> RAGPipeline:
    load_dotenv_if_present()
    return build_pipeline(Settings.from_env())


def cmd_ingest(args: argparse.Namespace) -> int:
    pipe = _pipeline()
    files = _files(args.paths)
    if not files:
        print("nothing to ingest (supported: .pdf, .txt, .docx)", file=sys.stderr)
        return 1
    added = pipe.ingest(files)
    print(f"ingested {len(files)} files -> {added} chunks (index now holds {len(pipe.store)})")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipe = _pipeline()
    if not pipe.store.ready:
        print("the index is empty; run `rag-chatbot ingest corpus/docs` first", file=sys.stderr)
        return 1
    if args.stream and not args.json:
        final = None
        for ev in pipe.stream(args.question, args.top_k):
            if ev["event"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["event"] == "done":
                final = ev
        print()
    else:
        final = pipe.ask(args.question, args.top_k)
        if not args.json:
            print(final["answer"])
    if args.json:
        print(
            json.dumps(
                {k: v for k, v in final.items() if k != "event"}, indent=2, ensure_ascii=False
            )
        )
    elif final:
        if final["abstained"]:
            print(f"  (declined: {final['abstain_reason']})")
        for s in final["sources"]:
            print(
                f"  [{s['rank']}] {s['source']} chunk {s['chunk']}  score {s['score']:.3f}  {s['snippet'][:90]}"
            )
        t = final["timings"]
        print(f"  retrieve {t['retrieve_ms']:.0f} ms, generate {t['generate_ms']:.0f} ms")
    return 0


def cmd_reindex(_: argparse.Namespace) -> int:
    pipe = _pipeline()
    print(f"re-embedded {pipe.reindex_raw_docs()} chunks from {pipe.settings.raw_docs_dir}")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    pipe = _pipeline()
    st = pipe.store
    print(
        f"embeddings: {st.fingerprint}\nllm: {pipe.llm.name}\nretriever: {pipe.settings.retriever}"
    )
    print(f"chunks: {len(st)} in {len(st.sources())} documents")
    for s in st.sources():
        print(f"  {s.source}: {s.chunks} chunks, {s.characters} characters")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    import uvicorn

    uvicorn.run("rag_chatbot.api:app", host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-chatbot", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="load, chunk, embed and index files or folders")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("ask", help="ask one question against the index")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--stream", action="store_true", help="print tokens as they arrive")
    p.add_argument("--json", action="store_true", help="print the full response as JSON")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser(
        "reindex", help="rebuild the index from data/raw_docs (after changing the embedder)"
    )
    p.set_defaults(fn=cmd_reindex)
    p = sub.add_parser("info", help="show what the index holds")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("serve", help="run the API and the web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (ConfigError, IndexMismatch, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
