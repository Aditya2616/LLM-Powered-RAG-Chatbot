"""Record real queries from the pipeline into docs/data/demo.json, the file the static demo replays.

    python scripts/record_demo.py --queries a01,a62,a71,u10,u18 [--gate 0.3] [--few-shot]

Everything in the file is produced by running the pipeline: the chunk layout, the 2-D projection of
the chunk and query vectors, the retrieved chunks with their scores, the token stream with the time
each token arrived, and the final answer. Nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_chatbot.config import Settings, pick_device  # noqa: E402
from rag_chatbot.embeddings import SentenceTransformerEmbeddings  # noqa: E402
from rag_chatbot.evaluation import load_questions  # noqa: E402
from rag_chatbot.llm import LocalLLM  # noqa: E402
from rag_chatbot.pipeline import QueryLog, RAGPipeline  # noqa: E402
from rag_chatbot.retrieval import Retriever  # noqa: E402
from run_eval import CORPUS, EMBEDDER, OVERLAP_RATIO, build_store, revision  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queries", required=True, help="comma-separated ids from corpus/questions.jsonl"
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--few-shot", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "demo.json"))
    args = ap.parse_args()

    device = pick_device(args.device)
    embedder = SentenceTransformerEmbeddings(EMBEDDER, device)
    store = build_store(embedder, args.chunk_size)
    cfg = Settings(
        top_k=args.k,
        retriever=args.retriever,
        min_dense_score=args.gate,
        few_shot=args.few_shot,
        chunk_size=args.chunk_size,
        chunk_overlap=round(args.chunk_size * OVERLAP_RATIO),
    )
    llm = LocalLLM(args.model, device)
    pipe = RAGPipeline(cfg, store, Retriever(store, args.retriever), llm)
    pipe.log = QueryLog(None)
    pipe.ask("warm up the model and the retriever")
    pipe.cache.clear()

    xy = store.project(store.matrix)
    chunks = [
        {
            "chunk_id": c.metadata["chunk_id"],
            "source": c.metadata["source"],
            "chunk": c.metadata["chunk"],
            "start": c.metadata["start"],
            "end": c.metadata["end"],
            "x": round(float(x), 4),
            "y": round(float(y), 4),
        }
        for c, (x, y) in zip(store.chunks, xy, strict=True)
    ]
    by_id = {q.id: q for q in load_questions(CORPUS / "questions.jsonl")}
    queries = []
    for qid in args.queries.split(","):
        q = by_id[qid]
        pipe.cache.clear()
        retrieval, tokens, done, t_gen = None, [], None, None
        for ev in pipe.stream(q.question):
            name = ev["event"]
            if name == "retrieval":
                retrieval, t_gen = (
                    {k: v for k, v in ev.items() if k != "event"},
                    time.perf_counter(),
                )
            elif name == "token":
                tokens.append(
                    {"t": round((time.perf_counter() - t_gen) * 1000, 1), "text": ev["text"]}
                )
            else:
                done = {k: v for k, v in ev.items() if k != "event"}
        queries.append(
            {
                "id": q.id,
                "question": q.question,
                "kind": q.kind,
                "answerable": q.answerable,
                "retrieval": retrieval,
                "tokens": tokens,
                "done": done,
            }
        )
        print(qid, "->", done["answer"][:80], "| abstained:", done["abstained"], flush=True)

    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted((CORPUS / "docs").glob("*.txt"))}
    data = {
        "meta": {
            "recorded": True,
            "date": dt.date.today().isoformat(),
            "device": device,
            "embedder": {
                "model": EMBEDDER,
                "revision": revision(EMBEDDER),
                "short": "MiniLM-L6",
                "dim": int(store.matrix.shape[1]),
            },
            "llm": {
                "model": args.model,
                "revision": revision(args.model),
                "short": args.model.split("/")[-1],
                "dtype": llm.dtype,
            },
            "retrieval": {
                "chunk_size": args.chunk_size,
                "chunk_overlap": round(args.chunk_size * OVERLAP_RATIO),
                "retriever": args.retriever,
                "k": args.k,
                "gate": args.gate,
                "few_shot": args.few_shot,
            },
            "corpus": {
                "documents": len(docs),
                "words": sum(len(t.split()) for t in docs.values()),
                "chunks": len(chunks),
            },
        },
        "documents": [{"name": n, "characters": len(t)} for n, t in docs.items()],
        "chunks": chunks,
        "queries": queries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("wrote", out, f"({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
