"""Run the measurements that the README reports and write them to results/*.json.

    python scripts/run_eval.py retrieval                      # chunk size x retriever grid, gate sweep
    python scripts/run_eval.py generation --model Qwen/Qwen2.5-0.5B-Instruct --chunk-size 500 --retriever hybrid
    python scripts/run_eval.py latency --model Qwen/Qwen2.5-0.5B-Instruct --chunk-size 500 --retriever hybrid
    python scripts/run_eval.py extractive --chunk-size 500 --retriever hybrid

Each stage records the device, dtype, model revisions and library versions next to its numbers.
Then ``python scripts/check_numbers.py --write`` turns the results into the README tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CORPUS = ROOT / "corpus"

from rag_chatbot.config import Settings, pick_device  # noqa: E402
from rag_chatbot.embeddings import SentenceTransformerEmbeddings  # noqa: E402
from rag_chatbot.evaluation import (  # noqa: E402
    grade_answer,
    load_questions,
    percentile,
    resolve_spans,
    score_retrieval,
    summarise_answers,
)
from rag_chatbot.llm import ExtractiveLLM, LocalLLM, Prompt  # noqa: E402
from rag_chatbot.loaders import load_documents, split_documents  # noqa: E402
from rag_chatbot.pipeline import QueryLog, RAGPipeline  # noqa: E402
from rag_chatbot.retrieval import CrossEncoderReranker, Retriever  # noqa: E402
from rag_chatbot.store import VectorStore  # noqa: E402
from rag_chatbot.text import is_abstention  # noqa: E402

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OVERLAP_RATIO = 0.15
CHUNK_SIZES = (250, 500, 1000, 2000)
KS = (1, 3, 5)
GATE_GRID = [round(x, 2) for x in np.arange(0.0, 0.801, 0.05)]


def revision(repo: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo, local_files_only=True)).name
    except Exception:
        return None


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in (
        "torch",
        "transformers",
        "sentence_transformers",
        "faiss",
        "langchain_core",
        "numpy",
    ):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = None
    return out


def cpu_load() -> float | None:
    """Total CPU load (percent) from Windows, or None where that is unavailable."""
    try:
        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def meta(device: str, extra: dict | None = None) -> dict:
    questions = load_questions(CORPUS / "questions.jsonl")
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted((CORPUS / "docs").glob("*.txt"))}
    return {
        "date": dt.date.today().isoformat(),
        "device": device,
        "versions": versions(),
        "corpus": {
            "documents": len(docs),
            "words": sum(len(t.split()) for t in docs.values()),
            "characters": sum(len(t) for t in docs.values()),
        },
        "questions": {
            "total": len(questions),
            "answerable": sum(q.answerable for q in questions),
            "unanswerable": sum(not q.answerable for q in questions),
            "direct": sum(q.kind == "direct" for q in questions),
            "paraphrase": sum(q.kind == "paraphrase" for q in questions),
            "near_miss": sum(q.kind == "near_miss" for q in questions),
            "off_topic": sum(q.kind == "off_topic" for q in questions),
        },
        **(extra or {}),
    }


def build_store(embedder, size: int, overlap: int | None = None) -> VectorStore:
    overlap = overlap if overlap is not None else round(size * OVERLAP_RATIO)
    chunks = split_documents(load_documents(sorted((CORPUS / "docs").glob("*.txt"))), size, overlap)
    store = VectorStore(embedder, embedder.fingerprint)
    store.add(chunks)
    return store


def write(name: str, data: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(
        json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print("wrote", path.relative_to(ROOT))


# ---------------------------------------------------------------------------------------------
def stage_retrieval(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    questions = load_questions(CORPUS / "questions.jsonl")
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted((CORPUS / "docs").glob("*.txt"))}
    spans = {q.id: resolve_spans(q, docs) for q in questions if q.answerable}
    embedder = SentenceTransformerEmbeddings(EMBEDDER, device)
    reranker = None
    rerank_note = "not run"
    if not args.no_rerank:
        try:
            reranker = CrossEncoderReranker(RERANKER, device)
            reranker.score("warm up", ["warm up"])
            rerank_note = "run"
        except Exception as exc:  # a missing download must not sink the other measurements
            reranker, rerank_note = None, f"not run ({type(exc).__name__})"
    rows = []
    for size in CHUNK_SIZES:
        store = build_store(embedder, size)
        n_chunks = len(store)
        modes = [("dense", None), ("bm25", None), ("hybrid", None)]
        if reranker:
            modes.append(("hybrid", reranker))
        for mode, rr in modes:
            retr = Retriever(store, mode, rr)
            ranked, ctx_chars = {}, []
            t0 = time.perf_counter()
            for q in questions:
                hits = retr.retrieve(q.question, 10).hits
                if q.answerable:
                    ranked[q.id] = [
                        (
                            h.chunk.metadata["source"],
                            h.chunk.metadata["start"],
                            h.chunk.metadata["end"],
                            h.chunk.page_content,
                        )
                        for h in hits
                    ]
                    ctx_chars.append(sum(len(h.chunk.page_content) for h in hits[:4]))
            elapsed = time.perf_counter() - t0
            sc = score_retrieval(questions, ranked, spans, KS)
            by_kind = {}
            for kind in ("direct", "paraphrase"):
                sub = [q for q in questions if q.answerable and q.kind == kind]
                s = score_retrieval(sub, {q.id: ranked[q.id] for q in sub}, spans, KS)
                by_kind[kind] = {
                    "n": s.n,
                    "recall": {str(k): v for k, v in s.recall.items()},
                    "mrr": s.mrr,
                }
            rows.append(
                {
                    "chunk_size": size,
                    "chunk_overlap": round(size * OVERLAP_RATIO),
                    "chunks": n_chunks,
                    "retriever": mode + ("+rerank" if rr else ""),
                    "n": sc.n,
                    "recall": {str(k): v for k, v in sc.recall.items()},
                    "mrr": sc.mrr,
                    "answer_in_context": {str(k): v for k, v in sc.answer_in_context.items()},
                    "mean_context_chars_k4": float(np.mean(ctx_chars)),
                    "by_kind": by_kind,
                    "seconds_for_100_queries": round(elapsed, 2),
                }
            )
            print(
                f"size {size:5d} {rows[-1]['retriever']:14s} MRR {sc.mrr:.3f} R@3 {sc.recall[3]:.3f}",
                flush=True,
            )
    data = {
        "meta": meta(
            device,
            {
                "embedder": {"model": EMBEDDER, "revision": revision(EMBEDDER)},
                "reranker": {
                    "model": RERANKER,
                    "revision": revision(RERANKER),
                    "status": rerank_note,
                },
                "chunk_overlap_ratio": OVERLAP_RATIO,
                "ks": list(KS),
                "mrr_depth": 10,
            },
        ),
        "rows": rows,
    }
    write("retrieval.json", data)


def top_dense(store: VectorStore, retr: Retriever, questions, k: int) -> dict[str, float]:
    return {
        q.id: max((h.dense for h in retr.retrieve(q.question, k).hits), default=0.0)
        for q in questions
    }


def pick_threshold(scores: dict[str, float], questions) -> float:
    """Gate threshold with the best balanced accuracy (abstain on unanswerable, pass answerable)."""
    best, best_t = -1.0, 0.0
    for t in GATE_GRID:
        una = [q for q in questions if not q.answerable]
        ans = [q for q in questions if q.answerable]
        bal = (
            sum(scores[q.id] < t for q in una) / len(una)
            + sum(scores[q.id] >= t for q in ans) / len(ans)
        ) / 2
        if bal > best + 1e-9:
            best, best_t = bal, t
    return best_t


def run_config(
    llm,
    name: str,
    questions,
    retr: Retriever,
    k: int,
    few_shot: bool,
    gate: float | None,
    closed_book: bool = False,
    max_new: int = 160,
) -> dict:
    """Answer every question under one configuration and grade it."""
    cfg = Settings(few_shot=few_shot, top_k=k, min_dense_score=gate, retriever=retr.mode)
    pipe = RAGPipeline(cfg, retr.store, retr, llm)
    pipe.log = QueryLog(None)
    records, gen_ms = [], []
    for q in questions:
        t0 = time.perf_counter()
        if closed_book:
            prompt = Prompt(
                system="Answer the question briefly.", user=q.question, question=q.question
            )
            answer = llm.generate(prompt, max_new, 0.0)
            context, retrieved, abstained = "", [], is_abstention(answer)
            gen_ms.append((time.perf_counter() - t0) * 1000)
        else:
            res = pipe.ask(q.question)
            answer, abstained = res["answer"], res["abstained"]
            context = " ".join(s["text"] for s in res["sources"])
            retrieved = [s["chunk_id"] for s in res["sources"]]
            gen_ms.append(res["timings"]["generate_ms"])
            pipe.cache.clear()
        rec = grade_answer(q, answer, abstained, context)
        rec["retrieved"] = retrieved
        if closed_book:
            rec["support"] = None
            rec["unsupported_numbers"] = []
        records.append(rec)
    summary_all = summarise_answers([_for_summary(r) for r in records])
    return {
        "name": name,
        "few_shot": few_shot,
        "gate": gate,
        "closed_book": closed_book,
        "k": k,
        "mean_generate_ms": float(np.mean(gen_ms)),
        "p95_generate_ms": percentile(gen_ms, 0.95),
        "summary": summary_all,
        "records": records,
    }


def _for_summary(r: dict) -> dict:
    return {**r, "support": 1.0 if r["support"] is None else r["support"]}


def subset_summary(cfg: dict, ids: set[str]) -> dict:
    return summarise_answers([_for_summary(r) for r in cfg["records"] if r["id"] in ids])


def stage_generation(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    questions = load_questions(CORPUS / "questions.jsonl")
    embedder = SentenceTransformerEmbeddings(EMBEDDER, device)
    store = build_store(embedder, args.chunk_size)
    retr = Retriever(store, args.retriever)
    split_a = {q.id for i, q in enumerate(questions) if i % 2 == 0}
    split_b = {q.id for i, q in enumerate(questions) if i % 2 == 1}
    # the abstention gate is tuned on half A (retrieval scores only) and judged on half B
    scores = top_dense(store, retr, questions, args.k)
    qa = [q for q in questions if q.id in split_a]
    threshold = pick_threshold({q.id: scores[q.id] for q in qa}, qa)
    sweep = []
    for t in GATE_GRID:
        una = [q for q in questions if not q.answerable]
        ans = [q for q in questions if q.answerable]
        sweep.append(
            {
                "threshold": t,
                "abstain_on_unanswerable": sum(scores[q.id] < t for q in una) / len(una),
                "false_abstain_on_answerable": sum(scores[q.id] < t for q in ans) / len(ans),
            }
        )
    llm = LocalLLM(args.model, device)
    llm.generate(
        Prompt(system="s", user="warm up"), 4, 0.0
    )  # load and warm up before timing anything
    configs = {}
    for name, kw in (
        ("closed_book", dict(few_shot=False, gate=None, closed_book=True)),
        ("rag", dict(few_shot=False, gate=None)),
        ("rag_few_shot", dict(few_shot=True, gate=None)),
        ("rag_gate", dict(few_shot=False, gate=threshold)),
        ("rag_few_shot_gate", dict(few_shot=True, gate=threshold)),
    ):
        configs[name] = run_config(llm, name, questions, retr, args.k, **kw)
        s = configs[name]["summary"]
        print(
            f"{name:18s} correct {s['correct_on_answerable']['rate']:.2f} abst-unans "
            f"{s['abstained_on_unanswerable']['rate']:.2f} false-abst {s['false_abstention_on_answerable']['rate']:.2f}",
            flush=True,
        )
    for cfg in configs.values():
        cfg["summary_half_b"] = subset_summary(cfg, split_b)
    data = {
        "meta": meta(
            device,
            {
                "llm": {"model": args.model, "revision": revision(args.model), "dtype": llm.dtype},
                "embedder": {"model": EMBEDDER, "revision": revision(EMBEDDER)},
                "retrieval": {
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": round(args.chunk_size * OVERLAP_RATIO),
                    "retriever": args.retriever,
                    "k": args.k,
                },
                "decoding": "greedy",
                "max_new_tokens": 160,
                "gate": {
                    "threshold": threshold,
                    "tuned_on": "even-numbered questions (half A)",
                    "judged_on": "odd-numbered questions (half B)",
                    "statistic": "highest cosine among the top-k chunks",
                    "n_half_a": len(split_a),
                    "n_half_b": len(split_b),
                },
            },
        ),
        "gate_sweep": sweep,
        "configs": configs,
    }
    write(f"generation_{args.model.split('/')[-1].lower()}.json", data)


def stage_extractive(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    questions = load_questions(CORPUS / "questions.jsonl")
    embedder = SentenceTransformerEmbeddings(EMBEDDER, device)
    store = build_store(embedder, args.chunk_size)
    retr = Retriever(store, args.retriever)
    cfg = run_config(
        ExtractiveLLM(), "extractive", questions, retr, args.k, few_shot=False, gate=None
    )
    data = {
        "meta": meta(
            device,
            {
                "llm": {"model": "extractive (no model)", "revision": None, "dtype": None},
                "embedder": {"model": EMBEDDER, "revision": revision(EMBEDDER)},
                "retrieval": {
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": round(args.chunk_size * OVERLAP_RATIO),
                    "retriever": args.retriever,
                    "k": args.k,
                },
            },
        ),
        "configs": {"extractive": cfg},
    }
    write("generation_extractive.json", data)


def stage_latency(args: argparse.Namespace) -> None:
    """End-to-end timings on the CPU: the one place where the machine must be quiet."""
    import torch

    torch.set_num_threads(args.threads)
    load_before = cpu_load()
    questions = load_questions(CORPUS / "questions.jsonl")
    sample = [q for q in questions if q.answerable][::8][:8] + [
        q for q in questions if not q.answerable
    ][::4][:6]
    t0 = time.perf_counter()
    embedder = SentenceTransformerEmbeddings(EMBEDDER, "cpu")
    embedder.embed_documents(["warm up"])
    t_embed_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    store = build_store(embedder, args.chunk_size)
    t_index = time.perf_counter() - t0
    retr = Retriever(store, args.retriever)
    llm = LocalLLM(args.model, "cpu")
    t0 = time.perf_counter()
    llm.generate(Prompt(system="s", user="warm up"), 4, 0.0)
    t_llm_load = time.perf_counter() - t0
    cfg = Settings(top_k=args.k, retriever=args.retriever)
    pipe = RAGPipeline(cfg, store, retr, llm)
    pipe.log = QueryLog(None)
    pipe.ask(sample[0].question)  # warm the retrieval path too
    rows = []
    for q in sample:
        pipe.cache.clear()
        t0 = time.perf_counter()
        res = pipe.ask(q.question)
        wall = (time.perf_counter() - t0) * 1000
        n_tok = len(llm._tok(res["answer"])["input_ids"])
        rows.append(
            {
                "id": q.id,
                "answerable": q.answerable,
                "retrieve_ms": res["timings"]["retrieve_ms"],
                "generate_ms": res["timings"]["generate_ms"],
                "total_ms": wall,
                "answer_tokens": n_tok,
            }
        )
    load_after = cpu_load()
    data = {
        "meta": meta(
            "cpu",
            {
                "llm": {"model": args.model, "revision": revision(args.model), "dtype": llm.dtype},
                "embedder": {"model": EMBEDDER, "revision": revision(EMBEDDER)},
                "retrieval": {
                    "chunk_size": args.chunk_size,
                    "retriever": args.retriever,
                    "k": args.k,
                },
                "torch_threads": args.threads,
                "logical_cpus": os.cpu_count(),
                "cpu_load_percent_before": load_before,
                "cpu_load_percent_after": load_after,
            },
        ),
        "queries": len(rows),
        "startup_seconds": {
            "embedder_load": t_embed_load,
            "index_build": t_index,
            "llm_load_and_first_call": t_llm_load,
            "chunks": len(store),
        },
        "retrieve_ms": {
            "p50": percentile([r["retrieve_ms"] for r in rows], 0.5),
            "p95": percentile([r["retrieve_ms"] for r in rows], 0.95),
        },
        "generate_ms": {
            "p50": percentile([r["generate_ms"] for r in rows], 0.5),
            "p95": percentile([r["generate_ms"] for r in rows], 0.95),
        },
        "total_ms": {
            "p50": percentile([r["total_ms"] for r in rows], 0.5),
            "p95": percentile([r["total_ms"] for r in rows], 0.95),
        },
        "rows": rows,
    }
    write(f"latency_cpu_{args.model.split('/')[-1].lower()}.json", data)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["retrieval", "generation", "extractive", "latency"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--retriever", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()
    {
        "retrieval": stage_retrieval,
        "generation": stage_generation,
        "extractive": stage_extractive,
        "latency": stage_latency,
    }[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
