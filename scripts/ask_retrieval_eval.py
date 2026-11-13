"""Retrieval check for the Ask-Aditya index on the factual questions of eval/ask_aditya_eval.jsonl.

A question counts as a hit when every fact it must contain appears in the top-k chunks (so it does not
depend on where chunk boundaries fall). Compares plain BM25, BM25 with the query heuristics, and (with
--dense, which loads MiniLM offline) the same fused with dense vectors by reciprocal rank.

    python scripts/ask_retrieval_eval.py [--index ask/private/index] [--dense] [--out results/ask_retrieval.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from askaditya.index import AskIndex, contextual_query  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", s).lower()


def covered(items, text: str) -> bool:
    return all(any(norm(a) in text for a in (it if isinstance(it, list) else [it])) for it in items)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=ROOT / "ask" / "private" / "index")
    ap.add_argument("--eval", type=Path, default=ROOT / "eval" / "ask_aditya_eval.jsonl")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    idx = AskIndex.load(args.index)
    qs = [json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line]
    qs = [q for q in qs if q["group"] == "factual"]
    model = None
    if args.dense:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [f"{c['title']} {c['section']} {c['text']}" for c in idx.chunks]
        idx.dense = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    rows = {}
    for name, kw in [("bm25", dict(heuristics=False)), ("bm25+heuristics", dict(heuristics=True))] + (
        [("hybrid (bm25+heuristics, dense)", dict(heuristics=True, dense=True))] if args.dense else []
    ):
        hit = 0
        miss = []
        for q in qs:
            query = contextual_query(q["question"], q["history"])
            vec = None
            if kw.get("dense"):
                vec = np.asarray(model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
            hits = idx.search(query, args.k, vec, heuristics=kw["heuristics"])
            text = norm("\n".join(h.chunk["text"] for h in hits))
            if covered(q["must_contain"], text):
                hit += 1
            else:
                miss.append(q["id"])
        rows[name] = {"hits": hit, "n": len(qs), "rate": hit / len(qs), "missed": miss}
        print(f"{name:34s} {hit}/{len(qs)} = {hit / len(qs):.3f}  missed {miss}")
    if args.out:
        args.out.write_text(json.dumps({"k": args.k, "index_version": idx.version, "rows": rows}, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
