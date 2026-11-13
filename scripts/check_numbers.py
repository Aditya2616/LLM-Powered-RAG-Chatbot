"""Keep every published number tied to the results files.

    python scripts/check_numbers.py --write   # regenerate the README blocks and docs/data/results.json
    python scripts/check_numbers.py --check   # fail if anything is out of sync (a fast test runs this)

Generated, never hand-typed: the tables and figures between ``<!-- BEGIN name -->`` and
``<!-- END name -->`` markers in README.md, and docs/data/results.json (what the interface shows).
``CLAIMS`` lists the qualitative statements the README makes; each is verified against the data.
It also checks that the recorded demo agrees with the corpus on disk.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
CORPUS = ROOT / "corpus"

pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
f3 = lambda x: f"{x:.3f}"  # noqa: E731


def rate(r: dict) -> str:
    return f"{r['k']}/{r['n']} ({pct(r['rate'])}%)"


def rate_ci(r: dict) -> str:
    return f"{pct(r['rate'])}% [{pct(r['ci'][0])}, {pct(r['ci'][1])}]"


def load() -> dict:
    d: dict = {"retrieval": json.loads((RESULTS / "retrieval.json").read_text(encoding="utf-8"))}
    d["gen"] = {}
    for p in sorted(RESULTS.glob("generation_*.json")):
        d["gen"][p.stem.removeprefix("generation_")] = json.loads(p.read_text(encoding="utf-8"))
    d["lat"] = {}
    for p in sorted(RESULTS.glob("latency_cpu_*.json")):
        d["lat"][p.stem.removeprefix("latency_cpu_")] = json.loads(p.read_text(encoding="utf-8"))
    d["demo"] = json.loads((DOCS / "data" / "demo.json").read_text(encoding="utf-8"))
    return d


def short(model: str) -> str:
    return model.split("/")[-1]


# ------------------------------------------------------------------------------- blocks ---------
def b_setup(d: dict) -> str:
    m = d["retrieval"]["meta"]
    q, c = m["questions"], m["corpus"]
    gens = [g for g in d["gen"].values() if g["meta"]["llm"]["dtype"]]
    g0 = gens[0]["meta"]
    return (
        f"The corpus is {c['documents']} invented policy documents ({c['words']:,} words, {c['characters']:,} characters) "
        f"for a fictional outdoor-gear retailer. The question set has {q['total']} questions: {q['answerable']} answerable "
        f"({q['direct']} worded close to the text, {q['paraphrase']} reworded) and {q['unanswerable']} that the documents cannot "
        f"answer ({q['near_miss']} on a topic the documents cover but a detail they do not, {q['off_topic']} off topic). "
        f"Retrieval ran on {m['device']} with {short(m['embedder']['model'])} embeddings (revision `{m['embedder']['revision'][:8]}`); "
        f"generation ran on {g0['device']} in {g0['llm']['dtype']} with greedy decoding and at most {g0['max_new_tokens']} new tokens, "
        f"retrieving the top {g0['retrieval']['k']} chunks with the {g0['retrieval']['retriever']} retriever at "
        f"{g0['retrieval']['chunk_size']}-character chunks unless a row says otherwise. Dates: retrieval {m['date']}, "
        f"generation {', '.join(sorted({g['meta']['date'] for g in gens}))}."
    )


def b_retrieval(d: dict) -> str:
    r = d["retrieval"]
    ks = r["meta"]["ks"]
    head = [
        "| Chunk size (chars) | Chunks | Retriever | "
        + " | ".join(f"Recall@{k}" for k in ks)
        + " | MRR | Answer facts in top-3 text | Mean top-4 text (chars) |",
        "|---|---|---|" + "---|" * (len(ks) + 3),
    ]
    rows = []
    for x in r["rows"]:
        label = x["retriever"]
        if label == "hybrid":
            label = "hybrid"
        rows.append(
            f"| {x['chunk_size']} | {x['chunks']} | {label} | "
            + " | ".join(pct(x["recall"][str(k)]) + "%" for k in ks)
            + f" | {f3(x['mrr'])} | {pct(x['answer_in_context']['3'])}% | {x['mean_context_chars_k4']:.0f} |"
        )
    return "\n".join(head + rows)


def b_retrieval_kind(d: dict) -> str:
    r = d["retrieval"]
    size = d["demo"]["meta"]["retrieval"]["chunk_size"]
    head = [
        "| Retriever (chunk size "
        + str(size)
        + ") | Worded close to the text: Recall@3 / MRR | Reworded: Recall@3 / MRR |",
        "|---|---|---|",
    ]
    rows = []
    for x in r["rows"]:
        if x["chunk_size"] != size:
            continue
        bk = x["by_kind"]
        rows.append(
            f"| {x['retriever']} | {pct(bk['direct']['recall']['3'])}% / {f3(bk['direct']['mrr'])} "
            f"(n={bk['direct']['n']}) | {pct(bk['paraphrase']['recall']['3'])}% / {f3(bk['paraphrase']['mrr'])} "
            f"(n={bk['paraphrase']['n']}) |"
        )
    return "\n".join(head + rows)


CFG_LABEL = {
    "closed_book": "No retrieval (question only)",
    "rag": "RAG, plain prompt",
    "rag_rerank": "RAG + cross-encoder reranker",
    "rag_few_shot": "RAG, two worked examples in the prompt",
    "rag_gate": "RAG + retrieval-score gate",
    "rag_few_shot_gate": "RAG, worked examples + gate",
    "extractive": "Extractive baseline (no model)",
}


def gen_table(g: dict, subset: str = "summary") -> str:
    head = [
        '| Setup | Correct on answerable | Wrong answer | False "I don\'t know" | "I don\'t know" on unanswerable | '
        "near-miss | off-topic | Right decision overall | Grounded (>=80% of words in context) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for name, c in g["configs"].items():
        s = c[subset]
        cb = c["closed_book"]
        rows.append(
            f"| {CFG_LABEL[name]} | {rate(s['correct_on_answerable'])} | {rate(s['wrong_answer_on_answerable'])} | "
            f"{rate(s['false_abstention_on_answerable'])} | {rate(s['abstained_on_unanswerable'])} | "
            f"{rate(s['abstained_on_near_miss'])} | {rate(s['abstained_on_off_topic'])} | {rate(s['decision_accuracy'])} | "
            + ("n/a" if cb else rate(s["grounded"]))
            + " |"
        )
    return "\n".join(head + rows)


def b_gen(d: dict, key: str) -> str:
    g = d["gen"][key]
    m = g["meta"]
    title = f"**{short(m['llm']['model'])}**, {m['retrieval']['chunk_size']}-character chunks, {m['retrieval']['retriever']} retrieval, top {m['retrieval']['k']}, all {m['questions']['total']} questions:\n\n"
    return title + gen_table(g)


def b_gate(d: dict) -> str:
    key = next(k for k in d["gen"] if "0.5b" in k and k.endswith("c500"))
    g = d["gen"][key]
    m = g["meta"]["gate"]
    sweep = [s for s in g["gate_sweep"] if round(s["threshold"] * 100) % 10 == 0]
    out = [
        f"Threshold tuned on the {m['n_half_a']} {m['tuned_on']}; statistic: {m['statistic']}. Chosen threshold: **{m['threshold']:.2f}**.\n",
        '| Threshold | "I don\'t know" on unanswerable (all 25) | False "I don\'t know" on answerable (all 75) |',
        "|---|---|---|",
    ]
    for s in sweep:
        out.append(
            f"| {s['threshold']:.2f} | {pct(s['abstain_on_unanswerable'])}% | {pct(s['false_abstain_on_answerable'])}% |"
        )
    out.append("")
    out.append(
        f"Judged on the {m['n_half_b']} {m['judged_on']} only ({short(g['meta']['llm']['model'])}):\n"
    )
    out.append(
        gen_table(
            {
                "configs": {
                    k: v
                    for k, v in g["configs"].items()
                    if k in ("rag", "rag_gate", "rag_few_shot", "rag_few_shot_gate")
                }
            },
            "summary_half_b",
        )
    )
    return "\n".join(out)


def b_latency(d: dict) -> str:
    if not d["lat"]:
        return "Not run."
    rows = [
        "| Model | Device | Threads | Queries | Retrieve p50 / p95 (ms) | Generate p50 / p95 (ms) | Total p50 / p95 (ms) | CPU load before / after |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in d["lat"].values():
        m = v["meta"]
        rows.append(
            f"| {short(m['llm']['model'])} ({m['llm']['dtype']}) | {m['device']} | {m['torch_threads']} of {m['logical_cpus']} logical | {v['queries']} | "
            f"{v['retrieve_ms']['p50']:.0f} / {v['retrieve_ms']['p95']:.0f} | {v['generate_ms']['p50']:.0f} / {v['generate_ms']['p95']:.0f} | "
            f"{v['total_ms']['p50']:.0f} / {v['total_ms']['p95']:.0f} | {m['cpu_load_percent_before']}% / {m['cpu_load_percent_after']}% |"
        )
    return "\n".join(rows)


def b_gpu_latency(d: dict) -> str:
    rows = [
        "| Model | Device | Mean generation per question (ms) | p95 (ms) | Questions |",
        "|---|---|---|---|---|",
    ]
    for g in d["gen"].values():
        c = g["configs"].get("rag")
        if not c or not g["meta"]["llm"]["dtype"]:
            continue
        rows.append(
            f"| {short(g['meta']['llm']['model'])} ({g['meta']['llm']['dtype']}, {g['meta']['retrieval']['chunk_size']}-char chunks) | {g['meta']['device']} | {c['mean_generate_ms']:.0f} | {c['p95_generate_ms']:.0f} | {g['meta']['questions']['total']} |"
        )
    return "\n".join(rows)


def headline_facts(d: dict) -> dict:
    """The numbers quoted in prose, computed once so prose and tables cannot disagree."""
    r = {(x["chunk_size"], x["retriever"]): x for x in d["retrieval"]["rows"]}
    size = d["demo"]["meta"]["retrieval"]["chunk_size"]
    k05 = next(k for k in d["gen"] if "0.5b" in k and k.endswith(f"c{size}"))
    k15 = next((k for k in d["gen"] if "1.5b" in k), None)
    g05 = d["gen"][k05]["configs"]
    out = {
        "size": size,
        "hyb_mrr": r[(size, "hybrid")]["mrr"],
        "dense_mrr": r[(size, "dense")]["mrr"],
        "bm25_mrr": r[(size, "bm25")]["mrr"],
        "hyb_r3": r[(size, "hybrid")]["recall"]["3"],
        "hyb_r1": r[(size, "hybrid")]["recall"]["1"],
        "rr_mrr": r[(size, "hybrid+rerank")]["mrr"] if (size, "hybrid+rerank") in r else None,
        "g05": g05,
    }
    if k15:
        out["g15"] = d["gen"][k15]["configs"]
    return out


def b_headline(d: dict) -> str:
    h = headline_facts(d)
    rag = h["g05"]["rag"]["summary"]
    cb = h["g05"]["closed_book"]["summary"]
    lines = [
        f"- Retrieval ({h['size']}-character chunks): hybrid MRR {f3(h['hyb_mrr'])}, dense {f3(h['dense_mrr'])}, BM25 {f3(h['bm25_mrr'])}; "
        f"hybrid puts a chunk holding the gold passage in the top 3 for {pct(h['hyb_r3'])}% of the 75 answerable questions.",
        f"- With Qwen2.5-0.5B-Instruct, RAG answers {rate(rag['correct_on_answerable'])} of the answerable questions correctly "
        f"(without retrieval: {rate(cb['correct_on_answerable'])}).",
        f'- It says "I don\'t know" on {rate(rag["abstained_on_unanswerable"])} of the unanswerable questions, and wrongly on '
        f"{rate(rag['false_abstention_on_answerable'])} of the answerable ones.",
    ]
    if "g15" in h:
        r15 = h["g15"]["rag"]["summary"]
        lines.append(
            f"- Qwen2.5-1.5B-Instruct: {rate(r15['correct_on_answerable'])} correct, "
            f"{rate(r15['abstained_on_unanswerable'])} abstentions on unanswerable, "
            f"{rate(r15['false_abstention_on_answerable'])} false abstentions."
        )
    return "\n".join(lines)


BLOCKS = {
    "setup": b_setup,
    "retrieval": b_retrieval,
    "retrieval_kind": b_retrieval_kind,
    "headline": b_headline,
    "gate": b_gate,
    "latency": b_latency,
    "gpu_latency": b_gpu_latency,
}


def dynamic_blocks(d: dict) -> dict:
    out = {name: fn(d) for name, fn in BLOCKS.items()}
    for key in d["gen"]:
        if key == "extractive":
            continue
        out[f"gen_{key}"] = b_gen(d, key)
    if "extractive" in d["gen"]:
        g = d["gen"]["extractive"]
        out["gen_extractive"] = (
            "No generator at all: return the retrieved sentence with the most question words "
            '(plural-stemmed), or "I don\'t know" if fewer than two match.\n\n' + gen_table(g)
        )
    return out


# ------------------------------------------------------------------------------ UI data --------
def ui_results(d: dict) -> dict:
    h = headline_facts(d)
    size = h["size"]
    r = d["retrieval"]["rows"]
    t1 = {
        "title": f"Retrieval, {size}-character chunks (75 answerable questions)",
        "head": ["Retriever", "Recall@1", "Recall@3", "MRR"],
        "rows": [
            [x["retriever"], pct(x["recall"]["1"]) + "%", pct(x["recall"]["3"]) + "%", f3(x["mrr"])]
            for x in r
            if x["chunk_size"] == size
        ],
        "hl": None,
    }
    for i, row in enumerate(t1["rows"]):
        if row[0] == "hybrid":
            t1["hl"] = i
    rows, hl = [], None
    for key, g in d["gen"].items():
        for name in ("closed_book", "rag", "rag_few_shot", "rag_gate", "extractive"):
            c = g["configs"].get(name)
            if not c or (name != "closed_book" and key.endswith("c1000")):
                continue
            s = c["summary"]
            who = (
                "extractive"
                if key == "extractive"
                else short(g["meta"]["llm"]["model"]).replace("-Instruct", "")
            )
            rows.append(
                [
                    f"{who}: {CFG_LABEL[name].lower()}",
                    pct(s["correct_on_answerable"]["rate"]) + "%",
                    pct(s["false_abstention_on_answerable"]["rate"]) + "%",
                    pct(s["abstained_on_unanswerable"]["rate"]) + "%",
                ]
            )
            if name == "rag" and "0.5b" in key and key.endswith(f"c{size}"):
                hl = len(rows) - 1
    t2 = {
        "title": "Answers (75 answerable, 25 unanswerable)",
        "head": ["Setup", "Correct", 'False "I don\'t know"', '"I don\'t know" on unanswerable'],
        "rows": rows,
        "hl": hl,
    }
    notes = [
        "The corpus is 11 invented documents of about 5,400 words and the questions are written by me, so scores are high and the set is small: a 1-point difference is under one question in the retrieval table.",
        "Answer correctness is a word-match against the facts the answer must contain. It is strict about wording and blind to meaning, in both directions.",
        "The model is a 0.5B-parameter open model. It follows the instruction to decline imperfectly, and the numbers above are that, not a tuned best case.",
        "This page replays a recorded run. Run the service locally to ask your own questions of your own documents.",
    ]
    return {
        "note": f"{d['demo']['meta']['date']}, GPU float16 for generation, bundled corpus",
        "tables": [t1, t2],
        "notes": notes,
    }


# ------------------------------------------------------------------------------- claims ---------
def claims(d: dict) -> list[tuple[str, bool]]:
    rows = d["retrieval"]["rows"]
    by = {(x["chunk_size"], x["retriever"]): x for x in rows}
    sizes = sorted({x["chunk_size"] for x in rows})
    h = headline_facts(d)
    rag = h["g05"]["rag"]["summary"]
    cb = h["g05"]["closed_book"]["summary"]
    out = [
        (
            "hybrid MRR is at least the better of dense and BM25 at the default chunk size minus 0.02",
            by[(h["size"], "hybrid")]["mrr"]
            >= max(by[(h["size"], "dense")]["mrr"], by[(h["size"], "bm25")]["mrr"]) - 0.02,
        ),
        (
            "hybrid beats dense on MRR at every chunk size",
            all(by[(s, "hybrid")]["mrr"] > by[(s, "dense")]["mrr"] for s in sizes),
        ),
        (
            "retrieval makes the 0.5B model more accurate than no retrieval",
            rag["correct_on_answerable"]["rate"] > cb["correct_on_answerable"]["rate"],
        ),
        (
            "every summary's counts add up",
            all(
                c["summary"]["correct_on_answerable"]["k"]
                + c["summary"]["false_abstention_on_answerable"]["k"]
                + c["summary"]["wrong_answer_on_answerable"]["k"]
                == c["summary"]["n_answerable"]
                for g in d["gen"].values()
                for c in g["configs"].values()
            ),
        ),
        (
            "every generation file covers the same questions as the retrieval file",
            all(
                g["meta"]["questions"] == d["retrieval"]["meta"]["questions"]
                for g in d["gen"].values()
            ),
        ),
    ]
    g05 = d["gen"]["qwen2.5-0.5b-instruct_c500"]["configs"]
    g15 = d["gen"]["qwen2.5-1.5b-instruct_c500"]["configs"]
    g10 = d["gen"]["qwen2.5-0.5b-instruct_c1000"]["configs"]
    ex = d["gen"]["extractive"]["configs"]["extractive"]["summary"]
    ab = lambda g, n: g[n]["summary"]["abstained_on_unanswerable"]["rate"]  # noqa: E731
    ok = lambda g, n: g[n]["summary"]["correct_on_answerable"]["rate"]  # noqa: E731
    out += [
        (
            "worked examples in the prompt lower the abstention rate on unanswerable questions for both models",
            ab(g05, "rag_few_shot") < ab(g05, "rag") and ab(g15, "rag_few_shot") < ab(g15, "rag"),
        ),
        (
            "the gate raises abstention on unanswerable questions but lowers correct answers, for both models",
            all(
                ab(g, "rag_gate") > ab(g, "rag") and ok(g, "rag_gate") < ok(g, "rag")
                for g in (g05, g15)
            ),
        ),
        (
            "the reranker raises MRR at every chunk size, but lowers Recall@3 at 500 characters",
            all(by[(s, "hybrid+rerank")]["mrr"] > by[(s, "hybrid")]["mrr"] for s in sizes)
            and by[(500, "hybrid+rerank")]["recall"]["3"] < by[(500, "hybrid")]["recall"]["3"],
        ),
        (
            "the reranker changes the number of correct answers by at most two questions at 500 characters, for both models",
            all(
                abs(
                    g["rag_rerank"]["summary"]["correct_on_answerable"]["k"]
                    - g["rag"]["summary"]["correct_on_answerable"]["k"]
                )
                <= 2
                for g in (g05, g15)
            ),
        ),
        (
            "the 1.5B model answers more correctly than the 0.5B model",
            ok(g15, "rag") > ok(g05, "rag"),
        ),
        (
            "neither model abstains on more than 3 of 4 unanswerable questions",
            ab(g05, "rag") < 0.75 and ab(g15, "rag") < 0.75,
        ),
        (
            "500-character chunks give the 0.5B model more correct answers than 1000-character chunks",
            ok(g05, "rag") > ok(g10, "rag"),
        ),
        (
            "near-miss questions are declined less often than off-topic ones for both models",
            all(
                g["rag"]["summary"]["abstained_on_near_miss"]["rate"]
                < g["rag"]["summary"]["abstained_on_off_topic"]["rate"]
                for g in (g05, g15)
            ),
        ),
        (
            "hybrid MRR is within 0.02 of the best single retriever at every chunk size",
            all(
                by[(s, "hybrid")]["mrr"]
                >= max(by[(s, "dense")]["mrr"], by[(s, "bm25")]["mrr"]) - 0.02
                for s in sizes
            ),
        ),
        (
            "the 0.5B model beats the extractive baseline on correct answers",
            ok(g05, "rag") > ex["correct_on_answerable"]["rate"],
        ),
        (
            "the 0.5B model without retrieval answers almost nothing correctly (under 5%)",
            g05["closed_book"]["summary"]["correct_on_answerable"]["rate"] < 0.05,
        ),
    ]
    demo = d["demo"]["meta"]
    corpus = d["retrieval"]["meta"]["corpus"]
    out.append(
        (
            "the demo's corpus counts match the corpus on disk",
            demo["corpus"]["documents"]
            == corpus["documents"]
            == len(list((CORPUS / "docs").glob("*.txt")))
            and demo["corpus"]["words"] == corpus["words"]
            and demo["corpus"]["chunks"] == len(d["demo"]["chunks"]),
        )
    )
    n_q = sum(
        1
        for _ in (CORPUS / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if _.strip()
    )
    out.append(
        (
            "the recorded question count matches questions.jsonl",
            d["retrieval"]["meta"]["questions"]["total"] == n_q,
        )
    )
    return out


# ------------------------------------------------------------------------------- driver ---------
def render_readme(text: str, blocks: dict) -> str:
    for name, body in blocks.items():
        pat = re.compile(
            rf"(<!-- BEGIN {re.escape(name)} -->\n)(.*?)(\n<!-- END {re.escape(name)} -->)", re.S
        )
        if pat.search(text):
            text = pat.sub(lambda m, body=body: m.group(1) + body + m.group(3), text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()
    d = load()
    bad = [name for name, ok in claims(d) if not ok]
    blocks = dynamic_blocks(d)
    text = README.read_text(encoding="utf-8")
    new = render_readme(text, blocks)
    ui = json.dumps(ui_results(d), indent=1, ensure_ascii=False) + "\n"
    ui_path = DOCS / "data" / "results.json"
    missing = [n for n in blocks if f"<!-- BEGIN {n} -->" not in text]
    if args.write:
        README.write_text(new, encoding="utf-8", newline="\n")
        ui_path.write_text(ui, encoding="utf-8", newline="\n")
        print("README blocks and docs/data/results.json written")
        if missing:
            print("note: README has no marker for:", ", ".join(missing))
    problems = []
    if args.check:
        if new != text:
            problems.append(
                "README.md is out of sync with results/ (run scripts/check_numbers.py --write)"
            )
        if not ui_path.exists() or ui_path.read_text(encoding="utf-8") != ui:
            problems.append(
                "docs/data/results.json is out of sync (run scripts/check_numbers.py --write)"
            )
        used = set(re.findall(r"<!-- BEGIN (\w+) -->", text))
        unknown = used - set(blocks)
        if unknown:
            problems.append("README has markers with no generator: " + ", ".join(sorted(unknown)))
    problems += [f"claim fails: {c}" for c in bad]
    for p in problems:
        print("FAIL:", p)
    if args.check and not problems:
        print("ok: README, docs/data/results.json and the stated claims agree with results/")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
