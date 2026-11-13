"""Metrics for the bundled question set: retrieval, answer correctness, groundedness, abstention.

Everything here is mechanical and deterministic. Answer correctness is a lexical check (the answer
must contain every required fact, normalised), which is strict about wording and blind to meaning;
the README says so next to the numbers.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .text import contains_phrase, normalise, token_support, tokens


@dataclass(frozen=True)
class Evidence:
    doc: str
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answerable: bool
    kind: str  # direct | paraphrase | near_miss | off_topic
    evidence: tuple[Evidence, ...] = ()
    answer_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class Span:
    doc: str
    start: int
    end: int


def load_questions(path: Path) -> list[Question]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(
            Question(
                id=r["id"],
                question=r["question"],
                answerable=r["answerable"],
                kind=r["kind"],
                evidence=tuple(Evidence(e["doc"], e["text"]) for e in r["evidence"]),
                answer_groups=tuple(tuple(g) for g in r["answer_groups"]),
            )
        )
    return out


def resolve_spans(q: Question, docs: dict[str, str]) -> list[Span]:
    """Character spans of the gold evidence inside the documents. Each must occur exactly once."""
    spans = []
    for e in q.evidence:
        text = docs[e.doc]
        if text.count(e.text) != 1:
            raise ValueError(f"{q.id}: evidence must occur exactly once in {e.doc}")
        i = text.index(e.text)
        spans.append(Span(e.doc, i, i + len(e.text)))
    return spans


def chunk_overlaps_gold(source: str, start: int, end: int, spans: Sequence[Span]) -> bool:
    """A chunk counts as holding the gold passage when it covers at least half of it, or at least 60
    characters of it, whichever is smaller. (Plain "any overlap" would credit a chunk that only shares
    one overlapping word with the passage.)"""
    for s in spans:
        if s.doc != source:
            continue
        overlap = min(end, s.end) - max(start, s.start)
        if overlap >= min(0.5 * (s.end - s.start), 60):
            return True
    return False


def answer_correct(answer: str, groups: Sequence[Sequence[str]]) -> bool:
    """Every group must be satisfied by at least one of its alternatives (whole-word, normalised)."""
    return bool(groups) and all(any(contains_phrase(answer, alt) for alt in g) for g in groups)


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def unsupported_numbers(answer: str, context: str) -> list[str]:
    ctx = set(tokens(context))
    return [n for n in _NUMBER.findall(normalise(answer)) if n not in ctx]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (as fractions)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class RetrievalScore:
    n: int
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    answer_in_context: dict[int, float] = field(default_factory=dict)


def score_retrieval(
    questions: Iterable[Question],
    ranked: dict[str, list[tuple[str, int, int, str]]],
    spans: dict[str, list[Span]],
    ks: Sequence[int],
) -> RetrievalScore:
    """``ranked[qid]`` is the retrieved chunks, best first, as ``(source, start, end, text)``.

    * recall@k: some chunk in the top k holds the gold passage (see ``chunk_overlaps_gold``).
    * MRR: mean over questions of 1 / rank of the first such chunk (0 if none in the list).
    * answer-in-context@k: the top-k chunks together contain every required answer fact. Unlike
      recall it does not depend on where chunk boundaries fall, so it is comparable across chunk sizes.
    """
    qs = [q for q in questions if q.answerable]
    rec = dict.fromkeys(ks, 0)
    aic = dict.fromkeys(ks, 0)
    rr = 0.0
    for q in qs:
        hits = ranked[q.id]
        first = next(
            (
                i + 1
                for i, (s, a, b, _) in enumerate(hits)
                if chunk_overlaps_gold(s, a, b, spans[q.id])
            ),
            None,
        )
        if first:
            rr += 1.0 / first
        for k in ks:
            if first and first <= k:
                rec[k] += 1
            if answer_correct(" ".join(t for *_, t in hits[:k]), q.answer_groups):
                aic[k] += 1
    n = len(qs)
    return RetrievalScore(n, {k: rec[k] / n for k in ks}, rr / n, {k: aic[k] / n for k in ks})


def summarise_answers(records: Sequence[dict]) -> dict:
    """Roll per-question generation records up into the headline rates (with counts and CIs).

    A record has: answerable, kind, abstained, correct, support, unsupported_numbers.
    """
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    given = [r for r in records if not r["abstained"]]
    out: dict = {"n": len(records), "n_answerable": len(ans), "n_unanswerable": len(una)}

    def rate(k: int, n: int) -> dict:
        lo, hi = wilson(k, n)
        return {"k": k, "n": n, "rate": k / n if n else 0.0, "ci": [lo, hi]}

    out["correct_on_answerable"] = rate(sum(r["correct"] for r in ans), len(ans))
    out["false_abstention_on_answerable"] = rate(sum(r["abstained"] for r in ans), len(ans))
    out["wrong_answer_on_answerable"] = rate(
        sum((not r["abstained"]) and (not r["correct"]) for r in ans), len(ans)
    )
    out["abstained_on_unanswerable"] = rate(sum(r["abstained"] for r in una), len(una))
    out["answered_unanswerable"] = rate(sum(not r["abstained"] for r in una), len(una))
    for kind in sorted({r["kind"] for r in una}):
        sub = [r for r in una if r["kind"] == kind]
        out[f"abstained_on_{kind}"] = rate(sum(r["abstained"] for r in sub), len(sub))
    for kind in sorted({r["kind"] for r in ans}):
        sub = [r for r in ans if r["kind"] == kind]
        out[f"correct_on_{kind}"] = rate(sum(r["correct"] for r in sub), len(sub))
    total_ok = sum(r["correct"] for r in ans) + sum(r["abstained"] for r in una)
    out["decision_accuracy"] = rate(total_ok, len(records))
    out["mean_support"] = sum(r["support"] for r in given) / len(given) if given else 0.0
    out["grounded"] = rate(sum(r["support"] >= 0.8 for r in given), len(given))
    out["with_unsupported_number"] = rate(
        sum(bool(r["unsupported_numbers"]) for r in given), len(given)
    )
    return out


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = (len(s) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def grade_answer(q: Question, answer: str, abstained: bool, context: str) -> dict:
    """One per-question record for ``summarise_answers``."""
    return {
        "id": q.id,
        "kind": q.kind,
        "answerable": q.answerable,
        "answer": answer,
        "abstained": abstained,
        "correct": (not abstained) and q.answerable and answer_correct(answer, q.answer_groups),
        "support": 1.0 if abstained else token_support(answer, context),
        "unsupported_numbers": [] if abstained else unsupported_numbers(answer, context),
    }
