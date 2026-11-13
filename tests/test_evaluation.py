import pytest

from rag_chatbot.evaluation import (
    Span,
    answer_correct,
    chunk_overlaps_gold,
    grade_answer,
    load_questions,
    percentile,
    resolve_spans,
    score_retrieval,
    summarise_answers,
    unsupported_numbers,
    wilson,
)
from tests.conftest import CORPUS, QUESTIONS

DOCS = {p.name: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.txt"))}


def test_question_set_shape():
    qs = load_questions(QUESTIONS)
    assert len(qs) == 100
    assert sum(q.answerable for q in qs) == 75
    assert len({q.id for q in qs}) == 100
    assert {q.kind for q in qs if q.answerable} == {"direct", "paraphrase"}
    assert {q.kind for q in qs if not q.answerable} == {"near_miss", "off_topic"}


def test_every_gold_passage_exists_once_and_contains_its_answer():
    for q in load_questions(QUESTIONS):
        if not q.answerable:
            assert not q.evidence and not q.answer_groups
            continue
        spans = resolve_spans(q, DOCS)
        assert spans, q.id
        evidence = " ".join(e.text for e in q.evidence)
        # a gold passage must be enough to answer: every required fact is in it (the second
        # span of a question stated twice is checked on its own as well)
        assert answer_correct(evidence, q.answer_groups), f"{q.id}: facts missing from evidence"
        for e in q.evidence:
            assert answer_correct(e.text, q.answer_groups), f"{q.id}: {e.doc}"


def test_chunk_overlap_rule():
    spans = [Span("a.txt", 100, 300)]  # 200 characters: needs min(100, 60) = 60 characters
    assert chunk_overlaps_gold("a.txt", 0, 200, spans)
    assert not chunk_overlaps_gold("a.txt", 0, 150, spans)
    assert not chunk_overlaps_gold("b.txt", 0, 400, spans)
    short = [Span("a.txt", 100, 130)]  # 30 characters: needs 15
    assert chunk_overlaps_gold("a.txt", 115, 500, short)
    assert not chunk_overlaps_gold("a.txt", 120, 500, short)


def test_answer_correct_requires_every_group():
    groups = [["30 days"], ["peakfreight"]]
    assert answer_correct("PeakFreight collects it within 30 days", groups)
    assert not answer_correct("within 30 days", groups)
    assert not answer_correct("anything", [])


def test_unsupported_numbers():
    assert unsupported_numbers("It takes 7 working days", "Refunds take 5 working days") == ["7"]
    assert unsupported_numbers("5 working days", "Refunds take 5 working days") == []


def test_score_retrieval_on_a_toy_case():
    qs = load_questions(QUESTIONS)[:2]
    spans = {q.id: resolve_spans(q, DOCS) for q in qs}
    s0 = spans[qs[0].id][0]
    s1 = spans[qs[1].id][0]
    ranked = {
        qs[0].id: [
            ("x.txt", 0, 10, "nothing"),
            (s0.doc, s0.start, s0.end, DOCS[s0.doc][s0.start : s0.end]),
        ],
        qs[1].id: [("x.txt", 0, 10, "nothing")],
    }
    r = score_retrieval(qs, ranked, spans, [1, 2])
    assert r.n == 2
    assert r.recall == {1: 0.0, 2: 0.5}
    assert r.mrr == pytest.approx(0.25)
    assert r.answer_in_context[2] == 0.5
    assert s1.doc  # second question has a gold span but nothing retrieved


def test_wilson_interval_properties():
    lo, hi = wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo < 1.0


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.5) == 0.0
    assert percentile([5], 0.95) == 5


def test_summary_counts_add_up():
    qs = load_questions(QUESTIONS)
    ans = [q for q in qs if q.answerable][:4]
    una = [q for q in qs if not q.answerable][:2]
    recs = [
        grade_answer(ans[0], " ".join(g[0] for g in ans[0].answer_groups), False, "ctx"),
        grade_answer(ans[1], "wrong", False, "ctx"),
        grade_answer(ans[2], "I don't know", True, "ctx"),
        grade_answer(ans[3], " ".join(g[0] for g in ans[3].answer_groups), False, "ctx"),
        grade_answer(una[0], "I don't know", True, "ctx"),
        grade_answer(una[1], "made up 99", False, "ctx"),
    ]
    s = summarise_answers(recs)
    assert (s["n"], s["n_answerable"], s["n_unanswerable"]) == (6, 4, 2)
    c, f, w = (
        s[k]["k"]
        for k in (
            "correct_on_answerable",
            "false_abstention_on_answerable",
            "wrong_answer_on_answerable",
        )
    )
    assert (c, f, w) == (2, 1, 1) and c + f + w == s["n_answerable"]
    assert s["abstained_on_unanswerable"]["k"] + s["answered_unanswerable"]["k"] == 2
    assert s["decision_accuracy"]["k"] == 3
    assert s["with_unsupported_number"]["k"] >= 1
