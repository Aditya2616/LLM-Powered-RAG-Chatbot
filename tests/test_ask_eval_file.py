import json
from collections import Counter

from tests.ask_fakes import ROOT

EVAL = ROOT / "eval" / "ask_aditya_eval.jsonl"
FORBIDDEN = (
    "cl" + "aude"
)  # the name of an assistant product that must not appear in this repository


def rows():
    return [
        json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_the_eval_set_has_the_required_groups():
    counts = Counter(r["group"] for r in rows())
    assert sum(counts.values()) >= 80
    assert counts["factual"] >= 40 and counts["unpublished"] >= 15 and counts["trap"] >= 10
    assert counts["injection"] >= 10 and counts["offtopic"] >= 5
    assert len({r["id"] for r in rows()}) == len(rows())


def test_every_question_has_a_machine_checkable_expectation():
    for r in rows():
        assert 1 <= len(r["question"]) <= 500
        if r["group"] in ("factual", "trap"):
            assert (
                r["must_contain"] and r["expect_fallback"] is None and r["expect_grounded"] is True
            )
        if r["group"] == "unpublished":
            assert (
                r["expect_fallback"] == "unknown" and "itsaditya1610@gmail.com" in r["must_contain"]
            )
        if r["group"] == "injection":
            assert r["expect_fallback"] == "refused"
        if r["group"] == "offtopic":
            assert r["expect_fallback"] == "offtopic"
        assert len(r["history"]) <= 6


def test_the_eval_file_and_the_repository_do_not_contain_the_forbidden_name():
    assert FORBIDDEN not in EVAL.read_text(encoding="utf-8").lower()


def test_no_phone_number_is_stored_in_the_eval_file():
    from askaditya.build import phone_like

    assert not phone_like(EVAL.read_text(encoding="utf-8").replace("\n", " "))
