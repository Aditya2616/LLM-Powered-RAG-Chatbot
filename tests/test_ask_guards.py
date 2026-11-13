import json
import logging

import pytest

from askaditya.api import create_app
from askaditya.config import AskSettings
from askaditya.guards import enforce_caveats, guards_in_scope, sanitise_follow_up
from askaditya.index import AskIndex
from askaditya.limits import MemoryLimiter
from askaditya.pipeline import AskPipeline
from askaditya.providers import ProviderUnavailable, Unavailable
from tests.ask_fakes import ScriptedProvider, make_chain

# paraphrases per claim, in the shapes a visitor might use: the claim itself, a how-much question, a how-good one
PARAPHRASES = {
    "modelrouter": [
        "How much has ModelRouter saved in production?",
        "How much does ModelRouter save?",
        "Does ModelRouter reduce costs for live traffic?",
        "What savings does the model router give real users?",
    ],
    "promptlab": [
        "Is PromptLab better than random search?",
        "Does PromptLab's evolution beat random search?",
        "Which wins in PromptLab, evolution or the random baseline?",
        "How does PromptLab compare with random search?",
    ],
    "sftforge": [
        "Does SFT Forge's cleaned data train a better model?",
        "Will SFT Forge improve my fine-tuned model?",
        "How much does cleaning with SFT Forge help training?",
        "Is data from SFT Forge worth training on?",
    ],
    "sentinel": [
        "Does Sentinel stop all prompt injections?",
        "How accurate is Sentinel?",
        "Can Sentinel catch every attack?",
        "How well does Sentinel protect a chatbot?",
    ],
    "agentflow": [
        "Does AgentFlow have a high task success rate?",
        "How often does AgentFlow finish its tasks successfully?",
        "How reliable is AgentFlow?",
        "What is AgentFlow's success rate?",
    ],
    "evalbench": [
        "Is EvalBench's rubric judge highly reliable?",
        "Can I trust EvalBench's judge scores?",
        "How accurate is the EvalBench judge?",
        "Does the EvalBench judge agree with humans?",
    ],
}
# what a model says without the caveat: the sort of sentence that came back live
BARE = {
    "modelrouter": "He keeps 95% of the big model's quality while using only 37% of the cost.",
    "promptlab": "PromptLab finds strong prompts by evolving them and scoring each on a dev set.",
    "sftforge": "SFT Forge cleans instruction data through dedupe, quality and safety gates before training.",
    "sentinel": "Sentinel blocks 54% of the deepset attacks with 1% false positives.",
    "agentflow": "AgentFlow plans a task, calls tools and verifies each result before it moves on.",
    "evalbench": "EvalBench scores model outputs with a rubric judge and groups the failures by mode.",
}
REQUIRED_WORDS = {
    "modelrouter": "offline",
    "promptlab": "tie",
    "sftforge": "no measurable",
    "sentinel": "one layer",
    "agentflow": "does not claim",
    "evalbench": "does not claim",
}
CASES = [(k, q) for k, qs in PARAPHRASES.items() for q in qs]


@pytest.mark.parametrize(("key", "question"), CASES)
def test_every_paraphrase_puts_the_right_guard_in_scope(key, question):
    assert key in {g.trap.key for g in guards_in_scope(question, BARE[key])}


@pytest.mark.parametrize(("key", "question"), CASES)
def test_a_bare_answer_always_gets_the_caveat_or_the_true_statement(key, question):
    text, note = enforce_caveats(question, BARE[key] + " Ask me about his projects.")
    assert note and REQUIRED_WORDS[key] in text.lower()
    assert "Ask me about" in text  # the follow-up survives an added caveat


def test_an_answer_that_already_carries_the_caveat_is_left_alone():
    ans = "No. ModelRouter's cost figures come from an offline analysis of recorded RouterBench outcomes. Ask me about his projects."
    assert enforce_caveats("How much does ModelRouter save?", ans) == (ans, None)


def test_the_live_failure_is_fixed():
    live = "He keeps 95% of the big model's quality while using only 37% of the cost. Ask me about the cost savings of ModelRouter."
    text, _ = enforce_caveats("How much has ModelRouter saved in production?", live)
    text = sanitise_follow_up(text)
    assert "95%" in text and "offline" in text and "recorded" in text
    assert text.endswith("Ask me about his other projects.") and "cost savings" not in text


def test_a_factual_question_keeps_its_number_and_only_gains_the_caveat():
    ans = "Sentinel blocks 54% of the deepset attacks. Ask me about his projects."
    text, note = enforce_caveats("What share of the deepset attacks does Sentinel block?", ans)
    assert (
        "54%" in text
        and "one layer" in text
        and note == "caveat:sentinel"
        and not text.startswith("No.")
    )


def test_an_answer_that_asserts_the_claim_is_replaced():
    text, note = enforce_caveats(
        "Does Sentinel stop all prompt injections?", "Sentinel blocks all attacks that people try."
    )
    assert text.startswith("No.") and note == "replaced:sentinel"


def test_an_unrelated_answer_is_never_touched():
    ans = "He works at Ethara AI as an LLM Post-Training Intern. Ask me about his projects."
    assert enforce_caveats("Where does he work?", ans) == (ans, None)
    assert sanitise_follow_up(ans) == ans


@pytest.mark.parametrize(
    "follow",
    [
        "Ask me about the cost savings of ModelRouter.",
        "Ask me about how reliable Sentinel is.",
        "Ask me about AgentFlow's success rate.",
        "Ask me about whether PromptLab is better than random search.",
        "Ask me about SFT Forge's effect on training.",
        "Ask me about EvalBench's judge accuracy.",
    ],
)
def test_follow_ups_never_invite_a_false_claim(follow):
    out = sanitise_follow_up(f"He built it. {follow}")
    assert out.endswith("Ask me about his other projects.") and follow not in out


def test_neutral_follow_ups_survive():
    assert sanitise_follow_up("He built it. Ask me about his evaluation work.").endswith(
        "his evaluation work."
    )


# ------------------------------------------------------ the pipeline, with a model that ignores the caveat

ROUTER_CHUNKS = [
    {
        "id": 0,
        "title": "ModelRouter README",
        "url": "https://github.com/Aditya2616/ModelRouter",
        "section": "ModelRouter",
        "text": "On RouterBench the router keeps 95% of the big model's quality at 37% of the cost. The figures come from an offline analysis of recorded outcomes.",
    }
]


def test_the_pipeline_adds_the_caveat_when_the_model_leaves_it_out():
    reply = "He keeps 95% of the big model's quality at 37% of the cost. Ask me about the cost savings of ModelRouter."
    pipe = AskPipeline(
        AskIndex(ROUTER_CHUNKS, "t"),
        make_chain([ScriptedProvider("fake", [reply])]),
        "me@example.test",
    )
    r = pipe.answer("How much has ModelRouter saved in production?")
    assert r.grounded and "offline" in r.answer.lower() and "cost savings" not in r.answer
    assert "offline" in pipe.answer("How much does ModelRouter save?").answer.lower()


# ---------------------------------------------------------------- reasons, retry-after and logging


class Clock:
    t = 1_800_000_030.0

    def __call__(self):
        return self.t


def test_unavailable_carries_reason_codes_and_a_retry_after_for_minute_budgets():
    s = AskSettings(token_caps_minute={"groq": 10, "groq-backup": 10}, token_caps_day={})
    chain = make_chain(
        [ScriptedProvider("groq", ["x"]), ScriptedProvider("groq-backup", ["x"])],
        s,
        MemoryLimiter(Clock()),
    )
    with pytest.raises(Unavailable) as exc:
        chain.generate("s", "u" * 100)
    assert exc.value.attempts == ["groq: budget_tokens_minute", "groq-backup: budget_tokens_minute"]
    assert 1 <= exc.value.retry_after <= 60


def test_failures_are_coded_and_a_daily_block_waits_for_the_next_day():
    s = AskSettings(token_caps_day={"groq": 10}, token_caps_minute={})
    gemini = ScriptedProvider("gemini", [ProviderUnavailable("gemini: 429", "rate_limit")])
    chain = make_chain([ScriptedProvider("groq", ["x"]), gemini], s)
    with pytest.raises(Unavailable) as exc:
        chain.generate("s", "u" * 100)
    assert exc.value.attempts == ["groq: budget_tokens_day", "gemini: 429"]
    assert exc.value.retry_after >= 1


def test_a_503_and_a_fallthrough_are_logged_without_question_or_keys(capsys):
    from fastapi.testclient import TestClient

    secret_question = "Where does he work at Northwind Labs? zebra-unique-token"
    cfg = AskSettings(token_caps_day={})
    chunks = [
        {
            "id": 0,
            "title": "Portfolio",
            "url": "https://x.test",
            "section": "About",
            "text": "Sam works at Northwind Labs as a Data Engineer.",
        }
    ]
    fail = ProviderUnavailable("groq: 429", "rate_limit")

    lim = MemoryLimiter()
    chain = make_chain(
        [
            ScriptedProvider("groq", [fail]),
            ScriptedProvider("gemini", ["Sam works at Northwind Labs. Ask me about his projects."]),
        ],
        cfg,
        lim,
    )
    ok = TestClient(create_app(cfg, index=AskIndex(chunks, "t"), chain=chain, limiter=lim)).post(
        "/api/ask", json={"question": secret_question}
    )
    assert ok.status_code == 200 and ok.headers["x-ask-provider"] == "gemini"

    lim2 = MemoryLimiter()
    chain2 = make_chain([ScriptedProvider("groq", [fail])], cfg, lim2)
    bad = TestClient(create_app(cfg, index=AskIndex(chunks, "t"), chain=chain2, limiter=lim2)).post(
        "/api/ask", json={"question": secret_question}
    )
    assert bad.status_code == 503

    err = capsys.readouterr().err
    lines = [json.loads(line)["ask"] for line in err.splitlines() if line.startswith('{"ask"')]
    assert any(
        e["status"] == 200 and e["attempts"] == ["groq: 429"] and e["provider"] == "gemini"
        for e in lines
    )
    assert any(e["status"] == 503 and e["attempts"] == ["groq: 429"] for e in lines)
    assert "zebra-unique-token" not in err and "Northwind" not in err
    logging.getLogger("askaditya").handlers.clear()
