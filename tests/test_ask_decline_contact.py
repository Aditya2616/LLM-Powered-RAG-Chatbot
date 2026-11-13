import httpx
import pytest

from askaditya import safety
from askaditya.build import phone_like
from askaditya.config import AskSettings
from askaditya.index import AskIndex
from askaditya.pipeline import AskPipeline
from askaditya.providers import ProviderUnavailable, Unavailable, _duration_s, _retry_after
from tests.ask_fakes import ScriptedProvider, make_chain
from tests.test_ask_budgets import Clock, MemoryLimiter, Metered, ProviderChain

GOOD = "Sam works at Northwind Labs as a Data Engineer since Mar 2024. Ask me about his projects."


@pytest.fixture(scope="module")
def index(persona_dir):
    return AskIndex.load(persona_dir)


def pipe(index, *providers):
    return AskPipeline(index, make_chain(list(providers)), "sam.rivera@example.test")


# ------------------------------------------------------------------------------ 1. decline retry
def test_a_decline_on_an_answerable_question_goes_to_the_next_provider_once(index):
    a, b = ScriptedProvider("groq", ["UNKNOWN"]), ScriptedProvider("groq-backup", [GOOD])
    r = pipe(index, a, b).answer("Where does Sam work now?")
    assert r.grounded and "Northwind" in r.answer and r.provider == "groq-backup"
    assert len(a.calls) == 1 and len(b.calls) == 1
    assert "decline_retry" in r.reason and "groq: declined" in r.attempts


def test_a_second_decline_is_kept_and_costs_one_extra_call_at_most(index):
    a, b, c = (ScriptedProvider(n, ["UNKNOWN"]) for n in ("groq", "groq-backup", "gemini"))
    r = pipe(index, a, b, c).answer("Where does Sam work now?")
    assert r.fallback == "unknown" and "decline_retry" in r.reason
    assert (len(a.calls), len(b.calls), len(c.calls)) == (1, 1, 0)


def test_no_other_provider_keeps_the_decline(index):
    a = ScriptedProvider("groq", ["UNKNOWN"])
    r = pipe(index, a).answer("Where does Sam work now?")
    assert (
        r.fallback == "unknown" and len(a.calls) == 2
    )  # the first answer and one same-model retry
    assert "decline_retry_same" in r.reason


def test_the_same_model_retry_uses_the_other_wording_and_can_rescue_the_answer(index):
    a = ScriptedProvider("groq", ["UNKNOWN", GOOD])
    r = pipe(index, a).answer("Where does Sam work now?")
    assert r.grounded and "Northwind" in r.answer and "decline_retry_same" in r.reason
    assert len(a.calls) == 2 and a.calls[0][0] != a.calls[1][0]
    assert "reply UNKNOWN only if none" in a.calls[1][0]


def test_a_question_the_passages_do_not_cover_is_not_retried(index):
    a, b = ScriptedProvider("groq", ["UNKNOWN"]), ScriptedProvider("groq-backup", [GOOD])
    r = pipe(index, a, b).answer("What is Sam's favourite zebra species?")
    assert r.fallback in ("unknown", "offtopic") and len(b.calls) == 0


def test_the_retry_is_counted_in_the_budgets(index):
    s = AskSettings(global_daily_cap=2, provider_daily_caps={})
    a, b = ScriptedProvider("groq", ["UNKNOWN"]), ScriptedProvider("groq-backup", [GOOD])
    p = AskPipeline(index, ProviderChain([a, b], MemoryLimiter(), s), "x@example.test")
    assert p.answer(
        "Where does Sam work now?"
    ).grounded  # two model calls: the decline and the retry
    with pytest.raises(Unavailable):
        p.answer("Where does Sam work now?")  # the global cap of 2 is spent


# ---------------------------------------------------------------------- 2. contact, no model
@pytest.mark.parametrize(
    "q",
    [
        "How can I contact Sam?",
        "What is his email address?",
        "Where can I find his LinkedIn?",
        "Link to his résumé please",
        "How do I reach him?",
        "Give me his GitHub profile",
    ],
)
def test_contact_questions_never_reach_a_model(index, q):
    assert safety.is_contact_question(q)
    a = ScriptedProvider("groq", ["UNKNOWN"])
    r = pipe(index, a).answer(q)
    assert r.stage == "contact" and r.grounded and not a.calls
    assert "sam.rivera@example.test" in r.answer
    assert r.sources and r.sources[0]["title"] == "Portfolio"
    assert not phone_like(r.answer)


@pytest.mark.parametrize(
    "q",
    [
        "What does Sam's email automation project do?",
        "Which projects use GitHub Actions?",
        "Where does he work?",
    ],
)
def test_other_questions_are_not_taken_for_contact_questions(q):
    assert not safety.is_contact_question(q)


def test_a_phone_request_is_still_deflected_before_the_contact_stage(index):
    r = pipe(index).answer("What is his phone number to contact him?")
    assert r.stage == "topic"


# ------------------------------------------------------------------------------ 3. cooldown
@pytest.mark.parametrize(
    "text,seconds",
    [
        ("12", 12),
        ("7.66s", 7.66),
        ("1m5.2s", 65.2),
        ("250ms", 0.25),
        ("2h", 7200),
        ("", None),
        ("soon", None),
    ],
)
def test_duration_parsing(text, seconds):
    got = _duration_s(text)
    assert (got is None) == (seconds is None) and (got is None or abs(got - seconds) < 1e-6)


def resp(**headers):
    return httpx.Response(429, headers=headers)


def test_the_cooldown_uses_retry_after_then_the_reset_headers():
    assert _retry_after(resp(**{"retry-after": "12"}), 99) == 13
    assert _retry_after(resp(**{"x-ratelimit-reset-tokens": "7.66s"}), 99) == 8
    assert (
        _retry_after(resp(**{"retry-after": "x", "x-ratelimit-reset-tokens": "1m5.2s"}), 99) == 66
    )
    assert _retry_after(resp(), 20) == 20


def test_the_primary_comes_back_when_the_window_the_provider_named_frees():
    clock = Clock()
    s = AskSettings(
        provider_daily_caps={}, token_caps_minute={}, token_caps_day={}, global_daily_cap=10**9
    )
    limited = ScriptedProvider("groq", [ProviderUnavailable("groq: 429", "rate_limit", 13), "back"])
    chain = ProviderChain([limited, Metered("groq-backup")], MemoryLimiter(clock), s)
    gen = chain.generate("s", "u")
    assert gen.provider == "groq-backup" and "groq: cooldown_set 13s" in gen.attempts
    clock.t += 8
    assert chain.generate("s", "u").provider == "groq-backup"  # still cooling down
    clock.t += 6  # 14 s after the 429
    assert chain.generate("s", "u").provider == "groq"
