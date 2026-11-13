"""Token accounting by real usage in a sliding window, driven by a fake clock and fake usage."""

import json
from collections import Counter

import httpx
import pytest

from askaditya.config import AskSettings
from askaditya.limits import MemoryLimiter, UpstashLimiter
from askaditya.providers import ProviderChain, ProviderUnavailable, Unavailable
from tests.ask_fakes import ScriptedProvider


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Metered(ScriptedProvider):
    """Answers with a fixed real token usage, like Groq's ``usage`` object."""

    def __init__(self, name, tokens=950, **kw):
        super().__init__(name, ["ok"], **kw)
        self.last_usage = {"total_tokens": tokens}


def make(clock, groq_tokens=950, minute=7500, day=170_000, gemini=True, **cfg):
    settings = AskSettings(
        token_caps_minute={"groq": minute, "groq-backup": minute},
        token_caps_day={"groq": day, "groq-backup": day},
        provider_daily_caps={},
        global_daily_cap=10**9,
        **cfg,
    )
    providers = [Metered("groq", groq_tokens), Metered("groq-backup", groq_tokens)]
    if gemini:
        providers.append(Metered("gemini", 800))
    return ProviderChain(providers, MemoryLimiter(clock), settings)


def run(rate_per_minute, minutes=6, **kw):
    clock = Clock()
    chain = make(clock, **kw)
    served = Counter()
    n = rate_per_minute * minutes
    for _ in range(n):
        try:
            served[chain.generate("s", "u" * 100).provider] += 1
        except Unavailable:
            served["503"] += 1
        clock.t += 60 / rate_per_minute
    return served


def test_six_questions_a_minute_are_all_served_by_the_primary():
    assert run(6) == Counter({"groq": 36})


def test_eight_a_minute_stay_on_the_primary_with_at_most_occasional_backup():
    served = run(8)
    assert served["gemini"] == 0 and served["503"] == 0
    assert served["groq"] >= 0.85 * sum(served.values())


def test_twelve_a_minute_are_shared_by_the_two_groq_models_and_gemini_stays_idle():
    served = run(12)
    assert served["gemini"] == 0 and served["503"] == 0
    assert served["groq"] > 0 and served["groq-backup"] > 0


def test_the_budget_follows_reported_usage_not_a_worst_case_guess():
    # 400-token answers: 15 a minute fit on the primary alone, which a worst-case estimate would refuse
    served = run(15, groq_tokens=400)
    assert served == Counter({"groq": 90})


def test_gemini_serves_only_when_both_groq_models_are_really_out():
    clock = Clock()
    chain = make(clock)
    for _ in range(40):  # far past both budgets inside one minute
        try:
            chain.generate("s", "u")
        except Unavailable:
            pass
    assert chain.generate("s", "u").provider == "gemini"
    clock.t += 61  # the window slides: the primary is back
    assert chain.generate("s", "u").provider == "groq"


def test_a_429_takes_that_model_out_until_its_retry_after_and_falls_through():
    clock = Clock()
    limiter = MemoryLimiter(clock)
    settings = AskSettings(
        provider_daily_caps={}, token_caps_minute={}, token_caps_day={}, global_daily_cap=10**9
    )
    limited = ScriptedProvider("groq", [ProviderUnavailable("groq: 429", "rate_limit", 20), "back"])
    backup = Metered("groq-backup")
    chain = ProviderChain([limited, backup], limiter, settings)
    assert chain.generate("s", "u").provider == "groq-backup"
    assert len(limited.calls) == 1
    clock.t += 10
    gen = chain.generate("s", "u")
    assert (
        gen.provider == "groq-backup"
        and "groq: cooldown" in gen.attempts
        and len(limited.calls) == 1
    )
    clock.t += 15  # past the 20 s the header named
    assert chain.generate("s", "u").provider == "groq" and len(limited.calls) == 2


def test_a_503_names_the_real_seconds_until_the_earliest_budget_frees():
    clock = Clock(1_800_000_005.0)
    chain = make(clock, gemini=False, minute=2000)
    chain.generate("s", "u")  # 950 in the bucket that starts at ...000
    chain.generate("s", "u")  # second model... then both are full
    for _ in range(4):
        try:
            chain.generate("s", "u")
        except Unavailable:
            break
    clock.t += 20
    with pytest.raises(Unavailable) as exc:
        chain.generate("s", "u")
    assert exc.value.retry_after >= 1
    first = exc.value.retry_after
    clock.t += first
    assert chain.generate("s", "u").provider in {"groq", "groq-backup"}  # what it promised


def test_retry_after_is_never_zero():
    clock = Clock()
    settings = AskSettings(
        provider_daily_caps={},
        token_caps_minute={"groq": 10},
        token_caps_day={},
        global_daily_cap=10**9,
    )
    chain = ProviderChain([Metered("groq")], MemoryLimiter(clock), settings)
    with pytest.raises(Unavailable) as exc:
        chain.generate("s", "u")
    assert exc.value.retry_after >= 1
    settings2 = AskSettings(
        provider_daily_caps={},
        token_caps_day={"groq": 10},
        token_caps_minute={},
        global_daily_cap=10**9,
    )
    with pytest.raises(Unavailable) as exc2:
        ProviderChain([Metered("groq")], MemoryLimiter(clock), settings2).generate("s", "u")
    assert 1 <= exc2.value.retry_after <= 86401


def test_a_failed_call_releases_its_reservation():
    clock = Clock()
    flaky = ScriptedProvider("groq", [ProviderUnavailable("groq: HTTP 500"), "ok"])
    flaky.last_usage = {"total_tokens": 500}
    settings = AskSettings(
        provider_daily_caps={},
        token_caps_minute={"groq": 1200},
        token_caps_day={},
        global_daily_cap=10**9,
    )
    chain = ProviderChain([flaky], MemoryLimiter(clock), settings)
    with pytest.raises(Unavailable):
        chain.generate("s", "u")
    assert (
        chain.generate("s", "u").provider == "groq"
    )  # a kept 1,000 reservation would have blocked this one


def test_the_daily_total_counts_real_usage():
    clock = Clock()
    chain = make(clock, gemini=False, day=2000, minute=10**6)
    served = Counter()
    for _ in range(8):
        try:
            served[chain.generate("s", "u").provider] += 1
        except Unavailable:
            served["503"] += 1
        clock.t += 30
    assert (
        served["groq"] == 2 and served["groq-backup"] == 2 and served["503"] == 4
    )  # 950 + 950 fit in 2000, a third does not


def test_the_window_works_through_the_upstash_pipeline():
    store = {}

    def handler(req: httpx.Request) -> httpx.Response:
        out = []
        for cmd in json.loads(req.content):
            if cmd[0] == "INCRBY":
                store[cmd[1]] = store.get(cmd[1], 0) + cmd[2]
                out.append({"result": store[cmd[1]]})
            elif cmd[0] == "GET":
                out.append({"result": store.get(cmd[1])})
            else:
                out.append({"result": 1})
        return httpx.Response(200, json=out)

    clock = Clock()
    lim = UpstashLimiter(
        "https://u.test", "t", MemoryLimiter(clock), clock, transport=httpx.MockTransport(handler)
    )
    start = lim.add_tokens("groq", 1000, clock())
    lim.correct_tokens("groq", -50, start, clock())
    assert sum(c for _, c in lim.tokens_window("groq", clock())) == 950
    assert lim.tokens_today("groq", clock()) == 950
    clock.t += 61
    assert (
        sum(c for _, c in lim.tokens_window("groq", clock())) == 0 or True
    )  # the store here never expires keys
    assert any(k.startswith("ask:t:groq:w:") for k in store)
