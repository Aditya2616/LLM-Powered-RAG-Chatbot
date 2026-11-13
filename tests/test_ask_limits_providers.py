import json

import httpx
import pytest

from askaditya.config import AskSettings
from askaditya.limits import MemoryLimiter, UpstashLimiter, make_limiter
from askaditya.providers import (
    GeminiProvider,
    GroqProvider,
    ProviderUnavailable,
    Unavailable,
    build_chain,
)
from tests.ask_fakes import ScriptedProvider, make_chain


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


# -------------------------------------------------------------------------------------- limiter


def test_per_minute_limit_blocks_the_eleventh_request_and_resets():
    clock = Clock()
    lim = MemoryLimiter(clock)
    results = [lim.check_ip("1.2.3.4", 10, 60)[0] for _ in range(11)]
    assert results == [True] * 10 + [False]
    allowed, retry = lim.check_ip("1.2.3.4", 10, 60)
    assert not allowed and 1 <= retry <= 61
    clock.t += 61
    assert lim.check_ip("1.2.3.4", 10, 60)[0]


def test_per_day_limit_and_independent_addresses():
    clock = Clock()
    lim = MemoryLimiter(clock)
    ok = 0
    for _ in range(70):
        clock.t += 61  # a new minute every time, so only the daily limit can stop us
        ok += lim.check_ip("9.9.9.9", 10, 60)[0]
    assert ok == 60
    assert lim.check_ip("8.8.8.8", 10, 60)[0]


def test_daily_counters_roll_over_at_midnight_utc():
    clock = Clock(1_800_000_000.0)
    lim = MemoryLimiter(clock)
    assert all(lim.take_global(3) for _ in range(3)) and not lim.take_global(3)
    clock.t += 86400
    assert lim.take_global(3)


def test_provider_budgets_are_separate():
    lim = MemoryLimiter(Clock())
    assert lim.take_provider("groq", 1) and not lim.take_provider("groq", 1)
    assert lim.take_provider("gemini", 1)


def test_addresses_are_hashed_in_keys():
    lim = MemoryLimiter(Clock())
    lim.check_ip("203.0.113.7", 10, 60)
    assert all("203.0.113.7" not in k for k in lim._d)


def test_upstash_limiter_uses_the_rest_pipeline():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((str(req.url), req.headers["authorization"], json.loads(req.content)))
        return httpx.Response(200, json=[{"result": 3}, {"result": 1}])

    lim = UpstashLimiter(
        "https://u.example.test/", "tok", MemoryLimiter(), transport=httpx.MockTransport(handler)
    )
    assert lim.hit("g:2026-01-01", 90000) == 3
    url, auth, body = seen[0]
    assert url == "https://u.example.test/pipeline" and auth == "Bearer tok"
    assert body == [["INCRBY", "ask:g:2026-01-01", 1], ["EXPIRE", "ask:g:2026-01-01", 90000, "NX"]]
    assert lim.mode == "upstash"


def test_upstash_failure_falls_back_to_memory_counting():
    lim = UpstashLimiter(
        "https://u.example.test",
        "tok",
        MemoryLimiter(),
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    assert [lim.hit("k", 60) for _ in range(3)] == [1, 2, 3]


def test_make_limiter_picks_the_backend_from_the_environment():
    assert make_limiter(AskSettings()).mode == "memory"
    s = AskSettings.from_env(
        {"UPSTASH_REDIS_REST_URL": "https://u.test", "UPSTASH_REDIS_REST_TOKEN": "t"}
    )
    assert make_limiter(s).mode == "upstash"
    s = AskSettings.from_env({"KV_REST_API_URL": "https://u.test", "KV_REST_API_TOKEN": "t"})
    assert make_limiter(s).mode == "upstash"


# --------------------------------------------------------------------------------------- settings


def test_no_model_id_has_a_default_and_unconfigured_providers_are_skipped():
    s = AskSettings()
    assert s.groq_model == "" and s.gemini_model == "" and s.groq_api_key == ""
    assert build_chain(s, MemoryLimiter()).providers == []
    s = AskSettings.from_env({"GROQ_API_KEY": "k"})  # a key without a model id is not enough
    assert build_chain(s, MemoryLimiter()).providers == []
    s = AskSettings.from_env(
        {"GROQ_API_KEY": "k", "GROQ_MODEL": "m1", "GEMINI_API_KEY": "g", "GEMINI_MODEL": "m2"}
    )
    assert build_chain(s, MemoryLimiter()).names == ["groq", "gemini"]
    s = AskSettings.from_env(
        {
            "ASK_PROVIDER_ORDER": "gemini,groq",
            "GROQ_API_KEY": "k",
            "GROQ_MODEL": "m1",
            "GEMINI_API_KEY": "g",
            "GEMINI_MODEL": "m2",
        }
    )
    assert build_chain(s, MemoryLimiter()).names == ["gemini", "groq"]


def test_limits_come_from_the_environment():
    s = AskSettings.from_env(
        {
            "ASK_IP_PER_MINUTE": "3",
            "ASK_IP_PER_DAY": "7",
            "ASK_GLOBAL_DAILY_CAP": "9",
            "ASK_PROVIDER_DAILY_CAPS": "groq=5,gemini=6",
            "ASK_TEST_HOOKS": "1",
        }
    )
    assert (s.ip_per_minute, s.ip_per_day, s.global_daily_cap, s.test_hooks) == (3, 7, 9, True)
    assert s.provider_daily_caps == {"groq": 5, "groq-backup": 150, "gemini": 6}
    assert AskSettings.from_env({}).test_hooks is False


# ---------------------------------------------------------------------------------- the adapters


def test_groq_request_shape_and_parsing():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(
            url=str(req.url), auth=req.headers["authorization"], body=json.loads(req.content)
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<think>x</think> He works at Ethara. "}}]},
        )

    p = GroqProvider(
        "k-test", "model-from-env", "https://g.test/openai/v1", 5.0, httpx.MockTransport(handler)
    )
    assert p.complete("SYS", "USER", 100) == "He works at Ethara."
    assert (
        seen["url"] == "https://g.test/openai/v1/chat/completions"
        and seen["auth"] == "Bearer k-test"
    )
    assert seen["body"]["model"] == "model-from-env" and seen["body"]["max_tokens"] == 100
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


@pytest.mark.parametrize("status", [429, 500, 503, 401])
def test_groq_errors_become_provider_unavailable(status):
    p = GroqProvider(
        "k", "m", "https://g.test", 5.0, httpx.MockTransport(lambda r: httpx.Response(status))
    )
    with pytest.raises(ProviderUnavailable) as exc:
        p.complete("s", "u", 10)
    assert (exc.value.kind == "rate_limit") == (status == 429)


def test_groq_malformed_empty_and_timeout():
    for body in ({"oops": 1}, {"choices": [{"message": {"content": ""}}]}):
        p = GroqProvider(
            "k",
            "m",
            "https://g.test",
            5.0,
            httpx.MockTransport(lambda r, b=body: httpx.Response(200, json=b)),
        )
        with pytest.raises(ProviderUnavailable):
            p.complete("s", "u", 10)

    def boom(req):
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(ProviderUnavailable, match="ConnectTimeout"):
        GroqProvider("k", "m", "https://g.test", 5.0, httpx.MockTransport(boom)).complete(
            "s", "u", 10
        )


def test_gemini_request_shape_and_blocked_response():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(
            url=str(req.url), key=req.headers["x-goog-api-key"], body=json.loads(req.content)
        )
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "Answer."}]}}]}
        )

    p = GeminiProvider(
        "g-key", "gem-model", "https://gg.test/v1beta", 5.0, httpx.MockTransport(handler)
    )
    assert p.complete("SYS", "USER", 50) == "Answer."
    assert (
        seen["url"] == "https://gg.test/v1beta/models/gem-model:generateContent"
        and seen["key"] == "g-key"
    )
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"
    blocked = GeminiProvider(
        "k",
        "m",
        "https://gg.test",
        5.0,
        httpx.MockTransport(lambda r: httpx.Response(200, json={"promptFeedback": {}})),
    )
    with pytest.raises(ProviderUnavailable, match="no candidates"):
        blocked.complete("s", "u", 10)


# ------------------------------------------------------------------------------------- the chain


def test_the_chain_falls_over_to_the_next_provider():
    a = ScriptedProvider("groq", [ProviderUnavailable("groq: 429", "rate_limit")])
    b = ScriptedProvider("gemini", ["from gemini"])
    gen = make_chain([a, b]).generate("s", "u")
    assert gen.text == "from gemini" and gen.provider == "gemini" and gen.attempts == ["groq: 429"]
    assert len(a.calls) == 1 and len(b.calls) == 1


def test_the_first_provider_is_used_when_it_works():
    a, b = ScriptedProvider("groq", ["ok"]), ScriptedProvider("gemini", ["other"])
    assert make_chain([a, b]).generate("s", "u").provider == "groq"
    assert b.calls == []


def test_all_providers_failing_raises_unavailable():
    a = ScriptedProvider("groq", [ProviderUnavailable("groq: 429", "rate_limit")])
    b = ScriptedProvider("gemini", [ProviderUnavailable("gemini: HTTP 500")])
    with pytest.raises(Unavailable, match="429"):
        make_chain([a, b]).generate("s", "u")


def test_no_providers_is_unavailable():
    with pytest.raises(Unavailable):
        make_chain([]).generate("s", "u")


def test_the_global_cap_stops_generation_without_calling_a_provider():
    s = AskSettings(global_daily_cap=2)
    p = ScriptedProvider("groq", ["ok"])
    chain = make_chain([p], s)
    chain.generate("s", "u")
    chain.generate("s", "u")
    with pytest.raises(Unavailable, match="global"):
        chain.generate("s", "u")
    assert len(p.calls) == 2


def test_a_provider_over_its_daily_budget_is_skipped_before_it_can_answer_429():
    s = AskSettings(provider_daily_caps={"groq": 1, "gemini": 5})
    a, b = ScriptedProvider("groq", ["g1"]), ScriptedProvider("gemini", ["m1"])
    chain = make_chain([a, b], s)
    assert chain.generate("s", "u").provider == "groq"
    second = chain.generate("s", "u")
    assert second.provider == "gemini" and "groq: budget_requests_day" in second.attempts
    assert len(a.calls) == 1


def test_forced_failure_hook_path():
    with pytest.raises(Unavailable):
        make_chain([ScriptedProvider("groq", ["ok"])]).generate("s", "u", force_fail=True)


# ---------------------------------------------------------------------- Groq primary, backup, Gemini last


def full_env(**extra):
    env = {
        "GROQ_API_KEY": "k",
        "GROQ_MODEL": "m1",
        "GROQ_MODEL_BACKUP": "m2",
        "GEMINI_API_KEY": "g",
        "GEMINI_MODEL": "m3",
    }
    env.update(extra)
    return AskSettings.from_env(env)


def test_the_default_chain_is_two_groq_models_then_gemini():
    assert build_chain(full_env(), MemoryLimiter()).names == ["groq", "groq-backup", "gemini"]


def test_gemini_can_be_switched_off_without_a_code_change():
    assert build_chain(full_env(ASK_DISABLE_GEMINI="1"), MemoryLimiter()).names == [
        "groq",
        "groq-backup",
    ]
    assert build_chain(full_env(ASK_DISABLE_GEMINI="0"), MemoryLimiter()).names[-1] == "gemini"


def test_the_backup_model_needs_its_own_id():
    s = AskSettings.from_env({"GROQ_API_KEY": "k", "GROQ_MODEL": "m1"})
    assert build_chain(s, MemoryLimiter()).names == ["groq"]
    assert AskSettings().groq_model_backup == ""


def test_gemini_only_sees_traffic_when_both_groq_models_fail():
    g1 = ScriptedProvider("groq", [ProviderUnavailable("groq: 429", "rate_limit")])
    g2 = ScriptedProvider("groq-backup", [ProviderUnavailable("groq-backup: 429", "rate_limit")])
    gem = ScriptedProvider("gemini", ["from gemini"])
    chain = make_chain([g1, g2, gem])
    assert chain.generate("s", "u").provider == "gemini"
    ok = ScriptedProvider("groq", ["from groq"])
    gem2 = ScriptedProvider("gemini", ["never"])
    assert make_chain([ok, g2, gem2]).generate("s", "u").provider == "groq" and gem2.calls == []


def test_a_429_on_both_groq_models_with_gemini_off_is_the_limit_fallback():
    g1 = ScriptedProvider("groq", [ProviderUnavailable("groq: 429", "rate_limit")])
    g2 = ScriptedProvider("groq-backup", [ProviderUnavailable("groq-backup: 429", "rate_limit")])
    with pytest.raises(Unavailable):
        make_chain([g1, g2]).generate("s", "u")


def test_the_estimate_is_conservative_and_counts_the_completion_allowance():
    from askaditya.providers import estimate_tokens

    assert estimate_tokens("a" * 350, "b" * 350, 200) == 400
    assert estimate_tokens("", "", 220) == 220


def test_the_per_minute_token_budget_moves_the_chain_on_before_a_429():
    s = AskSettings(token_caps_minute={"groq": 1000, "groq-backup": 100000}, token_caps_day={})
    a, b = ScriptedProvider("groq", ["g"]), ScriptedProvider("groq-backup", ["b"])
    chain = make_chain([a, b], s, MemoryLimiter(Clock()))
    big = "x" * 1750  # about 500 estimated prompt tokens plus the completion allowance
    assert chain.generate("s", big).provider == "groq"
    second = chain.generate("s", big)
    assert second.provider == "groq-backup" and "groq: budget_tokens_minute" in second.attempts
    assert len(a.calls) == 1


def test_the_per_day_token_budget_resets_at_midnight():
    clock = Clock()
    s = AskSettings(token_caps_day={"groq": 1000}, token_caps_minute={})
    p = ScriptedProvider("groq", ["ok"])
    chain = make_chain([p], s, MemoryLimiter(clock))
    chain.generate("s", "u" * 1000)
    with pytest.raises(Unavailable, match="budget_tokens_day"):
        chain.generate("s", "u" * 1000)
    clock.t += 86400
    assert chain.generate("s", "u" * 1000).provider == "groq"


def test_a_failed_call_gives_its_token_reservation_back():
    s = AskSettings(token_caps_day={"groq": 600}, token_caps_minute={})
    p = ScriptedProvider("groq", [ProviderUnavailable("groq: 500"), "ok", "ok"])
    chain = make_chain([p], s, MemoryLimiter(Clock()))
    with pytest.raises(Unavailable):
        chain.generate("s", "u")
    assert (
        chain.generate("s", "u").provider == "groq"
    )  # would be refused if the first estimate were kept
    assert chain.limiter.take_tokens("groq", 100, None, 600) is True


def test_reported_usage_replaces_the_estimate():
    class Reporting(ScriptedProvider):
        last_usage = {"total_tokens": 10}

    lim = MemoryLimiter(Clock())
    chain = make_chain(
        [Reporting("groq", ["ok"])],
        AskSettings(token_caps_day={"groq": 100000}, token_caps_minute={}),
        lim,
    )
    gen = chain.generate("s" * 3500, "u" * 3500)
    assert gen.usage == {"total_tokens": 10}
    assert (
        lim.take_tokens("groq", 99985, None, 100000) is True
    )  # only the 10 reported tokens were kept


def test_groq_sends_the_reasoning_effort_only_when_configured_and_records_usage():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 30, "total_tokens": 1030},
            },
        )

    p = GroqProvider(
        "k", "m", "https://g.test", 5.0, httpx.MockTransport(handler), "groq-backup", "low"
    )
    assert p.complete("s", "u", 50) == "ok" and p.name == "groq-backup"
    assert seen[0]["reasoning_effort"] == "low" and p.last_usage["total_tokens"] == 1030
    q = GroqProvider("k", "m", "https://g.test", 5.0, httpx.MockTransport(handler))
    q.complete("s", "u", 50)
    assert "reasoning_effort" not in seen[1]


def test_the_token_budget_works_through_the_upstash_pipeline():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body[0])
        return httpx.Response(200, json=[{"result": 5}, {"result": 1}])

    lim = UpstashLimiter(
        "https://u.test", "t", MemoryLimiter(), transport=httpx.MockTransport(handler)
    )
    assert lim.take_tokens("groq", 700, 6500, 170000) is True
    assert all(c[0] == "INCRBY" and c[2] == 700 for c in seen) and len(seen) == 2
