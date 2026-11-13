import importlib.util

from tests.ask_fakes import ROOT

spec = importlib.util.spec_from_file_location("run_ask_eval", ROOT / "scripts" / "run_ask_eval.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def res(status=200, provider=None, stage=None, fallback=None):
    events = [("done", {"fallback": fallback})]
    return {"status": status, "provider": provider, "stage": stage, "events": events}


def test_the_provider_header_decides_whether_a_question_reached_a_model():
    assert runner.reached_model(res(provider="groq"))
    assert not runner.reached_model(res(provider=""))  # a fixed reply: header present but empty
    assert not runner.reached_model(res(status=429, provider="groq"))
    assert not runner.reached_model(res(status=503))


def test_without_the_header_every_non_deterministic_answer_counts_as_generated():
    assert runner.reached_model(res())
    assert runner.reached_model(res(fallback="unknown"))
    assert not runner.reached_model(res(fallback="offtopic"))
    assert not runner.reached_model(res(fallback="refused"))
    assert runner.reached_model(res(stage="model"))


def test_effective_rate_is_questions_per_minute():
    assert runner.effective_rate([0.0]) is None
    assert round(runner.effective_rate([0.0, 8.0, 16.0, 24.0]), 1) == 7.5
    assert runner.effective_rate([float(i * 9) for i in range(11)]) < runner.MAX_SAFE_QPM
    assert runner.effective_rate([i * 0.5 for i in range(40)]) > runner.MAX_SAFE_QPM
