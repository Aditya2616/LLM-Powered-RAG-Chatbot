import json

import pytest

from rag_chatbot.cli import main
from tests.conftest import CORPUS


@pytest.fixture
def offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "hashing")
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.chdir(tmp_path)


def test_cli_ingest_ask_info(offline_env, capsys):
    assert main(["ingest", str(CORPUS)]) == 0
    assert "11 files" in capsys.readouterr().out
    assert main(["ask", "When are PayPal refunds issued?"]) == 0
    out = capsys.readouterr().out
    assert "2 working days" in out and "returns_and_refunds.txt" in out
    assert main(["ask", "Who is the Data Protection Officer?", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "Imran Qureshi" in data["answer"] and data["sources"]
    assert main(["info"]) == 0
    assert "11 documents" in capsys.readouterr().out


def test_cli_ask_on_an_empty_index_says_what_to_do(offline_env, capsys):
    assert main(["ask", "anything"]) == 1
    assert "ingest" in capsys.readouterr().err


def test_cli_reports_bad_configuration(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "hashing")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["ingest", str(CORPUS)]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


@pytest.mark.slow
def test_real_models_answer_and_decline(tmp_path):
    """Loads MiniLM and Qwen2.5-0.5B-Instruct (downloaded on first use). Run with: pytest -m slow."""
    from rag_chatbot.config import Settings
    from rag_chatbot.pipeline import build_pipeline

    cfg = Settings(data_dir=tmp_path / "d")
    pipe = build_pipeline(cfg)
    pipe.ingest(sorted(CORPUS.glob("*.txt")))
    ok = pipe.ask("How many days of annual leave does a full-time employee get?")
    assert "28" in ok["answer"] and ok["sources"][0]["source"] == "leave_and_time_off.txt"
    streamed = "".join(
        e["text"]
        for e in pipe.stream("Who is the Data Protection Officer?")
        if e["event"] == "token"
    )
    assert "Imran" in streamed
