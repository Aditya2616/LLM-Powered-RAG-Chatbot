import json

import pytest

from rag_chatbot.config import Settings
from rag_chatbot.llm import ExtractiveLLM
from tests.conftest import CORPUS, ScriptedLLM


def test_answer_comes_with_citations_and_timings(loaded_pipeline):
    r = loaded_pipeline.ask("What is the time limit for sending back something I have not used?")
    assert r["answer"] == "Thirty days."
    assert not r["abstained"] and r["abstain_reason"] is None and not r["cached"]
    src = r["sources"]
    assert 1 <= len(src) <= 3
    for s in src:
        assert s["source"].endswith(".txt") and s["chunk_id"] == f"{s['source']}#{s['chunk']}"
        assert 0 <= s["start"] < s["end"]
        assert s["text"] and s["snippet"] and "support" in s
    assert r["timings"]["total_ms"] >= r["timings"]["retrieve_ms"] >= 0
    assert r["model"] == {"embeddings": "hashing:384", "llm": "scripted"}


def test_prompt_contains_numbered_context_and_the_question(loaded_pipeline):
    loaded_pipeline.ask("Who is the Data Protection Officer?")
    prompt = loaded_pipeline.llm.prompts[-1]
    assert "[1] (source:" in prompt.user and "Who is the Data Protection Officer?" in prompt.user
    assert "ONLY" in prompt.system and "I don't know" in prompt.system
    assert any("Imran Qureshi" in p for p in prompt.passages)


def test_model_declining_is_reported_as_an_abstention(make_pipeline):
    pipe = make_pipeline(ScriptedLLM("I don't know."))
    pipe.ingest(sorted(CORPUS.glob("*.txt")))
    r = pipe.ask("What is the capital of France?")
    assert r["abstained"] and r["abstain_reason"] == "model_declined"
    assert r["answer"] == "I don't know"
    assert all("support" not in s for s in r["sources"])


def test_gate_skips_the_model_when_retrieval_is_weak(make_pipeline):
    llm = ScriptedLLM("should not be used")
    pipe = make_pipeline(llm, min_dense_score=0.99)
    pipe.ingest(sorted(CORPUS.glob("*.txt")))
    r = pipe.ask("What is the capital of France?")
    assert r["abstained"] and r["abstain_reason"] == "low_retrieval_score"
    assert llm.prompts == []
    assert r["timings"]["generate_ms"] == 0


def test_gate_lets_a_strong_match_through(make_pipeline):
    pipe = make_pipeline(ScriptedLLM("Keyhold"), min_dense_score=0.2)
    pipe.ingest(sorted(CORPUS.glob("*.txt")))
    r = pipe.ask("What is the name of the company password manager Keyhold?")
    assert not r["abstained"]


def test_empty_index_abstains_without_calling_the_model(make_pipeline):
    llm = ScriptedLLM()
    r = make_pipeline(llm).ask("anything at all")
    assert r["abstain_reason"] == "empty_index" and llm.prompts == [] and r["sources"] == []


def test_blank_question_is_rejected(loaded_pipeline):
    with pytest.raises(ValueError, match="empty"):
        loaded_pipeline.ask("   ")


def test_repeat_question_is_served_from_the_cache(loaded_pipeline):
    a = loaded_pipeline.ask("How long are order records stored?")
    b = loaded_pipeline.ask("How long are order records stored?")
    assert not a["cached"] and b["cached"]
    assert len(loaded_pipeline.llm.prompts) == 1
    assert b["answer"] == a["answer"] and b["sources"] == a["sources"]


def test_cache_is_cleared_when_documents_change(loaded_pipeline, tmp_path):
    loaded_pipeline.ask("How long are order records stored?")
    extra = tmp_path / "extra.txt"
    extra.write_text("Order records are archived in the vault.", encoding="utf-8")
    loaded_pipeline.ingest([extra])
    assert not loaded_pipeline.ask("How long are order records stored?")["cached"]


def test_cache_is_lru_and_expires(monkeypatch):
    from rag_chatbot.pipeline import QueryCache

    c = QueryCache(2, 10)
    c.set(("a",), {"v": 1})
    c.set(("b",), {"v": 2})
    assert c.get(("a",)) == {"v": 1}  # refreshes a
    c.set(("c",), {"v": 3})  # evicts b
    assert c.get(("b",)) is None and c.get(("a",)) and c.get(("c",))
    now = __import__("time").time()
    monkeypatch.setattr("time.time", lambda: now + 11)
    assert c.get(("a",)) is None


def test_history_is_used_per_session_and_can_be_disabled(loaded_pipeline):
    p = loaded_pipeline
    p.ask("How long are order records stored?", session_id="s1")
    p.ask("And CCTV footage?", session_id="s1")
    assert "Conversation history" in p.llm.prompts[-1].user
    assert "order records" in p.llm.prompts[-1].user
    p.ask("And analytics cookies?", session_id="s2")
    assert "Conversation history" not in p.llm.prompts[-1].user
    p.ask("And backups?", session_id="s1", use_history=False)
    assert "Conversation history" not in p.llm.prompts[-1].user


def test_history_is_capped(make_pipeline):
    pipe = make_pipeline(ScriptedLLM(), max_history_turns=2)
    pipe.ingest(sorted(CORPUS.glob("*.txt")))
    for i in range(4):
        pipe.ask(f"question number {i} about backups", session_id="s")
    assert len(pipe.history["s"]) == 2


def test_query_log_is_written(loaded_pipeline, settings):
    loaded_pipeline.ask("Who is the Data Protection Officer?")
    loaded_pipeline.ask("Who is the Data Protection Officer?")
    lines = [
        json.loads(x) for x in settings.query_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["cached"] for r in lines] == [False, True]
    assert lines[0]["question"] == "Who is the Data Protection Officer?"
    assert lines[0]["sources"] and lines[0]["total_ms"] >= 0


def test_stream_event_order(loaded_pipeline):
    events = list(loaded_pipeline.stream("How long are order records stored?"))
    names = [e["event"] for e in events]
    assert names[0] == "retrieval" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    assert "".join(e["text"] for e in events if e["event"] == "token") == events[-1]["answer"]
    assert events[0]["sources"] and "query_xy" in events[0]


def test_stream_on_a_cache_hit_and_on_abstention(loaded_pipeline, make_pipeline):
    q = "How long are order records stored?"
    list(loaded_pipeline.stream(q))
    again = list(loaded_pipeline.stream(q))
    assert again[0]["cached"] and again[-1]["cached"]
    pipe = make_pipeline(ScriptedLLM("I don't know"))
    pipe.ingest(sorted(CORPUS.glob("*.txt")))
    names = [e["event"] for e in pipe.stream("Capital of France?")]
    assert names[-1] == "done" and names[0] == "retrieval"


def test_few_shot_adds_turns_before_the_question(make_pipeline):
    llm = ScriptedLLM()
    pipe = make_pipeline(llm, few_shot=True)
    pipe.ingest(sorted(CORPUS.glob("*.txt")))
    pipe.ask("Who is the Data Protection Officer?")
    chat = llm.prompts[-1].chat()
    assert [m["role"] for m in chat] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert chat[4]["content"] == "I don't know"


def test_retriever_can_be_chosen_per_question(loaded_pipeline):
    assert loaded_pipeline.ask("Keyhold", retriever="bm25")["retriever"] == "bm25"
    assert loaded_pipeline.ask("Keyhold", retriever="dense")["retriever"] == "dense"


def test_end_to_end_with_the_extractive_generator(make_pipeline):
    pipe = make_pipeline(ExtractiveLLM(), chunk_size=500, chunk_overlap=75)
    pipe.ingest(sorted(CORPUS.glob("*.txt")))
    r = pipe.ask("How long are order records stored?", top_k=4)
    assert "7 years" in r["answer"]
    assert r["sources"][0]["support"] >= 0.5


def test_reindex_rebuilds_from_raw_docs(make_pipeline, settings):
    pipe = make_pipeline(ScriptedLLM())
    settings.raw_docs_dir.mkdir(parents=True)
    (settings.raw_docs_dir / "a.txt").write_text("Alpha facts live here. " * 20, encoding="utf-8")
    assert pipe.reindex_raw_docs() > 0
    assert [s.source for s in pipe.store.sources()] == ["a.txt"]
    assert isinstance(pipe.settings, Settings)
