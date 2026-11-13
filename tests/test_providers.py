"""Hosted adapters, exercised against a stubbed HTTP transport. No network, no keys."""

import json

import httpx
import pytest

from rag_chatbot.config import ConfigError, Settings
from rag_chatbot.embeddings import (
    GeminiEmbeddings,
    HashingEmbeddings,
    OpenAIEmbeddings,
    ProviderError,
    get_embeddings,
)
from rag_chatbot.llm import (
    ExtractiveLLM,
    GeminiLLM,
    OpenAICompatLLM,
    Prompt,
    get_llm,
)

PROMPT = Prompt(system="SYS", user="USER", question="q", passages=[])


def stub(handler):
    return httpx.MockTransport(handler)


def sse(*events: str) -> bytes:
    return ("".join(f"data: {e}\n\n" for e in events)).encode()


def test_openai_chat_request_shape_and_parsing():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " 30 days. "}}]})

    llm = OpenAICompatLLM("k-test", "model-x", "https://example.test/v1/", stub(handler))
    assert llm.generate(PROMPT, 50, 0.0) == "30 days."
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k-test"
    assert seen["body"]["model"] == "model-x"
    assert seen["body"]["max_tokens"] == 50
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


def test_openai_stream_yields_deltas_and_stops_at_done():
    chunks = [
        json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
        json.dumps({"choices": [{"delta": {"content": "Thirty"}}]}),
        json.dumps({"choices": [{"delta": {"content": " days"}}]}),
        "[DONE]",
    ]
    llm = OpenAICompatLLM(
        "k", "m", transport=stub(lambda r: httpx.Response(200, content=sse(*chunks)))
    )
    assert "".join(llm.stream(PROMPT, 10, 0.0)) == "Thirty days"


def test_openai_stream_error_status():
    llm = OpenAICompatLLM("k", "m", transport=stub(lambda r: httpx.Response(401, text="no")))
    with pytest.raises(ProviderError, match="401"):
        list(llm.stream(PROMPT, 10, 0.0))


def test_retry_on_429_then_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert OpenAICompatLLM("k", "m", transport=stub(handler)).generate(PROMPT, 5, 0.0) == "ok"
    assert calls["n"] == 3


def test_client_error_is_not_retried_and_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(ProviderError, match="400"):
        OpenAICompatLLM("k", "m", transport=stub(handler)).generate(PROMPT, 5, 0.0)
    assert calls["n"] == 1


def test_malformed_response_is_a_provider_error():
    llm = OpenAICompatLLM("k", "m", transport=stub(lambda r: httpx.Response(200, json={"oops": 1})))
    with pytest.raises(ProviderError, match="unexpected response"):
        llm.generate(PROMPT, 5, 0.0)


def test_gemini_generate_and_stream():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["key"] = req.headers["x-goog-api-key"]
        seen["body"] = json.loads(req.content)
        if "streamGenerateContent" in str(req.url):
            parts = [
                {"candidates": [{"content": {"parts": [{"text": t}]}}]} for t in ("Five", " days")
            ]
            return httpx.Response(200, content=sse(*[json.dumps(p) for p in parts]))
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "Five days"}]}}]}
        )

    llm = GeminiLLM("g-key", "some-model", "https://g.test/v1beta", stub(handler))
    assert llm.generate(PROMPT, 20, 0.2) == "Five days"
    assert seen["url"] == "https://g.test/v1beta/models/some-model:generateContent"
    assert seen["key"] == "g-key"
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert seen["body"]["generationConfig"] == {"temperature": 0.2, "maxOutputTokens": 20}
    assert "".join(llm.stream(PROMPT, 20, 0.0)) == "Five days"
    assert "streamGenerateContent?alt=sse" in seen["url"]


def test_gemini_blocked_response_is_reported():
    llm = GeminiLLM(
        "k", "m", transport=stub(lambda r: httpx.Response(200, json={"promptFeedback": {}}))
    )
    with pytest.raises(ProviderError, match="no candidates"):
        llm.generate(PROMPT, 5, 0.0)


def test_openai_embeddings_batches_and_normalises():
    sizes = []

    def handler(req):
        body = json.loads(req.content)
        sizes.append(len(body["input"]))
        data = [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(len(body["input"])))]
        return httpx.Response(200, json={"data": data})

    emb = OpenAIEmbeddings("k", "emb-model", transport=stub(handler), batch_size=2)
    out = emb.embed_documents(["a", "b", "c"])
    assert sizes == [2, 1]
    assert out[0] == pytest.approx([0.6, 0.8])
    assert emb.embed_query("q") == pytest.approx([0.6, 0.8])


def test_gemini_embeddings_use_task_types():
    tasks = []

    def handler(req):
        body = json.loads(req.content)
        tasks.append({r["taskType"] for r in body["requests"]})
        return httpx.Response(
            200, json={"embeddings": [{"values": [1.0, 0.0]} for _ in body["requests"]]}
        )

    emb = GeminiEmbeddings("k", "text-model", transport=stub(handler))
    emb.embed_documents(["a", "b"])
    emb.embed_query("q")
    assert tasks == [{"RETRIEVAL_DOCUMENT"}, {"RETRIEVAL_QUERY"}]
    assert emb.model == "models/text-model"


def test_embedding_count_mismatch_is_rejected():
    emb = OpenAIEmbeddings(
        "k", "m", transport=stub(lambda r: httpx.Response(200, json={"data": []}))
    )
    with pytest.raises(ProviderError, match="different number"):
        emb.embed_documents(["a"])


def test_no_hosted_model_id_has_a_default():
    s = Settings()
    assert s.openai_model == "" and s.gemini_model == "" and s.openai_embedding_model == ""
    assert s.gemini_embedding_model == ""


def test_hosted_provider_without_model_id_or_key_is_a_clear_error():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_llm(Settings(llm_provider="openai"))
    with pytest.raises(ConfigError, match="OPENAI_MODEL"):
        get_llm(Settings(llm_provider="openai", openai_api_key="k"))
    with pytest.raises(ConfigError, match="GEMINI_MODEL"):
        get_llm(Settings(llm_provider="gemini", gemini_api_key="k"))
    with pytest.raises(ConfigError, match="GEMINI_EMBEDDING_MODEL"):
        get_embeddings(Settings(embeddings_provider="gemini", gemini_api_key="k"))


def test_factories_build_the_selected_adapter():
    t = stub(lambda r: httpx.Response(500))
    s = Settings(
        llm_provider="openai",
        openai_api_key="k",
        openai_model="m",
        embeddings_provider="gemini",
        gemini_api_key="g",
        gemini_embedding_model="e",
    )
    assert isinstance(get_llm(s, t), OpenAICompatLLM)
    assert isinstance(get_embeddings(s, t), GeminiEmbeddings)
    assert isinstance(get_llm(Settings(llm_provider="extractive")), ExtractiveLLM)
    assert isinstance(get_embeddings(Settings(embeddings_provider="hashing")), HashingEmbeddings)


def test_any_openai_compatible_server_works_through_base_url():
    seen = {}

    def handler(req):
        seen["host"] = req.url.host
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    s = Settings.from_env(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "k",
            "OPENAI_MODEL": "any",
            "OPENAI_BASE_URL": "https://other-host.test/v1",
        }
    )
    get_llm(s, stub(handler)).generate(PROMPT, 5, 0.0)
    assert seen["host"] == "other-host.test"


def test_extractive_llm_returns_best_sentence_or_abstains():
    p = Prompt(
        system="",
        user="",
        question="When are PayPal refunds issued?",
        passages=[
            "Card refunds take 5 working days. PayPal refunds are issued within 2 working days."
        ],
    )
    assert ExtractiveLLM().generate(p) == "PayPal refunds are issued within 2 working days."
    off = Prompt(
        system="", user="", question="Capital of France?", passages=["Opening hours are 9:00."]
    )
    assert ExtractiveLLM().generate(off) == "I don't know"
    assert "".join(ExtractiveLLM().stream(p)) == ExtractiveLLM().generate(p)


def test_settings_validation():
    with pytest.raises(ConfigError, match="EMBEDDINGS_PROVIDER"):
        Settings.from_env({"EMBEDDINGS_PROVIDER": "nope"})
    with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
        Settings.from_env({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"})
    with pytest.raises(ConfigError, match="TOP_K"):
        Settings.from_env({"TOP_K": "x"})
    s = Settings.from_env(
        {"MIN_DENSE_SCORE": "0.3", "FEW_SHOT": "true", "RAG_DATA_DIR": "somewhere"}
    )
    assert (
        s.min_dense_score == 0.3 and s.few_shot and str(s.vector_store_dir).endswith("vector_store")
    )
