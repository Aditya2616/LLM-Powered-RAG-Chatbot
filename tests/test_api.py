import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from rag_chatbot.api import create_app
from tests.conftest import CORPUS, ScriptedLLM


@pytest.fixture
def client(make_pipeline, settings, tmp_path):
    pipe = make_pipeline(ScriptedLLM("Seven years."))
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<title>ui</title>", encoding="utf-8")
    app = create_app(settings, pipe, web_dir=web, sample_dir=CORPUS)
    with TestClient(app) as c:
        yield c


def upload(client, files):
    return client.post("/upload", files=[("files", f) for f in files])


def test_health_before_and_after_ingest(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["vector_store_ready"] is False and h["chunks"] == 0
    assert (
        h["embeddings"] == "hashing:384" and h["llm"] == "scripted" and h["retriever"] == "hybrid"
    )
    assert client.post("/documents/sample").status_code == 200
    h = client.get("/health").json()
    assert h["vector_store_ready"] and h["documents"] == 11 and h["chunks"] > 11


def test_upload_query_and_sources(client):
    r = upload(
        client,
        [("faq.txt", b"Order records are kept for 7 years because of tax rules.", "text/plain")],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (
        body["files"] == ["faq.txt"]
        and body["rejected"] == []
        and body["chunks_added"] == body["total_chunks"]
    )
    q = client.post("/query", json={"question": "How long are order records kept?"})
    assert q.status_code == 200
    data = q.json()
    assert data["answer"] == "Seven years." and data["sources"][0]["source"] == "faq.txt"
    assert data["abstained"] is False and data["timings"]["total_ms"] >= 0
    assert (
        client.post("/query", json={"question": "x", "return_sources": False}).json()["sources"]
        == []
    )


def test_upload_partial_rejection(client):
    r = upload(
        client,
        [
            ("good.txt", b"Hello there, this is a document.", "text/plain"),
            ("bad.exe", b"MZ", "application/octet-stream"),
        ],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["files"] == ["good.txt"]
    assert (
        body["rejected"][0]["file"] == "bad.exe" and "unsupported" in body["rejected"][0]["reason"]
    )


def test_upload_all_invalid_is_400_and_oversize_is_413(client, settings):
    r = upload(client, [("a.pdf", b"not a pdf", "application/pdf")])
    assert r.status_code == 400 and "PDF" in json.dumps(r.json())
    big = b"x" * (settings.max_upload_bytes + 1)
    assert upload(client, [("big.txt", big, "text/plain")]).status_code == 413
    assert client.get("/documents").json() == []


def test_upload_path_traversal_name_is_neutralised(client, settings):
    r = upload(client, [("../../evil.txt", b"Some harmless text about nothing.", "text/plain")])
    assert r.status_code == 200 and r.json()["files"] == ["evil.txt"]
    assert (settings.raw_docs_dir / "evil.txt").exists()
    assert not (settings.data_dir.parent / "evil.txt").exists()


def test_upload_docx_without_text_is_422(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document xmlns:w='x'/>")
    r = upload(client, [("empty.docx", buf.getvalue(), "application/octet-stream")])
    assert r.status_code in (422,)


def test_too_many_files(client, settings):
    files = [
        (f"f{i}.txt", b"text content", "text/plain") for i in range(settings.max_upload_files + 1)
    ]
    assert upload(client, files).status_code == 413


def test_query_validation(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "   "}).status_code == 400
    assert client.post("/query", json={"question": "x", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"question": "x", "retriever": "magic"}).status_code == 422


def test_stream_endpoint_sends_sse_events(client):
    client.post("/documents/sample")
    with client.stream(
        "POST", "/query/stream", json={"question": "How long are order records stored?"}
    ) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    names = [n for n, _ in events]
    assert names[0] == "retrieval" and names[-1] == "done" and "token" in names
    assert "".join(d["text"] for n, d in events if n == "token") == events[-1][1]["answer"]
    assert events[0][1]["sources"][0]["chunk_id"]


def test_documents_list_delete_and_map(client):
    client.post("/documents/sample")
    docs = client.get("/documents").json()
    assert len(docs) == 11 and all(d["chunks"] > 0 for d in docs)
    m = client.get("/index/map").json()
    assert len(m["chunks"]) == sum(d["chunks"] for d in docs)
    assert {"chunk_id", "source", "x", "y"} <= set(m["chunks"][0])
    assert client.delete("/documents/warranty_policy.txt").json()["removed_chunks"] > 0
    assert client.delete("/documents/warranty_policy.txt").status_code == 404
    assert len(client.get("/documents").json()) == 10


def test_ui_is_served_at_the_root_and_api_routes_are_not_shadowed(client):
    assert "<title>ui</title>" in client.get("/").text
    assert client.get("/health").json()["status"] == "ok"


def test_index_mismatch_is_reported_not_crashed(settings):
    from langchain_core.documents import Document

    from rag_chatbot.embeddings import HashingEmbeddings
    from rag_chatbot.store import VectorStore

    other = VectorStore(HashingEmbeddings(64), "hashing:64", settings.vector_store_dir)
    other.add([Document(page_content="alpha beta gamma", metadata={"source": "a.txt"})])
    other.save()
    with TestClient(create_app(settings, web_dir=None)) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and "re-ingest" in h["index_error"]
        r = c.post("/query", json={"question": "anything"})
        assert r.status_code == 409 and "re-ingest" in r.json()["detail"]
