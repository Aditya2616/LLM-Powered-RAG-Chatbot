import numpy as np
import pytest
from langchain_core.documents import Document

from rag_chatbot.embeddings import HashingEmbeddings
from rag_chatbot.loaders import load_documents, split_documents
from rag_chatbot.bm25 import BM25Index, rrf_fuse
from rag_chatbot.retrieval import Retriever
from rag_chatbot.store import IndexMismatch, VectorStore
from tests.conftest import CORPUS


def chunks_of(*names, size=300):
    return split_documents(load_documents([CORPUS / n for n in names]), size, 40)


def all_chunks():
    return chunks_of(*[p.name for p in sorted(CORPUS.glob("*.txt"))])


def make_store(directory=None):
    emb = HashingEmbeddings()
    return VectorStore(emb, emb.fingerprint, directory)


def test_dense_search_returns_cosine_sorted_scores():
    st = make_store()
    st.add(chunks_of("returns_and_refunds.txt", "warranty_policy.txt"))
    qv = st.embed_query("how long is the warranty on a tent")
    found = st.dense_search(qv, 5)
    scores = [s for _, s in found]
    assert scores == sorted(scores, reverse=True)
    assert all(abs(s - float(st.cosine_all(qv)[i])) < 1e-5 for i, s in found)
    assert st.chunks[found[0][0]].metadata["source"] == "warranty_policy.txt"


def test_persistence_roundtrip_and_no_pickle(tmp_path):
    st = make_store(tmp_path)
    st.add(chunks_of("shipping_and_delivery.txt"))
    st.save()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "chunks.jsonl",
        "index.faiss",
        "meta.json",
    ]
    again = make_store(tmp_path)
    assert again.load()
    assert len(again) == len(st)
    qv = st.embed_query("next-day delivery cut-off")
    assert again.dense_search(qv, 1) == st.dense_search(qv, 1)
    assert again.chunks[0].metadata == st.chunks[0].metadata


def test_load_refuses_a_different_embedder(tmp_path):
    st = make_store(tmp_path)
    st.add(chunks_of("shipping_and_delivery.txt"))
    st.save()
    other = VectorStore(HashingEmbeddings(64), "hashing:64", tmp_path)
    with pytest.raises(IndexMismatch, match="re-ingest"):
        other.load()


def test_load_of_an_empty_directory_is_false(tmp_path):
    assert make_store(tmp_path).load() is False


def test_re_adding_a_source_replaces_it_and_remove_works():
    st = make_store()
    st.add(chunks_of("returns_and_refunds.txt", "warranty_policy.txt"))
    again = chunks_of("returns_and_refunds.txt", size=200)
    st.add(again)
    sources = {s.source: s.chunks for s in st.sources()}
    assert sources["returns_and_refunds.txt"] == len(again)
    assert "warranty_policy.txt" in sources
    assert st.remove_source("warranty_policy.txt") > 0
    assert {s.source for s in st.sources()} == {"returns_and_refunds.txt"}
    assert st.remove_source("nope.txt") == 0


def test_dimension_mismatch_is_rejected():
    st = make_store()
    st.add([Document(page_content="alpha beta", metadata={"source": "a"})])
    st.embeddings = HashingEmbeddings(64)
    with pytest.raises(IndexMismatch):
        st.add([Document(page_content="gamma", metadata={"source": "b"})])


def test_bm25_prefers_the_rare_matching_word():
    docs = [
        Document(page_content=t) for t in ["the cat sat", "the dog barked loudly", "a zebra grazed"]
    ]
    idx = BM25Index([d.page_content for d in docs])
    assert int(np.argmax(idx.scores("zebra"))) == 2
    assert idx.scores("nonexistent").sum() == 0
    assert int(np.argmax(idx.scores("dogs dog"))) == 1


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_retriever_modes_find_the_right_document(mode):
    st = make_store()
    st.add(all_chunks())
    res = Retriever(st, mode).retrieve("Who is the Data Protection Officer?", 3)
    assert res.mode == mode
    assert [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
    assert res.hits[0].chunk.metadata["source"] == "customer_data_and_privacy.txt"
    h = res.hits[0]
    assert -1.0 <= h.dense <= 1.0 and h.bm25 >= 0.0


def test_exact_term_query_finds_its_document():
    st = make_store()
    st.add(all_chunks())
    res = Retriever(st, "hybrid").retrieve("Keyhold", 2)
    assert res.hits[0].chunk.metadata["source"] == "it_security_policy.txt"


def test_retrieve_on_an_empty_store():
    assert Retriever(make_store(), "hybrid").retrieve("anything", 3).hits == []


def test_reranker_reorders_candidates():
    class Flip:
        def score(self, query, texts):
            return [float(i) for i in range(len(texts))]  # prefers the last candidate

    st = make_store()
    st.add(chunks_of("returns_and_refunds.txt"))
    plain = Retriever(st, "dense").retrieve("refund", 2).hits
    reranked = Retriever(st, "dense", Flip()).retrieve("refund", 2).hits
    assert [h.index for h in plain] != [h.index for h in reranked]


def test_rrf_fuse_rewards_agreement():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]])
    order = [i for i, _ in fused]
    assert order[0] == 1 and set(order) == {1, 2, 3, 4}
    assert fused[0][1] > fused[-1][1]
