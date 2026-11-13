from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from rag_chatbot.config import Settings
from rag_chatbot.embeddings import HashingEmbeddings
from rag_chatbot.llm import ExtractiveLLM, Prompt
from rag_chatbot.pipeline import RAGPipeline
from rag_chatbot.retrieval import Retriever
from rag_chatbot.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "docs"
QUESTIONS = ROOT / "corpus" / "questions.jsonl"


class ScriptedLLM:
    """A stand-in generator: returns a fixed reply and records every prompt it was given."""

    name = "scripted"

    def __init__(self, reply: str = "Thirty days.") -> None:
        self.reply = reply
        self.prompts: list[Prompt] = []

    def generate(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> str:
        self.prompts.append(prompt)
        return self.reply

    def stream(self, prompt: Prompt, max_new_tokens: int, temperature: float) -> Iterator[str]:
        self.prompts.append(prompt)
        words = self.reply.split(" ")
        for i, w in enumerate(words):
            yield w if i == 0 else " " + w


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        embeddings_provider="hashing",
        llm_provider="extractive",
        chunk_size=300,
        chunk_overlap=40,
        top_k=3,
    )


@pytest.fixture
def make_pipeline(settings: Settings):
    def build(llm=None, **changes) -> RAGPipeline:
        cfg = settings.with_(**changes)
        emb = HashingEmbeddings()
        store = VectorStore(emb, emb.fingerprint, cfg.vector_store_dir)
        return RAGPipeline(cfg, store, Retriever(store, cfg.retriever), llm or ExtractiveLLM())

    return build


@pytest.fixture
def loaded_pipeline(make_pipeline):
    pipe = make_pipeline(ScriptedLLM())
    pipe.ingest(sorted(CORPUS.glob("*.txt")))
    return pipe
