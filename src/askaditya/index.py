"""The runtime index: chunks plus BM25, loaded from ``index.json``. NumPy only."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag_chatbot.bm25 import BM25Index, rrf_fuse

PROJECTS = {
    "evalbench": "EvalBench",
    "agentflow": "AgentFlow",
    "promptlab": "PromptLab",
    "sft forge": "SFT-Forge",
    "sft-forge": "SFT-Forge",
    "sftforge": "SFT-Forge",
    "diffuselab": "DiffuseLab",
    "visionlens": "VisionLens",
    "voicescribe": "VoiceScribe",
    "sentinel": "Sentinel",
    "modelrouter": "ModelRouter",
    "model router": "ModelRouter",
    "rag chatbot": "LLM-Powered-RAG-Chatbot",
    "rag-chatbot": "LLM-Powered-RAG-Chatbot",
}

# Words a visitor uses -> the words the documents use for the same thing (appended to the query).
EXPANSIONS = [
    (r"\b(email|e-mail|contact|reach|linkedin|github|get in touch|message)\b", "Contact"),
    (
        r"\b(work(s|ing)? (now|currently|right now)|currently|current job|employer|intern(ship)?)\b",
        "Experience",
    ),
    (r"\b(certif\w*|courses?|credentials?)\b", "Certifications"),
    (
        r"\b(study|studies|studied|education|degree|university|college|graduat\w*|b\.?tech)\b",
        "Education",
    ),
    (r"\b(skills?|tech stack|technolog\w*|tools?|languages?|frameworks?|libraries)\b", "Skills"),
    (r"\b(open to|looking for|hiring|opportunit\w*)\b", "About"),
    (r"\b(based|located|where does he live|city|country)\b", "About"),
]


def expand_query(query: str) -> str:
    extra = [add for pattern, add in EXPANSIONS if re.search(pattern, query, flags=re.I)]
    return query + (" " + " ".join(extra) if extra else "")


def named_project(query: str) -> str | None:
    q = query.lower()
    for key, name in PROJECTS.items():
        if key in q:
            return name
    return None


_ANAPHORA = re.compile(
    r"\b(it|its|they|them|their|this|that|those|these|there|the same|too|also|and)\b|^(and|what about|how about)\b",
    re.I,
)


def contextual_query(question: str, history: list[dict]) -> str:
    """Add the previous visitor question when the new one leans on it ("how many rules does it use?")."""
    if history and (len(question.split()) <= 5 or _ANAPHORA.search(question)):
        prior = [h["text"] for h in history if h["role"] == "user"]
        if prior:
            return prior[-1] + " " + question
    return question


LEAD_BOOST = 2.0  # README lead sections (no "A > B" path) hold the summary and the headline numbers
PROJECT_BOOST = 1.2  # other chunks of a README the visitor named
PROFILE_BOOST = 1.5  # résumé and portfolio chunks when no project is named

PROFILE_TITLES = {"Portfolio", "Résumé"}


@dataclass
class Hit:
    chunk: dict
    score: float
    rank: int


class AskIndex:
    def __init__(
        self,
        chunks: list[dict],
        version: str,
        traps: list[dict] | None = None,
        dense: object | None = None,
    ) -> None:
        self.chunks = chunks
        self.version = version
        self.traps = traps or []
        self._bm25 = BM25Index([self._search_text(c) for c in chunks])
        self.dense = dense  # optional (n, d) array of unit vectors, only used offline

    @staticmethod
    def _search_text(c: dict) -> str:
        """What BM25 indexes. Profile chunks carry their section name (Contact, Education...) twice so a
        heading match counts; README chunks carry only the last heading, because the project name is
        handled by the query-side project boost and would otherwise lift every chunk of a README alike."""
        if c["title"] in PROFILE_TITLES:
            return f"{c['section']} {c['section']} {c['text']}"
        return f"{c['section'].split(' > ')[-1]} {c['text']}"

    @classmethod
    def load(cls, directory: Path) -> AskIndex:
        data = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        dense = None
        npy = directory / "dense.npy"
        if npy.is_file():  # present only after an offline evaluation build; never needed to serve
            import numpy as np

            dense = np.load(npy)
        return cls(data["chunks"], data["version"], data.get("traps", []), dense)

    def search(
        self, query: str, k: int = 5, query_vec: object | None = None, heuristics: bool = True
    ) -> list[Hit]:
        """BM25 (with a few query-side heuristics), fused with dense cosine by reciprocal rank when a
        query vector is given. ``heuristics=False`` gives plain BM25, for measuring what they add."""
        proj = named_project(query) if heuristics else None
        scores = self._bm25.scores(expand_query(query) if heuristics else query)
        if heuristics:
            for i, c in enumerate(self.chunks):
                if proj:
                    if c["title"].startswith(proj):
                        scores[i] *= LEAD_BOOST if " > " not in c["section"] else PROJECT_BOOST
                elif c["title"] in PROFILE_TITLES:
                    scores[i] *= PROFILE_BOOST
        order = [int(i) for i in scores.argsort()[::-1] if scores[i] > 0][: max(20, 4 * k)]
        if query_vec is not None and self.dense is not None:
            cos = self.dense @ query_vec
            dense_order = [int(i) for i in cos.argsort()[::-1][: max(20, 4 * k)]]
            fused = rrf_fuse([dense_order, order])[:k]
            return [Hit(self.chunks[i], float(s), r + 1) for r, (i, s) in enumerate(fused)]
        return [Hit(self.chunks[i], float(scores[i]), r + 1) for r, i in enumerate(order[:k])]
