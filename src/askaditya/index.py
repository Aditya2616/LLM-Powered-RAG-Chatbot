"""The runtime index: chunks plus BM25, loaded from ``index.json``. NumPy only."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_chatbot.bm25 import BM25Index, bm25_tokens, rrf_fuse

_EXTRA_STOP = {"use", "uses", "used", "using"}  # in nearly every README chunk: only noise


def ask_tokens(text: str) -> list[str]:
    """BM25 tokens with possessives removed, so "RouterBench's" matches a question about RouterBench."""
    return [t for t in bm25_tokens(re.sub(r"['’]s\b", "", text)) if t not in _EXTRA_STOP]


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
    (r"\bword error rate\b", "WER"),
    (
        r"\b(best|worst|highest|lowest|top|scores?|scored|accuracy|pass rates?|how (well|fast|accurate))\b",
        "Results found measured",
    ),
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
    (r"\b(ethara|genesis|15 ?forteen|post-?training|his (role|job|work))\b", "Experience"),
    (
        r"\b(introduc\w*|summar(y|ise|ize)|who is|overview|background|bio)\b",
        "About",
    ),
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


_ASSISTANT_TALK = re.compile(
    r"ask[- ]?aditya|ask my ai|portfolio (assistant|chatbot)|this (assistant|chatbot|bot)|the (assistant|chatbot)\b"
    r"|placeholder|index build|rebuild\w* the index|private corpus|corpus version",
    re.I,
)


def about_assistant(query: str) -> bool:
    """True when a question is about the chatbot project or the assistant itself; only then may the README
    sections that describe the assistant (marked ``meta`` in the index) be retrieved."""
    return named_project(query) == "LLM-Powered-RAG-Chatbot" or bool(_ASSISTANT_TALK.search(query))


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

BOILERPLATE = re.compile(r"(Quickstart|Reproduce|Layout|Credits|Licen[cs]e)", re.I)
BOILERPLATE_FACTOR = 0.6
SIDE_SECTION = re.compile(
    r"(ablation|sweep|optional|perplexity|appendix|alternative|sensitivity)", re.I
)
SIDE_FACTOR = 0.6

PROFILE_TITLES = {"Portfolio", "Résumé"}
TABLE_BOOST = (
    1.8  # a whole results table, for "best / highest / lowest" questions about a named project
)
_SUPERLATIVE = re.compile(
    r"\b(best|worst|highest|lowest|top|most|least|fastest|slowest|better|largest|smallest|rank\w*|compare|comparison)\b",
    re.I,
)
# "What F1 did X reach?", "what is X's recall on unseen attacks?": a question for one measured figure
METRIC_QUESTION = re.compile(
    r"\b(f1|recall|precision|accuracy|pass rates?|scores?|scored|reach(?:ed)?|achiev\w*|got|get|wer|"
    r"word error rate|cost|share|percent\w*|rate|best|highest|lowest|how (?:much|many|well|high))\b|%",
    re.I,
)
HEADLINE_ROW_BOOST = 1.6  # a README row marked default / headline
HELD_OUT_BOOST = 1.2
VARIANT_FACTOR = (
    0.7  # optional variants (verifier, ablation) and the dev set, unless the question names them
)
_VARIANT = re.compile(r"\b(verifier|ablation|optional|variant|dev|0\.5b|1\.5b)\b", re.I)
ACRONYM_BOOST = (
    1.4  # a chunk that spells an upper-case term of the question the same way (RLHF, WER, SFT)
)
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,6}\b")


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
        profile: list[dict] | None = None,
    ) -> None:
        self.chunks = chunks
        self.profile = profile or []  # the portfolio sections, whole, for the structured answers
        self.version = version
        self.traps = traps or []
        # whole-table chunks repeat text that is already indexed, so they have their own small index and only
        # compete for "best / highest / lowest" questions about a named project (see ``search``)
        self._vocab: set[str] | None = None
        self._main_ids = [i for i, c in enumerate(chunks) if c.get("kind") != "table"]
        self._meta_ids = [i for i, c in enumerate(chunks) if c.get("meta")]
        self._table_ids = [i for i, c in enumerate(chunks) if c.get("kind") == "table"]
        self._bm25 = BM25Index(
            [self._search_text(chunks[i]) for i in self._main_ids], tokenizer=ask_tokens
        )
        self._tables = (
            BM25Index([self._search_text(chunks[i]) for i in self._table_ids], tokenizer=ask_tokens)
            if self._table_ids
            else None
        )
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
        return cls(
            data["chunks"], data["version"], data.get("traps", []), dense, data.get("profile", [])
        )

    def coverage(self, text: str) -> float:
        """The share of the question's search words (stop words and "use" removed) that occur anywhere in the
        index. A question about something the documents never mention has low coverage."""
        if self._vocab is None:
            self._vocab = set(self._bm25.postings) | (
                set(self._tables.postings) if self._tables else set()
            )
        words = set(ask_tokens(text))
        return sum(w in self._vocab for w in words) / len(words) if words else 1.0

    def _lead_chunk(self, project: str) -> int | None:
        for i, c in enumerate(self.chunks):
            if c["title"].startswith(project) and " > " not in c["section"]:
                return i
        return None

    def search(
        self, query: str, k: int = 5, query_vec: object | None = None, heuristics: bool = True
    ) -> list[Hit]:
        """BM25 (with a few query-side heuristics), fused with dense cosine by reciprocal rank when a
        query vector is given. ``heuristics=False`` gives plain BM25, for measuring what they add."""
        proj = named_project(query) if heuristics else None
        scoring_query = expand_query(query) if heuristics else query
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        scores[self._main_ids] = self._bm25.scores(scoring_query)
        if not about_assistant(query):
            for i in self._meta_ids:
                scores[i] = (
                    0.0  # text about the assistant itself never answers a question about him
                )
        if proj and self._tables is not None and _SUPERLATIVE.search(query):
            table_scores = self._tables.scores(scoring_query)
            for j, i in enumerate(self._table_ids):
                if self.chunks[i]["title"].startswith(proj):
                    scores[i] = table_scores[j] * TABLE_BOOST
        acronyms = set(_ACRONYM.findall(query)) if heuristics else set()
        if heuristics and proj and METRIC_QUESTION.search(query):
            for i, c in enumerate(self.chunks):
                if scores[i] <= 0 or not c["title"].startswith(proj):
                    continue
                low = c["text"].lower()
                if "(default / headline row)" in low:
                    scores[i] *= HEADLINE_ROW_BOOST
                if "held-out" in low or "held out" in low:
                    scores[i] *= HELD_OUT_BOOST
                if _VARIANT.search(low) and not _VARIANT.search(query):
                    scores[i] *= VARIANT_FACTOR
        if heuristics:
            for i, c in enumerate(self.chunks):
                if acronyms and acronyms & set(_ACRONYM.findall(c["text"])):
                    scores[i] *= ACRONYM_BOOST
                if proj:
                    if c["title"].startswith(proj):
                        scores[i] *= LEAD_BOOST if " > " not in c["section"] else PROJECT_BOOST
                elif c["title"] in PROFILE_TITLES:
                    scores[i] *= PROFILE_BOOST
        if heuristics:
            for i, c in enumerate(self.chunks):
                if c["title"] not in PROFILE_TITLES and BOILERPLATE.search(c["section"]):
                    scores[i] *= (
                        BOILERPLATE_FACTOR  # commands and file listings rarely hold the answer
                    )
                m = SIDE_SECTION.search(c["section"]) if c["title"] not in PROFILE_TITLES else None
                if m and not re.search(m.group(0), query, re.I):
                    scores[i] *= (
                        SIDE_FACTOR  # a side experiment's figures must not crowd out the headline
                    )
        order = [int(i) for i in scores.argsort()[::-1] if scores[i] > 0][: max(20, 4 * k)]
        lead = self._lead_chunk(proj) if proj else None
        if lead is not None and lead not in order[:k]:
            # the opening summary of a README the visitor named is always eligible: it holds the headline
            order = [*order[: k - 1], lead, *[i for i in order[k - 1 :] if i != lead]]
            scores[lead] = max(
                float(scores[lead]), float(scores[order[k - 2]]) * 0.5 if k > 1 else 0.0
            )
        if query_vec is not None and self.dense is not None:
            cos = self.dense @ query_vec
            dense_order = [int(i) for i in cos.argsort()[::-1][: max(20, 4 * k)]]
            fused = rrf_fuse([dense_order, order])[:k]
            return [Hit(self.chunks[i], float(s), r + 1) for r, (i, s) in enumerate(fused)]
        return [Hit(self.chunks[i], float(scores[i]), r + 1) for r, i in enumerate(order[:k])]
