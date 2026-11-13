"""Small text helpers shared by retrieval, the pipeline and the evaluation code."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for from with as is are was were be been being it its this
    that these those i you he she we they my your our their me him her us them do does did done have has had
    not no so than then there here what which who whom when where why how can could should would will shall
    may might must into about over under up down out off per""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.':][a-z0-9]+)*")
_IDK = re.compile(
    r"^\W*(?:i\s+(?:do\s*not|don'?t|dont)\s+know|i\s+do\s+not\s+have\s+enough\s+information)\b",
    re.I,
)

ABSTAIN_TEXT = "I don't know"


def normalise(text: str) -> str:
    """Lower-case and flatten the typographic variation that does not change a fact.

    Currency symbols become words, thousands separators and percent signs are normalised, dashes become
    spaces, curly quotes become straight ones. Used for answer matching and groundedness only.
    """
    t = (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace(" ", " ")
        .replace(" ", " ")
        .replace(" ", " ")
    )
    t = re.sub(r"£\s?(\d[\d,]*(?:\.\d+)?)", lambda m: f"{m.group(1)} pounds", t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"(\d)\s?%", r"\1 percent", t)
    t = re.sub(r"(\d)\s?(?:pm|p\.m\.)\b", r"\1 pm", t)
    t = re.sub(r"(\d)\s?(?:am|a\.m\.)\b", r"\1 am", t)
    t = re.sub(r"[‐-―-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def tokens(text: str) -> list[str]:
    """Lower-case word tokens (numbers and decimals kept whole), no stop-word removal."""
    return _TOKEN.findall(normalise(text))


def stem(token: str) -> str:
    """A deliberately tiny stemmer: drop a plural ``s`` (refunds -> refund, but not "class" or "this")."""
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-word (or whole-number) match of an already normalised phrase inside normalised text."""
    pattern = r"(?<![a-z0-9])" + re.escape(normalise(phrase)) + r"(?![a-z0-9])"
    return re.search(pattern, normalise(haystack)) is not None


def is_abstention(answer: str) -> bool:
    """True when the answer is the model declining ("I don't know", also "I do not know")."""
    return bool(_IDK.match(answer.strip().replace("’", "'")))


def snippet(text: str, limit: int) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0]
    return (cut or flat[:limit]) + "..."


def token_support(answer: str, context: str) -> float:
    """Share of the answer's content words (numbers included) that appear in the context.

    A lexical proxy for groundedness: 1.0 means every content word of the answer is in the context,
    0.0 means none is. An empty answer scores 1.0 (it asserts nothing).
    """
    words = content_tokens(answer)
    if not words:
        return 1.0
    ctx = set(content_tokens(context))
    return sum(w in ctx for w in words) / len(words)
