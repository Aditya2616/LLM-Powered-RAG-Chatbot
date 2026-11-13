"""Shared helpers for the Ask-Aditya tests: a scripted provider and a persona index built from ask/persona."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from askaditya.build import build
from askaditya.config import AskSettings
from askaditya.index import AskIndex
from askaditya.limits import MemoryLimiter
from askaditya.providers import ProviderChain, ProviderUnavailable

ROOT = Path(__file__).resolve().parents[1]
PERSONA = ROOT / "ask" / "persona"


class ScriptedProvider:
    """Returns queued replies (or raises) and records every call."""

    def __init__(self, name: str = "fake", replies=None) -> None:
        self.name = name
        self.replies = list(replies or [])
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise ProviderUnavailable(f"{self.name}: nothing scripted", "empty")
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r


def make_resume_pdf(path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in (PERSONA / "resume.txt").read_text(encoding="utf-8").splitlines():
        pdf.cell(0, 6, line.replace("–", "-"), new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def build_persona(out: Path, work: Path, platform: str | None = None) -> dict:
    pdf = work / "resume.pdf"
    make_resume_pdf(pdf)
    return build(
        pdf,
        PERSONA / "corpus.md",
        [PERSONA / "projects"],
        out,
        platform=platform,
        today=dt.date(2026, 1, 2),
    )


def persona_index(tmp: Path) -> AskIndex:
    build_persona(tmp / "idx", tmp)
    return AskIndex.load(tmp / "idx")


def make_chain(providers, settings: AskSettings | None = None, limiter=None) -> ProviderChain:
    cfg = settings or AskSettings()
    return ProviderChain(providers, limiter or MemoryLimiter(), cfg)
