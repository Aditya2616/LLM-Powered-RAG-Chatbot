"""Build the "Ask Aditya" index from the private source documents.

    python scripts/build_ask_index.py --resume Aditya_Jain_Resume.pdf --corpus corpus.md --readmes ../ProjectA ../ProjectB

Reads the résumé (PDF), the exported portfolio corpus (Markdown) and each project README, cleans them,
removes phone numbers at ingestion, chunks them with this repository's own splitter, and writes
``index.json`` and ``VERSION`` (date plus a SHA-256 for every source file) into an output directory that
is git-ignored and included in the deployment bundle. The sources themselves never enter git.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER = "[ASSISTANT-PLATFORM]"
DEFAULT_PLATFORM_NAME = "an AI assistant platform"
GITHUB = "https://github.com/Aditya2616"
PORTFOLIO = "https://adityajain16.vercel.app"
RESUME_URL = f"{PORTFOLIO}/Aditya_Jain_Resume.pdf"

# Any run of digits that could be a phone number: 9 or more digits with optional separators.
PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().-]{7,}\d(?![\w])")


def phone_like(text: str) -> list[str]:
    """Digit runs that look like a phone number (at least nine digits, ISO dates excluded)."""
    out = []
    for m in PHONE.finditer(text):
        run = m.group(0)
        digits = re.sub(r"\D", "", run)
        compact = run.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", compact) or re.fullmatch(
            r"\d{1,6}\s?-\s?\d{1,6}", compact
        ):
            continue  # an ISO date or a numeric range such as 11000-12000
        if len(digits) >= 9:
            out.append(run)
    return out


def strip_phones(text: str) -> str:
    for run in phone_like(text):
        text = text.replace(run, "")
    return text


@dataclass
class Section:
    title: str
    url: str
    section: str
    text: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def platform_name(private_dir: Path | None = None) -> str:
    """The product name for the placeholder: an environment variable or a git-ignored local file."""
    env = os.environ.get("ASK_PLATFORM_NAME", "").strip()
    if env:
        return env
    for candidate in [
        private_dir / "platform_name.txt" if private_dir else None,
        Path("ask/private/platform_name.txt"),
    ]:
        if candidate and candidate.is_file():
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value
    return DEFAULT_PLATFORM_NAME


# ---------------------------------------------------------------------------------------- cleaning
def clean_markdown(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"```mermaid.*?```", "", md, flags=re.S)
    md = re.sub(r"<img[^>]*>", "", md)
    md = re.sub(r"</?(?:p|div|details|summary|br|picture|source)[^>]*>", "", md)
    md = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    md = re.sub(r"\[\]\([^)]*\)", "", md)
    md = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", md)
    md = re.sub(r"^\s*\|?\s*:?-{3,}[\s|:-]*$", "", md, flags=re.M)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def split_sections(md: str, doc_title: str) -> list[tuple[str, str]]:
    """(heading path, body) pairs, one per heading, in order."""
    path: dict[int, str] = {}
    out: list[tuple[str, list[str]]] = [(doc_title, [])]
    in_code = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            path = {k: v for k, v in path.items() if k < level}
            path[level] = m.group(2).strip()
            out.append((" > ".join(path[k] for k in sorted(path)), []))
        else:
            out[-1][1].append(line)
    return [(h, "\n".join(b).strip()) for h, b in out if "\n".join(b).strip()]


# ------------------------------------------------------------------------------------------ sources
def read_resume(pdf: Path) -> list[Section]:
    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf)).pages)
    text = strip_phones(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\|\s*\|", "|", text).replace(" |  |", " |")
    return [Section("Résumé", RESUME_URL, "Résumé", text.strip())]


FALSE_CLAIM = re.compile(r"^- (?P<claim>.+?) \((?P<truth>[^()]*)\)\.?\s*$")


def read_corpus(md_path: Path, platform: str) -> tuple[list[Section], list[dict]]:
    raw = md_path.read_text(encoding="utf-8")
    # the file starts with a note to the maintainer and carries one parenthetical about the placeholder:
    raw = raw.split("\n## ", 1)[1]
    raw = "## " + raw
    raw = re.sub(r"\n\s*\(NOTE:.*?\)\n", "\n", raw, flags=re.S)
    raw = re.sub(r"\n- The résumé PDF also carries a phone number[^\n]*", "", raw)
    traps: list[dict] = []
    lines = []
    in_false = False
    for line in raw.splitlines():
        if line.startswith("### Claims that are FALSE"):
            in_false = True
            lines.append("### What he does not claim about the projects")
            continue
        if in_false and line.startswith("#"):
            in_false = False
        if in_false and line.startswith("- "):
            m = FALSE_CLAIM.match(line)
            if m:
                claim, truth = m.group("claim"), m.group("truth")
                traps.append({"claim": claim, "truth": truth})
                lines.append(f"- It is not true that {claim}. In fact: {truth}.")
                continue
        lines.append(line)
    raw = "\n".join(lines).replace(PLACEHOLDER, platform)
    sections = [
        Section("Portfolio", PORTFOLIO, h, b)
        for h, b in split_sections(clean_markdown(raw), "Portfolio")
    ]
    return sections, traps


def project_dirs(paths: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    for p in paths:
        if (p / "README.md").is_file():
            dirs.append(p)
        else:
            dirs.extend(sorted(d for d in p.iterdir() if (d / "README.md").is_file()))
    return dirs


def read_readme(project: Path) -> list[Section]:
    name = project.name
    md = clean_markdown((project / "README.md").read_text(encoding="utf-8"))
    title = f"{name} README"
    return [Section(title, f"{GITHUB}/{name}", h, b) for h, b in split_sections(md, name)]


# ------------------------------------------------------------------------------------------- build
def build(
    resume: Path,
    corpus: Path,
    readmes: list[Path],
    out_dir: Path,
    chunk_size: int = 500,
    overlap: int = 80,
    platform: str | None = None,
    today: dt.date | None = None,
) -> dict:
    from langchain_core.documents import Document

    from rag_chatbot.loaders import split_documents

    platform = platform or platform_name(out_dir)
    sections = read_resume(resume)
    corpus_sections, traps = read_corpus(corpus, platform)
    sections += corpus_sections
    projects = project_dirs(readmes)
    for p in projects:
        sections += read_readme(p)
    docs = []
    for i, s in enumerate(sections):
        docs.append(
            Document(
                page_content=s.text,
                metadata={
                    "source": f"{s.title}|{i}",
                    "title": s.title,
                    "url": s.url,
                    "section": s.section,
                },
            )
        )
    chunks = split_documents(docs, chunk_size, overlap)
    items = [
        {
            "id": n,
            "title": c.metadata["title"],
            "url": c.metadata["url"],
            "section": c.metadata["section"],
            "text": c.page_content.strip(),
        }
        for n, c in enumerate(chunks)
        if c.page_content.strip()
    ]
    leaked = [it["id"] for it in items if phone_like(it["text"]) and it["title"] == "Résumé"]
    if leaked:
        raise RuntimeError(f"phone-like digit runs survived in résumé chunks {leaked}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {"resume": sha256(resume), "corpus": sha256(corpus)}
    sources.update({f"readme:{p.name}": sha256(p / "README.md") for p in projects})
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:8]
    today = today or dt.date.today()
    version = {
        "corpus_version": f"{today.isoformat()}-{digest}",
        "built": today.isoformat(),
        "sources": sources,
        "chunks": len(items),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    (out_dir / "index.json").write_text(
        json.dumps(
            {"version": version["corpus_version"], "chunks": items, "traps": traps},
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "VERSION").write_text(
        json.dumps(version, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return version
