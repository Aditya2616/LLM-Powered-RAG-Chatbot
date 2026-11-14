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
from itertools import zip_longest
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


# Sections of a README that describe the assistant itself (the placeholder text, the index build, the private
# corpus). They are kept in the index but marked ``meta``: retrieval leaves them out unless the visitor asks about
# the chatbot project or the assistant, so they never answer a question about Aditya's skills or experience.
ASSISTANT_META = re.compile(r"Ask-Aditya|portfolio assistant|Ask my AI", re.I)


def is_assistant_meta(title: str, section: str) -> bool:
    return title.endswith(" README") and bool(ASSISTANT_META.search(section))


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
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}[\s|:-]*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


_ONE_LETTER = {"P": "precision", "R": "recall"}


def expand_cell(head: str, value: str) -> str | None:
    """A composite cell made unambiguous: header ``Action P / R / F1`` over value ``0.93 / 0.84 / 0.89`` becomes
    ``Action precision 0.93; Action recall 0.84; Action F1 0.89``. Only when the header and the value split on
    `` / `` into the same number of parts and every part after the first header part is a single word (so the
    first part carries the shared prefix, as in ``Owners ok / wrong / none`` or ``CPT 90% / 95% / 99%``);
    anything else returns None and stays as it was. P and R are spelled out only when both are present."""
    hp = [p.strip() for p in head.split(" / ")]
    vp = [p.strip() for p in value.split(" / ")]
    if len(hp) < 2 or len(hp) != len(vp) or any(not p or " " in p for p in hp[1:]):
        return None
    first = hp[0].split()
    names = [first[-1], *hp[1:]]
    prefix = " ".join(first[:-1])
    if {"P", "R"} <= set(names):
        names = [_ONE_LETTER.get(n, n) for n in names]
    return "; ".join(
        f"{prefix + ' ' if prefix else ''}{n} {v}" for n, v in zip(names, vp, strict=True)
    )


def _headline(label: str) -> bool:
    """The row a README marks as its default or headline: ``(default)`` in the label, or a bold label."""
    return "(default)" in label.lower() or (label.startswith("**") and label.endswith("**"))


def tables_to_rows(md: str) -> str:
    """Rewrite every Markdown table so that each row carries its own column names, as one bullet:
    ``| qwen | 44.6% |`` under ``| Model | Pass rate |`` becomes ``- Model: qwen; Pass rate: 44.6%``.
    Chunking splits a table anywhere, and a row without its header says nothing."""
    lines, out, i = md.split("\n"), [], 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code = not in_code
        if (
            not in_code
            and line.lstrip().startswith("|")
            and i + 1 < len(lines)
            and _SEPARATOR.match(lines[i + 1])
        ):
            head = _cells(line)
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                row = _cells(lines[i])
                pairs = [
                    f"{h}: {expand_cell(h, c) or c}" if h else c
                    for h, c in zip(head, row, strict=False)
                    if c and c != "-"
                ]
                if pairs:
                    mark = "(default / headline row) " if row and _headline(row[0]) else ""
                    out.append("- " + mark + "; ".join(pairs))
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


MAX_TABLE_CHARS = 1500


def table_sections(md: str, doc_title: str) -> list[tuple[str, str]]:
    """(heading path, compact text) for every Markdown table of a document that is short enough to be read whole:
    the column names once, then one line per row. These are kept as single chunks (never split), so a question
    about the best or the lowest value can see every row together."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    lines, path, out, i, in_code = md.split("\n"), {}, [], 0, False
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            path = {k: v for k, v in path.items() if k < level}
            path[level] = m.group(2).strip()
        elif (
            not in_code
            and line.lstrip().startswith("|")
            and i + 1 < len(lines)
            and _SEPARATOR.match(lines[i + 1])
        ):
            head = _cells(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            heading = " > ".join(path[k] for k in sorted(path)) or doc_title
            text = f"Table under {heading.split(' > ')[-1]}. Columns: " + " | ".join(head) + "\n"
            text += "\n".join(
                ("(default / headline row) " if r and _headline(r[0]) else "")
                + " | ".join(
                    (expand_cell(h, c) or c) if h else c
                    for h, c in zip_longest(head, r, fillvalue="")
                )
                for r in rows
            )
            text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
            if 2 <= len(rows) <= 14 and len(text) <= MAX_TABLE_CHARS:
                out.append((heading, text))
            continue
        i += 1
    return out


def clean_markdown(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = tables_to_rows(md)
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
    whole_tables = []
    for p in projects:
        raw = (p / "README.md").read_text(encoding="utf-8")
        for heading, text in table_sections(raw, p.name):
            whole_tables.append((f"{p.name} README", f"{GITHUB}/{p.name}", heading, text))
    items = [
        {
            "id": n,
            "title": c.metadata["title"],
            "url": c.metadata["url"],
            "section": c.metadata["section"],
            "text": c.page_content.strip(),
            **(
                {"meta": True}
                if is_assistant_meta(c.metadata["title"], c.metadata["section"])
                else {}
            ),
        }
        for n, c in enumerate(chunks)
        if c.page_content.strip()
    ]
    for title, url, heading, text in whole_tables:
        items.append(
            {
                "id": len(items),
                "title": title,
                "url": url,
                "section": heading,
                "text": text,
                "kind": "table",
            }
        )
    profile = [{"section": s.section, "text": s.text} for s in corpus_sections]
    leaked = [it["id"] for it in items if phone_like(it["text"]) and it["title"] == "Résumé"]
    if leaked:
        raise RuntimeError(f"phone-like digit runs survived in résumé chunks {leaked}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {"resume": sha256(resume), "corpus": sha256(corpus)}
    sources["platform_name"] = (
        "default"
        if platform == DEFAULT_PLATFORM_NAME
        else hashlib.sha256(platform.encode()).hexdigest()
    )
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
            {
                "version": version["corpus_version"],
                "chunks": items,
                "traps": traps,
                "profile": profile,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "VERSION").write_text(
        json.dumps(version, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return version
