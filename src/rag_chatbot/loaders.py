"""Document loading and chunking (PDF, TXT and DOCX) and the checks applied to uploads."""

from __future__ import annotations

import io
import logging
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".docx"})


class UploadRejected(ValueError):
    """An uploaded file failed validation. ``str(exc)`` is safe to show to the caller."""


def sanitise_filename(name: str) -> str:
    """Keep only the base name, and only characters that are safe in a path and in a log line."""
    base = Path(name.replace("\\", "/")).name.strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).lstrip(". ")
    if not base:
        raise UploadRejected("file has no usable name")
    return base[:120]


def validate_upload(name: str, data: bytes, max_bytes: int) -> str:
    """Return the sanitised file name, or raise ``UploadRejected`` with the reason.

    Checks the extension, the size, that the content really is what the extension says (PDF header, DOCX
    zip with a word/document.xml part, UTF-8 text without NUL bytes) and that it is not empty.
    """
    safe = sanitise_filename(name)
    suffix = Path(safe).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UploadRejected(f"unsupported file type {suffix or '(none)'}; use .pdf, .txt or .docx")
    if not data:
        raise UploadRejected("file is empty")
    if len(data) > max_bytes:
        raise UploadRejected(f"file is larger than the {max_bytes / (1024 * 1024):.4g} MB limit")
    if suffix == ".pdf":
        if data.lstrip()[:5] != b"%PDF-":
            raise UploadRejected("file does not look like a PDF")
    elif suffix == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                has_body = "word/document.xml" in z.namelist()
        except zipfile.BadZipFile as exc:
            raise UploadRejected("file does not look like a Word document") from exc
        if not has_body:
            raise UploadRejected("file does not look like a Word document")
    else:
        if b"\x00" in data:
            raise UploadRejected("file is not plain text")
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UploadRejected("text files must be UTF-8") from exc
    return safe


def load_documents(paths: Iterable[Path]) -> list[Document]:
    """Load files into LangChain ``Document`` objects with ``source`` (and ``page`` for PDFs)."""
    documents: list[Document] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logger.warning("Unsupported file type: %s", path)
            continue
        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loaded = PyPDFLoader(str(path)).load()
        elif suffix == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader

            loaded = Docx2txtLoader(str(path)).load()
        else:
            loaded = [Document(page_content=path.read_text(encoding="utf-8-sig"), metadata={})]
        for doc in loaded:
            doc.metadata["source"] = path.name
        documents.extend(d for d in loaded if d.page_content.strip())
    return documents


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split with ``RecursiveCharacterTextSplitter`` and number the chunks per source.

    Every chunk carries ``source``, ``chunk`` (0-based index within the source), ``chunk_id``
    (``source#chunk``) and ``start`` / ``end`` (character offsets into that page's text, which for a
    text file is the whole document). The evaluation uses those offsets to decide whether a retrieved
    chunk contains the gold passage.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        n = counters.get(source, 0)
        counters[source] = n + 1
        start = int(chunk.metadata.pop("start_index", 0))
        chunk.metadata.update(
            chunk=n, chunk_id=f"{source}#{n}", start=start, end=start + len(chunk.page_content)
        )
    return chunks
