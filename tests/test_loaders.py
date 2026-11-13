import io
import zipfile

import pytest

from rag_chatbot.loaders import (
    UploadRejected,
    load_documents,
    sanitise_filename,
    split_documents,
    validate_upload,
)


def test_split_metadata_points_back_into_the_text(tmp_path):
    text = ("Alpha beta gamma. " * 40 + "\n\n") * 4
    f = tmp_path / "notes.txt"
    f.write_text(text, encoding="utf-8")
    chunks = split_documents(load_documents([f]), 200, 30)
    assert len(chunks) > 3
    for i, c in enumerate(chunks):
        assert c.metadata["source"] == "notes.txt"
        assert c.metadata["chunk"] == i
        assert c.metadata["chunk_id"] == f"notes.txt#{i}"
        assert text[c.metadata["start"] : c.metadata["end"]] == c.page_content
        assert len(c.page_content) <= 200


def test_chunk_size_changes_the_number_of_chunks(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("word " * 600, encoding="utf-8")
    docs = load_documents([f])
    assert len(split_documents(docs, 100, 10)) > len(split_documents(docs, 400, 10))


def test_load_pdf_and_docx_roundtrip(tmp_path):
    from docx import Document as Word
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, "The warranty on a tent is three years.")
    pdf.add_page()
    pdf.multi_cell(0, 8, "Repairs take fifteen working days.")
    pdf.output(str(tmp_path / "w.pdf"))
    word = Word()
    word.add_paragraph("Probation lasts three months.")
    word.save(str(tmp_path / "p.docx"))
    docs = load_documents([tmp_path / "w.pdf", tmp_path / "p.docx"])
    by = {(d.metadata["source"], d.metadata.get("page")): d.page_content for d in docs}
    assert "warranty on a tent" in by[("w.pdf", 0)]
    assert "fifteen working days" in by[("w.pdf", 1)]
    assert "Probation lasts three months" in by[("p.docx", None)]


def test_unsupported_and_empty_files_are_skipped(tmp_path):
    (tmp_path / "a.csv").write_text("x,y")
    (tmp_path / "b.txt").write_text("   \n")
    assert load_documents([tmp_path / "a.csv", tmp_path / "b.txt"]) == []


def test_sanitise_filename():
    assert sanitise_filename("../../etc/passwd") == "passwd"
    assert sanitise_filename("C:\\temp\\my file (1).txt") == "my file _1_.txt"
    assert sanitise_filename(".hidden.txt") == "hidden.txt"
    with pytest.raises(UploadRejected):
        sanitise_filename("///")


def _docx_bytes(with_body=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml" if with_body else "other.xml", "<x/>")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("a.exe", b"MZ", "unsupported"),
        ("a.txt", b"", "empty"),
        ("a.txt", b"x" * 2000, "larger"),
        ("a.pdf", b"not a pdf", "PDF"),
        ("a.docx", b"not a zip", "Word"),
        ("a.docx", _docx_bytes(False), "Word"),
        ("a.txt", b"abc\x00def", "plain text"),
        ("a.txt", b"\xff\xfe\xfa", "UTF-8"),
    ],
)
def test_validate_upload_rejects(name, data, message):
    with pytest.raises(UploadRejected, match=message):
        validate_upload(name, data, max_bytes=1000)


def test_validate_upload_accepts():
    assert validate_upload("ok.txt", "caf\u00e9".encode(), 100) == "ok.txt"
    assert validate_upload("x.pdf", b"%PDF-1.4 ...", 100) == "x.pdf"
    assert validate_upload("y.docx", _docx_bytes(), 10_000) == "y.docx"
