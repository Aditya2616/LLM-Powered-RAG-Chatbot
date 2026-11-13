import json

from askaditya import build as b
from askaditya.index import AskIndex
from tests.ask_fakes import PERSONA, build_persona


def test_the_index_and_version_are_written(tmp_path):
    version = build_persona(tmp_path / "idx", tmp_path)
    assert (tmp_path / "idx" / "index.json").is_file() and (tmp_path / "idx" / "VERSION").is_file()
    v = json.loads((tmp_path / "idx" / "VERSION").read_text(encoding="utf-8"))
    assert v == version
    assert (
        v["corpus_version"].startswith("2026-01-02-")
        and len(v["corpus_version"]) == len("2026-01-02-") + 8
    )
    assert set(v["sources"]) == {
        "resume",
        "corpus",
        "platform_name",
        "readme:Atlas",
        "readme:Beacon",
    }
    assert all(len(h) == 64 for k, h in v["sources"].items() if k != "platform_name")
    idx = AskIndex.load(tmp_path / "idx")
    assert idx.version == v["corpus_version"] and len(idx.chunks) == v["chunks"] > 5


def test_the_version_changes_when_a_source_changes(tmp_path):
    a = build_persona(tmp_path / "a", tmp_path)
    readme = tmp_path / "proj" / "Atlas"
    readme.mkdir(parents=True)
    (readme / "README.md").write_text("# Atlas\nA different text.", encoding="utf-8")
    from tests.ask_fakes import make_resume_pdf

    make_resume_pdf(tmp_path / "r.pdf")
    c = b.build(tmp_path / "r.pdf", PERSONA / "corpus.md", [readme], tmp_path / "c")
    assert a["corpus_version"] != c["corpus_version"]


def test_the_phone_number_never_enters_the_index(tmp_path):
    number = "+91 " + "98765" + " " + "43210"  # built here, never stored in a fixture
    build_persona(
        tmp_path / "idx", tmp_path, extra_lines=(f"Springfield | {number} | sam@example.test",)
    )
    raw = (tmp_path / "idx" / "index.json").read_text(encoding="utf-8")
    assert "98765" not in raw and "43210" not in raw
    chunks_text = " ".join(c["text"] for c in AskIndex.load(tmp_path / "idx").chunks)
    assert not b.phone_like(chunks_text.replace(chr(10), " "))
    resume = [c for c in AskIndex.load(tmp_path / "idx").chunks if c["title"] == "Résumé"]
    assert resume and "sam.rivera@example.test" in " ".join(c["text"] for c in resume)


def test_phone_like_and_strip():
    assert (
        b.phone_like("call +91 98765 43210 now")
        and b.phone_like("9876543210")
        and b.phone_like("(011) 2345-6789")
    )
    assert not b.phone_like("rows 11000-12000, dated 2025-11-13, 408 prompts, Feb 2026 - Present")
    assert b.strip_phones("a 98765 43210 b") == "a  b"


def test_the_placeholder_is_replaced_by_the_default_or_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("ASK_PLATFORM_NAME", raising=False)
    build_persona(tmp_path / "d", tmp_path)
    text = (tmp_path / "d" / "index.json").read_text(encoding="utf-8")
    assert "ASSISTANT-PLATFORM" not in text and b.DEFAULT_PLATFORM_NAME in text
    monkeypatch.setenv("ASK_PLATFORM_NAME", "Platform Name Example")
    build_persona(tmp_path / "e", tmp_path)
    text = (tmp_path / "e" / "index.json").read_text(encoding="utf-8")
    assert "Platform Name Example" in text and "ASSISTANT-PLATFORM" not in text


def test_the_name_can_come_from_a_git_ignored_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ASK_PLATFORM_NAME", raising=False)
    out = tmp_path / "idx"
    out.mkdir()
    (out / "platform_name.txt").write_text("Local File Name\n", encoding="utf-8")
    assert b.platform_name(out) == "Local File Name"


def test_maintainer_notes_and_markup_do_not_reach_the_index(tmp_path):
    build_persona(tmp_path / "idx", tmp_path)
    raw = (tmp_path / "idx" / "index.json").read_text(encoding="utf-8")
    for junk in [
        "maintainers",
        "generated comment",
        "flowchart",
        "![diagram]",
        "example.test/ci.svg",
        "NOTE:",
        "must NOT",
    ]:
        assert junk not in raw, junk


def test_false_claims_become_traps_and_true_statements(tmp_path):
    build_persona(tmp_path / "idx", tmp_path)
    data = json.loads((tmp_path / "idx" / "index.json").read_text(encoding="utf-8"))
    assert {t["claim"] for t in data["traps"]} == {
        "Atlas fixes every spreadsheet automatically",
        "Beacon forecasts live demand",
    }
    text = " ".join(c["text"] for c in data["chunks"])
    assert (
        "It is not true that Atlas fixes every spreadsheet automatically. In fact: it flags rows it cannot repair."
        in text
    )


def test_every_chunk_carries_a_title_and_a_url(tmp_path):
    build_persona(tmp_path / "idx", tmp_path)
    for c in AskIndex.load(tmp_path / "idx").chunks:
        assert c["title"] and c["url"].startswith("https://") and c["text"].strip()
    urls = {c["url"] for c in AskIndex.load(tmp_path / "idx").chunks}
    assert "https://github.com/Aditya2616/Atlas" in urls


def test_chunks_are_small_enough_for_the_prompt(tmp_path):
    build_persona(tmp_path / "idx", tmp_path)
    assert max(len(c["text"]) for c in AskIndex.load(tmp_path / "idx").chunks) <= 500


def test_markdown_cleaning():
    md = "[![ci](a.svg)](x)\n# T\n<!-- c -->\n<img src='a'>\ntext [link](http://u) more\n|---|---|\n| a | b |\n```mermaid\nx\n```"
    out = b.clean_markdown(md)
    assert (
        "<!--" not in out
        and "<img" not in out
        and "http://u" not in out
        and "mermaid" not in out
        and "---" not in out
    )
    assert "text link more" in out and "| a | b |" in out


def test_sections_follow_the_heading_path():
    secs = b.split_sections("# A\nintro\n## B\nbody\n### C\ndeep\n## D\nmore", "Doc")
    assert [h for h, _ in secs] == ["A", "A > B", "A > B > C", "A > D"]
