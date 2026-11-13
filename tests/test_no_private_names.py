"""The tracked tree must not contain the name of the assistant product that only the private index may use."""

import os
import subprocess

from tests.ask_fakes import ROOT

FORBIDDEN = "cl" + "aude"


def tracked_text():
    names = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=ROOT
    ).stdout.split("\n")
    for name in filter(None, names):
        path = ROOT / name
        if path.suffix in {".png", ".jpg", ".webp", ".faiss", ".pdf"} or not path.is_file():
            continue
        yield name, path.read_text(encoding="utf-8", errors="ignore").lower()


def test_no_tracked_file_contains_the_forbidden_name():
    hits = [n for n, text in tracked_text() if FORBIDDEN in text]
    assert hits == []


def test_no_tracked_file_contains_the_configured_platform_name():
    configured = (
        (os.environ.get("ASK_PLATFORM_NAME") or os.environ.get("ASSISTANT_PLATFORM_NAME") or "")
        .strip()
        .lower()
    )
    if not configured:
        return  # nothing configured here: the check above covers the known name
    assert [n for n, text in tracked_text() if configured in text] == []
