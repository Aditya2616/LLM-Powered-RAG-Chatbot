"""Build the Ask-Aditya index: python scripts/build_ask_index.py --resume R.pdf --corpus C.md --readmes DIR [DIR ...]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from askaditya.build import build  # noqa: E402


def bundle() -> None:
    """Copy the service code next to ask/app.py. The copies are git-ignored; src/ stays the source of truth."""
    import shutil

    ask = ROOT / "ask"
    for target in (ask / "askaditya", ask / "rag_chatbot"):
        if target.exists():
            shutil.rmtree(target)
    shutil.copytree(
        ROOT / "src" / "askaditya", ask / "askaditya", ignore=shutil.ignore_patterns("__pycache__")
    )
    (ask / "rag_chatbot").mkdir()
    for name in (
        "__init__.py",
        "text.py",
        "bm25.py",
    ):  # the light modules the service shares with the RAG chatbot
        shutil.copy2(ROOT / "src" / "rag_chatbot" / name, ask / "rag_chatbot" / name)
    size = sum(
        f.stat().st_size for f in ask.rglob("*") if f.is_file() and "__pycache__" not in f.parts
    )
    print(
        f"bundle ready in {ask} ({size / 1e6:.1f} MB); deploy with: cd ask && vercel deploy --prod"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--readmes", required=True, nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "ask" / "private" / "index")
    ap.add_argument(
        "--bundle", action="store_true", help="also copy the service code into ask/ for deployment"
    )
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=80)
    args = ap.parse_args()
    version = build(args.resume, args.corpus, args.readmes, args.out, args.chunk_size, args.overlap)
    print(json.dumps({k: v for k, v in version.items() if k != "sources"}, indent=1))
    print(f"wrote {args.out} ({len(version['sources'])} sources)")
    if args.bundle:
        bundle()
    return 0


if __name__ == "__main__":
    sys.exit(main())
