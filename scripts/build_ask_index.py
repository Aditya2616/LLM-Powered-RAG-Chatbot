"""Build the Ask-Aditya index: python scripts/build_ask_index.py --resume R.pdf --corpus C.md --readmes DIR [DIR ...]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from askaditya.build import build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--readmes", required=True, nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "ask" / "private" / "index")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=80)
    args = ap.parse_args()
    version = build(args.resume, args.corpus, args.readmes, args.out, args.chunk_size, args.overlap)
    print(json.dumps({k: v for k, v in version.items() if k != "sources"}, indent=1))
    print(f"wrote {args.out} ({len(version['sources'])} sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
