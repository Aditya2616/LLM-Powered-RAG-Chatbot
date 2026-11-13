"""Entry point for the Vercel Python runtime (and for `uvicorn app:app` run from this directory).

`scripts/build_ask_index.py --bundle` copies the service code next to this file and writes the index into
private/index; both are git-ignored and are included in the deployment because .vercelignore is present.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from askaditya.api import create_app  # noqa: E402
from askaditya.config import AskSettings  # noqa: E402

app = create_app(
    AskSettings.from_env({"ASK_INDEX_DIR": str(HERE / "private" / "index"), **os.environ})
)
