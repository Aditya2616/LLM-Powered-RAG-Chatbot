"""Load the interface in headless Chromium and check it, then capture the README screenshots of the static demo.

    python scripts/check_ui.py [--shots docs/assets] [--live]

Checks at 1440x900 and 390x844 (touch): no console errors or page errors, no horizontal scroll, every
button, link and input at least 44 px tall (and wide, for buttons), for the static page served from
docs/ and (with --live) for the page served by the FastAPI app with the offline providers.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SIZES = {
    "desktop": dict(viewport={"width": 1440, "height": 900}),
    "phone": dict(
        viewport={"width": 390, "height": 844},
        device_scale_factor=2,
        has_touch=True,
        is_mobile=True,
    ),
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def static_server():
    port = free_port()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        srv.shutdown()


@contextlib.contextmanager
def live_server(tmp: Path):
    port = free_port()
    env = dict(
        os.environ, EMBEDDINGS_PROVIDER="hashing", LLM_PROVIDER="extractive", RAG_DATA_DIR=str(tmp)
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "rag_chatbot.cli", "serve", "--port", str(port)],
        env=env,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/"
    try:
        for _ in range(60):
            try:
                import urllib.request

                urllib.request.urlopen(url + "health", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


SMALL = """() => {
  const bad = [];
  for (const el of document.querySelectorAll('button, a, input, .qchip, .cite')) {
    const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
    if (!r.width || !r.height || cs.visibility === 'hidden' || cs.display === 'none') continue;
    if (r.bottom < 0 || el.closest('[hidden]')) continue;
    const isBtn = el.tagName === 'BUTTON' || el.tagName === 'INPUT';
    if (r.height < 43.5 || (isBtn && r.width < 43.5)) bad.push(`${el.tagName}.${el.className} ${Math.round(r.width)}x${Math.round(r.height)} "${(el.textContent || '').trim().slice(0, 24)}"`);
  }
  return bad;
}"""


def shoot(page, path: Path, scroll_sel: str, last_sel: str, pad: int = 3) -> None:
    """Scroll an element to just under the sticky bar and capture down to the bottom of another element,
    so the picture ends on a card boundary and nothing is clipped or hidden under the bar."""
    page.evaluate("document.documentElement.style.scrollBehavior = 'auto'")
    page.evaluate(
        "(s) => { const e = document.querySelector(s); window.scrollTo(0, e.getBoundingClientRect().top + scrollY - 76); }",
        scroll_sel,
    )
    page.wait_for_timeout(1500)
    vp = page.viewport_size
    bottom = page.evaluate(
        "(s) => document.querySelector(s).getBoundingClientRect().bottom", last_sel
    )
    height = min(vp["height"], int(bottom) + pad)
    page.screenshot(path=str(path), clip={"x": 0, "y": 0, "width": vp["width"], "height": height})


def run_one(p, name: str, url: str, size: str, shots: Path | None, live: bool) -> list[str]:
    problems: list[str] = []
    browser = p.chromium.launch()
    ctx = browser.new_context(**SIZES[size])
    page = ctx.new_page()
    logs: list[str] = []
    page.on(
        "console",
        lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None,
    )
    page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    page.on(
        "requestfailed",
        lambda r: logs.append(f"requestfailed: {r.url}") if "fonts.g" not in r.url else None,
    )
    page.goto(url, wait_until="load")
    page.wait_for_function("document.querySelector('#qchips .qchip')", timeout=20000)
    if live:
        if page.locator("#loadSample").is_visible():
            page.click("#loadSample")
            page.wait_for_selector("#loadSample", state="hidden", timeout=30000)
        page.fill("#liveQ", "When are PayPal refunds issued?")
        page.click("#liveForm button[type=submit]")
        page.wait_for_function(
            "['answered','declined'].includes(document.querySelector('#stateChip').textContent)",
            timeout=30000,
        )
    else:
        page.wait_for_function(
            "document.querySelector('#stateChip').textContent === 'answered'", timeout=60000
        )
    page.wait_for_timeout(1200)
    chunk = "#chunk-1" if size == "phone" else "#chunk-3"
    if shots and not live:
        if size == "desktop":
            shoot(page, shots / "desktop-answered-map.png", "#rail", ".ask")
        shoot(
            page,
            shots / f"{size}-answered.png",
            "#answerCard",
            "#answerCard" if size == "phone" else chunk,
        )
    # a question the documents cannot answer
    if not live:
        idx = page.evaluate(
            "[...document.querySelectorAll('.qchip')].findIndex(b => b.classList.contains('decline'))"
        )
        page.click(f".qchip[data-i='{idx}']")
        page.wait_for_function(
            "document.querySelector('#stateChip').textContent === 'declined'", timeout=60000
        )
        page.wait_for_timeout(1500)
        if shots:
            shoot(page, shots / f"{size}-declined.png", "#answerCard", chunk)
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    if overflow > 0:
        problems.append(f"horizontal overflow by {overflow}px")
    small = page.evaluate(SMALL)
    if small:
        problems.append("targets under 44px: " + "; ".join(small[:6]))
    for entry in logs:
        problems.append(entry)
    browser.close()
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--only", default=None, help="desktop or phone")
    args = ap.parse_args()
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    report: dict[str, list[str]] = {}
    with sync_playwright() as p:
        with static_server() as url:
            for size in SIZES:
                if args.only in (None, size):
                    report[f"static/{size}"] = run_one(p, "static", url, size, shots, False)
        if args.live:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp, live_server(Path(tmp)) as url:
                for size in SIZES:
                    if args.only in (None, size):
                        print("served", size, flush=True)
                        report[f"served/{size}"] = run_one(p, "served", url, size, None, True)
    print(json.dumps(report, indent=1))
    return 1 if any(report.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
