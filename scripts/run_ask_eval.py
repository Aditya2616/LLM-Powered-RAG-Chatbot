"""Evaluate the Ask-Aditya HTTP API on eval/ask_aditya_eval.jsonl.

    # against a running service (local or deployed)
    python scripts/run_ask_eval.py --base-url https://<deployment> --tag live [--eval-token T]

    # start a local server from the built index, measure cold start, and probe 429 / 503 / CORS
    python scripts/run_ask_eval.py --spawn --tag local-qwen1.5b \
        --env ASK_PROVIDER_ORDER=local --env ASK_LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct

It talks to the real HTTP API only. For every question it checks the contract (status, SSE framing, the
fields of ``done``, the tokens adding up to the answer) and the question's own expectations (facts it must
contain, claims it must not make, the expected fallback, the expected source), and it measures the time to
the first ``token`` event. Results go to results/ask_eval_<tag>.json; check_numbers.py renders the README table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from askaditya.build import phone_like  # noqa: E402

FALLBACKS = {None, "unknown", "offtopic", "refused", "limit"}
FORBIDDEN = "cl" + "aude"
GROUPS = ["factual", "unpublished", "trap", "injection", "offtopic"]


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‑", "-").replace("–", "-")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"(\d)\s+%", r"\1%", s)
    return re.sub(r"\s+", " ", s)


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = (len(s) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def ask(client: httpx.Client, base: str, q: dict, headers: dict) -> dict:
    """POST one question and return status, parsed events, timings and contract problems."""
    t0 = time.perf_counter()
    body = {"question": q["question"], "history": q.get("history", [])}
    out = {"status": None, "problems": [], "first_token_ms": None, "total_ms": None, "events": []}
    try:
        with client.stream("POST", f"{base}/api/ask", json=body, headers=headers, timeout=60) as r:
            out["status"] = r.status_code
            out["stage"] = r.headers.get("x-ask-stage")
            out["provider"] = r.headers.get("x-ask-provider")
            out["tokens"] = r.headers.get("x-ask-tokens")
            out["reason"] = r.headers.get("x-ask-reason")
            ctype = r.headers.get("content-type", "")
            if r.status_code != 200:
                r.read()
                out["body"] = r.text[:300]
                out["total_ms"] = (time.perf_counter() - t0) * 1000
                return out
            if not ctype.startswith("text/event-stream"):
                out["problems"].append(f"content-type {ctype}")
            buf = ""
            for chunk in r.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    lines = block.split("\n")
                    if (
                        len(lines) != 2
                        or not lines[0].startswith("event: ")
                        or not lines[1].startswith("data: ")
                    ):
                        out["problems"].append(f"bad SSE block {block[:60]!r}")
                        continue
                    name, data = lines[0][7:], json.loads(lines[1][6:])
                    if name == "token" and out["first_token_ms"] is None:
                        out["first_token_ms"] = (time.perf_counter() - t0) * 1000
                    out["events"].append((name, data))
            if buf.strip():
                out["problems"].append("trailing bytes after the last event")
    except (httpx.HTTPError, ValueError) as exc:
        out["problems"].append(f"{type(exc).__name__}: {exc}")
    out["total_ms"] = (time.perf_counter() - t0) * 1000
    return out


def check_contract(res: dict) -> dict | None:
    ev = res["events"]
    if not ev or ev[-1][0] != "done" or any(n != "token" for n, _ in ev[:-1]):
        res["problems"].append("events are not token* then done")
        return None
    done = ev[-1][1]
    for key, typ in [("answer", str), ("sources", list), ("grounded", bool)]:
        if not isinstance(done.get(key), typ):
            res["problems"].append(f"done.{key} is not {typ.__name__}")
    if "fallback" not in done or done["fallback"] not in FALLBACKS:
        res["problems"].append(f"done.fallback {done.get('fallback')!r}")
    for s in done.get("sources", []):
        if set(s) != {"title", "url"}:
            res["problems"].append("a source is not {title,url}")
    if not all(set(d) == {"t"} and isinstance(d["t"], str) for n, d in ev[:-1]):
        res["problems"].append("a token event is not {t}")
    if "".join(d["t"] for n, d in ev[:-1]) != done.get("answer"):
        res["problems"].append("tokens do not add up to the answer")
    return done


def judge(q: dict, res: dict) -> list[str]:
    """The failures (empty = pass) for one question."""
    fails = list(res["problems"])
    if res["status"] != 200:
        return fails + [f"HTTP {res['status']}"]
    done = check_contract(res)
    fails = list(res["problems"])
    if done is None:
        return fails
    answer = done["answer"]
    low = norm(answer)
    if phone_like(answer):
        fails.append("a phone-like digit run in the answer")
    if FORBIDDEN in low:
        fails.append("forbidden product name in the answer")
    if len(answer.split()) > 110:
        fails.append("longer than 110 words")
    if done["fallback"] != q["expect_fallback"]:
        fails.append(f"fallback {done['fallback']!r}, expected {q['expect_fallback']!r}")
    if done["grounded"] != q["expect_grounded"]:
        fails.append(f"grounded {done['grounded']}, expected {q['expect_grounded']}")
    for item in q["must_contain"]:
        alts = item if isinstance(item, list) else [item]
        if not any(norm(a) in low for a in alts):
            fails.append(f"missing {alts}")
    for pat in q["must_not_contain"]:
        if re.search(pat, answer, flags=re.I):
            fails.append(f"contains forbidden pattern {pat!r}")
    if q["group"] in ("factual", "trap") and done["fallback"] is None:
        if not done["sources"]:
            fails.append("grounded answer without sources")
        elif q["expect_source"]:
            hay = " ".join(f"{s['title']} {s['url']}" for s in done["sources"]).lower()
            if not any(e.lower() in hay for e in q["expect_source"]):
                fails.append(f"no source matching {q['expect_source']}")
    return fails


def _platform_names() -> list[str]:
    """The product name that the private index uses; it must never be stored in a results file."""
    names = [
        os.environ.get(k, "").strip() for k in ("ASK_PLATFORM_NAME", "ASSISTANT_PLATFORM_NAME")
    ]
    return [n for n in names if n]


def scrub(value):
    """Replace the private product name in any string inside ``value`` (case-insensitively)."""
    names = _platform_names()
    if not names or value is None:
        return value
    if isinstance(value, str):
        for n in names:
            value = re.sub(re.escape(n), "[PLATFORM]", value, flags=re.I)
        return value
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, env_extra: dict[str, str]) -> None:
        self.port = free_port()
        env = dict(os.environ)
        env.update(env_extra)
        env.setdefault("ASK_INDEX_DIR", str(ROOT / "ask" / "private" / "index"))
        code = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import uvicorn; from askaditya.api import create_app; uvicorn.run(create_app(), host='127.0.0.1', port={self.port}, log_level='warning')"
        self.t_spawn = time.perf_counter()
        self.proc = subprocess.Popen(
            [sys.executable, "-c", code],
            env=env,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.base = f"http://127.0.0.1:{self.port}"

    def wait_ready(self, timeout: float = 60) -> float:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if httpx.get(f"{self.base}/api/health", timeout=1).status_code == 200:
                    return (time.perf_counter() - self.t_spawn) * 1000
            except httpx.HTTPError:
                time.sleep(0.2)
        raise RuntimeError("the server did not become ready")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def probes(env: dict[str, str]) -> dict:
    """Contract behaviour that needs its own server settings: 429, 503 limit, CORS, bad input."""
    out: dict = {}
    q = {"question": "Where does Aditya work right now?", "history": []}
    base_env = {k: v for k, v in env.items()}
    # 429: three requests a minute
    s = Server({**base_env, "ASK_IP_PER_MINUTE": "2"})
    try:
        s.wait_ready()
        with httpx.Client(timeout=180) as c:
            cheap = {
                "question": "What is his salary?",
                "history": [],
            }  # a fixed reply: no generation to slow the probe down
            codes = [ask(c, s.base, cheap, {})["status"] for _ in range(3)]
            r = c.post(f"{s.base}/api/ask", json={"question": "x"})
            out["rate_limit"] = {
                "statuses": codes,
                "retry_after_header": r.headers.get("retry-after"),
                "pass": codes == [200, 200, 429] and r.status_code == 429,
            }
    finally:
        s.stop()
    # 503 limit: one generation allowed per day, then the cap is spent; a deterministic answer still works
    s = Server({**base_env, "ASK_GLOBAL_DAILY_CAP": "1", "ASK_TEST_HOOKS": "1"})
    try:
        s.wait_ready()
        with httpx.Client(timeout=180) as c:
            first = ask(c, s.base, q, {})
            second = c.post(f"{s.base}/api/ask", json=q)
            hook = c.post(f"{s.base}/api/ask", json=q, headers={"x-ask-test": "limit"})
            salary = ask(c, s.base, {"question": "What is his salary?", "history": []}, {})
            body = (
                second.json()
                if second.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            hbody = (
                hook.json()
                if hook.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            out["limit"] = {
                "first_status": first["status"],
                "cap_spent_status": second.status_code,
                "cap_spent_fallback": body.get("fallback"),
                "hook_status": hook.status_code,
                "hook_fallback": hbody.get("fallback"),
                "deterministic_still_answers": salary["status"] == 200,
                "sample_body": body,
                "pass": first["status"] == 200
                and second.status_code == 503
                and body.get("fallback") == "limit"
                and hook.status_code == 503
                and hbody.get("fallback") == "limit"
                and salary["status"] == 200,
            }
    finally:
        s.stop()
    # CORS and 400
    s = Server(base_env)
    try:
        s.wait_ready()
        with httpx.Client(timeout=180) as c:
            ok = c.options(
                f"{s.base}/api/ask",
                headers={
                    "Origin": "https://adityajain16.vercel.app",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            ok2 = c.get(f"{s.base}/api/health", headers={"Origin": "http://127.0.0.1:8765"})
            bad = c.get(f"{s.base}/api/health", headers={"Origin": "https://evil.example"})
            e400 = [
                c.post(f"{s.base}/api/ask", json=b).status_code
                for b in (
                    {},
                    {"question": ""},
                    {"question": "x" * 501},
                    {"question": "a", "history": [{"role": "x", "text": "y"}]},
                )
            ]
            out["cors"] = {
                "preflight_allowed": ok.headers.get("access-control-allow-origin")
                == "https://adityajain16.vercel.app"
                and ok.status_code == 200,
                "local_origin_allowed": ok2.headers.get("access-control-allow-origin")
                == "http://127.0.0.1:8765",
                "other_origin_blocked": "access-control-allow-origin" not in bad.headers,
                "bad_input_statuses": e400,
            }
            out["cors"]["pass"] = all(
                [
                    out["cors"]["preflight_allowed"],
                    out["cors"]["local_origin_allowed"],
                    out["cors"]["other_origin_blocked"],
                    e400 == [400, 400, 400, 400],
                ]
            )
    finally:
        s.stop()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default=None)
    ap.add_argument(
        "--spawn", action="store_true", help="start a local server from ask/private/index"
    )
    ap.add_argument(
        "--env", action="append", default=[], help="KEY=VALUE for the spawned server (repeatable)"
    )
    ap.add_argument("--eval", type=Path, default=ROOT / "eval" / "ask_aditya_eval.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-token", default=os.environ.get("ASK_EVAL_TOKEN", ""))
    ap.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="seconds between questions (for a deployment with per-IP limits)",
    )
    ap.add_argument(
        "--note", default="", help="free text recorded with the run (model, device, load)"
    )
    ap.add_argument(
        "--pause-generated",
        type=float,
        default=0.0,
        help="seconds to wait after a question that reached a model (token-per-minute budgets)",
    )
    ap.add_argument(
        "--ids", default="", help="comma-separated question ids to run (iterate on failures)"
    )
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    args = ap.parse_args()

    env = dict(kv.split("=", 1) for kv in args.env)
    questions = [
        json.loads(line)
        for line in args.eval.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]
    server = None
    cold = {}
    if args.spawn:
        env.setdefault("ASK_IP_PER_MINUTE", "100000")
        env.setdefault("ASK_IP_PER_DAY", "100000")
        env.setdefault("ASK_GLOBAL_DAILY_CAP", "100000")
        env.setdefault("ASK_PROVIDER_DAILY_CAPS", "local=100000,groq=100000,gemini=100000")
        server = Server(env)
        cold["ready_ms"] = server.wait_ready()
        base = server.base
    elif args.base_url:
        base = args.base_url.rstrip("/")
    else:
        ap.error("give --base-url or --spawn")
    headers = {"x-ask-eval": args.eval_token} if args.eval_token else {}
    results = []
    try:
        with httpx.Client() as c:
            health = c.get(f"{base}/api/health", timeout=30).json()
            for i, q in enumerate(questions):
                res = ask(c, base, q, headers)
                if i == 0:
                    cold["first_ask_ms"] = res["total_ms"]
                    cold["first_ask_first_token_ms"] = res["first_token_ms"]
                fails = judge(q, res)
                done = (
                    res["events"][-1][1] if res["events"] and res["events"][-1][0] == "done" else {}
                )
                results.append(
                    {
                        "id": q["id"],
                        "group": q["group"],
                        "question": q["question"],
                        "pass": not fails,
                        "failures": fails,
                        "status": res["status"],
                        "stage": res.get("stage"),
                        "reason": res.get("reason"),
                        "first_token_ms": res["first_token_ms"],
                        "total_ms": res["total_ms"],
                        "provider": res.get("provider"),
                        "tokens_prompt_completion": res.get("tokens"),
                        "answer": scrub(done.get("answer")),
                        "fallback": done.get("fallback"),
                        "grounded": done.get("grounded"),
                        "sources": scrub(done.get("sources")),
                    }
                )
                print(
                    f"{q['id']:>4} {'PASS' if not fails else 'FAIL'} {q['group']:11s} {res['first_token_ms'] or 0:7.0f} ms  {'; '.join(fails)[:100]}",
                    flush=True,
                )
                if args.pause:
                    time.sleep(args.pause)
                if args.pause_generated and res.get("stage") in ("model", "verify", "trap"):
                    time.sleep(args.pause_generated)
    finally:
        if server:
            server.stop()
    try:
        extra = {} if args.no_probes or not args.spawn else probes(env)
    except Exception as exc:  # a probe failure must not lose the question results
        extra = {"error": {"pass": False, "detail": f"{type(exc).__name__}: {exc}"}}
    groups = {}
    for g in GROUPS:
        rs = [r for r in results if r["group"] == g]
        groups[g] = {"passed": sum(r["pass"] for r in rs), "n": len(rs)}
    warm = [r["first_token_ms"] for r in results[1:] if r["first_token_ms"] is not None]
    model_rs = [
        r["first_token_ms"]
        for r in results[1:]
        if r["first_token_ms"] is not None and r["group"] in ("factual", "trap")
    ]
    summary = {
        "meta": {
            "tag": args.tag,
            "note": args.note,
            "date": dt.date.today().isoformat(),
            "mode": "spawned local server" if args.spawn else "base url",
            "corpus_version": health.get("corpus_version"),
            "n_questions": len(results),
            "server_env": {
                k: v
                for k, v in env.items()
                if "KEY" not in k
                and "TOKEN" not in k
                and not k.startswith("ASK_IP")
                and "CAP" not in k
            },
        },
        "groups": groups,
        "total": {"passed": sum(r["pass"] for r in results), "n": len(results)},
        "first_token_ms": {
            "mean_all": sum(warm) / len(warm) if warm else None,
            "p95_all": percentile(warm, 0.95),
            "mean_generated": sum(model_rs) / len(model_rs) if model_rs else None,
            "p95_generated": percentile(model_rs, 0.95),
            "n_all": len(warm),
            "n_generated": len(model_rs),
            "first_request_excluded": True,
        },
        "cold_start": cold,
        "probes": extra,
        "results": results,
    }
    summary = scrub(summary)
    blob = json.dumps(summary, ensure_ascii=False)
    assert not any(n.lower() in blob.lower() for n in _platform_names()), (
        "the product name reached the results"
    )
    assert FORBIDDEN not in blob.lower(), "the forbidden name appeared in the results"
    out = ROOT / "results" / f"ask_eval_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "groups": groups,
                "total": summary["total"],
                "first_token_ms": summary["first_token_ms"],
                "cold": cold,
                "probes": {k: v.get("pass") for k, v in extra.items()},
            },
            indent=1,
        )
    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
