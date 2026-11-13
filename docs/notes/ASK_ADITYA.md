# Ask-Aditya: provider and platform facts

Looked up on 2026-11-13 from the official pages named below. Limits change; re-check before relying on a number.
The service never enables billing: it only uses free-tier keys, keeps its own request and token budgets so it stops
asking before a provider would answer 429, and ends every request path in a 503 `limit` response when the free quota
is gone.

## The chain

1. **Groq, primary model** (`GROQ_MODEL`).
2. **Groq, backup model** (`GROQ_MODEL_BACKUP`), because Groq's limits apply per model, so a second model is a second
   budget on the same key.
3. **Gemini, free tier** (`GEMINI_MODEL`), strictly last: it only sees traffic when both Groq models are out of budget
   or answering errors. `ASK_DISABLE_GEMINI=1` switches it off without a code change.

No model id is in the repository; all three come from the environment. When none is left, the API answers 503 with
`fallback: "limit"`. The same output verification applies to every provider.

## Groq

- Free tier: the billing FAQ (https://console.groq.com/docs/billing-faqs) says moving from the Free tier to the
  Developer tier "you'll need to provide a valid payment method", that a Developer account can go back to the Free tier,
  and the spend-limit page (https://console.groq.com/docs/spend-limits) says spend limits exist only on paid plans.
- Rate limits (https://console.groq.com/docs/rate-limits): for `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and
  `qwen/qwen3.8-27b` the page lists 30 requests/min, 1,000 requests/day, 8,000 tokens/min and 200,000 tokens/day. It calls
  these a summary with exceptions; the exact limits for an account are on https://console.groq.com/settings/limits.
  Exceeding a limit returns HTTP 429 with a `retry-after` header.
- Models visible to the project's key on 2026-11-13 (its `/openai/v1/models` response): `openai/gpt-oss-120b`,
  `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.8-27b`, `allam-2-7b`, two Llama prompt-guard models,
  two Orpheus speech models and two Whisper models.
- Endpoint: `https://api.groq.com/openai/v1/chat/completions` (OpenAI-compatible).
- **Not verified:** whether the free plan requires a card at sign-up (the FAQ only says a card is needed to upgrade) and
  the wording of what happens on overage on the free plan (the documented behaviour is HTTP 429); the free tier's
  data-retention terms were not read.

## Google Gemini API, key without billing (last resort)

- Free tier prices (https://ai.google.dev/gemini-api/docs/pricing): `gemini-3.5-flash-lite` is "Free of charge" for
  input and output. Going to a paid tier requires setting up billing first (https://ai.google.dev/gemini-api/docs/rate-limits).
  Per-model daily limits are shown only in AI Studio (https://aistudio.google.com/rate-limit).
- Models with `generateContent` visible to the project's key on 2026-11-13 include `gemini-2.5-flash`,
  `gemini-2.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.5-flash-lite` and several newer preview models.
- **Terms caveat** (https://ai.google.dev/gemini-api/terms, "Unpaid Services" and the introduction): "You may use only
  Paid Services when making API Clients available to users in the European Economic Area, Switzerland, or the United
  Kingdom", and for unpaid quota "Google uses the content you submit to the Services and any generated responses to
  provide, improve, and develop Google products", with human reviewers possibly reading inputs and outputs. A public
  portfolio cannot exclude visitors from those regions, so Gemini on the free tier is kept as the last resort, can be
  switched off, and the README says so. Visitor questions that reach it may be used by Google.

## Vercel Hobby (hosting)

- Python runtime (https://vercel.com/docs/functions/runtimes/python, updated 2026-08-12): ASGI and WSGI apps; versions
  3.12 (default), 3.13, 3.14, set by `.python-version`, `pyproject.toml` or `Pipfile.lock`; entrypoints `app.py`,
  `index.py`, `server.py`, `main.py` (also under `src/` or `app/`) defining a top-level `app`; dependencies from
  `requirements.txt` or `pyproject.toml`; "Vercel Functions support streaming responses when using the Python runtime";
  bundle limit 500 MB uncompressed; by default all project files reachable at build time are included.
- `vercel.json` `functions[...]` takes `maxDuration`, `includeFiles` and `excludeFiles` globs
  (https://vercel.com/docs/project-configuration/vercel-json).
- Found on the first real deploy (2026-11-13): a `functions` pattern for `app.py` is rejected ("doesn't match any Serverless Functions inside the `api` directory"), and without a framework the project deploys as static files (404 on every route). `ask/vercel.json` therefore sets only `"framework": "fastapi"`; the whole project directory, including `private/index`, is bundled with the function (checked: `/api/health` returns the built corpus version).
- `.vercelignore` (https://vercel.com/docs/deployments/vercelignore): everything is uploaded except the default
  exclusions and what `.vercelignore` lists, so a git-ignored build directory is uploaded by `vercel deploy` from a local
  folder. A git integration would not have files that are not in git, so this project is deployed from the CLI.
- Limits (https://vercel.com/docs/functions/limitations): Hobby 300 s maximum duration, 2 GB / 1 vCPU, 4.5 MB request
  payload (the response limit does not apply to streamed responses).

## Upstash Redis (shared rate-limit state)

- Free plan (https://upstash.com/pricing/redis): $0, 250 MB data, 10 GB monthly bandwidth, 500K monthly commands; "Once
  you enter your credit card, your database will be upgraded to the pay-as-you-go plan", so a database without a card
  stays on the free plan.
- A question costs a handful of commands: the per-address counters (2), the global counter (1), the provider request
  counter (1) and the token counters (2, plus 2 when a reservation is corrected), each an INCRBY with an EXPIRE in one
  pipeline request. That is at most about 20 commands a question, so the free monthly allowance covers about 25,000
  questions, far above the daily caps below.

## Token budget and caps

A question's prompt is the fixed instructions plus five retrieved chunks of at most 500 characters. The measured
tokens per question (prompt and completion, from the providers' `usage`) are in the README's token table; the prompt stays
well under 1,500 tokens. Budgets use a conservative estimate (one token per 3.5 characters plus the whole
completion allowance) and are corrected with the reported usage after the call.

| Cap | Default | Why |
|---|---|---|
| per address | 10 / minute, 60 / day | the contract |
| Groq tokens per model | 6,500 / minute, 170,000 / day | below 8,000 and 200,000 |
| Groq requests per model | 150 / day | below 1,000; the tokens bind first |
| Gemini requests | 100 / day | the exact free quota is in AI Studio |
| global | 350 model calls / day | below the sum of the budgets; only calls that reach a model count |

When a model is out of budget or answers 429, errors or times out, the chain moves to the next one; when none is left, the
API answers 503 with `fallback: "limit"`. Deterministic replies (deflections, refusals, the assistant's own description)
never reach a model and never spend a budget.

## Models chosen (measured 2026-11-13 through the real HTTP API on 127.0.0.1)

Model ids live only in the environment; these are the values the evaluation used. The tables are in the README.

- `GROQ_MODEL=qwen/qwen3.8-27b` (primary): fastest and most accurate in the screening and the full run, mean first token
  under 0.3 s, no reasoning-token overhead.
- `GROQ_MODEL_BACKUP=openai/gpt-oss-120b` with `GROQ_BACKUP_REASONING_EFFORT=low`: a reasoning model, so it needs the low
  effort and a completion allowance of 400 tokens to leave room for the reasoning; slower and a little less accurate.
- `GEMINI_MODEL=gemini-3.5-flash-lite` (last resort): slowest of the three (mean first token about 1.3 s, p95 about 5 s),
  and it answered as accurately as the others.
- `openai/gpt-oss-20b` was screened on three questions only (1 to 2 s each, one wrong "UNKNOWN") and not evaluated fully.
- `ASK_MAX_NEW_TOKENS=400` for all three; the token budgets count that allowance in their estimate, so they are
  conservative.

## Evaluation expectations changed since the first version

The first version of `eval/ask_aditya_eval.jsonl` had 84 questions. Changes, each of which accepts only the same fact from
the sources:

| Question | Change | Reason |
|---|---|---|
| f25 (SFT Forge's three gates) | `dedupe` became `dedup` | the README says "dedupe"; "deduplication" is the same word and the same gate |
| f41, t05, t06 (SFT Forge experiment) | added "not find any measurable" and "no measurable benefit" | the source says the experiment found no measurable benefit; a model that writes "did not find any measurable improvement" states the same fact |
| f45, f46 (added) | two questions about the identity and administration skills | cover the skill line that names a product; the expectations use neighbouring words (SSO, SCIM, Enterprise Administration) so the product name stays out of the repository |

No expectation was relaxed to make a model pass: the remaining misses (f21 on all three hosted legs, f18 on the backup)
are retrieval and model-declined cases and still count as failures.

## Security of the evaluation bypass and the test hooks

- `ASK_EVAL_TOKEN` is compared with `hmac.compare_digest` on bytes (a non-ASCII header cannot raise), is ignored when it is
  not set, is never logged, and only skips the per-address limits: validation (400), the injection screen, the topic
  deflections and the global cap still apply (tests: `test_eval_token_security`).
- `ASK_TEST_HOOKS` is read from the environment only (`ASK_TEST_HOOKS=1`); no request header can enable it
  (`test_test_hooks_cannot_be_enabled_by_a_request`). Leave it unset in production.
