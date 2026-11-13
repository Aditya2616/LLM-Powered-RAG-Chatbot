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

A question's prompt is the fixed instructions plus five retrieved chunks of at most 500 characters. Measured on the
Groq models (`usage` in the response): about 620 to 730 prompt tokens and 25 to 130 completion tokens, so roughly 650 to
850 tokens a question, well under 1,500. Budgets use a conservative estimate (one token per 3.5 characters plus the whole
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
