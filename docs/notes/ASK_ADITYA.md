# Ask-Aditya: provider and platform facts

Looked up on 2026-11-13 from the official pages named below. Limits change; re-check before relying on a number.
The service never enables billing: it only uses free-tier keys, and every request path ends in a 503 `limit`
response when the free quota is gone.

## Groq (first provider in the chain)

- Free tier exists. Billing FAQ (https://console.groq.com/docs/billing-faqs): "To upgrade from the Free tier to the
  Developer tier, you'll need to provide a valid payment method", and a Developer account can downgrade to the Free
  tier. Spend limits are a paid-plan feature (https://console.groq.com/docs/spend-limits), so on the free tier there is
  nothing that can bill.
- Rate limits (https://console.groq.com/docs/rate-limits, markdown view). Rows listed on the page, per model id:
  `openai/gpt-oss-20b` 30 requests/min, 1,000 requests/day, 8,000 tokens/min, 200,000 tokens/day; the same four numbers
  for `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`. The page calls them a high-level summary with exceptions, and says
  the exact limits for an organisation are on https://console.groq.com/settings/limits (needs the account).
- Exceeding a limit returns HTTP 429 with a `retry-after` header.
- Endpoint: `https://api.groq.com/openai/v1/chat/completions` (OpenAI-compatible).
- Not checked: the data-retention terms of the free tier.

## Google Gemini API, AI Studio key without billing (second provider)

- Pricing page (https://ai.google.dev/gemini-api/docs/pricing): `gemini-3.5-flash-lite` and `gemini-embedding-001`
  list "Free of charge" in the Free Tier column for input and output tokens. The same page lists "Used to improve our
  products: Yes" for the free tier and "No" for the paid tier, so questions sent through a free key may be used by
  Google. Visitors' questions go to this provider, which is worth stating on the portfolio.
- Rate limits (https://ai.google.dev/gemini-api/docs/rate-limits): limits are per project, not per key; requests per
  day reset at midnight Pacific time; the per-model RPM, TPM and RPD numbers are shown in AI Studio
  (https://aistudio.google.com/rate-limit) for the signed-in project and are not printed on the docs page.
  Going from the Free tier to a paid tier requires setting up billing first, so a project without billing stays free.
- REST: `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`, key in `x-goog-api-key`.

## Vercel Hobby (hosting)

- Python runtime (https://vercel.com/docs/functions/runtimes/python, updated 2026-08-12): ASGI and WSGI apps; versions
  3.12 (default), 3.13, 3.14, set by `.python-version`, `pyproject.toml` or `Pipfile.lock`; entrypoints `app.py`,
  `index.py`, `server.py`, `main.py` (also under `src/` or `app/`) defining a top-level `app`; dependencies from
  `requirements.txt` or `pyproject.toml`; "Vercel Functions support streaming responses when using the Python runtime";
  bundle limit 500 MB uncompressed; "By default, Python Vercel Functions include all files from your project that are
  reachable at build time".
- `vercel.json` `functions[...]` takes `maxDuration`, `includeFiles` and `excludeFiles` globs
  (https://vercel.com/docs/project-configuration/vercel-json).
- `.vercelignore` (https://vercel.com/docs/deployments/vercelignore): all files are uploaded except the default
  exclusions and what `.vercelignore` lists, so a git-ignored build directory is uploaded by `vercel deploy` from a local
  folder. Deploying from a git integration would not include files that are not in git, so this project is deployed from
  the CLI.
- Limits (https://vercel.com/docs/functions/limitations): Hobby 300 s maximum duration, 2 GB / 1 vCPU, 4.5 MB request
  payload (the response limit does not apply to streamed responses).

## Upstash Redis (shared rate-limit state)

- Free plan (https://upstash.com/pricing/redis): $0, 250 MB data, 10 GB monthly bandwidth, 500K monthly commands;
  "Once you enter your credit card, your database will be upgraded to the pay-as-you-go plan", so a database without a card
  stays on the free plan.
- One question costs at most about 12 commands (four counters, each an INCR and an EXPIRE), so the free monthly allowance
  covers roughly 40,000 questions, far above the daily caps below.

## Caps chosen

| Cap | Value | Why |
|---|---|---|
| per address | 10 / minute, 60 / day | the contract |
| Groq budget | 100 requests / day | a request is about 1,000 to 1,300 tokens, so the 200,000 tokens/day limit allows about 150 to 200 |
| Gemini budget | 100 requests / day | well inside any free-tier daily quota listed for a Flash-Lite model; the exact number is in AI Studio |
| global | 120 requests / day | below the sum of the two budgets; only requests that reach a model count |

When a provider answers 429, errors or times out, the chain moves to the next one; when none is left, or a cap is spent,
the API answers 503 with `fallback: "limit"`.
