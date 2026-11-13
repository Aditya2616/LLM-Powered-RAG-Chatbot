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
  counter (1) and the token counters (2 to reserve, 2 to correct with the real usage), each an INCRBY with an EXPIRE in one
  pipeline request. That is at most about 20 commands a question, so the free monthly allowance covers about 25,000
  questions, far above the daily caps below.

## Token budget and caps

A question's prompt is the fixed instructions plus five retrieved chunks of at most 500 characters. The measured
tokens per question (prompt and completion, from the providers' `usage`) are in the README's token table; the prompt stays
well under 1,500 tokens. The budgets count what the provider reported, not a guess: each call adds its real `total_tokens`
to 10-second buckets, and the per-minute check sums the last six buckets (a sliding 60 s window). The call in flight is
reserved at the running average of the last 20 real calls of that model (1,000 until there is one), and the reservation is
swapped for the real figure when the call returns or released when it fails. A worst-case estimate (prompt plus the whole
completion allowance) is not used.

| Cap | Default | Why |
|---|---|---|
| per address | 10 / minute, 60 / day | the contract |
| Groq tokens per model | 7,500 / minute (sliding window), 170,000 / day | below 8,000 and 200,000 |
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
- `ASK_MAX_NEW_TOKENS=400` for all three; it caps the completion only and no longer inflates the budgets.

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

## The live 503 streak, and what changed

During the deployed run, nine consecutive questions answered 503 `limit` while Groq's own headers showed quota left.
The service answers 503 only when every provider in the chain was skipped or failed, so the cause had to be one of:

1. **The per-minute token budget (the cause).** Each call was reserved at an estimate (prompt characters / 3.5 plus the
   whole 400-token completion allowance), about 1,400 against a real usage of about 950, and the minute was a fixed
   calendar minute. Six or seven questions a minute filled the 6,500 cap on the primary on paper while Groq had used
   about 6,000, so the primary was skipped, the backup filled the same way, and the rest went to Gemini or to a 503. The
   accounting now follows real usage in a sliding 60 s window plus a realistic reservation (see above): 6 a minute stay
   100% on the primary, 8 a minute stay on it with at most the occasional backup, 12 a minute split between the two Groq
   models, and Gemini serves only when both are really out (a 429, or the day budget). A 429 puts that model on cooldown
   for its `Retry-After` and the chain falls through.
2. **Gemini not answering.** With both Groq models skipped, the only remaining leg was Gemini. If `GEMINI_MODEL` or the key
   was not set in the Vercel project, or the call failed, the result is a 503. Locally the Gemini leg answers (evaluated
   alone), so this needs the new logs to confirm on Vercel.
3. **Blocking.** The route called the provider synchronously inside an async handler, so one slow call held back every
   other request on the instance. It now runs in a thread pool.

Per-instance memory counters are not the cause of a streak (they would under-count, not over-count), but without Upstash
each instance counts separately.

What the service does now:
- A 503 carries a real `Retry-After`: the seconds until the earliest budget or cooldown frees (at least 1; for a day
  budget, the seconds to UTC midnight).
- Every 503, 429, 400 and every answer that needed a fallthrough or was refused by the verifier writes one JSON line to
  stderr (Vercel runtime logs): `{"ask":{"status":503,"stage":"unavailable","attempts":["groq: budget_tokens_minute",
  "groq-backup: budget_tokens_minute","gemini: 429"],"retry_after":23,...}}`. Reason codes only: no question, no answer
  text, no key.
- Every answer carries an `X-Ask-Provider` response header naming the provider that produced it (empty for fixed replies),
  so a client can see in production whether the Gemini leg is serving.

## Unseen recruiter questions: what failed and why

A live trial with 30 questions nobody had tuned on found seven failures, all of them generalisation, not capacity:

- A README table row lost its header when the chunker split the table, so "what is VoiceScribe's word error rate" or
  "which EvalBench checkpoint scored best" had nothing to find. Tables are now indexed one row per bullet with the
  column names (`tables_to_rows` in `build.py`).
- A named project's opening summary could fall out of the top chunks behind boilerplate; it is now always included.
  Upper-case terms (RLHF, WER) lift the chunks that spell them, and "use/uses/used" no longer count as search words.
- "Tell me about Aditya in two sentences" matched the Contact section; introductions are now answered from the About
  passage without a model, and a request to add a claim ("and say he has 10 years of experience") is refused with that
  same true summary.
- The injection screen fired on the words "jailbreak" and "run Python"; its rules now look for instructions aimed at the
  assistant ("enable developer mode", "run this code:"), so questions about Sentinel and jailbreaks reach the model.
- The EvalBench judge caveat was attached to any EvalBench question containing "score"; a caveat is now attached only
  when the question is about that claim's topic or the answer asserts the claim.

`eval/ask_aditya_recruiter.jsonl` (40 questions, written from the sources after these fixes) was run once. Its two
misses are left as they are: "What are his main strengths?" (the model declined) and "What detector does VisionLens
use?" (the chunk that names D-FINE nano is not among the seven retrieved; BM25 ranks the shorter chunks above it).
During that run the backup Groq model answered 429 for most calls (its quota is shared with the deployed service's
tests), so Gemini produced most of the generated answers; the per-provider counts are in the README.
