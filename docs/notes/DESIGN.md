# Design notes

Why the project looks the way it does. Numbers live in `results/` and the README, not here.

## Scope

The original service was a LangChain, FAISS and FastAPI wrapper around one hosted model. It had no tests,
no data and no measurement of whether the "I don't know" fallback worked. This version keeps the same stack and
endpoints and adds what was missing: local models by default, a bundled corpus with a gold question set,
measured retrieval and abstention, citations, streaming and a viewer.

## Decisions

- **Local first.** Defaults are MiniLM embeddings and Qwen2.5-0.5B-Instruct, so the project runs with no key. Hosted
  providers are adapters over plain HTTP (`httpx`) and are tested against a stubbed transport only; none has been
  called for real. Hosted model ids have no default and come from the environment.
- **Thin LangChain use.** LangChain supplies `Document`, `Embeddings`, the recursive splitter, the document loaders and
  the FAISS vector store. The generators and the embedding adapters are small classes of mine, so the install does not
  pull a provider SDK per vendor.
- **Exact index, no pickle.** `IndexFlatIP` over normalised vectors (cosine). The store is saved as a FAISS index,
  a JSONL file of chunks and a small JSON file, and loading it never unpickles anything. A store built with a different
  embedder is refused rather than silently queried with incomparable vectors.
- **Hybrid retrieval.** Dense and BM25 rankings are fused with reciprocal rank fusion (k = 60). Each hit carries its
  dense cosine, BM25 score and fused score so the UI and the gate can use whichever they need.
- **Citations are what the model saw.** A source in a response is a chunk that was put in the prompt, with its offsets
  in the document. `support` is the share of the answer's content words found in that chunk: a lexical hint about which
  chunk the answer drew on, not proof of it.
- **Abstention has two paths.** The model can decline ("I don't know"), and an optional gate can decline before the
  model is called when the best cosine similarity is below a threshold. The gate is off by default; the README
  reports what it does on held-out questions.
- **Gold passages are spans.** Each answerable question names the exact text that answers it. A retrieved chunk counts
  only if it covers enough of that span (half of it, or 60 characters, whichever is smaller), so results are
  comparable across chunk sizes.
- **Streaming.** `/query/stream` sends server-sent events: `retrieval` (chunks and scores), `token`, `done` (the same
  payload as `/query`).

## What was exercised for real, and what was not

Real: MiniLM embeddings, the two Qwen models, the cross-encoder reranker, FAISS, BM25, the FastAPI service with the
offline providers, the CLI, the interface in headless Chromium.
Stubbed in tests only: the OpenAI-compatible and Gemini adapters (requests, parsing, streaming, retries, errors).
Not built here: the Docker image (no Docker on the development machine).
