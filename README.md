# LLM-Powered RAG Chatbot

[![ci](https://github.com/Aditya2616/LLM-Powered-RAG-Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Aditya2616/LLM-Powered-RAG-Chatbot/actions/workflows/ci.yml) [![licence: MIT](https://img.shields.io/badge/licence-MIT-d7ff3f.svg)](LICENSE)

<!-- BEGIN intro -->
Question answering over your own documents, with the answer tied to the text it came from. It ingests PDF, TXT and DOCX files, splits them into chunks, embeds them into a FAISS index, retrieves the best chunks for a question (dense, BM25 or a fusion of both), and instructs a language model to answer from those chunks only and to say "I don't know" when they do not contain the answer (measured below: it does so on 56.0% to 72.0% of unanswerable questions, depending on the model). Python, LangChain, FAISS, FastAPI, Streamlit and Docker. It runs with no API key: the default embedder and generator are small open models that run on a laptop.
<!-- END intro -->

<!-- BEGIN lead -->
Headline result, on the 100 questions bundled with the repository (a small invented corpus): Qwen2.5-0.5B-Instruct answers 52/75 (69.3%) of the answerable questions correctly and says "I don't know" on 14/25 (56.0%) of the unanswerable ones; Qwen2.5-1.5B-Instruct answers 70/75 (93.3%) and declines 18/25 (72.0%). Declining reliably is the weak spot, and the results below say where.
<!-- END lead -->

The part I care about is the "I don't know" path, so the repository ships a small corpus and a question set (answerable and unanswerable) and reports how often the system answers correctly, declines when it should, and declines when it should not.

![The viewer at the Generate step: chunks of the documents as points in embedding space, with the question's query point joined to the four numbered chunks retrieved for it](docs/assets/desktop-answered-map.png)

| An answerable question | A question the documents cannot answer |
|---|---|
| <img width="260" alt="Phone view of the answer card: the answer, four citation buttons with the share of the answer found in each chunk, and retrieval and generation timings" src="docs/assets/phone-answered.png"> | <img width="260" alt="Phone view of the answer card showing the orange declined state, I don't know, and the retrieved chunks with the question words underlined" src="docs/assets/phone-declined.png"> |

![Desktop view of a declined question: the answer card says I don't know and three retrieved chunks sit beside it with scores](docs/assets/desktop-declined.png)

![Desktop view of an answered question: the answer card with citations beside three retrieved chunks, matching words highlighted](docs/assets/desktop-answered.png)

## How it works

The interface in `docs/` walks the pipeline step by step (ingest, chunk, embed, retrieve, generate) with real data: the bars are the documents, the points are their chunks projected from the embedding space, and the lime points are what the retriever returned for the question, with their scores. The static page replays queries recorded from a real run (it says so in the page); run the service locally and the same page talks to your own documents.

```mermaid
flowchart LR
    D[PDF / TXT / DOCX] --> L[Loaders + RecursiveCharacterTextSplitter]
    L --> C[Chunks with source, id, char offsets]
    C --> E[Embedder: MiniLM, OpenAI or Gemini]
    E --> F[(FAISS exact index + chunks.jsonl)]
    Q[Question] --> R{Retriever}
    F --> R
    C --> B[BM25]
    B --> R
    R -->|top-k chunks with scores| G{Gate: best cosine below threshold?}
    G -->|no| M[Generator: local Qwen, OpenAI-compatible or Gemini]
    G -->|yes| X["I don't know"]
    M --> A[Answer + citations + timings]
    M -->|declines| X
    A --> U[FastAPI: /query, /query/stream, /upload, /health]
    X --> U
    U --> W[Web viewer and Streamlit client]
```

Everything is in `src/rag_chatbot/`: `loaders.py` (loading, chunking, upload checks), `embeddings.py` and `llm.py` (provider adapters), `store.py` (FAISS and persistence), `retrieval.py` (dense, BM25, fusion, optional reranker), `pipeline.py` (prompt, gate, cache, history, query log, citations), `api.py` (service), `evaluation.py` (metrics), `cli.py`.

## Results

### Caveats

- **The corpus is small and invented.** 11 documents about a fictional outdoor-gear retailer, written by me so that they are licence-clean and so that no model has seen them. The questions are also mine. Retrieval scores are high partly because the corpus is easy: the documents are short and cleanly sectioned. This measures whether the pipeline works and which options help, not how it will do on your messy PDFs.
- **Small samples.** 75 answerable and 25 unanswerable questions. Tables give counts as k/n; at 75 questions a 95% Wilson interval is roughly plus or minus 10 percentage points (the intervals are in the `results/*.json` files), and a difference of one or two questions is noise.
- **Answer correctness is a word match.** An answer counts as correct if it contains every required fact (a number, a name) after normalising currency and punctuation. It is strict about wording and blind to meaning, so it can fail a right answer phrased differently and pass a wrong answer that happens to include the right words. "Grounded" is likewise lexical: the share of the answer's words that appear in the retrieved text. Neither is a human judgement.
- **The generator is tiny.** The default model has 0.5 billion parameters. It follows "answer only from the context, otherwise say I don't know" only partly, and the tables show where it does not.
- **Hosted providers were not run.** The OpenAI-compatible and Gemini adapters are tested against a stubbed HTTP transport. They have not been run against a real service, because I have no keys. The Docker image is built and queried in CI on every push with the no-model providers, and was run once in CI with the default local models.

<!-- BEGIN setup -->
The corpus is 11 invented policy documents (5,403 words, 30,830 characters) for a fictional outdoor-gear retailer. The question set has 100 questions: 75 answerable (42 worded close to the text, 33 reworded) and 25 that the documents cannot answer (17 on a topic the documents cover but a detail they do not, 8 off topic). Retrieval ran on cuda with all-MiniLM-L6-v2 embeddings (revision `1110a243`); generation ran on cuda in float16 with greedy decoding and at most 160 new tokens, retrieving the top 4 chunks with the hybrid retriever at 500-character chunks unless a table says otherwise. Dates: retrieval 2026-11-13, generation 2026-11-13.
<!-- END setup -->

### Headline

<!-- BEGIN headline -->
- Retrieval (500-character chunks): hybrid MRR 0.920, dense 0.874, BM25 0.869; hybrid puts a chunk holding the gold passage in the top 3 for 98.7% of the 75 answerable questions.
- With Qwen2.5-0.5B-Instruct, RAG answers 52/75 (69.3%) of the answerable questions correctly (without retrieval: 2/75 (2.7%)).
- It says "I don't know" on 14/25 (56.0%) of the unanswerable questions, and wrongly on 11/75 (14.7%) of the answerable ones.
- Qwen2.5-1.5B-Instruct: 70/75 (93.3%) correct, 18/25 (72.0%) abstentions on unanswerable, 0/75 (0.0%) false abstentions.
<!-- END headline -->

### What I take from this

Each statement below is checked against the results files by `scripts/check_numbers.py`.

- **Hybrid retrieval is the safest choice, not a large win.** It beats dense search on MRR at every chunk size and is within a few points of the best single retriever everywhere. On a corpus this easy the retrievers are close, and the differences are a handful of questions.
- **The cross-encoder reranker raises MRR at every chunk size, but it barely moved the answers.** It lowers Recall@3 at 500 characters, and at that chunk size it changes the number of correct answers by at most two questions for either model, which is noise. It stays optional (`RERANKER_MODEL`) and off by default.
<!-- BEGIN chunk_bullet -->
- **500-character chunks are the default because of the generator, not the retriever.** Larger chunks retrieve about as well (hybrid MRR 0.937 at 1,000 characters against 0.920 at 500; hybrid Recall@3 98.7% against 98.7%), but the 0.5B model answers more questions correctly from 500-character chunks (52/75) than from 1,000-character ones (45/75).
<!-- END chunk_bullet -->
- **Retrieval is what makes the model useful here.** Without it the 0.5B model answers almost nothing correctly, because the documents are invented and it cannot know them. The no-retrieval row is a sanity check that the questions really need the documents.
- **The "I don't know" path is the weak spot.** Neither model declines on more than three quarters of the unanswerable questions, and the questions about a detail the documents leave out (near-miss) are declined less often than the off-topic ones. The small model is more likely to produce a plausible sentence from a chunk that mentions the topic.
- **Adding two worked examples to the prompt made abstention worse for both models.** The examples lower the rate of "I don't know" on unanswerable questions, so the default prompt has none (`FEW_SHOT` turns them on).
- **The retrieval-score gate trades answers for refusals.** It raises abstention on unanswerable questions for both models, and costs a large share of the correct answers. The threshold the half-A tuning picked (0.55) is aggressive; the sweep shows milder thresholds that catch fewer unanswerable questions at almost no cost. The sweep table is computed over all 100 questions, so it is in-sample; only the half-B table is held out, and at 50 questions it is rough. The gate is off by default.
- **A larger model helps more than any retrieval option.** The 1.5B model answers more of the answerable questions correctly than the 0.5B model at the same retrieval settings, and I would use it first if the hardware allows.
- **A model is worth having.** The 0.5B model beats the extractive no-model baseline on correct answers. The baseline is perfectly grounded by construction, since it only returns retrieved sentences.

### Retrieval: chunk size and retriever

Recall@k: some chunk in the top k covers the gold passage (at least half of it, or 60 characters). MRR: mean reciprocal rank of the first such chunk, over the top 10. "Answer facts in top-3 text": the top-3 chunks together contain every fact the answer needs, which does not depend on where chunk boundaries fall, so it is comparable across chunk sizes. The reranker rows add the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0) over the hybrid candidates.

<!-- BEGIN retrieval -->
| Chunk size (chars) | Chunks | Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | Answer facts in top-3 text | Mean top-4 text (chars) |
|---|---|---|---|---|---|---|---|---|
| 250 | 212 | dense | 81.3% | 93.3% | 97.3% | 0.880 | 93.3% | 656 |
| 250 | 212 | bm25 | 76.0% | 92.0% | 94.7% | 0.842 | 92.0% | 674 |
| 250 | 212 | hybrid | 82.7% | 93.3% | 98.7% | 0.893 | 93.3% | 688 |
| 250 | 212 | hybrid+rerank | 94.7% | 98.7% | 100.0% | 0.969 | 97.3% | 754 |
| 500 | 90 | dense | 77.3% | 97.3% | 98.7% | 0.874 | 98.7% | 1417 |
| 500 | 90 | bm25 | 78.7% | 97.3% | 98.7% | 0.869 | 97.3% | 1405 |
| 500 | 90 | hybrid | 85.3% | 98.7% | 98.7% | 0.920 | 98.7% | 1440 |
| 500 | 90 | hybrid+rerank | 92.0% | 96.0% | 97.3% | 0.947 | 97.3% | 1455 |
| 1000 | 39 | dense | 84.0% | 94.7% | 97.3% | 0.894 | 96.0% | 3208 |
| 1000 | 39 | bm25 | 80.0% | 96.0% | 100.0% | 0.882 | 96.0% | 3146 |
| 1000 | 39 | hybrid | 89.3% | 98.7% | 100.0% | 0.937 | 100.0% | 3201 |
| 1000 | 39 | hybrid+rerank | 96.0% | 100.0% | 100.0% | 0.980 | 100.0% | 3226 |
| 2000 | 22 | dense | 73.3% | 88.0% | 94.7% | 0.820 | 90.7% | 5928 |
| 2000 | 22 | bm25 | 88.0% | 97.3% | 98.7% | 0.923 | 97.3% | 5979 |
| 2000 | 22 | hybrid | 84.0% | 100.0% | 100.0% | 0.911 | 100.0% | 6130 |
| 2000 | 22 | hybrid+rerank | 92.0% | 98.7% | 100.0% | 0.956 | 100.0% | 6112 |
<!-- END retrieval -->

Split by how the question is worded (the default chunk size):

<!-- BEGIN retrieval_kind -->
| Retriever (chunk size 500) | Worded close to the text: Recall@3 / MRR | Reworded: Recall@3 / MRR |
|---|---|---|
| dense | 97.6% / 0.883 (n=42) | 97.0% / 0.862 (n=33) |
| bm25 | 100.0% / 0.909 (n=42) | 93.9% / 0.819 (n=33) |
| hybrid | 100.0% / 0.952 (n=42) | 97.0% / 0.879 (n=33) |
| hybrid+rerank | 97.6% / 0.968 (n=42) | 93.9% / 0.920 (n=33) |
<!-- END retrieval_kind -->

### Answers

<!-- BEGIN gen_qwen2.5-0.5b-instruct_c500 -->
**Qwen2.5-0.5B-Instruct**, 500-character chunks, hybrid retrieval, top 4, all 100 questions:

| Setup | Correct on answerable | Wrong answer | False "I don't know" | "I don't know" on unanswerable | near-miss | off-topic | Right decision overall | Grounded (>=80% of words in context) |
|---|---|---|---|---|---|---|---|---|
| No retrieval (question only) | 2/75 (2.7%) | 73/75 (97.3%) | 0/75 (0.0%) | 0/25 (0.0%) | 0/17 (0.0%) | 0/8 (0.0%) | 2/100 (2.0%) | n/a |
| RAG, plain prompt | 52/75 (69.3%) | 12/75 (16.0%) | 11/75 (14.7%) | 14/25 (56.0%) | 8/17 (47.1%) | 6/8 (75.0%) | 66/100 (66.0%) | 55/75 (73.3%) |
| RAG + cross-encoder reranker | 53/75 (70.7%) | 11/75 (14.7%) | 11/75 (14.7%) | 12/25 (48.0%) | 6/17 (35.3%) | 6/8 (75.0%) | 65/100 (65.0%) | 59/77 (76.6%) |
| RAG, two worked examples in the prompt | 53/75 (70.7%) | 21/75 (28.0%) | 1/75 (1.3%) | 5/25 (20.0%) | 1/17 (5.9%) | 4/8 (50.0%) | 58/100 (58.0%) | 67/94 (71.3%) |
| RAG + retrieval-score gate | 33/75 (44.0%) | 8/75 (10.7%) | 34/75 (45.3%) | 22/25 (88.0%) | 14/17 (82.4%) | 8/8 (100.0%) | 55/100 (55.0%) | 37/44 (84.1%) |
| RAG, worked examples + gate | 33/75 (44.0%) | 16/75 (21.3%) | 26/75 (34.7%) | 21/25 (84.0%) | 13/17 (76.5%) | 8/8 (100.0%) | 54/100 (54.0%) | 43/53 (81.1%) |
<!-- END gen_qwen2.5-0.5b-instruct_c500 -->

<!-- BEGIN gen_qwen2.5-1.5b-instruct_c500 -->
**Qwen2.5-1.5B-Instruct**, 500-character chunks, hybrid retrieval, top 4, all 100 questions:

| Setup | Correct on answerable | Wrong answer | False "I don't know" | "I don't know" on unanswerable | near-miss | off-topic | Right decision overall | Grounded (>=80% of words in context) |
|---|---|---|---|---|---|---|---|---|
| No retrieval (question only) | 6/75 (8.0%) | 69/75 (92.0%) | 0/75 (0.0%) | 0/25 (0.0%) | 0/17 (0.0%) | 0/8 (0.0%) | 6/100 (6.0%) | n/a |
| RAG, plain prompt | 70/75 (93.3%) | 5/75 (6.7%) | 0/75 (0.0%) | 18/25 (72.0%) | 10/17 (58.8%) | 8/8 (100.0%) | 88/100 (88.0%) | 63/82 (76.8%) |
| RAG + cross-encoder reranker | 69/75 (92.0%) | 6/75 (8.0%) | 0/75 (0.0%) | 17/25 (68.0%) | 9/17 (52.9%) | 8/8 (100.0%) | 86/100 (86.0%) | 64/83 (77.1%) |
| RAG, two worked examples in the prompt | 69/75 (92.0%) | 5/75 (6.7%) | 1/75 (1.3%) | 12/25 (48.0%) | 5/17 (29.4%) | 7/8 (87.5%) | 81/100 (81.0%) | 63/87 (72.4%) |
| RAG + retrieval-score gate | 47/75 (62.7%) | 3/75 (4.0%) | 25/75 (33.3%) | 22/25 (88.0%) | 14/17 (82.4%) | 8/8 (100.0%) | 69/100 (69.0%) | 42/53 (79.2%) |
| RAG, worked examples + gate | 46/75 (61.3%) | 3/75 (4.0%) | 26/75 (34.7%) | 22/25 (88.0%) | 14/17 (82.4%) | 8/8 (100.0%) | 68/100 (68.0%) | 42/52 (80.8%) |
<!-- END gen_qwen2.5-1.5b-instruct_c500 -->

With 1,000-character chunks:

<!-- BEGIN gen_qwen2.5-0.5b-instruct_c1000 -->
**Qwen2.5-0.5B-Instruct**, 1000-character chunks, hybrid retrieval, top 4, all 100 questions:

| Setup | Correct on answerable | Wrong answer | False "I don't know" | "I don't know" on unanswerable | near-miss | off-topic | Right decision overall | Grounded (>=80% of words in context) |
|---|---|---|---|---|---|---|---|---|
| No retrieval (question only) | 2/75 (2.7%) | 73/75 (97.3%) | 0/75 (0.0%) | 0/25 (0.0%) | 0/17 (0.0%) | 0/8 (0.0%) | 2/100 (2.0%) | n/a |
| RAG, plain prompt | 45/75 (60.0%) | 19/75 (25.3%) | 11/75 (14.7%) | 7/25 (28.0%) | 3/17 (17.6%) | 4/8 (50.0%) | 52/100 (52.0%) | 61/82 (74.4%) |
| RAG + cross-encoder reranker | 50/75 (66.7%) | 13/75 (17.3%) | 12/75 (16.0%) | 9/25 (36.0%) | 6/17 (35.3%) | 3/8 (37.5%) | 59/100 (59.0%) | 59/79 (74.7%) |
| RAG, two worked examples in the prompt | 45/75 (60.0%) | 26/75 (34.7%) | 4/75 (5.3%) | 1/25 (4.0%) | 0/17 (0.0%) | 1/8 (12.5%) | 46/100 (46.0%) | 68/95 (71.6%) |
| RAG + retrieval-score gate | 37/75 (49.3%) | 17/75 (22.7%) | 21/75 (28.0%) | 17/25 (68.0%) | 10/17 (58.8%) | 7/8 (87.5%) | 54/100 (54.0%) | 50/62 (80.6%) |
| RAG, worked examples + gate | 34/75 (45.3%) | 26/75 (34.7%) | 15/75 (20.0%) | 14/25 (56.0%) | 7/17 (41.2%) | 7/8 (87.5%) | 48/100 (48.0%) | 57/71 (80.3%) |
<!-- END gen_qwen2.5-0.5b-instruct_c1000 -->

<!-- BEGIN gen_extractive -->
No generator at all: return the retrieved sentence with the most question words (plural-stemmed), or "I don't know" if fewer than two match.

| Setup | Correct on answerable | Wrong answer | False "I don't know" | "I don't know" on unanswerable | near-miss | off-topic | Right decision overall | Grounded (>=80% of words in context) |
|---|---|---|---|---|---|---|---|---|
| Extractive baseline (no model) | 39/75 (52.0%) | 28/75 (37.3%) | 8/75 (10.7%) | 11/25 (44.0%) | 5/17 (29.4%) | 6/8 (75.0%) | 50/100 (50.0%) | 81/81 (100.0%) |
<!-- END gen_extractive -->

### The retrieval-score gate

<!-- BEGIN gate -->
Threshold tuned on the 50 even-numbered questions (half A); statistic: highest cosine among the top-k chunks. Chosen threshold: **0.55**.

| Threshold | "I don't know" on unanswerable (all 25) | False "I don't know" on answerable (all 75) |
|---|---|---|
| 0.00 | 0.0% | 0.0% |
| 0.10 | 8.0% | 0.0% |
| 0.20 | 8.0% | 0.0% |
| 0.30 | 32.0% | 1.3% |
| 0.40 | 36.0% | 6.7% |
| 0.50 | 60.0% | 16.0% |
| 0.55 | 80.0% | 33.3% |
| 0.60 | 84.0% | 49.3% |
| 0.70 | 100.0% | 74.7% |
| 0.80 | 100.0% | 93.3% |

Judged on the 50 odd-numbered questions (half B) only (Qwen2.5-0.5B-Instruct):

| Setup | Correct on answerable | Wrong answer | False "I don't know" | "I don't know" on unanswerable | near-miss | off-topic | Right decision overall | Grounded (>=80% of words in context) |
|---|---|---|---|---|---|---|---|---|
| RAG, plain prompt | 26/37 (70.3%) | 7/37 (18.9%) | 4/37 (10.8%) | 10/13 (76.9%) | 6/9 (66.7%) | 4/4 (100.0%) | 36/50 (72.0%) | 29/36 (80.6%) |
| RAG, two worked examples in the prompt | 28/37 (75.7%) | 8/37 (21.6%) | 1/37 (2.7%) | 4/13 (30.8%) | 1/9 (11.1%) | 3/4 (75.0%) | 32/50 (64.0%) | 34/45 (75.6%) |
| RAG + retrieval-score gate | 19/37 (51.4%) | 6/37 (16.2%) | 12/37 (32.4%) | 11/13 (84.6%) | 7/9 (77.8%) | 4/4 (100.0%) | 30/50 (60.0%) | 22/27 (81.5%) |
| RAG, worked examples + gate | 21/37 (56.8%) | 5/37 (13.5%) | 11/37 (29.7%) | 10/13 (76.9%) | 6/9 (66.7%) | 4/4 (100.0%) | 31/50 (62.0%) | 23/29 (79.3%) |
<!-- END gate -->

### Latency

CPU timing, end to end for one question (embed the question, search, generate), on the machine described in the table:

<!-- BEGIN latency -->
| Model | Device | Threads | Queries | Retrieve p50 / p95 (ms) | Generate p50 / p95 (ms) | Total p50 / p95 (ms) | CPU load before / after |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (float32) | cpu | 4 of 12 logical | 14 | 10 / 10 | 3148 / 4234 | 3158 / 4244 | 17.0% / 23.0% |
<!-- END latency -->

GPU generation time per question during the runs above (not a CPU claim):

<!-- BEGIN gpu_latency -->
| Model | Device | Mean generation per question (ms) | p95 (ms) | Questions |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (float16, 1000-char chunks) | cuda | 327 | 731 | 100 |
| Qwen2.5-0.5B-Instruct (float16, 500-char chunks) | cuda | 459 | 1067 | 100 |
| Qwen2.5-1.5B-Instruct (float16, 500-char chunks) | cuda | 566 | 1060 | 100 |
<!-- END gpu_latency -->

## Quickstart

Offline, no key. The first run downloads two small models (about 90 MB for MiniLM, about 1 GB for Qwen2.5-0.5B-Instruct) from the Hugging Face hub; after that it is offline.

```bash
git clone https://github.com/Aditya2616/LLM-Powered-RAG-Chatbot.git
cd LLM-Powered-RAG-Chatbot
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[local]"
rag-chatbot ingest corpus/docs
rag-chatbot ask "How long does a PayPal refund take?"
rag-chatbot ask "What is the capital of France?"        # not in the documents: expect "I don't know"
rag-chatbot serve                                         # http://127.0.0.1:8000 : the viewer and the API
```

With no model download at all (a lexical embedder and an extractive generator that returns the best-matching sentence), useful for checking the plumbing:

```bash
pip install -e .
EMBEDDINGS_PROVIDER=hashing LLM_PROVIDER=extractive rag-chatbot ingest corpus/docs
EMBEDDINGS_PROVIDER=hashing LLM_PROVIDER=extractive rag-chatbot ask "When are PayPal refunds issued?"
```

On Windows PowerShell set the variables with `$env:EMBEDDINGS_PROVIDER = "hashing"`. Copy `.env.example` to `.env` to keep settings; every variable is optional.

### Models and providers

| | Default | Alternatives |
|---|---|---|
| Embeddings (`EMBEDDINGS_PROVIDER`) | `huggingface`: `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0), local | `openai` (any OpenAI-compatible `/embeddings`), `gemini`, `hashing` (no model) |
| Generator (`LLM_PROVIDER`) | `local`: `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0), float16 on CUDA, float32 on CPU | `openai` (any OpenAI-compatible `/chat/completions`, set `OPENAI_BASE_URL`), `gemini`, `extractive` (no model) |

Hosted model ids are never defaulted: set `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`, `GEMINI_MODEL` and `GEMINI_EMBEDDING_MODEL` yourself. Pointing `OPENAI_BASE_URL` at another provider's compatible endpoint is how to use one that is not named here; that path is untested beyond the stubs. Use another local model with `LOCAL_LLM_MODEL` or `HF_EMBEDDING_MODEL`. An index records which embedder built it and is refused if you change embedders; run `rag-chatbot reindex`.

### API

`rag-chatbot serve` starts the service (interactive docs at `/docs`).

```bash
# upload documents (.pdf, .txt, .docx; validated by content, 10 MB and 10 files by default)
curl -X POST http://localhost:8000/upload -F "files=@corpus/docs/returns_and_refunds.txt"

# or load the bundled corpus
curl -X POST http://localhost:8000/documents/sample

# ask
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How long does a PayPal refund take?", "top_k": 4}'

# stream: server-sent events "retrieval" (chunks and scores), "token" (text), "done" (the /query payload)
curl -N -X POST http://localhost:8000/query/stream -H "Content-Type: application/json" \
  -d '{"question": "What is the staff discount after probation?"}'

curl http://localhost:8000/health
curl http://localhost:8000/documents

# remove one document and its chunks from the index
curl -X DELETE http://localhost:8000/documents/returns_and_refunds.txt

# 2-D projection of every chunk (what the viewer draws)
curl http://localhost:8000/index/map
```

`rag-chatbot info` prints the embedder, generator, retriever and per-document chunk counts of the current index.

A `/query` response carries the answer, whether it abstained and why (`model_declined`, `low_retrieval_score` or `empty_index`), the sources, and timings. The shape (the values here are illustrative, not a recorded response):

```json
{
  "answer": "PayPal refunds are issued within 2 working days of inspection.",
  "abstained": false,
  "abstain_reason": null,
  "sources": [{"rank": 1, "source": "returns_and_refunds.txt", "chunk": 4, "chunk_id": "returns_and_refunds.txt#4",
               "start": 1180, "end": 1646, "score": 0.0328, "dense": 0.6012, "bm25": 7.41, "support": 0.82,
               "snippet": "...", "text": "..."}],
  "cached": false,
  "retriever": "hybrid",
  "model": {"embeddings": "huggingface:sentence-transformers/all-MiniLM-L6-v2", "llm": "local:Qwen/Qwen2.5-0.5B-Instruct"},
  "timings": {"retrieve_ms": 12.3, "generate_ms": 640.1, "total_ms": 653.0}
}
```

A source is a chunk that was put in the prompt; `support` is the share of the answer's content words found in that chunk. The service also keeps an in-memory LRU cache with a time-to-live, short per-session chat history, and a JSON-lines query log (`data/query_log.jsonl`). The index persists in `data/vector_store/`.

Streamlit client (shows scores and the reason for a refusal): start the service, then `pip install -e ".[streamlit]"` and `streamlit run frontend/streamlit_app.py`.

Docker: `docker build -t rag-chatbot .` then `docker run -p 8000:8000 -v rag-data:/data -v rag-models:/models rag-chatbot`. The image installs CPU PyTorch and downloads the models on first use. CI builds the image and queries the running container (see `.github/workflows/ci.yml`).

## Reproduce

```bash
pip install -e ".[local,dev]"
python scripts/run_eval.py retrieval
python scripts/run_eval.py generation --model Qwen/Qwen2.5-0.5B-Instruct --chunk-size 500
python scripts/run_eval.py generation --model Qwen/Qwen2.5-0.5B-Instruct --chunk-size 1000
python scripts/run_eval.py generation --model Qwen/Qwen2.5-1.5B-Instruct --chunk-size 500
python scripts/run_eval.py extractive
python scripts/run_eval.py latency --model Qwen/Qwen2.5-0.5B-Instruct --threads 4
python scripts/record_demo.py --queries <ids from corpus/questions.jsonl>
python scripts/check_numbers.py --write
pytest                      # fast, offline, no model downloads
pytest -m slow              # loads the real models
ruff check .
```

The fast suite uses a hashing embedder and scripted generators, so it needs neither torch nor a download; that is what CI runs; CI also builds the Docker image and queries the running container. The interface check (`python scripts/check_ui.py --live --shots docs/assets`, needs `pip install -e ".[check]"` and `playwright install chromium`) loads the page at 1440x900 and 390x844 and fails on console errors, horizontal scroll or touch targets under 44 px.

## Layout

```
src/rag_chatbot/   the package (see How it works)
src/askaditya/     the Ask-Aditya service (see Ask-Aditya): safety stages, providers, limits, API
ask/               the deployable app (app.py, vercel.json) and a fictional persona for tests
eval/              ask_aditya_eval.jsonl, the 86-question evaluation set
corpus/docs/       the 11 bundled documents; corpus/questions.jsonl the gold questions
scripts/           run_eval.py, record_demo.py, check_numbers.py, check_ui.py, build_ask_index.py, run_ask_eval.py, ask_retrieval_eval.py
results/           measured results (JSON), the source of every table above
docs/              the static viewer, data/ (recorded run), assets/ (screenshots), notes/
frontend/          the Streamlit client
tests/             fast offline suite; the model test is marked slow
```

## Static demo

`docs/` is a static site: serve it with any file server (`python -m http.server -d docs`) or GitHub Pages. It replays `docs/data/demo.json`, which `scripts/record_demo.py` wrote from a real run (chunk layout, 2-D projection, retrieved chunks and scores, the token stream with its real timing, the final answers), including a question on a topic the documents cover but a detail they do not, and an off-topic one. When the same page is served by the service it switches to live mode and sends your questions to it.

## Ask-Aditya: the portfolio assistant

The same retrieval code also powers the "Ask my AI" box on [Aditya's portfolio](https://adityajain16.vercel.app): a small HTTP API that answers questions about him, grounded only in his résumé, his portfolio text and the READMEs of his ten projects, and that says "I don't know" and points to his email when the sources do not support an answer. Code: `src/askaditya/`, deployable app: `ask/`. It has not been deployed yet: everything below was measured against the real HTTP API running on 127.0.0.1, once with a local 1.5B model and once with the free-tier hosted models (each leg of the chain evaluated alone).

```mermaid
flowchart LR
    V[Visitor] -->|POST /api/ask| C[CORS, per-address limits]
    C --> S[validate, injection screen]
    S --> T{unpublished topic, about the assistant, off topic?}
    T -->|fixed reply| E
    T --> R[BM25 and query heuristics over the index]
    R --> G[provider chain: Groq, Groq backup, Gemini]
    G -->|all failed or cap spent| X[503, fallback limit]
    G --> Y[verify the answer]
    Y -->|ok| E[SSE: token events, then done]
    Y -->|fails| U[I don't know and the email]
    U --> E
```

**What runs, in order.** (1) Input validation. (2) An injection screen: this repository's own rule list over the question and every history item (it does not use Sentinel). (3) Deterministic replies for topics he has not published (salary, grades, notice period, relocation or remote work, visa, phone number, research-paper details, 15FORTEEN details, personal life); they never reach a model. (4) A fixed answer about the assistant itself, a short introduction of him from the About passage, a refusal to add claims about him that the sources do not make, a polite refusal of off-topic questions, and the contact details (how to reach him, his LinkedIn, GitHub and résumé link) taken straight from the indexed Contact passage, never a phone number. (5) Retrieval. (6) Generation with a strict grounded prompt (third person, at most about 80 words, one follow-up suggestion). When a model declines although most of the question's words occur in the top five retrieved chunks, one other model of the chain is asked once before the answer becomes "I don't know" (counted in the budgets, logged as `decline_retry`). (7) Verification before anything is sent: no phone-like digit run, every number in the answer must occur in the retrieved passages, none of the six false claims about his projects may be asserted (a question that tries one gets the true statement from the sources), the capitalised names and most words must come from the passages, and a length cap. An answer that fails becomes "I don't know" plus his email.

Because verification has to come first, the answer is generated completely, checked, and only then streamed as `token` events followed by `done`. Time to first token is therefore the whole generation time, not the first word.

**Retrieval.** The index is a few hundred 500-character chunks (seven are put in the prompt), built by this repository's own splitter. At runtime it is BM25 only (NumPy, no model, no download) plus a few query-side heuristics: section names such as Contact or Education are matched, a named project boosts its README (and its lead section, which holds the summary and the headline numbers), and a short follow-up inherits the previous question. BM25 was chosen over a hosted embedding call because it needs no key, sends no question to a third party and adds no latency or failure mode; the comparison below shows what dense vectors would add. The retrieval parameters were tuned on this same question set, so the retrieval figure is optimistic.

<!-- BEGIN ask_retrieval -->
| Retrieval (top 7 chunks) | Factual questions whose facts are all in the retrieved text |
|---|---|
| bm25 | 39/46 (84.8%) |
| bm25+heuristics | 45/46 (97.8%) |
| hybrid (bm25+heuristics, dense) | 45/46 (97.8%) |
<!-- END ask_retrieval -->

(The dense row uses MiniLM vectors computed offline as a stand-in; a hosted embedding model was not run.)

**API.** `POST /api/ask` with `{"question": string (1 to 500 characters), "history": [{"role": "user"|"assistant", "text": string}]}` (at most 6 items) returns `text/event-stream`: repeated `event: token` / `data: {"t": "..."}`, then `event: done` / `data: {"answer": string, "sources": [{"title", "url"}], "grounded": boolean, "fallback": null|"unknown"|"offtopic"|"refused"|"limit"}`. `GET /api/health` returns `{"ok": true, "corpus_version": "..."}`. Errors: 400 for bad input, 429 with `Retry-After` when a per-address limit is hit (10 a minute, 60 a day), 503 with `fallback: "limit"` when every provider failed or a daily cap is spent. CORS allows only `https://adityajain16.vercel.app` and `http://127.0.0.1:8765`, including the preflight.

**Providers and cost.** The generator is a chain: a primary Groq model, a backup Groq model (Groq's limits are per model, so it is a second budget on the same key), and then Gemini on its free tier as the last resort before the 503 `limit` response. There are no model ids in the code: set `GROQ_MODEL`, `GROQ_MODEL_BACKUP` and `GEMINI_MODEL`; a local model can be used in development. `ASK_DISABLE_GEMINI=1` switches Gemini off without a code change. No payment method is ever needed: a free key answers 429 when its quota is gone, and the service also keeps its own request and token budgets per model (Groq's free limits are 8,000 tokens a minute and 200,000 a day per model, and the prompt is kept compact: see the token table below), so it stops asking first. The token budget follows what each provider reported over a sliding 60-second window, with a reservation for the call in flight at the recent average (about 950 tokens), so a model is used up to its real limit; a 429 puts that model on cooldown for the time the provider names (`Retry-After`, else its `x-ratelimit-reset-tokens`), and the cooldown applied is logged. The limits, the dates they were read and what could not be verified are in `docs/notes/ASK_ADITYA.md`. Cost per 1,000 questions on the free tiers is $0 by construction; nothing here has been billed or measured on a real account.

**Shared limits.** Per-address and daily counters live in Upstash Redis over its REST API when `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` (or the Vercel integration's `KV_REST_API_URL` and `KV_REST_API_TOKEN`) are set: every serverless instance sees the same counts. Without them the counters are in process memory, which is exact on one long-lived server but per instance on a serverless platform (each instance counts separately and resets when recycled), so the limits become approximate and the global cap can be exceeded by up to the number of warm instances. If Upstash is unreachable the service falls back to memory counting rather than stopping.

**Environment variable names.** `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_MODEL_BACKUP`, `GROQ_REASONING_EFFORT` and `GROQ_BACKUP_REASONING_EFFORT` (optional, for reasoning models), `GEMINI_API_KEY`, `GEMINI_MODEL`, `ASK_DISABLE_GEMINI`, `ASK_PROVIDER_ORDER` (default `groq,groq-backup,gemini`), `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`, `ASK_IP_PER_MINUTE`, `ASK_IP_PER_DAY`, `ASK_GLOBAL_DAILY_CAP`, `ASK_PROVIDER_DAILY_CAPS`, `ASK_TOKEN_CAPS_DAY`, `ASK_TOKEN_CAPS_MINUTE`, `ASK_MAX_NEW_TOKENS`, `ASK_INDEX_DIR`, `ASK_PLATFORM_NAME` (build time), `ASK_EVAL_TOKEN` (lets the evaluation runner bypass the per-address limits; unset means no bypass), `ASK_TEST_HOOKS` (`1` enables a header that forces the `limit` path and exposes the stage, provider and token counts of each answer; leave it off in production) and `ASK_LOCAL_MODEL` (development only).

**Rebuilding the index.** The résumé, the exported portfolio text and the project READMEs are private inputs and are not in this repository, and neither is the built index. One command reads them from their local paths, removes phone numbers before chunking, fills in the product name that the corpus keeps as a stand-in token (from `ASK_PLATFORM_NAME` or a git-ignored `ask/private/platform_name.txt`; the default text is "an AI assistant platform"), and writes `index.json` and `VERSION` (the date and the SHA-256 of every source) into the git-ignored `ask/private/index`, then copies the service code into `ask/`:

```bash
python scripts/build_ask_index.py --resume Aditya_Jain_Resume.pdf --corpus corpus.md --bundle \
    --readmes ../SFT-Forge ../VoiceScribe ../VisionLens ../Sentinel ../AgentFlow ../DiffuseLab ../PromptLab ../EvalBench ../ModelRouter ../LLM-Powered-RAG-Chatbot
cd ask && vercel deploy --prod      # a CLI deploy, because the index is not in git (see ask/.vercelignore)
```

`GET /api/health` reports the `corpus_version`. The repository ships a small fictional persona (`ask/persona/`) so the tests and CI run without the private sources.

**Evaluation.** `eval/ask_aditya_eval.jsonl` has 86 questions with machine-checkable expectations, written from the three sources: 46 factual (exact numbers and names must appear), 15 about unpublished topics (the deflection, the email, and never a phone number), 10 traps built from the six false claims, 10 injection attempts and 5 off-topic questions. `scripts/run_ask_eval.py` calls the real HTTP API, checks the server-sent-event framing and every field of `done`, applies each question's expectations, measures time to first token, and probes the 429, 503 `limit` and CORS behaviour on servers with deliberately tiny limits.

**Evaluating a live deployment without tripping its limits.** The free Groq tiers allow about 8,000 tokens a minute per model and an answer costs about 950, so a deployment holds roughly 7 questions a minute on the primary; faster runs spill to the backup and Gemini and then end in 503s, which measures the run, not the service. Pace the run at one question every 8 to 10 seconds: `python scripts/run_ask_eval.py --base-url https://<deployment> --tag live --pause 8` (with `--pause-generated` for an extra wait after questions that reached a model). The runner reads the `X-Ask-Provider` response header, which is on every answer, to tell which questions reached a model (an empty value is a fixed reply; without the header every 200 answer that is not a refusal or off-topic reply counts as generated), records that provider per question in the results, prints the effective questions a minute at the end of the run and warns when it is above 7. Each address is also held to 10 questions a minute and 60 a day, and the whole deployment to 350 model calls a day, so the 86 questions plus the 12 held-out ones need two days of per-address budget or the evaluation token.

<!-- BEGIN ask_eval -->
| Run | Factual | Unpublished topics | Traps | Injection attempts | Off-topic | All | First token mean / p95 (ms) | Cold start: ready / first answer (ms) |
|---|---|---|---|---|---|---|---|---|
| gemini-last: gemini leg alone (gemini-3.5-flash-lite); 86 questions, 2026-11-13, corpus 2026-11-13-9e087a2e | 46/46 (100.0%) | 15/15 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 5/5 (100.0%) | 86/86 (100.0%) | 1325 / 3907 (answered by the model: 2047 / 5389) | 1790 / 2345 |
| groq-backup: groq-backup leg alone (openai/gpt-oss-120b); 86 questions, 2026-11-13, corpus 2026-11-13-9e087a2e | 46/46 (100.0%) | 15/15 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 5/5 (100.0%) | 86/86 (100.0%) | 493 / 1091 (answered by the model: 761 / 1189) | 1779 / 2977 |
| groq-primary: groq leg alone (qwen/qwen3.8-27b); 86 questions, 2026-11-13, corpus 2026-11-13-9e087a2e | 46/46 (100.0%) | 15/15 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 5/5 (100.0%) | 86/86 (100.0%) | 296 / 744 (answered by the model: 457 / 768) | 1781 / 527 |
| live: live Vercel deployment ask-aditya (Hobby), chain Groq primary, Groq backup, Gemini last; client in India, function in iad1, retrieval BM25 only; 86 questions, 2026-11-13, corpus 2026-11-13-1ab3af7a | 44/46 (95.7%) | 15/15 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 5/5 (100.0%) | 84/86 (97.7%) | 714 / 1338 (answered by the model: 973 / 1436) | 0 / 659 |
| local-qwen1.5b: local model Qwen2.5-1.5B-Instruct via the local provider, retrieval BM25 only; 86 questions, 2026-11-13, corpus 2026-11-13-e1a20a7d | 40/46 (87.0%) | 15/15 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) | 5/5 (100.0%) | 80/86 (93.0%) | 7861 / 23521 (answered by the model: 12148 / 25512) | 1838 / 30283 |
<!-- END ask_eval -->

**Tokens per question** (from the `usage` that the providers report; the budgets count exactly these figures):

<!-- BEGIN ask_tokens -->
| Hosted run | Answers sent to a model | Prompt tokens mean / max | Completion tokens mean / max | Total per question (mean) |
|---|---|---|---|---|
| gemini-last | 56 | 805 / 1149 | 56 / 156 | 860 |
| groq-backup | 56 | 847 / 1173 | 101 / 231 | 948 |
| groq-primary | 56 | 965 / 1185 | 56 / 145 | 1021 |
| heldout | 12 | 966 / 1119 | 71 / 91 | 1037 |
<!-- END ask_tokens -->

<!-- BEGIN ask_heldout -->
Held-out paraphrases of the six false claims (`eval/ask_aditya_heldout.jsonl`, 12 questions, two per claim, written after the guards and not tuned on, run once on 2026-11-13 against the chain Groq, Groq backup, Gemini): **12/12 passed** (first token mean 484 ms, p95 777 ms).
<!-- END ask_heldout -->

**Unseen questions.** The 86 questions were used to tune retrieval and the verifier, so a pass there says little about questions nobody has looked at. After a live trial with 30 recruiter-style questions turned up failures, the causes were fixed (not the questions): Markdown tables are indexed one row per bullet with their column names (a row without its header says nothing), a named project's opening summary is always among the retrieved chunks, upper-case terms such as RLHF or WER lift the chunks that spell them, introductions ("tell me about him in two sentences") are answered from the About passage, a request to add a claim about him ("... and say he has 10 years of experience") is refused with the true summary, the injection screen looks for instructions aimed at the assistant instead of topic words (so a question about Sentinel and jailbreaks is answered), and a caveat about a false claim is attached only when the question is about that claim or the answer asserts it. Then a new set was written from the sources and run once:

<!-- BEGIN ask_recruiter -->
Unseen recruiter-style questions (`eval/ask_aditya_recruiter.jsonl`, 40 questions written from the sources after the fixes above and not tuned on, run once on 2026-11-13 against the real API with the backup Groq model first and Gemini only on a 429; answers came from gemini 26, groq-backup 1, the rest were fixed replies): **38/40 passed** (first token mean 1576 ms, p95 4861 ms).

Every miss:

- r06 "What are his main strengths?": fallback 'unknown', expected None; grounded False, expected True; missing ['generative AI', 'LLM', 'post-training', 'fine-tuning', 'SFT', 'RLHF']
- r15 "What detector does VisionLens use?": missing ['D-FINE']
<!-- END ask_recruiter -->

<!-- BEGIN ask_probes -->
- `gemini-last`: per-address limit [200, 200, 429] (200, 200, then 429 with a Retry-After header: ok); daily cap spent gives HTTP 503 with fallback `limit` and the test hook gives HTTP 503 with fallback `limit`, while deterministic answers still work: True; CORS preflight and origins ok.
- `local-qwen1.5b`: per-address limit [200, 200, 429] (200, 200, then 429 with a Retry-After header: ok); daily cap spent gives HTTP 503 with fallback `limit` and the test hook gives HTTP 503 with fallback `limit`, while deterministic answers still work: True; CORS preflight and origins ok.
- Fallback order with the real providers: with both Groq token budgets spent the answer came from `gemini` (HTTP 200, ok); with Gemini switched off as well the API answered HTTP 503 with fallback `limit` (ok).
<!-- END ask_probes -->

**Caveats.** The latency figures come from a Windows PC with a development server, not from the deployed function, and the cold start of a Vercel function has not been measured. The accuracy figures are on one run each of a small question set that was also used to tune retrieval and the verifier, so a point or two is noise and the retrieval figure is optimistic; the 12 held-out paraphrases of the false claims were written after the guards and run once. The primary Groq leg was run in full on the final code; for the backup and Gemini legs the ten trap questions and every question that had failed before were re-run after the last retrieval and guard changes, and their other rows come from the previous full run of the same leg (so a point there is an older measurement). The `live` row is the deployed service, measured by the owner before those changes. Each hosted model answered with the same prompt, retrieval and verifier; the questions it still fails are mostly retrieval misses (for example the number of LLMs and prompts in ModelRouter's data) or answers the verifier refuses because a figure is not literally in the retrieved text. The verifier is lexical: it cannot catch a sentence that reuses words from the sources in a wrong relation (the local model once attributed VoiceScribe's demo voice to the wrong model and the answer passed), so a more capable generator reduces that risk but nothing here removes it. The injection screen is a list of patterns and will miss attacks phrased differently, and the topic detector is keyword based, so an unusual phrasing of an unpublished topic can reach the model, where the grounded prompt and the verifier are the remaining defences. Nothing here is called production-ready.

## Limitations

Short version; the longer notes are in `docs/notes/DESIGN.md`.

- Exact FAISS search over normalised vectors (cosine). Fine for thousands of chunks; for millions you would want an approximate index, which I have not built or measured.
- BM25 is my own small implementation with a plural-stripping stemmer, so it is English only.
- Chunking is by characters, with 15% overlap in the experiments. Tables, scanned PDFs (no OCR) and layout are not handled.
- Chat history is pasted into the prompt as text; there is no question rewriting, so follow-up questions that depend on earlier turns retrieve poorly.
- The cache and the history are in memory and per process; the service is meant to run as a single worker.
- There is no authentication, rate limiting or per-user isolation. It is a local tool, not a hosted product.
- Ask-Aditya's last-resort provider, Gemini on its free tier, is under Google's unpaid-quota terms: the content may be used to improve Google products, and the terms allow only paid services for API clients offered to users in the EEA, Switzerland or the UK. It only sees traffic when both Groq models are out of budget and can be switched off with `ASK_DISABLE_GEMINI=1` (see `docs/notes/ASK_ADITYA.md`).
- The abstention gate depends on one threshold tuned on a small question set; the results above show how far to trust it.

## Licence and credits

MIT, copyright Aditya Jain (see `LICENSE`). Models: `sentence-transformers/all-MiniLM-L6-v2`, `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct` and `cross-encoder/ms-marco-MiniLM-L-6-v2`, all Apache-2.0 per their model cards; they are downloaded, not bundled. The corpus and questions are my own work, covered by the same MIT licence.

Built by [Aditya Jain](https://adityajain16.vercel.app).
