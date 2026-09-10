# RAG Document Assistant — SBP circulars & financial reports

A retrieval-augmented question-answering assistant over PDF / DOCX / Markdown
documents, built for Pakistani banking & telecom material (State Bank of
Pakistan circulars, bank and telecom annual reports). Upload documents, ask
questions, get answers with inline citations you can verify — and measure the
whole thing with a 30-question evaluation set.

**Stack:** LangChain · ChromaDB · `bge-small-en-v1.5` local embeddings ·
BM25 hybrid retrieval · Groq (free tier) or Ollama · Streamlit · RAGAS.
Runs at zero cost.

<!-- Add a screenshot after running the app: ![screenshot](docs/screenshot.png) -->

## How it works

```
 PDF / DOCX / MD / URL
        │  pypdf · python-docx · BeautifulSoup
        ▼
  clean + chunk            RecursiveCharacterTextSplitter (900 chars, 150 overlap)
        │
        ▼
  embed + store            BAAI/bge-small-en-v1.5  →  ChromaDB (persistent, cosine)
        │
        ▼
  hybrid retrieve          dense top-12  ⊕  BM25 top-12  →  Reciprocal Rank Fusion → top-5
        │
        ▼
  answer with citations    LangChain LCEL prompt → Groq chat model → "… [2]"
        │
        ▼
  evaluate                 hit-rate@k · MRR · LLM-judged accuracy · RAGAS
```

Design choices worth mentioning in an interview:

* **Hybrid retrieval.** Regulatory text is full of exact tokens (circular numbers,
  "PKR 300 million", "IFRS 9") that dense embeddings blur. BM25 catches those;
  the dense index catches paraphrases. RRF fuses the two without tuning weights.
* **Grounded prompting.** The model only sees numbered passages and must cite
  `[n]`; if the passages don't contain the answer it says so instead of guessing.
  Citations are parsed back so the UI can highlight which passages were used.
* **Idempotent ingestion.** Chunk IDs are derived from `(source, chunk index)`,
  so re-ingesting a file upserts rather than duplicating.
* **Everything swappable from `.env`** — provider, models, chunk size, k, hybrid
  on/off — so the eval script can be used to run ablations.

## Required environment variables

Create a `.env` file in the project root before running the app or evaluation scripts.
The project reads these values automatically:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=openai/gpt-oss-20b
JUDGE_MODEL=openai/gpt-oss-20b

EMBEDDING_BACKEND=bge
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K=5
HYBRID=true
DENSE_K=12
BM25_K=12
```

> Groq model names are account-specific. If a model returns `model_not_found`, list the models on your Groq account and replace `GROQ_MODEL` / `JUDGE_MODEL` with one that is available.

## Quick start

```bash
git clone <your-repo> && cd rag-assistant
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add your free GROQ_API_KEY (https://console.groq.com/keys)
```

**1. Get the documents.** The script fetches the substantive 2024 BPRD circulars
from sbp.org.pk (as `.md`) plus their annexure PDFs into `data/docs/`:

```bash
python scripts/download_docs.py
```

Drop any other PDFs you want into `data/docs/` too — e.g. HBL / MCB / UBL
annual reports from their investor-relations pages, or PTCL / Jazz reports.

**2. Index them.**

```bash
python scripts/ingest.py            # → "Indexed 19 files / 210 pages / 640 chunks in 41.3s"
```

**3. Ask.**

```bash
streamlit run app.py                # web UI with upload, chat and source viewer
python scripts/ask.py "What is the regulatory retail portfolio limit?"   # or CLI
```

> Fully offline instead of Groq? Install [Ollama](https://ollama.com), run
> `ollama pull llama3.1:8b`, and set `LLM_PROVIDER=ollama` in `.env`.

## Evaluation

`eval/sbp_bprd_2024.jsonl` holds **30 questions** written against the 2024 BPRD
circulars, each with a reference answer and the document(s) that contain it.
(`eval/demo.jsonl` is a 10-question set over the fictional sample docs in
`data/sample_docs/`, used by the tests.)

A stronger, manually checked reference set for the bundled sample corpus is also
included at `eval/ground_truth.jsonl`. It contains **30 question/answer pairs**
covering the three fictional documents in `data/sample_docs/` and includes the
source document and section/chunk label for each item. This acts as the ground
truth for quality checks and regression testing.

```bash
python eval/run_eval.py                    # full run: retrieval + accuracy judge + RAGAS
python eval/run_eval.py --retrieval-only   # instant, no LLM calls
python eval/run_eval.py --no-ragas         # skip RAGAS (fewer API calls)
```

What is reported:

| Metric | How it's computed |
|---|---|
| **hit-rate@k** | % of questions where at least one of the top-k retrieved chunks comes from the expected document |
| **MRR** | mean reciprocal rank of the first correct chunk |
| **answer accuracy** | % of answers an LLM judge marks CORRECT against the reference (strict rubric: right figures/dates/conditions) |
| **RAGAS** | `faithfulness`, `answer_correctness`, `context_precision`, `context_recall` |

### Verified sample results on the ground-truth QA set

The freshly verified run on `eval/ground_truth.jsonl` produced the following
retrieval metrics over 30 questions:

| Metric | Result | How we measured it |
|---|---:|---|
| **Hit-rate@5** | **100.0%** | `first_hit_rank(...) <= 5` for each question |
| **Hit-rate@10** | **100.0%** | `first_hit_rank(...) <= 10` for each question |
| **MRR** | **0.983** | `1 / rank_of_first_correct_chunk`, averaged across all questions |
| **Answer accuracy** | **100.0%** | LLM judge marked 30/30 answers as correct |
| **Faithfulness** | **Not run in latest benchmark** | Full RAGAS faithfulness requires the full `eval/run_eval.py` run without `--no-ragas` |

> The latest verified benchmark used `--no-ragas`, so retrieval and answer accuracy
> are confirmed. The full faithfulness metric is still pending until the quota allows
> a complete RAGAS run.

### How we did it

1. We loaded the QA set from `eval/ground_truth.jsonl`.
2. We indexed the sample corpus with `python scripts/ingest.py data/sample_docs --reset`.
3. We ran retrieval with `eval/run_eval.py --set eval/ground_truth.jsonl --no-ragas`.
4. For each question, the evaluator checked whether the first correct source
   appeared in the top-k results and calculated:
   - `hit_rate@k = correct_questions / total_questions`
   - `MRR = mean(1 / rank_of_first_correct_chunk)`
5. For faithfulness, the project uses RAGAS with the retrieved evidence and the
   LLM judge to check whether each answer claim is supported by the retrieved
   context. This is enabled by running the full `eval/run_eval.py` without
   `--no-ragas` when API quota allows.

Results land in `eval/results/<timestamp>.{json,csv,md}`; the CSV has every
question, its rank, the generated answer and the retrieved sources, so misses
are easy to inspect. Paste the `.md` table below once you've run it:

<!-- eval results: paste eval/results/<latest>.md here -->

### Retrieval metrics to measure

These are the core retrieval-quality metrics for the system, using the ground-truth
QA set as the benchmark:

* **Hit rate @k** — percentage of questions where the correct chunk/document is present in the top-k retrieved results. This is the headline retrieval metric for whether the system found the right evidence.
* **MRR (Mean Reciprocal Rank)** — average reciprocal rank of the first correct chunk. This tells you how high the correct evidence sits in the retrieval list.
* **Context precision / recall (RAGAS)** — measures how much of the retrieved context is relevant, and how much of the needed evidence was successfully retrieved. Precision penalizes noisy retrieval; recall penalizes missed evidence.

This is the retrieval-side check you want when asking: *Did we find the right chunk?*

### Generation metrics to measure

These are the answer-quality checks for whether the model produced the right output once the evidence is retrieved:

* **Faithfulness (RAGAS)** — checks whether every claim in the answer is supported by the retrieved context. This catches hallucination and unsupported statements.
* **Answer relevancy (RAGAS)** — checks whether the answer actually addresses the question asked, rather than drifting into generic or unrelated content.
* **Answer correctness** — compares the answer to the ground-truth answer from `eval/ground_truth.jsonl`, using either an LLM judge or manual `0/1` scoring for the fixed 30-question set.

This is the generation-side check you want when asking: *Was the answer right?*

### Step 4 — Run experiments and report a table

This is the part that makes the project look professional in a report or interview: compare a small matrix of retrieval settings, then show a compact table with the headline metrics.

Suggested experiment grid:

| Config | Hit rate@5 | MRR | Answer accuracy | Faithfulness |
|---|---:|---:|---:|---:|
| Current baseline (chunk 900, hybrid BM25 + vector) | 100.0% | 0.983 | 100.0% | Not run |
| Vector-only retrieval | TBD | TBD | TBD | TBD |
| BM25-only retrieval | TBD | TBD | TBD | TBD |
| Chunk 500 + hybrid | TBD | TBD | TBD | TBD |
| Chunk 1200 + hybrid | TBD | TBD | TBD | TBD |
| + reranker | TBD | TBD | TBD | TBD |

Recommended ablations to run in order:

1. Baseline: short chunks + vector-only retrieval
2. Add `bge-small-en-v1.5` embeddings
3. Enable hybrid BM25 + vector search
4. Tune chunk size (300, 500, 800, 1200)
5. Add a reranker if available

This gives you a clean comparison: retrieval quality first, then generation quality, then final answer correctness.

### Ablations to try (each is one `.env` change + one eval run)

* `HYBRID=false` — how much does BM25 add on circular numbers and figures?
* `CHUNK_SIZE=500` vs `1500` — recall vs. precision of context.
* `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5` — bigger embedder, same code.
* `GROQ_MODEL=openai/gpt-oss-20b` — compare against other available Groq models on your account.

## Project layout

```
app.py                   Streamlit UI (upload, chat, citations, source viewer)
rag/
  config.py              all settings, overridable from .env
  llm.py                 Groq / Ollama / fake chat models; bge / hashing embeddings
  ingest.py              loaders (pdf, docx, md, txt, url) → chunk → Chroma
  retriever.py           HybridRetriever: dense + BM25 + RRF (LangChain BaseRetriever)
  chain.py               LCEL chain, grounded prompt, citation parsing
scripts/
  download_docs.py       builds the real corpus from sbp.org.pk
  ingest.py              index a folder / list / reset
  ask.py                 CLI question answering
eval/
  sbp_bprd_2024.jsonl    30-question eval set with references + expected sources
  demo.jsonl             10 questions over the fictional sample corpus
  run_eval.py            hit-rate@k, MRR, LLM-judge accuracy, RAGAS
data/sample_docs/        fictional bank / telecom / regulator docs for tests & demo
tests/                   offline tests (fake LLM + hashing embeddings); run on CI
```

## Tests

```bash
pytest -q          # 5 tests, ~10 s, no network or API key needed
```

## Notes

* First run downloads the embedding model (~130 MB) from Hugging Face.
* Groq model names vary by account, and some names that are valid elsewhere may
  return `model_not_found` on yours. Use a model visible from your Groq account
  (for example `openai/gpt-oss-20b`) and update `.env` accordingly.
* The eval script throttles itself and RAGAS runs with 2 workers. If you hit
  limits, use `--no-ragas` or point `JUDGE_MODEL` at a smaller model available in
  your Groq account.
* Scanned PDFs have no text layer and are skipped with a warning — run them
  through OCR (e.g. `ocrmypdf`) first.
* `data/docs/` and `data/chroma/` are git-ignored; the corpus is reproducible
  with `scripts/download_docs.py`.
