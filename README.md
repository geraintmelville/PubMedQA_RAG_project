# PubMedQA retrieval experiments

This repository prepares PubMedQA data, builds BM25 and dense retrieval indexes, compares retrieval results over document, section, and sentence chunks, and supports a small LLM answer-generation check.

## Requirements

- Python with `numpy` and `sentence-transformers` installed for indexing and retrieval.
- The `openai` package and an `OPENAI_API_KEY` environment variable for LLM answers.
- `transformers` and a locally cached tokenizer for the diagnostic's context-token counts.
- Network access for the initial dataset downloads and for downloading embedding or reranker models that are not already cached. The diagnostic only loads its tokenizer from the local Hugging Face cache.

There is currently no dependency lockfile or `requirements.txt` in the repository. From the repository root, create and activate a virtual environment, then install the packages used by the scripts:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install numpy sentence-transformers transformers
python -m pip install openai
```

The scripts below are shown for PowerShell and should be run from the repository root. `data/` is ignored by Git and is generated locally.

## Prepare the corpus

```powershell
python scripts\prepare_corpus.py
```

This downloads the labeled PubMedQA file (`ori_pqal.json`) and unlabeled file (`ori_pqau.json`), normalizes the records, creates question splits and relevance judgments, and writes three chunk collections. The data sources and download URLs are defined in `src/processing/load.py`.

The corpus contains every labeled PubMedQA document plus 10,000 distractor documents sampled from the unlabeled set with seed `42`. Documents shared by both input sets are deduplicated. Questions are shuffled with that same seeded generator and split 80/20 into `dev` and `test`; the exact split is reproducible for the same inputs. Each labeled question has one qrel pointing to its PubMed document.

Generated files:

```text
data/
├── raw/
│   ├── ori_pqal.json
│   └── ori_pqau.json
└── processed/
    ├── documents.jsonl
    ├── manifest.json
    ├── questions/
    │   ├── all.jsonl
    │   ├── dev.jsonl
    │   └── test.jsonl
    ├── qrels/
    │   ├── all.tsv
    │   ├── dev.tsv
    │   └── test.tsv
    └── chunks/
        ├── document.jsonl
        ├── section.jsonl
        └── sentence.jsonl
```

`manifest.json` records input checksums and record counts. The raw files are downloaded again each time `prepare_corpus.py` runs. The chunkers in `src/processing/` can also be run individually after corpus preparation, but `prepare_corpus.py` is the normal entry point.

## Build retrieval indexes

After preparing the corpus, build the indexes with:

```powershell
python scripts\prepare_indexes.py
```

This creates BM25 indexes for all three chunkings and dense indexes for both configured embedding models:

- `sentence-transformers/all-MiniLM-L6-v2`
- `NeuML/pubmedbert-base-embeddings`

The resulting paths are `data/indexes/bm25/<chunking>/` and `data/indexes/<model>/<chunking>/`, with model-name `/` characters replaced by `--`. For example, the MiniLM index directory is `data/indexes/sentence-transformers--all-MiniLM-L6-v2/section/`. Each index contains metadata and refers to its shared processed chunk file rather than copying it. At query time, the metadata checksum is verified so changed chunk files are detected.

Index preparation skips indexes whose metadata, model, and corpus checksum are already current. Dense embeddings are written in resumable blocks; an interrupted build can continue from its saved progress. The default embedding batch size is 64. Override it with `DENSE_BATCH_SIZE`, and set `DENSE_DEVICE` to `cuda` or `cpu` to choose the device; without it, Sentence Transformers selects the device.

## Run the retrieval diagnostic

`scripts/diagnostic.py` runs a baseline on the labeled `dev` questions. It compares BM25 and the two dense models across document, section, and sentence chunks, for nine configurations. It measures recall at small, medium, and large cutoffs (document k = 1, 3, and 5). Section and sentence k values are scaled using average whitespace-word counts per chunk so the estimated retrieved text is comparable across chunkings.

Run it after the corpus and indexes are ready:

```powershell
python scripts\diagnostic.py
```

The script settings, including split and tokenizer, are constants near the top of `scripts/diagnostic.py`; it does not currently accept command-line options. Each invocation writes `summary.csv`, `per_question.csv`, and `run.json` into the next numbered directory under `data/experiments/diagnostic/<split>/`. Token estimates use the configured tokenizer from the local Hugging Face cache and count selected chunk text only (excluding prompts and special tokens). Recall is the fraction of relevant source documents retrieved, averaged across questions. Retrieval latency is warm per-question scoring and top-k selection latency; it excludes model loading and token counting. The diagnostic does not run an answer-generating model. To evaluate `test`, change `SPLIT` in the script before running it.

## Build an experiment report

After at least one experiment has produced a `run.json` and summary CSV, generate a standalone HTML report:

```powershell
python scripts\report.py
```

By default, the report scans `data/experiments/` recursively and writes `data/experiments/report.html`. Open that file directly in a browser. The report embeds run metadata and summary rows, so it needs no web server or external libraries. It reads the summary CSV named in `run.json` (`summary_csv`), or falls back to `summary.csv`, `metrics.csv`, or `results.csv`. Diagnostic runs show chunking, retrieval, latency, average tokens, and recall for each size. Older diagnostic runs without size-specific recall need to be rerun to populate those values.

To use a different experiment root or output path:

```powershell
python scripts\report.py --experiments-dir path\to\experiments --output path\to\report.html
```

## Use retrieval from Python

Scorers return the highest-scoring `(chunk, score)` pairs in descending score order; the default is 10 results. Pass a larger `top_k` to retain more candidates. Set `top_k=None` only when every score is needed; that returns all scores in corpus order. Hybrid retrieval fuses bounded BM25 and dense candidate lists using reciprocal-rank fusion (RRF), since their raw scores have different scales. Reranking applies a cross-encoder to the best initial candidates.

```python
from src.scoring.bm25 import bm25_score
from src.scoring.dense import dense_score
from src.retrieval.hybrid import hybrid
from src.retrieval.rerank import rerank

question = "Does treatment X reduce outcome Y?"
model = "sentence-transformers/all-MiniLM-L6-v2"

bm25_top10 = bm25_score("section", question, {"k1": 1.5, "b": 0.75})
dense_top10 = dense_score("section", question, model)

bm25_candidates = bm25_score("section", question, top_k=50)
dense_candidates = dense_score("section", question, model, top_k=50)
hybrid_candidates = hybrid(
    bm25_candidates, dense_candidates, top_k=50, candidate_k=50
)
reranked_top10 = rerank(hybrid_candidates, question, top_k=10)
```

Dense query scoring requires the same embedding model used to build the selected index. Reranking uses `ncbi/MedCPT-Cross-Encoder` by default and downloads or loads that model when first used. Models are cached in-process for repeated queries.

## Generate an answer

`src/prompt/prompt.py` provides `answer_question(question, chunking_method,
scoring_method, retrieval_method)`. The scoring method is `"bm25"`, `"miniLM"`,
or `"Bert"`. The retrieval method is `None` for no context, `"top_k"` for
highest-scoring chunks, `"hybrid"` to combine BM25 and dense scoring, or
`"rerank"` to rerank candidates. `hybrid` requires a dense scoring method.
Pass all four arguments explicitly. The function returns a dictionary with the
answer, end-to-end latency, token counts, and estimated USD cost. Model,
passage limits, and token prices are constants at the top of
`src/prompt/prompt.py`.

Put your API key in a `.env` file in the repository root, then call the function:

```env
OPENAI_API_KEY=your-api-key
```

```powershell
python -c "from src.prompt.prompt import answer_question; print(answer_question('Does treatment X reduce outcome Y?', 'section', 'miniLM', 'hybrid'))"
```

Answers are normalized to exactly `yes`, `maybe`, or `no`. Current model
availability and pricing are listed in [OpenAI's API documentation](https://developers.openai.com/api/docs/pricing).

## Run the LLM experiment

After preparing the corpus and required indexes, add `OPENAI_API_KEY` to the
repository-root `.env` file and run the full answer-generation configuration
grid on either split:

```powershell
python scripts\llm_run.py dev
python scripts\llm_run.py test
```

The only command-line argument is `dev` or `test`. Each invocation gets the
next unused numbered directory, such as `data/experiments/llm/dev/1/`, so
previous runs are retained. The grid covers the no-context baseline and every
valid top-k, hybrid, and rerank configuration supported by `prompt.py` (25
configurations total). This makes one invocation issue 25 model calls per
question in the selected split.

Each run contains `run.json` metadata, `summary.csv` with per-configuration
accuracy, macro-F1, per-label precision/recall/F1, latency percentiles and
token counts, `per_question.csv` with individual predictions and timings, and
`summary.json` with confusion matrices and full metric details. Partial
results and a failed status are recorded if a run stops with an error.

## Repository map

```text
scripts/
├── prepare_corpus.py    # Download and prepare source data and chunks
├── prepare_indexes.py   # Build all configured BM25 and dense indexes
├── diagnostic.py        # Run the fixed retrieval baseline
├── llm_run.py           # Run sequential LLM answer-generation experiments
└── report.py            # Combine experiment summaries into HTML
src/
├── processing/          # Source loading, corpus creation, and chunking
├── indexing/            # BM25 and dense index builders
├── scoring/             # BM25 and dense query scoring
├── retrieval/           # Hybrid fusion and cross-encoder reranking
├── prompt/              # Context assembly and answer generation
└── common.py            # Shared paths, model caches, and helpers
notebook.ipynb            # Exploratory PubMedQA example
plan.md                   # Project plan and notes
```
