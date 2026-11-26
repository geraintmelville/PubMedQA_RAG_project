# PubMedQA retrieval experiments

This repository prepares PubMedQA data, builds BM25 and dense retrieval indexes, compares retrieval results over document, section, and sentence chunks, and supports a small LLM answer-generation check.

## Requirements

- Python with `numpy` and `sentence-transformers` installed for indexing and retrieval.
- The `openai` and `python-dotenv` packages and an `OPENAI_API_KEY` environment variable for LLM answers.
- Network access for the initial dataset downloads and for downloading embedding or reranker models that are not already cached.

There is no dependency lockfile. The project metadata in `pyproject.toml` declares the retrieval and diagnostic dependencies; LLM and notebook dependencies are optional extras. From the repository root, create and activate a virtual environment, then install the project in editable mode:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[llm,notebook]"
```

Editable installation makes the `src` package importable from Python launched
outside the repository, including notebooks and tests using this environment.
Use `python -m pip install -e .` if you only need retrieval and diagnostic
dependencies, or add `[llm]` and/or `[notebook]` for those features.

The scripts below are shown for PowerShell and should be run from the repository root. `data/` is ignored by Git and is generated locally.

## Prepare the corpus

```powershell
python scripts\prepare_corpus.py
```

This downloads the labeled PubMedQA file (`ori_pqal.json`) and unlabeled file (`ori_pqau.json`), normalizes the records, creates question splits and relevance judgments, and writes three chunk collections. The data sources and download URLs are defined in `src/processing/load.py`.

The corpus contains every labeled PubMedQA document plus 10,000 distractor documents sampled from unlabeled records whose IDs are not in the labeled set, with seed `42`. The manifest's `duplicate_count` records how many IDs the two source files share. Questions are shuffled with that same seeded generator and split 25/75 into `dev` and `test`; the exact split is reproducible for the same inputs. Each labeled question has one qrel pointing to its PubMed document.

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

`manifest.json` records input checksums and record counts. Existing raw files are reused, and only missing files are downloaded. The chunkers in `src/processing/` can also be run individually after corpus preparation, but `prepare_corpus.py` is the normal entry point.

## Build retrieval indexes

After preparing the corpus, build the indexes with:

```powershell
python scripts\prepare_indexes.py
```

To prepare only one or more dense models, select their nicknames. For example, to build only GTE:

```powershell
python -m scripts.prepare_indexes --models GTE
```

This creates BM25 indexes for all three chunkings and dense indexes only for the selected models. Without `--models`, all configured dense models are selected:

- `sentence-transformers/all-MiniLM-L6-v2`
- `NeuML/pubmedbert-base-embeddings`
- `Alibaba-NLP/gte-modernbert-base`

The resulting paths are `data/indexes/bm25/<chunking>/` and `data/indexes/<model>/<chunking>/`, with model-name `/` characters replaced by `--`. For example, the MiniLM index directory is `data/indexes/sentence-transformers--all-MiniLM-L6-v2/section/`. Each index contains metadata and refers to its shared processed chunk file rather than copying it. When indexes are prepared, the metadata checksum is compared with the chunk file to decide whether the index needs rebuilding.

Index preparation skips indexes whose metadata, model, and corpus checksum are already current. Adding a model therefore creates its dense indexes while reusing current BM25 and dense indexes for the existing models. Dense embeddings are encoded and saved as a NumPy array. Sentence Transformers selects the device and batch behavior.

## Run the retrieval diagnostic

`scripts/diagnostic.py` runs a baseline on a selected labeled split. It compares BM25 and the three dense models across document, section, and sentence chunks, for twelve configurations. It measures recall at small, medium, and large cutoffs (document k = 1, 3, and 5). Section and sentence k values are scaled using average whitespace-word counts per chunk so the estimated retrieved text is comparable across chunkings.

Run it after the corpus and indexes are ready:

```powershell
python scripts\diagnostic.py dev
python scripts\diagnostic.py test
```

Each invocation writes `per_question.csv` first, then derives `summary.csv` with `pandas.DataFrame.groupby(["chunking", "method"]).mean()`, and records run metadata in `run.json`. Context size estimates count whitespace-separated words in selected chunk text only (excluding prompts), matching the word-count basis used to scale k across chunkings. Recall is the fraction of relevant source documents retrieved, averaged across questions. Retrieval latency is warm per-question scoring and top-k selection latency; it excludes model loading and word counting. The diagnostic does not run an answer-generating model.

## Explore results

Open `scripts/results.ipynb` from the repository root in Jupyter after running one or more experiments:

```powershell
jupyter notebook scripts/results.ipynb
```

The notebook reads the latest numbered diagnostic and LLM run for the selected split, summarizes per-question retrieval with pandas, plots medium-budget recall and answer accuracy, and reports percentile bootstrap confidence intervals for recall. Change `SPLIT` in the first code cell to `test` to inspect the test runs. Run metadata records diagnostic latency as warm retrieval and top-k selection time (model loading and word counting excluded), and context size as whitespace-separated words in selected chunks (prompts excluded). Section and sentence cutoffs scale document cutoffs by average chunk word count. LLM latency is end-to-end and can include first-use loading; token usage comes from the Responses API and averages exclude unreported usage; cost is estimated using the rates configured in `src/prompt.py`, including the cached-input rate.

## Use retrieval from Python

Scorers return the highest-scoring `(chunk, score)` pairs in descending score order; the default is 10 results. Pass a larger `top_k` to retain more candidates. Set `top_k=None` only when every score is needed; that returns all scores in corpus order. Hybrid retrieval fuses bounded BM25 and dense candidate lists using reciprocal-rank fusion (RRF), since their raw scores have different scales. Reranking applies a cross-encoder to the best initial candidates.

```python
from src.retrieval.search import search

question = "Does treatment X reduce outcome Y?"

bm25_top10 = search("section", question, scoring="bm25")
dense_top10 = search("section", question, scoring="miniLM")
hybrid_top10 = search("section", question, scoring="miniLM", method="hybrid")
reranked_top10 = search("section", question, scoring="miniLM", method="rerank")
```

Dense query scoring requires the same embedding model used to build the selected index. Reranking uses `ncbi/MedCPT-Cross-Encoder` by default and downloads or loads that model when first used. Models are cached in-process for repeated queries.

## Generate an answer

`src/prompt.py` provides `answer_question(question, chunking_method,
scoring_method, retrieval_method)`. The scoring method is `"bm25"`, `"miniLM"`,
or `"Bert"`. The retrieval method is `None` for no context, `"top_k"` for
highest-scoring chunks, `"hybrid"` to combine BM25 and dense scoring, or
`"rerank"` to rerank candidates. `hybrid` requires a dense scoring method.
Pass all four arguments explicitly. The function returns a dictionary with the
answer, end-to-end latency, token counts, and estimated USD cost. Model,
passage limits, and token prices are constants at the top of
`src/prompt.py`.

Put your API key in a `.env` file in the repository root, then call the function:

```env
OPENAI_API_KEY=your-api-key
```

```powershell
python -c "from src.prompt import answer_question; print(answer_question('Does treatment X reduce outcome Y?', 'section', 'miniLM', 'hybrid'))"
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
├── diagnostic.py        # Run retrieval diagnostics on dev or test
├── llm_run.py           # Run sequential LLM answer-generation experiments
└── results.ipynb        # Experiment summaries, plots, and bootstrap CIs
src/
├── processing/          # Source loading, corpus creation, and chunking
├── retrieval/           # BM25, dense, hybrid, reranking, and search
├── prompt.py            # Context assembly and answer generation
└── common.py            # Shared paths, model caches, and helpers
plan.md                   # Project plan and notes
```
