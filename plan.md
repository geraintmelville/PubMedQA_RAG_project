# Plan: How Much Do Chunking, Embedding and Retrieval Choices Matter? A Controlled Study on PubMedQA

A controlled study with a research question, using a RAG pipeline as the test bench. Generation is a downstream check rather than the headline deliverable.

## 1. Framing (write this first, in the README)

- **Research question:** On biomedical question answering, how much does retrieval quality depend on chunking strategy, embedding model and retrieval method, and how much of any apparent advantage is an artefact of lexical overlap between questions and abstracts?
- **Sub-questions:**
  1. Does chunk granularity (whole abstract, section, sentence window) change retrieval quality, and does the answer depend on the retriever?
  2. Do biomedical embeddings beat general-purpose ones, and by how much?
  3. When does hybrid retrieval or reranking pay for its extra cost?
  4. How well do conclusions drawn on the original questions survive on paraphrased questions with less shared vocabulary?
- **Why it's worth doing:** most RAG write-ups report one pipeline's result. This one isolates the effect of each design decision, with uncertainty estimates, and shows which decisions matter and which don't.
- **Success criteria (decide now):** a complete factorial experiment grid, confidence intervals on every headline comparison, at least one finding that contradicts a default assumption or confirms it with evidence, and a reproducible one-command run.

## 2. Data setup

1. Load `pqa_labeled` (about 1k questions with yes/no/maybe labels and long answers).
2. Pool every question's context abstracts into **one shared corpus**. Add distractors from `pqa_unlabeled` or `pqa_artificial`, and treat corpus size as an experimental variable (for example 1k, 10k and 50k abstracts) so you can measure how methods degrade as retrieval gets harder.
3. Record each question's gold abstract(s) as retrieval ground truth. Also record sentence-level gold evidence where the structure allows, since chunking experiments need finer-grained relevance judgements.
4. Split into dev (for tuning) and held-out test (touched once). Fix random seeds. Tune hyperparameters only on dev.

## 3. Code structure

Keep the code organized around a repeatable path from the raw corpora to retrieval results. The current repository layout is:

```text
PubMedQA_RAG_project/
├── data/
│   ├── raw/                 # downloaded JSON, unchanged
│   └── processed/           # normalized corpus, chunks, indexes, and experiments
├── src/
│   ├── processing/          # source loading, corpus creation, and chunking
│   ├── retrieval/           # BM25, dense, hybrid, reranking, and search
│   ├── prompt.py            # context assembly and answer generation
│   └── common.py            # shared paths, model caches, and helpers
├── scripts/
│   ├── prepare_corpus.py
│   ├── prepare_indexes.py
│   ├── diagnostic.py
│   ├── llm_run.py
│   └── results.ipynb        # experiment summaries, plots, and bootstrap CIs
```

Keep raw inputs unchanged, and write derived data and experiment outputs separately. Use shared document, question, and qrel formats across retrieval methods so comparisons remain consistent.

## 4. Lexical overlap analysis (before building anything)

Measure overlap between each question and its gold abstract: token overlap, Jaccard, and fraction of question terms present. Compare against overlap with random abstracts.

**Role in the study:** PubMedQA questions derive from abstract titles, so overlap is probably very high, which would favour lexical methods. This analysis defines the **controlled variable** for the rest of the project. Split questions into overlap tertiles (high, medium, low), and add an LLM-paraphrased slice rewritten to avoid shared vocabulary. Every later result is reported by slice, so you can say where each method's advantage comes from.

Validate the paraphrases: check overlap actually drops and that a sample still preserves the question's meaning (hand-check about 50).

## 5. Experiment design

### Factors

| Factor | Levels |
| --- | --- |
| Chunking | whole abstract; section-level (background/methods/results/conclusion); sentence windows (sizes 1, 3, 5, with and without overlap); optionally fixed-token windows as a naive baseline |
| Lexical | BM25 (tune k1 and b; with and without stemming and stopword handling) |
| Embedding model | General-purpose MiniLM (`sentence-transformers/all-MiniLM-L6-v2`); biomedical PubMedBERT (`NeuML/pubmedbert-base-embeddings`); Alibaba GTE (`Alibaba-NLP/gte-modernbert-base`) |
| Retrieval method | BM25 only; each dense encoder; hybrid BM25 + dense via reciprocal rank fusion |
| Reranking | none vs. cross-encoder over the top 20-50 candidates |
| Corpus size | small, medium, large (from the distractor setup) |

### Design principle

Vary one factor at a time from a fixed baseline first (cheap, interpretable), then run the full grid driven by a single config file. For the grid, fit a simple model (for example regression on per-question reciprocal rank or recall) to estimate **main effects and interactions**, such as whether chunk size matters more for dense than for BM25.

### Metrics

- **Retrieval:** recall@k (k = 1, 5, 10, 20), MRR, nDCG@10. For sub-abstract chunks, report both chunk-level and abstract-level recall (a hit if any chunk from the gold abstract is retrieved).
- **Efficiency:** index build time, index size, query latency, and cost per query for each configuration.
- **Slices:** overlap tertiles, paraphrased slice, question type, and corpus size.
- **Uncertainty:** bootstrap confidence intervals over questions, and paired bootstrap or permutation tests for comparing configurations.

## 6. Analysis of embedding behaviour

This section is where the project goes beyond a benchmark table.

- **Embedding quality diagnostics:** similarity distributions for relevant vs. non-relevant pairs; how well each model separates them; hubness (documents that appear in too many top-k lists).
- **Chunking and embeddings:** does embedding quality degrade on very short or very long chunks? Plot retrieval quality against chunk length.
- **Context windows:** report each encoder's maximum sequence length and the fraction of chunks truncated at that limit, by chunking strategy.
- **Failure inspection:** for each retrieval method, categorise 30-50 misses (vocabulary mismatch, chunk boundary split the evidence, near-duplicate distractor, ambiguous question, label noise).
- **Complementarity:** measure the overlap between BM25 misses and dense misses. This is the empirical justification (or refutation) for hybrid retrieval.
- Optional: a small embedding visualisation (UMAP) coloured by chunk type or question category, if it adds insight.

## 7. Downstream check: does retrieval quality affect answers?

Keep this deliberately small. Its purpose is to test whether retrieval metrics predict end-task quality, not to build a product.

- Fix one generator and a simple prompt (cite passage IDs, answer yes/no/maybe, allow "insufficient evidence").
- Run it on 3-5 contrasting retrieval configurations from the grid (best, worst, cheapest, and so on).
- Measure accuracy and macro-F1 on yes/no/maybe, citation validity (cited IDs exist in the retrieved set), and abstention rate.
- Compare against oracle retrieval (gold abstracts supplied directly) and no retrieval, to bound how much retrieval can matter.
- **Key question:** does a retrieval gain of X points in recall@5 translate into any answer gain, or does generator behaviour dominate?

LLM-as-judge faithfulness evaluation is a stretch goal here, with the 50-example hand-labelled calibration if you include it.

## 8. Practical conclusions

- **Cost-versus-quality frontier:** plot retrieval quality against latency and cost for every configuration, and identify which ones sit on the Pareto frontier.
- **Recommendations:** state clearly which configuration you'd choose for different constraints (lowest cost, highest recall, best on low-overlap questions).
- **Threats to validity:** lexical leakage from title-derived questions, small labelled set, pooled-corpus artefacts (gold abstracts are easier to find than in a real index), model choice limited to what you tested, and generalisation beyond PubMed-style abstracts.

## 9. Deliverables

- A repo with pinned dependencies, a config-driven experiment runner, cached embeddings and indexes, and a one-command reproduction script.
- A results report or notebook: the experiment grid, main-effect and interaction analysis, slice breakdowns, embedding diagnostics, error taxonomy, and the cost/quality frontier.
- A small demo (Streamlit or Gradio) that lets you pick a chunking, embedding and retrieval configuration, run a query, and compare retrieved passages side by side.
- A README that leads with the research question, the headline findings and the limitations, with the architecture after that.

## 10. Suggested timeline

| Phase | Work | Rough time |
| --- | --- | --- |
| 1 | Data setup, splits, overlap analysis, paraphrased slice, BM25 baseline | 3-4 days |
| 2 | Chunking pipelines, embedding models, dense and hybrid retrieval, one-factor-at-a-time runs | 3-4 days |
| 3 | Reranking, full grid, statistical analysis (CIs, interactions) | 3-4 days |
| 4 | Embedding diagnostics, error taxonomy, downstream generation check | 3 days |
| 5 | Cost/latency analysis, demo, README and write-up | 2-3 days |

## 11. Risks to manage

- **Combinatorial explosion:** the grid can get large. Cache chunks and embeddings, run the one-factor sweep first, and prune dominated levels before the full grid.
- **Lexical leakage:** handled by overlap tertiles and the paraphrased slice.
- **Over-tuning on dev:** tune BM25 parameters and fusion weights on dev only, and report test once.
- **Over-claiming from a small set:** use bootstrap CIs, and say "no detectable difference" when intervals overlap.
- **Scope creep:** park query rewriting, metadata filtering, multi-hop questions and fine-tuning embeddings as stretch goals. Fine-tuning a biomedical embedding on the dev set is a good stretch extension if time allows.

## What changed from the original

- The research question and slice-based analysis now lead, replacing the product framing.
- Corpus size, chunk length, and embedding model are experimental factors.
- A new section analyses embedding behaviour and complementarity between retrievers.
- Statistical rigour is stronger: interaction effects and paired significance tests.
- Generation shrank to a downstream check on a few configurations, with faithfulness judging as a stretch goal.
- The business layer shrank to a cost/quality frontier and recommendations.
