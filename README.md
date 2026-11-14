# PubMedQA preparation

Run the data preparation and indexing scripts from the repository root:

```powershell
python scripts\prepare_corpus.py
python scripts\prepare_indexes.py
```

`prepare_corpus.py` downloads the labeled and unlabeled source files to
`data/raw/`, normalizes only the labeled gold documents and 10,000 sampled
distractors directly into `data/interim/corpus.jsonl`, writes the combined
questions and manifest to `data/interim/`, then writes `qrels.tsv`,
`dev_questions.jsonl`, `dev_qrels.tsv`, `test_questions.jsonl`, and
`test_qrels.tsv` to `data/corpus/` alongside the document, section, and
sentence chunks. This is the single script for corpus preparation; its
download, corpus, and chunking steps run in sequence.

`prepare_indexes.py` builds BM25, dense, hybrid, and reranking indexes for all
three chunking methods. Dense and hybrid indexes use both embedding models
defined at the top of that script. Model based indexes are stored under
`data/indexes/<chunking>/<retriever>/<model>/`; BM25 indexes are stored under
`data/indexes/<chunking>/bm25/`.

The chunking scripts in `src/chunking/` can also be run individually. The four
index builders in `src/indexing/` can be run individually after
editing the example corpus, model, and output paths near the bottom of each
file. The dense, hybrid, and reranking builders take a model name; BM25 does not.
