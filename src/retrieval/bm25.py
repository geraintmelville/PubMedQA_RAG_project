"""BM25 index construction and scoring."""

import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.common import (
    BM25_INDEX_FILENAME,
    CHUNK_PATHS,
    INDEXES_DIR,
    INDEX_METADATA_FILENAME,
    PROJECT_ROOT,
    _sha256,
    _write_json_atomic,
    load_index_chunks,
    load_jsonl,
    select_top_k_arrays,
)


DEFAULT_PARAMETERS = {"k1": 1.5, "b": 0.75}
INDEX_FORMAT_VERSION = 2


def tokenize(text):
    """Return lowercase word tokens shared by index construction and search."""
    return re.findall(r"\b\w+\b", text.lower())


def build_index(corpus_path, index_path):
    """Build a BM25 index from a chunk JSONL file."""
    corpus_path = Path(corpus_path).resolve()
    index_path = Path(index_path)
    corpus_hash = _sha256(corpus_path)
    metadata_path = index_path / INDEX_METADATA_FILENAME
    bm25_path = index_path / BM25_INDEX_FILENAME
    if metadata_path.is_file() and bm25_path.is_file():
        try:
            with metadata_path.open(encoding="utf-8") as input_file:
                metadata = json.load(input_file)
            if (
                metadata.get("retriever") == "bm25"
                and metadata.get("format_version") == INDEX_FORMAT_VERSION
                and metadata.get("source_corpus")
                == corpus_path.relative_to(PROJECT_ROOT).as_posix()
                and metadata.get("source_sha256") == corpus_hash
            ):
                print(f"BM25 index is current; skipping {index_path}")
                return False
        except (OSError, json.JSONDecodeError):
            pass

    corpus = load_jsonl(corpus_path)
    document_frequencies = Counter()
    postings = defaultdict(list)
    document_lengths = []

    for document_index, chunk in enumerate(corpus):
        frequencies = Counter(tokenize(chunk["text"]))
        document_lengths.append(sum(frequencies.values()))
        for term in frequencies:
            document_frequencies[term] += 1
            postings[term].append((document_index, frequencies[term]))

    index_path.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        bm25_path,
        {
            "document_frequencies": dict(document_frequencies),
            "postings": dict(postings),
            "document_lengths": document_lengths,
            "average_document_length": (
                sum(document_lengths) / len(document_lengths)
                if document_lengths
                else 0.0
            ),
        },
    )
    _write_json_atomic(
        metadata_path,
        {
            "retriever": "bm25",
            "format_version": INDEX_FORMAT_VERSION,
            "source_corpus": corpus_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_sha256": corpus_hash,
            "chunk_count": len(corpus),
        },
        indent=2,
    )
    _load_bm25_index.cache_clear()
    print(f"Saved BM25 index with {len(corpus)} chunks to {index_path}")
    return True


@lru_cache(maxsize=3)
def _load_bm25_index(index_path):
    """Keep the active BM25 index parsed across queries in one run."""
    with (index_path / BM25_INDEX_FILENAME).open(encoding="utf-8") as input_file:
        return json.load(input_file)


def score(chunking_method, query, parameters=None, top_k=10):
    """Return top ``(chunk, BM25 score)`` pairs, or all scores if top_k is None."""
    index_path = INDEXES_DIR / "bm25" / chunking_method
    index = _load_bm25_index(index_path)
    chunks = load_index_chunks(index_path)
    parameters = parameters or {}
    k1 = parameters.get("k1", DEFAULT_PARAMETERS["k1"])
    b = parameters.get("b", DEFAULT_PARAMETERS["b"])
    query_terms = set(tokenize(query))
    document_count = len(chunks)
    average_length = index["average_document_length"]
    document_lengths = np.asarray(index["document_lengths"], dtype=np.float64)
    length_norms = (
        1 - b + b * document_lengths / average_length
        if average_length
        else np.zeros(document_count, dtype=np.float64)
    )
    scores = np.zeros(document_count, dtype=np.float64)

    for term in query_terms:
        document_frequency = index["document_frequencies"].get(term, 0)
        if not document_frequency:
            continue
        inverse_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for document_index, frequency in index["postings"].get(term, []):
            scores[document_index] += inverse_frequency * (
                frequency * (k1 + 1)
                / (frequency + k1 * length_norms[document_index])
            )
    if top_k is not None:
        return select_top_k_arrays(chunks, scores, top_k)
    return list(zip(chunks, scores.tolist()))


# Backward-compatible names for scripts and external notebooks.
build_bm25_index = build_index
bm25_score = score
