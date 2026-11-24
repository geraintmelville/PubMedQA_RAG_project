"""BM25 scoring over the prepared BM25 indexes."""

import json
import math
import re
from functools import lru_cache

import numpy as np

from src.common import (
    BM25_INDEX_FILENAME,
    INDEXES_DIR,
    load_index_chunks,
    select_top_k_arrays,
)


DEFAULT_PARAMETERS = {"k1": 1.5, "b": 0.75}


@lru_cache(maxsize=1)
def _load_bm25_index(index_path):
    """Keep the active BM25 index parsed across queries in one run."""
    with (index_path / BM25_INDEX_FILENAME).open(encoding="utf-8") as input_file:
        return json.load(input_file)


def bm25_score(chunking_method, query, parameters=None, top_k=10):
    """Return top ``(chunk, BM25 score)`` pairs, or all scores if top_k is None."""
    parameters = parameters or {}
    index_path = INDEXES_DIR / "bm25" / chunking_method
    index = _load_bm25_index(index_path)
    chunks = load_index_chunks(index_path)

    k1 = parameters.get("k1", DEFAULT_PARAMETERS["k1"])
    b = parameters.get("b", DEFAULT_PARAMETERS["b"])
    query_terms = set(re.findall(r"\b\w+\b", query.lower()))
    document_count = len(chunks)
    average_length = index["average_document_length"]
    scores = np.zeros(document_count, dtype=np.float64)

    for term in query_terms:
        document_frequency = index["document_frequencies"].get(term, 0)
        if not document_frequency:
            continue
        inverse_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for document_index in index["postings"].get(term, []):
            frequency = index["term_frequencies"][document_index].get(term, 0)
            length = index["document_lengths"][document_index]
            length_norm = (
                1 - b + b * length / average_length if average_length else 0.0
            )
            scores[document_index] += inverse_frequency * (
                frequency * (k1 + 1) / (frequency + k1 * length_norm)
            )
    if top_k is not None:
        return select_top_k_arrays(chunks, scores, top_k)
    return list(zip(chunks, scores.tolist()))
