"""Dispatch retrieval methods through the shared scoring implementations."""

from src.common import EMBEDDING_MODELS
from src.retrieval.hybrid import DEFAULT_CANDIDATE_K, hybrid
from src.retrieval.rerank import DEFAULT_CANDIDATE_K as DEFAULT_RERANK_CANDIDATE_K
from src.retrieval.rerank import rerank
from src.retrieval.bm25 import score as bm25_score


def _score(chunking, query, scoring, top_k):
    if scoring == "bm25":
        return bm25_score(chunking, query, top_k=top_k)
    if scoring not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown scoring method: {scoring}")
    # Keep torch and sentence-transformers out of BM25-only runs.
    from src.retrieval.dense import score as dense_score

    return dense_score(chunking, query, EMBEDDING_MODELS[scoring], top_k=top_k)


def search(chunking, query, scoring, method="top_k", top_k=10, candidate_k=None):
    """Retrieve chunks using top-k scoring, reciprocal-rank fusion, or reranking.

    ``scoring`` is ``bm25``, ``miniLM``, or ``Bert``; ``method`` is
    ``top_k``, ``hybrid``, or ``rerank``.
    """
    if method == "top_k":
        return _score(chunking, query, scoring, top_k)
    if method == "hybrid":
        if scoring == "bm25":
            raise ValueError("hybrid needs a dense scoring method")
        k = candidate_k or DEFAULT_CANDIDATE_K
        return hybrid(
            _score(chunking, query, "bm25", k),
            _score(chunking, query, scoring, k),
            top_k=top_k,
        )
    if method == "rerank":
        k = candidate_k or DEFAULT_RERANK_CANDIDATE_K
        return rerank(
            _score(chunking, query, scoring, k),
            query,
            top_k=top_k,
        )
    raise ValueError(method)
