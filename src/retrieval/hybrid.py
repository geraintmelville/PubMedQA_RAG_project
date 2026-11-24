"""Hybrid fusion of independently scored candidate sets."""

from src.common import select_top_k


DEFAULT_CANDIDATE_K = 100
RRF_CONSTANT = 60


def hybrid(
    bm25_scored,
    dense_scored,
    top_k=10,
    candidate_k=DEFAULT_CANDIDATE_K,
    rrf_constant=RRF_CONSTANT,
):
    """Fuse BM25 and dense scores by reciprocal rank and return the best pairs.

    The two inputs can use unrelated score scales: only their within-method
    rankings contribute to the fused score. Returned scores are RRF scores.
    """
    fused = {}
    for scored_chunks in (bm25_scored, dense_scored):
        candidates = select_top_k(scored_chunks, candidate_k)
        ranked = sorted(candidates, key=lambda pair: pair[1], reverse=True)
        for rank, (chunk, _) in enumerate(ranked, start=1):
            chunk_id = chunk["id"]
            previous_chunk, previous_score = fused.get(chunk_id, (chunk, 0.0))
            fused[chunk_id] = (
                previous_chunk,
                previous_score + 1 / (rrf_constant + rank),
            )

    return select_top_k(list(fused.values()), top_k)
