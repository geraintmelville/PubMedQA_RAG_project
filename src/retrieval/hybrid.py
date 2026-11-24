"""Hybrid fusion of independently scored candidate sets."""


DEFAULT_CANDIDATE_K = 100
RRF_CONSTANT = 60


def hybrid(
    bm25_ranked,
    dense_ranked,
    top_k=10,
    k=RRF_CONSTANT,
):
    """Fuse BM25 and dense scores by reciprocal rank and return the best pairs.

    The two inputs can use unrelated score scales: only their within-method
    rankings contribute to the fused score. Returned scores are RRF scores.
    """
    scores, chunks = {}, {}
    for ranked in (bm25_ranked, dense_ranked):
        for rank, (chunk, _) in enumerate(ranked, start=1):
            chunk_id = chunk["id"]
            chunks[chunk_id] = chunk
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    best = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [(chunks[chunk_id], scores[chunk_id]) for chunk_id in best]
