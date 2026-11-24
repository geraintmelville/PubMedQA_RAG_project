"""Cross-encoder reranking of scored candidate chunks."""

from src.common import get_cross_encoder, select_top_k


DEFAULT_CANDIDATE_K = 50
RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"


def rerank(
    scored_chunks,
    query,
    top_k=10,
    candidate_k=DEFAULT_CANDIDATE_K,
    model_name=RERANKER_MODEL,
):
    """Rerank the best initial candidates and return cross-encoder score pairs."""
    candidates = select_top_k(scored_chunks, candidate_k)
    if not candidates:
        return []

    reranker = get_cross_encoder(model_name)
    pairs = [(query, chunk["text"]) for chunk, _ in candidates]
    scores = reranker.predict(pairs)
    rescored = [(chunk, float(score)) for (chunk, _), score in zip(candidates, scores)]
    return select_top_k(rescored, top_k)
