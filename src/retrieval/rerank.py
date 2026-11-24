"""Cross-encoder reranking of scored candidate chunks."""

from src.common import get_cross_encoder, select_top_k


DEFAULT_CANDIDATE_K = 50
RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"


def rerank(
    scored_chunks,
    query,
    top_k=10,
    model_name=RERANKER_MODEL,
):
    """Rerank the best initial candidates and return cross-encoder score pairs."""
    if not scored_chunks:
        return []

    reranker = get_cross_encoder(model_name)
    pairs = [(query, chunk["text"]) for chunk, _ in scored_chunks]
    scores = reranker.predict(pairs)
    rescored = [(chunk, float(score)) for (chunk, _), score in zip(scored_chunks, scores)]
    return select_top_k(rescored, top_k)
