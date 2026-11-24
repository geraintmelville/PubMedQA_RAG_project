"""Dense vector scoring over the prepared embedding indexes."""

import numpy as np

from src.common import (
    DENSE_EMBEDDINGS_FILENAME,
    INDEXES_DIR,
    get_embedding_model,
    load_index_chunks,
    select_top_k_arrays,
)


def dense_score(chunking_method, query, model, top_k=10):
    """Return ``(chunk, cosine similarity)`` pairs.

    ``model`` is the embedding model name used to build the index.
    Without ``top_k``, results are returned in corpus order. With ``top_k``,
    the highest-scoring results are returned in descending score order.
    Top-k selection happens on the NumPy score array before Python result
    objects are created, avoiding a pair for every indexed chunk.
    """
    if not model:
        raise ValueError(
            "model must be the embedding model name used to build the index"
        )
    index_path = INDEXES_DIR / model.replace("/", "--") / chunking_method
    chunks = load_index_chunks(index_path)
    embeddings = np.load(index_path / DENSE_EMBEDDINGS_FILENAME, mmap_mode="r")
    embedding_model = get_embedding_model(model)
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_embedding

    if top_k is not None:
        return select_top_k_arrays(chunks, scores, top_k)

    return list(zip(chunks, scores.tolist()))
