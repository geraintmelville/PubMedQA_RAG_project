import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.common import (
    CHUNK_PATHS,
    DENSE_EMBEDDINGS_FILENAME,
    INDEXES_DIR,
    INDEX_METADATA_FILENAME,
    PROJECT_ROOT,
    _sha256,
    _write_json_atomic,
    get_embedding_model,
    load_index_chunks,
    load_jsonl,
    select_top_k_arrays,
)


@lru_cache(maxsize=3)
def _load_embeddings(embeddings_path):
    """Keep dense indexes memory-mapped across queries in one run."""
    return np.load(embeddings_path, mmap_mode="r")


def _index_is_current(index_path, corpus_path, model_name, corpus_hash):
    metadata_path = index_path / INDEX_METADATA_FILENAME
    embeddings_path = index_path / DENSE_EMBEDDINGS_FILENAME
    if not metadata_path.is_file() or not embeddings_path.is_file():
        return False
    try:
        with metadata_path.open(encoding="utf-8") as input_file:
            metadata = json.load(input_file)
        embeddings = np.load(embeddings_path, mmap_mode="r")
        embedding_count = embeddings.shape[0]
    except (OSError, ValueError, IndexError, json.JSONDecodeError):
        return False
    return (
        metadata.get("retriever") == "dense"
        and metadata.get("embedding_model") == model_name
        and metadata.get("source_corpus")
        == corpus_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        and metadata.get("source_sha256") == corpus_hash
        and metadata.get("chunk_count") == embedding_count
    )


def build_index(corpus_path, index_path, model_name, model=None):
    corpus_path = Path(corpus_path).resolve()
    index_path = Path(index_path)
    corpus_hash = _sha256(corpus_path)
    if _index_is_current(index_path, corpus_path, model_name, corpus_hash):
        print(f"Dense index is current; skipping {index_path}")
        return False

    corpus = load_jsonl(corpus_path)

    if model is None:
        model = get_embedding_model(model_name)
    index_path.mkdir(parents=True, exist_ok=True)
    _load_embeddings.cache_clear()
    np.save(
        index_path / DENSE_EMBEDDINGS_FILENAME,
        model.encode([chunk["text"] for chunk in corpus], normalize_embeddings=True),
    )

    _write_json_atomic(
        index_path / INDEX_METADATA_FILENAME,
        {
            "retriever": "dense",
            "embedding_model": model_name,
            "source_corpus": corpus_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_sha256": corpus_hash,
            "chunk_count": len(corpus),
            "normalize_embeddings": True,
        },
        indent=2,
    )

    print(f"Saved dense index with {len(corpus)} chunks to {index_path}")
    return True


def score(chunking_method, query, model, top_k=10):
    """Return dense cosine similarity pairs for a prepared embedding index."""
    if not model:
        raise ValueError("model must be the embedding model name used to build the index")
    index_path = INDEXES_DIR / model.replace("/", "--") / chunking_method
    chunks = load_index_chunks(index_path)
    embeddings = _load_embeddings(index_path / DENSE_EMBEDDINGS_FILENAME)
    embedding_model = get_embedding_model(model)
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_embedding
    if top_k is not None:
        return select_top_k_arrays(chunks, scores, top_k)
    return list(zip(chunks, scores.tolist()))


build_dense_index = build_index
dense_score = score


if __name__ == "__main__":
    corpus_path = CHUNK_PATHS["section"]
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    index_path = INDEXES_DIR / model_name.replace("/", "--") / "section"
    build_dense_index(corpus_path, index_path, model_name)
