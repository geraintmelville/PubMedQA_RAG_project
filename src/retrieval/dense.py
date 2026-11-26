import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.common import (
    DENSE_EMBEDDINGS_FILENAME,
    INDEX_METADATA_FILENAME,
    PROJECT_ROOT,
    dense_index_path,
    sha256_file,
    write_json,
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
        and isinstance(metadata.get("max_seq_length"), int)
        and isinstance(metadata.get("truncation_rate"), (int, float))
        and isinstance(metadata.get("truncated_chunk_count"), int)
        and isinstance(metadata.get("truncation_sample_count"), int)
        and metadata.get("truncation_sample_count") == embedding_count
    )


def _truncation_stats(tokenizer, texts, max_seq_length, batch_size=256):
    """Count corpus texts whose tokenized input exceeds the encoder limit."""
    truncated_count = 0
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        truncated_count += sum(
            len(input_ids) > max_seq_length for input_ids in tokenized["input_ids"]
        )
    return {
        "truncation_rate": truncated_count / len(texts) if texts else 0.0,
        "truncated_chunk_count": truncated_count,
        "truncation_sample_count": len(texts),
    }


def build_index(corpus_path, index_path, model_name, model=None):
    corpus_path = Path(corpus_path).resolve()
    index_path = Path(index_path)
    corpus_hash = sha256_file(corpus_path)
    if model is None:
        model = get_embedding_model(model_name)
    max_seq_length = int(model.max_seq_length)
    if _index_is_current(index_path, corpus_path, model_name, corpus_hash):
        with (index_path / INDEX_METADATA_FILENAME).open(encoding="utf-8") as input_file:
            metadata = json.load(input_file)
        if metadata.get("max_seq_length") == max_seq_length:
            print(f"Dense index is current; skipping {index_path}")
            return False

    corpus = load_jsonl(corpus_path)
    texts = [chunk["text"] for chunk in corpus]
    truncation = _truncation_stats(model.tokenizer, texts, max_seq_length)
    index_path.mkdir(parents=True, exist_ok=True)
    _load_embeddings.cache_clear()
    print(
        f"Encoding {len(corpus)} chunks with {model_name} for {index_path.name}",
        flush=True,
    )
    np.save(
        index_path / DENSE_EMBEDDINGS_FILENAME,
        model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
    )

    write_json(
        index_path / INDEX_METADATA_FILENAME,
        {
            "retriever": "dense",
            "embedding_model": model_name,
            "source_corpus": corpus_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_sha256": corpus_hash,
            "chunk_count": len(corpus),
            "normalize_embeddings": True,
            "max_seq_length": max_seq_length,
            **truncation,
        },
        indent=2,
    )

    print(f"Saved dense index with {len(corpus)} chunks to {index_path}")
    return True


def score(chunking_method, query, model_name, top_k=10):
    """Return dense cosine similarity pairs for a prepared embedding index."""
    if not model_name:
        raise ValueError("model_name must be the embedding model name used to build the index")
    index_path = dense_index_path(model_name, chunking_method)
    chunks = load_index_chunks(index_path)
    embeddings = _load_embeddings(index_path / DENSE_EMBEDDINGS_FILENAME)
    embedding_model = get_embedding_model(model_name)
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_embedding
    if top_k is not None:
        return select_top_k_arrays(chunks, scores, top_k)
    return list(zip(chunks, scores.tolist()))


