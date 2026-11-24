import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.common import (
    CHUNK_PATHS,
    DENSE_EMBEDDINGS_FILENAME,
    DENSE_PARTIAL_EMBEDDINGS_FILENAME,
    DENSE_PROGRESS_FILENAME,
    INDEXES_DIR,
    INDEX_METADATA_FILENAME,
    PROJECT_ROOT,
    load_jsonl,
)


@lru_cache(maxsize=None)
def _load_embedding_model(model_name, device):
    return SentenceTransformer(model_name, device=device)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _write_dense_embeddings(
    model, texts, index_path, corpus_hash, model_name, batch_size, device
):
    count = len(texts)
    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        raise ValueError(f"Could not determine embedding size for {model_name}")

    partial_path = index_path / DENSE_PARTIAL_EMBEDDINGS_FILENAME
    progress_path = index_path / DENSE_PROGRESS_FILENAME
    checkpoint = {
        "source_sha256": corpus_hash,
        "embedding_model": model_name,
        "chunk_count": count,
        "embedding_dimension": dimension,
        "batch_size": batch_size,
        "device": device,
    }
    completed_rows = 0
    if partial_path.is_file() and progress_path.is_file():
        try:
            with progress_path.open(encoding="utf-8") as input_file:
                progress = json.load(input_file)
            partial = np.load(partial_path, mmap_mode="r+")
            partial_shape = partial.shape
            partial._mmap.close()
            if (
                all(progress.get(key) == value for key, value in checkpoint.items())
                and partial_shape == (count, dimension)
            ):
                completed_rows = min(progress.get("completed_rows", 0), count)
            else:
                partial_path.unlink()
        except (OSError, ValueError, IndexError, json.JSONDecodeError):
            partial_path.unlink(missing_ok=True)

    embeddings = np.lib.format.open_memmap(
        partial_path,
        mode="r+" if completed_rows else "w+",
        dtype=np.float32,
        shape=(count, dimension),
    )
    checkpoint_size = max(batch_size * 16, batch_size)
    for start in range(completed_rows, count, checkpoint_size):
        end = min(start + checkpoint_size, count)
        encoded = model.encode(
            texts[start:end],
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        embeddings[start:end] = encoded
        embeddings.flush()
        _write_json_atomic(
            progress_path,
            {**checkpoint, "completed_rows": end},
        )
        print(f"Saved embedding checkpoint {end}/{count} for {model_name}")
    embeddings.flush()
    del embeddings
    os.replace(partial_path, index_path / DENSE_EMBEDDINGS_FILENAME)
    progress_path.unlink(missing_ok=True)


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


def build_dense_index(
    corpus_path, index_path, model_name, model=None, batch_size=64, device=None
):
    corpus_path = Path(corpus_path).resolve()
    index_path = Path(index_path)
    corpus_hash = _sha256(corpus_path)
    if _index_is_current(index_path, corpus_path, model_name, corpus_hash):
        print(f"Dense index is current; skipping {index_path}")
        return False

    corpus = load_jsonl(corpus_path)

    if model is None:
        model = _load_embedding_model(model_name, device)
    index_path.mkdir(parents=True, exist_ok=True)
    _write_dense_embeddings(
        model,
        [chunk["text"] for chunk in corpus],
        index_path,
        corpus_hash,
        model_name,
        batch_size,
        device,
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
    )

    print(f"Saved dense index with {len(corpus)} chunks to {index_path}")
    return True


if __name__ == "__main__":
    corpus_path = CHUNK_PATHS["section"]
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    index_path = INDEXES_DIR / model_name.replace("/", "--") / "section"
    build_dense_index(corpus_path, index_path, model_name)
