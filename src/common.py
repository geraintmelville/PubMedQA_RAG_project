"""Shared project paths and helpers used across the pipeline."""

import csv
import hashlib
import heapq
import json
import math
import statistics
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QUESTIONS_DIR = PROCESSED_DIR / "questions"
QRELS_DIR = PROCESSED_DIR / "qrels"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
INDEXES_DIR = DATA_DIR / "indexes"

LABELED_DATA_PATH = RAW_DIR / "ori_pqal.json"
UNLABELED_DATA_PATH = RAW_DIR / "ori_pqau.json"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.jsonl"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
INDEX_METADATA_FILENAME = "metadata.json"
BM25_INDEX_FILENAME = "bm25.json"
DENSE_EMBEDDINGS_FILENAME = "embeddings.npy"
DENSE_PARTIAL_EMBEDDINGS_FILENAME = "embeddings.partial.npy"
DENSE_PROGRESS_FILENAME = "progress.json"
CHUNK_PATHS = {
    chunking: CHUNKS_DIR / f"{chunking}.jsonl"
    for chunking in ("document", "section", "sentence")
}


def load_jsonl(path):
    """Load non-empty JSON Lines records from ``path``."""
    with Path(path).open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def load_qrels(path):
    """Load a tab-separated qrels file into query-to-document relevance maps."""
    relevance = {}
    with Path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(
                    f"Expected four tab-separated qrel fields in {path}: {line!r}"
                )
            query_id, _, document_id, grade = fields
            relevance.setdefault(query_id, {})[document_id] = int(grade)
    return relevance


def write_csv(path, rows):
    """Write dictionaries as CSV, deriving columns from their first appearance."""
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean(values, default=None):
    """Return the arithmetic mean, or ``default`` when ``values`` is empty."""
    return statistics.fmean(values) if values else default


def rounded_mean(values, digits, default=None):
    """Return the rounded arithmetic mean, or ``default`` for no values."""
    value = mean(values)
    return round(value, digits) if value is not None else default


def percentile(values, percentile_value):
    """Return a linearly interpolated percentile, or ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def classification_metrics(rows, labels):
    """Calculate accuracy, per-class scores, and a confusion matrix."""
    confusion = {label: {predicted: 0 for predicted in labels} for label in labels}
    for row in rows:
        gold, predicted = row["gold_label"], row["predicted_label"]
        confusion[gold][predicted] += 1

    per_class = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in labels if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in labels if other != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "f1": round(f1, 5),
            "support": sum(confusion[label].values()),
        }

    correct = sum(confusion[label][label] for label in labels)
    total = len(rows)
    return {
        "accuracy": round(correct / total, 5) if total else None,
        "macro_f1": round(mean([per_class[label]["f1"] for label in labels]), 5)
        if total
        else None,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def allocate_run_dir(experiments_dir, split):
    """Create and return the next numbered run directory for a split."""
    split_dir = Path(experiments_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    run_number = 1
    while True:
        run_dir = split_dir / str(run_number)
        try:
            run_dir.mkdir()
            return run_number, run_dir
        except FileExistsError:
            run_number += 1


def load_index_chunks(index_path):
    """Load the shared source corpus and verify it matches an index manifest."""
    index_path = Path(index_path)
    with (index_path / INDEX_METADATA_FILENAME).open(encoding="utf-8") as input_file:
        metadata = json.load(input_file)
    source_path = (PROJECT_ROOT / metadata["source_corpus"]).resolve()
    expected_hash = metadata["source_sha256"]
    expected_count = metadata["chunk_count"]
    return _load_verified_chunks(str(source_path), expected_hash, expected_count)


@lru_cache(maxsize=8)
def _load_verified_chunks(source_path, expected_hash, expected_count):
    digest = hashlib.sha256()
    with Path(source_path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Source chunks at {source_path} changed since the index was built; "
            "run scripts/prepare_indexes.py again."
        )
    chunks = load_jsonl(source_path)
    if len(chunks) != expected_count:
        raise ValueError(
            f"Source chunks at {source_path} contain {len(chunks)} rows, "
            f"but the index expects {expected_count}."
        )
    return chunks


@lru_cache(maxsize=None)
def get_embedding_model(model_name):
    """Load and cache an embedding model for repeated query encoding."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=None)
def get_cross_encoder(model_name):
    """Load and cache a cross-encoder used to rerank retrieved candidates."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def select_top_k(scored_chunks, k=10):
    """Return the highest-scoring ``(chunk, score)`` pairs."""
    return heapq.nlargest(max(k, 0), scored_chunks, key=lambda pair: pair[1])


def select_top_k_arrays(chunks, scores, k):
    """Select top-scoring chunk indices before creating Python result pairs."""
    import numpy as np

    k = min(max(int(k), 0), len(scores))
    if k == 0:
        return []
    if k < len(scores):
        indices = np.argpartition(scores, len(scores) - k)[-k:]
    else:
        indices = np.arange(len(scores))
    indices = indices[np.argsort(scores[indices])[::-1]]
    return [(chunks[int(index)], float(scores[index])) for index in indices]
