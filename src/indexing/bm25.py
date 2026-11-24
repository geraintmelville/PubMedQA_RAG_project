import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.common import (
    BM25_INDEX_FILENAME,
    CHUNK_PATHS,
    INDEXES_DIR,
    INDEX_METADATA_FILENAME,
    PROJECT_ROOT,
    load_jsonl,
)



def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path, value, indent=None):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, indent=indent)
        output_file.write("\n")
    os.replace(temporary_path, path)


def build_bm25_index(corpus_path, index_path):
    corpus_path = Path(corpus_path).resolve()
    index_path = Path(index_path)
    corpus_hash = _sha256(corpus_path)
    metadata_path = index_path / INDEX_METADATA_FILENAME
    bm25_path = index_path / BM25_INDEX_FILENAME
    if metadata_path.is_file() and bm25_path.is_file():
        try:
            with metadata_path.open(encoding="utf-8") as input_file:
                metadata = json.load(input_file)
            if (
                metadata.get("retriever") == "bm25"
                and metadata.get("source_corpus")
                == corpus_path.relative_to(PROJECT_ROOT).as_posix()
                and metadata.get("source_sha256") == corpus_hash
            ):
                print(f"BM25 index is current; skipping {index_path}")
                return False
        except (OSError, json.JSONDecodeError):
            pass

    corpus = load_jsonl(corpus_path)
    term_frequencies = []
    document_frequencies = Counter()
    postings = defaultdict(list)
    document_lengths = []

    for document_index, chunk in enumerate(corpus):
        frequencies = Counter(re.findall(r"\b\w+\b", chunk["text"].lower()))
        term_frequencies.append(dict(frequencies))
        document_lengths.append(sum(frequencies.values()))
        for term in frequencies:
            document_frequencies[term] += 1
            postings[term].append(document_index)

    index_path.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        bm25_path,
        {
            "k1": 1.5,
            "b": 0.75,
            "term_frequencies": term_frequencies,
            "document_frequencies": dict(document_frequencies),
            "postings": dict(postings),
            "document_lengths": document_lengths,
            "average_document_length": (
                sum(document_lengths) / len(document_lengths)
                if document_lengths
                else 0.0
            ),
        },
    )
    _write_json_atomic(
        metadata_path,
        {
            "retriever": "bm25",
            "source_corpus": corpus_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_sha256": corpus_hash,
            "chunk_count": len(corpus),
        },
        indent=2,
    )

    print(f"Saved BM25 index with {len(corpus)} chunks to {index_path}")
    return True


if __name__ == "__main__":
    corpus_path = CHUNK_PATHS["section"]
    index_path = INDEXES_DIR / "bm25" / "section"
    build_bm25_index(corpus_path, index_path)
