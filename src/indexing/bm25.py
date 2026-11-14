import json
import re
from collections import Counter, defaultdict
from pathlib import Path


project_dir = Path(__file__).resolve().parents[2]


def read_corpus(corpus_path):
    with corpus_path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def build_bm25_index(corpus_path, index_path):
    corpus = read_corpus(corpus_path)
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
    with (index_path / "chunks.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for chunk in corpus:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with (index_path / "bm25.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        json.dump(
            {
                "k1": 1.5,
                "b": 0.75,
                "term_frequencies": term_frequencies,
                "document_frequencies": dict(document_frequencies),
                "postings": dict(postings),
                "document_lengths": document_lengths,
                "average_document_length": sum(document_lengths) / len(document_lengths),
            },
            output_file,
        )

    with (index_path / "metadata.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        json.dump(
            {
                "retriever": "bm25",
                "source_corpus": str(corpus_path),
                "chunk_count": len(corpus),
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")

    print(f"Saved BM25 index with {len(corpus)} chunks to {index_path}")


if __name__ == "__main__":
    corpus_path = project_dir / "data" / "corpus" / "section_chunks.jsonl"
    index_path = project_dir / "data" / "indexes" / "section" / "bm25"
    build_bm25_index(corpus_path, index_path)
