import json
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / "src"))

from indexing.bm25 import build_bm25_index
from indexing.dense import build_dense_index


def build_hybrid_index(corpus_path, index_path, model_name):
    build_bm25_index(corpus_path, index_path / "bm25")
    build_dense_index(corpus_path, index_path / "dense", model_name)

    with (index_path / "metadata.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        json.dump(
            {
                "retriever": "hybrid",
                "embedding_model": model_name,
                "source_corpus": str(corpus_path),
                "rrf_constant": 60,
                "candidate_limit": 100,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")

    print(f"Saved hybrid index to {index_path}")


if __name__ == "__main__":
    corpus_path = project_dir / "data" / "corpus" / "section_chunks.jsonl"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_folder = model_name.replace("/", "--")
    index_path = project_dir / "data" / "indexes" / "section" / "hybrid" / model_folder
    build_hybrid_index(corpus_path, index_path, model_name)
