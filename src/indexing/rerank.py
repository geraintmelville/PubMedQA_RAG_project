import json
import sys
from pathlib import Path


project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / "src"))

from indexing.bm25 import build_bm25_index


def build_rerank_index(corpus_path, index_path, model_name):
    build_bm25_index(corpus_path, index_path / "bm25")

    with (index_path / "metadata.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        json.dump(
            {
                "retriever": "rerank",
                "reranker_model": model_name,
                "source_corpus": str(corpus_path),
                "candidate_retriever": "bm25",
                "candidate_limit": 50,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")

    print(f"Saved reranking index to {index_path}")


if __name__ == "__main__":
    corpus_path = project_dir / "data" / "corpus" / "section_chunks.jsonl"
    model_name = "ncbi/MedCPT-Cross-Encoder"
    model_folder = model_name.replace("/", "--")
    index_path = project_dir / "data" / "indexes" / "section" / "rerank" / model_folder
    build_rerank_index(corpus_path, index_path, model_name)
