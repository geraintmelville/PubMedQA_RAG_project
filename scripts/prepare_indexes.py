import sys
from pathlib import Path


project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir / "src"))

from src.indexing.bm25 import build_bm25_index
from src.indexing.dense import build_dense_index
from src.indexing.hybrid import build_hybrid_index
from src.indexing.rerank import build_rerank_index


GENERAL_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
BIOMEDICAL_EMBEDDER = "NeuML/pubmedbert-base-embeddings"
RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"
CHUNKING_METHODS = ("document", "section", "sentence")
EMBEDDING_MODELS = (GENERAL_EMBEDDER, BIOMEDICAL_EMBEDDER)


def main():
    for chunking in CHUNKING_METHODS:
        chunks_path = project_dir / "data" / "corpus" / f"{chunking}_chunks.jsonl"
        indexes_dir = project_dir / "data" / "indexes" / chunking

        build_bm25_index(chunks_path, indexes_dir / "bm25")
        for model in EMBEDDING_MODELS:
            model_folder = model.replace("/", "--")
            build_dense_index(
                chunks_path, indexes_dir / "dense" / model_folder, model
            )
            build_hybrid_index(
                chunks_path, indexes_dir / "hybrid" / model_folder, model
            )

        reranker_folder = RERANKER_MODEL.replace("/", "--")
        build_rerank_index(
            chunks_path,
            indexes_dir / "rerank" / reranker_folder,
            RERANKER_MODEL,
        )


if __name__ == "__main__":
    main()
