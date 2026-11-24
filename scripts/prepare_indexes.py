import os
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.common import CHUNK_PATHS, INDEXES_DIR
from src.indexing.bm25 import build_bm25_index
from src.indexing.dense import build_dense_index


GENERAL_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
BIOMEDICAL_EMBEDDER = "NeuML/pubmedbert-base-embeddings"
CHUNKING_METHODS = ("document", "section", "sentence")
EMBEDDING_MODELS = (GENERAL_EMBEDDER, BIOMEDICAL_EMBEDDER)
def main():
    chunk_paths = {chunking: CHUNK_PATHS[chunking] for chunking in CHUNKING_METHODS}
    missing_paths = [path for path in chunk_paths.values() if not path.is_file()]
    if missing_paths:
        missing_files = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing chunk files. Run scripts/prepare_corpus.py first:\n"
            f"{missing_files}"
        )

    for chunking in CHUNKING_METHODS:
        chunks_path = chunk_paths[chunking]
        build_bm25_index(chunks_path, INDEXES_DIR / "bm25" / chunking)

    batch_size = int(os.environ.get("DENSE_BATCH_SIZE", "64"))
    device = os.environ.get("DENSE_DEVICE") or None
    for model_name in EMBEDDING_MODELS:
        for chunking in CHUNKING_METHODS:
            build_dense_index(
                chunk_paths[chunking],
                INDEXES_DIR / model_name.replace("/", "--") / chunking,
                model_name,
                batch_size=batch_size,
                device=device,
            )


if __name__ == "__main__":
    main()
