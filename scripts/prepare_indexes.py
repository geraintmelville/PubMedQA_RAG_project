from src.common import CHUNK_PATHS, EMBEDDING_MODELS, INDEXES_DIR, dense_index_path
from src.retrieval.bm25 import build_index as build_bm25_index
from src.retrieval.dense import build_index as build_dense_index


def main():
    missing_paths = [path for path in CHUNK_PATHS.values() if not path.is_file()]
    if missing_paths:
        missing_files = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing chunk files. Run scripts/prepare_corpus.py first:\n"
            f"{missing_files}"
        )

    for chunking, chunks_path in CHUNK_PATHS.items():
        build_bm25_index(chunks_path, INDEXES_DIR / "bm25" / chunking)

    for model_name in EMBEDDING_MODELS.values():
        for chunking, chunks_path in CHUNK_PATHS.items():
            build_dense_index(
                chunks_path,
                dense_index_path(model_name, chunking),
                model_name,
            )


if __name__ == "__main__":
    main()
