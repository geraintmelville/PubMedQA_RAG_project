from src.common import CHUNK_PATHS, EMBEDDING_MODELS, INDEXES_DIR
from src.retrieval.bm25 import build_index as build_bm25_index
from src.retrieval.dense import build_index as build_dense_index


EMBEDDING_MODEL_NAMES = tuple(EMBEDDING_MODELS.values())


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

    for model_name in EMBEDDING_MODEL_NAMES:
        for chunking, chunks_path in CHUNK_PATHS.items():
            build_dense_index(
                chunks_path,
                INDEXES_DIR / model_name.replace("/", "--") / chunking,
                model_name,
            )


if __name__ == "__main__":
    main()
