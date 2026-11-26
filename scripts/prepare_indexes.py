import argparse

from src.common import CHUNK_PATHS, EMBEDDING_MODELS, INDEXES_DIR, dense_index_path
from src.retrieval.bm25 import build_index as build_bm25_index
from src.retrieval.dense import build_index as build_dense_index


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare BM25 and dense indexes.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=EMBEDDING_MODELS,
        default=None,
        metavar="MODEL",
        help=(
            "dense model nickname(s) to prepare; defaults to all configured models "
            f"({', '.join(EMBEDDING_MODELS)})"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    missing_paths = [path for path in CHUNK_PATHS.values() if not path.is_file()]
    if missing_paths:
        missing_files = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing chunk files. Run scripts/prepare_corpus.py first:\n"
            f"{missing_files}"
        )

    selected_models = (
        set(args.models) if args.models is not None else set(EMBEDDING_MODELS)
    )
    model_nicknames = [
        nickname for nickname in EMBEDDING_MODELS if nickname in selected_models
    ]
    print(f"Preparing dense indexes for: {', '.join(model_nicknames)}", flush=True)

    for chunking, chunks_path in CHUNK_PATHS.items():
        build_bm25_index(chunks_path, INDEXES_DIR / "bm25" / chunking)

    for nickname in model_nicknames:
        model_name = EMBEDDING_MODELS[nickname]
        for chunking, chunks_path in CHUNK_PATHS.items():
            build_dense_index(
                chunks_path,
                dense_index_path(model_name, chunking),
                model_name,
            )


if __name__ == "__main__":
    main()
