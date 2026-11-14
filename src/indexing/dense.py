import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


project_dir = Path(__file__).resolve().parents[2]


def build_dense_index(corpus_path, index_path, model_name):
    with corpus_path.open(encoding="utf-8") as input_file:
        corpus = [json.loads(line) for line in input_file if line.strip()]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [chunk["text"] for chunk in corpus],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    index_path.mkdir(parents=True, exist_ok=True)
    with (index_path / "chunks.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for chunk in corpus:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    np.save(index_path / "embeddings.npy", embeddings)

    with (index_path / "metadata.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        json.dump(
            {
                "retriever": "dense",
                "embedding_model": model_name,
                "source_corpus": str(corpus_path),
                "chunk_count": len(corpus),
                "normalize_embeddings": True,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")

    print(f"Saved dense index with {len(corpus)} chunks to {index_path}")


if __name__ == "__main__":
    corpus_path = project_dir / "data" / "corpus" / "section_chunks.jsonl"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_folder = model_name.replace("/", "--")
    index_path = project_dir / "data" / "indexes" / "section" / "dense" / model_folder
    build_dense_index(corpus_path, index_path, model_name)
