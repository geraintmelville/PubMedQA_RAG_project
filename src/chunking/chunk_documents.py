import json
from pathlib import Path


project_dir = Path(__file__).resolve().parents[2]
input_path = project_dir / "data" / "interim" / "corpus.jsonl"
output_path = project_dir / "data" / "corpus" / "document_chunks.jsonl"


def chunk_documents():
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as input_file:
        documents = [json.loads(line) for line in input_file if line.strip()]

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for document in documents:
            text = "\n\n".join(
                context if isinstance(context, str) else context["text"]
                for context in document["contexts"]
            )
            chunk = {
                "id": f"{document['ID']}:document",
                "document_id": document["ID"],
                "chunk_type": "document",
                "chunk_index": 0,
                "source_split": document["source_split"],
                "text": text,
            }
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote document chunks to {output_path}")


if __name__ == "__main__":
    chunk_documents()
