import json

from src.common import CHUNK_PATHS, DOCUMENTS_PATH, load_jsonl

input_path = DOCUMENTS_PATH
output_path = CHUNK_PATHS["document"]


def chunk_documents():
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents = load_jsonl(input_path)

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
