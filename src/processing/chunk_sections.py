import json

from src.common import CHUNK_PATHS, DOCUMENTS_PATH, load_jsonl

input_path = DOCUMENTS_PATH
output_path = CHUNK_PATHS["section"]


def chunk_sections():
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents = load_jsonl(input_path)

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for document in documents:
            for chunk_index, context in enumerate(document["contexts"]):
                chunk = {
                    "id": f"{document['ID']}:section:{chunk_index}",
                    "document_id": document["ID"],
                    "chunk_type": "section",
                    "chunk_index": chunk_index,
                    "section_index": chunk_index,
                    "source_split": document["source_split"],
                    "text": context if isinstance(context, str) else context["text"],
                }
                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote section chunks to {output_path}")


if __name__ == "__main__":
    chunk_sections()
