import json
import re

from src.common import CHUNK_PATHS, DOCUMENTS_PATH, load_jsonl

input_path = DOCUMENTS_PATH
output_path = CHUNK_PATHS["sentence"]


def chunk_sentences():
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents = load_jsonl(input_path)

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for document in documents:
            chunk_index = 0
            for context_index, context in enumerate(document["contexts"]):
                context_text = context if isinstance(context, str) else context["text"]
                sentences = re.split(r"(?<=[.!?])\s+", context_text.strip())
                for sentence in sentences:
                    if not sentence:
                        continue
                    chunk = {
                        "id": f"{document['ID']}:sentence:{chunk_index}",
                        "document_id": document["ID"],
                        "chunk_type": "sentence",
                        "chunk_index": chunk_index,
                        "context_index": context_index,
                        "source_split": document["source_split"],
                        "text": sentence,
                    }
                    output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    chunk_index += 1

    print(f"Wrote sentence chunks to {output_path}")


if __name__ == "__main__":
    chunk_sentences()
