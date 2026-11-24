"""Build document, section, and sentence chunks from normalized documents."""

import json
import re

from src.common import CHUNK_PATHS, DOCUMENTS_PATH, load_jsonl


def _document_chunks(document):
    text = "\n\n".join(context["text"] for context in document["contexts"])
    yield {
        "id": f"{document['ID']}:document",
        "document_id": document["ID"],
        "chunk_type": "document",
        "chunk_index": 0,
        "source": document["source"],
        "text": text,
    }


def _section_chunks(document):
    for chunk_index, context in enumerate(document["contexts"]):
        yield {
            "id": f"{document['ID']}:section:{chunk_index}",
            "document_id": document["ID"],
            "chunk_type": "section",
            "chunk_index": chunk_index,
            "section_index": chunk_index,
            "source": document["source"],
            "text": context["text"],
        }


def _sentence_chunks(document):
    chunk_index = 0
    for context_index, context in enumerate(document["contexts"]):
        sentences = re.split(r"(?<=[.!?])\s+", context["text"].strip())
        for sentence in sentences:
            if not sentence:
                continue
            yield {
                "id": f"{document['ID']}:sentence:{chunk_index}",
                "document_id": document["ID"],
                "chunk_type": "sentence",
                "chunk_index": chunk_index,
                "context_index": context_index,
                "source": document["source"],
                "text": sentence,
            }
            chunk_index += 1


# Keep this mapping in sync with CHUNK_PATHS: each key identifies both the
# chunking strategy and its output file.
CHUNKERS = {
    "document": _document_chunks,
    "section": _section_chunks,
    "sentence": _sentence_chunks,
}


def chunk(name):
    """Write chunks for one of the strategies named in ``CHUNK_PATHS``."""
    try:
        chunks_for_document = CHUNKERS[name]
        output_path = CHUNK_PATHS[name]
    except KeyError:
        raise ValueError(
            f"Unknown chunking strategy {name!r}; choose from {', '.join(CHUNKERS)}."
        ) from None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents = load_jsonl(DOCUMENTS_PATH)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for document in documents:
            for record in chunks_for_document(document):
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {name} chunks to {output_path}")


def chunk_all():
    """Write every chunk corpus declared in ``CHUNK_PATHS``."""
    for name in CHUNK_PATHS:
        chunk(name)


if __name__ == "__main__":
    chunk_all()
