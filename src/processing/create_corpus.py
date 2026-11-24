import hashlib
import json
import random

from src.common import (
    DOCUMENTS_PATH,
    LABELED_DATA_PATH,
    MANIFEST_PATH,
    PROCESSED_DIR,
    QRELS_DIR,
    QUESTIONS_DIR,
    UNLABELED_DATA_PATH,
)


def _normalize_document(pmid, record, source_split):
    if len(record["LABELS"]) != len(record["CONTEXTS"]):
        raise ValueError(f"PubMed ID {pmid} has mismatched LABELS and CONTEXTS.")
    return {
        "ID": int(pmid),
        "contexts": [
            {"section": section, "text": text}
            for section, text in zip(record["LABELS"], record["CONTEXTS"])
        ],
        "source_split": source_split,
    }


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_qrels(path, questions, qrels_by_question):
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for question in sorted(questions, key=lambda item: item["id"]):
            output_file.write("\t".join(qrels_by_question[question["id"]]) + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_corpus(labeled_records, unlabeled_records):
    """Write normalized documents, question splits, qrels, and a manifest."""
    labeled_path = LABELED_DATA_PATH
    unlabeled_path = UNLABELED_DATA_PATH
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    QRELS_DIR.mkdir(parents=True, exist_ok=True)

    gold_ids = set(labeled_records)
    distractor_ids = sorted(set(unlabeled_records) - gold_ids)
    random_generator = random.Random(42)
    sampled_distractor_ids = random_generator.sample(distractor_ids, 10_000)
    selected_ids = gold_ids | set(sampled_distractor_ids)

    corpus_documents = []
    for pmid in selected_ids:
        sources = []
        if pmid in labeled_records:
            sources.append(("labeled", labeled_records[pmid]))
        if pmid in unlabeled_records:
            sources.append(("unlabeled", unlabeled_records[pmid]))
        normalized = [
            _normalize_document(pmid, record, name)
            for name, record in sources
        ]
        if (
            len(normalized) == 2
            and normalized[0]["contexts"] != normalized[1]["contexts"]
        ):
            raise ValueError(
                f"PubMed ID {pmid} has different CONTEXTS in both input files."
            )
        document = normalized[0]
        document["source_split"] = [name for name, _ in sources]
        corpus_documents.append(document)
    corpus_documents.sort(key=lambda document: document["ID"])
    _write_jsonl(DOCUMENTS_PATH, corpus_documents)

    questions = [
        {
            "id": f"q:{pmid}",
            "pubmed_id": pmid,
            "question": record["QUESTION"],
            "label": record["final_decision"],
            "long_answer": record["LONG_ANSWER"],
        }
        for pmid, record in labeled_records.items()
    ]
    questions.sort(key=lambda question: question["id"])
    random_generator.shuffle(questions)
    split_index = int(len(questions) * 0.8)
    question_splits = {"dev": questions[:split_index], "test": questions[split_index:]}
    qrels_by_question = {
        f"q:{pmid}": (f"q:{pmid}", "Q0", str(int(pmid)), "1")
        for pmid in labeled_records
    }

    _write_jsonl(QUESTIONS_DIR / "all.jsonl", questions)
    _write_qrels(QRELS_DIR / "all.tsv", questions, qrels_by_question)
    for split_name, split_questions in question_splits.items():
        split_questions = sorted(split_questions, key=lambda item: item["id"])
        _write_jsonl(QUESTIONS_DIR / f"{split_name}.jsonl", split_questions)
        _write_qrels(QRELS_DIR / f"{split_name}.tsv", split_questions, qrels_by_question)

    input_checksums = {
        "labeled": _sha256(labeled_path),
        "unlabeled": _sha256(unlabeled_path),
    }
    manifest = {
        "schema_version": "1.0",
        "inputs": {
            "labeled": {"path": str(labeled_path), "sha256": input_checksums["labeled"]},
            "unlabeled": {"path": str(unlabeled_path), "sha256": input_checksums["unlabeled"]},
        },
        "input_record_counts": {
            "labeled": len(labeled_records),
            "unlabeled": len(unlabeled_records),
        },
        "output_record_counts": {
            "documents": len(corpus_documents),
            "questions": len(questions),
            "qrels": len(qrels_by_question),
        },
        "duplicate_count": len(gold_ids & set(unlabeled_records)),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    gold_count = sum("labeled" in document["source_split"] for document in corpus_documents)
    print(
        f"Wrote {len(corpus_documents)} corpus documents ({gold_count} gold, "
        f"{len(corpus_documents) - gold_count} distractors) to {PROCESSED_DIR}"
    )
