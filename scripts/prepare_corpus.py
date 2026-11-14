import hashlib
import json
import random
import sys
from pathlib import Path
from urllib.request import urlretrieve


project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir / "src"))

from src.chunking.chunk_documents import chunk_documents
from src.chunking.chunk_sections import chunk_sections
from src.chunking.chunk_sentences import chunk_sentences


raw_dir = project_dir / "data" / "raw"
interim_dir = project_dir / "data" / "interim"
corpus_dir = project_dir / "data" / "corpus"


def load():
    raw_dir.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
        raw_dir / "ori_pqal.json",
    )
    urlretrieve(
        "https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/3b487bf64f698d2211163fb64cde6b102cd29bd2/data/ori_pqau.json",
        raw_dir / "ori_pqau.json",
    )
    print(f"Downloaded PubMedQA-L and PubMedQA-U to {raw_dir}")


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


def create_corpus():
    labeled_path = raw_dir / "ori_pqal.json"
    unlabeled_path = raw_dir / "ori_pqau.json"
    with labeled_path.open(encoding="utf-8") as input_file:
        labeled_records = json.load(input_file)
    with unlabeled_path.open(encoding="utf-8") as input_file:
        unlabeled_records = json.load(input_file)

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
        if len(normalized) == 2 and normalized[0]["contexts"] != normalized[1]["contexts"]:
            raise ValueError(f"PubMed ID {pmid} has different CONTEXTS in both input files.")
        document = normalized[0]
        document["source_split"] = [name for name, _ in sources]
        corpus_documents.append(document)
    corpus_documents.sort(key=lambda document: document["ID"])

    interim_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with (interim_dir / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as output_file:
        for document in corpus_documents:
            output_file.write(json.dumps(document, ensure_ascii=False) + "\n")

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

    with (interim_dir / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as output_file:
        for question in questions:
            output_file.write(json.dumps(question, ensure_ascii=False) + "\n")
    with (corpus_dir / "qrels.tsv").open("w", encoding="utf-8", newline="\n") as output_file:
        for question_id in sorted(qrels_by_question):
            output_file.write("\t".join(qrels_by_question[question_id]) + "\n")

    for split_name, split_questions in question_splits.items():
        with (corpus_dir / f"{split_name}_questions.jsonl").open("w", encoding="utf-8", newline="\n") as output_file:
            for question in sorted(split_questions, key=lambda item: item["id"]):
                output_file.write(json.dumps(question, ensure_ascii=False) + "\n")
        with (corpus_dir / f"{split_name}_qrels.tsv").open("w", encoding="utf-8", newline="\n") as output_file:
            for question in sorted(split_questions, key=lambda item: item["id"]):
                output_file.write("\t".join(qrels_by_question[question["id"]]) + "\n")

    input_checksums = {}
    for name, path in (("labeled", labeled_path), ("unlabeled", unlabeled_path)):
        sha256 = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                sha256.update(chunk)
        input_checksums[name] = sha256.hexdigest()
    manifest = {
        "schema_version": "1.0",
        "inputs": {
            "labeled": {"path": str(labeled_path), "sha256": input_checksums["labeled"]},
            "unlabeled": {"path": str(unlabeled_path), "sha256": input_checksums["unlabeled"]},
        },
        "input_record_counts": {"labeled": len(labeled_records), "unlabeled": len(unlabeled_records)},
        "output_record_counts": {
            "documents": len(corpus_documents),
            "questions": len(questions),
            "qrels": len(qrels_by_question),
        },
        "duplicate_count": len(gold_ids & set(unlabeled_records)),
    }
    with (interim_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    gold_count = sum("labeled" in document["source_split"] for document in corpus_documents)
    print(
        f"Wrote {len(corpus_documents)} corpus documents ({gold_count} gold, "
        f"{len(corpus_documents) - gold_count} distractors) to {interim_dir}"
    )


def main():
    load()
    create_corpus()
    chunk_documents()
    chunk_sections()
    chunk_sentences()


if __name__ == "__main__":
    main()
