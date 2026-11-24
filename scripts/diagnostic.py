"""Run a small retrieval baseline on the prepared PubMedQA dev/test split.

The baseline compares BM25 with two dense embedding models. It evaluates only
the top-k chunks selected for the context and does not call an answer LLM.
"""

import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (  # noqa: E402
    CHUNK_PATHS,
    QRELS_DIR,
    QUESTIONS_DIR,
    allocate_run_dir,
    load_jsonl,
    load_qrels,
    mean,
    write_csv,
)
from src.scoring.bm25 import bm25_score  # noqa: E402
from src.scoring.dense import dense_score  # noqa: E402


# Run settings. Edit these values here when a different diagnostic run is needed.
SPLIT = "dev"
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments" / "diagnostic"

# Keep the report nickname beside each full model ID used to load it.
EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "miniLM",
    "NeuML/pubmedbert-base-embeddings": "Bert",
}

# Document cutoffs define the small, medium, and large retrieval budgets. For
# section and sentence chunks, k is scaled by average whitespace-word count so
# the estimated amount of retrieved text stays roughly constant.
SIZE_NAMES = ("small", "medium", "large")
DOCUMENT_K_BY_SIZE = {"small": 1, "medium": 3, "large": 5}


def _load_tokenizer(model_name):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Token counts need transformers. Install the project's model dependencies."
        ) from error

    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Tokenizer {model_name!r} is not available in the local model cache. "
            "Set TOKENIZER_MODEL to a locally cached tokenizer."
        ) from error


def _context_token_count(chunks, tokenizer):
    return sum(
        len(tokenizer.encode(chunk["text"], add_special_tokens=False))
        for chunk, _ in chunks
    )


def _average_chunk_lengths():
    averages = {}
    for chunking, path in CHUNK_PATHS.items():
        chunks = load_jsonl(path)
        averages[chunking] = (
            sum(len(chunk["text"].split()) for chunk in chunks) / len(chunks)
            if chunks
            else 1.0
        )
    return averages


def _top_k_by_size(average_chunk_lengths):
    document_average = average_chunk_lengths["document"]
    return {
        chunking: {
            size: max(
                1,
                round(
                    document_k
                    * document_average
                    / average_chunk_lengths[chunking]
                ),
            )
            for size, document_k in DOCUMENT_K_BY_SIZE.items()
        }
        for chunking in CHUNK_PATHS
    }


def _recall(context_chunks, qrel):
    gold_ids = {str(doc_id) for doc_id, grade in qrel.items() if grade > 0}
    if not gold_ids:
        return 0.0
    retrieved_ids = {str(chunk["document_id"]) for chunk, _ in context_chunks}
    return len(gold_ids & retrieved_ids) / len(gold_ids)


def _score(method, question, chunking, model, k):
    if method == "bm25":
        return bm25_score(chunking, question, top_k=k)
    else:
        return dense_score(chunking, question, model, top_k=k)


def _run_configuration(questions, qrels, tokenizer, chunking, method, model, ks):
    max_k = max(ks.values())
    config = {
        "chunking": chunking,
        "method": method,
    }

    print(f"  Loading index/model and warming up {method} {model}...", flush=True)
    # Load model and index state before collecting query-time measurements.
    _score(method, questions[0]["question"], chunking, model, max_k)

    detail_rows = []
    latencies = []
    token_counts = {size: [] for size in SIZE_NAMES}
    recall_values = {size: [] for size in SIZE_NAMES}
    progress_interval = max(1, min(50, len(questions) // 10 or 1))
    for question_index, question in enumerate(questions, 1):
        start = time.perf_counter()
        context_chunks = _score(method, question["question"], chunking, model, max_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        detail = {
            **config,
            "question_id": question["id"],
            "retrieval_latency_ms": round(latency_ms, 3),
        }
        qrel = qrels[question["id"]]
        for size in SIZE_NAMES:
            size_chunks = context_chunks[:ks[size]]
            token_count = _context_token_count(size_chunks, tokenizer)
            recall = _recall(size_chunks, qrel)
            token_counts[size].append(token_count)
            recall_values[size].append(recall)
            detail[f"k_{size}"] = ks[size]
            detail[f"tokens_{size}"] = token_count
            detail[f"recall_{size}"] = round(recall, 5)
        detail_rows.append(detail)
        if question_index % progress_interval == 0 or question_index == len(questions):
            print(
                f"  {chunking} {method} {model}: "
                f"{question_index}/{len(questions)} questions",
                flush=True,
            )

    summary_row = {
        **config,
        "mean_retrieval_latency_ms": round(mean(latencies, default=0.0), 3),
    }
    for size in SIZE_NAMES:
        summary_row[f"mean_tokens_{size}"] = round(
            mean(token_counts[size], default=0.0), 2
        )
    for size in SIZE_NAMES:
        summary_row[f"recall_{size}"] = round(
            mean(recall_values[size], default=0.0), 5
        )
    return summary_row, detail_rows


def run():
    questions = load_jsonl(QUESTIONS_DIR / f"{SPLIT}.jsonl")
    qrels = load_qrels(QRELS_DIR / f"{SPLIT}.tsv")
    questions = [question for question in questions if question["id"] in qrels]
    if not questions:
        raise ValueError(f"No questions found for split {SPLIT!r}.")

    tokenizer = _load_tokenizer(TOKENIZER_MODEL)
    average_chunk_lengths = _average_chunk_lengths()
    top_k_by_size = _top_k_by_size(average_chunk_lengths)
    run_number, output_dir = allocate_run_dir(EXPERIMENTS_DIR, SPLIT)

    configurations = []
    for chunking in CHUNK_PATHS:
        configurations.append((chunking, "bm25", ""))
        configurations.extend((chunking, nickname, model) for model, nickname in EMBEDDING_MODELS.items())

    summaries = []
    details = []
    total = len(configurations)
    for index, (chunking, method, model) in enumerate(configurations, 1):
        print(
            f"[{index}/{total}] Starting {chunking} {method} {model} "
            f"k={top_k_by_size[chunking]}",
            flush=True,
        )
        summary, per_question = _run_configuration(
            questions, qrels, tokenizer, chunking, method, model,
            top_k_by_size[chunking],
        )
        summaries.append(summary)
        details.extend(per_question)
        print(
            f"[{index}/{total}] Completed {chunking} {method} {model}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "per_question.csv", details)
    metadata = {
        "name": f"Diagnostic {SPLIT} run {run_number}",
        "experiment": "diagnostic",
        "split": SPLIT,
        "run_number": run_number,
        "status": "completed",
        "question_count": len(questions),
        "document_k_by_size": DOCUMENT_K_BY_SIZE,
        "top_k_by_size": top_k_by_size,
        "average_chunk_words": average_chunk_lengths,
        "k_scaling_note": "Section and sentence k values scale document cutoffs by the corpus average whitespace-word count per chunk; observed context tokens are measured with the configured tokenizer.",
        "embedding_models": EMBEDDING_MODELS,
        "tokenizer_model": tokenizer.name_or_path,
        "token_count_note": "Sum of selected chunk text tokens; excludes prompts and special tokens.",
        "latency_note": "Warm per-question retrieval and top-k selection; model loading is excluded. Token counting is excluded.",
        "configurations": len(configurations),
        "summary_csv": "summary.csv",
        "per_question_csv": "per_question.csv",
    }
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(summaries)} configurations to {output_dir.resolve()}")


if __name__ == "__main__":
    run()
