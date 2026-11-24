"""Run a small retrieval baseline on the prepared PubMedQA dev/test split.

The baseline compares BM25 with two dense embedding models. It evaluates only
the top-k chunks selected for the context and does not call an answer LLM.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common import (
    CHUNK_PATHS,
    EMBEDDING_MODELS,
    EMBEDDING_MODEL_NICKNAMES,
    QRELS_DIR,
    QUESTIONS_DIR,
    allocate_run_dir,
    load_jsonl,
    load_qrels,
    write_csv,
)
from src.retrieval.search import search


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments" / "diagnostic"

# Document cutoffs define the small, medium, and large retrieval budgets. For
# section and sentence chunks, k is scaled by average whitespace-word count so
# the estimated amount of retrieved text stays roughly constant.
SIZE_NAMES = ("small", "medium", "large")
DOCUMENT_K_BY_SIZE = {"small": 1, "medium": 3, "large": 5}


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


def _run_configuration(questions, qrels, chunking, method, model, ks):
    max_k = max(ks.values())
    config = {
        "chunking": chunking,
        "method": method,
    }

    print(f"  Loading index/model and warming up {method} {model}...", flush=True)
    # Load model and index state before collecting query-time measurements.
    search(
        chunking,
        questions[0]["question"],
        scoring=method,
        method="top_k",
        top_k=max_k,
    )

    detail_rows = []
    for question in tqdm(questions, desc=f"{chunking} {method}", unit="question"):
        start = time.perf_counter()
        context_chunks = search(
            chunking,
            question["question"],
            scoring=method,
            method="top_k",
            top_k=max_k,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        detail = {
            **config,
            "question_id": question["id"],
            "retrieval_latency_ms": round(latency_ms, 3),
        }
        qrel = qrels[question["id"]]
        # The cutoffs are nested. Count each ranked chunk once and save the
        # cumulative word totals needed by the three cutoffs.
        word_totals = []
        word_total = 0
        for chunk, _ in context_chunks:
            word_total += len(chunk["text"].split())
            word_totals.append(word_total)
        size_metrics = {}
        for size in SIZE_NAMES:
            size_chunks = context_chunks[:ks[size]]
            word_count = (
                word_totals[min(ks[size], len(word_totals)) - 1]
                if word_totals
                else 0
            )
            size_metrics.update(
                {
                    f"k_{size}": ks[size],
                    f"words_{size}": word_count,
                    f"recall_{size}": round(_recall(size_chunks, qrel), 5),
                }
            )
        detail.update(size_metrics)
        detail_rows.append(detail)
    return detail_rows


def run(split):
    questions = load_jsonl(QUESTIONS_DIR / f"{split}.jsonl")
    qrels = load_qrels(QRELS_DIR / f"{split}.tsv")
    if not questions:
        raise ValueError(f"No questions found for split {split!r}.")

    average_chunk_lengths = _average_chunk_lengths()
    top_k_by_size = _top_k_by_size(average_chunk_lengths)
    run_number, output_dir = allocate_run_dir(EXPERIMENTS_DIR, split)

    configurations = []
    for chunking in CHUNK_PATHS:
        configurations.append((chunking, "bm25", ""))
        configurations.extend(
            (chunking, nickname, model)
            for nickname, model in EMBEDDING_MODELS.items()
        )

    details = []
    total = len(configurations)
    for index, (chunking, method, model) in enumerate(configurations, 1):
        print(
            f"[{index}/{total}] Starting {chunking} {method} {model} "
            f"k={top_k_by_size[chunking]}",
            flush=True,
        )
        per_question = _run_configuration(
            questions, qrels, chunking, method, model,
            top_k_by_size[chunking],
        )
        details.extend(per_question)
        print(
            f"[{index}/{total}] Completed {chunking} {method} {model}",
            flush=True,
        )

    write_csv(output_dir / "per_question.csv", details)
    df = pd.read_csv(output_dir / "per_question.csv", dtype={"question_id": str})
    summary = df.groupby(["chunking", "method"]).mean(numeric_only=True)
    summary = summary.rename(
        columns={
            "retrieval_latency_ms": "mean_retrieval_latency_ms",
            **{f"words_{size}": f"mean_words_{size}" for size in SIZE_NAMES},
        }
    ).reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)
    metadata = {
        "name": f"Diagnostic {split} run {run_number}",
        "experiment": "diagnostic",
        "split": split,
        "run_number": run_number,
        "status": "completed",
        "question_count": len(questions),
        "document_k_by_size": DOCUMENT_K_BY_SIZE,
        "top_k_by_size": top_k_by_size,
        "average_chunk_words": average_chunk_lengths,
        "embedding_models": EMBEDDING_MODEL_NICKNAMES,
        "configurations": len(configurations),
        "summary_csv": "summary.csv",
        "per_question_csv": "per_question.csv",
    }
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(configurations)} configurations to {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("dev", "test"))
    run(parser.parse_args().split)
