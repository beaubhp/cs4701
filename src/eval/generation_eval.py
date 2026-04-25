from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from src.corpus.schema import read_jsonl, write_json, write_jsonl
from src.eval.benchmark import load_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated policy QA answers heuristically.")
    parser.add_argument("generations", type=Path, help="Path to generations JSONL")
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-output", type=Path)
    return parser.parse_args()


def evaluate_generations(
    generations_path: Path,
    questions_path: Path,
    chunks_path: Path,
    output_path: Path,
    review_output_path: Path,
) -> dict[str, Any]:
    questions = {question["question_id"]: question for question in load_questions(questions_path)}
    chunks = {chunk["chunk_id"]: chunk for chunk in read_jsonl(chunks_path)}
    generations = read_jsonl(generations_path)

    rows = []
    review_rows = []
    for generation in generations:
        question = questions[generation["question_id"]]
        analyzed = analyze_generation(generation, question, chunks)
        rows.append(analyzed)
        review_rows.append(review_row(generation, question, analyzed))

    metrics = aggregate_metrics(rows)
    write_json(output_path, metrics)
    write_jsonl(review_output_path, review_rows)
    print_summary(metrics, output_path, review_output_path)
    return metrics


def analyze_generation(
    generation: dict[str, Any],
    question: dict[str, Any],
    chunks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    citations = generation.get("citations", [])
    retrieved_ids = {item["chunk_id"] for item in generation.get("retrieved_chunks", [])}
    gold_ids = set(question.get("gold_chunk_ids", []))
    qrel_relevant = {qrel["chunk_id"] for qrel in question.get("qrels", []) if qrel.get("relevance") == 3}

    cited_existing = [chunk_id for chunk_id in citations if chunk_id in chunks]
    cited_missing = [chunk_id for chunk_id in citations if chunk_id not in chunks]
    cited_not_retrieved = [chunk_id for chunk_id in citations if retrieved_ids and chunk_id not in retrieved_ids]

    return {
        "question_id": generation["question_id"],
        "system": generation["system"],
        "answerable": question["answerable"],
        "abstained": bool(generation["abstained"]),
        "answered": not bool(generation["abstained"]),
        "citation_count": len(citations),
        "citations_exist": len(cited_missing) == 0,
        "citations_in_retrieved": len(cited_not_retrieved) == 0,
        "has_gold_citation": bool(set(citations) & (gold_ids | qrel_relevant)),
        "missing_citations": cited_missing,
        "not_retrieved_citations": cited_not_retrieved,
        "answer_nonempty": bool(generation.get("answer", "").strip()),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    return {
        "num_generations": len(rows),
        "system": rows[0]["system"],
        "answer_rate": rate(rows, "answered"),
        "abstention_rate": rate(rows, "abstained"),
        "answerable": {
            "num_questions": len(answerable),
            "over_abstention_rate": rate(answerable, "abstained"),
            "answered_rate": rate(answerable, "answered"),
            "gold_citation_hit_rate": rate(answerable, "has_gold_citation"),
        },
        "unanswerable": {
            "num_questions": len(unanswerable),
            "correct_abstention_rate": rate(unanswerable, "abstained"),
            "false_answer_rate": rate(unanswerable, "answered"),
        },
        "citations": {
            "citation_validity_rate": rate(rows, "citations_exist"),
            "citation_in_retrieved_rate": rate(rows, "citations_in_retrieved"),
            "avg_citation_count": mean(row["citation_count"] for row in rows),
        },
    }


def rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return mean(1.0 if row[field] else 0.0 for row in rows)


def review_row(generation: dict[str, Any], question: dict[str, Any], analyzed: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": generation["question_id"],
        "system": generation["system"],
        "question": generation["question"],
        "answerable": question["answerable"],
        "gold_answer_short": question["gold_answer_short"],
        "gold_chunk_ids": question["gold_chunk_ids"],
        "model_answer": generation["answer"],
        "abstained": generation["abstained"],
        "abstention_reason": generation["abstention_reason"],
        "citations": generation["citations"],
        "retrieved_chunk_ids": [item["chunk_id"] for item in generation.get("retrieved_chunks", [])],
        "heuristics": analyzed,
        "manual_correct": None,
        "manual_citation_supported": None,
        "manual_hallucinated": None,
        "manual_notes": "",
    }


def print_summary(metrics: dict[str, Any], output_path: Path, review_output_path: Path) -> None:
    print(f"Wrote metrics to {output_path}")
    print(f"Wrote review template to {review_output_path}")
    print(
        "Summary:",
        f"system={metrics.get('system')}",
        f"answer_rate={metrics.get('answer_rate')}",
        f"abstention_rate={metrics.get('abstention_rate')}",
    )


def default_output_paths(generations_path: Path) -> tuple[Path, Path]:
    stem = generations_path.stem
    return (
        Path(f"data/results/{stem}_eval.json"),
        Path(f"data/results/{stem}_review_template.jsonl"),
    )


def main() -> None:
    args = parse_args()
    default_output, default_review = default_output_paths(args.generations)
    evaluate_generations(
        generations_path=args.generations,
        questions_path=args.questions,
        chunks_path=args.chunks,
        output_path=args.output or default_output,
        review_output_path=args.review_output or default_review,
    )


if __name__ == "__main__":
    main()
