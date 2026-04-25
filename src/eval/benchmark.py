from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.corpus.schema import read_jsonl


ANSWERABLE_REQUIRED_FIELDS = {
    "question_id",
    "question",
    "answerable",
    "topic",
    "question_type",
    "difficulty",
    "requires_multi_chunk",
    "requires_multi_doc",
    "gold_doc_ids",
    "gold_chunk_ids",
    "gold_answer_short",
    "evidence",
    "qrels",
    "unanswerable_type",
    "negative_rationale",
    "near_miss_doc_ids",
    "notes",
}

VALID_RELEVANCE = {0, 1, 2, 3}


@dataclass(frozen=True)
class BenchmarkValidationResult:
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_chunks(path: Path) -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: chunk for chunk in read_jsonl(path)}


def load_questions(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def validate_benchmark(questions_path: Path, chunks_path: Path) -> BenchmarkValidationResult:
    questions = load_questions(questions_path)
    chunks = load_chunks(chunks_path)
    chunk_doc_ids = {chunk["doc_id"] for chunk in chunks.values()}

    errors: list[str] = []
    warnings: list[str] = []
    question_ids: Counter[str] = Counter()
    answerability_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()

    for index, question in enumerate(questions, start=1):
        qid = question.get("question_id", f"<line {index}>")
        question_ids[qid] += 1

        missing = sorted(field for field in ANSWERABLE_REQUIRED_FIELDS if field not in question)
        if missing:
            errors.append(f"{qid}: missing required fields: {', '.join(missing)}")
            continue

        answerable = question["answerable"]
        answerability_counts["answerable" if answerable else "unanswerable"] += 1
        topic_counts[question["topic"]] += 1
        type_counts[question["question_type"]] += 1
        difficulty_counts[question["difficulty"]] += 1

        gold_doc_ids = question["gold_doc_ids"]
        gold_chunk_ids = question["gold_chunk_ids"]
        evidence = question["evidence"]
        qrels = question["qrels"]

        if not isinstance(answerable, bool):
            errors.append(f"{qid}: answerable must be a boolean")

        for doc_id in gold_doc_ids:
            doc_counts[doc_id] += 1
            if doc_id not in chunk_doc_ids:
                errors.append(f"{qid}: unknown gold_doc_id {doc_id}")

        for chunk_id in gold_chunk_ids:
            if chunk_id not in chunks:
                errors.append(f"{qid}: unknown gold_chunk_id {chunk_id}")
            elif chunks[chunk_id]["doc_id"] not in gold_doc_ids:
                errors.append(f"{qid}: gold_chunk_id {chunk_id} doc_id is not in gold_doc_ids")

        validate_evidence(qid, evidence, chunks, errors)
        validate_qrels(qid, qrels, chunks, errors)

        if answerable:
            if not gold_doc_ids:
                errors.append(f"{qid}: answerable question has no gold_doc_ids")
            if not gold_chunk_ids:
                errors.append(f"{qid}: answerable question has no gold_chunk_ids")
            if not question["gold_answer_short"]:
                errors.append(f"{qid}: answerable question has no gold_answer_short")
            if not any(qrel.get("relevance") == 3 for qrel in qrels):
                errors.append(f"{qid}: answerable question needs at least one relevance=3 qrel")
            if question["unanswerable_type"] is not None:
                errors.append(f"{qid}: answerable question should have null unanswerable_type")
            if question["negative_rationale"] is not None:
                errors.append(f"{qid}: answerable question should have null negative_rationale")
        else:
            if gold_doc_ids or gold_chunk_ids:
                errors.append(f"{qid}: unanswerable question should not have gold ids")
            if question["gold_answer_short"] is not None:
                errors.append(f"{qid}: unanswerable question should have null gold_answer_short")
            if evidence:
                errors.append(f"{qid}: unanswerable question should not have evidence")
            if any(qrel.get("relevance") == 3 for qrel in qrels):
                errors.append(f"{qid}: unanswerable question cannot have relevance=3 qrel")
            if not question["unanswerable_type"]:
                errors.append(f"{qid}: unanswerable question needs unanswerable_type")
            if not question["negative_rationale"]:
                errors.append(f"{qid}: unanswerable question needs negative_rationale")
            if not question["near_miss_doc_ids"]:
                warnings.append(f"{qid}: unanswerable question has no near_miss_doc_ids")

    duplicate_ids = [qid for qid, count in question_ids.items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate question_id values: {', '.join(sorted(duplicate_ids))}")

    duplicate_questions = find_duplicate_questions(questions)
    for qid_a, qid_b in duplicate_questions:
        warnings.append(f"near-duplicate normalized question text: {qid_a} and {qid_b}")

    summary = {
        "num_questions": len(questions),
        "answerability": dict(answerability_counts),
        "topics": dict(topic_counts),
        "gold_docs": dict(doc_counts),
        "question_types": dict(type_counts),
        "difficulties": dict(difficulty_counts),
    }
    return BenchmarkValidationResult(errors=errors, warnings=warnings, summary=summary)


def validate_evidence(
    question_id: str,
    evidence_items: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    for idx, evidence in enumerate(evidence_items):
        prefix = f"{question_id}: evidence[{idx}]"
        chunk_id = evidence.get("chunk_id")
        doc_id = evidence.get("doc_id")
        quote = evidence.get("quote")
        if chunk_id not in chunks:
            errors.append(f"{prefix}: unknown chunk_id {chunk_id}")
            continue
        chunk = chunks[chunk_id]
        if doc_id != chunk["doc_id"]:
            errors.append(f"{prefix}: doc_id {doc_id} does not match chunk {chunk_id}")
        if not quote:
            errors.append(f"{prefix}: missing quote")
        elif quote not in chunk["text"]:
            errors.append(f"{prefix}: quote not found in chunk text")


def validate_qrels(
    question_id: str,
    qrels: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    for idx, qrel in enumerate(qrels):
        prefix = f"{question_id}: qrels[{idx}]"
        chunk_id = qrel.get("chunk_id")
        doc_id = qrel.get("doc_id")
        relevance = qrel.get("relevance")
        if chunk_id not in chunks:
            errors.append(f"{prefix}: unknown chunk_id {chunk_id}")
            continue
        chunk = chunks[chunk_id]
        if doc_id != chunk["doc_id"]:
            errors.append(f"{prefix}: doc_id {doc_id} does not match chunk {chunk_id}")
        if relevance not in VALID_RELEVANCE:
            errors.append(f"{prefix}: relevance must be one of {sorted(VALID_RELEVANCE)}")


def normalize_question_text(text: str) -> str:
    return " ".join("".join(char.lower() if char.isalnum() else " " for char in text).split())


def find_duplicate_questions(questions: list[dict[str, Any]]) -> list[tuple[str, str]]:
    seen: defaultdict[str, list[str]] = defaultdict(list)
    duplicates: list[tuple[str, str]] = []
    for question in questions:
        normalized = normalize_question_text(question.get("question", ""))
        for previous in seen[normalized]:
            duplicates.append((previous, question.get("question_id", "<unknown>")))
        seen[normalized].append(question.get("question_id", "<unknown>"))
    return duplicates
