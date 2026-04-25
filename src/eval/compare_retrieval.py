from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.corpus.schema import read_jsonl, write_json


DEFAULT_K_VALUES = [1, 3, 5, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two retrieval prediction files.")
    parser.add_argument("--left", type=Path, default=Path("data/results/retrieval_bm25_predictions.jsonl"))
    parser.add_argument("--right", type=Path, default=Path("data/results/retrieval_dense_predictions.jsonl"))
    parser.add_argument("--left-name", default="bm25")
    parser.add_argument("--right-name", default="dense")
    parser.add_argument("--output", type=Path, default=Path("data/results/retrieval_comparison.json"))
    return parser.parse_args()


def compare_predictions(
    left_path: Path,
    right_path: Path,
    left_name: str,
    right_name: str,
    output_path: Path,
) -> dict[str, Any]:
    left_rows = {row["question_id"]: row for row in read_jsonl(left_path)}
    right_rows = {row["question_id"]: row for row in read_jsonl(right_path)}
    if left_rows.keys() != right_rows.keys():
        missing_left = sorted(right_rows.keys() - left_rows.keys())
        missing_right = sorted(left_rows.keys() - right_rows.keys())
        raise ValueError(f"Prediction IDs differ. missing_left={missing_left}, missing_right={missing_right}")

    answerable_ids = [qid for qid, row in left_rows.items() if row["answerable"]]
    overall = {
        f"k={k}": summarize_at_k(answerable_ids, left_rows, right_rows, left_name, right_name, k)
        for k in DEFAULT_K_VALUES
    }

    grouped = {
        "topic": grouped_summaries(answerable_ids, left_rows, right_rows, left_name, right_name, "topic"),
        "question_type": grouped_summaries(answerable_ids, left_rows, right_rows, left_name, right_name, "question_type"),
        "difficulty": grouped_summaries(answerable_ids, left_rows, right_rows, left_name, right_name, "difficulty"),
    }

    details = per_question_details(answerable_ids, left_rows, right_rows, left_name, right_name)
    comparison = {
        "left": left_name,
        "right": right_name,
        "overall": overall,
        "grouped": grouped,
        "per_question": details,
    }
    write_json(output_path, comparison)
    return comparison


def summarize_at_k(
    question_ids: list[str],
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
    left_name: str,
    right_name: str,
    k: int,
) -> dict[str, Any]:
    buckets = {
        "both_hit": [],
        f"{left_name}_only": [],
        f"{right_name}_only": [],
        "both_miss": [],
    }
    for qid in question_ids:
        left_hit = hit_at_k(left_rows[qid], k)
        right_hit = hit_at_k(right_rows[qid], k)
        if left_hit and right_hit:
            buckets["both_hit"].append(qid)
        elif left_hit:
            buckets[f"{left_name}_only"].append(qid)
        elif right_hit:
            buckets[f"{right_name}_only"].append(qid)
        else:
            buckets["both_miss"].append(qid)
    return {
        "counts": {key: len(value) for key, value in buckets.items()},
        "question_ids": buckets,
    }


def grouped_summaries(
    question_ids: list[str],
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
    left_name: str,
    right_name: str,
    field: str,
) -> dict[str, Any]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for qid in question_ids:
        groups[left_rows[qid][field]].append(qid)
    return {
        group: {
            f"k={k}": summarize_at_k(ids, left_rows, right_rows, left_name, right_name, k)["counts"]
            for k in DEFAULT_K_VALUES
        }
        for group, ids in sorted(groups.items())
    }


def per_question_details(
    question_ids: list[str],
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
    left_name: str,
    right_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for qid in question_ids:
        left = left_rows[qid]
        right = right_rows[qid]
        rows.append(
            {
                "question_id": qid,
                "question": left["question"],
                "topic": left["topic"],
                "question_type": left["question_type"],
                "difficulty": left["difficulty"],
                f"{left_name}_rank": first_relevant_rank(left),
                f"{right_name}_rank": first_relevant_rank(right),
                f"{left_name}_top_chunk": left["retrieved"][0]["chunk_id"] if left["retrieved"] else None,
                f"{right_name}_top_chunk": right["retrieved"][0]["chunk_id"] if right["retrieved"] else None,
            }
        )
    return rows


def hit_at_k(row: dict[str, Any], k: int) -> bool:
    return any(item["relevance"] >= 3 for item in row["retrieved"][:k])


def first_relevant_rank(row: dict[str, Any]) -> int | None:
    for item in row["retrieved"]:
        if item["relevance"] >= 3:
            return item["rank"]
    return None


def main() -> None:
    args = parse_args()
    comparison = compare_predictions(
        left_path=args.left,
        right_path=args.right,
        left_name=args.left_name,
        right_name=args.right_name,
        output_path=args.output,
    )
    print(json.dumps(comparison["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
