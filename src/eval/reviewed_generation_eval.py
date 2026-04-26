from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.corpus.schema import read_jsonl

DEFAULT_TEMPLATE_PATTERN = "generations_*_review_template.jsonl"
DEFAULT_RESULTS_DIR = Path("data/results")
DEFAULT_COMBINED_OUTPUT = DEFAULT_RESULTS_DIR / "reviewed_generation_metrics.json"
PROBLEM_SEVERITY = "major"
VALID_SEVERITIES = {"none", "minor", "major"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate filled generation review templates.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing generations_*_review_template.jsonl files.",
    )
    parser.add_argument(
        "--template",
        action="append",
        type=Path,
        dest="templates",
        help="Specific review template to aggregate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=DEFAULT_COMBINED_OUTPUT,
        help="Path for the all-systems reviewed metrics JSON.",
    )
    return parser.parse_args()


def aggregate_templates(
    templates: list[Path],
    combined_output: Path,
) -> dict[str, dict[str, Any]]:
    if not templates:
        raise ValueError("No review template files found.")

    combined: dict[str, dict[str, Any]] = {}
    for template_path in templates:
        rows = read_jsonl(template_path)
        metrics = aggregate_review_rows(rows)
        system = metrics["system"]
        output_path = reviewed_output_path(template_path)
        write_review_json(output_path, metrics)
        combined[system] = metrics
        print(f"Wrote {output_path}")

    write_review_json(combined_output, combined)
    print(f"Wrote {combined_output}")
    return combined


def aggregate_review_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty review template.")

    system = rows[0]["system"]
    if any(row.get("system") != system for row in rows):
        raise ValueError(f"Review template mixes systems; first system is {system}.")

    validate_review_rows(rows)

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    citation_rows = [row for row in rows if row.get("manual_citation_supported") is not None]

    return {
        "system": system,
        "num_reviewed": len(rows),
        "reviewed_correct_rate": bool_rate(rows, "manual_correct"),
        "reviewed_hallucination_rate": bool_rate(rows, "manual_hallucinated"),
        "major_hallucination_rate": severity_rate(rows, PROBLEM_SEVERITY),
        "answerable": {
            "num_questions": len(answerable),
            "reviewed_accuracy": bool_rate(answerable, "manual_correct"),
            "reviewed_hallucination_rate": bool_rate(answerable, "manual_hallucinated"),
            "reviewed_over_abstention_rate": abstention_rate(answerable),
        },
        "unanswerable": {
            "num_questions": len(unanswerable),
            "reviewed_correct_refusal_rate": bool_rate(unanswerable, "review_abstention_correct"),
            "reviewed_false_answer_rate": false_answer_rate(unanswerable),
            "reviewed_hallucination_rate": bool_rate(unanswerable, "manual_hallucinated"),
        },
        "citations": {
            "num_applicable_rows": len(citation_rows),
            "reviewed_citation_support_rate": bool_rate(citation_rows, "manual_citation_supported"),
        },
        "flagged_rows": flagged_rows(rows),
    }


def validate_review_rows(rows: list[dict[str, Any]]) -> None:
    missing = []
    invalid = []
    for row in rows:
        question_id = row.get("question_id", "<missing question_id>")
        for field in ("manual_correct", "manual_hallucinated", "review_hallucination_severity"):
            if row.get(field) is None:
                missing.append(f"{question_id}:{field}")
        if not row.get("answerable") and row.get("review_abstention_correct") is None:
            missing.append(f"{question_id}:review_abstention_correct")
        if row.get("system") != "non_rag" and row.get("manual_citation_supported") is None:
            missing.append(f"{question_id}:manual_citation_supported")
        severity = row.get("review_hallucination_severity")
        if severity is not None and severity not in VALID_SEVERITIES:
            invalid.append(f"{question_id}:review_hallucination_severity={severity}")
        if row.get("manual_hallucinated") is True and severity == "none":
            invalid.append(f"{question_id}:hallucinated_true_with_none_severity")
        if row.get("manual_hallucinated") is False and severity in {"minor", "major"}:
            invalid.append(f"{question_id}:hallucinated_false_with_{severity}_severity")
    if missing:
        raise ValueError("Missing reviewed labels: " + ", ".join(missing))
    if invalid:
        raise ValueError("Invalid reviewed labels: " + ", ".join(invalid))


def bool_rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return mean(1.0 if row.get(field) else 0.0 for row in rows)


def severity_rate(rows: list[dict[str, Any]], severity: str) -> float:
    if not rows:
        return 0.0
    return mean(1.0 if row.get("review_hallucination_severity") == severity else 0.0 for row in rows)


def abstention_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return mean(1.0 if row.get("abstained") else 0.0 for row in rows)


def false_answer_rate(unanswerable_rows: list[dict[str, Any]]) -> float:
    if not unanswerable_rows:
        return 0.0
    return mean(0.0 if row.get("review_abstention_correct") else 1.0 for row in unanswerable_rows)


def flagged_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flagged = []
    for row in rows:
        if row_has_review_issue(row):
            flagged.append(
                {
                    "question_id": row["question_id"],
                    "manual_correct": row["manual_correct"],
                    "manual_citation_supported": row.get("manual_citation_supported"),
                    "manual_hallucinated": row["manual_hallucinated"],
                    "manual_notes": row.get("manual_notes", ""),
                }
            )
    return flagged


def row_has_review_issue(row: dict[str, Any]) -> bool:
    return (
        row.get("manual_correct") is False
        or row.get("manual_citation_supported") is False
        or row.get("manual_hallucinated") is True
        or row.get("review_abstention_correct") is False
    )


def reviewed_output_path(template_path: Path) -> Path:
    suffix = "_review_template"
    if not template_path.stem.endswith(suffix):
        raise ValueError(f"Unexpected review template name: {template_path}")
    return template_path.with_name(f"{template_path.stem.removesuffix(suffix)}_reviewed_eval.json")


def write_review_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2, sort_keys=True)
        file.write("\n")


def discover_templates(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob(DEFAULT_TEMPLATE_PATTERN))


def main() -> None:
    args = parse_args()
    templates = args.templates or discover_templates(args.results_dir)
    aggregate_templates(templates=templates, combined_output=args.combined_output)


if __name__ == "__main__":
    main()
