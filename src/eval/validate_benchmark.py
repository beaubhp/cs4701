from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.benchmark import validate_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the benchmark JSONL against fixed corpus chunks.")
    parser.add_argument("questions", type=Path, help="Path to data/benchmark/questions.jsonl")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_benchmark(args.questions, args.chunks)
    print(json.dumps(result.summary, indent=2, sort_keys=True))

    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")

    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
