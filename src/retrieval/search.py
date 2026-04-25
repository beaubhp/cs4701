from __future__ import annotations

import argparse
from pathlib import Path

from src.retrieval.bm25 import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Cornell policy chunks with local BM25.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--expand", action="store_true", help="Use deterministic query expansion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = BM25Index.from_jsonl(args.chunks)
    results = index.search(args.query, top_k=args.top_k, expand=args.expand)
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split())[:300]
        print(f"{rank}. {result.chunk_id} [{result.doc_id}] score={result.score:.4f}")
        print(f"   section: {result.section}")
        print(f"   {preview}")


if __name__ == "__main__":
    main()
