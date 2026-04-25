from __future__ import annotations

import argparse
from pathlib import Path

from src.retrieval.dense import DEFAULT_MODEL, DenseIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Cornell policy chunks with dense embeddings.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = DenseIndex.from_jsonl(
        args.chunks,
        model_name=args.model,
        cache_path=args.cache,
        rebuild_cache=args.rebuild_cache,
    )
    results = index.search(args.query, top_k=args.top_k)
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split())[:300]
        print(f"{rank}. {result.chunk_id} [{result.doc_id}] score={result.score:.4f}")
        print(f"   section: {result.section}")
        print(f"   {preview}")


if __name__ == "__main__":
    main()
