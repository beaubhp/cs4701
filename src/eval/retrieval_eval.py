from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from src.corpus.schema import write_json, write_jsonl
from src.eval.benchmark import load_questions, validate_benchmark
from src.retrieval.base import Retriever, SearchResult
from src.retrieval.bm25 import BM25Index
from src.retrieval.rerank import DEFAULT_RERANKER_MODEL


DEFAULT_K_VALUES = [1, 3, 5, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against benchmark qrels.")
    parser.add_argument("questions", type=Path, help="Path to data/benchmark/questions.jsonl")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--retriever", choices=["bm25", "dense", "bm25_rerank", "dense_rerank"], default="bm25")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--predictions",
        type=Path,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--expand", action="store_true", help="Use deterministic query expansion.")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    return parser.parse_args()


def evaluate(
    questions_path: Path,
    chunks_path: Path,
    output_path: Path,
    predictions_path: Path,
    top_k: int,
    expand: bool,
    retriever_name: str = "bm25",
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_cache: Path | None = None,
    rebuild_embeddings: bool = False,
    candidate_k: int = 20,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
) -> dict[str, Any]:
    validation = validate_benchmark(questions_path, chunks_path)
    if not validation.ok:
        for error in validation.errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    questions = load_questions(questions_path)
    retriever = build_retriever(
        retriever_name=retriever_name,
        chunks_path=chunks_path,
        expand=expand,
        dense_model=dense_model,
        embedding_cache=embedding_cache,
        rebuild_embeddings=rebuild_embeddings,
        candidate_k=candidate_k,
        reranker_model=reranker_model,
    )
    k_values = [k for k in DEFAULT_K_VALUES if k <= top_k]
    if top_k not in k_values:
        k_values.append(top_k)
    k_values = sorted(set(k_values))

    predictions: list[dict[str, Any]] = []
    answerable_rows: list[dict[str, Any]] = []
    unanswerable_rows: list[dict[str, Any]] = []

    for question in questions:
        results = retriever.search(question["question"], top_k=top_k)
        qrel_map = {qrel["chunk_id"]: qrel["relevance"] for qrel in question.get("qrels", [])}
        row = prediction_row(question, results, qrel_map)
        predictions.append(row)
        if question["answerable"]:
            answerable_rows.append(row)
        else:
            unanswerable_rows.append(row)

    metrics = {
        "benchmark": validation.summary,
        "retriever": retriever_metadata(retriever, top_k=top_k, k_values=k_values, expand=expand),
        "answerable": answerable_metrics(answerable_rows, k_values),
        "unanswerable": unanswerable_metrics(unanswerable_rows, k_values),
        "by_topic": grouped_answerable_metrics(answerable_rows, "topic", k_values),
        "by_question_type": grouped_answerable_metrics(answerable_rows, "question_type", k_values),
        "by_difficulty": grouped_answerable_metrics(answerable_rows, "difficulty", k_values),
    }

    write_json(output_path, metrics)
    write_jsonl(predictions_path, predictions)
    return metrics


def default_output_paths(retriever_name: str) -> tuple[Path, Path]:
    return (
        Path(f"data/results/retrieval_{retriever_name}.json"),
        Path(f"data/results/retrieval_{retriever_name}_predictions.jsonl"),
    )


def build_retriever(
    retriever_name: str,
    chunks_path: Path,
    expand: bool,
    dense_model: str,
    embedding_cache: Path | None,
    rebuild_embeddings: bool,
    candidate_k: int,
    reranker_model: str,
) -> Retriever:
    if retriever_name == "bm25":
        return BM25RetrieverWrapper(BM25Index.from_jsonl(chunks_path), expand=expand)
    if retriever_name == "dense":
        from src.retrieval.dense import DenseIndex

        return DenseIndex.from_jsonl(
            chunks_path,
            model_name=dense_model,
            cache_path=embedding_cache,
            rebuild_cache=rebuild_embeddings,
        )
    if retriever_name == "bm25_rerank":
        from src.retrieval.rerank import RerankingRetriever

        base_retriever = BM25RetrieverWrapper(BM25Index.from_jsonl(chunks_path), expand=expand)
        return RerankingRetriever(base_retriever, candidate_k=candidate_k, model_name=reranker_model)
    if retriever_name == "dense_rerank":
        from src.retrieval.dense import DenseIndex
        from src.retrieval.rerank import RerankingRetriever

        base_retriever = DenseIndex.from_jsonl(
            chunks_path,
            model_name=dense_model,
            cache_path=embedding_cache,
            rebuild_cache=rebuild_embeddings,
        )
        return RerankingRetriever(base_retriever, candidate_k=candidate_k, model_name=reranker_model)
    raise ValueError(f"Unsupported retriever: {retriever_name}")


class BM25RetrieverWrapper:
    def __init__(self, index: BM25Index, expand: bool = False) -> None:
        self.index = index
        self.expand = expand

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self.index.search(query, top_k=top_k, expand=self.expand)

    def describe(self) -> dict:
        metadata = self.index.describe()
        metadata["query_expansion"] = self.expand
        return metadata


def retriever_metadata(retriever: Retriever, top_k: int, k_values: list[int], expand: bool) -> dict[str, Any]:
    metadata = retriever.describe()
    metadata["top_k"] = top_k
    metadata["k_values"] = k_values
    metadata.setdefault("query_expansion", expand)
    return metadata


def prediction_row(
    question: dict[str, Any],
    results: list[SearchResult],
    qrel_map: dict[str, int],
) -> dict[str, Any]:
    near_miss_doc_ids = set(question.get("near_miss_doc_ids", []))
    retrieved = []
    for rank, result in enumerate(results, start=1):
        retrieved.append(
            {
                "rank": rank,
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "section": result.section,
                "score": result.score,
                "relevance": qrel_map.get(result.chunk_id, 0),
                "is_gold_doc": result.doc_id in set(question.get("gold_doc_ids", [])),
                "is_near_miss_doc": result.doc_id in near_miss_doc_ids,
                "text_preview": " ".join(result.text.split())[:500],
            }
        )
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "answerable": question["answerable"],
        "topic": question["topic"],
        "question_type": question["question_type"],
        "difficulty": question["difficulty"],
        "gold_chunk_ids": question["gold_chunk_ids"],
        "gold_doc_ids": question["gold_doc_ids"],
        "near_miss_doc_ids": question["near_miss_doc_ids"],
        "qrel_relevances": list(qrel_map.values()),
        "retrieved": retrieved,
    }


def answerable_metrics(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    if not rows:
        return {}

    metrics: dict[str, Any] = {"num_questions": len(rows), "mrr": mean(reciprocal_rank(row) for row in rows)}
    for k in k_values:
        metrics[f"recall@{k}"] = mean(has_relevant_chunk(row, k) for row in rows)
        metrics[f"doc_recall@{k}"] = mean(has_gold_doc(row, k) for row in rows)
        metrics[f"ndcg@{k}"] = mean(ndcg(row, k) for row in rows)
    return metrics


def unanswerable_metrics(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    if not rows:
        return {}

    top_scores = [row["retrieved"][0]["score"] for row in rows if row["retrieved"]]
    metrics: dict[str, Any] = {
        "num_questions": len(rows),
        "avg_top_score": mean(top_scores) if top_scores else 0.0,
        "max_top_score": max(top_scores) if top_scores else 0.0,
    }
    for k in k_values:
        metrics[f"near_miss_doc@{k}"] = mean(has_near_miss_doc(row, k) for row in rows)
    return metrics


def grouped_answerable_metrics(
    rows: list[dict[str, Any]],
    field: str,
    k_values: list[int],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[field], []).append(row)
    return {key: answerable_metrics(value, k_values) for key, value in sorted(grouped.items())}


def has_relevant_chunk(row: dict[str, Any], k: int) -> bool:
    return any(item["relevance"] >= 3 for item in row["retrieved"][:k])


def has_gold_doc(row: dict[str, Any], k: int) -> bool:
    gold_doc_ids = set(row["gold_doc_ids"])
    return any(item["doc_id"] in gold_doc_ids for item in row["retrieved"][:k])


def has_near_miss_doc(row: dict[str, Any], k: int) -> bool:
    near_miss_doc_ids = set(row["near_miss_doc_ids"])
    return any(item["doc_id"] in near_miss_doc_ids for item in row["retrieved"][:k])


def reciprocal_rank(row: dict[str, Any]) -> float:
    for item in row["retrieved"]:
        if item["relevance"] >= 3:
            return 1.0 / item["rank"]
    return 0.0


def ndcg(row: dict[str, Any], k: int) -> float:
    gains = [item["relevance"] for item in row["retrieved"][:k]]
    dcg = discounted_cumulative_gain(gains)
    ideal_gains = sorted([qrel_relevance for qrel_relevance in qrel_relevances(row)], reverse=True)[:k]
    ideal = discounted_cumulative_gain(ideal_gains)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def qrel_relevances(row: dict[str, Any]) -> list[int]:
    return [relevance for relevance in row.get("qrel_relevances", []) if relevance > 0]


def discounted_cumulative_gain(gains: list[int]) -> float:
    return sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def main() -> None:
    args = parse_args()
    default_output, default_predictions = default_output_paths(args.retriever)
    metrics = evaluate(
        questions_path=args.questions,
        chunks_path=args.chunks,
        output_path=args.output or default_output,
        predictions_path=args.predictions or default_predictions,
        top_k=args.top_k,
        expand=args.expand,
        retriever_name=args.retriever,
        dense_model=args.dense_model,
        embedding_cache=args.embedding_cache,
        rebuild_embeddings=args.rebuild_embeddings,
        candidate_k=args.candidate_k,
        reranker_model=args.reranker_model,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
