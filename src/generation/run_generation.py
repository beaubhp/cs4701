from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.corpus.schema import read_jsonl, write_jsonl
from src.eval.benchmark import load_questions, validate_benchmark
from src.generation.iterkey import ITERKEY_PROMPT_VERSION, run_iterkey
from src.generation.llm import DEFAULT_MODEL, DEFAULT_TEMPERATURE, OpenAIGenerator
from src.generation.prompts import (
    NON_RAG_PROMPT_VERSION,
    RAG_PROMPT_VERSION,
    build_non_rag_prompt,
    build_rag_prompt,
)
from src.retrieval.base import SearchResult
from src.retrieval.bm25 import BM25Index
from src.retrieval.rerank import DEFAULT_RERANKER_MODEL


SYSTEMS = {"non_rag", "bm25_rag", "dense_rag", "bm25_rerank_rag", "dense_rerank_rag", "iterkey_rag"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate policy QA answers with OpenAI.")
    parser.add_argument("questions", type=Path, help="Path to data/benchmark/questions.jsonl")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id", action="append", dest="question_ids")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and output placeholder rows without API calls.")
    parser.add_argument("--include-raw-response", action="store_true", help="Store full raw OpenAI response payloads.")
    return parser.parse_args()


def run_generation(
    questions_path: Path,
    chunks_path: Path,
    system: str,
    output_path: Path,
    top_k: int,
    candidate_k: int,
    reranker_model: str,
    max_iterations: int,
    model: str,
    temperature: float,
    limit: int | None = None,
    question_ids: list[str] | None = None,
    dry_run: bool = False,
    include_raw_response: bool = False,
) -> None:
    validation = validate_benchmark(questions_path, chunks_path)
    if not validation.ok:
        for error in validation.errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    questions = select_questions(load_questions(questions_path), limit=limit, question_ids=question_ids)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in read_jsonl(chunks_path)}
    retriever = build_retriever(system, chunks_path, candidate_k=candidate_k, reranker_model=reranker_model)
    generator = None if dry_run else build_generator(model, temperature)

    rows = []
    for question in questions:
        iterkey_trace = None
        if system == "iterkey_rag" and not dry_run:
            assert generator is not None
            assert isinstance(retriever, BM25Index)
            parsed, retrieved_results, iterkey_trace = run_iterkey(
                question=question,
                retriever=retriever,
                chunks_by_id=chunks_by_id,
                generator=generator,
                top_k=top_k,
                max_iterations=max_iterations,
            )
            prompt_version = ITERKEY_PROMPT_VERSION
            raw_response = None
            request_id = None
            usage = None
        elif system == "non_rag":
            retrieved_results = []
            instructions, user_input = build_non_rag_prompt(question)
            prompt_version = NON_RAG_PROMPT_VERSION
            if dry_run:
                parsed = dry_run_response()
                raw_response = None
                request_id = None
                usage = None
            else:
                assert generator is not None
                result = generator.generate_structured(instructions, user_input)
                parsed = result.parsed
                raw_response = result.raw_response
                request_id = result.request_id
                usage = result.usage
        else:
            retrieved_results = [] if retriever is None else retriever.search(question["question"], top_k=top_k)
            retrieved_chunks = [chunks_by_id[result.chunk_id] for result in retrieved_results]
            instructions, user_input = build_rag_prompt(question, retrieved_chunks)
            prompt_version = ITERKEY_PROMPT_VERSION if system == "iterkey_rag" else RAG_PROMPT_VERSION
            if dry_run:
                parsed = dry_run_response()
                raw_response = None
                request_id = None
                usage = None
            else:
                assert generator is not None
                result = generator.generate_structured(instructions, user_input)
                parsed = result.parsed
                raw_response = result.raw_response
                request_id = result.request_id
                usage = result.usage

        rows.append(
            generation_row(
                question=question,
                system=system,
                parsed=parsed,
                retrieved_results=retrieved_results,
                model=model,
                temperature=temperature,
                prompt_version=prompt_version,
                raw_response=raw_response,
                request_id=request_id,
                usage=usage,
                dry_run=dry_run,
                include_raw_response=include_raw_response,
                iterkey_trace=iterkey_trace,
            )
        )

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} generations to {output_path}")


def select_questions(
    questions: list[dict[str, Any]],
    limit: int | None,
    question_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if question_ids:
        wanted = set(question_ids)
        selected = [question for question in questions if question["question_id"] in wanted]
        missing = wanted - {question["question_id"] for question in selected}
        if missing:
            raise ValueError(f"Unknown question_id values: {', '.join(sorted(missing))}")
        return selected
    if limit is not None:
        return questions[:limit]
    return questions


def dry_run_response() -> dict[str, Any]:
    return {
        "answer": "",
        "abstained": True,
        "abstention_reason": "dry_run",
        "citations": [],
    }


def build_retriever(system: str, chunks_path: Path, candidate_k: int, reranker_model: str):
    if system == "non_rag":
        return None
    if system == "iterkey_rag":
        return BM25Index.from_jsonl(chunks_path)
    if system == "bm25_rag":
        return BM25Index.from_jsonl(chunks_path)
    if system == "dense_rag":
        from src.retrieval.dense import DenseIndex

        return DenseIndex.from_jsonl(chunks_path)
    if system == "bm25_rerank_rag":
        from src.retrieval.rerank import RerankingRetriever

        return RerankingRetriever(BM25Index.from_jsonl(chunks_path), candidate_k=candidate_k, model_name=reranker_model)
    if system == "dense_rerank_rag":
        from src.retrieval.dense import DenseIndex
        from src.retrieval.rerank import RerankingRetriever

        return RerankingRetriever(DenseIndex.from_jsonl(chunks_path), candidate_k=candidate_k, model_name=reranker_model)
    raise ValueError(f"Unsupported generation system: {system}")


def build_generator(model: str, temperature: float) -> OpenAIGenerator:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or export it in your shell.")
    return OpenAIGenerator(model=model, temperature=temperature)


def generation_row(
    question: dict[str, Any],
    system: str,
    parsed: dict[str, Any],
    retrieved_results: list[SearchResult],
    model: str,
    temperature: float,
    prompt_version: str,
    raw_response: dict[str, Any] | None,
    request_id: str | None,
    usage: dict[str, Any] | None,
    dry_run: bool,
    include_raw_response: bool,
    iterkey_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "system": system,
        "question": question["question"],
        "answer": parsed["answer"],
        "abstained": parsed["abstained"],
        "abstention_reason": parsed["abstention_reason"],
        "citations": parsed["citations"],
        "retrieved_chunks": [retrieved_chunk_row(result) for result in retrieved_results],
        "model": model,
        "temperature": temperature,
        "prompt_version": prompt_version,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "request_id": request_id,
        "usage": usage,
        "dry_run": dry_run,
        "raw_response": raw_response if include_raw_response else None,
        "iterkey_trace": iterkey_trace,
    }


def retrieved_chunk_row(result: SearchResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "section": result.section,
        "score": result.score,
        "title": result.title,
        "chunk_index": result.chunk_index,
        "text_preview": " ".join(result.text.split())[:500],
    }


def default_output_path(system: str) -> Path:
    return Path(f"data/results/generations_{system}.jsonl")


def main() -> None:
    args = parse_args()
    run_generation(
        questions_path=args.questions,
        chunks_path=args.chunks,
        system=args.system,
        output_path=args.output or default_output_path(args.system),
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        reranker_model=args.reranker_model,
        max_iterations=args.max_iterations,
        model=args.model,
        temperature=args.temperature,
        limit=args.limit,
        question_ids=args.question_ids,
        dry_run=args.dry_run,
        include_raw_response=args.include_raw_response,
    )


if __name__ == "__main__":
    main()
