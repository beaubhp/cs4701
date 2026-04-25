from __future__ import annotations

from typing import Any

from src.generation.llm import OpenAIGenerator
from src.generation.prompts import build_rag_prompt
from src.retrieval.base import SearchResult
from src.retrieval.bm25 import BM25Index


ITERKEY_PROMPT_VERSION = "iterkey_v1"

KEYWORDS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["keywords"],
}

VALIDATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_supported": {"type": "boolean"},
    },
    "required": ["is_supported"],
}


def run_iterkey(
    question: dict[str, Any],
    retriever: BM25Index,
    chunks_by_id: dict[str, dict[str, Any]],
    generator: OpenAIGenerator,
    top_k: int,
    max_iterations: int,
) -> tuple[dict[str, Any], list[SearchResult], list[dict[str, Any]]]:
    previous_keywords: list[str] = []
    last_answer: dict[str, Any] | None = None
    last_results: list[SearchResult] = []
    trace: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        keywords = (
            generate_keywords(generator, question)
            if iteration == 1
            else refine_keywords(generator, question, previous_keywords)
        )
        previous_keywords = keywords
        expanded_query = expanded_query_text(question["question"], keywords)
        results = retriever.search(expanded_query, top_k=top_k)
        retrieved_chunks = [chunks_by_id[result.chunk_id] for result in results]
        instructions, user_input = build_rag_prompt(question, retrieved_chunks)
        answer_result = generator.generate_structured(instructions, user_input)
        answer = answer_result.parsed
        is_supported = validate_answer(generator, question, answer, retrieved_chunks)

        trace.append(
            {
                "iteration": iteration,
                "keywords": keywords,
                "expanded_query": expanded_query,
                "retrieved_chunk_ids": [result.chunk_id for result in results],
                "answer": answer["answer"],
                "abstained": answer["abstained"],
                "citations": answer["citations"],
                "is_supported": is_supported,
            }
        )

        last_answer = answer
        last_results = results
        if is_supported:
            return answer, results, trace

    return final_answer(last_answer, last_results, trace), last_results, trace


def generate_keywords(generator: OpenAIGenerator, question: dict[str, Any]) -> list[str]:
    instructions = """You are an assistant that generates keywords for information retrieval.

Return structured JSON only."""
    user_input = f"""Generate a list of important keywords related to the Query: {question['question']}.
Focus on keywords that are relevant and likely to appear in Cornell policy documents for BM25 search in a RAG framework.
Return only concise retrieval keywords."""
    result = generator.generate_with_schema(instructions, user_input, KEYWORDS_SCHEMA, "iterkey_keywords")
    return clean_keywords(result.parsed["keywords"])


def refine_keywords(generator: OpenAIGenerator, question: dict[str, Any], previous_keywords: list[str]) -> list[str]:
    instructions = """You refine keywords to improve document retrieval for BM25 search in a RAG framework.

Return structured JSON only."""
    user_input = f"""Refine the keyword selection process to improve retrieval of documents with the correct answer.
Query: {question['question']}
Previous Keywords: {previous_keywords}

Return a refined list of concise retrieval keywords."""
    result = generator.generate_with_schema(instructions, user_input, KEYWORDS_SCHEMA, "iterkey_keywords")
    return clean_keywords(result.parsed["keywords"])


def validate_answer(
    generator: OpenAIGenerator,
    question: dict[str, Any],
    answer: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
) -> bool:
    instructions = """You validate whether an answer is supported by retrieved Cornell policy documents.

Return structured JSON only."""
    docs = "\n\n".join(
        f"[chunk_id: {chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )
    user_input = f"""Is the following answer fully supported by the retrieved documents and responsive to the query?

Query: {question['question']}
Answer: {answer['answer']}
Abstained: {answer['abstained']}
Citations: {answer['citations']}

Retrieved Documents:
{docs}

Return true only if every substantive claim is supported by the retrieved documents or if abstention is appropriate because the documents do not contain the answer."""
    result = generator.generate_with_schema(instructions, user_input, VALIDATION_SCHEMA, "iterkey_validation")
    return bool(result.parsed["is_supported"])


def final_answer(
    last_answer: dict[str, Any] | None,
    last_results: list[SearchResult],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    if last_answer and last_answer.get("abstained"):
        return last_answer
    return {
        "answer": "",
        "abstained": True,
        "abstention_reason": f"IterKey validation did not confirm support after {len(trace)} iterations.",
        "citations": [result.chunk_id for result in last_results],
    }


def expanded_query_text(question: str, keywords: list[str]) -> str:
    return " ".join([question, *keywords]).strip()


def clean_keywords(keywords: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for keyword in keywords:
        text = " ".join(str(keyword).strip().split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned
