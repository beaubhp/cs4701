from __future__ import annotations

from typing import Any


NON_RAG_PROMPT_VERSION = "non_rag_v1"
RAG_PROMPT_VERSION = "rag_v1"


def build_non_rag_prompt(question: dict[str, Any]) -> tuple[str, str]:
    instructions = """You answer questions about Cornell University policy.

You are not being given the fixed Cornell policy corpus for this question. Answer from your general knowledge and reasoning as directly as possible. This is a non-RAG baseline designed to test what an LLM says without retrieved evidence.

Rules:
- Do not cite sources because no evidence passages were provided.
- Set citations to an empty list.
- Do not abstain merely because no corpus evidence was provided.
- Abstain only if the question is impossible to answer at all or is not a Cornell policy question.
- Return structured JSON only."""
    user_input = f"""Question ID: {question['question_id']}
Question: {question['question']}

Answer the question from general knowledge without retrieved evidence."""
    return instructions, user_input


def build_rag_prompt(question: dict[str, Any], retrieved_chunks: list[dict[str, Any]]) -> tuple[str, str]:
    instructions = """You answer questions about Cornell University policy using only the provided evidence chunks.

Rules:
- Use only the provided evidence chunks.
- Every substantive claim in the answer must be supported by the cited chunks.
- Citations must be exact chunk_id values from the provided evidence.
- If the evidence does not support the answer, abstain instead of guessing.
- If the question asks for a specific detail that is not in the evidence, abstain.
- Return structured JSON only."""

    evidence_blocks = []
    for chunk in retrieved_chunks:
        evidence_blocks.append(
            f"""[chunk_id: {chunk['chunk_id']}]
title: {chunk['title']}
section: {chunk.get('section')}
text:
{chunk['text']}"""
        )

    user_input = f"""Question ID: {question['question_id']}
Question: {question['question']}

Evidence chunks:
{chr(10).join(evidence_blocks)}

Answer using only the evidence chunks above."""
    return instructions, user_input
