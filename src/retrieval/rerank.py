from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from src.retrieval.base import Retriever, SearchResult


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    def __init__(
        self,
        base_retriever: Retriever,
        candidate_k: int = 20,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        candidates = self.base_retriever.search(query, top_k=max(top_k, self.candidate_k))
        if not candidates:
            return []

        pairs = [(query, rerank_text(candidate)) for candidate in candidates]
        scores = self.model.predict(pairs)
        scored = [
            (float(score), rank, candidate)
            for rank, (score, candidate) in enumerate(zip(scores, candidates, strict=True), start=1)
        ]
        scored.sort(key=lambda item: (-item[0], item[1], item[2].doc_id, item[2].chunk_index))
        return [with_score(candidate, score) for score, _, candidate in scored[:top_k]]

    def describe(self) -> dict[str, Any]:
        return {
            "name": "cross_encoder_reranker",
            "reranker_model": self.model_name,
            "candidate_k": self.candidate_k,
            "base_retriever": self.base_retriever.describe(),
        }


def rerank_text(result: SearchResult) -> str:
    parts = [result.title, result.section or "", result.text]
    return "\n\n".join(part for part in parts if part)


def with_score(result: SearchResult, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=result.chunk_id,
        doc_id=result.doc_id,
        section=result.section,
        score=score,
        title=result.title,
        text=result.text,
        chunk_index=result.chunk_index,
    )
