from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.corpus.schema import read_jsonl
from src.retrieval.tokenize import expand_query_text, tokenize


TITLE_WEIGHT = 2
SECTION_WEIGHT = 3
TOPIC_WEIGHT = 1
BODY_WEIGHT = 1


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    doc_id: str
    section: str | None
    score: float
    title: str
    text: str
    chunk_index: int


@dataclass(frozen=True)
class IndexedChunk:
    chunk: dict[str, Any]
    tokens: list[str]
    term_freqs: Counter[str]
    length: int
    title_section_terms: set[str]


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.2, b: float = 0.65) -> None:
        self.k1 = k1
        self.b = b
        self.indexed_chunks = [index_chunk(chunk) for chunk in chunks]
        self.num_docs = len(self.indexed_chunks)
        self.avg_doc_len = average_doc_length(self.indexed_chunks)
        self.doc_freqs = document_frequencies(self.indexed_chunks)
        self.idf = {
            term: math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freqs.items()
        }

    @classmethod
    def from_jsonl(cls, chunks_path: Path, k1: float = 1.2, b: float = 0.65) -> BM25Index:
        return cls(read_jsonl(chunks_path), k1=k1, b=b)

    def search(self, query: str, top_k: int = 5, expand: bool = False) -> list[SearchResult]:
        query_text = expand_query_text(query) if expand else query
        query_terms = list(dict.fromkeys(tokenize(query_text)))
        if not query_terms:
            return []

        scored: list[tuple[float, int, IndexedChunk]] = []
        for indexed in self.indexed_chunks:
            score = self.score_terms(query_terms, indexed)
            if score <= 0:
                continue
            field_match_count = len(set(query_terms) & indexed.title_section_terms)
            scored.append((score, field_match_count, indexed))

        scored.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2].chunk["doc_id"],
                item[2].chunk["chunk_index"],
            )
        )
        return [to_search_result(score, indexed) for score, _, indexed in scored[:top_k]]

    def score_terms(self, query_terms: list[str], indexed: IndexedChunk) -> float:
        if indexed.length == 0 or self.avg_doc_len == 0:
            return 0.0

        score = 0.0
        norm = 1 - self.b + self.b * (indexed.length / self.avg_doc_len)
        for term in query_terms:
            tf = indexed.term_freqs.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * norm
            score += self.idf.get(term, 0.0) * (numerator / denominator)
        return score


def index_chunk(chunk: dict[str, Any]) -> IndexedChunk:
    title_tokens = tokenize(chunk.get("title") or "")
    section_tokens = tokenize(chunk.get("section") or "")
    topic_tokens = tokenize(chunk.get("topic") or "")
    body_tokens = tokenize(body_text(chunk))

    tokens = (
        title_tokens * TITLE_WEIGHT
        + section_tokens * SECTION_WEIGHT
        + topic_tokens * TOPIC_WEIGHT
        + body_tokens * BODY_WEIGHT
    )
    return IndexedChunk(
        chunk=chunk,
        tokens=tokens,
        term_freqs=Counter(tokens),
        length=len(tokens),
        title_section_terms=set(title_tokens + section_tokens),
    )


def body_text(chunk: dict[str, Any]) -> str:
    text = chunk.get("text") or ""
    section = chunk.get("section")
    if section and text.startswith(section):
        return text[len(section) :].strip()
    return text


def average_doc_length(indexed_chunks: list[IndexedChunk]) -> float:
    if not indexed_chunks:
        return 0.0
    return sum(indexed.length for indexed in indexed_chunks) / len(indexed_chunks)


def document_frequencies(indexed_chunks: list[IndexedChunk]) -> Counter[str]:
    doc_freqs: Counter[str] = Counter()
    for indexed in indexed_chunks:
        doc_freqs.update(set(indexed.tokens))
    return doc_freqs


def to_search_result(score: float, indexed: IndexedChunk) -> SearchResult:
    chunk = indexed.chunk
    return SearchResult(
        chunk_id=chunk["chunk_id"],
        doc_id=chunk["doc_id"],
        section=chunk.get("section"),
        score=score,
        title=chunk["title"],
        text=chunk["text"],
        chunk_index=chunk["chunk_index"],
    )
