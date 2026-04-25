from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    doc_id: str
    section: str | None
    score: float
    title: str
    text: str
    chunk_index: int


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        ...

    def describe(self) -> dict:
        ...
