from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.corpus.schema import file_sha256, read_jsonl
from src.retrieval.base import SearchResult


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseIndex:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        model_name: str,
        chunks_sha256: str,
        cache_path: Path | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.model_name = model_name
        self.chunks_sha256 = chunks_sha256
        self.cache_path = cache_path
        self.model = model or SentenceTransformer(model_name)

    @classmethod
    def from_jsonl(
        cls,
        chunks_path: Path,
        model_name: str = DEFAULT_MODEL,
        cache_path: Path | None = None,
        rebuild_cache: bool = False,
    ) -> DenseIndex:
        chunks = read_jsonl(chunks_path)
        chunks_sha256 = file_sha256(chunks_path)
        cache_path = cache_path or default_cache_path(model_name)

        if not rebuild_cache:
            cached = load_cached_embeddings(cache_path, chunks, model_name, chunks_sha256)
            if cached is not None:
                return cls(chunks, cached, model_name, chunks_sha256, cache_path=cache_path)

        model = SentenceTransformer(model_name)
        texts = [embedding_text(chunk) for chunk in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        save_cached_embeddings(cache_path, chunks, embeddings, model_name, chunks_sha256)
        return cls(chunks, embeddings, model_name, chunks_sha256, cache_path=cache_path, model=model)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_vector = np.asarray(query_embedding, dtype=np.float32)[0]
        scores = self.embeddings @ query_vector

        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda idx: (-float(scores[idx]), self.chunks[idx]["doc_id"], self.chunks[idx]["chunk_index"]),
        )
        return [to_search_result(float(scores[idx]), self.chunks[idx]) for idx in ranked_indices[:top_k]]

    def describe(self) -> dict:
        return {
            "name": "local_dense_sentence_transformers",
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0,
            "chunks_sha256": self.chunks_sha256,
            "cache_path": str(self.cache_path) if self.cache_path else None,
        }


def embedding_text(chunk: dict[str, Any]) -> str:
    parts = [chunk.get("title") or "", chunk.get("section") or "", chunk.get("topic") or "", chunk.get("text") or ""]
    return "\n\n".join(part for part in parts if part)


def default_cache_path(model_name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name.split("/")[-1]).strip("-")
    return Path(f"data/processed/embeddings_{slug}.npz")


def load_cached_embeddings(
    cache_path: Path,
    chunks: list[dict[str, Any]],
    model_name: str,
    chunks_sha256: str,
) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    cache = np.load(cache_path, allow_pickle=False)
    metadata = json.loads(str(cache["metadata"]))
    expected_chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    cached_chunk_ids = [str(value) for value in cache["chunk_ids"]]
    if metadata.get("model_name") != model_name:
        return None
    if metadata.get("chunks_sha256") != chunks_sha256:
        return None
    if cached_chunk_ids != expected_chunk_ids:
        return None
    return np.asarray(cache["embeddings"], dtype=np.float32)


def save_cached_embeddings(
    cache_path: Path,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    model_name: str,
    chunks_sha256: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "chunks_sha256": chunks_sha256,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    np.savez_compressed(
        cache_path,
        embeddings=embeddings.astype(np.float32),
        chunk_ids=np.array([chunk["chunk_id"] for chunk in chunks]),
        metadata=json.dumps(metadata, sort_keys=True),
    )


def to_search_result(score: float, chunk: dict[str, Any]) -> SearchResult:
    return SearchResult(
        chunk_id=chunk["chunk_id"],
        doc_id=chunk["doc_id"],
        section=chunk.get("section"),
        score=score,
        title=chunk["title"],
        text=chunk["text"],
        chunk_index=chunk["chunk_index"],
    )
