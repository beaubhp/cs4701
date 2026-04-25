from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class Source:
    doc_id: str
    title: str
    source_type: str
    source_url: str
    include: bool = True
    metadata_url: str | None = None
    topic: str | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotMetadata:
    doc_id: str
    title: str
    source_type: str
    source_url: str
    snapshot_path: str
    sha256: str
    fetched_at: str
    metadata_url: str | None = None
    topic: str | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    source_type: str
    source_url: str
    snapshot_path: str
    sha256: str
    fetched_at: str
    text: str
    metadata_url: str | None = None
    topic: str | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str | None
    source_url: str
    text: str
    word_count: int
    chunk_index: int
    source_type: str
    metadata_url: str | None = None
    topic: str | None = None


SOURCE_FIELDS = {
    "doc_id",
    "title",
    "source_type",
    "source_url",
    "include",
    "metadata_url",
    "topic",
    "reason",
}


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources(path: Path) -> list[Source]:
    with path.open("r", encoding="utf-8") as file:
        raw_sources = yaml.safe_load(file) or []

    sources: list[Source] = []
    for item in raw_sources:
        known = {key: item.get(key) for key in SOURCE_FIELDS if key in item}
        extra = {key: value for key, value in item.items() if key not in SOURCE_FIELDS}
        sources.append(Source(**known, extra=extra))
    return sources


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                payload = asdict(row)
            else:
                payload = row
            file.write(json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "__dataclass_fields__"):
        payload = asdict(payload)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, default=str, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
