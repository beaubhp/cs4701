from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.corpus.schema import (
    Document,
    SnapshotMetadata,
    Source,
    file_sha256,
    load_sources,
    read_jsonl,
    utc_now_iso,
    write_json,
    write_jsonl,
)


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def source_extension(source_type: str) -> str:
    if source_type == "pdf":
        return "pdf"
    if source_type == "html":
        return "html"
    raise ValueError(f"Unsupported source_type: {source_type}")


def snapshot_source(source: Source, raw_dir: Path, refresh: bool) -> SnapshotMetadata:
    source_dir = raw_dir / source.doc_id
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = source_dir / f"source.{source_extension(source.source_type)}"
    metadata_path = source_dir / "metadata.json"

    if snapshot_path.exists() and not refresh:
        fetched_at = _existing_fetched_at(metadata_path)
    else:
        response = requests.get(source.source_url, timeout=60)
        response.raise_for_status()
        snapshot_path.write_bytes(response.content)
        fetched_at = utc_now_iso()

    metadata = SnapshotMetadata(
        doc_id=source.doc_id,
        title=source.title,
        source_type=source.source_type,
        source_url=source.source_url,
        snapshot_path=str(snapshot_path),
        sha256=file_sha256(snapshot_path),
        fetched_at=fetched_at,
        metadata_url=source.metadata_url,
        topic=source.topic,
        reason=source.reason,
        extra=source.extra,
    )
    write_json(metadata_path, metadata)
    return metadata


def _existing_fetched_at(metadata_path: Path) -> str:
    if metadata_path.exists():
        rows = read_jsonl(metadata_path) if metadata_path.suffix == ".jsonl" else []
        if rows and rows[0].get("fetched_at"):
            return rows[0]["fetched_at"]
        try:
            import json

            with metadata_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if payload.get("fetched_at"):
                return payload["fetched_at"]
        except Exception:
            pass
    return utc_now_iso()


def extract_text(metadata: SnapshotMetadata) -> str:
    path = Path(metadata.snapshot_path)
    if metadata.source_type == "pdf":
        return extract_pdf_text(path)
    if metadata.source_type == "html":
        return extract_html_text(path)
    raise ValueError(f"Unsupported source_type: {metadata.source_type}")


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {index}]\n{text}")
    return normalize_text("\n\n".join(pages), source_type="pdf")


def extract_html_text(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "form"]):
        tag.decompose()
    for selector in ["nav", "header", "footer", "aside"]:
        for tag in soup.select(selector):
            tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    parts: list[str] = []
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"], recursive=True):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            parts.append(f"\n{text}\n")
        elif element.name == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)

    if not parts:
        parts = list(root.stripped_strings)
    return normalize_text("\n".join(parts), source_type="html")


def normalize_text(text: str, source_type: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\uf025", "").replace("\uf075", "")
    text = text.replace("\uf0b7", "-").replace("\uf09f", "-").replace("\uf0fe", "-")
    text = re.sub(r"[\uf000-\uf8ff]", "", text)
    raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    lines: list[str] = []
    blank_seen = False
    for line in raw_lines:
        if source_type == "pdf":
            line = clean_pdf_line(line)
        if not line:
            if not blank_seen and lines:
                lines.append("")
            blank_seen = True
            continue
        blank_seen = False
        if source_type == "pdf" and re.fullmatch(r"\d+", line):
            continue
        if source_type == "pdf" and is_repeated_policy_pdf_chrome(line):
            continue
        lines.append(line)

    if source_type == "pdf":
        lines = merge_wrapped_pdf_lines(lines)

    normalized = "\n".join(lines)
    if source_type == "pdf":
        normalized = clean_pdf_text(normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def clean_pdf_line(line: str) -> str:
    line = re.sub(r"Cornell University Policy Office\s+policy\.cornell\.edu", "", line)
    line = re.sub(r"University Policy Office\s+www\.policy\.cornell\.edu", "", line)
    line = re.sub(r"^Cornell University Policy Office\b", "", line)
    line = re.sub(r"^policy\.cornell\.edu\b", "", line)
    line = re.sub(r"\b([A-Za-z]+)\s+-\s+([A-Za-z]+)", r"\1-\2", line)
    line = re.sub(r"\s{2,}", " ", line).strip()
    if is_toc_or_index_line(line):
        return ""
    return line


def clean_pdf_text(text: str) -> str:
    text = re.sub(
        r"Provost for Enrollment Last Updated: March 18, 2019 Processing and Reporting Changes in Student Enrollment Status Under Title IV",
        "",
        text,
    )
    text = re.sub(r"\b([A-Za-z]+)-\s+([A-Za-z]+)\b", r"\1\2", text)
    text = re.sub(r"\b([A-Za-z]+)\s+-\s+([A-Za-z]+)", r"\1-\2", text)
    replacements = {
        "welcomin g": "welcoming",
        "prope rty": "property",
        "Cod e": "Code",
        "th e": "the",
        "dep rives": "deprives",
        "c ondition": "condition",
        "inten tionally": "intentionally",
        "do cuments": "documents",
        "appea r": "appear",
        "Heari ng": "Hearing",
        "conside red": "considered",
        "de -recognized": "de-recognized",
        "non -punitive": "non-punitive",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def is_toc_or_index_line(line: str) -> bool:
    stripped = line.strip()
    if stripped in {"Table of Contents", "CONTENTS", "INDEX"}:
        return True
    if re.search(r"(\.{5,}|_{5,})\s*\d+", stripped):
        return True
    if len(stripped) > 20:
        leader_chars = stripped.count(".") + stripped.count("_")
        if leader_chars / len(stripped) > 0.25:
            return True
    return False


def is_repeated_policy_pdf_chrome(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "Cornell University Policy Office policy.cornell.edu" in stripped:
        return True
    if stripped == "University Policy Office www.policy.cornell.edu":
        return True
    if stripped.startswith("Cornell Policy Library") or stripped.startswith("Library Volume:"):
        return True
    if stripped.startswith("Volume:") or stripped.startswith("Chapter:"):
        return True
    if re.fullmatch(r"Policy\s+7\.\d+\s+Last Updated:.*", stripped):
        return True
    if stripped.startswith("Responsible Executive:") or stripped.startswith("Responsible Office"):
        return True
    if stripped.startswith("Originally Issued:") or stripped.startswith("Current Version Approved:"):
        return True
    if stripped.startswith("Cornell University Policy 7.1,"):
        return True
    if stripped.startswith("University Policy 7."):
        return True
    if stripped.startswith("Last updated:"):
        return True
    if re.fullmatch(r"POLICY\s+\d+\.\d+", stripped):
        return True
    if re.fullmatch(r"University Policy Office\s+www\.policy\.cornell\.edu", stripped):
        return True
    if stripped in {
        "Processing and Reporting Changes in Student Enrollment Status Under Title IV",
        "Voluntary Leave of Absence for Students",
    }:
        return True
    return False


def merge_wrapped_pdf_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if not merged or not line:
            merged.append(line)
            continue

        previous = merged[-1]
        if should_join_pdf_line(previous, line):
            merged[-1] = f"{previous} {line}"
        else:
            merged.append(line)
    return merged


def should_join_pdf_line(previous: str, current: str) -> bool:
    if not previous or previous.startswith("[Page ") or current.startswith("[Page "):
        return False
    if looks_like_heading(previous) or looks_like_heading(current):
        return False
    if re.match(r"^(\d+\.|[a-zA-Z]\.|[ivxlcdmIVXLCDM]+\.)\s+", current):
        return False
    if previous.endswith((".", "?", "!", ":", ";", ")")):
        return False
    return True


def looks_like_heading(line: str) -> bool:
    if len(line) > 120:
        return False
    if re.match(r"^(Article|Section|Part)\s+\w+", line, flags=re.IGNORECASE):
        return True
    if re.match(r"^[IVXLCDM]+\.\s+.+", line):
        return True
    letters = [char for char in line if char.isalpha()]
    return bool(letters) and sum(char.isupper() for char in letters) / len(letters) > 0.75


def build_documents(snapshots: list[SnapshotMetadata]) -> list[Document]:
    documents: list[Document] = []
    for metadata in snapshots:
        documents.append(
            Document(
                doc_id=metadata.doc_id,
                title=metadata.title,
                source_type=metadata.source_type,
                source_url=metadata.source_url,
                snapshot_path=metadata.snapshot_path,
                sha256=metadata.sha256,
                fetched_at=metadata.fetched_at,
                text=extract_text(metadata),
                metadata_url=metadata.metadata_url,
                topic=metadata.topic,
                reason=metadata.reason,
                extra=metadata.extra,
            )
        )
    return documents


def ingest(sources_path: Path, raw_dir: Path, processed_dir: Path, refresh: bool) -> None:
    sources = [source for source in load_sources(sources_path) if source.include]
    snapshots = [snapshot_source(source, raw_dir, refresh) for source in sources]
    documents = build_documents(snapshots)

    write_jsonl(processed_dir / "snapshot_manifest.jsonl", snapshots)
    write_jsonl(processed_dir / "documents.jsonl", documents)

    print(f"Wrote {len(snapshots)} snapshot rows to {processed_dir / 'snapshot_manifest.jsonl'}")
    print(f"Wrote {len(documents)} documents to {processed_dir / 'documents.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Cornell policy sources and extract normalized text.")
    parser.add_argument("sources", type=Path, help="Path to data/sources.yml")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--refresh", action="store_true", help="Re-download sources even if raw snapshots exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(args.sources, args.raw_dir, args.processed_dir, args.refresh)


if __name__ == "__main__":
    main()
