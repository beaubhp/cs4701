from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from pathlib import Path

from src.corpus.schema import Chunk, read_jsonl, write_jsonl


DEFAULT_OUTPUT_PATH = Path("data/processed/chunks.jsonl")
MIN_CHUNK_WORDS = 50

FAKE_HEADINGS = {
    "CORNELL UNIVERSITY",
    "POLICY LIBRARY",
    "MOST CURRENT VERSION OF THIS POLICY",
    "CORNELL TECH",
}

KNOWN_EMBEDDED_HEADINGS = [
    "A. Alcohol",
    "B. Assault and Endangering Behavior",
    "C. Bribery",
    "D. Collusion or Complicity",
    "E. Disorderly Conduct",
    "F. Disruption of University Activities",
    "G. Drug-Related Behavior",
    "H. Failure to Comply",
    "I. Fire Safety",
    "J. Harassment",
    "K. Hazing",
    "L. Invasion of Privacy and Appropriation of Identity",
    "M. Misconduct Related to Student Organizations or Groups",
    "N. Misrepresentation",
    "O. Obstruction with Code of Conduct Investigation and Adjudication Process",
    "P. Property Damage",
    "Q. Public Urination or Defecation",
    "R. Retaliation",
    "S. Theft",
    "T. Unauthorized Access",
    "U. Violation of Public Law(s)",
    "V. Weapons",
]


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def split_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def tail_words(text: str, count: int) -> str:
    words = split_words(text)
    if len(words) <= count:
        return " ".join(words)
    return " ".join(words[-count:])


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 140:
        return False
    if stripped.startswith("[Page "):
        return False
    if stripped in FAKE_HEADINGS:
        return False
    if is_metadata_line(stripped):
        return False
    if re.match(r"^(#{1,4}\s+)?[IVXLCDM]+\.\s+\S+", stripped):
        return True
    if re.match(r"^(Article|Section|Part)\s+[A-Z0-9IVXLCDM]+", stripped, flags=re.IGNORECASE):
        return True
    if re.match(r"^\d+(?:\.\d+)+\s+[A-Z][A-Za-z]", stripped) and word_count(stripped) <= 14:
        return True
    if is_lettered_heading(stripped):
        return True

    letters = [char for char in stripped if char.isalpha()]
    if len(letters) >= 4:
        upper_ratio = sum(char.isupper() for char in letters) / len(letters)
        if upper_ratio > 0.75 and word_count(stripped) <= 14:
            return True
    return False


def is_lettered_heading(line: str) -> bool:
    if not re.match(r"^[A-Z]\.\s+[A-Z][A-Za-z]", line):
        return False
    if line.endswith("-") or "," in line or ";" in line:
        return False
    return word_count(line) <= 8


def is_metadata_line(line: str) -> bool:
    metadata_prefixes = (
        "Volume:",
        "Chapter:",
        "Responsible Executive:",
        "Responsible Office",
        "Originally Issued:",
        "Current Version Approved:",
        "Last Updated:",
    )
    return line.startswith(metadata_prefixes)


def split_known_embedded_heading(line: str) -> tuple[str, str] | None:
    for heading in sorted(KNOWN_EMBEDDED_HEADINGS, key=len, reverse=True):
        if line == heading:
            return heading, ""
        if line.startswith(f"{heading} "):
            return heading, line[len(heading) :].strip()
    return None


def iter_section_blocks(text: str) -> Iterator[tuple[str | None, str]]:
    section: str | None = None
    paragraph_lines: list[str] = []

    def flush_paragraph() -> tuple[str | None, str] | None:
        if not paragraph_lines:
            return None
        paragraph = " ".join(paragraph_lines).strip()
        paragraph_lines.clear()
        if not paragraph:
            return None
        return section, paragraph

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flushed = flush_paragraph()
            if flushed:
                yield flushed
            continue
        if line.startswith("[Page "):
            flushed = flush_paragraph()
            if flushed:
                yield flushed
            continue
        if line.lower().endswith(", continued") or is_metadata_line(line):
            continue
        embedded = split_known_embedded_heading(line)
        if embedded:
            flushed = flush_paragraph()
            if flushed:
                yield flushed
            section, body = embedded
            if body:
                paragraph_lines.append(body)
            continue
        if looks_like_heading(line):
            flushed = flush_paragraph()
            if flushed:
                yield flushed
            section = line.lstrip("#").strip()
            continue
        paragraph_lines.append(line)

    flushed = flush_paragraph()
    if flushed:
        yield flushed


def split_oversized_block(block: str, max_words: int) -> list[str]:
    if word_count(block) <= max_words:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+", block)
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = word_count(sentence)
        if current and current_words + sentence_words > max_words:
            pieces.append(" ".join(current).strip())
            current = []
            current_words = 0
        if sentence_words > max_words:
            words = split_words(sentence)
            for start in range(0, len(words), max_words):
                pieces.append(" ".join(words[start : start + max_words]))
            continue
        current.append(sentence)
        current_words += sentence_words

    if current:
        pieces.append(" ".join(current).strip())
    return pieces


def is_non_substantive_front_matter(text: str, section: str | None) -> bool:
    if word_count(text) < MIN_CHUNK_WORDS and not re.search(r"[a-z]{2,}[^.!?]*[.!?]", text):
        return True
    if section is not None or word_count(text) >= MIN_CHUNK_WORDS:
        return False
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ["approved by the board", "effective august", "posted for campus notice", "policy library"]
    )


def make_chunks_for_document(
    document: dict,
    target_words: int,
    max_words: int,
    overlap_words: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_section: str | None = None
    current_blocks: list[str] = []
    current_words = 0
    pending_overlap: str | None = None

    def emit_chunk() -> None:
        nonlocal current_blocks, current_words, pending_overlap
        if not current_blocks:
            return

        text = "\n\n".join(current_blocks).strip()
        if not text:
            current_blocks = []
            current_words = 0
            return
        if is_non_substantive_front_matter(text, current_section):
            pending_overlap = None
            current_blocks = []
            current_words = 0
            return

        if current_section and not text.startswith(current_section):
            text = f"{current_section}\n\n{text}"

        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document['doc_id']}__{chunk_index:04d}",
                doc_id=document["doc_id"],
                title=document["title"],
                section=current_section,
                source_url=document["source_url"],
                text=text,
                word_count=word_count(text),
                chunk_index=chunk_index,
                source_type=document["source_type"],
                metadata_url=document.get("metadata_url"),
                topic=document.get("topic"),
            )
        )

        pending_overlap = tail_words(text, overlap_words) if overlap_words > 0 else None
        current_blocks = []
        current_words = 0

    for section, raw_block in iter_section_blocks(document["text"]):
        for block in split_oversized_block(raw_block, max_words):
            block_words = word_count(block)
            section_changed = bool(current_blocks) and section != current_section
            would_overflow = current_blocks and current_words + block_words > max_words

            if section_changed:
                emit_chunk()
                pending_overlap = None
            elif would_overflow:
                emit_chunk()

            if not current_blocks:
                current_section = section
                if pending_overlap:
                    current_blocks.append(pending_overlap)
                    current_words += word_count(pending_overlap)

            current_blocks.append(block)
            current_words += block_words

            if current_words >= target_words:
                emit_chunk()

    emit_chunk()
    return chunks


def chunk_documents(
    documents_path: Path,
    output_path: Path,
    target_words: int,
    max_words: int,
    overlap_words: int,
) -> None:
    documents = read_jsonl(documents_path)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(make_chunks_for_document(document, target_words, max_words, overlap_words))

    write_jsonl(output_path, chunks)
    print(f"Wrote {len(chunks)} chunks to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create retrieval chunks from normalized corpus documents.")
    parser.add_argument("documents", type=Path, help="Path to data/processed/documents.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-words", type=int, default=320)
    parser.add_argument("--max-words", type=int, default=380)
    parser.add_argument("--overlap-words", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_documents(
        documents_path=args.documents,
        output_path=args.output,
        target_words=args.target_words,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
    )


if __name__ == "__main__":
    main()
