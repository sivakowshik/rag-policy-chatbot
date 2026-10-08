"""Split markdown documents into overlapping, heading-aware chunks.

Pure Python with no Azure dependency, so it can be unit tested offline.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    source: str  # file name, used for citations
    title: str  # nearest markdown heading
    content: str
    chunk_index: int


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs, splitting on markdown headings."""
    sections: list[tuple[str, str]] = []
    heading = "Introduction"
    buffer: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            if "".join(buffer).strip():
                sections.append((heading, "\n".join(buffer).strip()))
            heading = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    if "".join(buffer).strip():
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~size-character windows, breaking on whitespace."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def _make_id(source: str, index: int) -> str:
    # Azure AI Search keys allow letters, digits, '_', '-' and '='.
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{index}"


def chunk_markdown(text: str, source: str, size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        for piece in _window(body, size, overlap):
            idx = len(chunks)
            chunks.append(
                Chunk(
                    id=_make_id(source, idx),
                    source=source,
                    title=heading,
                    content=f"{heading}\n{piece}",
                    chunk_index=idx,
                )
            )
    return chunks


def load_and_chunk(folder: Path, size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name, size, overlap))
    return chunks
