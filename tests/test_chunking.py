"""Offline tests for the chunker (no Azure needed)."""
from pathlib import Path

import pytest

from ragbot.chunking import chunk_markdown, load_and_chunk

SAMPLE = """# Closing procedure
Only trained closers may close.

## Cash
Count the till twice.
"""


def test_splits_on_headings():
    chunks = chunk_markdown(SAMPLE, "closing.md")
    assert [c.title for c in chunks] == ["Closing procedure", "Cash"]
    assert all(c.source == "closing.md" for c in chunks)


def test_ids_are_unique_and_search_safe():
    chunks = chunk_markdown(SAMPLE * 3, "x.md")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(ch.isalnum() or ch in "-_=" for i in ids for ch in i)


def test_long_section_is_windowed_with_overlap():
    body = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_markdown(f"# Long\n{body}", "long.md", size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.content) <= 300 + len("Long\n") for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_markdown("# A\n" + "x " * 500, "a.md", size=100, overlap=100)


def test_sample_docs_load():
    folder = Path(__file__).resolve().parent.parent / "data" / "sample_sops"
    chunks = load_and_chunk(folder)
    assert len(chunks) > 0
    assert {c.source for c in chunks} >= {"closing-procedure.md"}
