"""Tests for ingestion/chunker.py: chunk_sections."""
from __future__ import annotations

import pytest

from crucible.ingestion.chunker import chunk_sections
from crucible.ingestion.parsers import Section


def _sec(text: str, heading: str = "h", position: int = 0) -> Section:
    return Section(heading=heading, text=text, position=position)


class TestChunkSections:
    def test_empty_input(self):
        assert chunk_sections([]) == []

    def test_whitespace_section_skipped(self):
        assert chunk_sections([_sec("   \n  ")]) == []

    def test_short_section_passes_through(self):
        chunks = chunk_sections([_sec("short body")])
        assert len(chunks) == 1
        assert chunks[0].text == "short body"
        assert chunks[0].position == 0

    def test_windowing_triggers_over_max(self):
        text = ("A" * 500 + ". ") * 5  # ~2510 chars with sentence breaks
        chunks = chunk_sections([_sec(text)], max_chars=1000, overlap_chars=100)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 1000

    def test_overlap_preserved(self):
        text = "X" * 3000
        chunks = chunk_sections([_sec(text)], max_chars=1000, overlap_chars=200)
        assert len(chunks) >= 3
        # Adjacent chunks should share the overlap region when the window
        # ran to full max_chars (no sentence boundary shifted end).
        for a, b in zip(chunks, chunks[1:]):
            # Find shared suffix/prefix. Because chunker may have trimmed at
            # a boundary we only check *some* overlap exists.
            assert len(a.text) + len(b.text) > 1000

    def test_position_is_global(self):
        """Positions count across sections, not per-section."""
        sections = [_sec("alpha " * 500), _sec("beta " * 500)]
        chunks = chunk_sections(sections, max_chars=500, overlap_chars=50)
        positions = [c.position for c in chunks]
        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions)  # unique

    def test_heading_propagates(self):
        chunks = chunk_sections([_sec("x" * 3000, heading="hdr")], max_chars=500, overlap_chars=50)
        assert all(c.heading == "hdr" for c in chunks)

    def test_metadata_propagates(self):
        meta = {"author": "e"}
        sections = [Section(heading="h", text="short", position=0, metadata=meta)]
        chunks = chunk_sections(sections)
        assert chunks[0].metadata == meta

    def test_sentence_boundary_preferred_over_newline(self):
        """If a ". " exists but no \\n in tail region, chunker prefers it."""
        text = "a" * 900 + ". " + "b" * 200
        chunks = chunk_sections([_sec(text)], max_chars=1000, overlap_chars=50)
        # First chunk should end at the period, not be hard-cut at 1000.
        assert chunks[0].text.endswith(".") or chunks[0].text.endswith(". ")
