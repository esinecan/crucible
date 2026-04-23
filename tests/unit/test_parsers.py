"""Tests for ingestion/parsers.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.ingestion.parsers import PARSERS, parse_file


@pytest.fixture
def tmp_md(tmp_path: Path) -> Path:
    p = tmp_path / "doc.md"
    p.write_text(
        "intro text\n"
        "# H1 One\n"
        "body under one\n"
        "## H2 Two\n"
        "body under two\n",
        encoding="utf-8",
    )
    return p


def test_parse_markdown_splits_on_headings(tmp_md):
    sections = parse_file(tmp_md)
    headings = [s.heading for s in sections]
    texts = [s.text for s in sections]
    assert "H1 One" in headings
    assert "H2 Two" in headings
    # intro lands under the file-stem heading
    assert any("intro text" in t for t in texts)


def test_parse_markdown_empty_file(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    sections = parse_file(p)
    # Parser returns at least one section (may be empty text)
    assert len(sections) >= 1


def test_parse_markdown_no_headings(tmp_path):
    p = tmp_path / "plain.md"
    p.write_text("just some text\nno headings here", encoding="utf-8")
    sections = parse_file(p)
    assert len(sections) == 1
    assert "just some text" in sections[0].text


def test_parse_json_dict(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"alpha": "one", "beta": {"nested": 2}}), encoding="utf-8")
    sections = parse_file(p)
    headings = [s.heading for s in sections]
    assert "alpha" in headings
    assert "beta" in headings


def test_parse_json_list(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(
        json.dumps([{"name": "first"}, {"title": "second"}, "plain"]),
        encoding="utf-8",
    )
    sections = parse_file(p)
    headings = [s.heading for s in sections]
    assert "first" in headings
    assert "second" in headings


def test_parse_json_malformed_falls_back_to_text(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    sections = parse_file(p)
    assert len(sections) == 1
    assert "{not valid json" in sections[0].text


def test_parse_text_paragraphs(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("para one\n\npara two\n\npara three", encoding="utf-8")
    sections = parse_file(p)
    texts = [s.text for s in sections]
    assert "para one" in texts
    assert "para two" in texts
    assert "para three" in texts


def test_parse_empty_text(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    sections = parse_file(p)
    assert len(sections) == 1


def test_unknown_extension_uses_text_parser(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("hello\n\nworld", encoding="utf-8")
    sections = parse_file(p)
    assert len(sections) == 2


def test_all_registered_extensions_callable():
    for ext, parser in PARSERS.items():
        assert callable(parser)
