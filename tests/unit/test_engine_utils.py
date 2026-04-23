"""Tests for insight/engine.py pure utilities: _cosine, _doc_distance,
_is_noisy, _parse_json_lenient.
"""
from __future__ import annotations

import math

import pytest

from crucible.insight.engine import (
    _cosine,
    _doc_distance,
    _is_noisy,
    _parse_json_lenient,
)


class TestCosine:
    def test_identical_vectors(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert _cosine([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_scale_invariant(self):
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]
        assert _cosine(a, b) == pytest.approx(1.0)


class TestDocDistance:
    def test_same_path(self):
        assert _doc_distance("a/b/c.md", "a/b/c.md") == 0.0

    def test_completely_different(self):
        assert _doc_distance("a/b.md", "x/y.md") == pytest.approx(1.0)

    def test_shared_prefix_reduces_distance(self):
        close = _doc_distance("a/b/c.md", "a/b/d.md")
        far = _doc_distance("a/b/c.md", "x/y/z.md")
        assert close < far


class TestIsNoisy:
    def test_literal_prefix(self):
        assert _is_noisy("docs/diffs/foo.txt", ["docs/diffs/*"])

    def test_extension_glob(self):
        assert _is_noisy("a/b.json", ["*.json"])

    def test_no_match(self):
        assert not _is_noisy("src/foo.py", ["*.json", "docs/diffs/*"])

    def test_empty_patterns(self):
        assert not _is_noisy("anything", [])


class TestParseJsonLenient:
    def test_plain_json_object(self):
        assert _parse_json_lenient('{"a": 1}') == {"a": 1}

    def test_plain_json_array(self):
        assert _parse_json_lenient('[1, 2, 3]') == [1, 2, 3]

    def test_markdown_fenced(self):
        text = 'prose\n```json\n{"x": 2}\n```\ntrailing'
        assert _parse_json_lenient(text) == {"x": 2}

    def test_markdown_fenced_no_lang(self):
        text = 'blah\n```\n{"x": 3}\n```'
        assert _parse_json_lenient(text) == {"x": 3}

    def test_embedded_object_in_prose(self):
        text = 'here is the output: {"a": 1} — done.'
        assert _parse_json_lenient(text) == {"a": 1}

    def test_embedded_array_in_prose(self):
        text = 'list: [1, 2, 3] ok'
        assert _parse_json_lenient(text) == [1, 2, 3]

    def test_garbage_returns_none(self):
        assert _parse_json_lenient("not json at all") is None

    def test_empty_string(self):
        assert _parse_json_lenient("") is None
