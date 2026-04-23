"""Tests for insight/engine.py pure utilities: _cosine, _doc_distance,
_is_noisy, _is_code_like_type, InsightEngine._filter_gap_rows.

JSON parsing was hoisted to llm_client.parse_json_lenient and its tests
now live in test_llm_client.py — kept here as a re-export so any
downstream caller importing from engine still works.
"""
from __future__ import annotations

import math

import pytest

from crucible.insight.engine import (
    InsightEngine,
    _cosine,
    _doc_distance,
    _is_code_like_type,
    _is_noisy,
)
from crucible.llm_client import parse_json_lenient as _parse_json_lenient


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


class TestIsCodeLikeType:
    """Code-aware gap heuristic depends on type detection. Explicit-set
    membership covers the common cocrucible-ontology types; token-based
    fallback catches unfamiliar code-corpus ontologies that emit names
    like SERVICE_COMPONENT or QUERY_ENGINE."""

    @pytest.mark.parametrize("etype", [
        "MODULE", "COMPONENT", "METHOD", "FUNCTION", "CLASS",
        "INSIGHT_ENGINE", "KNOWLEDGE_GRAPH", "EXTRACTION_PIPELINE",
        "REASONING_STRATEGY", "ONTOLOGY",
    ])
    def test_explicit_set_returns_true(self, etype):
        assert _is_code_like_type(etype) is True

    @pytest.mark.parametrize("etype", [
        "PRICING_ENGINE", "SERVICE_COMPONENT", "STORAGE_PIPELINE",
        "RANK_ALGORITHM", "GRAPH_METHOD",
    ])
    def test_token_fallback_returns_true(self, etype):
        assert _is_code_like_type(etype) is True

    @pytest.mark.parametrize("etype", [
        "PERSON", "PLACE", "EVENT", "POLITICAL_PARTY", "DOCUMENT",
        "CONCEPT", "ENTITY", "RELATION",
    ])
    def test_prose_types_return_false(self, etype):
        assert _is_code_like_type(etype) is False

    @pytest.mark.parametrize("etype", [None, ""])
    def test_empty_returns_false(self, etype):
        assert _is_code_like_type(etype) is False


class TestGapRowsFilter:
    """_filter_gap_rows applies the code-aware threshold + scoring without
    a Neo4j round-trip. Cocrucible's 4 hallucinated gap insights all came
    through because every well-encapsulated function looks like a gap under
    the prose heuristic; the stricter rule for code types should drop them
    while leaving genuinely under-covered prose entities alone."""

    def _row(self, name="X", type="MODULE", incoming=3, mentions=1):
        return {
            "name": name,
            "type": type,
            "incoming": incoming,
            "mentions": mentions,
        }

    def test_module_with_small_deficit_filtered(self):
        """incoming=3, mentions=1 → deficit=2, below the code threshold of 5.
        Under the prose rule this would surface; the cocrucible "MCTS skips
        backpropagation" hallucination came from rows like this."""
        rows = [self._row("get_dirty_entities", "MODULE", incoming=3, mentions=1)]
        out = InsightEngine._filter_gap_rows(rows)
        assert out == []

    def test_person_with_small_deficit_kept(self):
        """Same numbers (incoming=3, mentions=1) but PERSON type — prose
        heuristic still applies; the entity surfaces as a gap candidate.
        Same concept-overlap rule for prose corpora."""
        rows = [self._row("Alice", "PERSON", incoming=3, mentions=1)]
        out = InsightEngine._filter_gap_rows(rows)
        assert len(out) == 1
        score, name, etype, incoming, mentions = out[0]
        assert name == "Alice"
        assert etype == "PERSON"
        assert incoming == 3 and mentions == 1
        assert 0.6 < score < 1.0

    def test_module_with_severe_deficit_kept(self):
        """incoming=10, mentions=1 → deficit=9 (>= 5) AND mentions=1 (<= 2).
        Both code-strict conditions satisfied; surfaces as a real gap."""
        rows = [self._row("orphan_method", "MODULE", incoming=10, mentions=1)]
        out = InsightEngine._filter_gap_rows(rows)
        assert len(out) == 1
        assert out[0][1] == "orphan_method"

    def test_module_with_high_mentions_filtered_even_if_diff_large(self):
        """incoming=20, mentions=5 → deficit=15 (>= 5) but mentions=5 (> 2).
        Code-strict requires BOTH conditions; this one fails the mentions cap."""
        rows = [self._row("popular_method", "MODULE", incoming=20, mentions=5)]
        out = InsightEngine._filter_gap_rows(rows)
        assert out == []

    def test_token_fallback_engine_treated_as_code(self):
        """ENGINE-suffixed type catches via token fallback."""
        rows = [self._row("ranker", "PRICING_ENGINE", incoming=3, mentions=1)]
        out = InsightEngine._filter_gap_rows(rows)
        # Treated as code → strict rule → deficit 2 < 5 → filtered
        assert out == []

    def test_max_results_cap(self):
        """The slice limit is on the post-filter list, not the pre-filter
        rows. Useful when a code-corpus produces many false candidates that
        get filtered out — the cap should still let through up to N real ones."""
        rows = [
            self._row(f"prose-{i}", "PERSON", incoming=3, mentions=1)
            for i in range(20)
        ]
        out = InsightEngine._filter_gap_rows(rows, max_results=5)
        assert len(out) == 5

    def test_mixed_types_preserves_real_gaps(self):
        """A mixed batch of prose + code types — all prose entities pass,
        only severely-deficient code entities pass."""
        rows = [
            self._row("Alice", "PERSON", incoming=3, mentions=1),       # pass
            self._row("get_x", "MODULE", incoming=3, mentions=1),       # filtered (low deficit)
            self._row("orphan_y", "MODULE", incoming=10, mentions=1),   # pass (severe)
            self._row("Bob", "PERSON", incoming=4, mentions=2),         # pass
            self._row("popular_z", "MODULE", incoming=10, mentions=5),  # filtered (mentions cap)
        ]
        out = InsightEngine._filter_gap_rows(rows)
        names = [r[1] for r in out]
        assert names == ["Alice", "orphan_y", "Bob"]
