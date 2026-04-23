"""Tests for extraction/__init__.py: sanitize_type_name + ID helpers.

These functions are load-bearing: sanitize_type_name is the security gate that
lets graph/client.py:upsert_relation interpolate relation types directly into
a Cypher string. If the sanitizer ever slips, that f-string becomes a Cypher
injection surface.
"""
from __future__ import annotations

import re

import pytest

from crucible.extraction import (
    make_entity_id,
    make_relation_id,
    sanitize_type_name,
)

_VALID = re.compile(r"^[A-Z][A-Z0-9_]*$")


class TestSanitizeTypeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("acquires", "ACQUIRES"),
            ("part of", "PART_OF"),
            ("owned-by", "OWNED_BY"),
            ("  spaced  ", "SPACED"),
            ("MixED_Case", "MIXED_CASE"),
            ("multi   space", "MULTI_SPACE"),
            ("trailing_", "TRAILING"),
            ("_leading", "LEADING"),
            ("under__score", "UNDER_SCORE"),
            ("with.punct!", "WITHPUNCT"),
        ],
    )
    def test_happy_forms(self, raw, expected):
        assert sanitize_type_name(raw) == expected

    def test_digit_prefix_gets_rel_prefix(self):
        assert sanitize_type_name("1to1") == "REL_1TO1"

    def test_output_always_matches_safe_pattern(self):
        for raw in [
            "hello world",
            "x-ray",
            "1st-party",
            "Noun.phrase!",
            "WHERE",  # keyword-shaped but sanitizer doesn't care
        ]:
            assert _VALID.match(sanitize_type_name(raw))

    @pytest.mark.parametrize("raw", ["", "   ", "!!!", "___", "  ---  "])
    def test_unrecoverable_raises(self, raw):
        with pytest.raises(ValueError):
            sanitize_type_name(raw)

    def test_cypher_hostile_chars_stripped(self):
        """Security-critical: no backtick / brace / semicolon survives."""
        raw = "evil`)-[:DELETE]->()--"
        out = sanitize_type_name(raw)
        assert _VALID.match(out)
        for ch in ["`", "{", "}", ";", "'", '"', "[", "]", "(", ")", "-"]:
            assert ch not in out


class TestMakeEntityId:
    def test_deterministic(self):
        a = make_entity_id("corp1", "PERSON", "Alice")
        b = make_entity_id("corp1", "PERSON", "Alice")
        assert a == b

    def test_name_case_insensitive(self):
        """Same name in different cases maps to same id (intentional)."""
        a = make_entity_id("corp1", "PERSON", "Alice")
        b = make_entity_id("corp1", "PERSON", "ALICE")
        c = make_entity_id("corp1", "PERSON", " alice ")
        assert a == b == c

    def test_distinct_corpus_distinct_id(self):
        a = make_entity_id("corp1", "PERSON", "Alice")
        b = make_entity_id("corp2", "PERSON", "Alice")
        assert a != b

    def test_distinct_type_distinct_id(self):
        a = make_entity_id("corp1", "PERSON", "Apple")
        b = make_entity_id("corp1", "COMPANY", "Apple")
        assert a != b

    def test_length_20(self):
        assert len(make_entity_id("c", "T", "n")) == 20


class TestMakeRelationId:
    def test_deterministic(self):
        a = make_relation_id("src", "KNOWS", "tgt")
        b = make_relation_id("src", "KNOWS", "tgt")
        assert a == b

    def test_distinct_type_distinct_id(self):
        a = make_relation_id("src", "KNOWS", "tgt")
        b = make_relation_id("src", "WORKS_WITH", "tgt")
        assert a != b

    def test_direction_matters(self):
        a = make_relation_id("src", "KNOWS", "tgt")
        b = make_relation_id("tgt", "KNOWS", "src")
        assert a != b

    def test_chunk_does_not_participate(self):
        """Regression guard against reintroducing chunk_id in the hash."""
        # same claim from two different chunks must collapse to one id
        a = make_relation_id("src", "KNOWS", "tgt")
        b = make_relation_id("src", "KNOWS", "tgt")
        assert a == b
