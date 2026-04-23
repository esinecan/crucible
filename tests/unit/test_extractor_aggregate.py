"""Unit tests for extraction/aggregate.py.

The aggregator is a pure function over per-chunk LLM output; these tests
lock in its deduplication, provenance-tracking, and alias-resolution contract.
"""
from __future__ import annotations

import pytest

from crucible.extraction import make_entity_id, make_relation_id
from crucible.extraction.aggregate import aggregate_document_extractions


VALID_TYPES = {"PERSON", "COMPANY", "CONCEPT", "PLACE"}
VALID_RELS = {"WORKS_AT", "FOUNDED", "LOCATED_IN", "KNOWS"}


def _ent(name, type="PERSON", desc="", aliases=None):
    return {
        "name": name,
        "type": type,
        "description": desc,
        "aliases": aliases or [],
    }


def _rel(source, target, type="KNOWS", evidence="", confidence=0.8):
    return {
        "source": source,
        "target": target,
        "type": type,
        "evidence": evidence,
        "confidence": confidence,
    }


class TestEmptyInput:
    def test_no_chunks(self):
        agg = aggregate_document_extractions([], "c1", VALID_TYPES, VALID_RELS)
        assert agg.entities == {}
        assert agg.relations == {}

    def test_empty_chunk(self):
        agg = aggregate_document_extractions(
            [("ch-1", [], [])], "c1", VALID_TYPES, VALID_RELS
        )
        assert agg.entities == {}
        assert agg.relations == {}


class TestEntityAggregation:
    def test_single_entity_single_chunk(self):
        agg = aggregate_document_extractions(
            [("ch-1", [_ent("Alice", "PERSON", "A person")], [])],
            "c1", VALID_TYPES, VALID_RELS,
        )
        assert len(agg.entities) == 1
        eid = make_entity_id("c1", "PERSON", "Alice")
        assert agg.entities[eid].name == "Alice"
        assert agg.entities[eid].source_chunks == ["ch-1"]
        assert agg.entities[eid].description == "A person"

    def test_same_entity_across_chunks_unions_source_chunks(self):
        chunks = [
            ("ch-1", [_ent("Alice", "PERSON")], []),
            ("ch-2", [_ent("Alice", "PERSON")], []),
            ("ch-3", [_ent("Alice", "PERSON")], []),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        assert len(agg.entities) == 1
        eid = make_entity_id("c1", "PERSON", "Alice")
        assert sorted(agg.entities[eid].source_chunks) == ["ch-1", "ch-2", "ch-3"]

    def test_longer_description_wins(self):
        chunks = [
            ("ch-1", [_ent("Alice", desc="short")], []),
            ("ch-2", [_ent("Alice", desc="a much longer description")], []),
            ("ch-3", [_ent("Alice", desc="mid len")], []),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        eid = make_entity_id("c1", "PERSON", "Alice")
        assert agg.entities[eid].description == "a much longer description"

    def test_aliases_union_across_chunks(self):
        chunks = [
            ("ch-1", [_ent("Alice", aliases=["Al"])], []),
            ("ch-2", [_ent("Alice", aliases=["Ally", "Al"])], []),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        eid = make_entity_id("c1", "PERSON", "Alice")
        assert sorted(agg.entities[eid].aliases) == ["Al", "Ally"]

    def test_same_name_different_type_kept_separate(self):
        """PERSON 'Apple' and COMPANY 'Apple' must NOT be conflated."""
        chunks = [
            ("ch-1", [_ent("Apple", "COMPANY")], []),
            ("ch-2", [_ent("Apple", "PERSON")], []),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        assert len(agg.entities) == 2
        assert make_entity_id("c1", "COMPANY", "Apple") in agg.entities
        assert make_entity_id("c1", "PERSON", "Apple") in agg.entities

    def test_unknown_type_coerced_to_concept(self):
        chunks = [("ch-1", [_ent("Widget", "MACHINE")], [])]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        eid = make_entity_id("c1", "CONCEPT", "Widget")
        assert eid in agg.entities
        assert agg.entities[eid].entity_type == "CONCEPT"

    def test_missing_name_skipped(self):
        chunks = [("ch-1", [{"type": "PERSON"}, _ent("Alice")], [])]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        assert len(agg.entities) == 1


class TestRelationAggregation:
    def test_relation_resolved_from_chunk_local_map(self):
        chunks = [
            ("ch-1",
             [_ent("Alice"), _ent("Acme", "COMPANY")],
             [_rel("Alice", "Acme", "WORKS_AT", evidence="a quote")]),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        assert len(agg.relations) == 1
        rel = next(iter(agg.relations.values()))
        assert rel.relation_type == "WORKS_AT"
        assert rel.source_chunks == ["ch-1"]
        assert rel.evidence == "a quote"

    def test_same_relation_across_chunks_dedupes(self):
        """(Alice WORKS_AT Acme) extracted from two chunks → one Relation."""
        chunks = [
            ("ch-1",
             [_ent("Alice"), _ent("Acme", "COMPANY")],
             [_rel("Alice", "Acme", "WORKS_AT", evidence="first quote", confidence=0.6)]),
            ("ch-2",
             [_ent("Alice"), _ent("Acme", "COMPANY")],
             [_rel("Alice", "Acme", "WORKS_AT", evidence="second quote", confidence=0.9)]),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        assert len(agg.relations) == 1
        rel = next(iter(agg.relations.values()))
        assert sorted(rel.source_chunks) == ["ch-1", "ch-2"]
        assert rel.confidence == 0.9  # max wins
        assert "first quote" in rel.evidence and "second quote" in rel.evidence

    def test_unresolved_endpoint_drops_relation(self):
        """Relation referencing a name that never got extracted is discarded."""
        chunks = [("ch-1", [_ent("Alice")], [_rel("Alice", "Nobody", "KNOWS")])]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        assert agg.relations == {}

    def test_unknown_relation_type_dropped(self):
        chunks = [
            ("ch-1",
             [_ent("Alice"), _ent("Bob")],
             [_rel("Alice", "Bob", "FOES_WITH")]),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        assert agg.relations == {}

    def test_confidence_clamped(self):
        chunks = [
            ("ch-1",
             [_ent("Alice"), _ent("Bob")],
             [_rel("Alice", "Bob", "KNOWS", confidence=5.0)]),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        rel = next(iter(agg.relations.values()))
        assert rel.confidence == 1.0

    def test_relation_id_collapses_chunk_provenance(self):
        """Cross-chunk same claim produces ONE rid (chunk_id is NOT in the hash)."""
        alice_id = make_entity_id("c1", "PERSON", "Alice")
        acme_id = make_entity_id("c1", "COMPANY", "Acme")
        expected_rid = make_relation_id(alice_id, "WORKS_AT", acme_id)

        chunks = [
            ("ch-1", [_ent("Alice"), _ent("Acme", "COMPANY")],
             [_rel("Alice", "Acme", "WORKS_AT")]),
            ("ch-2", [_ent("Alice"), _ent("Acme", "COMPANY")],
             [_rel("Alice", "Acme", "WORKS_AT")]),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        assert list(agg.relations.keys()) == [expected_rid]


class TestAliasResolution:
    def test_alias_within_chunk_resolves_to_canonical(self):
        chunks = [
            ("ch-1",
             [_ent("Alice", aliases=["Al"]), _ent("Bob")],
             [_rel("Al", "Bob", "KNOWS")]),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        assert len(agg.relations) == 1
        rel = next(iter(agg.relations.values()))
        assert rel.source_entity_id == make_entity_id("c1", "PERSON", "Alice")

    def test_ambiguous_alias_without_topological_signal_drops_relation(self):
        """Two PERSONs share alias 'A'; the relation references 'A' in a new
        chunk with no co-occurring context — drop rather than risk wrong merge.
        """
        chunks = [
            ("ch-1", [_ent("Alice", aliases=["A"])], []),
            ("ch-2", [_ent("Anton", aliases=["A"])], []),
            ("ch-3",
             [_ent("Bob")],
             [_rel("A", "Bob", "KNOWS")]),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        # No topological signal (Alice and Anton have no chunks in common with
        # Bob). The ambiguous 'A' must not force a merge.
        assert agg.relations == {}

    def test_ambiguous_alias_with_topological_signal_resolves(self):
        """'A' is ambiguous, but Alice shares chunks with Cat (also present in
        the relation's chunk). Anton does not. Resolve to Alice.
        """
        chunks = [
            ("ch-1",
             [_ent("Alice", aliases=["A"]), _ent("Cat", "CONCEPT")],
             []),
            ("ch-2",
             [_ent("Anton", aliases=["A"])],
             []),
            ("ch-3",
             [_ent("Cat", "CONCEPT"), _ent("Bob")],
             [_rel("A", "Bob", "KNOWS")]),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        # ch-3's chunk_local doesn't have 'A' (neither Alice nor Anton
        # extracted here), so falls through to doc-wide disambiguation.
        # Alice shares ch-1 with Cat; Anton shares nothing with Cat.
        assert len(agg.relations) == 1
        rel = next(iter(agg.relations.values()))
        assert rel.source_entity_id == make_entity_id("c1", "PERSON", "Alice")


class TestProvenance:
    def test_entity_source_chunks_are_deduplicated(self):
        """Same chunk's dict-level repeat doesn't double-count."""
        chunks = [
            ("ch-1", [_ent("Alice"), _ent("Alice", desc="dup")], []),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        eid = make_entity_id("c1", "PERSON", "Alice")
        assert agg.entities[eid].source_chunks == ["ch-1"]

    def test_relation_source_chunks_are_deduplicated(self):
        chunks = [
            ("ch-1",
             [_ent("Alice"), _ent("Bob")],
             [_rel("Alice", "Bob", "KNOWS"), _rel("Alice", "Bob", "KNOWS")]),
        ]
        agg = aggregate_document_extractions(chunks, "c1", VALID_TYPES, VALID_RELS)
        rel = next(iter(agg.relations.values()))
        assert rel.source_chunks == ["ch-1"]
