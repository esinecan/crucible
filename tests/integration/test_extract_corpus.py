"""End-to-end integration test for EntityExtractor.extract_corpus.

The LLM is stubbed via _extract_chunk_raw and embeddings via fake_embed so
the test is deterministic and free of network calls — everything downstream
of the LLM (aggregation, upserts, chunk marking) runs against a real Neo4j.
"""
from __future__ import annotations

import pytest

from crucible.extraction import make_entity_id
from crucible.extraction.extractor import EntityExtractor
from crucible.extraction.ontology import Ontology, OntologyClass, OntologyRelation
from crucible.models import Chunk, Corpus, Document


pytestmark = pytest.mark.integration


def _fake_ontology(corpus_id: str = "c1") -> Ontology:
    return Ontology(
        corpus_id=corpus_id,
        classes=[
            OntologyClass(name="PERSON", description="A named individual"),
            OntologyClass(name="COMPANY", description="An organization"),
            OntologyClass(name="CONCEPT", description="Catch-all"),
        ],
        relations=[
            OntologyRelation(
                name="WORKS_AT", description="employed at",
                domain="PERSON", range="COMPANY",
            ),
            OntologyRelation(
                name="KNOWS", description="knows someone",
                domain="PERSON", range="PERSON",
            ),
        ],
    )


def _seed_corpus(graph, fake_embed, corpus_id: str = "c1") -> str:
    """Seed one corpus + one doc + 3 chunks. Returns doc_id."""
    import math
    graph.upsert_corpus(Corpus(id=corpus_id, name="test"))
    doc_id = "doc-1"
    graph.upsert_document(Document(
        id=doc_id, corpus_id=corpus_id, path="notes.md", title="notes",
        source_type="md", content_hash="h",
    ))
    chunks = []
    for i in range(3):
        v = [math.sin(i + j) for j in range(768)]
        norm = math.sqrt(sum(x * x for x in v))
        embedding = [x / norm for x in v]
        chunks.append(Chunk(
            id=f"ch-{i}", document_id=doc_id, text=f"chunk {i}",
            position=i, embedding=embedding,
        ))
    graph.upsert_chunks(chunks)
    return doc_id


def _patch_extractor(monkeypatch, mapping: dict[str, tuple[list, list]]):
    """Patch _extract_chunk_raw to return deterministic LLM output per chunk.

    mapping: chunk_id -> (entities_raw, relations_raw)
    """
    def _stub(self, chunk, corpus_id):
        return mapping.get(chunk["id"], ([], []))
    monkeypatch.setattr(EntityExtractor, "_extract_chunk_raw", _stub)


class TestExtractCorpusHappyPath:
    def test_entities_deduped_across_chunks(self, graph, config, fake_embed, monkeypatch):
        """Alice extracted from all 3 chunks → one Entity, source_chunks=[ch-0,ch-1,ch-2]."""
        _seed_corpus(graph, fake_embed)
        _patch_extractor(monkeypatch, {
            "ch-0": ([{"name": "Alice", "type": "PERSON", "description": "alice short"}], []),
            "ch-1": ([{"name": "Alice", "type": "PERSON", "description": "alice longer description"}], []),
            "ch-2": ([{"name": "Alice", "type": "PERSON", "description": "med len"}], []),
        })

        extractor = EntityExtractor(config, graph, _fake_ontology())
        stats = extractor.extract_corpus("c1", embed=True, workers=2)

        assert stats["entities"] == 1
        assert stats["chunks"] == 3
        assert stats["documents"] == 1

        alice_id = make_entity_id("c1", "PERSON", "Alice")
        rows = graph.cypher_read(
            "MATCH (e:Entity {id: $eid}) "
            "RETURN e.source_chunks AS chunks, e.description AS desc",
            eid=alice_id,
        )
        assert sorted(rows[0]["chunks"]) == ["ch-0", "ch-1", "ch-2"]
        assert rows[0]["desc"] == "alice longer description"  # longest wins

    def test_relations_deduped_across_chunks(self, graph, config, fake_embed, monkeypatch):
        """Same (Alice WORKS_AT Acme) claim in two chunks → one Relation."""
        _seed_corpus(graph, fake_embed)
        ents = [
            {"name": "Alice", "type": "PERSON"},
            {"name": "Acme", "type": "COMPANY"},
        ]
        rels = [{
            "source": "Alice", "target": "Acme", "type": "WORKS_AT",
            "evidence": "employed", "confidence": 0.8,
        }]
        _patch_extractor(monkeypatch, {
            "ch-0": (ents, rels),
            "ch-1": (ents, rels),
            "ch-2": ([], []),
        })

        extractor = EntityExtractor(config, graph, _fake_ontology())
        stats = extractor.extract_corpus("c1", embed=True, workers=2)

        assert stats["relations"] == 1
        rows = graph.cypher_read(
            "MATCH ()-[r:WORKS_AT]->() "
            "RETURN r.source_chunks AS chunks, r.confidence AS conf"
        )
        assert len(rows) == 1
        assert sorted(rows[0]["chunks"]) == ["ch-0", "ch-1"]

    def test_mentioned_in_edges_created(self, graph, config, fake_embed, monkeypatch):
        _seed_corpus(graph, fake_embed)
        _patch_extractor(monkeypatch, {
            "ch-0": ([{"name": "Alice", "type": "PERSON"}], []),
            "ch-1": ([{"name": "Alice", "type": "PERSON"}], []),
            "ch-2": ([], []),
        })
        extractor = EntityExtractor(config, graph, _fake_ontology())
        extractor.extract_corpus("c1", embed=True, workers=2)

        rows = graph.cypher_read(
            "MATCH (e:Entity)-[:MENTIONED_IN]->(ch:Chunk) "
            "RETURN e.name AS name, collect(ch.id) AS chunks"
        )
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"
        assert sorted(rows[0]["chunks"]) == ["ch-0", "ch-1"]

    def test_all_chunks_marked_extracted(self, graph, config, fake_embed, monkeypatch):
        _seed_corpus(graph, fake_embed)
        _patch_extractor(monkeypatch, {
            "ch-0": ([], []), "ch-1": ([], []), "ch-2": ([], []),
        })
        extractor = EntityExtractor(config, graph, _fake_ontology())
        extractor.extract_corpus("c1", embed=True, workers=2)

        # Second invocation should see zero pending docs.
        docs = graph.get_unextracted_docs("c1")
        assert docs == []


class TestExtractCorpusErrorPolicy:
    def test_transient_error_leaves_chunk_unmarked(self, graph, config, fake_embed, monkeypatch):
        """On a transient LLM error, the chunk is NOT marked → next run retries."""
        import httpx
        from crucible.extraction.extractor import TransientExtractionError
        _seed_corpus(graph, fake_embed)

        def _stub(self, chunk, corpus_id):
            if chunk["id"] == "ch-1":
                raise TransientExtractionError("rate limited")
            return ([], [])
        monkeypatch.setattr(EntityExtractor, "_extract_chunk_raw", _stub)

        extractor = EntityExtractor(config, graph, _fake_ontology())
        stats = extractor.extract_corpus("c1", embed=False, workers=2)
        assert stats["errors"] == 1

        # ch-0 and ch-2 marked; ch-1 still pending
        rows = graph.cypher_read(
            "MATCH (ch:Chunk) "
            "OPTIONAL MATCH (ch)-[r:EXTRACTION_DONE]->(:Corpus) "
            "RETURN ch.id AS id, r IS NOT NULL AS marked "
            "ORDER BY ch.id"
        )
        by_id = {r["id"]: r["marked"] for r in rows}
        assert by_id["ch-0"] is True
        assert by_id["ch-1"] is False
        assert by_id["ch-2"] is True

    def test_permanent_error_marks_chunk(self, graph, config, fake_embed, monkeypatch):
        """On a permanent error (e.g. ValueError from bad JSON), still mark the
        chunk so the next run doesn't loop.
        """
        _seed_corpus(graph, fake_embed)

        def _stub(self, chunk, corpus_id):
            if chunk["id"] == "ch-1":
                raise ValueError("malformed LLM output")
            return ([], [])
        monkeypatch.setattr(EntityExtractor, "_extract_chunk_raw", _stub)

        extractor = EntityExtractor(config, graph, _fake_ontology())
        extractor.extract_corpus("c1", embed=False, workers=2)

        # All three marked (poison permanent failure, move on)
        docs = graph.get_unextracted_docs("c1")
        assert docs == []
