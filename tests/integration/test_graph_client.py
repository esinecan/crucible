"""Integration tests for graph/client.py against a real Neo4j container.

Scope: lock in current behavior so the planned security + correctness refactor
(cypher_read rewrite, transaction boundaries, MERGE-after-MATCH orphan fix)
doesn't silently regress anything else. Tests that document known-broken
behavior use `xfail(strict=True)` — flipping to pass signals the fix landed.
"""
from __future__ import annotations

import pytest

from crucible.models import (
    Chunk,
    Corpus,
    Document,
    Insight,
    ReasoningNode,
)


pytestmark = pytest.mark.integration


# ── Helpers ─────────────────────────────────────────────────

def _make_corpus(cid: str = "corp-1", name: str = "test") -> Corpus:
    return Corpus(id=cid, name=name)


def _make_document(cid: str = "corp-1", did: str = "doc-1") -> Document:
    return Document(
        id=did,
        corpus_id=cid,
        path="notes/a.md",
        title="a",
        source_type="md",
        content_hash="hash1",
    )


def _fake_vec(seed: int, dim: int = 768) -> list[float]:
    # unit-ish vector with a seed-sensitive tilt
    import math

    raw = [math.sin(seed + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


# ── Corpus / Document / Chunks ──────────────────────────────

class TestCorpusUpsert:
    def test_idempotent(self, graph):
        c = _make_corpus()
        graph.upsert_corpus(c)
        graph.upsert_corpus(c)  # second call should be a no-op update
        rows = graph.cypher_read("MATCH (c:Corpus) RETURN c.id AS id, c.name AS name")
        assert rows == [{"id": "corp-1", "name": "test"}]


class TestDocumentUpsert:
    def test_links_to_existing_corpus(self, graph):
        graph.upsert_corpus(_make_corpus())
        graph.upsert_document(_make_document())
        rows = graph.cypher_read(
            "MATCH (d:Document)-[:BELONGS_TO]->(c:Corpus) "
            "RETURN d.id AS did, c.id AS cid"
        )
        assert rows == [{"did": "doc-1", "cid": "corp-1"}]

    def test_without_corpus_should_not_orphan(self, graph):
        """upsert_document must not leave an orphaned Document when the
        Corpus is absent. Accepts 'raise' or 'auto-create' as valid fixes.
        """
        try:
            graph.upsert_document(_make_document(cid="ghost-corpus"))
        except ValueError:
            pass  # raise-strategy is an acceptable fix
        rows = graph.cypher_read(
            "MATCH (d:Document) "
            "OPTIONAL MATCH (d)-[r:BELONGS_TO]->(c) "
            "RETURN d.id AS did, c.id AS cid"
        )
        assert rows == [] or (rows and rows[0]["cid"] is not None)


class TestChunks:
    def test_upsert_and_sequential_links(self, graph):
        graph.upsert_corpus(_make_corpus())
        graph.upsert_document(_make_document())
        chunks = [
            Chunk(
                id=f"ch-{i}",
                document_id="doc-1",
                text=f"chunk number {i}",
                position=i,
                embedding=_fake_vec(i),
            )
            for i in range(3)
        ]
        graph.upsert_chunks(chunks)
        graph.link_sequential("doc-1")

        rows = graph.cypher_read(
            "MATCH (a:Chunk)-[:NEXT]->(b:Chunk) "
            "RETURN a.id AS a, b.id AS b ORDER BY a"
        )
        assert rows == [
            {"a": "ch-0", "b": "ch-1"},
            {"a": "ch-1", "b": "ch-2"},
        ]

    def test_chunk_part_of_document(self, graph):
        graph.upsert_corpus(_make_corpus())
        graph.upsert_document(_make_document())
        graph.upsert_chunks([
            Chunk(id="ch-A", document_id="doc-1", text="x", position=0, embedding=_fake_vec(0))
        ])
        rows = graph.cypher_read(
            "MATCH (ch:Chunk)-[:PART_OF]->(d:Document) "
            "RETURN ch.id AS ch, d.id AS d"
        )
        assert rows == [{"ch": "ch-A", "d": "doc-1"}]


# ── Search ──────────────────────────────────────────────────

class TestSearch:
    def _seed(self, graph):
        graph.upsert_corpus(_make_corpus())
        graph.upsert_document(_make_document())
        graph.upsert_chunks([
            Chunk(id="ch-alpha", document_id="doc-1", text="alpha beta", position=0, embedding=_fake_vec(1)),
            Chunk(id="ch-gamma", document_id="doc-1", text="gamma delta", position=1, embedding=_fake_vec(2)),
        ])

    def test_vector_search_returns_ranked(self, graph):
        self._seed(graph)
        # Index population can briefly lag after upsert. Poll for readiness.
        import time
        for _ in range(20):
            rows = graph.vector_search(_fake_vec(1), limit=5)
            if rows:
                break
            time.sleep(0.2)
        assert rows, "vector_search returned empty even after waiting"
        assert rows[0]["id"] == "ch-alpha"

    def test_fulltext_search_returns_matches(self, graph):
        self._seed(graph)
        import time
        for _ in range(20):
            rows = graph.fulltext_search("gamma", limit=5)
            if rows:
                break
            time.sleep(0.2)
        assert rows, "fulltext_search returned empty even after waiting"
        assert rows[0]["id"] == "ch-gamma"


# ── cypher_read security ────────────────────────────────────

class TestCypherReadSecurity:
    def test_simple_read_works(self, graph):
        graph.upsert_corpus(_make_corpus())
        rows = graph.cypher_read("MATCH (c:Corpus) RETURN c.id AS id")
        assert rows == [{"id": "corp-1"}]

    @pytest.mark.parametrize("bad", [
        "CREATE (n:Pwned {id: 'x'})",
        "MATCH (n:Corpus) DELETE n",
        "MATCH (n:Corpus) SET n.name = 'evil'",
        "DROP INDEX corpus_id",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n:Corpus) REMOVE n.name",
    ])
    def test_obvious_writes_rejected(self, graph, bad):
        """Canonical write forms are correctly blocked today."""
        with pytest.raises(ValueError):
            graph.cypher_read(bad)

    def test_space_concatenated_merge_is_rejected(self, graph):
        """Historic bypass: `MERGE(n:...)` evaded the old token blocklist.
        Driver-level read mode refuses it now.
        """
        with pytest.raises(ValueError):
            graph.cypher_read("MERGE(n:Pwned {v: 1}) RETURN n")

    def test_foreach_merge_bypass_is_rejected(self, graph):
        """Historic bypass: FOREACH + concatenated MERGE. Also refused now."""
        with pytest.raises(ValueError):
            graph.cypher_read(
                "WITH 1 AS x FOREACH (i IN [x] | MERGE(m:Pwned {v: i}))"
            )

    def test_bypass_has_no_write_side_effect(self, graph):
        """Even if a future bypass tried to slip through, the node must not
        land — an invariant that holds regardless of how the guard is
        implemented.
        """
        try:
            graph.cypher_read("MERGE(n:Pwned {v: 1}) RETURN n")
        except Exception:
            pass
        rows = graph.cypher_read("MATCH (n:Pwned) RETURN count(n) AS c")
        assert rows[0]["c"] == 0

    def test_accepts_query_parameters(self, graph):
        """Regression test: cypher_read now accepts parameter kwargs for
        safe ID interpolation (previously only the query string was allowed,
        forcing callers into f-string Cypher).
        """
        graph.upsert_corpus(_make_corpus(cid="p-corp"))
        rows = graph.cypher_read(
            "MATCH (c:Corpus {id: $cid}) RETURN c.id AS id", cid="p-corp"
        )
        assert rows == [{"id": "p-corp"}]


# ── Insight ─────────────────────────────────────────────────

class TestInsight:
    def test_upsert_with_chunk_source_links(self, graph):
        graph.upsert_corpus(_make_corpus())
        graph.upsert_document(_make_document())
        graph.upsert_chunks([
            Chunk(id="ch-1", document_id="doc-1", text="x", position=0, embedding=_fake_vec(0)),
        ])
        insight = Insight(
            id="ins-1",
            corpus_id="corp-1",
            text="interesting",
            strategy="bridge",
            score=0.9,
            source_chunk_ids=["ch-1"],
            embedding=_fake_vec(100),
        )
        graph.upsert_insight(insight)
        rows = graph.cypher_read(
            "MATCH (i:Insight)-[:DERIVED_FROM]->(ch:Chunk) "
            "RETURN i.id AS i, ch.id AS ch"
        )
        assert rows == [{"i": "ins-1", "ch": "ch-1"}]

    def test_l2_insight_links_to_source_insight(self, graph):
        graph.upsert_corpus(_make_corpus())
        l1 = Insight(
            id="l1", corpus_id="corp-1", text="a", strategy="bridge",
            score=0.8, layer=1, embedding=_fake_vec(10),
        )
        l2 = Insight(
            id="l2", corpus_id="corp-1", text="b", strategy="meta",
            score=0.85, layer=2, source_insight_ids=["l1"],
            embedding=_fake_vec(11),
        )
        graph.upsert_insight(l1)
        graph.upsert_insight(l2)
        rows = graph.cypher_read(
            "MATCH (l2:Insight {id: 'l2'})-[:DERIVED_FROM]->(l1:Insight) "
            "RETURN l1.id AS src"
        )
        assert rows == [{"src": "l1"}]


# ── Reasoning tree ──────────────────────────────────────────

class TestReasoningTreePersistence:
    def test_roundtrip(self, graph):
        root = ReasoningNode(
            id="n-root", tree_id="tree-1", query="q", action_taken="root", depth=0,
        )
        child = ReasoningNode(
            id="n-child", tree_id="tree-1", query="q",
            action_taken="search", parent_id="n-root", depth=1,
            evidence_ids=[],
        )
        graph.upsert_reasoning_node(root)
        graph.upsert_reasoning_node(child)
        rows = graph.get_reasoning_tree("tree-1")
        ids = {r["id"] for r in rows}
        assert ids == {"n-root", "n-child"}
        by_id = {r["id"]: r for r in rows}
        assert by_id["n-child"]["parent_id"] == "n-root"
        assert by_id["n-root"]["parent_id"] is None


# ── Stats ───────────────────────────────────────────────────

class TestStats:
    def test_counts_basic(self, graph):
        graph.upsert_corpus(_make_corpus())
        graph.upsert_document(_make_document())
        graph.upsert_chunks([
            Chunk(id="ch-1", document_id="doc-1", text="x", position=0, embedding=_fake_vec(0)),
        ])
        s = graph.stats()
        assert s["corpora"] == 1
        assert s["docs"] == 1
        assert s["chunks"] == 1
        assert s["insights_l1"] == 0
        assert s["insights_l2"] == 0
