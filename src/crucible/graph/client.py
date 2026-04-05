from __future__ import annotations

import json

from neo4j import GraphDatabase

from ..config import Config
from ..models import Chunk, Corpus, Document, Insight, ReasoningNode
from .schema import ensure_schema


class CrucibleGraph:
    def __init__(self, config: Config):
        self._driver = GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        )
        with self._driver.session() as s:
            ensure_schema(s)

    def close(self):
        self._driver.close()

    # ── Corpus ──────────────────────────────────────────────

    def upsert_corpus(self, corpus: Corpus) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (c:Corpus {id: $id}) "
                "SET c.name = $name, c.description = $desc, "
                "c.source_path = $path, c.created_at = $ts",
                id=corpus.id,
                name=corpus.name,
                desc=corpus.description,
                path=corpus.source_path,
                ts=corpus.created_at,
            )

    # ── Document ────────────────────────────────────────────

    def upsert_document(self, doc: Document) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (d:Document {id: $id}) "
                "SET d.corpus_id=$cid, d.path=$path, d.title=$title, "
                "d.source_type=$st, d.content_hash=$hash, "
                "d.metadata=$meta, d.ingested_at=$ts "
                "WITH d "
                "MATCH (c:Corpus {id: $cid}) "
                "MERGE (d)-[:BELONGS_TO]->(c)",
                id=doc.id,
                cid=doc.corpus_id,
                path=doc.path,
                title=doc.title,
                st=doc.source_type,
                hash=doc.content_hash,
                meta=json.dumps(doc.metadata),
                ts=doc.ingested_at,
            )

    # ── Chunks ──────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        rows = [
            {
                "id": c.id,
                "doc_id": c.document_id,
                "text": c.text,
                "pos": c.position,
                "heading": c.heading,
                "embedding": c.embedding,
                "meta": json.dumps(c.metadata),
                "ts": c.created_at,
            }
            for c in chunks
        ]
        with self._driver.session() as s:
            s.run(
                "UNWIND $rows AS r "
                "MERGE (ch:Chunk {id: r.id}) "
                "SET ch.document_id=r.doc_id, ch.text=r.text, "
                "ch.position=r.pos, ch.heading=r.heading, "
                "ch.embedding=r.embedding, ch.metadata=r.meta, "
                "ch.created_at=r.ts "
                "WITH ch, r "
                "MATCH (d:Document {id: r.doc_id}) "
                "MERGE (ch)-[:PART_OF]->(d)",
                rows=rows,
            )

    def link_sequential(self, document_id: str) -> None:
        with self._driver.session() as s:
            s.run(
                "MATCH (ch:Chunk {document_id: $did}) "
                "WITH ch ORDER BY ch.position "
                "WITH collect(ch) AS cs "
                "UNWIND range(0, size(cs)-2) AS i "
                "WITH cs[i] AS a, cs[i+1] AS b "
                "MERGE (a)-[:NEXT]->(b)",
                did=document_id,
            )

    # ── Search ──────────────────────────────────────────────

    def vector_search(
        self,
        embedding: list[float],
        limit: int = 10,
        corpus_id: str | None = None,
    ) -> list[dict]:
        q = (
            "CALL db.index.vector.queryNodes('chunk_embedding', $k, $emb) "
            "YIELD node, score "
            "MATCH (node)-[:PART_OF]->(d:Document) "
        )
        if corpus_id:
            q += "WHERE d.corpus_id = $cid "
        q += (
            "RETURN node.id AS id, node.text AS text, node.heading AS heading, "
            "d.path AS doc_path, d.title AS doc_title, score "
            "ORDER BY score DESC"
        )
        with self._driver.session() as s:
            return [dict(r) for r in s.run(q, emb=embedding, k=limit, cid=corpus_id)]

    def fulltext_search(self, query: str, limit: int = 10) -> list[dict]:
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "CALL db.index.fulltext.queryNodes('chunk_text', $q) "
                    "YIELD node, score "
                    "MATCH (node)-[:PART_OF]->(d:Document) "
                    "RETURN node.id AS id, node.text AS text, node.heading AS heading, "
                    "d.path AS doc_path, score "
                    "ORDER BY score DESC LIMIT $lim",
                    q=query,
                    lim=limit,
                )
            ]

    def vector_search_insights(
        self,
        embedding: list[float],
        limit: int = 5,
        corpus_id: str | None = None,
    ) -> list[dict]:
        """Semantic search over insight embeddings.

        Returns insights ranked by cosine similarity to the query embedding.
        Each result includes the insight text, strategy, score, and source info.
        Used by MCTS answer mode to include discovered insights as evidence.
        """
        q = (
            "CALL db.index.vector.queryNodes('insight_embedding', $k, $emb) "
            "YIELD node, score "
        )
        if corpus_id:
            q += "WHERE node.corpus_id = $cid "
        q += (
            "RETURN node.id AS id, node.text AS text, node.strategy AS strategy, "
            "node.score AS insight_score, node.layer AS layer, score "
            "ORDER BY score DESC"
        )
        with self._driver.session() as s:
            return [dict(r) for r in s.run(q, emb=embedding, k=limit, cid=corpus_id)]

    def cypher_read(self, query: str) -> list[dict]:
        blocked = {"CREATE", "DELETE", "SET", "REMOVE", "MERGE", "DROP", "DETACH"}
        tokens = set(query.upper().split())
        for kw in blocked:
            if kw in tokens:
                raise ValueError(f"Write keyword '{kw}' blocked in read-only cypher")
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query)]

    # ── Insight ─────────────────────────────────────────────

    def upsert_insight(self, insight: Insight) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (i:Insight {id: $id}) "
                "SET i.corpus_id=$cid, i.text=$text, i.strategy=$strat, "
                "i.score=$score, i.novelty=$nov, i.relevance=$rel, "
                "i.layer=$layer, i.embedding=$emb, "
                "i.context=$ctx, i.created_at=$ts",
                id=insight.id,
                cid=insight.corpus_id,
                text=insight.text,
                strat=insight.strategy,
                score=insight.score,
                nov=insight.novelty,
                rel=insight.relevance,
                layer=insight.layer,
                emb=insight.embedding or None,
                ctx=json.dumps(insight.context),
                ts=insight.created_at,
            )
            # L1: link to source chunks
            for chunk_id in insight.source_chunk_ids:
                s.run(
                    "MATCH (i:Insight {id: $iid}), (ch:Chunk {id: $cid}) "
                    "MERGE (i)-[:DERIVED_FROM]->(ch)",
                    iid=insight.id,
                    cid=chunk_id,
                )
            # L2: link to source insights
            for src_id in insight.source_insight_ids:
                s.run(
                    "MATCH (i:Insight {id: $iid}), (src:Insight {id: $sid}) "
                    "MERGE (i)-[:DERIVED_FROM]->(src)",
                    iid=insight.id,
                    sid=src_id,
                )

    # ── Stats & Sampling ────────────────────────────────────

    def stats(self) -> dict:
        with self._driver.session() as s:
            r = s.run(
                "OPTIONAL MATCH (co:Corpus) WITH count(co) AS corpora "
                "OPTIONAL MATCH (d:Document) WITH corpora, count(d) AS docs "
                "OPTIONAL MATCH (ch:Chunk) WITH corpora, docs, count(ch) AS chunks "
                "OPTIONAL MATCH (i:Insight {layer: 1}) "
                "WITH corpora, docs, chunks, count(i) AS l1 "
                "OPTIONAL MATCH (i2:Insight {layer: 2}) "
                "RETURN corpora, docs, chunks, l1 AS insights_l1, count(i2) AS insights_l2"
            ).single()
            return dict(r) if r else {}

    def sample_chunks(self, n: int = 50, corpus_id: str | None = None) -> list[dict]:
        q = (
            "MATCH (ch:Chunk)-[:PART_OF]->(d:Document) "
            "WHERE ch.embedding IS NOT NULL "
        )
        if corpus_id:
            q += "AND d.corpus_id = $cid "
        q += (
            "RETURN ch.id AS id, ch.text AS text, ch.embedding AS embedding, "
            "d.path AS doc_path, rand() AS r ORDER BY r LIMIT $n"
        )
        with self._driver.session() as s:
            return [dict(r) for r in s.run(q, cid=corpus_id, n=n)]

    def sample_insights(self, n: int = 50, layer: int = 1) -> list[dict]:
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "MATCH (i:Insight) "
                    "WHERE i.layer = $layer AND i.embedding IS NOT NULL "
                    "RETURN i.id AS id, i.text AS text, i.embedding AS embedding, "
                    "i.strategy AS strategy, i.score AS original_score, "
                    "rand() AS r ORDER BY r LIMIT $n",
                    layer=layer,
                    n=n,
                )
            ]

    def get_unembedded_insights(self, batch_size: int = 100) -> list[dict]:
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "MATCH (i:Insight) WHERE i.embedding IS NULL "
                    "RETURN i.id AS id, i.text AS text LIMIT $n",
                    n=batch_size,
                )
            ]

    def set_insight_embeddings(self, updates: list[tuple[str, list[float]]]) -> None:
        with self._driver.session() as s:
            s.run(
                "UNWIND $rows AS r "
                "MATCH (i:Insight {id: r.id}) "
                "SET i.embedding = r.embedding",
                rows=[{"id": uid, "embedding": emb} for uid, emb in updates],
            )

    def set_insight_layers(self, layer: int) -> int:
        """Set layer on all insights that don't have one yet."""
        with self._driver.session() as s:
            r = s.run(
                "MATCH (i:Insight) WHERE i.layer IS NULL "
                "SET i.layer = $layer RETURN count(i) AS updated",
                layer=layer,
            ).single()
            return r["updated"] if r else 0

    # ── Reasoning Tree ─────────────────────────────────────

    def upsert_reasoning_node(self, node: ReasoningNode) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (rn:ReasoningNode {id: $id}) "
                "SET rn.tree_id=$tid, rn.query=$q, "
                "rn.evidence_ids=$ev, rn.partial_answer=$ans, "
                "rn.action_taken=$act, rn.score=$score, "
                "rn.visits=$vis, rn.total_score=$ts, "
                "rn.depth=$depth, rn.context=$ctx, "
                "rn.created_at=$cat",
                id=node.id,
                tid=node.tree_id,
                q=node.query,
                ev=node.evidence_ids,
                ans=node.partial_answer,
                act=node.action_taken,
                score=node.score,
                vis=node.visits,
                ts=node.total_score,
                depth=node.depth,
                ctx=json.dumps(node.context),
                cat=node.created_at,
            )
            if node.parent_id:
                s.run(
                    "MATCH (child:ReasoningNode {id: $cid}), "
                    "(parent:ReasoningNode {id: $pid}) "
                    "MERGE (child)-[:CHILD_OF]->(parent)",
                    cid=node.id,
                    pid=node.parent_id,
                )
            # Link evidence (chunks and insights)
            for eid in node.evidence_ids:
                s.run(
                    "MATCH (rn:ReasoningNode {id: $rid}) "
                    "OPTIONAL MATCH (ch:Chunk {id: $eid}) "
                    "OPTIONAL MATCH (i:Insight {id: $eid}) "
                    "FOREACH (_ IN CASE WHEN ch IS NOT NULL THEN [1] ELSE [] END | "
                    "  MERGE (rn)-[:USES_EVIDENCE]->(ch)) "
                    "FOREACH (_ IN CASE WHEN i IS NOT NULL THEN [1] ELSE [] END | "
                    "  MERGE (rn)-[:USES_EVIDENCE]->(i))",
                    rid=node.id,
                    eid=eid,
                )

    def get_reasoning_tree(self, tree_id: str) -> list[dict]:
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "MATCH (rn:ReasoningNode {tree_id: $tid}) "
                    "OPTIONAL MATCH (rn)-[:CHILD_OF]->(parent:ReasoningNode) "
                    "RETURN rn.id AS id, rn.query AS query, "
                    "rn.partial_answer AS partial_answer, "
                    "rn.action_taken AS action_taken, "
                    "rn.score AS score, rn.visits AS visits, "
                    "rn.total_score AS total_score, "
                    "rn.depth AS depth, rn.evidence_ids AS evidence_ids, "
                    "rn.context AS context, "
                    "parent.id AS parent_id "
                    "ORDER BY rn.depth, rn.id",
                    tid=tree_id,
                )
            ]

    def backpropagate_score(self, node_id: str, score: float) -> None:
        """Walk up the tree from node_id to root, updating visits and total_score."""
        with self._driver.session() as s:
            s.run(
                "MATCH path = (leaf:ReasoningNode {id: $nid})-[:CHILD_OF*0..]->(ancestor:ReasoningNode) "
                "SET ancestor.visits = ancestor.visits + 1, "
                "ancestor.total_score = ancestor.total_score + $score",
                nid=node_id,
                score=score,
            )
