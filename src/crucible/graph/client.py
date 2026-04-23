from __future__ import annotations

import json

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from ..config import Config
from ..extraction import sanitize_rel_type
from ..models import Chunk, Corpus, Document, Entity, Insight, ReasoningNode, Relation
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
        def _tx(tx):
            found = tx.run(
                "MATCH (c:Corpus {id: $cid}) RETURN c.id AS cid",
                cid=doc.corpus_id,
            ).single()
            if found is None:
                raise ValueError(
                    f"Corpus {doc.corpus_id!r} not found; call upsert_corpus first"
                )
            tx.run(
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

        with self._driver.session() as s:
            s.execute_write(_tx)

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

    def cypher_read(self, query: str, **params) -> list[dict]:
        """Execute a read-only Cypher query.

        Uses a managed read transaction so Neo4j itself refuses writes — no
        keyword blocklist to bypass. Parameters are passed as kwargs; prefer
        them over string interpolation to avoid Cypher injection.
        """
        def _read(tx):
            return [dict(r) for r in tx.run(query, **params)]

        try:
            with self._driver.session() as s:
                return s.execute_read(_read)
        except ClientError as e:
            # Translate the driver's write-in-read-transaction rejection into
            # the ValueError callers already expect.
            msg = str(e).lower()
            if "write" in msg or "read" in msg or "access" in msg:
                raise ValueError(f"Write operation blocked in read-only cypher: {e}") from e
            raise

    # ── Insight ─────────────────────────────────────────────

    def upsert_insight(self, insight: Insight) -> None:
        def _tx(tx):
            tx.run(
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
            if insight.source_chunk_ids:
                tx.run(
                    "MATCH (i:Insight {id: $iid}) "
                    "UNWIND $cids AS cid "
                    "MATCH (ch:Chunk {id: cid}) "
                    "MERGE (i)-[:DERIVED_FROM]->(ch)",
                    iid=insight.id,
                    cids=insight.source_chunk_ids,
                )
            if insight.source_insight_ids:
                tx.run(
                    "MATCH (i:Insight {id: $iid}) "
                    "UNWIND $sids AS sid "
                    "MATCH (src:Insight {id: sid}) "
                    "MERGE (i)-[:DERIVED_FROM]->(src)",
                    iid=insight.id,
                    sids=insight.source_insight_ids,
                )

        with self._driver.session() as s:
            s.execute_write(_tx)

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
            rows = []
            for r in s.run(
                "MATCH (i:Insight) "
                "WHERE i.layer = $layer AND i.embedding IS NOT NULL "
                "OPTIONAL MATCH (i)-[:DERIVED_FROM]->(ch:Chunk) "
                "RETURN i.id AS id, i.text AS text, i.embedding AS embedding, "
                "i.strategy AS strategy, i.score AS original_score, "
                "i.context AS context, "
                "collect(ch.id) AS source_chunk_ids, "
                "rand() AS r ORDER BY r LIMIT $n",
                layer=layer,
                n=n,
            ):
                d = dict(r)
                # Parse context from Neo4j string if needed
                if isinstance(d.get("context"), str):
                    try:
                        d["context"] = json.loads(d["context"])
                    except (json.JSONDecodeError, TypeError):
                        d["context"] = {}
                elif d.get("context") is None:
                    d["context"] = {}
                rows.append(d)
            return rows

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
        def _tx(tx):
            tx.run(
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
                tx.run(
                    "MATCH (child:ReasoningNode {id: $cid}), "
                    "(parent:ReasoningNode {id: $pid}) "
                    "MERGE (child)-[:CHILD_OF]->(parent)",
                    cid=node.id,
                    pid=node.parent_id,
                )
            if node.evidence_ids:
                # Link evidence chunks + insights in two UNWIND passes.
                # Non-matching ids are silently dropped by MATCH, same as
                # the original FOREACH/OPTIONAL-MATCH behavior.
                tx.run(
                    "MATCH (rn:ReasoningNode {id: $rid}) "
                    "UNWIND $eids AS eid "
                    "MATCH (ch:Chunk {id: eid}) "
                    "MERGE (rn)-[:USES_EVIDENCE]->(ch)",
                    rid=node.id,
                    eids=node.evidence_ids,
                )
                tx.run(
                    "MATCH (rn:ReasoningNode {id: $rid}) "
                    "UNWIND $eids AS eid "
                    "MATCH (i:Insight {id: eid}) "
                    "MERGE (rn)-[:USES_EVIDENCE]->(i)",
                    rid=node.id,
                    eids=node.evidence_ids,
                )

        with self._driver.session() as s:
            s.execute_write(_tx)

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

    # ── Entity Extraction ──────────────────────────────────

    def upsert_entity(self, entity: Entity) -> None:
        """Upsert an Entity, unioning source_chunks and aliases across calls.

        source_chunks is the authoritative mention trace — the mention_count
        readings in queries derive from size(source_chunks). description still
        uses 'longer wins' as a stopgap; async synthesis is the planned fix.
        """
        def _tx(tx):
            tx.run(
                "MERGE (e:Entity {id: $id}) "
                "ON CREATE SET e.corpus_id=$cid, e.name=$name, "
                "  e.entity_type=$etype, e.description=$desc, "
                "  e.aliases=$aliases, e.embedding=$emb, "
                "  e.source_chunks=$chunks, e.properties=$props, "
                "  e.created_at=$ts "
                "ON MATCH SET "
                "  e.source_chunks = e.source_chunks + "
                "    [x IN $chunks WHERE NOT x IN e.source_chunks], "
                "  e.aliases = e.aliases + "
                "    [x IN $aliases WHERE NOT x IN e.aliases], "
                "  e.description = CASE WHEN size(e.description) < size($desc) "
                "    THEN $desc ELSE e.description END",
                id=entity.id,
                cid=entity.corpus_id,
                name=entity.name,
                etype=entity.entity_type,
                desc=entity.description,
                aliases=entity.aliases,
                emb=entity.embedding or None,
                chunks=entity.source_chunks,
                props=json.dumps(entity.properties),
                ts=entity.created_at,
            )

        with self._driver.session() as s:
            s.execute_write(_tx)

    def upsert_relation(self, relation: Relation) -> None:
        """Create or augment a typed relationship between entities.

        Dynamic Cypher is safe because relation_type passes through
        sanitize_rel_type which pins it to ^[A-Z][A-Z0-9_]*$. The relation id
        is hash(src, rel_type, tgt), so re-extracting the same claim from a
        new chunk augments source_chunks rather than creating a new edge.
        """
        rel_type = sanitize_rel_type(relation.relation_type)
        query = (
            f"MATCH (src:Entity {{id: $src_id}}), (tgt:Entity {{id: $tgt_id}}) "
            f"MERGE (src)-[r:{rel_type} {{id: $rid}}]->(tgt) "
            f"ON CREATE SET "
            f"  r.evidence=$ev, r.confidence=$conf, "
            f"  r.source_chunks=$chunks, r.properties=$props, "
            f"  r.created_at=$ts "
            f"ON MATCH SET "
            f"  r.source_chunks = r.source_chunks + "
            f"    [x IN $chunks WHERE NOT x IN r.source_chunks], "
            f"  r.confidence = CASE WHEN r.confidence < $conf "
            f"    THEN $conf ELSE r.confidence END"
        )

        def _tx(tx):
            tx.run(
                query,
                src_id=relation.source_entity_id,
                tgt_id=relation.target_entity_id,
                rid=relation.id,
                ev=relation.evidence,
                conf=relation.confidence,
                chunks=relation.source_chunks,
                props=json.dumps(relation.properties),
                ts=relation.created_at,
            )

        with self._driver.session() as s:
            s.execute_write(_tx)

    def link_entity_to_chunk(self, entity_id: str, chunk_id: str) -> None:
        with self._driver.session() as s:
            s.run(
                "MATCH (e:Entity {id: $eid}), (ch:Chunk {id: $cid}) "
                "MERGE (e)-[:MENTIONED_IN]->(ch)",
                eid=entity_id,
                cid=chunk_id,
            )

    def entity_search(self, query: str, limit: int = 10) -> list[dict]:
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "CALL db.index.fulltext.queryNodes('entity_fulltext', $q) "
                    "YIELD node, score "
                    "RETURN node.id AS id, node.name AS name, "
                    "node.entity_type AS entity_type, "
                    "node.description AS description, "
                    "size(node.source_chunks) AS mention_count, score "
                    "ORDER BY score DESC LIMIT $lim",
                    q=query,
                    lim=limit,
                )
            ]

    def entity_neighbors(self, entity_id: str, limit: int = 20) -> list[dict]:
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "MATCH (e:Entity {id: $eid})-[r]-(neighbor:Entity) "
                    "RETURN neighbor.id AS id, neighbor.name AS name, "
                    "neighbor.entity_type AS entity_type, "
                    "type(r) AS relation_type, "
                    "startNode(r).id AS from_id, endNode(r).id AS to_id, "
                    "r.evidence AS evidence, r.confidence AS confidence "
                    "ORDER BY r.confidence DESC LIMIT $lim",
                    eid=entity_id,
                    lim=limit,
                )
            ]

    def get_entities_for_chunk(self, chunk_id: str) -> list[dict]:
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "MATCH (e:Entity)-[:MENTIONED_IN]->(ch:Chunk {id: $cid}) "
                    "RETURN e.id AS id, e.name AS name, "
                    "e.entity_type AS entity_type, "
                    "e.description AS description",
                    cid=chunk_id,
                )
            ]

    def get_chunks_for_entity(self, entity_id: str) -> list[dict]:
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "MATCH (e:Entity {id: $eid})-[:MENTIONED_IN]->(ch:Chunk)"
                    "-[:PART_OF]->(d:Document) "
                    "RETURN ch.id AS id, ch.text AS text, "
                    "d.path AS doc_path, d.title AS doc_title "
                    "ORDER BY d.path, ch.position",
                    eid=entity_id,
                )
            ]

    def get_unextracted_docs(self, corpus_id: str) -> list[dict]:
        """List documents that still have at least one un-extracted chunk.

        Order is by document path for deterministic test behavior.
        """
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "MATCH (d:Document {corpus_id: $cid})<-[:PART_OF]-(ch:Chunk) "
                    "WHERE NOT exists((ch)-[:EXTRACTION_DONE]->(:Corpus {id: $cid})) "
                    "WITH d, count(ch) AS pending "
                    "WHERE pending > 0 "
                    "RETURN d.id AS doc_id, d.path AS path, pending "
                    "ORDER BY d.path",
                    cid=corpus_id,
                )
            ]

    def get_unextracted_chunks_for_doc(
        self, doc_id: str, corpus_id: str
    ) -> list[dict]:
        """All unextracted chunks for one document, ordered by position."""
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "MATCH (ch:Chunk)-[:PART_OF]->(d:Document {id: $did, corpus_id: $cid}) "
                    "WHERE NOT exists((ch)-[:EXTRACTION_DONE]->(:Corpus {id: $cid})) "
                    "RETURN ch.id AS id, ch.text AS text, "
                    "ch.heading AS heading, d.path AS doc_path "
                    "ORDER BY ch.position",
                    did=doc_id,
                    cid=corpus_id,
                )
            ]

    def mark_chunk_extracted(self, chunk_id: str, corpus_id: str) -> None:
        with self._driver.session() as s:
            s.run(
                "MATCH (ch:Chunk {id: $cid}), (co:Corpus {id: $corp}) "
                "MERGE (ch)-[:EXTRACTION_DONE]->(co)",
                cid=chunk_id,
                corp=corpus_id,
            )

    def get_all_chunk_texts(self, corpus_id: str, only_missing: bool = False) -> list[dict]:
        """Get chunk IDs and texts for a corpus.

        If only_missing=True, returns only chunks where embedding IS NULL.
        """
        q = "MATCH (ch:Chunk)-[:PART_OF]->(d:Document {corpus_id: $cid}) "
        if only_missing:
            q += "WHERE ch.embedding IS NULL "
        q += "RETURN ch.id AS id, ch.text AS text ORDER BY ch.id"
        with self._driver.session() as s:
            return [dict(r) for r in s.run(q, cid=corpus_id)]

    def get_unembedded_entities(self, corpus_id: str) -> list[dict]:
        """Get entities without embeddings."""
        with self._driver.session() as s:
            return [
                dict(r)
                for r in s.run(
                    "MATCH (e:Entity {corpus_id: $cid}) "
                    "WHERE e.embedding IS NULL "
                    "RETURN e.id AS id, e.name AS name, e.description AS desc",
                    cid=corpus_id,
                )
            ]

    def set_chunk_embeddings(self, updates: list[tuple[str, list[float]]]) -> None:
        """Bulk update chunk embeddings."""
        with self._driver.session() as s:
            s.run(
                "UNWIND $rows AS r "
                "MATCH (ch:Chunk {id: r.id}) "
                "SET ch.embedding = r.embedding",
                rows=[{"id": uid, "embedding": emb} for uid, emb in updates],
            )

    def set_entity_embeddings(self, updates: list[tuple[str, list[float]]]) -> None:
        """Bulk update entity embeddings."""
        with self._driver.session() as s:
            s.run(
                "UNWIND $rows AS r "
                "MATCH (e:Entity {id: r.id}) "
                "SET e.embedding = r.embedding",
                rows=[{"id": uid, "embedding": emb} for uid, emb in updates],
            )

    def entity_stats(self) -> dict:
        with self._driver.session() as s:
            r = s.run(
                "OPTIONAL MATCH (e:Entity) WITH count(e) AS entities "
                "OPTIONAL MATCH (:Entity)-[r]->(:Entity) "
                "WHERE type(r) <> 'MENTIONED_IN' "
                "WITH entities, count(r) AS relations "
                "RETURN entities, relations"
            ).single()
            return dict(r) if r else {}

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
