"""Expansion actions for MCTS reasoning tree.

Each action takes a ReasoningNode and graph, produces a child node
with new evidence or a refined answer. Actions are registered by name
and selectable by the MCTS engine.
"""
from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod

import httpx

from ..config import Config
from ..embeddings import embed_text
from ..graph.client import CrucibleGraph
from ..models import ReasoningNode


class Action(ABC):
    name: str

    @abstractmethod
    def expand(
        self, parent: ReasoningNode, graph: CrucibleGraph, config: Config
    ) -> ReasoningNode | None:
        """Produce a child node from the parent. Return None if action yields nothing."""


class SearchAction(Action):
    """Hybrid vector + fulltext + insight + entity search for evidence.

    Four search channels, deduplicated and ranked:
    1. Vector search on chunk embeddings (semantic similarity)
    2. Fulltext search on chunk text (keyword matching)
    3. Vector search on insight embeddings (discovered patterns)
    4. Entity-based retrieval (find entities in query, get their chunks)
    """

    name = "search"

    def expand(self, parent: ReasoningNode, graph: CrucibleGraph, config: Config) -> ReasoningNode | None:
        challenge = parent.context.get("challenge", "")
        if not parent.evidence_ids:
            search_text = parent.query
        elif challenge:
            search_text = f"{parent.query} {challenge[:200]}"
        else:
            search_text = parent.query

        embedding = embed_text(search_text, config)
        seen = set(parent.evidence_ids)
        scored: dict[str, tuple[float, dict]] = {}

        # Channel 1: vector search on chunks
        for r in graph.vector_search(embedding, limit=8):
            if r["id"] not in seen:
                scored[r["id"]] = (r["score"], r)

        # Channel 2: fulltext search on chunks
        for r in graph.fulltext_search(parent.query, limit=8):
            if r["id"] not in seen:
                rid = r["id"]
                if rid in scored:
                    if r["score"] > scored[rid][0]:
                        scored[rid] = (r["score"], r)
                else:
                    scored[rid] = (r["score"], r)

        # Channel 3: insight-aware search
        for r in graph.vector_search_insights(embedding, limit=5):
            if r["id"] not in seen:
                boosted = r["score"] * 1.1
                scored[r["id"]] = (boosted, r)

        # Channel 4: entity-based retrieval
        try:
            entity_hits = graph.entity_search(parent.query, limit=5)
            for hit in entity_hits:
                chunks = graph.get_chunks_for_entity(hit["id"])
                for ch in chunks[:3]:
                    if ch["id"] not in seen and ch["id"] not in scored:
                        scored[ch["id"]] = (0.8, ch)
        except Exception:
            pass  # no entities or index not ready

        if not scored:
            return None

        ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)[:10]
        new_evidence = [r["id"] for _, r in ranked]

        return ReasoningNode(
            id=uuid.uuid4().hex[:20],
            tree_id=parent.tree_id,
            query=parent.query,
            evidence_ids=parent.evidence_ids + new_evidence,
            partial_answer=parent.partial_answer,
            action_taken=self.name,
            parent_id=parent.id,
            depth=parent.depth + 1,
            context={
                "new_evidence_count": len(new_evidence),
                "search_text": search_text[:100],
                "insight_evidence": sum(1 for _, r in ranked if "strategy" in r),
                "entity_evidence": sum(1 for _, r in ranked if "doc_title" in r and r.get("doc_title")),
            },
        )


class FollowRefAction(Action):
    """Traverse graph edges from evidence to find adjacent content.

    Follows four types of edges from the most recent evidence:
    - NEXT: sequential chunks in the same document (read ahead)
    - DERIVED_FROM: if evidence is an insight, follow to its source chunks
    - Entity traversal: chunk → entities → related entities → their chunks
      This is SEMANTIC traversal — following meaning, not document structure.
    """

    name = "follow_ref"

    def expand(self, parent: ReasoningNode, graph: CrucibleGraph, config: Config) -> ReasoningNode | None:
        if not parent.evidence_ids:
            return None

        recent = parent.evidence_ids[-3:]
        new_evidence = []
        seen = set(parent.evidence_ids)
        entity_hops = []

        for eid in recent:
            # Follow NEXT edges (sequential context)
            neighbors = graph.cypher_read(
                f"MATCH (n {{id: '{eid}'}})-[:NEXT]->(next:Chunk) "
                f"RETURN next.id AS id LIMIT 2"
            )
            # Follow DERIVED_FROM (insight → source chunks)
            neighbors += graph.cypher_read(
                f"MATCH (n {{id: '{eid}'}})-[:DERIVED_FROM]->(src) "
                f"RETURN src.id AS id LIMIT 3"
            )
            for r in neighbors:
                if r["id"] and r["id"] not in seen:
                    seen.add(r["id"])
                    new_evidence.append(r["id"])

            # Entity traversal: chunk → entity → related entity → chunk
            try:
                entities = graph.get_entities_for_chunk(eid)
                for ent in entities[:2]:
                    related = graph.entity_neighbors(ent["id"], limit=3)
                    for rel in related:
                        rel_entity_id = rel["id"]
                        rel_chunks = graph.get_chunks_for_entity(rel_entity_id)
                        for ch in rel_chunks[:2]:
                            if ch["id"] not in seen:
                                seen.add(ch["id"])
                                new_evidence.append(ch["id"])
                                entity_hops.append(
                                    f"{ent['name']} -[{rel['relation_type']}]-> {rel['name']}"
                                )
            except Exception:
                pass  # no entities yet

        if not new_evidence:
            return None

        return ReasoningNode(
            id=uuid.uuid4().hex[:20],
            tree_id=parent.tree_id,
            query=parent.query,
            evidence_ids=parent.evidence_ids + new_evidence,
            partial_answer=parent.partial_answer,
            action_taken=self.name,
            parent_id=parent.id,
            depth=parent.depth + 1,
            context={
                "followed_from": recent,
                "new_count": len(new_evidence),
                "entity_hops": entity_hops[:5],
            },
        )


class ChallengeAction(Action):
    """Adversarial prosecution of the current partial answer."""

    name = "challenge"

    def expand(self, parent: ReasoningNode, graph: CrucibleGraph, config: Config) -> ReasoningNode | None:
        if not parent.partial_answer:
            return None

        challenge = self._prosecute(parent.query, parent.partial_answer, config)
        if not challenge:
            return None

        return ReasoningNode(
            id=uuid.uuid4().hex[:20],
            tree_id=parent.tree_id,
            query=parent.query,
            evidence_ids=parent.evidence_ids,
            partial_answer=parent.partial_answer,
            action_taken=self.name,
            parent_id=parent.id,
            depth=parent.depth + 1,
            context={"challenge": challenge},
        )

    def _prosecute(self, query: str, answer: str, config: Config) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("CRUCIBLE_EVAL_MODEL", "deepseek-chat")
        try:
            resp = httpx.post(
                f"{base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": config.domain_preamble + (
                                "You are a prosecutor stress-testing an answer. Find gaps, "
                                "unsupported claims, missing evidence, logical errors, and "
                                "counter-arguments. Be specific. 2-3 sentences."
                            ),
                        },
                        {"role": "user", "content": f"Query: {query}\n\nAnswer: {answer}"},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return ""


class SynthesizeAction(Action):
    """Build or refine an answer from accumulated evidence.

    Gathers text from evidence items (both chunks and insights),
    combines with the query and any active challenge, and asks the
    LLM to produce or refine a grounded answer.

    If a prior partial_answer exists and a challenge was raised,
    the LLM is instructed to address the challenge directly rather
    than starting from scratch. This is the "refine" path that
    produces progressively better answers through the MCTS tree.
    """

    name = "synthesize"

    def expand(self, parent: ReasoningNode, graph: CrucibleGraph, config: Config) -> ReasoningNode | None:
        if not parent.evidence_ids:
            return None

        # Gather evidence texts from both Chunks and Insights
        evidence_texts = []
        for eid in parent.evidence_ids[-10:]:  # last 10 evidence items
            results = graph.cypher_read(
                f"MATCH (n) WHERE n.id = '{eid}' "
                f"RETURN n.text AS text, labels(n) AS labels LIMIT 1"
            )
            if results and results[0].get("text"):
                text = results[0]["text"][:500]
                labels = results[0].get("labels", [])
                if "Insight" in (labels or []):
                    text = f"[Insight] {text}"
                evidence_texts.append(text)

        if not evidence_texts:
            return None

        answer = self._synthesize(parent.query, evidence_texts, parent.partial_answer, parent.context.get("challenge", ""), config)
        if not answer:
            return None

        return ReasoningNode(
            id=uuid.uuid4().hex[:20],
            tree_id=parent.tree_id,
            query=parent.query,
            evidence_ids=parent.evidence_ids,
            partial_answer=answer,
            action_taken=self.name,
            parent_id=parent.id,
            depth=parent.depth + 1,
            context={"evidence_count": len(evidence_texts), "refined": bool(parent.partial_answer)},
        )

    def _synthesize(self, query: str, evidence: list[str], prior_answer: str, challenge: str, config: Config) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("CRUCIBLE_EVAL_MODEL", "deepseek-chat")

        evidence_block = "\n---\n".join(evidence)
        user_msg = f"Query: {query}\n\nEvidence:\n{evidence_block}"
        if prior_answer:
            user_msg += f"\n\nPrevious answer (refine, don't restart):\n{prior_answer}"
        if challenge:
            user_msg += f"\n\nChallenge to address:\n{challenge}"

        try:
            resp = httpx.post(
                f"{base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": config.domain_preamble + (
                                "You are a knowledge analyst. Synthesize a grounded answer "
                                "from the evidence provided. Cite specific evidence. "
                                "If refining a prior answer, address the challenge directly. "
                                "Be specific and concise."
                            ),
                        },
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return ""


class EntityWalkAction(Action):
    """Pure graph traversal from query entities. No embedding, no LLM.

    Extracts entity names from the query via fulltext search on the entity
    index, then walks the entity graph following typed relations, collecting
    chunks along the path. Fast and deterministic.
    """

    name = "entity_walk"

    def expand(self, parent: ReasoningNode, graph: CrucibleGraph, config: Config) -> ReasoningNode | None:
        # Find entities mentioned in the query
        try:
            query_entities = graph.entity_search(parent.query, limit=5)
        except Exception:
            return None

        if not query_entities:
            return None

        seen = set(parent.evidence_ids)
        new_evidence = []
        walked_paths = []

        for ent in query_entities[:3]:
            # Get chunks directly mentioning this entity
            direct_chunks = graph.get_chunks_for_entity(ent["id"])
            for ch in direct_chunks[:2]:
                if ch["id"] not in seen:
                    seen.add(ch["id"])
                    new_evidence.append(ch["id"])

            # Walk one hop: entity → related entities → their chunks
            neighbors = graph.entity_neighbors(ent["id"], limit=5)
            for rel in neighbors:
                rel_chunks = graph.get_chunks_for_entity(rel["id"])
                for ch in rel_chunks[:1]:
                    if ch["id"] not in seen:
                        seen.add(ch["id"])
                        new_evidence.append(ch["id"])
                        walked_paths.append(
                            f"{ent['name']} -[{rel['relation_type']}]-> {rel['name']}"
                        )

        if not new_evidence:
            return None

        return ReasoningNode(
            id=uuid.uuid4().hex[:20],
            tree_id=parent.tree_id,
            query=parent.query,
            evidence_ids=parent.evidence_ids + new_evidence,
            partial_answer=parent.partial_answer,
            action_taken=self.name,
            parent_id=parent.id,
            depth=parent.depth + 1,
            context={
                "query_entities": [e["name"] for e in query_entities[:3]],
                "new_count": len(new_evidence),
                "walked_paths": walked_paths[:10],
            },
        )


# ── Action Registry ────────────────────────────────────────

DEFAULT_ACTIONS: list[Action] = [
    SearchAction(),
    FollowRefAction(),
    ChallengeAction(),
    SynthesizeAction(),
    EntityWalkAction(),
]

ACTION_REGISTRY: dict[str, Action] = {a.name: a for a in DEFAULT_ACTIONS}
