"""Insight engine — multi-strategy exploration with Thompson Sampling."""
from __future__ import annotations

import json
import math
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path

import httpx

from ..config import Config
from ..embeddings import embed_batch, embed_text
from ..graph.client import CrucibleGraph
from ..models import Insight
from .evaluator import AdversarialEvaluator, EvalContext, EvalResult

# ── Defaults ────────────────────────────────────────────────

STRATEGIES = ["bridge", "outlier", "hub", "meta", "contradiction", "gap"]

DEFAULT_NOISE_PATTERNS = [
    "docs/diffs/*",
    "*.json",
    # Dev-environment clutter that slips into corpora when a repo is ingested
    ".history/*",
    ".lh/*",
    ".venv/*",
    "node_modules/*",
    "__pycache__/*",
    "dist/*",
    "build/*",
    ".git/*",
]

MIN_CHUNK_LENGTH = 80

# ── LLM Prompt Templates (domain injected at runtime) ──────

def _synthesize_system(domain: str) -> str:
    return domain + (
        "You are a knowledge analyst. Given raw data about a connection found "
        "in a knowledge base, explain in 2-3 sentences what this connection "
        "means and why it matters. Be specific about what a reader would learn "
        "from this that they couldn't see from either source alone. "
        "Do not restate the raw text — interpret it."
    )

def _contradiction_system(domain: str) -> str:
    return domain + (
        "Two passages from a knowledge base discuss similar topics. "
        "Determine if they contain a genuine contradiction — incompatible facts, "
        "claims, or positions about the same entity, event, or process.\n\n"
        "Surface-level topic overlap is NOT a contradiction. Different perspectives "
        "are NOT contradictions unless they assert incompatible facts.\n\n"
        'Respond with JSON only: {"contradiction": true/false, "explanation": "one sentence if true, empty string if false"}'
    )

def _gap_system(domain: str) -> str:
    return domain + (
        "You are reviewing a sample of documents from a knowledge base. "
        "Identify 3-5 concepts, entities, or phenomena that are REFERENCED or "
        "ASSUMED but never fully explained. These are knowledge gaps — things "
        "a new reader would need to look up elsewhere.\n\n"
        "Focus on domain-specific gaps (not generic common knowledge).\n\n"
        "Respond with JSON array only:\n"
        '[{"topic": "...", "referenced_in": "brief quote showing the reference", '
        '"why_gap": "one sentence on why this is a gap"}]'
    )


# ── Utilities ───────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _is_noisy(path: str, patterns: list[str]) -> bool:
    return any(fnmatch(path, p) for p in patterns)


def _doc_distance(path_a: str, path_b: str) -> float:
    parts_a = Path(path_a).parts
    parts_b = Path(path_b).parts
    shared = 0
    for a, b in zip(parts_a, parts_b):
        if a == b:
            shared += 1
        else:
            break
    total = max(len(parts_a), len(parts_b))
    return 1.0 - (shared / total) if total > 0 else 0.0


def _parse_json_lenient(text: str) -> dict | list | None:
    """Parse JSON from LLM output, tolerant of markdown fences and preamble."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding JSON object or array
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Thompson Sampler ────────────────────────────────────────

class ThompsonBandit:
    """Beta-Bernoulli Thompson Sampling over strategy arms."""

    def __init__(self, arms: list[str]):
        self.alpha: dict[str, float] = {a: 1.0 for a in arms}
        self.beta: dict[str, float] = {a: 1.0 for a in arms}

    def select(self) -> str:
        samples = {
            arm: random.betavariate(self.alpha[arm], self.beta[arm])
            for arm in self.alpha
        }
        return max(samples, key=samples.get)

    def update(self, arm: str, score: float) -> None:
        if random.random() < score:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def load_from_log(self, path: Path) -> None:
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            arm = entry.get("strategy", "")
            arm_name = arm.replace("_discovery", "").replace("_detection", "")
            if arm_name in self.alpha:
                self.update(arm_name, entry.get("score", 0.5))

    def posteriors(self) -> dict[str, dict]:
        return {
            arm: {
                "alpha": self.alpha[arm],
                "beta": self.beta[arm],
                "mean": self.alpha[arm] / (self.alpha[arm] + self.beta[arm]),
                "observations": self.alpha[arm] + self.beta[arm] - 2,
            }
            for arm in self.alpha
        }


# ── Insight Engine ──────────────────────────────────────────

class InsightEngine:
    def __init__(
        self,
        config: Config,
        graph: CrucibleGraph,
        reward_log: str | Path | None = None,
        noise_patterns: list[str] | None = None,
        use_evaluator: bool = True,
    ):
        self.config = config
        self.graph = graph
        self.reward_log = Path(reward_log) if reward_log else Path("insight_rewards.jsonl")
        self.noise_patterns = noise_patterns if noise_patterns is not None else DEFAULT_NOISE_PATTERNS
        self.bandit = ThompsonBandit(STRATEGIES)
        self.bandit.load_from_log(self.reward_log)
        self.evaluator = AdversarialEvaluator(config) if use_evaluator else None

    # ── Shared LLM Client ──────────────────────────────────

    def _llm(self, system: str, user: str, max_tokens: int = 400) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("CRUCIBLE_EVAL_MODEL", "deepseek-chat")
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.7,
                "max_tokens": max_tokens,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _get_samples(
        self, corpus_id: str | None, raw_count: int, final_count: int
    ) -> list[dict]:
        raw = self.graph.sample_chunks(n=raw_count, corpus_id=corpus_id)
        filtered = [
            s for s in raw
            if not _is_noisy(s["doc_path"], self.noise_patterns)
            and len(s["text"]) >= MIN_CHUNK_LENGTH
        ]
        return filtered[:final_count]

    # ── Synthesis ───────────────────────────────────────────

    def _synthesize(self, insight: Insight) -> str:
        """Replace template insight text with LLM-generated explanation."""
        resp = self._llm(
            _synthesize_system(self.config.domain_preamble),
            f"Raw insight data:\n\n{insight.text}",
            max_tokens=250,
        )
        return f"[{insight.strategy}] {resp}\n\n---\n{insight.text}"

    # ── Harness Context Gathering ──────────────────────────

    def _gather_eval_context(self, insight: Insight) -> EvalContext:
        """Gather graph context for evaluator agents. Zero LLM cost."""
        source_chunks: list[str] = []
        source_entities: list[str] = []
        entity_paths: list[str] = []
        related_insights: list[str] = []
        seen_entities: set[str] = set()

        # 1. Source chunk texts
        for cid in insight.source_chunk_ids[:5]:
            results = self.graph.cypher_read(
                "MATCH (c:Chunk {id: $cid}) RETURN c.text AS text LIMIT 1",
                cid=cid,
            )
            if results and results[0].get("text"):
                source_chunks.append(results[0]["text"][:500])

        # 2. Entities from source chunks + their 1-hop paths
        for cid in insight.source_chunk_ids[:5]:
            try:
                entities = self.graph.get_entities_for_chunk(cid)
                for ent in entities[:5]:
                    ent_key = ent["name"].lower()
                    if ent_key not in seen_entities:
                        seen_entities.add(ent_key)
                        source_entities.append(f"{ent['name']} ({ent.get('entity_type', 'UNKNOWN')})")
                        # 1-hop entity paths
                        try:
                            neighbors = self.graph.entity_neighbors(ent["id"], limit=3)
                            for rel in neighbors:
                                entity_paths.append(
                                    f"{ent['name']} -[{rel['relation_type']}]-> {rel['name']}"
                                )
                        except Exception:
                            pass
            except Exception:
                pass  # no entities yet

        # 3. Related existing insights via embedding similarity
        if insight.embedding:
            try:
                similar = self.graph.vector_search_insights(insight.embedding, limit=5)
                for s in similar:
                    if s["id"] != insight.id and s.get("score", 0) >= 0.65:
                        related_insights.append(s.get("text", "")[:300])
                        if len(related_insights) >= 3:
                            break
            except Exception:
                pass

        return EvalContext(
            source_chunks=source_chunks,
            source_entities=source_entities[:10],
            related_insights=related_insights,
            entity_paths=entity_paths[:10],
            strategy=insight.strategy,
        )

    # ── Strategy: Bridge Discovery ──────────────────────────

    def _bridge_cycle(
        self, corpus_id: str | None = None, max_insights: int = 10
    ) -> list[Insight]:
        # Embedding-based bridges
        samples = self._get_samples(corpus_id, 240, 120)
        candidates: list[tuple[float, dict, dict, str]] = []  # (score, a, b, source)

        if len(samples) >= 2:
            for i, a in enumerate(samples):
                for b in samples[i + 1:]:
                    if a["doc_path"] == b["doc_path"]:
                        continue
                    sim = _cosine(a["embedding"], b["embedding"])
                    if sim >= 0.65:
                        dist = _doc_distance(a["doc_path"], b["doc_path"])
                        avg_len = (len(a["text"]) + len(b["text"])) / 2
                        length_f = min(avg_len / 400, 1.0)
                        score = sim * (0.5 + 0.5 * dist) * (0.6 + 0.4 * length_f)
                        candidates.append((score, a, b, "embedding"))

        # Entity co-occurrence bridges: same entity mentioned in different documents
        try:
            entity_bridges = self._entity_bridge_candidates(corpus_id)
            candidates.extend(entity_bridges)
        except Exception:
            pass  # no entities yet, graceful degradation

        seen_pairs: set[tuple[str, str]] = set()
        insights: list[Insight] = []

        for score, a, b, source in sorted(candidates, key=lambda x: x[0], reverse=True):
            pair = tuple(sorted([a["doc_path"], b["doc_path"]]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            entity_note = ""
            if source == "entity":
                entity_note = f"\nBridge entity: {a.get('bridge_entity', '?')}\n"

            insight = Insight(
                id=uuid.uuid4().hex[:20],
                corpus_id=corpus_id or "",
                text=(
                    f"Bridge: [{a['doc_path']}] <-> [{b['doc_path']}] "
                    f"(score={score:.3f}, via {source})\n"
                    f"{entity_note}\n"
                    f"--- A ---\n{a['text'][:500]}\n\n"
                    f"--- B ---\n{b['text'][:500]}"
                ),
                strategy="bridge",
                score=score,
                novelty=score,
                relevance=1.0,
                source_chunk_ids=[a["id"], b["id"]],
                context={
                    "doc_a": a["doc_path"], "doc_b": b["doc_path"],
                    "source": source,
                    "bridge_entity": a.get("bridge_entity", ""),
                },
            )
            insights.append(insight)
            if len(insights) >= max_insights:
                break

        return insights

    def _entity_bridge_candidates(
        self, corpus_id: str | None
    ) -> list[tuple[float, dict, dict, str]]:
        """Find cross-document bridges via shared entities."""
        with self.graph._driver.session() as s:
            rows = [
                dict(r)
                for r in s.run(
                    "MATCH (c1:Chunk)<-[:MENTIONED_IN]-(e:Entity)-[:MENTIONED_IN]->(c2:Chunk) "
                    "MATCH (c1)-[:PART_OF]->(d1:Document), (c2)-[:PART_OF]->(d2:Document) "
                    "WHERE d1 <> d2 AND c1.id < c2.id "
                    "WITH e, c1, c2, d1, d2, rand() AS r "
                    "ORDER BY r LIMIT 50 "
                    "RETURN e.name AS entity, c1.id AS c1_id, c1.text AS c1_text, "
                    "d1.path AS d1_path, c2.id AS c2_id, c2.text AS c2_text, "
                    "d2.path AS d2_path, size(e.source_chunks) AS mc"
                )
            ]

        candidates = []
        for row in rows:
            # Higher mention_count = more central entity = stronger bridge signal
            mc_factor = min(row["mc"] / 10, 1.0)
            dist = _doc_distance(row["d1_path"], row["d2_path"])
            score = 0.7 * (0.5 + 0.5 * dist) * (0.5 + 0.5 * mc_factor)

            a = {"id": row["c1_id"], "text": row["c1_text"],
                 "doc_path": row["d1_path"], "bridge_entity": row["entity"]}
            b = {"id": row["c2_id"], "text": row["c2_text"],
                 "doc_path": row["d2_path"], "bridge_entity": row["entity"]}
            candidates.append((score, a, b, "entity"))

        return candidates

    # ── Strategy: Outlier Detection ─────────────────────────

    def _outlier_cycle(
        self, corpus_id: str | None = None, max_insights: int = 10
    ) -> list[Insight]:
        samples = self._get_samples(corpus_id, 200, 100)
        if len(samples) < 5:
            return []

        scored: list[tuple[float, dict, float]] = []
        for chunk in samples:
            sims = [
                _cosine(chunk["embedding"], other["embedding"])
                for other in samples
                if other["id"] != chunk["id"]
            ]
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            isolation = 1.0 - avg_sim
            length_f = min(len(chunk["text"]) / 400, 1.0)
            score = isolation * (0.5 + 0.5 * length_f)
            scored.append((score, chunk, avg_sim))

        scored.sort(key=lambda x: x[0], reverse=True)

        seen_docs: set[str] = set()
        insights: list[Insight] = []

        for score, chunk, avg_sim in scored:
            if chunk["doc_path"] in seen_docs:
                continue
            seen_docs.add(chunk["doc_path"])

            insight = Insight(
                id=uuid.uuid4().hex[:20],
                corpus_id=corpus_id or "",
                text=(
                    f"Outlier in [{chunk['doc_path']}] "
                    f"(isolation={score:.3f}, avg_sim={avg_sim:.3f})\n\n"
                    f"This chunk is unlike most other content in the corpus:\n\n"
                    f"{chunk['text'][:600]}"
                ),
                strategy="outlier",
                score=score,
                novelty=score,
                relevance=1.0,
                source_chunk_ids=[chunk["id"]],
                context={
                    "doc_path": chunk["doc_path"],
                    "avg_similarity": avg_sim,
                    "isolation": 1.0 - avg_sim,
                },
            )
            insights.append(insight)
            if len(insights) >= max_insights:
                break

        return insights

    # ── Strategy: Hub Detection ─────────────────────────────

    def _hub_cycle(
        self, corpus_id: str | None = None, max_insights: int = 10
    ) -> list[Insight]:
        insights: list[Insight] = []

        # Entity-based hub detection: entities spanning many documents
        try:
            entity_hubs = self._entity_hub_candidates(corpus_id)
            for score, entity_name, entity_type, doc_count, docs, chunk_id, chunk_text in entity_hubs:
                if len(insights) >= max_insights // 2:
                    break
                top_docs = docs[:8]
                insight = Insight(
                    id=uuid.uuid4().hex[:20],
                    corpus_id=corpus_id or "",
                    text=(
                        f"Hub entity: {entity_name} ({entity_type}) "
                        f"(score={score:.3f}, spans {doc_count} docs)\n\n"
                        f"Appears in: {', '.join(top_docs)}\n\n"
                        f"Representative content:\n{chunk_text[:500]}"
                    ),
                    strategy="hub",
                    score=score,
                    novelty=score,
                    relevance=1.0,
                    source_chunk_ids=[chunk_id] if chunk_id else [],
                    context={
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                        "doc_count": doc_count,
                        "top_docs": top_docs,
                        "source": "entity",
                    },
                )
                insights.append(insight)
        except Exception:
            pass  # no entities yet

        # Embedding-based hub detection (fills remaining slots)
        remaining = max_insights - len(insights)
        if remaining > 0:
            samples = self._get_samples(corpus_id, 200, 100)
            if len(samples) >= 5:
                all_docs = set(s["doc_path"] for s in samples)
                scored: list[tuple[float, dict, set, float]] = []
                for chunk in samples:
                    connected_docs: set[str] = set()
                    total_sim = 0.0
                    count = 0
                    for other in samples:
                        if other["doc_path"] == chunk["doc_path"]:
                            continue
                        sim = _cosine(chunk["embedding"], other["embedding"])
                        if sim >= 0.65:
                            connected_docs.add(other["doc_path"])
                            total_sim += sim
                            count += 1
                    if not connected_docs:
                        continue
                    connectivity = len(connected_docs) / max(len(all_docs) - 1, 1)
                    avg_strength = total_sim / count if count else 0.0
                    score = connectivity * (0.4 + 0.6 * avg_strength)
                    scored.append((score, chunk, connected_docs, avg_strength))

                scored.sort(key=lambda x: x[0], reverse=True)
                seen_docs: set[str] = set()

                for score, chunk, connected_docs, avg_strength in scored:
                    if chunk["doc_path"] in seen_docs:
                        continue
                    seen_docs.add(chunk["doc_path"])
                    top_connections = sorted(connected_docs)[:8]
                    insight = Insight(
                        id=uuid.uuid4().hex[:20],
                        corpus_id=corpus_id or "",
                        text=(
                            f"Hub in [{chunk['doc_path']}] "
                            f"(connectivity={score:.3f}, reaches {len(connected_docs)} docs)\n\n"
                            f"Connects to: {', '.join(top_connections)}\n\n"
                            f"Core content:\n{chunk['text'][:500]}"
                        ),
                        strategy="hub",
                        score=score,
                        novelty=score,
                        relevance=1.0,
                        source_chunk_ids=[chunk["id"]],
                        context={
                            "doc_path": chunk["doc_path"],
                            "connected_doc_count": len(connected_docs),
                            "avg_connection_strength": avg_strength,
                            "top_connections": top_connections,
                            "source": "embedding",
                        },
                    )
                    insights.append(insight)
                    if len(insights) >= max_insights:
                        break

        return insights

    def _entity_hub_candidates(
        self, corpus_id: str | None
    ) -> list[tuple[float, str, str, int, list[str], str, str]]:
        """Find hub entities that span many documents."""
        with self.graph._driver.session() as s:
            rows = [
                dict(r)
                for r in s.run(
                    "MATCH (e:Entity)-[:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(d:Document) "
                    "WITH e, count(DISTINCT d) AS doc_count, "
                    "collect(DISTINCT d.path) AS docs, "
                    "collect(ch.id)[0] AS sample_chunk_id, "
                    "collect(ch.text)[0] AS sample_text "
                    "WHERE doc_count >= 3 "
                    "RETURN e.name AS name, e.entity_type AS type, "
                    "doc_count, docs, sample_chunk_id, sample_text "
                    "ORDER BY doc_count DESC LIMIT 20"
                )
            ]

        total_docs = max(len(set(d for r in rows for d in r["docs"])), 1)
        results = []
        for row in rows:
            connectivity = row["doc_count"] / total_docs
            score = min(connectivity * 1.5, 1.0)
            results.append((
                score, row["name"], row["type"], row["doc_count"],
                row["docs"], row["sample_chunk_id"], row["sample_text"],
            ))
        return results

    # ── Strategy: Meta-Bridge (L1 → L2) ────────────────────

    def _meta_cycle(
        self, corpus_id: str | None = None, max_insights: int = 10
    ) -> list[Insight]:
        samples = self.graph.sample_insights(n=60, layer=1)
        if len(samples) < 2:
            return []

        candidates: list[tuple[float, dict, dict]] = []
        for i, a in enumerate(samples):
            for b in samples[i + 1:]:
                if a["strategy"] == b["strategy"]:
                    continue
                sim = _cosine(a["embedding"], b["embedding"])
                if sim >= 0.65:
                    candidates.append((sim, a, b))

        seen_pairs: set[tuple[str, str]] = set()
        insights: list[Insight] = []

        for sim, a, b in sorted(candidates, key=lambda x: x[0], reverse=True):
            pair = tuple(sorted([a["id"], b["id"]]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            strategy_bonus = 1.0 if a["strategy"] != b["strategy"] else 0.7
            avg_original = (a["original_score"] + b["original_score"]) / 2
            score = sim * strategy_bonus * (0.5 + 0.5 * avg_original)

            # Build text with eval context from L1 insights
            text = (
                f"Meta-bridge: [{a['strategy']}] <-> [{b['strategy']}] "
                f"(sim={sim:.3f})\n\n"
                f"--- L1 Insight A ({a['strategy']}) ---\n{a['text'][:400]}\n"
            )
            a_ctx = a.get("context", {})
            if a_ctx.get("advocate"):
                text += f"  Advocate: {a_ctx['advocate'][:200]}\n"
                text += f"  Skeptic: {a_ctx['skeptic'][:200]}\n"

            text += f"\n--- L1 Insight B ({b['strategy']}) ---\n{b['text'][:400]}\n"
            b_ctx = b.get("context", {})
            if b_ctx.get("advocate"):
                text += f"  Advocate: {b_ctx['advocate'][:200]}\n"
                text += f"  Skeptic: {b_ctx['skeptic'][:200]}\n"

            # Entity overlap between source chunks
            shared_entities: set[str] = set()
            try:
                a_ents: set[str] = set()
                for cid in a.get("source_chunk_ids", [])[:3]:
                    for ent in self.graph.get_entities_for_chunk(cid):
                        a_ents.add(ent["name"])
                b_ents: set[str] = set()
                for cid in b.get("source_chunk_ids", [])[:3]:
                    for ent in self.graph.get_entities_for_chunk(cid):
                        b_ents.add(ent["name"])
                shared_entities = a_ents & b_ents
                if shared_entities:
                    text += f"\nShared entities: {', '.join(list(shared_entities)[:10])}\n"
            except Exception:
                pass

            emb = embed_text(text, self.config)
            insight = Insight(
                id=uuid.uuid4().hex[:20],
                corpus_id=corpus_id or "",
                text=text,
                strategy="meta",
                score=score,
                novelty=score,
                relevance=1.0,
                layer=2,
                embedding=emb,
                source_insight_ids=[a["id"], b["id"]],
                context={
                    "insight_a_id": a["id"],
                    "insight_a_strategy": a["strategy"],
                    "insight_b_id": b["id"],
                    "insight_b_strategy": b["strategy"],
                    "similarity": sim,
                    "shared_entities": list(shared_entities)[:10],
                    "has_eval_context": bool(a_ctx.get("advocate") or b_ctx.get("advocate")),
                },
            )
            insights.append(insight)
            if len(insights) >= max_insights:
                break

        return insights

    # ── Strategy: Contradiction Detection (LLM) ────────────

    def _contradiction_cycle(
        self, corpus_id: str | None = None, max_insights: int = 10
    ) -> list[Insight]:
        """Find chunks that discuss the same topic but assert incompatible facts.

        Two candidate sources, both verified by LLM:
        1. Entity pre-filter: chunks sharing an entity across different docs.
           Much more targeted than embedding similarity — same entity = same topic.
        2. Embedding similarity: high-similarity cross-doc pairs (original approach).
        """
        pairs: list[tuple[float, dict, dict, str]] = []  # (priority, a, b, source)

        # Channel 1: entity-sharing pairs (higher priority — same entity = same topic)
        try:
            entity_pairs = self._entity_contradiction_candidates(corpus_id)
            pairs.extend(entity_pairs)
        except Exception:
            pass

        # Channel 2: embedding-similarity pairs (fallback)
        samples = self._get_samples(corpus_id, 200, 100)
        if len(samples) >= 2:
            for i, a in enumerate(samples):
                for b in samples[i + 1:]:
                    if a["doc_path"] == b["doc_path"]:
                        continue
                    sim = _cosine(a["embedding"], b["embedding"])
                    if 0.65 <= sim <= 0.85:
                        pairs.append((sim, a, b, "embedding"))

        pairs.sort(key=lambda x: x[0], reverse=True)

        insights: list[Insight] = []
        seen_pairs: set[tuple[str, str]] = set()

        for priority, a, b, source in pairs[:30]:
            pair = tuple(sorted([a.get("doc_path", a.get("id", "")),
                                 b.get("doc_path", b.get("id", ""))]))
            if pair in seen_pairs:
                continue

            try:
                resp = self._llm(
                    _contradiction_system(self.config.domain_preamble),
                    f"Passage A (from {a.get('doc_path', '?')}):\n{a['text'][:600]}\n\n"
                    f"Passage B (from {b.get('doc_path', '?')}):\n{b['text'][:600]}",
                    max_tokens=200,
                )
                parsed = _parse_json_lenient(resp)
                if not parsed or not isinstance(parsed, dict):
                    continue
                if not parsed.get("contradiction", False):
                    continue
            except Exception:
                continue

            seen_pairs.add(pair)
            explanation = parsed.get("explanation", "")
            entity_note = f" (shared entity: {a.get('shared_entity', '?')})" if source == "entity" else ""

            insight = Insight(
                id=uuid.uuid4().hex[:20],
                corpus_id=corpus_id or "",
                text=(
                    f"Contradiction between [{a.get('doc_path', '?')}] and "
                    f"[{b.get('doc_path', '?')}]{entity_note}\n\n"
                    f"{explanation}\n\n"
                    f"--- A ---\n{a['text'][:500]}\n\n"
                    f"--- B ---\n{b['text'][:500]}"
                ),
                strategy="contradiction",
                score=0.8,
                novelty=0.8,
                relevance=1.0,
                source_chunk_ids=[a["id"], b["id"]],
                context={
                    "doc_a": a.get("doc_path", ""), "doc_b": b.get("doc_path", ""),
                    "source": source,
                    "shared_entity": a.get("shared_entity", ""),
                    "explanation": explanation,
                },
            )
            insights.append(insight)
            if len(insights) >= max_insights:
                break

        return insights

    def _entity_contradiction_candidates(
        self, corpus_id: str | None
    ) -> list[tuple[float, dict, dict, str]]:
        """Find cross-doc chunk pairs that share an entity — contradiction candidates."""
        with self.graph._driver.session() as s:
            rows = [
                dict(r)
                for r in s.run(
                    "MATCH (c1:Chunk)<-[:MENTIONED_IN]-(e:Entity)-[:MENTIONED_IN]->(c2:Chunk) "
                    "MATCH (c1)-[:PART_OF]->(d1:Document), (c2)-[:PART_OF]->(d2:Document) "
                    "WHERE d1 <> d2 AND c1.id < c2.id "
                    "WITH e, c1, c2, d1, d2, rand() AS r "
                    "ORDER BY r LIMIT 30 "
                    "RETURN e.name AS entity, c1.id AS c1_id, c1.text AS c1_text, "
                    "d1.path AS d1_path, c2.id AS c2_id, c2.text AS c2_text, "
                    "d2.path AS d2_path"
                )
            ]

        candidates = []
        for row in rows:
            a = {"id": row["c1_id"], "text": row["c1_text"],
                 "doc_path": row["d1_path"], "shared_entity": row["entity"]}
            b = {"id": row["c2_id"], "text": row["c2_text"],
                 "doc_path": row["d2_path"], "shared_entity": row["entity"]}
            # Entity pairs get priority boost over embedding pairs
            candidates.append((0.85, a, b, "entity"))

        return candidates

    # ── Strategy: Gap Analysis (LLM + Entity Graph) ─────────

    def _gap_cycle(
        self, corpus_id: str | None = None, max_insights: int = 10
    ) -> list[Insight]:
        """Identify knowledge gaps via entity graph structure + LLM.

        Two channels:
        1. Entity graph: entities with many incoming relations but few
           MENTIONED_IN edges = referenced by others but poorly covered.
           Zero LLM cost.
        2. LLM: samples chunks, asks what's assumed but missing.
           One LLM call per cycle.
        """
        insights: list[Insight] = []

        # Channel 1: graph-structural gaps (zero LLM cost)
        try:
            graph_gaps = self._entity_gap_candidates(corpus_id)
            for score, name, etype, incoming, mentions in graph_gaps:
                if len(insights) >= max_insights // 2:
                    break
                insight = Insight(
                    id=uuid.uuid4().hex[:20],
                    corpus_id=corpus_id or "",
                    text=(
                        f"Knowledge gap (graph): {name} ({etype})\n\n"
                        f"Referenced by {incoming} other entities but only "
                        f"directly mentioned in {mentions} chunks.\n\n"
                        f"This entity is talked ABOUT but not talked about directly."
                    ),
                    strategy="gap",
                    score=score,
                    novelty=score,
                    relevance=0.9,
                    context={
                        "entity_name": name,
                        "entity_type": etype,
                        "incoming_relations": incoming,
                        "mention_count": mentions,
                        "source": "entity_graph",
                    },
                )
                insights.append(insight)
        except Exception:
            pass  # no entities yet

        # Channel 2: LLM-based gap detection (fills remaining slots)
        remaining = max_insights - len(insights)
        if remaining > 0:
            samples = self._get_samples(corpus_id, 100, 30)
            if len(samples) >= 5:
                sample_text = "\n\n---\n\n".join(
                    f"[{s['doc_path']}]\n{s['text'][:300]}" for s in samples
                )
                try:
                    resp = self._llm(
                        _gap_system(self.config.domain_preamble),
                        f"Knowledge base sample:\n\n{sample_text}",
                        max_tokens=500,
                    )
                    parsed = _parse_json_lenient(resp)
                    if parsed and isinstance(parsed, list):
                        for item in parsed[:remaining]:
                            if not isinstance(item, dict):
                                continue
                            topic = item.get("topic", "")
                            if not topic:
                                continue
                            insight = Insight(
                                id=uuid.uuid4().hex[:20],
                                corpus_id=corpus_id or "",
                                text=(
                                    f"Knowledge gap: {topic}\n\n"
                                    f"Referenced as: \"{item.get('referenced_in', '')}\"\n\n"
                                    f"Why this is a gap: {item.get('why_gap', '')}"
                                ),
                                strategy="gap",
                                score=0.7,
                                novelty=0.7,
                                relevance=0.8,
                                context={
                                    "topic": topic,
                                    "referenced_in": item.get("referenced_in", ""),
                                    "why_gap": item.get("why_gap", ""),
                                    "source": "llm",
                                },
                            )
                            insights.append(insight)
                except Exception:
                    pass

        return insights

    def _entity_gap_candidates(
        self, corpus_id: str | None
    ) -> list[tuple[float, str, str, int, int]]:
        """Find entities that are referenced by others but have sparse direct coverage."""
        with self.graph._driver.session() as s:
            rows = [
                dict(r)
                for r in s.run(
                    "MATCH (e:Entity)<-[r]-(other:Entity) "
                    "WHERE type(r) <> 'MENTIONED_IN' "
                    "WITH e, count(r) AS incoming "
                    "WHERE incoming >= 2 "
                    "OPTIONAL MATCH (e)-[:MENTIONED_IN]->(ch:Chunk) "
                    "WITH e, incoming, count(ch) AS mentions "
                    "WHERE mentions < incoming "
                    "RETURN e.name AS name, e.entity_type AS type, "
                    "incoming, mentions "
                    "ORDER BY incoming - mentions DESC LIMIT 15"
                )
            ]

        results = []
        for row in rows:
            gap_ratio = (row["incoming"] - row["mentions"]) / max(row["incoming"], 1)
            score = min(0.6 + 0.4 * gap_ratio, 1.0)
            results.append((
                score, row["name"], row["type"],
                row["incoming"], row["mentions"],
            ))
        return results

    # ── Insight Embedding ───────────────────────────────────

    def embed_insights(self) -> int:
        total = 0
        while True:
            batch = self.graph.get_unembedded_insights(batch_size=50)
            if not batch:
                break
            texts = [b["text"] for b in batch]
            embeddings = embed_batch(texts, self.config, batch_size=50)
            updates = [(b["id"], emb) for b, emb in zip(batch, embeddings)]
            self.graph.set_insight_embeddings(updates)
            total += len(updates)
        return total

    # ── Main Entry Point ────────────────────────────────────

    def _write_cycle_artifacts(
        self, output_dir: Path, stage: str, insights: list[Insight]
    ) -> None:
        """Write intermediate artifacts to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        data = []
        for ins in insights:
            entry = {
                "id": ins.id,
                "strategy": ins.strategy,
                "score": ins.score,
                "novelty": ins.novelty,
                "relevance": ins.relevance,
                "layer": ins.layer,
                "text": ins.text[:500],
                "source_chunk_ids": ins.source_chunk_ids,
                "source_insight_ids": ins.source_insight_ids,
                "context": ins.context,
                "created_at": ins.created_at,
            }
            data.append(entry)
        (output_dir / f"{stage}.json").write_text(
            json.dumps(data, indent=2, default=str)
        )

    def run_cycle(
        self,
        corpus_id: str | None = None,
        max_insights: int = 10,
        strategy: str | None = None,
        output_dir: str | Path | None = None,
        persist: bool = True,
    ) -> tuple[str, list[Insight], Path]:
        """Run one exploration cycle.

        If strategy is None, Thompson Sampling selects the arm.
        Returns (strategy_used, insights, output_path).
        """
        picked = strategy or self.bandit.select()

        # Set up output directory for cycle artifacts
        if output_dir:
            cycle_dir = Path(output_dir)
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            cycle_dir = Path("output") / f"explore-cycle-{ts}"

        dispatch = {
            "bridge": self._bridge_cycle,
            "outlier": self._outlier_cycle,
            "hub": self._hub_cycle,
            "meta": self._meta_cycle,
            "contradiction": self._contradiction_cycle,
            "gap": self._gap_cycle,
        }

        if picked not in dispatch:
            raise ValueError(f"Unknown strategy: {picked}")

        insights = dispatch[picked](corpus_id, max_insights)

        # Embed L1 insights inline so they're ready for future meta cycles
        for insight in insights:
            if insight.layer == 1 and not insight.embedding:
                insight.embedding = embed_text(insight.text, self.config)

        # Artifact 1: candidates (pre-eval, heuristic scores)
        if insights:
            self._write_cycle_artifacts(cycle_dir, "candidates", insights)

        # Synthesis: LLM rewrites template text into meaningful explanation
        if self.evaluator:
            for insight in insights:
                try:
                    insight.text = self._synthesize(insight)
                except Exception:
                    pass  # keep original text

        # Adversarial evaluation: replace heuristic scores with debate scores
        if self.evaluator:
            for insight in insights:
                try:
                    ctx = self._gather_eval_context(insight)
                    result = self.evaluator.evaluate(insight.text, context=ctx)
                    insight.score = result.score
                    insight.novelty = result.novelty
                    insight.relevance = result.relevance
                    insight.context["actionability"] = result.actionability
                    insight.context["advocate"] = result.advocate_arg
                    insight.context["skeptic"] = result.skeptic_arg
                    insight.context["adversarial"] = True
                    insight.context["eval_context"] = {
                        "source_chunks": len(ctx.source_chunks),
                        "entities": ctx.source_entities[:5],
                        "entity_paths": ctx.entity_paths[:5],
                        "related_insights": len(ctx.related_insights),
                    }
                except Exception as e:
                    insight.context["eval_error"] = str(e)

        # Artifact 2: evaluated (post-eval with debate output)
        if insights:
            self._write_cycle_artifacts(cycle_dir, "evaluated", insights)

        # Persist and log
        for insight in insights:
            if persist:
                self.graph.upsert_insight(insight)
            mode = "insight" if self.evaluator else "heuristic_only"
            self._log_reward(
                mode=mode,
                strategy=picked,
                score=insight.score,
                novelty=insight.novelty,
                relevance=insight.relevance,
                insight_id=insight.id,
                layer=insight.layer,
                adversarial=bool(self.evaluator),
                context=insight.context,
            )
            # Discount heuristic scores so they don't dominate the posterior
            bandit_score = insight.score if self.evaluator else insight.score * 0.5
            self.bandit.update(picked, bandit_score)

        # Artifact 3: final persisted insights
        if insights:
            self._write_cycle_artifacts(cycle_dir, "insights", insights)

        return picked, insights, cycle_dir

    def _log_reward(self, **kwargs) -> None:
        kwargs["mode"] = kwargs.get("mode", "insight")
        kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.reward_log, "a") as f:
            f.write(json.dumps(kwargs, default=str) + "\n")

    def feedback_from_answer(self, evidence_ids: list[str], answer_score: float) -> int:
        """Bonus reward for insights that contributed to a good answer.

        Called after MCTS search completes. Insights referenced as evidence
        in high-scoring answer paths get their strategy rewarded in the bandit.
        """
        if answer_score < 0.5:
            return 0  # only reward from good answers

        if not evidence_ids:
            return 0

        bonus = answer_score * 0.3  # scaled bonus, not full score

        # One round-trip: resolve all evidence ids to any matching insights.
        results = self.graph.cypher_read(
            "UNWIND $eids AS eid "
            "MATCH (i:Insight {id: eid}) "
            "RETURN i.id AS id, i.strategy AS strategy",
            eids=evidence_ids,
        )

        rewarded = 0
        for row in results:
            strategy = row["strategy"]
            arm = strategy.replace("_discovery", "").replace("_detection", "")
            if arm in self.bandit.alpha:
                self.bandit.update(arm, bonus)
                self._log_reward(
                    mode="answer_feedback",
                    strategy=arm,
                    score=bonus,
                    insight_id=row["id"],
                    answer_score=answer_score,
                )
                rewarded += 1

        return rewarded

    def apply_human_feedback(self, insight_id: str, score: float) -> str | None:
        """Apply human feedback score to an insight. Weighted 3x in bandit.

        Score is clamped to [0.0, 1.0] — it feeds a Bernoulli bandit update.
        """
        score = max(0.0, min(1.0, float(score)))
        results = self.graph.cypher_read(
            "MATCH (i:Insight {id: $iid}) RETURN i.strategy AS strategy",
            iid=insight_id,
        )
        if not results:
            return None
        strategy = results[0]["strategy"]
        arm = strategy.replace("_discovery", "").replace("_detection", "")
        if arm not in self.bandit.alpha:
            return None

        # 3x weight: update bandit three times
        for _ in range(3):
            self.bandit.update(arm, score)

        self._log_reward(
            mode="human_feedback",
            strategy=arm,
            score=score,
            insight_id=insight_id,
        )

        # Update insight score in graph
        with self.graph._driver.session() as s:
            s.run(
                "MATCH (i:Insight {id: $iid}) SET i.human_score = $score",
                iid=insight_id, score=score,
            )

        return arm
