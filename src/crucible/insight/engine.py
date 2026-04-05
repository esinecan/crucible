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
from .evaluator import AdversarialEvaluator, EvalResult

# ── Defaults ────────────────────────────────────────────────

STRATEGIES = ["bridge", "outlier", "hub", "meta", "contradiction", "gap"]

DEFAULT_NOISE_PATTERNS = [
    "docs/diffs/*",
    "*.json",
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
        use_evaluator: bool = False,
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

    # ── Strategy: Bridge Discovery ──────────────────────────

    def _bridge_cycle(
        self, corpus_id: str | None = None, max_insights: int = 10
    ) -> list[Insight]:
        samples = self._get_samples(corpus_id, 240, 120)
        if len(samples) < 2:
            return []

        candidates: list[tuple[float, dict, dict]] = []
        for i, a in enumerate(samples):
            for b in samples[i + 1:]:
                if a["doc_path"] == b["doc_path"]:
                    continue
                sim = _cosine(a["embedding"], b["embedding"])
                if sim >= 0.55:
                    candidates.append((sim, a, b))

        seen_pairs: set[tuple[str, str]] = set()
        insights: list[Insight] = []

        for sim, a, b in sorted(candidates, key=lambda x: x[0], reverse=True):
            pair = tuple(sorted([a["doc_path"], b["doc_path"]]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            dist = _doc_distance(a["doc_path"], b["doc_path"])
            avg_len = (len(a["text"]) + len(b["text"])) / 2
            length_f = min(avg_len / 400, 1.0)
            score = sim * (0.5 + 0.5 * dist) * (0.6 + 0.4 * length_f)

            insight = Insight(
                id=uuid.uuid4().hex[:20],
                corpus_id=corpus_id or "",
                text=(
                    f"Bridge: [{a['doc_path']}] <-> [{b['doc_path']}] "
                    f"(sim={sim:.3f})\n\n"
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
                    "similarity": sim, "structural_distance": dist,
                },
            )
            insights.append(insight)
            if len(insights) >= max_insights:
                break

        return insights

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
        samples = self._get_samples(corpus_id, 200, 100)
        if len(samples) < 5:
            return []

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
                if sim >= 0.50:
                    connected_docs.add(other["doc_path"])
                    total_sim += sim
                    count += 1

            if not connected_docs:
                continue

            connectivity = len(connected_docs) / max(len(all_docs) - 1, 1)
            avg_connection_strength = total_sim / count if count else 0.0
            score = connectivity * (0.4 + 0.6 * avg_connection_strength)
            scored.append((score, chunk, connected_docs, avg_connection_strength))

        scored.sort(key=lambda x: x[0], reverse=True)

        seen_docs: set[str] = set()
        insights: list[Insight] = []

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
                },
            )
            insights.append(insight)
            if len(insights) >= max_insights:
                break

        return insights

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
                if sim >= 0.55:
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

            text = (
                f"Meta-bridge: [{a['strategy']}] <-> [{b['strategy']}] "
                f"(sim={sim:.3f})\n\n"
                f"--- L1 Insight A ({a['strategy']}) ---\n{a['text'][:400]}\n\n"
                f"--- L1 Insight B ({b['strategy']}) ---\n{b['text'][:400]}"
            )

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

        Uses embeddings to find same-topic pairs, then LLM to detect actual
        contradictions. Embeddings alone can't find these — contradictions
        have HIGH similarity.
        """
        samples = self._get_samples(corpus_id, 200, 100)
        if len(samples) < 2:
            return []

        # Find high-similarity cross-document pairs (same topic, different docs)
        pairs: list[tuple[float, dict, dict]] = []
        for i, a in enumerate(samples):
            for b in samples[i + 1:]:
                if a["doc_path"] == b["doc_path"]:
                    continue
                sim = _cosine(a["embedding"], b["embedding"])
                if 0.60 <= sim <= 0.90:  # Same topic but not identical
                    pairs.append((sim, a, b))

        # Sort by similarity (most topically aligned first) and check top N
        pairs.sort(key=lambda x: x[0], reverse=True)

        insights: list[Insight] = []
        seen_pairs: set[tuple[str, str]] = set()

        for sim, a, b in pairs[:30]:  # Check at most 30 pairs
            pair = tuple(sorted([a["doc_path"], b["doc_path"]]))
            if pair in seen_pairs:
                continue

            # Ask LLM to check for contradiction
            try:
                resp = self._llm(
                    _contradiction_system(self.config.domain_preamble),
                    f"Passage A (from {a['doc_path']}):\n{a['text'][:600]}\n\n"
                    f"Passage B (from {b['doc_path']}):\n{b['text'][:600]}",
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

            insight = Insight(
                id=uuid.uuid4().hex[:20],
                corpus_id=corpus_id or "",
                text=(
                    f"Contradiction between [{a['doc_path']}] and [{b['doc_path']}]\n\n"
                    f"{explanation}\n\n"
                    f"--- A ---\n{a['text'][:500]}\n\n"
                    f"--- B ---\n{b['text'][:500]}"
                ),
                strategy="contradiction",
                score=0.8,  # Contradictions are inherently high-value
                novelty=0.8,
                relevance=1.0,
                source_chunk_ids=[a["id"], b["id"]],
                context={
                    "doc_a": a["doc_path"], "doc_b": b["doc_path"],
                    "similarity": sim,
                    "explanation": explanation,
                },
            )
            insights.append(insight)
            if len(insights) >= max_insights:
                break

        return insights

    # ── Strategy: Gap Analysis (LLM) ───────────────────────

    def _gap_cycle(
        self, corpus_id: str | None = None, max_insights: int = 10
    ) -> list[Insight]:
        """Identify knowledge gaps — concepts referenced but never explained.

        Samples chunks, asks LLM what's assumed but missing. One LLM call
        per cycle, cheapest strategy.
        """
        samples = self._get_samples(corpus_id, 100, 30)
        if len(samples) < 5:
            return []

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
            if not parsed or not isinstance(parsed, list):
                return []
        except Exception:
            return []

        insights: list[Insight] = []
        for item in parsed[:max_insights]:
            if not isinstance(item, dict):
                continue
            topic = item.get("topic", "")
            referenced_in = item.get("referenced_in", "")
            why_gap = item.get("why_gap", "")

            if not topic:
                continue

            insight = Insight(
                id=uuid.uuid4().hex[:20],
                corpus_id=corpus_id or "",
                text=(
                    f"Knowledge gap: {topic}\n\n"
                    f"Referenced as: \"{referenced_in}\"\n\n"
                    f"Why this is a gap: {why_gap}"
                ),
                strategy="gap",
                score=0.7,  # Gaps are valuable but unverified
                novelty=0.7,
                relevance=0.8,
                context={
                    "topic": topic,
                    "referenced_in": referenced_in,
                    "why_gap": why_gap,
                },
            )
            insights.append(insight)

        return insights

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

    def run_cycle(
        self,
        corpus_id: str | None = None,
        max_insights: int = 10,
        strategy: str | None = None,
    ) -> tuple[str, list[Insight]]:
        """Run one exploration cycle.

        If strategy is None, Thompson Sampling selects the arm.
        Returns (strategy_used, insights).
        """
        picked = strategy or self.bandit.select()

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
                    result = self.evaluator.evaluate(insight.text)
                    insight.score = result.score
                    insight.novelty = result.novelty
                    insight.relevance = result.relevance
                    insight.context["actionability"] = result.actionability
                    insight.context["advocate"] = result.advocate_arg
                    insight.context["skeptic"] = result.skeptic_arg
                    insight.context["adversarial"] = True
                except Exception as e:
                    insight.context["eval_error"] = str(e)

        # Persist and log
        for insight in insights:
            self.graph.upsert_insight(insight)
            self._log_reward(
                strategy=picked,
                score=insight.score,
                novelty=insight.novelty,
                relevance=insight.relevance,
                insight_id=insight.id,
                layer=insight.layer,
                adversarial=bool(self.evaluator),
                context=insight.context,
            )
            self.bandit.update(picked, insight.score)

        return picked, insights

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

        bonus = answer_score * 0.3  # scaled bonus, not full score
        rewarded = 0

        for eid in evidence_ids:
            # Check if this evidence ID is an insight
            results = self.graph.cypher_read(
                f"MATCH (i:Insight {{id: '{eid}'}}) RETURN i.strategy AS strategy"
            )
            if results:
                strategy = results[0]["strategy"]
                arm = strategy.replace("_discovery", "").replace("_detection", "")
                if arm in self.bandit.alpha:
                    self.bandit.update(arm, bonus)
                    self._log_reward(
                        mode="answer_feedback",
                        strategy=arm,
                        score=bonus,
                        insight_id=eid,
                        answer_score=answer_score,
                    )
                    rewarded += 1

        return rewarded
