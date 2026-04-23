"""Regression tests for the f-string → parameterized Cypher sweep.

Covers the user-reachable callsites that previously used f-string
interpolation of IDs into Cypher (insight/engine.py:1089, 1110 and
reasoning/actions.py:197, 202). LLM-dependent callsites (SynthesizeAction,
MCTSEngine._simulate, _gather_eval_context) are reached indirectly by the
same API and out of scope here.
"""
from __future__ import annotations

import math

import pytest

from crucible.insight.engine import InsightEngine
from crucible.models import Chunk, Corpus, Document, Insight, ReasoningNode
from crucible.reasoning.actions import FollowRefAction


pytestmark = pytest.mark.integration


def _fake_vec(seed: int, dim: int = 768) -> list[float]:
    raw = [math.sin(seed + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _seed_insight(graph, iid: str = "ins-1", strategy: str = "bridge") -> None:
    graph.upsert_corpus(Corpus(id="c1", name="c"))
    graph.upsert_insight(
        Insight(
            id=iid,
            corpus_id="c1",
            text="hello",
            strategy=strategy,
            score=0.5,
            layer=1,
            embedding=_fake_vec(1),
        )
    )


def _engine(config, graph, tmp_path) -> InsightEngine:
    """InsightEngine wired to a tmp reward log, no LLM evaluator."""
    return InsightEngine(
        config,
        graph,
        reward_log=tmp_path / "rewards.jsonl",
        use_evaluator=False,
    )


class TestApplyHumanFeedback:
    """engine.py:1109 — MCP-exposed path; insight_id comes from the client."""

    def test_returns_strategy_on_hit(self, graph, config, tmp_path):
        _seed_insight(graph)
        engine = _engine(config, graph, tmp_path)
        assert engine.apply_human_feedback("ins-1", 0.8) == "bridge"

    def test_returns_none_on_miss(self, graph, config, tmp_path):
        _seed_insight(graph)
        engine = _engine(config, graph, tmp_path)
        assert engine.apply_human_feedback("no-such-id", 0.8) is None

    @pytest.mark.parametrize("raw,clamped", [(5.0, 1.0), (-3.0, 0.0), (0.5, 0.5)])
    def test_score_is_clamped(self, graph, config, tmp_path, raw, clamped):
        """Scores outside [0,1] are clamped, not rejected — Bernoulli needs a
        probability.
        """
        _seed_insight(graph)
        engine = _engine(config, graph, tmp_path)
        before = engine.bandit.alpha["bridge"], engine.bandit.beta["bridge"]
        arm = engine.apply_human_feedback("ins-1", raw)
        after = engine.bandit.alpha["bridge"], engine.bandit.beta["bridge"]
        assert arm == "bridge"
        # The update should still have fired (3x weight -> 3 updates)
        assert (after[0] - before[0]) + (after[1] - before[1]) == 3

    def test_injection_payload_matches_nothing(self, graph, config, tmp_path):
        """With the old f-string, a closing `'` + trailing MATCH could alter
        the query. With parameter binding, the whole string is treated as a
        node id literal — no match, no write.
        """
        _seed_insight(graph)
        engine = _engine(config, graph, tmp_path)
        payload = "ins-1'}) MATCH (n) DETACH DELETE n //"
        assert engine.apply_human_feedback(payload, 0.8) is None
        # Sanity: ins-1 still exists (no DELETE fired)
        rows = graph.cypher_read("MATCH (i:Insight) RETURN i.id AS id")
        assert rows == [{"id": "ins-1"}]


class TestFeedbackFromAnswer:
    """engine.py:1088 — called from `crucible answer --feedback`."""

    def test_rewards_known_insights_only(self, graph, config, tmp_path):
        _seed_insight(graph)
        engine = _engine(config, graph, tmp_path)
        count = engine.feedback_from_answer(
            ["ins-1", "chunk-id-not-an-insight"], answer_score=0.8
        )
        assert count == 1

    def test_skips_low_scoring_answers(self, graph, config, tmp_path):
        _seed_insight(graph)
        engine = _engine(config, graph, tmp_path)
        assert engine.feedback_from_answer(["ins-1"], answer_score=0.3) == 0

    def test_empty_evidence_ids(self, graph, config, tmp_path):
        _seed_insight(graph)
        engine = _engine(config, graph, tmp_path)
        assert engine.feedback_from_answer([], answer_score=0.9) == 0


class TestFollowRefActionTraversal:
    """actions.py:197, 202 — graph-level traversal from evidence chunks."""

    def _seed_sequential_doc(self, graph):
        graph.upsert_corpus(Corpus(id="c1", name="c"))
        graph.upsert_document(
            Document(
                id="d1",
                corpus_id="c1",
                path="d.md",
                title="d",
                source_type="md",
                content_hash="h",
            )
        )
        chunks = [
            Chunk(
                id=f"ch-{i}",
                document_id="d1",
                text=f"chunk {i}",
                position=i,
                embedding=_fake_vec(i),
            )
            for i in range(3)
        ]
        graph.upsert_chunks(chunks)
        graph.link_sequential("d1")

    def test_follows_next_from_evidence(self, graph, config):
        self._seed_sequential_doc(graph)
        parent = ReasoningNode(
            id="p",
            tree_id="t",
            query="q",
            action_taken="root",
            evidence_ids=["ch-0"],
        )
        child = FollowRefAction().expand(parent, graph, config)
        assert child is not None
        new_ids = set(child.evidence_ids) - {"ch-0"}
        assert "ch-1" in new_ids

    def test_no_evidence_returns_none(self, graph, config):
        parent = ReasoningNode(
            id="p", tree_id="t", query="q", action_taken="root",
        )
        assert FollowRefAction().expand(parent, graph, config) is None

    def test_injection_payload_is_inert(self, graph, config):
        """Malicious eid shouldn't smuggle Cypher through actions.py:197/202."""
        self._seed_sequential_doc(graph)
        evil = "ch-0'}) DETACH DELETE n //"
        parent = ReasoningNode(
            id="p",
            tree_id="t",
            query="q",
            action_taken="root",
            evidence_ids=[evil],
        )
        # Should either return None (no match) or a child with zero new
        # evidence — never delete the seed data.
        result = FollowRefAction().expand(parent, graph, config)
        # Main check: data still intact.
        rows = graph.cypher_read(
            "MATCH (ch:Chunk) RETURN count(ch) AS c"
        )
        assert rows[0]["c"] == 3
