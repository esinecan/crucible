"""Tests for the bandits.

Thompson Sampling lives in insight/engine.py; UCB1 ActionBandit in
reasoning/mcts.py. Both are thin and deterministic once you control RNG.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from crucible.insight.engine import STRATEGIES, ThompsonBandit
from crucible.models import ReasoningNode
from crucible.reasoning.actions import SearchAction, SynthesizeAction
from crucible.reasoning.mcts import ActionBandit


class TestThompsonBandit:
    def test_init_uniform_priors(self):
        b = ThompsonBandit(["a", "b", "c"])
        for arm in ["a", "b", "c"]:
            assert b.alpha[arm] == 1.0
            assert b.beta[arm] == 1.0

    def test_update_with_score_1_biases_alpha(self):
        random.seed(0)
        b = ThompsonBandit(["x"])
        # score=1.0 means random.random() < 1.0 always → alpha bump
        for _ in range(10):
            b.update("x", 1.0)
        assert b.alpha["x"] == 11.0
        assert b.beta["x"] == 1.0

    def test_update_with_score_0_biases_beta(self):
        random.seed(0)
        b = ThompsonBandit(["x"])
        for _ in range(10):
            b.update("x", 0.0)
        assert b.alpha["x"] == 1.0
        assert b.beta["x"] == 11.0

    def test_select_returns_known_arm(self):
        random.seed(42)
        b = ThompsonBandit(STRATEGIES)
        arm = b.select()
        assert arm in STRATEGIES

    def test_posteriors_structure(self):
        b = ThompsonBandit(["x"])
        b.update("x", 1.0)
        p = b.posteriors()
        assert "x" in p
        assert set(p["x"].keys()) == {"alpha", "beta", "mean", "observations"}

    def test_load_from_log_replays(self, tmp_path):
        log = tmp_path / "rewards.jsonl"
        # Simulate a log with two updates for "bridge"
        log.write_text(
            '{"strategy": "bridge", "score": 1.0}\n'
            '{"strategy": "bridge", "score": 1.0}\n',
            encoding="utf-8",
        )
        random.seed(0)
        b = ThompsonBandit(STRATEGIES)
        b.load_from_log(log)
        # Two score=1.0 updates → alpha bumped twice (Bernoulli always lands heads)
        assert b.alpha["bridge"] == 3.0

    def test_load_from_log_missing_file_noop(self, tmp_path):
        b = ThompsonBandit(["x"])
        b.load_from_log(tmp_path / "nonexistent.jsonl")
        assert b.alpha["x"] == 1.0
        assert b.beta["x"] == 1.0

    def test_load_from_log_strips_suffixes(self, tmp_path):
        log = tmp_path / "rewards.jsonl"
        log.write_text(
            '{"strategy": "bridge_discovery", "score": 1.0}\n'
            '{"strategy": "contradiction_detection", "score": 1.0}\n',
            encoding="utf-8",
        )
        random.seed(0)
        b = ThompsonBandit(STRATEGIES)
        b.load_from_log(log)
        assert b.alpha["bridge"] == 2.0
        assert b.alpha["contradiction"] == 2.0


class TestActionBandit:
    def _node(self, evidence=None, answer="", depth=0) -> ReasoningNode:
        return ReasoningNode(
            id="n1",
            tree_id="t1",
            query="q",
            evidence_ids=list(evidence or []),
            partial_answer=answer,
            action_taken="root",
            depth=depth,
        )

    def test_must_search_without_evidence(self):
        ab = ActionBandit([SearchAction(), SynthesizeAction()])
        node = self._node(evidence=[], answer="")
        assert ab.select(node).name == "search"

    def test_must_synthesize_with_evidence_no_answer(self):
        ab = ActionBandit([SearchAction(), SynthesizeAction()])
        node = self._node(evidence=["c1"], answer="")
        assert ab.select(node).name == "synthesize"

    def test_update_tracks_stats(self):
        ab = ActionBandit([SearchAction(), SynthesizeAction()])
        ab.update("search", 0.7)
        ab.update("search", 0.9)
        stats = ab.stats()
        assert stats["search"]["visits"] == 2
        assert stats["search"]["avg_score"] == pytest.approx(0.8)

    def test_update_ignores_unknown_action(self):
        ab = ActionBandit([SearchAction()])
        ab.update("nope", 1.0)
        assert ab.stats()["search"]["visits"] == 0

    def test_ucb1_prefers_unvisited(self):
        """UCB1 returns +inf for zero-visit arms → always picked first."""
        ab = ActionBandit([SearchAction(), SynthesizeAction()])
        ab.update("search", 0.9)
        # Evidence present + answer present → eligible: all non-hard-constrained
        node = self._node(evidence=["c1"], answer="partial")
        picked = ab.select(node)
        assert picked.name == "synthesize"  # unvisited wins
