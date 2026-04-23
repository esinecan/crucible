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


class TestThompsonBanditPseudoCount:
    """Pseudo-count update preserves the signal that the Bernoulli coin flip
    throws away. It's also deterministic across replays of the same log.
    """

    def test_pseudo_count_is_default(self):
        b = ThompsonBandit(["x"])
        b.update("x", 0.7)
        # 0.7 added to alpha, 0.3 to beta. Bernoulli would have produced
        # integer increments, not fractional.
        assert b.alpha["x"] == pytest.approx(1.7)
        assert b.beta["x"] == pytest.approx(1.3)

    def test_pseudo_count_clamps_out_of_range(self):
        b = ThompsonBandit(["x"], update_mode="pseudo_count")
        b.update("x", 1.5)
        b.update("x", -0.3)
        # 1.5 clamped to 1.0 → alpha+1.0, beta+0.0
        # -0.3 clamped to 0.0 → alpha+0.0, beta+1.0
        assert b.alpha["x"] == pytest.approx(2.0)
        assert b.beta["x"] == pytest.approx(2.0)

    def test_bernoulli_still_available_via_flag(self):
        random.seed(0)
        b = ThompsonBandit(["x"], update_mode="bernoulli")
        # Score of 1.0 under Bernoulli always lands heads
        b.update("x", 1.0)
        assert b.alpha["x"] == 2.0
        assert b.beta["x"] == 1.0

    def test_pseudo_count_log_replay_is_deterministic(self, tmp_path):
        """Same log → same posteriors, regardless of how many times you boot."""
        log = tmp_path / "rewards.jsonl"
        scores = [0.7, 0.4, 0.6, 0.55, 0.8, 0.3, 0.9]
        log.write_text(
            "\n".join(f'{{"strategy": "bridge", "score": {s}}}' for s in scores)
            + "\n",
            encoding="utf-8",
        )
        runs = []
        for _ in range(3):
            b = ThompsonBandit(STRATEGIES)
            b.load_from_log(log)
            runs.append((b.alpha["bridge"], b.beta["bridge"]))
        # All three loads produce identical (alpha, beta). Bernoulli would not.
        assert runs[0] == runs[1] == runs[2]
        # Total alpha contribution = sum(scores); beta = sum(1-scores). Plus prior 1.0.
        assert runs[0][0] == pytest.approx(1.0 + sum(scores))
        assert runs[0][1] == pytest.approx(1.0 + sum(1 - s for s in scores))

    def test_bernoulli_log_replay_is_non_deterministic(self, tmp_path):
        """The motivating bug: same log, different posteriors per boot."""
        log = tmp_path / "rewards.jsonl"
        # Scores near 0.5 → maximum Bernoulli variance
        log.write_text(
            "\n".join(
                f'{{"strategy": "bridge", "score": 0.5}}' for _ in range(60)
            )
            + "\n",
            encoding="utf-8",
        )
        posteriors = set()
        for seed in range(5):
            random.seed(seed)
            b = ThompsonBandit(STRATEGIES, update_mode="bernoulli")
            b.load_from_log(log)
            posteriors.add((b.alpha["bridge"], b.beta["bridge"]))
        # Different seeds → different posteriors. Exactly the drift we're fixing.
        assert len(posteriors) > 1


class TestThompsonBanditSnapshot:
    """Snapshot persistence — the JSONL reward log becomes audit trail, the
    snapshot becomes state.
    """

    def test_snapshot_roundtrip(self, tmp_path):
        snap = tmp_path / "rewards.snapshot.json"
        b1 = ThompsonBandit(STRATEGIES, snapshot_path=snap)
        b1.update("bridge", 0.8)
        b1.update("hub", 0.3)
        # Snapshot written; a fresh bandit reads it directly.
        b2 = ThompsonBandit(STRATEGIES, snapshot_path=snap)
        assert b2.load_snapshot() is True
        assert b2.alpha["bridge"] == b1.alpha["bridge"]
        assert b2.beta["bridge"] == b1.beta["bridge"]
        assert b2.alpha["hub"] == b1.alpha["hub"]

    def test_snapshot_missing_returns_false(self, tmp_path):
        b = ThompsonBandit(STRATEGIES, snapshot_path=tmp_path / "absent.json")
        assert b.load_snapshot() is False

    def test_corrupted_snapshot_returns_false(self, tmp_path):
        snap = tmp_path / "broken.snapshot.json"
        snap.write_text("{not valid json", encoding="utf-8")
        b = ThompsonBandit(STRATEGIES, snapshot_path=snap)
        assert b.load_snapshot() is False
        # Priors intact so caller can fall back to log replay.
        assert b.alpha["bridge"] == 1.0

    def test_wrong_version_snapshot_rejected(self, tmp_path):
        snap = tmp_path / "wrongver.snapshot.json"
        snap.write_text(
            '{"version": 99, "alpha": {"bridge": 5.0}, "beta": {"bridge": 2.0}}',
            encoding="utf-8",
        )
        b = ThompsonBandit(STRATEGIES, snapshot_path=snap)
        assert b.load_snapshot() is False
        assert b.alpha["bridge"] == 1.0  # priors unchanged

    def test_log_replay_writes_snapshot(self, tmp_path):
        log = tmp_path / "rewards.jsonl"
        snap = tmp_path / "rewards.snapshot.json"
        log.write_text(
            '{"strategy": "bridge", "score": 0.7}\n'
            '{"strategy": "bridge", "score": 0.6}\n',
            encoding="utf-8",
        )
        b = ThompsonBandit(STRATEGIES, snapshot_path=snap)
        b.load_from_log(log)
        # After replay the snapshot must exist so subsequent boots can skip the log.
        assert snap.exists()
        import json
        data = json.loads(snap.read_text())
        assert data["version"] == ThompsonBandit.SNAPSHOT_VERSION
        assert data["alpha"]["bridge"] == pytest.approx(1.0 + 0.7 + 0.6)


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


class TestActionBanditSnapshot:
    def test_snapshot_roundtrip(self, tmp_path):
        snap = tmp_path / "action.snapshot.json"
        ab1 = ActionBandit(
            [SearchAction(), SynthesizeAction()], snapshot_path=snap,
        )
        ab1.update("search", 0.7)
        ab1.update("synthesize", 0.9)
        ab1.update("search", 0.6)
        # Fresh bandit reconstructs from snapshot.
        ab2 = ActionBandit(
            [SearchAction(), SynthesizeAction()], snapshot_path=snap,
        )
        s1, s2 = ab1.stats(), ab2.stats()
        assert s2["search"]["visits"] == s1["search"]["visits"] == 2
        assert s2["search"]["avg_score"] == pytest.approx(s1["search"]["avg_score"])
        assert s2["synthesize"]["visits"] == 1

    def test_log_replay_writes_snapshot(self, tmp_path):
        log = tmp_path / "actions.jsonl"
        snap = tmp_path / "actions.snapshot.json"
        log.write_text(
            '{"mode": "answer_action", "action": "search", "score": 0.8}\n'
            '{"mode": "answer_action", "action": "search", "score": 0.6}\n',
            encoding="utf-8",
        )
        ab = ActionBandit(
            [SearchAction(), SynthesizeAction()],
            persistence_log=log,
            snapshot_path=snap,
        )
        # Constructor replayed the log; check stats and snapshot exist.
        assert ab.stats()["search"]["visits"] == 2
        assert snap.exists()

    def test_snapshot_short_circuits_log_on_second_boot(self, tmp_path):
        """Snapshot exists → log is NOT replayed a second time."""
        log = tmp_path / "actions.jsonl"
        snap = tmp_path / "actions.snapshot.json"
        log.write_text(
            '{"mode": "answer_action", "action": "search", "score": 0.8}\n',
            encoding="utf-8",
        )
        ab1 = ActionBandit(
            [SearchAction(), SynthesizeAction()],
            persistence_log=log, snapshot_path=snap,
        )
        assert ab1.stats()["search"]["visits"] == 1
        # Append more entries to the log — a snapshot-first boot must ignore them.
        with open(log, "a") as f:
            f.write('{"mode": "answer_action", "action": "search", "score": 0.9}\n')
        ab2 = ActionBandit(
            [SearchAction(), SynthesizeAction()],
            persistence_log=log, snapshot_path=snap,
        )
        # Still 1 visit — snapshot authoritative, log ignored.
        assert ab2.stats()["search"]["visits"] == 1
