"""Tests for ActionBandit persistence — replaying state from the reward log.

Mirrors the ThompsonBandit load_from_log contract: observations are appended
per update() call, then replayed at construction time so learned selection
state survives across CLI invocations.
"""
from __future__ import annotations

import json

from crucible.reasoning.actions import SearchAction, SynthesizeAction
from crucible.reasoning.mcts import ActionBandit


class TestRoundtripPersistence:
    def test_empty_log_leaves_priors(self, tmp_path):
        log = tmp_path / "rewards.jsonl"
        ab = ActionBandit(
            [SearchAction(), SynthesizeAction()], persistence_log=log
        )
        stats = ab.stats()
        assert stats["search"]["visits"] == 0
        assert stats["synthesize"]["visits"] == 0

    def test_updates_are_appended(self, tmp_path):
        log = tmp_path / "rewards.jsonl"
        ab = ActionBandit([SearchAction()], persistence_log=log)
        ab.update("search", 0.7)
        ab.update("search", 0.9)

        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        entries = [json.loads(l) for l in lines]
        assert all(e["mode"] == "answer_action" for e in entries)
        assert all(e["action"] == "search" for e in entries)
        assert [e["score"] for e in entries] == [0.7, 0.9]

    def test_reload_reconstructs_state(self, tmp_path):
        log = tmp_path / "rewards.jsonl"

        first = ActionBandit(
            [SearchAction(), SynthesizeAction()], persistence_log=log
        )
        first.update("search", 0.8)
        first.update("search", 0.6)
        first.update("synthesize", 0.9)

        second = ActionBandit(
            [SearchAction(), SynthesizeAction()], persistence_log=log
        )
        s = second.stats()
        assert s["search"]["visits"] == 2
        assert s["search"]["avg_score"] == 0.7  # (0.8 + 0.6) / 2
        assert s["synthesize"]["visits"] == 1
        assert s["synthesize"]["avg_score"] == 0.9


class TestLogResilience:
    def test_ignores_non_action_entries(self, tmp_path):
        """The shared reward log may contain answer/summary entries too."""
        log = tmp_path / "rewards.jsonl"
        log.write_text(
            json.dumps({"mode": "answer", "best_score": 0.9}) + "\n"
            + json.dumps({"mode": "answer_action", "action": "search", "score": 0.5}) + "\n"
            + json.dumps({"mode": "insight", "strategy": "bridge", "score": 0.6}) + "\n"
        )
        ab = ActionBandit([SearchAction()], persistence_log=log)
        s = ab.stats()
        assert s["search"]["visits"] == 1
        assert s["search"]["avg_score"] == 0.5

    def test_ignores_unknown_action_names(self, tmp_path):
        """Log entries for actions no longer registered must not crash."""
        log = tmp_path / "rewards.jsonl"
        log.write_text(
            json.dumps({"mode": "answer_action", "action": "legacy_action", "score": 0.5}) + "\n"
            + json.dumps({"mode": "answer_action", "action": "search", "score": 0.7}) + "\n"
        )
        ab = ActionBandit([SearchAction()], persistence_log=log)
        assert ab.stats()["search"]["visits"] == 1

    def test_skips_malformed_lines(self, tmp_path):
        log = tmp_path / "rewards.jsonl"
        log.write_text(
            "not json at all\n"
            + json.dumps({"mode": "answer_action", "action": "search", "score": 0.4}) + "\n"
            + "\n"  # blank
            + json.dumps({"mode": "answer_action", "action": "search"}) + "\n"  # missing score
        )
        ab = ActionBandit([SearchAction()], persistence_log=log)
        assert ab.stats()["search"]["visits"] == 1
        assert ab.stats()["search"]["avg_score"] == 0.4

    def test_missing_log_file_is_clean_slate(self, tmp_path):
        ab = ActionBandit([SearchAction()], persistence_log=tmp_path / "nope.jsonl")
        assert ab.stats()["search"]["visits"] == 0

    def test_no_persistence_still_works(self):
        """Backwards-compat: constructed without a log still selects/updates."""
        ab = ActionBandit([SearchAction(), SynthesizeAction()])
        ab.update("search", 0.8)
        assert ab.stats()["search"]["visits"] == 1


class TestCrossSessionContinuity:
    def test_third_session_sees_both_predecessors(self, tmp_path):
        log = tmp_path / "rewards.jsonl"
        a = ActionBandit([SearchAction()], persistence_log=log)
        a.update("search", 0.5)
        b = ActionBandit([SearchAction()], persistence_log=log)
        b.update("search", 0.9)
        c = ActionBandit([SearchAction()], persistence_log=log)
        assert c.stats()["search"]["visits"] == 2
        assert c.stats()["search"]["avg_score"] == 0.7
