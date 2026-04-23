"""Unit tests for the pre-persist grounding check.

Stubs the LLM client and the graph layer. Verifies the verdict-handling
contract: contradicted-with-confidence drops the insight (and logs
'dropped_grounding'), unsupported persists with a marker (so the bandit
discount applies downstream), supported is untouched, layer-2 (meta)
insights bypass the check entirely.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from crucible.config import Config
from crucible.insight.engine import InsightEngine
from crucible.models import Insight


def _config(tmp_path, grounding_check=True) -> Config:
    return Config(
        deepseek_api_key="test",
        state_dir=str(tmp_path / "state"),
        grounding_check=grounding_check,
    )


def _l1(iid: str, text: str = "claim text", source_chunks=None) -> Insight:
    return Insight(
        id=iid,
        corpus_id="corp",
        text=text,
        strategy="hub",
        score=0.7,
        layer=1,
        source_chunk_ids=source_chunks or ["chunk-1"],
    )


def _l2(iid: str = "l2-1") -> Insight:
    return Insight(
        id=iid,
        corpus_id="corp",
        text="meta bridge text",
        strategy="meta",
        score=0.6,
        layer=2,
        source_insight_ids=["l1-a", "l1-b"],
    )


def _engine(tmp_path, grounding_check=True) -> InsightEngine:
    config = _config(tmp_path, grounding_check=grounding_check)
    graph = MagicMock()
    graph.cypher_read.return_value = [{"text": "actual source chunk text"}]
    eng = InsightEngine(
        config, graph,
        reward_log=tmp_path / "rewards.jsonl",
        use_evaluator=False,
    )
    return eng


class TestGroundCheck:
    def test_returns_unsupported_when_no_source_chunks(self, tmp_path):
        eng = _engine(tmp_path)
        ins = _l1("i-1", source_chunks=[])
        verdict, conf = eng._ground_check(ins)
        assert verdict == "unsupported"
        assert conf == 0.0

    def test_returns_unsupported_when_chunk_fetch_empty(self, tmp_path):
        eng = _engine(tmp_path)
        eng.graph.cypher_read.return_value = []  # chunk not found
        ins = _l1("i-2")
        verdict, conf = eng._ground_check(ins)
        assert verdict == "unsupported"
        assert conf == 0.0

    def test_returns_unsupported_when_llm_raises(self, tmp_path):
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        eng._llm_client.chat_json.side_effect = RuntimeError("network blip")
        verdict, conf = eng._ground_check(_l1("i-3"))
        assert verdict == "unsupported"
        assert conf == 0.0

    def test_parses_support_verdict(self, tmp_path):
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        eng._llm_client.chat_json.return_value = {
            "verdict": "support", "confidence": 0.85,
        }
        verdict, conf = eng._ground_check(_l1("i-4"))
        assert verdict == "support"
        assert conf == pytest.approx(0.85)

    def test_parses_contradict_verdict(self, tmp_path):
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        eng._llm_client.chat_json.return_value = {
            "verdict": "contradict", "confidence": 0.9,
        }
        verdict, conf = eng._ground_check(_l1("i-5"))
        assert verdict == "contradict"
        assert conf == pytest.approx(0.9)

    def test_unknown_verdict_normalizes_to_unsupported(self, tmp_path):
        """LLM occasionally returns 'partial' or 'mixed' — fall back to
        unsupported rather than crashing or trusting an unknown label."""
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        eng._llm_client.chat_json.return_value = {
            "verdict": "partially supported", "confidence": 0.7,
        }
        verdict, conf = eng._ground_check(_l1("i-6"))
        assert verdict == "unsupported"
        assert conf == pytest.approx(0.7)

    def test_clamps_confidence_to_unit_interval(self, tmp_path):
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        eng._llm_client.chat_json.return_value = {
            "verdict": "support", "confidence": 1.7,
        }
        _, conf = eng._ground_check(_l1("i-7"))
        assert conf == 1.0


class TestFilterByGrounding:
    def test_drops_contradicted_at_high_confidence(self, tmp_path):
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        # Three L1 insights, three different verdicts.
        responses = [
            {"verdict": "contradict", "confidence": 0.9},   # dropped
            {"verdict": "unsupported", "confidence": 0.5},  # marked
            {"verdict": "support", "confidence": 0.8},      # untouched
        ]
        eng._llm_client.chat_json.side_effect = responses
        insights = [_l1("a"), _l1("b"), _l1("c")]
        kept = eng._filter_by_grounding(insights, "hub")
        assert [k.id for k in kept] == ["b", "c"]
        assert kept[0].context["grounding"] == "unsupported"
        assert kept[1].context["grounding"] == "support"

    def test_keeps_contradicted_below_drop_confidence(self, tmp_path):
        """Low-confidence contradiction is treated as 'unsupported' rather
        than dropped — the model wasn't sure enough to discard the insight."""
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        eng._llm_client.chat_json.return_value = {
            "verdict": "contradict", "confidence": 0.5,
        }
        insights = [_l1("a")]
        kept = eng._filter_by_grounding(insights, "hub")
        assert [k.id for k in kept] == ["a"]
        # Marked as 'contradict' so caller can apply its own policy if it
        # wants stricter behavior; current run_cycle only acts on
        # 'unsupported' for the bandit discount.
        assert kept[0].context["grounding"] == "contradict"
        assert kept[0].context["grounding_confidence"] == pytest.approx(0.5)

    def test_l2_insights_bypass_grounding(self, tmp_path):
        """Meta insights are abstractions over multiple L1s — their grounding
        is in those L1s, not in any single chunk. Grounding check would
        always return unsupported for L2, which is wrong."""
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        # If grounding ran on L2, this would return contradict; we expect
        # the helper to skip the L2 entirely so chat_json is never called.
        eng._llm_client.chat_json.return_value = {
            "verdict": "contradict", "confidence": 0.95,
        }
        insights = [_l2("l2-meta")]
        kept = eng._filter_by_grounding(insights, "meta")
        assert [k.id for k in kept] == ["l2-meta"]
        # The L2 insight should not have a grounding marker because we
        # bypassed the check.
        assert "grounding" not in kept[0].context
        eng._llm_client.chat_json.assert_not_called()

    def test_dropped_insight_logged_to_reward_log(self, tmp_path):
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        eng._llm_client.chat_json.return_value = {
            "verdict": "contradict", "confidence": 0.9,
        }
        eng._filter_by_grounding([_l1("dropped-id")], "gap")
        # Reward log captured the drop event.
        assert eng.reward_log.exists()
        entries = [
            json.loads(line)
            for line in eng.reward_log.read_text().splitlines()
            if line.strip()
        ]
        dropped = [e for e in entries if e.get("mode") == "dropped_grounding"]
        assert len(dropped) == 1
        assert dropped[0]["insight_id"] == "dropped-id"
        assert dropped[0]["strategy"] == "gap"
        assert dropped[0]["grounding_confidence"] == pytest.approx(0.9)

    def test_empty_list_passes_through(self, tmp_path):
        eng = _engine(tmp_path)
        eng._llm_client = MagicMock()
        kept = eng._filter_by_grounding([], "hub")
        assert kept == []
        eng._llm_client.chat_json.assert_not_called()


class TestGroundingFlagDefault:
    """The grounding check is opt-in via env var. Default behavior must be
    unchanged so existing scripts and prior tests aren't affected."""

    def test_default_off_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("CRUCIBLE_GROUNDING_CHECK", raising=False)
        assert Config().grounding_check is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_env_values_enable(self, monkeypatch, val):
        monkeypatch.setenv("CRUCIBLE_GROUNDING_CHECK", val)
        assert Config().grounding_check is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "garbage"])
    def test_falsy_env_values_disable(self, monkeypatch, val):
        monkeypatch.setenv("CRUCIBLE_GROUNDING_CHECK", val)
        assert Config().grounding_check is False
