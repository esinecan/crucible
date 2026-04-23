"""Integration tests for meta-strategy dedup + anti-cluster.

Cross-cycle dedup: a re-run of meta on the same L1 inventory must not
re-bridge already-existing (a_id, b_id) pairs. The cocrucible experiment
produced 4 L2s where 2 were the same gap×hub bridge with different
LLM-synthesized text — this is what we're preventing here.

Anti-cluster: when several top-similarity candidates share the same
(a_strategy, b_strategy) pair, the cycle defers later same-pair candidates
so other strategy combinations get airtime first.
"""
from __future__ import annotations

import math

import pytest

from crucible.insight.engine import InsightEngine
from crucible.models import Corpus, Insight


pytestmark = pytest.mark.integration


def _vec(seed: int, dim: int = 768) -> list[float]:
    raw = [math.sin(seed + i * 0.01) for i in range(dim)]
    n = math.sqrt(sum(x * x for x in raw))
    return [x / n for x in raw]


def _l1(iid: str, strategy: str, vec_seed: int, score: float = 0.7) -> Insight:
    return Insight(
        id=iid,
        corpus_id="corp-meta",
        text=f"L1 {iid} via {strategy}",
        strategy=strategy,
        score=score,
        layer=1,
        embedding=_vec(vec_seed),
    )


def _engine(config, graph, tmp_path) -> InsightEngine:
    return InsightEngine(
        config, graph,
        reward_log=tmp_path / "rewards.jsonl",
        use_evaluator=False,
    )


def _seed_l1_pool(graph, n_per_strategy: dict[str, int]) -> None:
    """Seed `n_per_strategy` L1 insights, all with similar embeddings so
    cross-strategy pairs reliably exceed the meta cosine threshold (0.65).
    Each strategy uses a slightly different vector seed offset; within a
    strategy, items get sequential seeds so they're not identical."""
    graph.upsert_corpus(Corpus(id="corp-meta", name="meta-test"))
    seed_offset = {
        "hub": 0, "gap": 50, "bridge": 100, "outlier": 150,
        "contradiction": 200, "meta": 250,
    }
    for strategy, n in n_per_strategy.items():
        base = seed_offset[strategy]
        for k in range(n):
            graph.upsert_insight(_l1(
                f"{strategy}-{k}", strategy, base + k * 0.5,
            ))


class TestMetaDedup:
    def test_rerun_skips_already_bridged_pairs(
        self, graph, config, fake_embed, monkeypatch, tmp_path,
    ):
        """First meta cycle bridges some pairs; the SECOND cycle on the same
        L1 pool must NOT re-bridge those pairs (verified via insight count
        delta and via existing_meta_bridge_pairs lookup)."""
        _seed_l1_pool(graph, {"hub": 3, "gap": 3})
        engine = _engine(config, graph, tmp_path)

        # First run: produce up to 4 L2s.
        first = engine._meta_cycle(corpus_id="corp-meta", max_insights=4)
        assert len(first) >= 1, "first meta cycle should produce at least one bridge"
        for ins in first:
            graph.upsert_insight(ins)
        first_ids = {ins.id for ins in first}

        # Verify the graph now reports those exact pairs as already-bridged.
        sample_ids = [f"{s}-{i}" for s in ("hub", "gap") for i in range(3)]
        bridged_after_first = graph.existing_meta_bridge_pairs(sample_ids)
        assert len(bridged_after_first) == len(first), (
            "every persisted L2 should appear as a bridged pair"
        )

        # Second run: same pool, dedup should kick in.
        second = engine._meta_cycle(corpus_id="corp-meta", max_insights=4)
        # If first run consumed all viable cross-strategy pairs, second
        # produces zero. If pool is large enough that there are leftover
        # non-bridged pairs, those get picked — but none should overlap
        # with first run's pairs.
        for ins in second:
            graph.upsert_insight(ins)
        second_pair_ids = {
            tuple(sorted(ins.source_insight_ids)) for ins in second
        }
        first_pair_ids = {
            tuple(sorted(ins.source_insight_ids)) for ins in first
        }
        assert second_pair_ids.isdisjoint(first_pair_ids), (
            "second cycle re-bridged a pair already covered by the first"
        )
        second_ids = {ins.id for ins in second}
        assert second_ids.isdisjoint(first_ids), (
            "second cycle produced an L2 with a colliding id"
        )

    def test_anti_cluster_promotes_strategy_diversity(
        self, graph, config, fake_embed, monkeypatch, tmp_path,
    ):
        """When the top-similarity candidates all share one (a_strategy,
        b_strategy) pair, the anti-cluster mechanism defers further
        same-pair candidates after two consecutive picks so other strategy
        combinations get a chance.

        Strategy: seed enough L1s so that hub×gap dominates similarity but
        bridge×outlier pairs also exist. Without anti-cluster, all 4 picks
        would be hub×gap. With it, the 3rd and 4th picks should reach into
        non-hub-gap pairs.
        """
        _seed_l1_pool(graph, {"hub": 4, "gap": 4, "bridge": 2, "outlier": 2})
        engine = _engine(config, graph, tmp_path)

        results = engine._meta_cycle(corpus_id="corp-meta", max_insights=4)
        assert len(results) == 4

        strategy_pairs = [
            tuple(sorted([
                ins.context["insight_a_strategy"],
                ins.context["insight_b_strategy"],
            ]))
            for ins in results
        ]
        # First two may share a pair (allowed); but at least one of picks
        # 3 or 4 must be a different strategy combination if any exist
        # in the candidate pool.
        unique_pairs = set(strategy_pairs)
        assert len(unique_pairs) >= 2, (
            f"anti-cluster failed; all picks were {strategy_pairs[0]}"
        )
