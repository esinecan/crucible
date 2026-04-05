"""MCTS engine for knowledge graph reasoning.

Given a query, builds a reasoning tree by:
1. Selection: UCT picks the most promising leaf to expand
2. Expansion: an action (search, follow_ref, challenge, synthesize) produces a child
3. Simulation: referee evaluates the partial answer quality
4. Backpropagation: score propagates up to root
"""
from __future__ import annotations

import json
import math
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..graph.client import CrucibleGraph
from ..models import ReasoningNode
from .actions import ACTION_REGISTRY, Action
from .referee import LLMReferee, Referee


class ActionBandit:
    """UCB1 selection over expansion actions.

    Each action has a visit count and total score. UCB1 balances
    exploitation (actions that improved answer score) with exploration
    (under-tried actions). This replaces the heuristic if/else in
    _pick_action with a learned policy.

    Unlike the insight bandit (Thompson Sampling / Beta-Bernoulli),
    this uses UCB1 because the action space is small (4 actions) and
    we want deterministic selection, not stochastic.
    """

    def __init__(self, actions: list[Action], c: float = 1.41):
        self._actions = {a.name: a for a in actions}
        self._visits: dict[str, int] = {a.name: 0 for a in actions}
        self._total: dict[str, float] = {a.name: 0.0 for a in actions}
        self._c = c

    def select(self, node: ReasoningNode) -> Action:
        """Pick the best action for this node via UCB1.

        Hard constraints applied first:
        - No evidence → must search
        - No answer + have evidence → must synthesize
        - challenge can't run without a partial answer
        Then UCB1 over the remaining eligible actions.
        """
        eligible = list(self._actions.keys())

        # Hard constraints
        if not node.evidence_ids:
            return self._actions["search"]
        if not node.partial_answer and "synthesize" in eligible:
            return self._actions["synthesize"]
        if not node.partial_answer:
            eligible = [a for a in eligible if a != "challenge"]

        total_visits = sum(self._visits[a] for a in eligible) or 1

        def ucb1(name: str) -> float:
            v = self._visits[name]
            if v == 0:
                return float("inf")
            exploit = self._total[name] / v
            explore = self._c * math.sqrt(math.log(total_visits) / v)
            return exploit + explore

        best = max(eligible, key=ucb1)
        return self._actions[best]

    def update(self, action_name: str, score: float) -> None:
        """Update action stats after observing a score."""
        if action_name in self._visits:
            self._visits[action_name] += 1
            self._total[action_name] += score

    def stats(self) -> dict[str, dict]:
        return {
            name: {
                "visits": self._visits[name],
                "avg_score": self._total[name] / self._visits[name] if self._visits[name] > 0 else 0.0,
            }
            for name in self._actions
        }


class MCTSEngine:
    def __init__(
        self,
        config: Config,
        graph: CrucibleGraph,
        referee: Referee | None = None,
        actions: list[Action] | None = None,
        reward_log: str | Path | None = None,
    ):
        self.config = config
        self.graph = graph
        self.referee = referee or LLMReferee(config)
        self.actions = actions or list(ACTION_REGISTRY.values())
        self.action_bandit = ActionBandit(self.actions, c=config.mcts_uct_c)
        self.reward_log = Path(reward_log) if reward_log else Path("answer_rewards.jsonl")
        self._nodes: dict[str, ReasoningNode] = {}

    # ── UCT Selection ──────────────────────────────────────

    def _uct_score(self, node: ReasoningNode, parent_visits: int) -> float:
        if node.visits == 0:
            return float("inf")
        exploit = node.total_score / node.visits
        explore = self.config.mcts_uct_c * math.sqrt(math.log(parent_visits) / node.visits)
        return exploit + explore

    def _select(self, root: ReasoningNode) -> ReasoningNode:
        """Walk down tree via UCT until we reach a leaf or under-visited node."""
        current = root
        while not current.is_leaf:
            children = [self._nodes[cid] for cid in current.children_ids if cid in self._nodes]
            if not children:
                break
            # If any child is unvisited, select it immediately
            unvisited = [c for c in children if c.visits == 0]
            if unvisited:
                return random.choice(unvisited)
            # UCT selection
            current = max(children, key=lambda c: self._uct_score(c, current.visits))
        return current

    # ── Expansion ──────────────────────────────────────────

    def _expand(self, node: ReasoningNode) -> ReasoningNode | None:
        """Generate a child node via the action bandit's UCB1 selection.

        The action bandit applies hard constraints first (must search if
        no evidence, must synthesize if no answer), then UCB1 over eligible
        actions. If the selected action returns None, tries alternatives.
        """
        action = self.action_bandit.select(node)
        child = action.expand(node, self.graph, self.config)
        if child is None:
            for alt in self.actions:
                if alt.name != action.name:
                    child = alt.expand(node, self.graph, self.config)
                    if child is not None:
                        break
        if child:
            self._nodes[child.id] = child
            node.children_ids.append(child.id)
        return child

    # ── Simulation ─────────────────────────────────────────

    def _simulate(self, node: ReasoningNode) -> float:
        """Evaluate the partial answer at this node."""
        if not node.partial_answer:
            return 0.1  # no answer yet, minimal score

        # Gather evidence texts for context
        evidence_texts = []
        for eid in node.evidence_ids[-5:]:
            try:
                results = self.graph.cypher_read(
                    f"MATCH (n) WHERE n.id = '{eid}' RETURN n.text AS text LIMIT 1"
                )
                if results and results[0].get("text"):
                    evidence_texts.append(results[0]["text"][:300])
            except Exception:
                continue

        result = self.referee.evaluate(node.query, node.partial_answer, evidence_texts)
        return result.score

    # ── Backpropagation ────────────────────────────────────

    def _backpropagate(self, node: ReasoningNode, score: float) -> None:
        """Walk up from node to root, updating visits and total_score."""
        current: ReasoningNode | None = node
        while current is not None:
            current.visits += 1
            current.total_score += score
            current.score = current.avg_score
            current = self._nodes.get(current.parent_id) if current.parent_id else None

    # ── Main Search Loop ───────────────────────────────────

    def search(
        self,
        query: str,
        max_iterations: int | None = None,
        max_depth: int | None = None,
        corpus_id: str | None = None,
    ) -> ReasoningTree:
        max_iterations = max_iterations or self.config.mcts_max_iterations
        max_depth = max_depth or self.config.mcts_max_depth

        tree_id = uuid.uuid4().hex[:16]
        root = ReasoningNode(
            id=uuid.uuid4().hex[:20],
            tree_id=tree_id,
            query=query,
            action_taken="root",
        )
        self._nodes = {root.id: root}

        best_score = 0.0
        best_node_id = root.id

        for iteration in range(max_iterations):
            # 1. Selection
            leaf = self._select(root)

            # 2. Expansion (if within depth limit)
            if leaf.depth < max_depth:
                child = self._expand(leaf)
                if child:
                    leaf = child

            # 3. Simulation
            score = self._simulate(leaf)

            # 4. Backpropagation
            self._backpropagate(leaf, score)

            # 4b. Update action bandit with this score
            if leaf.action_taken and leaf.action_taken != "root":
                self.action_bandit.update(leaf.action_taken, score)

            # Track best
            if score > best_score:
                best_score = score
                best_node_id = leaf.id

            print(
                f"  iter {iteration+1}/{max_iterations}: "
                f"depth={leaf.depth} action={leaf.action_taken} "
                f"score={score:.3f} best={best_score:.3f}"
            )

        # Print action bandit stats
        print("\nAction bandit:")
        for name, s in sorted(self.action_bandit.stats().items(), key=lambda x: x[1]["avg_score"], reverse=True):
            if s["visits"] > 0:
                print(f"  {name:12s}  avg={s['avg_score']:.3f}  visits={s['visits']}")

        # Persist tree to graph
        for node in self._nodes.values():
            self.graph.upsert_reasoning_node(node)

        best = self._nodes[best_node_id]

        # Log reward
        self._log_reward(
            tree_id=tree_id,
            query=query,
            best_score=best_score,
            best_answer=best.partial_answer[:500],
            iterations=max_iterations,
            total_nodes=len(self._nodes),
            evidence_count=len(best.evidence_ids),
        )

        return ReasoningTree(
            tree_id=tree_id,
            query=query,
            root=root,
            nodes=self._nodes,
            best_node=best,
            iterations=max_iterations,
        )

    def _log_reward(self, **kwargs) -> None:
        kwargs["mode"] = "answer"
        kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.reward_log, "a") as f:
            f.write(json.dumps(kwargs, default=str) + "\n")


class ReasoningTree:
    """Result of an MCTS search."""

    def __init__(
        self,
        tree_id: str,
        query: str,
        root: ReasoningNode,
        nodes: dict[str, ReasoningNode],
        best_node: ReasoningNode,
        iterations: int,
    ):
        self.tree_id = tree_id
        self.query = query
        self.root = root
        self.nodes = nodes
        self.best_node = best_node
        self.iterations = iterations

    @property
    def best_answer(self) -> str:
        return self.best_node.partial_answer

    @property
    def best_score(self) -> float:
        return self.best_node.score

    @property
    def evidence_ids(self) -> list[str]:
        return self.best_node.evidence_ids

    def get_path_to_best(self) -> list[ReasoningNode]:
        """Walk from best node up to root."""
        path = []
        current: ReasoningNode | None = self.best_node
        while current is not None:
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        return list(reversed(path))

    def summary(self) -> str:
        path = self.get_path_to_best()
        lines = [
            f"Tree {self.tree_id}: {self.iterations} iterations, {len(self.nodes)} nodes",
            f"Query: {self.query}",
            f"Best score: {self.best_score:.3f} (depth {self.best_node.depth})",
            f"Evidence: {len(self.evidence_ids)} items",
            "",
            "Path to best answer:",
        ]
        for node in path:
            lines.append(
                f"  d={node.depth} [{node.action_taken}] "
                f"score={node.avg_score:.3f} visits={node.visits} "
                f"evidence={len(node.evidence_ids)}"
            )
        lines.append("")
        lines.append("Best answer:")
        lines.append(self.best_answer[:1000] if self.best_answer else "(no answer)")
        return "\n".join(lines)
