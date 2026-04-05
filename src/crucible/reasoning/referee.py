"""Pluggable referee interface for MCTS simulation step.

Default: LLMReferee wraps the existing Advocate/Skeptic/Referee debate.
Domain adapters can implement SymbolicReferee for deterministic evaluation.
HybridReferee tries symbolic first, falls back to LLM.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import Config
from ..insight.evaluator import AdversarialEvaluator


@dataclass
class RefereeResult:
    score: float  # 0.0-1.0 overall quality
    novelty: float = 0.5
    relevance: float = 0.5
    grounding: float = 0.5  # how well-grounded in evidence
    reasoning: str = ""  # explanation of score


class Referee(ABC):
    @abstractmethod
    def evaluate(self, query: str, partial_answer: str, evidence: list[str]) -> RefereeResult:
        """Score a partial answer given the query and gathered evidence."""


class LLMReferee(Referee):
    """Wraps the existing adversarial evaluator for MCTS use."""

    def __init__(self, config: Config):
        self._evaluator = AdversarialEvaluator(config)
        self._domain = config.domain_preamble

    def evaluate(self, query: str, partial_answer: str, evidence: list[str]) -> RefereeResult:
        evidence_text = "\n---\n".join(evidence[:5])  # cap context
        text = (
            f"Query: {query}\n\n"
            f"Answer: {partial_answer}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        result = self._evaluator.evaluate(text)
        return RefereeResult(
            score=result.score,
            novelty=result.novelty,
            relevance=result.relevance,
            grounding=result.actionability,
            reasoning=f"Advocate: {result.advocate_arg}\nSkeptic: {result.skeptic_arg}",
        )


class SymbolicReferee(Referee):
    """Stub for domain adapters that evaluate deterministically.

    Burokrat would implement this with predicate evaluation.
    Other domains can implement with test execution, rule matching, etc.
    """

    def evaluate(self, query: str, partial_answer: str, evidence: list[str]) -> RefereeResult:
        raise NotImplementedError("SymbolicReferee requires a domain adapter implementation")


class HybridReferee(Referee):
    """Try symbolic first, fall back to LLM for what symbolic can't cover."""

    def __init__(self, symbolic: SymbolicReferee, llm: LLMReferee):
        self._symbolic = symbolic
        self._llm = llm

    def evaluate(self, query: str, partial_answer: str, evidence: list[str]) -> RefereeResult:
        try:
            return self._symbolic.evaluate(query, partial_answer, evidence)
        except NotImplementedError:
            return self._llm.evaluate(query, partial_answer, evidence)
