"""Adversarial insight evaluator — Advocate/Skeptic/Referee debate."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

from ..config import Config

# ── Prompt Builders (domain injected at runtime) ────────────

def _advocate_system(domain: str) -> str:
    return domain + (
        "You are an insight advocate. Argue why this candidate insight from "
        "a knowledge base is a genuinely valuable discovery.\n\n"
        "Focus on:\n"
        "- What non-obvious connection or pattern does this reveal?\n"
        "- What would someone learn that they couldn't see from individual sources?\n"
        "- Does this challenge assumptions or reveal hidden connections?\n\n"
        "Be specific. Cite evidence from the text. 2-3 sentences max."
    )

def _skeptic_system(domain: str) -> str:
    return domain + (
        "You are an insight skeptic. Argue why this candidate insight is NOT "
        "interesting or valuable.\n\n"
        "Focus on:\n"
        "- Is this just surface-level keyword or topic overlap?\n"
        "- Would any informed reader already know this?\n"
        "- Is the connection trivial or obvious from the shared context?\n\n"
        "Be specific. 2-3 sentences max."
    )

def _referee_system(domain: str) -> str:
    return domain + (
        "You are a referee. An advocate argued an insight is valuable, "
        "a skeptic argued it's not. Score the insight on three dimensions:\n\n"
        "- novelty (0.0-1.0): How surprising to an informed reader?\n"
        "- relevance (0.0-1.0): How useful for deeper understanding?\n"
        "- actionability (0.0-1.0): Does it suggest a concrete line of inquiry?\n\n"
        "Return ONLY valid JSON: {\"novelty\": X, \"relevance\": Y, \"actionability\": Z}"
    )


@dataclass
class EvalContext:
    """Evidence gathered by the harness for evaluator agents."""
    source_chunks: list[str]       # text of source chunks (max 5, 500 chars each)
    source_entities: list[str]     # "EntityName (TYPE)" strings
    related_insights: list[str]    # text of similar existing insights (max 3)
    entity_paths: list[str]        # "A -[REL]-> B" entity connection strings
    strategy: str = ""             # which strategy produced this insight


@dataclass
class EvalResult:
    novelty: float
    relevance: float
    actionability: float
    advocate_arg: str
    skeptic_arg: str

    @property
    def score(self) -> float:
        return (self.novelty + self.relevance + self.actionability) / 3


class AdversarialEvaluator:
    def __init__(self, config: Config):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("CRUCIBLE_EVAL_MODEL", "deepseek-chat")
        self.domain = config.domain_preamble

    def evaluate(self, insight_text: str, context: EvalContext | None = None) -> EvalResult:
        user_msg = f"Evaluate this insight:\n\n{insight_text}"
        if context:
            if context.source_chunks:
                user_msg += "\n\nSOURCE EVIDENCE:\n" + "\n---\n".join(context.source_chunks[:5])
            if context.source_entities:
                user_msg += "\n\nENTITIES INVOLVED:\n" + ", ".join(context.source_entities[:10])
            if context.entity_paths:
                user_msg += "\n\nENTITY CONNECTIONS:\n" + "\n".join(context.entity_paths[:5])
            if context.related_insights:
                user_msg += "\n\nEXISTING RELATED INSIGHTS:\n" + "\n---\n".join(context.related_insights[:3])

        advocate_arg = self._chat(_advocate_system(self.domain), user_msg)
        skeptic_arg = self._chat(_skeptic_system(self.domain), user_msg)

        referee_prompt = (
            f"INSIGHT:\n{insight_text}\n\n"
            f"ADVOCATE:\n{advocate_arg}\n\n"
            f"SKEPTIC:\n{skeptic_arg}\n\n"
            f"Score the insight as JSON."
        )
        referee_resp = self._chat(_referee_system(self.domain), referee_prompt)
        scores = self._parse_scores(referee_resp)

        return EvalResult(
            novelty=scores.get("novelty", 0.5),
            relevance=scores.get("relevance", 0.5),
            actionability=scores.get("actionability", 0.5),
            advocate_arg=advocate_arg,
            skeptic_arg=skeptic_arg,
        )

    def _chat(self, system: str, user: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.7,
                "max_tokens": 400,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse_scores(self, text: str) -> dict[str, float]:
        """Extract scores from referee response, tolerant of messy output."""
        try:
            data = json.loads(text)
            return self._clamp(data)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[^}]*\"novelty\"[^}]*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return self._clamp(data)
            except json.JSONDecodeError:
                pass

        scores = {}
        for key in ("novelty", "relevance", "actionability"):
            m = re.search(rf'"{key}"\s*:\s*([\d.]+)', text)
            if m:
                scores[key] = float(m.group(1))
        return self._clamp(scores) if scores else {"novelty": 0.5, "relevance": 0.5, "actionability": 0.5}

    def _clamp(self, data: dict) -> dict[str, float]:
        return {
            k: max(0.0, min(1.0, float(v)))
            for k, v in data.items()
            if k in ("novelty", "relevance", "actionability")
        }
