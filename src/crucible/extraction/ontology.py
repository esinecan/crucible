"""Ontology generation from corpus insights.

The insight engine discovers corpus-level patterns. This module synthesizes
those patterns into a structured ontology (entity classes + relation types)
that guides per-chunk entity extraction.

Strategy → ontology construct mapping:
  hub         → central entity classes (the hub topic is a class)
  bridge      → relation types (the nature of the bridge is a relation)
  gap         → missing classes (referenced but undefined concepts)
  meta (L2)   → class hierarchy (meta-patterns suggest superclasses)
  outlier     → specialized/niche classes
  contradiction → attributes with conflicting definitions
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..config import Config
from ..graph.client import CrucibleGraph
from . import sanitize_type_name


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OntologyClass:
    name: str  # UPPER_SNAKE_CASE
    description: str
    parent: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class OntologyRelation:
    name: str  # UPPER_SNAKE_CASE
    description: str
    domain: str  # source class name
    range: str  # target class name
    examples: list[str] = field(default_factory=list)


@dataclass
class Ontology:
    corpus_id: str
    classes: list[OntologyClass] = field(default_factory=list)
    relations: list[OntologyRelation] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def to_json(self) -> str:
        return json.dumps({
            "corpus_id": self.corpus_id,
            "classes": [
                {"name": c.name, "description": c.description,
                 "parent": c.parent, "examples": c.examples}
                for c in self.classes
            ],
            "relations": [
                {"name": r.name, "description": r.description,
                 "domain": r.domain, "range": r.range, "examples": r.examples}
                for r in self.relations
            ],
            "created_at": self.created_at,
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Ontology:
        data = json.loads(text)
        return cls(
            corpus_id=data["corpus_id"],
            classes=[
                OntologyClass(**c) for c in data.get("classes", [])
            ],
            relations=[
                OntologyRelation(**r) for r in data.get("relations", [])
            ],
            created_at=data.get("created_at", ""),
        )

    def to_extraction_prompt(self) -> str:
        lines = ["Extract entities and relations according to this ontology:\n"]
        lines.append("ENTITY TYPES:")
        for c in self.classes:
            parent = f" (subclass of {c.parent})" if c.parent else ""
            lines.append(f"  - {c.name}{parent}: {c.description}")
        lines.append("\nRELATION TYPES:")
        for r in self.relations:
            lines.append(f"  - {r.name}: {r.domain} -> {r.range} ({r.description})")
        return "\n".join(lines)

    def to_turtle(self) -> str:
        lines = [
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix crucible: <http://crucible.dev/ontology/> .",
            "",
            "crucible:Ontology a owl:Ontology ;",
            f'    rdfs:comment "Generated from corpus {self.corpus_id}" .',
            "",
        ]
        for c in self.classes:
            lines.append(f"crucible:{c.name} a owl:Class ;")
            lines.append(f'    rdfs:label "{c.name}" ;')
            lines.append(f'    rdfs:comment "{c.description}" .')
            if c.parent:
                lines.append(f"crucible:{c.name} rdfs:subClassOf crucible:{c.parent} .")
            lines.append("")
        for r in self.relations:
            lines.append(f"crucible:{r.name} a owl:ObjectProperty ;")
            lines.append(f'    rdfs:label "{r.name}" ;')
            lines.append(f'    rdfs:comment "{r.description}" ;')
            lines.append(f"    rdfs:domain crucible:{r.domain} ;")
            lines.append(f"    rdfs:range crucible:{r.range} .")
            lines.append("")
        return "\n".join(lines)

    @property
    def class_names(self) -> set[str]:
        return {c.name for c in self.classes}

    @property
    def relation_names(self) -> set[str]:
        return {r.name for r in self.relations}


# ── Ontology Generation ──────────────────────────────────

ONTOLOGY_SYSTEM = """{domain}You are an ontology engineer. Given insights discovered from a text corpus, design a minimal ontology of entity classes and relation types.

Each insight type maps to a specific ontology construct:

- HUB insights reveal central concepts that connect many documents. These are ENTITY CLASSES.
- BRIDGE insights reveal connections between different topics. These suggest RELATION TYPES.
- GAP insights reveal referenced-but-undefined concepts. These are MISSING CLASSES (include them).
- META insights (L2, connecting other insights) suggest CLASS HIERARCHY — when a meta insight unifies two concepts, the unifying concept may be a parent class.
- OUTLIER insights reveal unique/specialized content. These suggest NICHE CLASSES.
- CONTRADICTION insights reveal conflicting claims. Note these as annotations on relevant classes.

Rules:
1. Class names: UPPER_SNAKE_CASE, singular nouns (PERSON, POLITICAL_PARTY, ECONOMIC_POLICY)
2. Relation names: UPPER_SNAKE_CASE, verbs or prepositions (CRITICIZES, MEMBER_OF, CAUSED_BY)
3. Keep it minimal but complete: 5-15 classes, 5-20 relations
4. Every relation must specify domain (source class) and range (target class)
5. Always include a CONCEPT class for abstract ideas that don't fit other classes
6. Derive ALL classes and relations FROM the insights, not from general knowledge
7. Include 2-3 brief examples per class/relation drawn from the insight text

Return JSON only:
{{
  "classes": [
    {{"name": "PERSON", "description": "Named individual", "parent": "", "examples": ["example1", "example2"]}}
  ],
  "relations": [
    {{"name": "CRITICIZES", "description": "One entity criticizes another", "domain": "PERSON", "range": "PERSON", "examples": ["X criticizes Y's policy"]}}
  ]
}}"""


def _parse_json_lenient(text: str) -> dict | list | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


class OntologyGenerator:
    def __init__(self, config: Config, graph: CrucibleGraph):
        self.config = config
        self.graph = graph

    def generate(self, corpus_id: str) -> Ontology:
        insights = self._load_insights(corpus_id)
        if not insights:
            raise ValueError(f"No insights found for corpus {corpus_id}. Run 'crucible explore' first.")
        raw = self._llm_generate(insights)
        return self._parse_response(raw, corpus_id)

    def _load_insights(self, corpus_id: str) -> dict[str, list[dict]]:
        # Many insights have empty corpus_id from explore runs without --corpus,
        # so the filter keeps those along with the scoped ones.
        strategies = ["hub", "bridge", "gap", "meta", "outlier", "contradiction"]
        result: dict[str, list[dict]] = {}
        with self.graph._driver.session() as s:
            for strategy in strategies:
                rows = [
                    dict(r)
                    for r in s.run(
                        "MATCH (i:Insight) "
                        "WHERE i.strategy = $strat "
                        "AND (i.corpus_id = $cid OR i.corpus_id IS NULL OR i.corpus_id = '') "
                        "RETURN i.text AS text, i.score AS score, "
                        "i.layer AS layer "
                        "ORDER BY i.score DESC LIMIT 20",
                        cid=corpus_id,
                        strat=strategy,
                    )
                ]
                if rows:
                    result[strategy] = rows
        return result

    def _build_user_prompt(self, insights: dict[str, list[dict]]) -> str:
        sections = []
        for strategy, items in insights.items():
            section = f"=== {strategy.upper()} INSIGHTS ({len(items)} total) ===\n"
            for item in items[:10]:
                score = item.get("score", 0)
                layer = item.get("layer", 1)
                text = item["text"][:400]
                section += f"- [score={score:.2f}, L{layer}] {text}\n\n"
            sections.append(section)
        return "\n".join(sections)

    def _llm_generate(self, insights: dict[str, list[dict]]) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("CRUCIBLE_EVAL_MODEL", "deepseek-chat")

        system = ONTOLOGY_SYSTEM.format(domain=self.config.domain_preamble)
        user = self._build_user_prompt(insights)

        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse_response(self, raw: str, corpus_id: str) -> Ontology:
        parsed = _parse_json_lenient(raw)
        if not parsed or not isinstance(parsed, dict):
            raise ValueError(f"Failed to parse ontology response: {raw[:200]}")

        classes = []
        for c in parsed.get("classes", []):
            if not isinstance(c, dict) or not c.get("name"):
                continue
            try:
                name = sanitize_type_name(c["name"])
            except ValueError:
                continue
            classes.append(OntologyClass(
                name=name,
                description=c.get("description", ""),
                parent=c.get("parent", ""),
                examples=c.get("examples", []),
            ))

        # Ensure CONCEPT class exists as catch-all
        class_names = {c.name for c in classes}
        if "CONCEPT" not in class_names:
            classes.append(OntologyClass(
                name="CONCEPT",
                description="Abstract idea or phenomenon not captured by other classes",
            ))

        relations = []
        for r in parsed.get("relations", []):
            if not isinstance(r, dict) or not r.get("name"):
                continue
            try:
                name = sanitize_type_name(r["name"])
            except ValueError:
                continue
            relations.append(OntologyRelation(
                name=name,
                description=r.get("description", ""),
                domain=r.get("domain", "CONCEPT"),
                range=r.get("range", "CONCEPT"),
                examples=r.get("examples", []),
            ))

        return Ontology(corpus_id=corpus_id, classes=classes, relations=relations)
