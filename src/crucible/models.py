from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Corpus:
    id: str
    name: str
    description: str = ""
    source_path: str = ""
    created_at: str = field(default_factory=_now)


@dataclass
class Document:
    id: str
    corpus_id: str
    path: str
    title: str
    source_type: str  # markdown, json, text, etc.
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    ingested_at: str = field(default_factory=_now)


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    position: int
    heading: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)


@dataclass
class Insight:
    id: str
    corpus_id: str
    text: str
    strategy: str
    score: float = 0.0
    novelty: float = 0.0
    relevance: float = 0.0
    layer: int = 1  # 1 = derived from chunks, 2 = derived from insights
    embedding: list[float] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)
    source_insight_ids: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)


@dataclass
class ReasoningNode:
    """A node in an MCTS reasoning tree."""

    id: str
    tree_id: str
    query: str
    evidence_ids: list[str] = field(default_factory=list)
    partial_answer: str = ""
    action_taken: str = ""  # which expansion action produced this node
    score: float = 0.0
    visits: int = 0
    total_score: float = 0.0
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    @property
    def avg_score(self) -> float:
        return self.total_score / self.visits if self.visits > 0 else 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0


@dataclass
class Entity:
    """A named entity extracted from corpus text via ontology-guided extraction."""

    id: str  # deterministic: hash(corpus_id, entity_type, lower(name))
    corpus_id: str
    name: str  # canonical name
    entity_type: str  # UPPER_SNAKE_CASE from ontology
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    mention_count: int = 1
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)


@dataclass
class Relation:
    """A typed relationship between two entities, extracted from chunk text."""

    id: str  # deterministic: hash(src_id, rel_type, tgt_id, chunk_id)
    source_entity_id: str
    target_entity_id: str
    relation_type: str  # UPPER_SNAKE_CASE from ontology
    evidence: str = ""  # quote from source text
    confidence: float = 0.0
    source_chunk_id: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
