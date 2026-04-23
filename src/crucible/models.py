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
    """A canonical entity — synthesized truth about a named thing.

    Entity is what we believe; Mention is what a specific chunk asserted.
    Keeping them separate lets us audit per-chunk provenance without mutating
    Entity state, and lets a synthesizer refine Entity.description from the
    union of Mentions without racing with the extraction pipeline.

    Source chunks are no longer stored on Entity — they're derived on read
    from `(m:Mention)<-[:EXTRACTS]-(:Chunk)` where `m-[:RESOLVES_TO]->entity`.
    """

    id: str  # deterministic: hash(corpus_id, entity_type, lower(name))
    corpus_id: str
    name: str  # canonical name
    entity_type: str  # UPPER_SNAKE_CASE from ontology
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    last_synthesized_at: str = ""  # ISO timestamp; empty = never synthesized
    created_at: str = field(default_factory=_now)


@dataclass
class Mention:
    """A single chunk's raw extraction of an entity. Immutable.

    One Mention per (chunk, entity) pair. `name_as_extracted` preserves the
    surface form in that chunk even when the canonical Entity.name differs
    (e.g. chunk said "Al", canonical is "Alice"). Description and aliases
    live here because each chunk asserts its own view; the synthesizer folds
    them into Entity.description downstream.
    """

    id: str  # deterministic: hash(chunk_id, entity_id)
    chunk_id: str
    entity_id: str  # resolves to canonical Entity
    name_as_extracted: str  # surface form in this chunk
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)


@dataclass
class Relation:
    """A typed relationship between two entities, aggregated across chunks.

    A relation is a claim (src --rel_type--> tgt), not a per-chunk observation.
    Multiple chunks asserting the same claim collapse to a single Relation
    whose source_chunks list preserves provenance.
    """

    id: str  # deterministic: hash(src_id, rel_type, tgt_id)
    source_entity_id: str
    target_entity_id: str
    relation_type: str  # UPPER_SNAKE_CASE from ontology
    evidence: str = ""  # representative quote(s) from source text
    confidence: float = 0.0
    source_chunks: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
