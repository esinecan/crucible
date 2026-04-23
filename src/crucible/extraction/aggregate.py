"""In-memory aggregation of per-chunk extractions for a single document.

The extractor runs the LLM per chunk; those raw results are then folded into
deduplicated Entity/Relation objects before anything touches Neo4j. This
eliminates the chunk-overlap artifacts (duplicate relation rows, inflated
mention counts) and lets us resolve aliases with doc-level context instead of
a flat name_to_id dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Entity, Relation
from . import make_entity_id, make_relation_id, sanitize_type_name


@dataclass
class DocAggregation:
    entities: dict[str, Entity] = field(default_factory=dict)   # eid -> Entity
    relations: dict[str, Relation] = field(default_factory=dict)  # rid -> Relation
    # chunk_id -> (alias_lower -> entity_id): the LLM's own scope of resolution
    # for each call. Used for relation resolution first (matches what the LLM
    # produced in that call).
    chunk_local: dict[str, dict[str, str]] = field(default_factory=dict)
    # alias_lower -> set of entity_ids seen doc-wide (for ambiguity detection)
    alias_index: dict[str, set[str]] = field(default_factory=dict)


def aggregate_document_extractions(
    per_chunk_results: list[tuple[str, list[dict], list[dict]]],
    corpus_id: str,
    valid_types: set[str],
    valid_rels: set[str],
) -> DocAggregation:
    """Fold per-chunk LLM extractions into deduplicated entities + relations.

    Args:
        per_chunk_results: list of (chunk_id, entities_raw, relations_raw),
            where the raw items are the LLM's parsed dicts.
        corpus_id: used to compute deterministic entity IDs.
        valid_types: ontology class names; unknown types are coerced to CONCEPT.
        valid_rels: ontology relation names; unknown relations are dropped.

    Returns:
        DocAggregation. Entities are deduped by (type, name.lower()); each
        entity's source_chunks tracks every chunk that extracted it. Relations
        are deduped by (src_id, rel_type, tgt_id); each relation's
        source_chunks tracks every chunk that extracted it, and evidence quotes
        are joined with ' || ' up to 500 chars.
    """
    agg = DocAggregation()

    # Pass 1: entities
    for chunk_id, entities_raw, _ in per_chunk_results:
        chunk_map: dict[str, str] = {}
        for raw in entities_raw:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            canon = _canonicalize_entity(raw, chunk_id, corpus_id, valid_types)
            if canon is None:
                continue
            eid, entity = canon

            existing = agg.entities.get(eid)
            if existing is None:
                agg.entities[eid] = entity
                effective = entity
            else:
                if chunk_id and chunk_id not in existing.source_chunks:
                    existing.source_chunks.append(chunk_id)
                for a in entity.aliases:
                    if a not in existing.aliases:
                        existing.aliases.append(a)
                if len(entity.description) > len(existing.description):
                    existing.description = entity.description
                effective = existing

            canonical_lower = effective.name.lower()
            chunk_map[canonical_lower] = eid
            chunk_map[effective.name] = eid
            agg.alias_index.setdefault(canonical_lower, set()).add(eid)
            for alias in effective.aliases:
                a_lower = alias.lower()
                chunk_map[a_lower] = eid
                chunk_map[alias] = eid
                agg.alias_index.setdefault(a_lower, set()).add(eid)
        agg.chunk_local[chunk_id] = chunk_map

    # Pass 2: relations
    for chunk_id, _, relations_raw in per_chunk_results:
        for raw in relations_raw:
            if not isinstance(raw, dict):
                continue
            src_name = (raw.get("source") or "").strip()
            tgt_name = (raw.get("target") or "").strip()
            if not src_name or not tgt_name:
                continue

            src_id = _resolve_name(src_name, chunk_id, agg)
            tgt_id = _resolve_name(tgt_name, chunk_id, agg)
            if not src_id or not tgt_id:
                continue

            try:
                rtype = sanitize_type_name(raw.get("type", ""))
            except ValueError:
                continue
            if rtype not in valid_rels:
                continue

            rid = make_relation_id(src_id, rtype, tgt_id)
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
            evidence = (raw.get("evidence") or "")[:500]

            existing_rel = agg.relations.get(rid)
            if existing_rel is None:
                agg.relations[rid] = Relation(
                    id=rid,
                    source_entity_id=src_id,
                    target_entity_id=tgt_id,
                    relation_type=rtype,
                    evidence=evidence,
                    confidence=confidence,
                    source_chunks=[chunk_id] if chunk_id else [],
                )
            else:
                if chunk_id and chunk_id not in existing_rel.source_chunks:
                    existing_rel.source_chunks.append(chunk_id)
                if confidence > existing_rel.confidence:
                    existing_rel.confidence = confidence
                if evidence and evidence not in existing_rel.evidence:
                    combined = (
                        existing_rel.evidence + " || " + evidence
                        if existing_rel.evidence
                        else evidence
                    )
                    existing_rel.evidence = combined[:500]

    return agg


def _canonicalize_entity(
    raw: dict, chunk_id: str, corpus_id: str, valid_types: set[str]
) -> tuple[str, Entity] | None:
    try:
        etype = sanitize_type_name(raw.get("type", "CONCEPT"))
    except ValueError:
        etype = "CONCEPT"
    if etype not in valid_types:
        etype = "CONCEPT"

    name = (raw.get("name") or "").strip()
    if not name:
        return None

    aliases_raw = raw.get("aliases", [])
    aliases = (
        [a.strip() for a in aliases_raw if isinstance(a, str) and a.strip()]
        if isinstance(aliases_raw, list)
        else []
    )

    eid = make_entity_id(corpus_id, etype, name)
    entity = Entity(
        id=eid,
        corpus_id=corpus_id,
        name=name,
        entity_type=etype,
        description=raw.get("description", "") or "",
        aliases=aliases,
        source_chunks=[chunk_id] if chunk_id else [],
    )
    return eid, entity


def _resolve_name(name: str, chunk_id: str, agg: DocAggregation) -> str | None:
    """Resolve a name or alias to an entity_id.

    Order of resolution:
      1. Exact match in the chunk-local map — matches the LLM's own scope in
         the call that produced this relation.
      2. Doc-wide alias index; if unambiguous, return directly.
      3. On collision, use topological scoring: for each candidate, count the
         source-chunk overlap with every other entity also present in the
         current chunk. Winner = most co-occurrences with the neighbourhood.
      4. If no candidate wins by a margin, return None — upstream drops the
         relation rather than risk a false merge.
    """
    chunk_map = agg.chunk_local.get(chunk_id, {})
    hit = chunk_map.get(name) or chunk_map.get(name.lower())
    if hit:
        return hit

    candidates = agg.alias_index.get(name.lower(), set())
    if not candidates:
        return None
    if len(candidates) == 1:
        return next(iter(candidates))

    neighbour_eids = set(chunk_map.values())
    best_eid: str | None = None
    best_score = 0
    for cid in candidates:
        cand = agg.entities.get(cid)
        if cand is None:
            continue
        overlap = 0
        for other in neighbour_eids:
            if other == cid:
                continue
            other_ent = agg.entities.get(other)
            if other_ent:
                overlap += len(set(cand.source_chunks) & set(other_ent.source_chunks))
        if overlap > best_score:
            best_score = overlap
            best_eid = cid
    return best_eid if best_score > 0 else None
