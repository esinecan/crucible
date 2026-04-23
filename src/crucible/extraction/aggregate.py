"""In-memory aggregation of per-chunk extractions for a single document.

The extractor runs the LLM per chunk; those raw results are then folded into
deduplicated Entity/Mention/Relation objects before anything touches Neo4j.
This eliminates the chunk-overlap artifacts (duplicate relation rows, inflated
mention counts) and lets us resolve aliases with doc-level context instead of
a flat name_to_id dict.

Mention is the per-chunk provenance record — one per (chunk, entity) pair.
Entity holds canonical/synthesized state only. Same aggregator, two outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Entity, Mention, Relation
from . import make_entity_id, make_mention_id, make_relation_id, sanitize_type_name


@dataclass
class DocAggregation:
    entities: dict[str, Entity] = field(default_factory=dict)   # eid -> Entity
    mentions: dict[str, Mention] = field(default_factory=dict)  # mid -> Mention
    relations: dict[str, Relation] = field(default_factory=dict)  # rid -> Relation
    # chunk_id -> (alias_lower -> entity_id): the LLM's own scope of resolution
    # for each call. Used for relation resolution first (matches what the LLM
    # produced in that call).
    chunk_local: dict[str, dict[str, str]] = field(default_factory=dict)
    # alias_lower -> set of entity_ids seen doc-wide (for ambiguity detection).
    # Populated from Mention records (not Entity.source_chunks, which is gone).
    alias_index: dict[str, set[str]] = field(default_factory=dict)
    # eid -> set of chunk_ids. Used by the topological tiebreaker in
    # _resolve_name; previously read from Entity.source_chunks.
    entity_chunks: dict[str, set[str]] = field(default_factory=dict)


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

    # Pass 1: entities + mentions
    for chunk_id, entities_raw, _ in per_chunk_results:
        chunk_map: dict[str, str] = {}
        for raw in entities_raw:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            canon = _canonicalize_entity(raw, corpus_id, valid_types)
            if canon is None:
                continue
            eid, entity, surface_name, raw_description, raw_aliases = canon

            existing = agg.entities.get(eid)
            if existing is None:
                agg.entities[eid] = entity
                effective = entity
            else:
                # Union aliases on the canonical entity. Description uses
                # longer-wins as a bootstrap — the synthesizer rewrites it
                # later. Source-chunks now live on Mentions, not Entity.
                for a in raw_aliases:
                    if a not in existing.aliases:
                        existing.aliases.append(a)
                if len(raw_description) > len(existing.description):
                    existing.description = raw_description
                effective = existing

            # One Mention per (chunk, entity) pair. Repeated extraction of
            # the same entity in the same chunk collapses here idempotently
            # via the deterministic mention_id.
            mid = make_mention_id(chunk_id, eid)
            existing_m = agg.mentions.get(mid)
            if existing_m is None:
                agg.mentions[mid] = Mention(
                    id=mid,
                    chunk_id=chunk_id,
                    entity_id=eid,
                    name_as_extracted=surface_name,
                    description=raw_description,
                    aliases=list(raw_aliases),
                )
            else:
                # Duplicate dict in the same chunk — keep richer description,
                # union aliases. name_as_extracted keeps its first assignment.
                if len(raw_description) > len(existing_m.description):
                    existing_m.description = raw_description
                for a in raw_aliases:
                    if a not in existing_m.aliases:
                        existing_m.aliases.append(a)

            agg.entity_chunks.setdefault(eid, set()).add(chunk_id)

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
    raw: dict, corpus_id: str, valid_types: set[str]
) -> tuple[str, Entity, str, str, list[str]] | None:
    """Return (eid, Entity, surface_name, description, aliases) on success.

    The Entity carries canonical state (description bootstrapped from this
    extraction, aliases). The trailing tuple elements are also handed back
    raw so the caller can populate the per-chunk Mention with the same
    surface form, description, and aliases it actually saw.
    """
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

    description = raw.get("description", "") or ""

    eid = make_entity_id(corpus_id, etype, name)
    entity = Entity(
        id=eid,
        corpus_id=corpus_id,
        name=name,
        entity_type=etype,
        description=description,
        aliases=list(aliases),
    )
    return eid, entity, name, description, aliases


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
        if cid not in agg.entity_chunks:
            continue
        cand_chunks = agg.entity_chunks[cid]
        overlap = 0
        for other in neighbour_eids:
            if other == cid:
                continue
            other_chunks = agg.entity_chunks.get(other)
            if other_chunks:
                overlap += len(cand_chunks & other_chunks)
        if overlap > best_score:
            best_score = overlap
            best_eid = cid
    return best_eid if best_score > 0 else None
