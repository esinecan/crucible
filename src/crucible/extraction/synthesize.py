"""Asynchronous entity-description synthesizer.

Phase 1 split per-chunk extraction (Mention) from canonical truth (Entity)
but left Entity.description as a stopgap "longest mention wins" — fine for
bootstrap, terrible for retrieval embeddings. This module rebuilds Entity
descriptions and alias sets from the union of Mentions via an LLM pass.

Design:
- Pull dirty entities (any Mention newer than entity.last_synthesized_at).
- Per dirty entity: gather its Mentions, prompt the model for a canonical
  description + alias set, write back, re-embed, stamp last_synthesized_at.
- Cap at config.max_synthesis_calls; report remaining-dirty count if hit.
- Re-embed in batches of 20 using whatever embed_batch is configured for
  the process — tests stub embeddings via the existing fake_embed fixture.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ..config import Config
from ..embeddings import embed_batch
from ..graph.client import CrucibleGraph
from ..llm_client import LLMClient

logger = logging.getLogger(__name__)


SYNTHESIS_SYSTEM = (
    "You are merging multiple per-chunk descriptions of the same named "
    "entity into one canonical description. Drop redundancy, resolve "
    "conflicts conservatively (prefer specific over vague, dated over "
    "current-tense), keep it under three sentences. Aliases: union all "
    "non-canonical surface forms across the mentions, deduplicated.\n\n"
    'Return JSON only: {"description": "...", "aliases": ["...", "..."]}'
)


def _build_user_prompt(entity_name: str, entity_type: str, mentions: list[dict]) -> str:
    lines = [f'Entity: "{entity_name}" ({entity_type})\n', "Mentions:"]
    for i, m in enumerate(mentions, start=1):
        surface = m.get("name_as_extracted") or entity_name
        desc = (m.get("description") or "").strip()
        aliases = m.get("aliases") or []
        block = [f"  {i}. surface={surface!r}"]
        if desc:
            block.append(f"     description: {desc}")
        if aliases:
            block.append(f"     aliases: {', '.join(aliases)}")
        lines.append("\n".join(block))
    return "\n".join(lines)


class EntitySynthesizer:
    """Batch synthesizer for canonical Entity descriptions."""

    def __init__(
        self,
        config: Config,
        graph: CrucibleGraph,
        llm_client: LLMClient | None = None,
    ):
        self.config = config
        self.graph = graph
        self._llm_client = llm_client or LLMClient(config)

    def synthesize_dirty(
        self,
        corpus_id: str,
        limit: int = 0,
        max_calls: int | None = None,
    ) -> dict[str, int]:
        """Synthesize all dirty entities (or up to `limit`).

        max_calls overrides config.max_synthesis_calls. When reached, exit
        cleanly and report remaining dirty count via stats['remaining'].
        """
        budget = max_calls if max_calls is not None else self.config.max_synthesis_calls
        dirty = self.graph.get_dirty_entities(corpus_id, limit=limit)
        stats = {
            "dirty_total": len(dirty),
            "synthesized": 0,
            "errors": 0,
            "embedded": 0,
            "remaining": 0,
        }

        if not dirty:
            return stats

        to_embed: list[tuple[str, str]] = []  # (entity_id, description)
        processed_count = 0

        for ent in dirty:
            if processed_count >= budget:
                stats["remaining"] = stats["dirty_total"] - stats["synthesized"]
                logger.info(
                    "Budget exhausted at %d calls; %d entities remain dirty",
                    budget, stats["remaining"],
                )
                break

            mentions = self.graph.get_mentions_for_entity(ent["id"])
            if not mentions:
                continue

            try:
                merged = self._synthesize_one(ent, mentions)
            except Exception as exc:
                logger.warning(
                    "Synthesis failed for entity %s (%s): %s",
                    ent["id"][:12], ent["name"], exc,
                )
                stats["errors"] += 1
                processed_count += 1
                continue

            now = datetime.now(timezone.utc).isoformat()
            self.graph.update_entity_synthesis(
                ent["id"], merged["description"], merged["aliases"], now,
            )
            to_embed.append((ent["id"], merged["description"]))
            stats["synthesized"] += 1
            processed_count += 1

        if to_embed:
            stats["embedded"] = self._embed_in_batches(to_embed)

        return stats

    def _synthesize_one(self, entity: dict, mentions: list[dict]) -> dict[str, Any]:
        """One LLM call → {description, aliases}. Falls back to bootstrap on
        malformed output: longest mention.description, union of aliases."""
        user = _build_user_prompt(entity["name"], entity["entity_type"], mentions)
        result = self._llm_client.chat_json(
            self.config.domain_preamble + SYNTHESIS_SYSTEM,
            user,
            temperature=0.2,
            max_tokens=600,
        )

        description = ""
        aliases: list[str] = []
        if isinstance(result, dict):
            description = (result.get("description") or "").strip()
            raw_aliases = result.get("aliases") or []
            if isinstance(raw_aliases, list):
                aliases = [
                    a.strip() for a in raw_aliases
                    if isinstance(a, str) and a.strip()
                ]

        if not description:
            # LLM returned nothing usable — bootstrap the same way Entity.upsert
            # does: longest seen description, union of aliases.
            description = max(
                (m.get("description") or "" for m in mentions),
                key=len,
                default="",
            )
        if not aliases:
            seen: set[str] = set()
            for m in mentions:
                for a in (m.get("aliases") or []):
                    if isinstance(a, str) and a.strip() and a not in seen:
                        seen.add(a)
                        aliases.append(a)

        return {"description": description, "aliases": aliases}

    def _embed_in_batches(
        self, items: list[tuple[str, str]], batch_size: int = 20,
    ) -> int:
        """Re-embed `name + description` per entity, write via set_entity_embeddings.

        Returns the count of embeddings actually written. Errors in one batch
        do not stop subsequent batches.
        """
        # Pull names once so the embedded text matches what extract uses
        # ("name: description"). Avoids a per-entity round trip.
        ids = [eid for eid, _ in items]
        with self.graph._driver.session() as s:
            rows = {
                r["id"]: r["name"]
                for r in s.run(
                    "MATCH (e:Entity) WHERE e.id IN $ids "
                    "RETURN e.id AS id, e.name AS name",
                    ids=ids,
                )
            }

        written = 0
        for i in range(0, len(items), batch_size):
            batch = items[i: i + batch_size]
            texts = [f"{rows.get(eid, '')}: {desc}" for eid, desc in batch]
            try:
                embeddings = embed_batch(texts, self.config, batch_size=batch_size)
            except Exception as exc:
                logger.warning("Embedding batch failed: %s", exc)
                continue
            self.graph.set_entity_embeddings(
                [(eid, emb) for (eid, _), emb in zip(batch, embeddings)]
            )
            written += len(batch)
        return written
