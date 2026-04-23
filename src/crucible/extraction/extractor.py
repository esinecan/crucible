"""Ontology-guided entity and relation extraction from chunk text."""
from __future__ import annotations

import logging
import time

import httpx
from tqdm import tqdm

from ..config import Config
from ..embeddings import embed_batch
from ..graph.client import CrucibleGraph
from ..llm_client import LLMClient, parse_json_lenient
from ..models import Entity
from .aggregate import aggregate_document_extractions
from .ontology import Ontology

logger = logging.getLogger(__name__)


class TransientExtractionError(Exception):
    """An error that should NOT mark a chunk as extracted. The chunk will be
    retried on a subsequent run (network blip, rate limit, 5xx).
    """


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRYABLE_EXC = (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)


def _is_transient(exc: BaseException) -> bool:
    """True for errors we'd want to retry on a later run rather than poison."""
    if isinstance(exc, TransientExtractionError):
        return True
    if isinstance(exc, _RETRYABLE_EXC):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


EXTRACTION_SYSTEM = """{domain}You are an entity-relation extractor. Given a text chunk and an ontology, extract all entities and relations present in the text.

{ontology_prompt}

Rules:
1. Only extract entities whose type matches one of the ENTITY TYPES above
2. Only extract relations whose type matches one of the RELATION TYPES above
3. Entity names should be canonical (full proper name, not abbreviation) when possible
4. Include aliases if the text uses alternate forms
5. Provide a brief evidence quote for each relation
6. Confidence: 0.9+ for explicitly stated, 0.6-0.8 for strongly implied, 0.3-0.5 for weakly implied
7. Extract 0-8 entities per chunk. Skip trivial mentions.
8. If no entities or relations are found, return empty arrays

Return JSON only:
{{
  "entities": [
    {{"name": "canonical name", "type": "ENTITY_TYPE", "description": "one sentence", "aliases": ["alt"]}}
  ],
  "relations": [
    {{"source": "entity name", "target": "entity name", "type": "RELATION_TYPE", "evidence": "quote from text", "confidence": 0.9}}
  ]
}}"""


class EntityExtractor:
    def __init__(self, config: Config, graph: CrucibleGraph, ontology: Ontology):
        self.config = config
        self.graph = graph
        self.ontology = ontology
        self._ontology_prompt = ontology.to_extraction_prompt()
        self._valid_types = ontology.class_names
        self._valid_rels = ontology.relation_names
        self._llm_client = LLMClient(config)

    def extract_corpus(
        self,
        corpus_id: str,
        batch_size: int = 50,  # retained for API compat; not used in doc-at-a-time flow
        embed: bool = True,
        limit: int = 0,
        workers: int = 10,
    ) -> dict:
        """Extract entities and relations from all unprocessed chunks.

        Processes one document at a time: chunks for that document are
        extracted in parallel, aggregated in memory (deduplicating entities
        and relations across overlap, unioning source_chunks), then persisted
        as one batch. Only successful + permanently-failed chunks are marked
        as extracted — transient errors leave the chunk un-marked so the next
        run retries it.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        stats = {
            "documents": 0, "chunks": 0, "entities": 0,
            "relations": 0, "errors": 0,
        }

        docs = self.graph.get_unextracted_docs(corpus_id)
        processed_chunks = 0

        for doc_row in docs:
            if limit and processed_chunks >= limit:
                break

            doc_id = doc_row["doc_id"]
            chunks = self.graph.get_unextracted_chunks_for_doc(doc_id, corpus_id)
            if not chunks:
                continue
            if limit:
                remaining = max(0, limit - processed_chunks)
                if remaining == 0:
                    break
                chunks = chunks[:remaining]

            per_chunk: list[tuple[str, list[dict], list[dict]]] = []
            successful: list[dict] = []
            permanent_err: list[dict] = []

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._extract_chunk_raw, c, corpus_id): c
                    for c in chunks
                }
                with tqdm(
                    total=len(futures),
                    desc=f"Extracting {doc_row.get('path', doc_id)[:40]}",
                    leave=False,
                ) as pbar:
                    for future in as_completed(futures):
                        chunk = futures[future]
                        try:
                            ents_raw, rels_raw = future.result()
                            per_chunk.append((chunk["id"], ents_raw, rels_raw))
                            successful.append(chunk)
                            stats["chunks"] += 1
                        except Exception as exc:
                            stats["errors"] += 1
                            if _is_transient(exc):
                                logger.warning(
                                    "Transient extraction error on %s: %s",
                                    chunk["id"][:12], exc,
                                )
                            else:
                                logger.warning(
                                    "Permanent extraction error on %s: %s",
                                    chunk["id"][:12], exc,
                                )
                                permanent_err.append(chunk)
                        pbar.update(1)

            agg = aggregate_document_extractions(
                per_chunk, corpus_id, self._valid_types, self._valid_rels,
            )

            if embed and agg.entities:
                self._embed_entities(list(agg.entities.values()))

            self._persist_aggregation(agg, corpus_id)

            # Mark successful AND permanent-error chunks so the next run
            # doesn't retry them. Transient-error chunks are intentionally
            # left unmarked.
            for chunk in successful + permanent_err:
                try:
                    self.graph.mark_chunk_extracted(chunk["id"], corpus_id)
                except Exception:
                    pass

            stats["documents"] += 1
            stats["entities"] += len(agg.entities)
            stats["relations"] += len(agg.relations)
            processed_chunks += len(chunks)

        return stats

    def _extract_chunk_raw(
        self, chunk: dict, corpus_id: str
    ) -> tuple[list[dict], list[dict]]:
        """Call the LLM on one chunk and return raw parsed entities/relations.

        Persistence and aggregation are the caller's responsibility. Returning
        dicts (not Entity/Relation objects) lets the aggregator resolve aliases
        with doc-level context instead of chunk-local scope.
        """
        system = EXTRACTION_SYSTEM.format(
            domain=self.config.domain_preamble,
            ontology_prompt=self._ontology_prompt,
        )
        doc_path = chunk.get("doc_path", "unknown")
        user = f"Text chunk (from {doc_path}):\n\n{chunk['text']}"

        resp = self._llm(system, user, max_tokens=1000)
        parsed = parse_json_lenient(resp)
        if not parsed or not isinstance(parsed, dict):
            return [], []

        ents = parsed.get("entities", []) or []
        rels = parsed.get("relations", []) or []
        return (
            [e for e in ents if isinstance(e, dict)],
            [r for r in rels if isinstance(r, dict)],
        )

    def _embed_entities(self, entities: list[Entity]) -> None:
        texts = [f"{e.name}: {e.description}" for e in entities]
        embeddings = embed_batch(texts, self.config)
        for entity, emb in zip(entities, embeddings):
            entity.embedding = emb

    def _persist_aggregation(self, agg, corpus_id: str) -> None:
        """Write aggregated entities, mentions, and relations.

        Order matters: entities first (Mentions reference them via
        RESOLVES_TO), then mentions (also requires the chunks to exist —
        they were upserted earlier in the ingestion pipeline), then relations
        (between entities, doesn't depend on Mentions). Each upsert method is
        individually transactional; a partial failure here leaves a
        consistent-but-incomplete document.
        """
        for entity in agg.entities.values():
            self.graph.upsert_entity(entity)

        for mention in agg.mentions.values():
            self.graph.upsert_mention(mention)

        for relation in agg.relations.values():
            self.graph.upsert_relation(relation)

    def _llm(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """Call DeepSeek with 3 attempts on retryable errors (429/5xx/network).

        Non-retryable HTTP errors (auth, 4xx) and malformed payloads raise
        without retry. On exhaustion the original transient error propagates.
        Wraps LLMClient.chat — the retry policy is extractor-domain logic
        (chunk-level recoverability) and stays out of the shared client.
        """
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return self._llm_client.chat(
                    system, user, temperature=0.2, max_tokens=max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                if _is_transient(exc) and attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(
                        "DeepSeek transient error (attempt %d/3), retrying in %ds: %s",
                        attempt + 1, wait, exc,
                    )
                    time.sleep(wait)
                    continue
                raise
        # Should be unreachable (loop either returns or raises), but keep mypy happy.
        raise last_exc  # type: ignore[misc]
