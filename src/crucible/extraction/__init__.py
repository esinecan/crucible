"""Entity-relation extraction: ontology generation and ontology-guided extraction."""
from __future__ import annotations

import hashlib
import re

_VALID_TYPE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def sanitize_type_name(raw: str) -> str:
    """Convert a raw string to a safe Neo4j relationship/entity type.

    Only allows A-Z, 0-9, underscore. Must start with a letter.
    Raises ValueError if the result is empty after sanitization.
    """
    cleaned = raw.upper().replace(" ", "_").replace("-", "_")
    cleaned = re.sub(r"[^A-Z0-9_]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if cleaned and not cleaned[0].isalpha():
        cleaned = "REL_" + cleaned
    if not cleaned or not _VALID_TYPE.match(cleaned):
        raise ValueError(f"Cannot sanitize type name: {raw!r} -> {cleaned!r}")
    return cleaned


sanitize_rel_type = sanitize_type_name
sanitize_entity_type = sanitize_type_name


def make_entity_id(corpus_id: str, entity_type: str, name: str) -> str:
    """Deterministic entity ID from corpus + type + normalized name."""
    raw = ":".join([corpus_id, entity_type, name.lower().strip()])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def make_relation_id(src_id: str, rel_type: str, tgt_id: str) -> str:
    """Deterministic relation ID from source + type + target.

    A relation is a claim, not a per-chunk observation. Chunk provenance lives
    on the Relation.source_chunks list, not in the identity.
    """
    raw = ":".join([src_id, rel_type, tgt_id])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]
