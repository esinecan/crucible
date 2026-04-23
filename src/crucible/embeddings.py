"""Embedding client for Crucible.

Uses Gemini (gemini-embedding-001 by default, 768-dim MRL) via the
google-genai SDK. Requires GEMINI_API_KEY. A Batch API path is available
for bulk re-embedding (50% cheaper, async).
"""
from __future__ import annotations

import logging
import os
import time

from .config import Config

logger = logging.getLogger(__name__)


# ── Gemini backend ────────────────────────────────────────

def _gemini_embed(texts: list[str], config: Config) -> list[list[float]]:
    from functools import lru_cache

    from google import genai
    from google.genai import types

    @lru_cache(maxsize=1)
    def _client():
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        return genai.Client(api_key=api_key)

    client = _client()
    model = os.environ.get("CRUCIBLE_GEMINI_EMBED_MODEL", "gemini-embedding-001")
    cleaned = [t if t and t.strip() else " " for t in texts]

    while True:
        try:
            result = client.models.embed_content(
                model=model,
                contents=cleaned,
                config=types.EmbedContentConfig(output_dimensionality=config.embed_dim),
            )
            return [list(e.values) for e in result.embeddings]
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                import re
                wait = 30
                m = re.search(r"retry in (\d+)", err)
                if m:
                    wait = int(m.group(1)) + 5
                logger.warning(f"Rate limited, waiting {wait}s")
                time.sleep(wait)
            elif "503" in err or "UNAVAILABLE" in err or "500" in err or "INTERNAL" in err:
                logger.warning(f"Server error, retrying in 10s: {err[:100]}")
                time.sleep(10)
            else:
                raise


# ── Public API ────────────────────────────────────────────

def embed_batch(
    texts: list[str], config: Config, batch_size: int = 20
) -> list[list[float]]:
    """Embed multiple texts. Returns list of 768-dim vectors.

    Batches of batch_size with 1s delay between batches for rate limiting.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        all_embeddings.extend(_gemini_embed(chunk, config))
        if i + batch_size < len(texts):
            time.sleep(1)
    return all_embeddings


def embed_text(text: str, config: Config) -> list[float]:
    """Embed a single text string. Returns a 768-dim vector."""
    return embed_batch([text], config, batch_size=1)[0]


# ── Gemini Batch API (for bulk re-embedding) ──────────────

def batch_embed_via_api(
    items: list[tuple[str, str]],
    config: Config,
    poll_interval: int = 30,
) -> dict[str, list[float]]:
    """Embed many texts via Gemini Batch API. 50% cheaper, no rate limits.

    Args:
        items: list of (key, text) tuples. Key is used to match results back.
        config: Crucible config.
        poll_interval: seconds between status checks.

    Returns:
        dict mapping key -> embedding vector.
    """
    import json
    import tempfile

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)
    model = os.environ.get("CRUCIBLE_GEMINI_EMBED_MODEL", "gemini-embedding-001")

    # Write JSONL
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        jsonl_path = f.name
        for key, text in items:
            cleaned = text.strip() if text and text.strip() else " "
            line = {
                "key": key,
                "request": {
                    "output_dimensionality": config.embed_dim,
                    "content": {"parts": [{"text": cleaned}]},
                },
            }
            f.write(json.dumps(line) + "\n")

    logger.info(f"Uploading {len(items)} items for batch embedding...")
    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(mime_type="jsonl"),
    )

    # Clean up temp file
    os.unlink(jsonl_path)

    logger.info(f"Creating batch embedding job (model={model})...")
    for attempt in range(5):
        try:
            batch_job = client.batches.create_embeddings(
                model=model,
                src=types.EmbeddingsBatchJobSource(file_name=uploaded.name),
            )
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited on batch create (attempt {attempt+1}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError("Failed to create batch job after 5 attempts")
    logger.info(f"Batch job created: {batch_job.name}")

    # Poll
    while True:
        batch_job = client.batches.get(name=batch_job.name)
        state = batch_job.state.name
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            break
        logger.info(f"Batch job state: {state}, waiting {poll_interval}s...")
        time.sleep(poll_interval)

    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch embedding job failed: {state}")

    # Download results
    logger.info("Downloading batch results...")
    result_bytes = client.files.download(file=batch_job.dest.file_name)
    result_text = result_bytes.decode("utf-8")

    results: dict[str, list[float]] = {}
    for line in result_text.splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        key = parsed.get("key", "")
        embedding = parsed.get("response", {}).get("embeddings", [{}])[0].get("values", [])
        if key and embedding:
            results[key] = embedding

    logger.info(f"Batch complete: {len(results)}/{len(items)} embeddings received")
    return results
