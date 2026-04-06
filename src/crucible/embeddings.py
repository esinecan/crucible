"""Embedding client for Crucible.

Supports two backends:
  - Gemini (default): gemini-embedding-001, 768-dim MRL, requires GEMINI_API_KEY
  - Ollama (fallback): local nomic-embed-text, 768-dim, no API key needed

Set CRUCIBLE_EMBED_BACKEND=ollama to use Ollama. Default is gemini.
"""
from __future__ import annotations

import logging
import os
import time

import httpx

from .config import Config

logger = logging.getLogger(__name__)

EMBED_BACKEND = os.environ.get("CRUCIBLE_EMBED_BACKEND", "gemini")


# ── Gemini backend ────────────────────────────────────────

def _gemini_embed(texts: list[str], config: Config) -> list[list[float]]:
    from functools import lru_cache

    from google import genai
    from google.genai import types

    @lru_cache(maxsize=1)
    def _client():
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Set it or use CRUCIBLE_EMBED_BACKEND=ollama")
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


# ── Ollama backend ────────────────────────────────────────

def _ollama_embed(texts: list[str], config: Config) -> list[list[float]]:
    cleaned = [t if t and t.strip() else " " for t in texts]
    try:
        resp = httpx.post(
            f"{config.ollama_url}/api/embed",
            json={"model": config.embed_model, "input": cleaned},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]
    except Exception:
        logger.warning("Ollama batch failed, falling back to one-by-one")
        all_embs: list[list[float]] = []
        for t in cleaned:
            try:
                r = httpx.post(
                    f"{config.ollama_url}/api/embed",
                    json={"model": config.embed_model, "input": [t]},
                    timeout=30.0,
                )
                r.raise_for_status()
                all_embs.extend(r.json()["embeddings"])
            except Exception:
                all_embs.append([0.0] * config.embed_dim)
        return all_embs


# ── Public API ────────────────────────────────────────────

def embed_batch(
    texts: list[str], config: Config, batch_size: int = 20
) -> list[list[float]]:
    """Embed multiple texts. Returns list of 768-dim vectors.

    Gemini: batches of batch_size with 1s delay for rate limiting.
    Ollama: batches of 200 (local, no rate limits).
    """
    if not texts:
        return []

    backend = EMBED_BACKEND

    if backend == "gemini":
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            all_embeddings.extend(_gemini_embed(chunk, config))
            if i + batch_size < len(texts):
                time.sleep(1)
        return all_embeddings

    # Ollama: larger batches, no delay
    all_embeddings = []
    for i in range(0, len(texts), 200):
        chunk = texts[i: i + 200]
        all_embeddings.extend(_ollama_embed(chunk, config))
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
