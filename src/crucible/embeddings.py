from __future__ import annotations

import httpx

from .config import Config


def embed_batch(
    texts: list[str], config: Config, batch_size: int = 200
) -> list[list[float]]:
    cleaned = [t if t and t.strip() else " " for t in texts]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        try:
            resp = httpx.post(
                f"{config.ollama_url}/api/embed",
                json={"model": config.embed_model, "input": batch},
                timeout=120.0,
            )
            resp.raise_for_status()
            all_embeddings.extend(resp.json()["embeddings"])
        except Exception:
            for t in batch:
                try:
                    r = httpx.post(
                        f"{config.ollama_url}/api/embed",
                        json={"model": config.embed_model, "input": [t]},
                        timeout=30.0,
                    )
                    r.raise_for_status()
                    all_embeddings.extend(r.json()["embeddings"])
                except Exception:
                    all_embeddings.append([0.0] * config.embed_dim)

    return all_embeddings


def embed_text(text: str, config: Config) -> list[float]:
    return embed_batch([text], config)[0]
