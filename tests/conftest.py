"""Shared test fixtures.

Unit tests should not touch Neo4j, Gemini, or DeepSeek. Integration tests use
a throwaway Neo4j container via testcontainers and stub out Gemini.
"""
from __future__ import annotations

import os
import socket
import time

import pytest

from crucible.config import Config


# ── Safety: block accidental network calls in unit tests ─────

@pytest.fixture
def no_gemini(monkeypatch):
    """Fail loudly if _gemini_embed is called in a test that mounts this."""
    from crucible import embeddings

    def _explode(*args, **kwargs):
        raise AssertionError("unit test tried to call _gemini_embed")

    monkeypatch.setattr(embeddings, "_gemini_embed", _explode)


# ── Deterministic fake embeddings for integration tests ─────

def _fake_vector(text: str, dim: int = 768) -> list[float]:
    """Deterministic pseudo-embedding derived from text. Unit vectors.

    Uses a tiny hash scheme so tests are reproducible without calling Gemini.
    Similar texts map near each other: same text => identical vector.
    """
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(h[i % len(h)] - 128) / 128.0 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


@pytest.fixture
def fake_embed(monkeypatch):
    """Stub embeddings.embed_text / embed_batch to deterministic vectors."""
    from crucible import embeddings

    def _embed_batch(texts, config, batch_size=20):
        return [_fake_vector(t, config.embed_dim) for t in texts]

    def _embed_text(text, config):
        return _fake_vector(text, config.embed_dim)

    monkeypatch.setattr(embeddings, "embed_batch", _embed_batch)
    monkeypatch.setattr(embeddings, "embed_text", _embed_text)
    # Also patch in modules that imported at module-load
    from crucible.ingestion import pipeline as _pipeline
    from crucible.reasoning import actions as _actions
    monkeypatch.setattr(_pipeline, "embed_batch", _embed_batch, raising=False)
    monkeypatch.setattr(_actions, "embed_text", _embed_text, raising=False)


# ── Neo4j via testcontainers (session-scoped) ───────────────

def _neo4j_available() -> bool:
    """Quick check whether Docker is reachable (for skipping integration tests)."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def neo4j_container():
    """Spin up Neo4j 5.26 once per test session.

    Uses raw DockerContainer rather than Neo4jContainer because the latter's
    log-based readiness wait is fragile under resource pressure (its predicate
    string isn't always emitted within its internal timeout). Bolt-polling is
    the actual ground truth for readiness.
    """
    if not _neo4j_available():
        pytest.skip("Docker not reachable; skipping integration tests")

    from neo4j import GraphDatabase
    from testcontainers.core.container import DockerContainer

    password = "test-pass"
    container = (
        DockerContainer("neo4j:5.26-community")
        .with_env("NEO4J_AUTH", f"neo4j/{password}")
        .with_exposed_ports(7687, 7474)
    )
    container.start()

    bolt_host = container.get_container_host_ip()
    bolt_port = container.get_exposed_port(7687)
    bolt_url = f"bolt://{bolt_host}:{bolt_port}"

    last_err: Exception | None = None
    for _ in range(300):  # up to 5 minutes for slow Docker / first-pull
        try:
            drv = GraphDatabase.driver(bolt_url, auth=("neo4j", password))
            with drv.session() as s:
                s.run("RETURN 1").single()
            drv.close()
            break
        except Exception as e:
            last_err = e
            time.sleep(1)
    else:
        status = "unknown"
        try:
            status = container.get_wrapped_container().status
        except Exception:
            pass
        container.stop()
        pytest.fail(
            f"Neo4j bolt not ready in 300s (status={status}, last_err={last_err})"
        )

    yield {"uri": bolt_url, "user": "neo4j", "password": password}
    container.stop()


@pytest.fixture
def config(neo4j_container) -> Config:
    """Config pointing at the test Neo4j container."""
    return Config(
        neo4j_uri=neo4j_container["uri"],
        neo4j_user=neo4j_container["user"],
        neo4j_password=neo4j_container["password"],
    )


@pytest.fixture
def graph(config):
    """Fresh CrucibleGraph per test; wipes all data before yielding."""
    from crucible.graph.client import CrucibleGraph

    g = CrucibleGraph(config)
    with g._driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    yield g
    g.close()
