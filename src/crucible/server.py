"""Crucible MCP server — exposes the knowledge graph and insight engine to LLMs."""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from .config import Config
from .embeddings import embed_text
from .graph.client import CrucibleGraph

config = Config()
graph = CrucibleGraph(config)
mcp = FastMCP("crucible")


@mcp.tool()
def search(query: str, limit: int = 10, corpus: str = "") -> str:
    """Semantic search across the knowledge graph. Returns chunks ranked by relevance."""
    embedding = embed_text(query, config)
    results = graph.vector_search(
        embedding, limit=limit, corpus_id=corpus or None
    )
    for r in results:
        r.pop("embedding", None)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def fulltext(query: str, limit: int = 10) -> str:
    """Full-text keyword search across chunk content. Supports Lucene syntax."""
    results = graph.fulltext_search(query, limit=limit)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def cypher(query: str) -> str:
    """Execute a read-only Cypher query against the knowledge graph."""
    results = graph.cypher_read(query)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def stats() -> str:
    """Get knowledge graph statistics: corpus count, documents, chunks, insights."""
    return json.dumps(graph.stats(), indent=2)


@mcp.tool()
def get_insights(corpus: str = "", strategy: str = "", limit: int = 10) -> str:
    """Browse insights derived by the insight engine."""
    q = "MATCH (i:Insight) "
    conditions = []
    if corpus:
        conditions.append("i.corpus_id = $cid")
    if strategy:
        conditions.append("i.strategy = $strat")
    if conditions:
        q += "WHERE " + " AND ".join(conditions) + " "
    q += (
        "OPTIONAL MATCH (i)-[:DERIVED_FROM]->(ch:Chunk)-[:PART_OF]->(d:Document) "
        "RETURN i.id AS id, i.text AS text, i.strategy AS strategy, "
        "i.score AS score, i.created_at AS created_at, "
        "collect(DISTINCT d.path) AS source_docs "
        "ORDER BY i.score DESC LIMIT $lim"
    )
    with graph._driver.session() as s:
        results = [
            dict(r) for r in s.run(q, cid=corpus, strat=strategy, lim=limit)
        ]
    return json.dumps(results, indent=2, default=str)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
