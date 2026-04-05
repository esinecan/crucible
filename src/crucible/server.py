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


@mcp.tool()
def answer(query: str, max_iterations: int = 20, max_depth: int = 5) -> str:
    """Answer a question using MCTS reasoning over the knowledge graph.

    Builds a reasoning tree: searches for evidence, synthesizes answers,
    challenges them adversarially, and refines. Returns the best grounded
    answer with its evidence chain and reasoning path.

    Args:
        query: The question to answer.
        max_iterations: MCTS iterations (more = better but slower). Default 20.
        max_depth: Maximum reasoning depth. Default 5.
    """
    from .reasoning.mcts import MCTSEngine

    engine = MCTSEngine(config, graph)
    tree = engine.search(query, max_iterations=max_iterations, max_depth=max_depth)

    result = {
        "tree_id": tree.tree_id,
        "query": tree.query,
        "best_score": tree.best_score,
        "best_answer": tree.best_answer,
        "evidence_count": len(tree.evidence_ids),
        "total_nodes": len(tree.nodes),
        "iterations": tree.iterations,
        "path": [
            {
                "depth": n.depth,
                "action": n.action_taken,
                "score": n.avg_score,
                "visits": n.visits,
                "evidence_count": len(n.evidence_ids),
            }
            for n in tree.get_path_to_best()
        ],
    }
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def reasoning_trees(tree_id: str = "") -> str:
    """List or retrieve MCTS reasoning trees.

    Without tree_id: lists all trees with their queries and scores.
    With tree_id: returns full tree detail including all nodes.
    """
    if tree_id:
        nodes = graph.get_reasoning_tree(tree_id)
        return json.dumps(nodes, indent=2, default=str)

    # List all trees (distinct tree_ids)
    with graph._driver.session() as s:
        results = [
            dict(r)
            for r in s.run(
                "MATCH (rn:ReasoningNode) WHERE rn.depth = 0 "
                "RETURN rn.tree_id AS tree_id, rn.query AS query, "
                "rn.score AS root_score, rn.visits AS visits, "
                "rn.created_at AS created_at "
                "ORDER BY rn.created_at DESC LIMIT 20"
            )
        ]
    return json.dumps(results, indent=2, default=str)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
