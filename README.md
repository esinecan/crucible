# Crucible

Universal knowledge base builder with two reasoning modes: **Insight Discovery** (pattern finding) and **Answer Search** (MCTS-based reasoning).

Ingest any corpus. Discover patterns. Answer questions. No domain-specific config required.

## Quick Start

```bash
# Prerequisites: Docker, Python 3.12+, GEMINI_API_KEY (and DEEPSEEK_API_KEY for eval/synthesis)

# 1. Start Neo4j
cd ~/dev5/crucible
docker compose up -d

# 2. Install
pip install -e .

# 3. Ingest a corpus
python -m crucible ingest ./my-notes --name my-kb

# 4. Discover patterns
python -m crucible explore --cycles 5 --eval

# 5. Answer questions
python -m crucible answer "What do I think about X?"
```

## Architecture

Crucible has two reasoning modes that share a graph and reinforce each other:

### Insight Mode (Exploration)

Discovers non-obvious patterns in your corpus using a **Thompson Sampling bandit** that selects between six strategies:

| Strategy | What it finds | How |
|----------|--------------|-----|
| **bridge** | Similar content across different documents | Cosine similarity on embeddings, cross-doc pairs |
| **outlier** | Anomalous content unlike the rest | Isolation score (1 - avg similarity to peers) |
| **hub** | Central chunks connecting many documents | Connectivity count above similarity threshold |
| **meta** | Patterns across L1 insights (L2) | Cross-strategy bridge on insight embeddings |
| **contradiction** | Incompatible claims about the same thing | LLM evaluation of high-similarity cross-doc pairs |
| **gap** | Referenced but unexplained concepts | LLM identifies missing knowledge from samples |

The bandit learns which strategies work best for your corpus. After each cycle, insight scores feed back as Bernoulli rewards — Beta(α,β) posteriors sharpen over time. Strategies that produce high-quality insights get selected more often.

**Adversarial evaluation** (optional, `--eval` flag): each insight goes through a three-role debate:
1. **Advocate** argues why the insight is valuable
2. **Skeptic** argues why it's not
3. **Referee** scores novelty, relevance, and actionability (0-1 each)

The debate score replaces the heuristic score and feeds the bandit.

### Answer Mode (Reasoning)

Given a question, builds a **reasoning tree** using **Monte Carlo Tree Search (MCTS)** with **UCT** (Upper Confidence bounds applied to Trees) for node selection.

Each iteration:
1. **Selection**: UCT balances exploitation (high-scoring paths) with exploration (under-visited paths)
2. **Expansion**: An action produces a child node:
   - `search` — vector + fulltext query against the graph
   - `follow_ref` — traverse graph edges from evidence chunks
   - `challenge` — adversarial prosecution of the current partial answer
   - `synthesize` — build/refine answer from accumulated evidence
3. **Simulation**: Referee evaluates the partial answer at this node
4. **Backpropagation**: Score propagates up to root, updating visit counts and averages

Actions are selected by an **Action Bandit** (UCB1), which learns from per-iteration scores which actions improve answer quality. Hard constraints apply first (must search before synthesize, must have an answer before challenge), then UCB1 over eligible actions.

The result: a grounded answer with an evidence chain, where every claim was stress-tested by the challenge action and scored by the referee.

### Feedback Loop

Answer mode feeds back into insight mode:
- Insights used as evidence in high-scoring answers get bonus rewards in the bandit
- This shifts Thompson Sampling posteriors toward strategies that produce *useful* insights (not just interesting ones)

```
Insight Mode ──discovers──→ Insights stored in graph
                                    │
Answer Mode ──queries────→ Chunks + Insights as evidence
         │
         └── scores ──→ bonus reward to contributing insights
                              │
                              └── bandit posteriors shift
```

## Graph Schema

```
Corpus
  └── Document (BELONGS_TO)
        └── Chunk (PART_OF, linked by NEXT)
              └── Insight L1 (DERIVED_FROM chunk)
                    └── Insight L2 (DERIVED_FROM insight)

ReasoningNode (CHILD_OF parent, USES_EVIDENCE → chunk/insight)
```

**Node types:**
- `Corpus` — a named collection of documents
- `Document` — a file (markdown, JSON, text, code)
- `Chunk` — a section of a document with a 768-dim embedding (gemini-embedding-001, MRL-truncated)
- `Insight` — a discovered pattern (L1 from chunks, L2 from L1 insights)
- `ReasoningNode` — a node in an MCTS reasoning tree

**Indexes:**
- Vector (cosine, 768-dim): `chunk_embedding`, `insight_embedding`
- Fulltext (Lucene): `chunk_text`, `insight_text`
- Property: document corpus/path, chunk document, insight corpus/strategy/layer, reasoning tree/depth

## CLI Reference

```bash
# Ingest a directory into the knowledge graph
python -m crucible ingest <path> --name <corpus_name> [--description "..."]

# Run insight discovery cycles
python -m crucible explore [--cycles N] [--max N] [--strategy NAME] [--eval]
#   --cycles: number of exploration cycles (default 1)
#   --max: max insights per cycle (default 5)
#   --strategy: force a strategy (bridge/outlier/hub/meta/contradiction/gap)
#   --eval: enable adversarial evaluation (requires DEEPSEEK_API_KEY)

# Answer a question via MCTS reasoning
python -m crucible answer "your question" [--iterations N] [--depth N] [--feedback]
#   --iterations: max MCTS iterations (default 20)
#   --depth: max tree depth (default 5)
#   --feedback: feed answer scores back to insight bandit

# Show graph statistics
python -m crucible stats

# Show Thompson Sampling bandit posteriors
python -m crucible bandit

# Backfill embeddings on insights
python -m crucible embed-insights

# Start MCP server (stdio transport)
python -m crucible serve
```

## Configuration

All config via environment variables (yaml config file planned):

| Variable | Default | Purpose |
|----------|---------|---------|
| `CRUCIBLE_NEO4J_URI` | `bolt://localhost:7689` | Neo4j Bolt endpoint |
| `CRUCIBLE_NEO4J_USER` | `neo4j` | Neo4j username |
| `CRUCIBLE_NEO4J_PASSWORD` | `nous-dev` | Neo4j password |
| `CRUCIBLE_GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `CRUCIBLE_EVAL_MODEL` | `deepseek-chat` | LLM for evaluation/synthesis |
| `CRUCIBLE_DOMAIN` | (empty) | Domain context injected into all LLM prompts |
| `GEMINI_API_KEY` | (required) | Gemini API key for embeddings |
| `DEEPSEEK_API_KEY` | (required for eval) | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API endpoint |

## Key Concepts

### Thompson Sampling (Bandit)

Each insight strategy is an "arm" with a Beta(α, β) distribution representing belief about its quality. To select which strategy to run:
1. Sample a random value from each arm's Beta distribution
2. Pick the arm with the highest sample

High-quality strategies get tight distributions around high values (exploited often). Under-explored strategies have wide distributions that occasionally sample high (explored naturally). No explicit explore/exploit switch needed.

**Bernoulli update**: referee scores (continuous 0-1) are converted to binary via a coin flip biased by the score. This feeds the Beta distribution: heads → α+1, tails → β+1.

### MCTS with UCT

Standard Monte Carlo Tree Search adapted for knowledge graph reasoning. UCT formula for node selection:

```
UCT(node) = avg_score / visits + c * sqrt(ln(parent_visits) / visits)
```

Where `c` (default 1.41 ≈ √2) controls the exploration/exploitation balance. Unvisited nodes get infinite UCT score (always explored first).

### Referee (Pluggable)

The referee determines what "good" means. Three implementations:
- **LLMReferee** (default): Advocate/Skeptic/Referee three-role debate via DeepSeek
- **SymbolicReferee**: Deterministic evaluation (domain adapters, e.g., legal predicate matching)
- **HybridReferee**: Symbolic first, LLM fallback

Referee quality directly determines bandit learning quality. A better referee → better reward signal → faster convergence on the best strategies.

## MCP Server

Exposes 5 tools via stdio transport:

| Tool | Purpose |
|------|---------|
| `search(query, limit, corpus)` | Semantic vector search |
| `fulltext(query, limit)` | Lucene fulltext search |
| `cypher(query)` | Read-only Cypher queries |
| `stats()` | Graph statistics |
| `get_insights(corpus, strategy, limit)` | Browse discovered insights |

## File Structure

```
src/crucible/
├── __init__.py
├── __main__.py              # CLI entry point
├── config.py                # Configuration (env vars, MCTS params)
├── models.py                # Data models (Corpus, Document, Chunk, Insight, ReasoningNode)
├── embeddings.py            # Gemini embedding client (sync + Batch API)
├── server.py                # MCP server
├── graph/
│   ├── client.py            # Neo4j CRUD, search, tree persistence
│   └── schema.py            # Indexes, constraints
├── ingestion/
│   ├── pipeline.py          # Resumable directory ingestion
│   ├── chunker.py           # Sentence-boundary-aware chunking
│   └── parsers.py           # File format parsers (md, json, txt, code)
├── insight/
│   ├── engine.py            # Thompson Sampling + 6 strategies + feedback loop
│   └── evaluator.py         # Adversarial Advocate/Skeptic/Referee debate
└── reasoning/
    ├── mcts.py              # MCTS engine + ReasoningTree
    ├── actions.py           # Expansion actions (search, follow_ref, challenge, synthesize)
    └── referee.py           # Pluggable referee interface (LLM, Symbolic, Hybrid)
```

## Proven Domains

| Domain | Corpus | Chunks | Insights | Key Feature |
|--------|--------|--------|----------|-------------|
| **Yepis** (geopolitics) | 104 docs | 4,772 | 76 L1, 10 L2 | Current events → essay synthesis |
| **Burokrat** (German law) | 6,862 laws | 89K norms | — | Symbolic predicate evaluation |
| **Agent-KB** (Forto platform) | 221 docs | 17,540 | 345 L1, 25 L2 | Team knowledge patterns |
