# Crucible Onboarding — What You Need to Know

This doc is for Eren picking up Crucible in a new session. It covers what the system does, how the pieces fit together, and where the work left off.

## What Crucible Is

A knowledge graph engine with three modes:
- **Insight mode** — discovers patterns in your corpus using a Thompson Sampling bandit over six strategies (bridge, outlier, hub, meta, contradiction, gap).
- **Extraction mode** — generates an ontology from those insights, then extracts typed entities + relations from chunks. Per-chunk Mentions sit between Chunks and Entities; an async synthesizer merges Mentions into canonical Entity descriptions.
- **Answer mode** — reasons toward answers via MCTS over five expansion actions (search, follow_ref, challenge, synthesize, entity_walk).

All three share a Neo4j graph. Insight discoveries feed extraction (ontology); extraction enriches insight strategies (entity-aware bridge/hub/contradiction/gap channels) and answer evidence (entity_walk action). Answer scores feed back to the insight bandit.

## How It Works (Plain English)

### Insight Mode

You have a corpus (essays, notes, transcripts, whatever). Crucible chunks it, embeds it, stores it in Neo4j. Then it runs exploration cycles:

1. **Bandit picks a strategy** (bridge, outlier, hub, meta, contradiction, gap)
2. **Strategy samples chunks** and looks for a pattern (e.g., bridge finds similar content across different documents)
3. **Evaluator scores the finding** (optionally via a three-way LLM debate: advocate argues it's valuable, skeptic argues it's not, referee scores)
4. **Score feeds back to the bandit** as a Bernoulli coin flip — the bandit learns which strategies produce good findings for this corpus

Over time: strategies that produce high-quality insights get selected more often. The system self-tunes.

### Answer Mode

You ask a question. MCTS builds a tree of reasoning paths:

1. **UCT selects** which node to expand (balances "this path looks good" with "this path is under-explored")
2. **Action bandit picks** an expansion action:
   - `search` — find relevant evidence in the graph
   - `follow_ref` — follow edges from existing evidence
   - `challenge` — adversarially attack the current answer
   - `synthesize` — build/refine an answer from evidence
3. **Referee evaluates** the partial answer at the new node
4. **Score backpropagates** up the tree

After N iterations, you get the best answer with its evidence chain.

### Feedback Loop

When answer mode uses insights as evidence and the answer scores well, those insights get bonus rewards in the insight bandit. This teaches the bandit to favor strategies that produce insights useful for answering questions, not just interesting ones.

## The Algorithms

| Name | What | Where |
|------|------|-------|
| **Thompson Sampling** | Picks which insight strategy to run. Beta-Bernoulli: each arm has Beta(a,b), sample from each, pick argmax. | `insight/engine.py` ThompsonBandit |
| **UCT** | Picks which MCTS node to expand. avg_score/visits + c*sqrt(ln(parent_visits)/visits). | `reasoning/mcts.py` MCTSEngine._uct_score |
| **UCB1** | Picks which expansion action to use. Same formula as UCT but over actions, not nodes. | `reasoning/mcts.py` ActionBandit |
| **Bernoulli** | Converts continuous referee scores (0-1) to binary for Beta updates. Coin flip biased by score. | `insight/engine.py` ThompsonBandit.update |
| **Adversarial Eval** | Three-role debate: Advocate/Skeptic/Referee. Replaces heuristic scoring. | `insight/evaluator.py` AdversarialEvaluator |

## Running It

### Prerequisites
- Docker (for Neo4j)
- Python 3.12+ with venv at `~/dev5/crucible/.venv/`
- `GEMINI_API_KEY` env var (for embeddings)
- `DEEPSEEK_API_KEY` env var (for evaluation and synthesis)

### Active Instances

| Instance | Neo4j Ports | Container | Password |
|----------|-------------|-----------|----------|
| Crucible (agent-kb) | 7476/7689 | nous-neo4j-1 | nous-dev |
| Burokrat (German law) | 7475/7688 | burokrat-neo4j | (check compose) |
| Yepis (geopolitics) | 7477/7690 | yepis-neo4j | yepis-dev |

### Common Commands

```bash
cd ~/dev5/crucible

# Yepis example (set env vars for the instance you want)
export CRUCIBLE_NEO4J_URI=bolt://localhost:7690
export CRUCIBLE_NEO4J_PASSWORD=yepis-dev
export CRUCIBLE_DOMAIN="Turkish political commentary knowledge base."

# Check stats
.venv/bin/python -m crucible stats

# Run insight discovery (5 cycles, adversarial eval is on by default)
.venv/bin/python -m crucible explore --cycles 5

# Generate an ontology from the discovered insights
.venv/bin/python -m crucible ontology --corpus my-kb --output my-ontology.json

# Extract typed entities + relations from chunks
.venv/bin/python -m crucible extract --corpus my-kb --ontology my-ontology.json --workers 10

# Synthesize canonical entity descriptions from per-chunk Mentions
.venv/bin/python -m crucible synthesize-entities --corpus my-kb

# Answer a question (10 iterations, depth 4)
.venv/bin/python -m crucible answer "your question here" --iterations 10 --depth 4

# Answer with feedback to insight bandit (3x boost for evidence insights from a high-scoring answer)
.venv/bin/python -m crucible answer "question" --iterations 10 --feedback

# Apply human feedback to an insight (3x weighted bandit signal — strongest available)
.venv/bin/python -m crucible rate <insight_id> 0.9

# Check bandit posteriors
.venv/bin/python -m crucible bandit

# Start MCP server
.venv/bin/python -m crucible serve
```

**Resetting entity state for a corpus (e.g. after schema changes or for a clean re-extract):**

```cypher
MATCH (e:Entity {corpus_id: $cid}) DETACH DELETE e;
MATCH (m:Mention) WHERE NOT (m)-[:RESOLVES_TO]->(:Entity) DETACH DELETE m;
MATCH (ch:Chunk)-[r:EXTRACTION_DONE]->(:Corpus {id: $cid}) DELETE r;
```

Then re-run `crucible extract --corpus <name>`.

## File Map

```
src/crucible/
├── config.py           — All settings: neo4j, deepseek/gemini keys, MCTS params,
│                         domain, bandit_update_mode, max_synthesis_calls
├── models.py           — Corpus, Document, Chunk, Insight, Mention, Entity, Relation, ReasoningNode
├── embeddings.py       — Gemini embedding client (sync batch + Batch API)
├── llm_client.py       — Unified DeepSeek client (chat, chat_json, parse_json_lenient)
├── server.py           — MCP server: 9 tools (search, fulltext, cypher, stats,
│                         get_insights, answer, reasoning_trees, entity_search,
│                         entity_graph, rate_insight)
├── __main__.py         — CLI: ingest, explore, answer, stats, bandit, ontology,
│                         extract, synthesize-entities, re-embed, embed-insights,
│                         rate, serve
│
├── graph/
│   ├── schema.py       — Neo4j indexes + constraints (auto-applied on connect)
│   └── client.py       — Neo4j operations: CRUD, vector/fulltext search,
│                         chunk/insight sampling, reasoning tree persistence,
│                         Mention upserts, corpus-name-or-id resolver
│
├── ingestion/
│   ├── pipeline.py     — Resumable directory ingestion with checkpoint
│   ├── chunker.py      — Sentence-boundary-aware sliding window
│   └── parsers.py      — markdown (heading-split), JSON (key-split), text (paragraph)
│
├── insight/
│   ├── engine.py       — Thompson Sampling bandit + 6 strategies + feedback loop
│   │                     Strategies: bridge, outlier, hub, meta, contradiction, gap
│   │                     Bandit snapshot (alpha,beta) on every update; pseudo-count default
│   └── evaluator.py    — Advocate/Skeptic/Referee three-role debate
│
├── reasoning/
│   ├── mcts.py         — MCTS engine (UCT selection, backprop) + ActionBandit (UCB1)
│   │                     with snapshot persistence + ReasoningTree result object
│   ├── actions.py      — 5 expansion actions: search, follow_ref, challenge,
│   │                     synthesize, entity_walk
│   └── referee.py      — Pluggable referee: LLMReferee (default), SymbolicReferee
│                         (slot/NotImplementedError), HybridReferee
│
└── extraction/
    ├── __init__.py     — sanitize_type_name, make_entity_id, make_relation_id, make_mention_id
    ├── ontology.py     — OntologyGenerator: insights → typed class/relation schema
    ├── aggregate.py    — In-memory doc-level aggregation: chunk-local +
    │                     topological alias resolution; emits Entities + Mentions
    ├── extractor.py    — EntityExtractor: per-chunk extraction with retry/transient
    │                     classification, ThreadPoolExecutor parallelism
    └── synthesize.py   — EntitySynthesizer: dirty-detection + LLM merge of
                          Mentions → canonical Entity description, batched re-embed,
                          cost cap via CRUCIBLE_MAX_SYNTHESIS_CALLS
```

## Where Work Left Off

### Recently completed (2026-04-23 session)
- **Phase 0.3 — Bandit determinism fix.** `pseudo_count` update mode (default) replaces Bernoulli coin-flip; (alpha, beta) snapshot persists to disk on every update so process boots are deterministic. Empirically validated: outlier posterior swung 0.27→0.73 across Bernoulli replays of insight_rewards.jsonl, now produces a single deterministic 0.539 with pseudo-count.
- **Phase 0.1 — Unified LLMClient.** New `src/crucible/llm_client.py` modeled on burokrat's pattern; replaces 5 duplicate httpx-based call sites across engine.py, evaluator.py, actions.py, extractor.py, ontology.py. `chat_json` uses DeepSeek's `response_format={"type":"json_object"}` with lenient regex fallback.
- **Phase 1 — Mention/Entity split.** New `Mention` node sitting between Chunk and Entity (`(Chunk)-[:EXTRACTS]->(Mention)-[:RESOLVES_TO]->(Entity)`). Entity.source_chunks dropped — provenance is Mention degree. Entity.last_synthesized_at added. Eight `MENTIONED_IN` query sites migrated to the new hop.
- **Phase 2 — EntitySynthesizer.** New `extract/synthesize.py`. Detects dirty entities (any Mention newer than last_synthesized_at), LLM-merges Mentions into canonical description + alias union, re-embeds in batches, caps at `CRUCIBLE_MAX_SYNTHESIS_CALLS`.
- **CLI UX**: `--corpus` accepts either name or hashed id (was previously silent-zero on name mismatch).
- **Neo4j noise**: UNRECOGNIZED + INFORMATION-severity notifications filtered at driver init so `EXTRACTION_DONE` and "constraint already exists" don't flood logs.

Test suite: 167 unit + 30 integration = 197+ tests, all green. See git log for per-phase commits.

### Still open (continuation plan in cortex task `b174f847`)
- **Phase 0.2 — Referee benchmark.** Pivoted from Yepis hand-scoring to **crucible-on-crucible**: ingest src/ + docs as a corpus and verify answer quality against ground truth we directly know. Initial run validated the pipeline; full benchmark CLI not yet wired.
- **Phase 4 — Observability cleanup.** Replace 12+ `except Exception: pass` sites with logged warnings. Audit the heuristic_discount=0.5 hardcoded in engine.py:1069.
- **Phase 5 — Architectural cleanup.** Split engine.py (1158 lines) into per-strategy modules. `state_dir` config so reward/snapshot files don't litter CWD. Adopt burokrat's `session()` `@contextmanager` pattern to retire `graph._driver` external access.
- **Phase 6 — Strategic.** Principle/lens strategy (NEXT.md's top fix). Multi-fire synthesize. Traversal bandit for `entity_walk`. Symbolic referee adapter (the only known referee that escapes the LLM-validates-LLM circular trust).

### Yepis POC Artifacts
- Yepis graph: 104 docs, 4772 chunks, 76+ insights
- Fixed transcripts: `~/dev5/yepis/fixed-transcripts/` (57 videos)
- Essay outputs: `~/dev5/yepis/output/yepis-essay-v2.md` (voice-focused), `v3.md` (thesis-focused)
- Current events briefing: `~/dev5/yepis/current-events-briefing.md`
- Brave augmentation: `~/dev5/yepis/brave-augmented-context.md`
- Backups: `~/yepis-backup-2026-04-03.zip` + `~/yepis-neo4j-2026-04-03.tar.gz`

## Key Design Decisions

1. **Two algorithms, two problems.** Thompson Sampling for insight strategy selection (which kebab shop to try). MCTS/UCT for answer reasoning (which path through a tree to explore). They're different problems — don't conflate them.

2. **Referee quality = system quality.** The referee's score is the reward signal for everything. Bad referee → bad bandit learning → bad strategy selection → bad insights. The three-role debate (Advocate/Skeptic/Referee) is expensive (3 LLM calls) but produces much better signal than a single-call scorer.

3. **Bernoulli as a bridge.** Referee gives continuous scores (0-1). Beta distributions need binary updates. The biased coin flip bridges them. You lose per-observation nuance but the aggregate converges correctly.

4. **Pluggable referee for domains.** LLM default works everywhere. Burokrat can plug in symbolic predicate evaluation for crisp, deterministic scoring where the domain allows it. The system doesn't need to know about German law — it just needs a referee that returns a float.

5. **Actions are a registry.** Add new expansion actions without touching MCTS core. Domain adapters register their own (e.g., `check_exception` for Burokrat, `brave_search` for current events).
