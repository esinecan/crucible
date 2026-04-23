# Crucible

Knowledge graph engine with two reasoning modes — **Insight Discovery** (pattern finding via Thompson Sampling over six strategies) and **Answer Search** (MCTS over query → search → synthesize → challenge actions) — plus an ontology-guided **Entity Extraction** layer that materializes a typed entity graph alongside the chunks.

Ingest a corpus. Discover patterns. Extract entities. Synthesize canonical descriptions. Answer questions with grounded evidence chains. Most things tunable via env vars; no domain-specific code required to run, though a `CRUCIBLE_DOMAIN` preamble sharpens LLM output.

## Quick Start

```bash
# Prerequisites: Docker, Python 3.12+, GEMINI_API_KEY, DEEPSEEK_API_KEY

# 1. Start Neo4j
cd ~/dev5/crucible
docker compose up -d

# 2. Install
pip install -e .

# 3. Ingest a directory of .md/.py/.json/.txt files into a corpus
python -m crucible ingest ./my-notes --name my-kb

# 4. Discover insights (Thompson Sampling picks among 6 strategies)
python -m crucible explore --cycles 5

# 5. Generate an ontology from those insights (one LLM call)
python -m crucible ontology --corpus my-kb --output my-kb-ontology.json

# 6. Extract entities + relations from chunks via the ontology
python -m crucible extract --corpus my-kb --ontology my-kb-ontology.json

# 7. Synthesize canonical entity descriptions from the per-chunk Mentions
python -m crucible synthesize-entities --corpus my-kb

# 8. Answer questions (MCTS over the chunks + insights + entities)
python -m crucible answer "What do I think about X?"

# Optional: rate an insight (3x weight in the bandit)
python -m crucible rate <insight_id> 0.9
```

`--corpus` accepts either the human name (`my-kb`) or the hashed corpus_id; the CLI resolves either.

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
              ├── Insight L1 (DERIVED_FROM chunk)
              │     └── Insight L2 (DERIVED_FROM insight)
              └── Mention (EXTRACTS) ─→ Entity (RESOLVES_TO)
                                              └── Entity (typed Entity-Entity relations)

ReasoningNode (CHILD_OF parent, USES_EVIDENCE → chunk/insight)
```

**Node types:**
- `Corpus` — a named collection of documents
- `Document` — a file (markdown, JSON, text, code)
- `Chunk` — a section of a document with a 768-dim embedding (gemini-embedding-001, MRL-truncated)
- `Insight` — a discovered pattern (L1 from chunks, L2 from L1 insights)
- `Entity` — a canonical, synthesized named thing extracted from chunk text via ontology-guided extraction
- `Mention` — a single chunk's raw extraction of an entity, immutable; preserves per-chunk surface form, description, and aliases
- `ReasoningNode` — a node in an MCTS reasoning tree

**Why Mention sits between Chunk and Entity:** the Mention layer separates per-chunk provenance ("what this chunk asserted") from canonical truth ("what we believe"). Entity descriptions are rebuilt asynchronously by `synthesize-entities` from the union of Mentions; per-chunk audit needs the Mention. One Mention per (chunk, entity) pair, idempotent under re-extraction.

**Indexes:**
- Vector (cosine, 768-dim): `chunk_embedding`, `insight_embedding`, `entity_embedding`
- Fulltext (Lucene): `chunk_text`, `insight_text`, `entity_fulltext`
- Property: document corpus/path, chunk document, insight corpus/strategy/layer, mention chunk/entity, entity corpus/type/name, reasoning tree/depth

## CLI Reference

```bash
# Ingest a directory into the knowledge graph (resumable via .crucible_progress.json checkpoint)
python -m crucible ingest <path> --name <corpus_name> [--description "..."]

# Run insight discovery cycles
python -m crucible explore [--cycles N] [--max N] [--strategy NAME] [--no-eval]
#   --cycles: number of exploration cycles (default 1)
#   --max: max insights per cycle (default 5)
#   --strategy: force a strategy (bridge/outlier/hub/meta/contradiction/gap)
#   --no-eval: skip adversarial evaluation, use heuristic scores only
#   --dry-run / --no-persist / --output-dir for inspecting cycle artifacts

# Generate an ontology from accumulated insights
python -m crucible ontology --corpus <name_or_id> [--output ontology.json]

# Extract entities and relations from chunks via the ontology
python -m crucible extract --corpus <name_or_id> [--ontology ontology.json] \
                           [--batch N] [--limit N] [--workers N] [--no-embed]

# Synthesize canonical Entity descriptions from per-chunk Mentions
python -m crucible synthesize-entities --corpus <name_or_id> \
                                       [--limit N] [--max-calls N]
#   Capped by CRUCIBLE_MAX_SYNTHESIS_CALLS (default 500)

# Re-embed a corpus with the current backend (sync, or --batch-api for 50% cheaper async)
python -m crucible re-embed --corpus <name_or_id> \
                            [--include-insights] [--include-entities] \
                            [--force] [--batch-api]

# Answer a question via MCTS reasoning
python -m crucible answer "your question" [--iterations N] [--depth N] [--feedback]
#   --iterations: max MCTS iterations (default 20)
#   --depth: max tree depth (default 5)
#   --feedback: feed answer scores back to insight bandit (3x boost for evidence insights)

# Apply human feedback to an insight (3x weight in the bandit — strongest signal)
python -m crucible rate <insight_id> <score>   # score in [0.0, 1.0]

# Show graph statistics
python -m crucible stats

# Show Thompson Sampling bandit posteriors
python -m crucible bandit

# Backfill embeddings on insights
python -m crucible embed-insights

# Start MCP server (stdio transport)
python -m crucible serve
```

`--corpus` accepts either the human name or the hashed corpus_id throughout.

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
| `CRUCIBLE_BANDIT_UPDATE_MODE` | `pseudo_count` | `pseudo_count` (default, deterministic, fractional) or `bernoulli` (legacy, coin-flip) |
| `CRUCIBLE_MAX_SYNTHESIS_CALLS` | `500` | Cost cap for `synthesize-entities` LLM calls per run |
| `CRUCIBLE_STATE_DIR` | `~/.crucible` | Per-corpus state root; reward log + bandit snapshot land in `<state_dir>/<corpus_id>/` when an engine is constructed with a corpus_id |
| `CRUCIBLE_DISABLED_STRATEGIES` | (empty) | Comma-separated insight strategies to exclude from Thompson Sampling. Example: `gap,outlier`. An explicit `--strategy gap` CLI flag still overrides. |
| `CRUCIBLE_GROUNDING_CHECK` | `false` | When truthy (`1`/`true`/`yes`/`on`), runs an extra LLM call per L1 insight to verify the claim against its source chunk. Contradicted insights are dropped before persist; unsupported insights persist with a marker and bandit reward is halved. Doubles per-insight LLM cost. |
| `GEMINI_API_KEY` | (required) | Gemini API key for embeddings |
| `DEEPSEEK_API_KEY` | (required) | DeepSeek API key for evaluation, synthesis, extraction |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API endpoint |

## Recipes

Patterns earned the hard way against Crucible's own source as a self-referential test corpus (see Validation Corpus section below). Each recipe lists the failure mode it addresses, the env-var/flag combination, and a copy-pasteable cocrucible example.

### 1. Per-corpus bandit isolation

**Without it:** a Thompson-Sampling bandit warmed by one corpus steers strategy selection on the next. 640 reward entries from a Yepis run had `hub` at α=256 entering a fresh cocrucible run; cocrucible's first 13 explore decisions picked `hub` 8 times before any cocrucible signal could counter the prior.

**With it:** `--corpus <name_or_id>` on `explore`, `answer`, and `bandit` routes the reward log + `(α, β)` snapshot to `<CRUCIBLE_STATE_DIR>/<corpus_id>/`, so each corpus learns its own arm distribution from clean priors.

```bash
# Inspect cocrucible's bandit (cold start = uniform priors)
python -m crucible bandit --corpus cocrucible
#   bridge   mean=0.500  α=1 β=1  obs=0
#   ...

# Run explore against cocrucible — updates land in
# ~/.crucible/<cocrucible_id>/insight_rewards.jsonl, not the global CWD log
python -m crucible explore --corpus cocrucible --cycles 5

# Same code, different corpus, fully independent posteriors
python -m crucible explore --corpus my-prose --cycles 5
python -m crucible bandit --corpus my-prose
```

`--corpus` accepts either the human name or the hashed corpus_id; `resolve_corpus_id` in graph/client.py handles the lookup.

### 2. Disable strategies that don't fit your corpus type

**The gap problem:** the gap strategy's prose-corpus rule (`incoming_relations > mentions = knowledge gap`) misfires on code corpora because every called function is referenced more times than it's mentioned. On cocrucible the gap channel produced 4/4 hallucinations including "MCTS skips backpropagation" (refuted by the first 30 lines of `reasoning/mcts.py`).

**Three escape hatches, smallest to largest:**

```bash
# Hatch 1: code-aware heuristic (automatic, no config). Stricter rule for
# entities of types like MODULE, INSIGHT_ENGINE, EXTRACTION_PIPELINE,
# REASONING_STRATEGY, or any UPPER_SNAKE_CASE class containing a code-y
# token (ENGINE, PIPELINE, ALGORITHM, METHOD, FUNCTION, CLASS, COMPONENT).
# Code-like entities must clear `incoming - mentions >= 5 AND mentions <= 2`
# to surface as a gap. Most well-encapsulated functions get filtered out.

# Hatch 2: turn off the gap strategy entirely for this corpus
CRUCIBLE_DISABLED_STRATEGIES=gap python -m crucible explore --corpus cocrucible --cycles 5

# Hatch 3: turn off multiple strategies known to misfit your corpus
CRUCIBLE_DISABLED_STRATEGIES=gap,outlier python -m crucible explore --corpus my-thin-corpus --cycles 5
```

`update()` still applies to disabled arms, so an explicit `--strategy gap` CLI flag still runs gap and updates its posterior — disabling is about the bandit's autonomous selection, not freezing the arm's state.

### 3. Catch hallucinations with the grounding check

**The problem:** the LLM-driven evaluator (Advocate / Skeptic / Referee) compresses scores into the 0.5–0.85 band; it doesn't reliably catch claims that are directly refutable by the source chunk the insight was built from. The cocrucible meta experiment surfaced 4 L2 insights, 3 of which were wrong, all scored within the same narrow band.

**With `CRUCIBLE_GROUNDING_CHECK=true`:** every L1 insight gets one extra LLM call after the evaluator runs. The model is shown the insight's claim and one of its source chunks and asked whether the source supports, contradicts, or is silent on the claim, with a 0–1 confidence. Contradicted insights at confidence ≥ 0.7 are **dropped before persist** (logged with `mode='dropped_grounding'` for audit). Unsupported insights persist with `context.grounding='unsupported'` and the bandit reward is halved.

```bash
# Re-run gap on cocrucible with grounding on. Hallucinated gap insights
# whose claims are directly refuted by source chunks (e.g., a claim that
# get_dirty_entities is "absent from the codebase" tested against the
# chunk that defines it) get filtered before they pollute the graph.
CRUCIBLE_GROUNDING_CHECK=true python -m crucible explore --corpus cocrucible --strategy gap --cycles 1

# Check the audit trail
cat ~/.crucible/<cocrucible_id>/insight_rewards.jsonl \
  | grep '"mode": "dropped_grounding"'
```

L2 (meta) insights bypass grounding because their claims are abstractions over multiple L1s, not single chunks; their grounding is transitively covered by whatever passed at the L1 layer.

**Cost:** doubles per-insight LLM cost. Off by default. Worth turning on when you're seeding a new corpus and don't yet trust the strategy mix; turn off once the bandit has converged and the referee is producing reliable scores for your domain.

### 4. Re-run meta and watch dedup work

**The problem:** re-running `explore --strategy meta` on the same L1 inventory used to produce slightly-reworded duplicates of existing L2 bridges. The cocrucible run produced 4 L2s where 2 were the same gap×hub pair with different LLM-synthesized text.

**With T2's dedup:** the cycle queries the graph for `(:Insight{layer:2,strategy:'meta'})-[:DERIVED_FROM]->(src)` to find any (a_id, b_id) pair already bridged, and skips those candidates. Anti-cluster reorders the queue when the last 2 accepted insights share the same (a_strategy, b_strategy) pair, deferring further same-pair candidates to the back.

```bash
# First meta cycle
python -m crucible explore --corpus cocrucible --strategy meta --cycles 1 --max 4

# Re-run on the same L1 pool — expect zero or few new bridges
python -m crucible explore --corpus cocrucible --strategy meta --cycles 1 --max 4
# Logger reports (at INFO):
#   meta: skipped N candidate pair(s) already bridged in graph
```

If your L1 pool is small (cocrucible has 13), the second cycle may produce 0 new L2s — that's correct, not a bug.

### 5. Inspect bandit drift / determinism

**The problem:** the legacy `bernoulli` update mode flips a biased coin per observation; the per-process `random` state means replaying the same reward log gives different posteriors each boot. Outlier strategy on the cocrucible reward log swung mean 0.27 → 0.73 across three replay seeds.

**With `pseudo_count` (default since this commit):** `alpha += score, beta += (1 - score)`. Deterministic across replays. Same input always gives the same posterior.

```bash
# Confirm a fresh boot of the cocrucible bandit gives the same numbers
python -m crucible bandit --corpus cocrucible
python -m crucible bandit --corpus cocrucible
# Both prints should match exactly (snapshot loaded, no replay variance).

# Reproduce historical Bernoulli drift if you need to compare against
# an old run from before the determinism fix
CRUCIBLE_BANDIT_UPDATE_MODE=bernoulli python -m crucible bandit --corpus cocrucible
```

### 6. Stack the fixes for a clean run

The patterns compose. Here's the conservative-mode preset for a brand-new code corpus where you want grounded, dedup-clean insight discovery from the first cycle:

```bash
export CRUCIBLE_DISABLED_STRATEGIES=gap   # skip the prose-rule strategy
export CRUCIBLE_GROUNDING_CHECK=true      # drop refuted insights before persist
export CRUCIBLE_DOMAIN="Description of your codebase, language, and conventions."

python -m crucible ingest ./my-codebase --name my-code
python -m crucible explore --corpus my-code --cycles 5 --max 4
python -m crucible ontology --corpus my-code --output my-code-ontology.json
python -m crucible extract --corpus my-code --ontology my-code-ontology.json --workers 10
python -m crucible synthesize-entities --corpus my-code
python -m crucible answer "your question" --corpus my-code --iterations 12 --depth 4
```

For a prose corpus, drop the first export and leave grounding off until you've seen the score distribution.

## Key Concepts

### Thompson Sampling (Bandit)

Each insight strategy is an "arm" with a Beta(α, β) distribution representing belief about its quality. To select which strategy to run:
1. Sample a random value from each arm's Beta distribution
2. Pick the arm with the highest sample

High-quality strategies get tight distributions around high values (exploited often). Under-explored strategies have wide distributions that occasionally sample high (explored naturally). No explicit explore/exploit switch needed.

**Update modes** (selected via `CRUCIBLE_BANDIT_UPDATE_MODE`):
- `pseudo_count` (default) — `alpha += score, beta += (1 - score)`. Deterministic across replays of the same reward log; preserves the magnitude of the referee signal.
- `bernoulli` (legacy) — `random.random() < score` flips a biased coin and bumps one side by 1. Same posterior in expectation, but the per-process `random` state means two boots of the same log produce different posteriors. Kept for reproducing historical runs.

**State persistence** — `(alpha, beta)` is snapshotted to `<reward_log>.snapshot.json` after every update. On boot the snapshot wins; the JSONL reward log is the audit trail, not the state, and is only replayed when the snapshot is missing or corrupted. The same pattern applies to the `ActionBandit` (UCB1 over MCTS expansion actions).

### MCTS with UCT

Standard Monte Carlo Tree Search adapted for knowledge graph reasoning. UCT formula for node selection:

```
UCT(node) = avg_score / visits + c * sqrt(ln(parent_visits) / visits)
```

Where `c` (default 1.41 ≈ √2) controls the exploration/exploitation balance. Unvisited nodes get infinite UCT score (always explored first).

### Referee (Pluggable)

The referee determines what "good" means. Three slots:
- **LLMReferee** (default, implemented): Advocate/Skeptic/Referee three-role debate via DeepSeek.
- **SymbolicReferee** (slot, currently `NotImplementedError`): deterministic domain adapter. The reference implementation lives outside this repo (e.g. burokrat's `evaluate_provision` for German legal predicates). A near-term plan item is a thin adapter to plug that in via `HybridReferee`.
- **HybridReferee** (implemented): tries symbolic first, falls back to LLM. Useful once a SymbolicReferee subclass is provided.

Referee quality directly determines bandit learning quality. A better referee → better reward signal → faster convergence on the best strategies. **The current LLMReferee compresses scores into the 0.5–0.85 band**, which is one of the system's known optimization ceilings; human ratings via `crucible rate` are weighted 3× and are the strongest signal available.

## MCP Server

Exposes 9 tools via stdio transport:

| Tool | Purpose |
|------|---------|
| `search(query, limit, corpus)` | Semantic vector search over chunks |
| `fulltext(query, limit)` | Lucene fulltext search over chunk text |
| `cypher(query)` | Read-only Cypher queries (driver-level write rejection) |
| `stats()` | Graph statistics (corpora, docs, chunks, insights, entities, relations) |
| `get_insights(corpus, strategy, limit)` | Browse discovered insights |
| `entity_search(query, limit)` | Fulltext search over entity names + descriptions |
| `entity_graph(entity_name, hops)` | Get an entity and its typed neighbors |
| `answer(query, max_iterations, max_depth)` | MCTS reasoning, returns answer + evidence chain + path summary |
| `reasoning_trees(tree_id)` | List or fetch a persisted MCTS tree |
| `rate_insight(insight_id, score)` | Apply human feedback (3× weight in the bandit) |

## File Structure

```
src/crucible/
├── __init__.py
├── __main__.py              # CLI entry point — 11 subcommands
├── config.py                # Configuration (env vars, MCTS params, bandit mode, synthesis cap)
├── models.py                # Data models (Corpus, Document, Chunk, Insight, Mention, Entity, Relation, ReasoningNode)
├── embeddings.py            # Gemini embedding client (sync + Batch API)
├── llm_client.py            # Unified DeepSeek client (chat, chat_json with response_format=json_object + lenient fallback)
├── server.py                # MCP server (9 tools via stdio)
├── graph/
│   ├── client.py            # Neo4j CRUD, search, tree persistence, corpus-name-or-id resolver
│   └── schema.py            # Indexes, constraints (Chunk, Insight, Entity, Mention, ReasoningNode)
├── ingestion/
│   ├── pipeline.py          # Resumable directory ingestion
│   ├── chunker.py           # Sentence-boundary-aware chunking
│   └── parsers.py           # File format parsers (md, json, txt, code)
├── insight/
│   ├── engine.py            # Thompson Sampling + 6 strategies + feedback loop + bandit snapshot
│   └── evaluator.py         # Adversarial Advocate/Skeptic/Referee debate
├── reasoning/
│   ├── mcts.py              # MCTS engine + ReasoningTree + ActionBandit (UCB1) with snapshot
│   ├── actions.py           # 5 expansion actions (search, follow_ref, challenge, synthesize, entity_walk)
│   └── referee.py           # Pluggable referee interface (LLM, Symbolic [stub], Hybrid)
└── extraction/
    ├── __init__.py          # Sanitizers + deterministic id helpers (entity, relation, mention)
    ├── ontology.py          # OntologyGenerator: insights → typed class/relation schema
    ├── aggregate.py         # In-memory doc aggregator: chunk-local + topological alias resolution
    ├── extractor.py         # EntityExtractor: ontology-guided per-chunk extraction with retry
    └── synthesize.py        # EntitySynthesizer: LLM-merges Mentions into canonical Entity
```

## Validation Corpus

Crucible is regularly run against its own source as a self-referential test corpus (~5K LOC + 5 docs → 27 docs, 798 chunks, 260+ entities, 504+ Mentions, 250+ relations, 13+ L1 insights, plus L2 meta bridges). The ontology generator surfaces the Mention/Entity schema directly from reading the source; the MCTS answer flow correctly identifies the bandit's pseudo-count vs Bernoulli distinction; the meta strategy's failure modes on this corpus motivated the four-tier discovery-layer fix (see Recipes 2–4). See `crucible-on-crucible` notes in NEXT.md.

Build it yourself by pointing `crucible ingest` at this repo's `src/crucible/` plus the markdown docs at the root, then walking the recipe in section 6 above. The corpus name `cocrucible` is conventional. A pre-baked Neo4j volume snapshot may be bundled with the repo so a fresh clone can boot directly into the recipes — when present, mount it with `docker compose -f docker-compose.yml -f docker-compose.bundle.yml up -d` (overlay TBD).

Other corpora the engine has been used on (numbers approximate, last-known counts):
- **Yepis** (geopolitics) — ~4.7K chunks, 76+ L1 insights, motivated NEXT.md's three optimization ceilings
- **Burokrat** (German law) — ~89K legal norms; symbolic referee adapter is in scope but not yet wired
- **Agent-KB** (Forto platform) — ~17.5K chunks of team knowledge
