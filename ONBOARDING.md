# Crucible Onboarding — What You Need to Know

This doc is for Eren picking up Crucible in a new session. It covers what the system does, how the pieces fit together, and where the work left off.

## What Crucible Is

A knowledge graph engine with two modes:
- **Insight mode**: discovers patterns in your corpus using a Thompson Sampling bandit
- **Answer mode**: reasons toward answers using MCTS tree search

Both share a Neo4j graph and reinforce each other through a feedback loop.

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
- Ollama with `nomic-embed-text` model pulled
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

# Run insight discovery (5 cycles, adversarial eval)
.venv/bin/python -m crucible explore --cycles 5 --eval

# Answer a question (10 iterations, depth 4)
.venv/bin/python -m crucible answer "your question here" --iterations 10 --depth 4

# Answer with feedback to insight bandit
.venv/bin/python -m crucible answer "question" --iterations 10 --feedback

# Check bandit posteriors
.venv/bin/python -m crucible bandit

# Start MCP server
.venv/bin/python -m crucible serve
```

## File Map

```
src/crucible/
├── config.py           — All settings: neo4j, ollama, deepseek, MCTS params, domain
├── models.py           — Corpus, Document, Chunk, Insight, ReasoningNode
├── embeddings.py       — Ollama embedding client (batch + single)
├── server.py           — MCP server: 7 tools (search, fulltext, cypher, stats,
│                         get_insights, answer, reasoning_trees)
├── __main__.py         — CLI: ingest, explore, answer, stats, bandit, serve
│
├── graph/
│   ├── schema.py       — Neo4j indexes + constraints (auto-applied on connect)
│   └── client.py       — All Neo4j operations: CRUD, vector/fulltext search,
│                         chunk/insight sampling, reasoning tree persistence
│
├── ingestion/
│   ├── pipeline.py     — Resumable directory ingestion with checkpoint
│   ├── chunker.py      — Sentence-boundary-aware sliding window
│   └── parsers.py      — markdown (heading-split), JSON (key-split), text (paragraph)
│
├── insight/
│   ├── engine.py       — Thompson Sampling bandit + 6 strategies + feedback loop
│   │                     Strategies: bridge, outlier, hub, meta, contradiction, gap
│   └── evaluator.py    — Advocate/Skeptic/Referee three-role debate
│
└── reasoning/
    ├── mcts.py         — MCTS engine (UCT selection, backprop) + ActionBandit (UCB1)
    │                     + ReasoningTree result object
    ├── actions.py      — 4 expansion actions: search, follow_ref, challenge, synthesize
    └── referee.py      — Pluggable referee: LLMReferee, SymbolicReferee, HybridReferee
```

## Where Work Left Off

### Completed (Phase A)
- MCTS core: UCT selection, expansion, simulation, backpropagation
- Query-focused search (embeds the query, not partial answer)
- Hybrid retrieval (vector + fulltext + insight search, merged by score)
- Insight-aware evidence (insights as first-class evidence in MCTS)
- UCB1 action bandit (learns which actions improve answer quality)
- MCP server: `answer` and `reasoning_trees` tools
- Feedback loop: answer scores reward contributing insights in the bandit
- README and docstrings

### Remaining (Phases B-D in Cortex task 678d080d)
- **Phase B**: crucible.yaml config file (replace env vars)
- **Phase C**: `crucible init` zero-friction setup + CLI polish
- **Phase D**: auto-domain detection, auto-referee, auto-tune MCTS, graceful degradation

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
