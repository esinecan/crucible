# Handover — post-Phase-2 continuation

Fresh-session brief for picking up after the 2026-04-23 session, which landed Phase 0.3 (bandit determinism), Phase 0.1 (LLM client extraction), Phase 1 (Mention/Entity split — the previous handover's §2), and Phase 2 (async description synthesis — the previous handover's §3). All committed, tests green.

The next-step plan lives in cortex task `b174f847` (`task_context("b174f847")` to load it). The remaining phases are observability cleanup, architectural cleanup, and the strategic items NEXT.md identified.

## Starting state

- Branch: `main`. Last commits: Phase 0/1/2 + bug fixes + doc refresh from this session.
- Test suite: `.venv/bin/python -m pytest tests/ -q` — 167 unit + 30+ integration. Integration tests need Docker for testcontainers Neo4j; full run ~4 min.
- Working corpora: Yepis exists in its own Neo4j instance; `cocrucible` test corpus exists in this repo's Neo4j (`docker compose up -d`).
- Bandit reward log (`insight_rewards.jsonl`, ~640 entries from Yepis runs) is gitignored. Snapshot files (`*.snapshot.json`) also gitignored — they're durable state, but reproducible from the log on first cold boot.

## Recently shipped (2026-04-23)

| Phase | Summary | Files |
|---|---|---|
| 0.3 | Bandit determinism: `pseudo_count` update + (alpha,beta) snapshot | `config.py`, `insight/engine.py`, `reasoning/mcts.py`, `tests/unit/test_bandits.py` |
| 0.1 | Unified `LLMClient` modeled on burokrat's pattern; 5 callsite migration | `llm_client.py` (new), `config.py`, `insight/engine.py`, `insight/evaluator.py`, `reasoning/actions.py`, `extraction/extractor.py`, `extraction/ontology.py`, `tests/unit/test_llm_client.py` |
| 1 | `Mention` node between Chunk and Entity; 8 query sites migrated; aggregator emits Mentions; topological alias tiebreaker uses agg.entity_chunks | `models.py`, `extraction/__init__.py`, `graph/schema.py`, `graph/client.py`, `insight/engine.py`, `extraction/aggregate.py`, `extraction/extractor.py` |
| 2 | `EntitySynthesizer` with budget cap, dirty-detection, batched re-embed | `extraction/synthesize.py` (new), `__main__.py`, `graph/client.py`, `tests/unit/test_synthesize.py` |
| UX | `--corpus` accepts name or hash; UNRECOGNIZED + INFO Neo4j notifications filtered | `graph/client.py`, `__main__.py`, `tests/integration/test_graph_client.py` |

## Empirical signals worth knowing

- **Pseudo-count bandit eliminated replay drift.** Outlier posterior swung 0.27→0.73 across Bernoulli replays of insight_rewards.jsonl; pseudo-count produces a single deterministic 0.539. See `memory/crucible-bandit-replay-drift.md` for the data table.
- **Crucible-on-crucible works end-to-end.** Ingest → explore → ontology → extract → synthesize → answer pipeline ran clean on this repo's own source. The MCTS answer correctly identified the Phase 0.3 design rationale ("Bernoulli flip flattens fractional signal; pseudo-count preserves it") from chunks of code I wrote in the same session.
- **The referee's score band is 0.5–0.85.** The Skeptic's recurring "merely restates the obvious" pattern compresses everything into mid-range. NEXT.md's 0.733→0.800→0.733 progression is consistent with referee variance, not search-quality signal. This is the system's known optimization ceiling.

## What's next (cortex task `b174f847`, Phases 4–6)

### Phase 4 — Observability cleanup (~2 days)

12+ `except Exception: pass` sites in `insight/engine.py` swallow entity-channel failures silently. Replace with `logger.warning(...)` so a broken bridge/hub/contradiction/gap channel is visible (not invisibly degrading bandit signal). Specific lines: `engine.py` 274, 288, 322, 497, 655, 704, 741, 850, 895; same pattern in `actions.py`.

Also audit `engine.py:1069`: `bandit_score = insight.score if self.evaluator else insight.score * 0.5` — the 0.5 discount is a magic number that pushes heuristic-mode insights below noise. Either expose as `Config.heuristic_discount` or rethink the policy entirely.

### Phase 5 — Architectural cleanup (~1 week, ship incrementally)

- **5.1** Split `insight/engine.py` (1158 lines) into `insight/strategies/{bridge,outlier,hub,meta,contradiction,gap}.py` registered like `reasoning/actions.py:ACTION_REGISTRY`. Cross-strategy entity helpers may justify a third tier `insight/entity_channels.py`. Budget 3 days.
- **5.2** `Config.state_dir` (default `~/.crucible/<corpus>/`) for reward logs + snapshots so they don't litter CWD.
- **5.3** Adopt burokrat's `session()` `@contextmanager` pattern (`~/dev5/burokrat/src/burokrat/graph/client.py:47`) and rewrite the 7 external `graph._driver.session()` callsites.
- **5.4** Server: `graph = CrucibleGraph(config)` at module import in `server.py:13` means the MCP server can't even start to surface a Neo4j unreachable error. Move to lazy property.
- **5.5** README's "Proven Domains" table — verify the chunk/insight counts are current. Yepis numbers may be stale.

### Phase 6 — Strategic (separate sessions, NEXT.md-backed)

- **6.1 Principle/lens strategy** — NEXT.md's top fix candidate. Pre-compute transferable principles as embedded insights so MCTS finds abstracted concepts, not raw chunks. New `insight/strategies/principle.py`. Yepis evaluation: 3.0/7 ground truth ceiling that this addresses.
- **6.2 Multi-fire `SynthesizeAction`** — currently fires once and answer freezes for the rest of the MCTS tree. Allow re-synthesis at the same depth, or remove the depth constraint. NEXT.md "synthesis token ceiling."
- **6.3 Traversal bandit for `entity_walk`** — second Thompson Sampling layer with competing strategies (neighbor walk, shortest path between query entities, random walk with restart, community detection sampling). Will share Phase 0.3's snapshot pattern.
- **6.4 Symbolic referee adapter to burokrat** — burokrat's `evaluate_provision` returns deterministic PASS/FAIL/UNKNOWN/CONDITIONAL per provision. Build adapter in `reasoning/referee_burokrat.py` that frames legal queries as person+provision evaluations. Once shipped, `HybridReferee` actually works for at least one domain. Per Disconfirmation 7 in the cortex plan, the adapter is ~200 lines of glue plus corpus-specific logic — verify scope before scheduling.
- **6.5** L3+ meta insights, or document why one level is enough.

### Zero-engineering, highest leverage

Per the original-insight section in NEXT.md: the `crucible rate` command and `rate_insight` MCP tool are the only mechanisms that break the LLM-validates-LLM circular trust. They've been called twice in 640 bandit observations. Surface a CLI prompt at the end of `explore` that asks for a 0–1 rating on each insight. No new code paths needed — `apply_human_feedback` already exists. The bottleneck is UX, not algorithm.

## Reference

- **Cortex plan**: `task_context("b174f847")` for the full multi-phase plan with disconfirmations.
- **Burokrat patterns to mirror**: `~/dev5/burokrat/src/burokrat/llm/client.py` (LLM client shape, already adopted), `~/dev5/burokrat/src/burokrat/graph/client.py` (session context manager, Phase 5.3), `~/dev5/burokrat/src/burokrat/engine/evaluator.py` (symbolic referee, Phase 6.4).
- **NEXT.md structural-ceiling section**: the architectural argument for why human-rating UX is the unlock.
- **`memory/crucible-bandit-replay-drift.md`** and **`memory/burokrat-as-crucible-template.md`**: durable findings from this session.

## Non-goals

- Full cross-corpus alias disambiguation (proposal 4 stretch). The within-doc version landed in cd39aa0 and is sufficient until volume forces the question.
- MCP stdout poisoning fix.
- Embedding retry bounds.
