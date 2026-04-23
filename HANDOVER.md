# Handover — discovery-layer fixes complete, strategy-level work next

Fresh-session brief for picking up Crucible after the 2026-04-23 session
shipped the four-tier discovery-layer fix (cortex task 44450c9e) and the
preceding handover refactor (cortex task b174f847). Both tasks are closed;
their commits are on `main` at `origin/main`. The open successor task is
`56afbe15` — gap-channel quality + strategy-specific grounding.

## What shipped

### b174f847 — handover refactor + audit findings (closed)
Commits: `cd39aa0`, `4c92d5c`, `8333bde`.

- Mention/Entity split. Per-chunk provenance on Mention nodes,
  canonical state on Entity, eight MENTIONED_IN query sites migrated
  to `(Chunk)-[:EXTRACTS]->(Mention)-[:RESOLVES_TO]->(Entity)`.
- Unified `LLMClient` (modeled on burokrat's client.py). Five duplicate
  httpx callsites collapsed to one, three duplicate `_parse_json_lenient`
  copies collapsed to one, all `os.getenv("DEEPSEEK_*")` reads live in
  `Config` exclusively.
- Bandit determinism. Pseudo-count is default (replay-deterministic);
  Bernoulli available behind a flag. `(alpha, beta)` snapshots next to
  reward log; JSONL becomes audit trail, not state.
- Async entity-description synthesizer. `crucible synthesize-entities`
  with cost cap and idempotent re-runs via `Entity.last_synthesized_at`.

### 44450c9e — discovery-layer fixes (closed, just shipped)
Commits: `d3c8bc2`, `512776c`, `4d21cc9`, `dad3622`, `5df642a`, `405e7d7`.

- **T1 per-corpus bandit state.** `Config.state_dir`; `--corpus` on
  explore/answer/bandit routes `(alpha, beta)` snapshots to
  `<state_dir>/<corpus_id>/`. A Yepis-warmed bandit no longer steers
  cocrucible discovery.
- **T2 meta dedup + anti-cluster.** `existing_meta_bridge_pairs()`
  skips already-bridged (a_id, b_id) pairs; anti-cluster defers further
  same-strategy-pair candidates after two consecutive picks.
- **T3 code-aware gap heuristic + disabled_strategies escape hatch.**
  Stricter rule for code-typed entities (`incoming - mentions >= 5 AND
  mentions <= 2`); `CRUCIBLE_DISABLED_STRATEGIES` excludes named arms.
- **T4 pre-persist grounding check.** Opt-in via
  `CRUCIBLE_GROUNDING_CHECK`. Drops L1s on high-confidence contradict,
  marks and discounts unsupported, bypasses L2.
- **T5 complete cocrucible extraction.** 697 entities, 715 relations,
  123 synthesized.
- **T6 revalidation.** Covered below.
- **README Recipes section** (`5df642a`) with cocrucible examples for
  every env-var pattern.
- **Lucene escape + gap provenance** (`405e7d7`). Two fixes surfaced
  during T6: `fulltext_search` didn't escape Lucene specials (hyphens
  in queries crashed); gap L1s had empty `source_chunk_ids` so
  grounding fell through to fail-closed default on every gap insight.

Total: `8333bde..405e7d7` on origin/main, **303/303 tests green**.

## Live validation summary (cocrucible, 2026-04-23)

### T6a — cross-module MCTS answer
Question: "Trace what happens when a user runs
`crucible synthesize-entities --corpus X`, from CLI to the final
updated Entity row in Neo4j…"

Best score 0.527 at depth 2 (12 iterations). Answer named
`__main__.py`, `extraction/synthesize.py`, `llm_client.py` (with a
minor path slip — said `extraction/llm_client.py`, actual is
`src/crucible/llm_client.py`), `graph/client.py:get_dirty_entities`,
and produced a structurally-correct dirty-detection Cypher (used
`last_updated_at` vs the actual `last_mention > e.last_synthesized_at`
— shape right, property names wrong). Truncated at Phase 3 because
synthesize fired once and never re-fired. **Pass.**

### T6b — meta strategy with T1 + T2 active
Fresh L2 baseline, 3 cycles, 4 max each → 12 L2s. Strategy pair split:
7 gap×hub, 5 contradiction×hub — anti-cluster reached the 1 contradiction
L1 that would otherwise be buried under hub×gap's top-similarity
cluster.

Top-scored L2s at 0.700 graded honestly:
- "EntityExtractor as bidirectional translator" — verifiable, novel
  framing.
- "LLMClient parameter drift undermines dual-channel design" —
  verifiable bridge between two correct L1s, novel synthesis.
- Three further parameter-drift variants (anti-cluster defers after
  2 consecutive; 3 same-pair picks still squeezed through).
- "MCTS skips tree expansion" reappeared once — grounding was off for
  this run's L1 generation because L1s predate T4. **Pass** on the
  ≥1 verifiable novel L2 criterion.

### T6c — grounding validation on gap
Two iterations. First revealed a wiring bug: gap L1s had no
`source_chunk_ids`, so grounding fell through to fail-closed
`('unsupported', 0.0)` for every insight. Committed `405e7d7` to
populate provenance from both gap channels (entity-graph fetches up
to 5 mention chunks via the Mention hop; LLM carries the first 5
sampled chunks).

Post-fix: 8 gap L1s persisted, 7 at grounding confidence 0.9, 1 at
0.0 (LLM error fallback). Verdicts all `unsupported`. Drop path
never fired.

**This is correct behavior**, not a T4 failure. Gap makes structural
claims ("Entity X referenced by N, mentioned in M chunks"). Grounding
samples one of those M chunks; that chunk supports X existing but is
silent on the meta-level sparsity claim. LLM correctly refuses to
issue `contradict` above the chunk's epistemic level. T4 degrades to
mark-and-halve: `context.grounding="unsupported"` persisted, bandit
reward halved. Over enough cycles, gap's posterior collapses on
cocrucible (corpus-isolated thanks to T1; Yepis unaffected).

## Current state

| Component | State |
|---|---|
| Source LOC | ~5500 Python across `src/crucible/` |
| Test count | 303 passing (254 unit + 49 integration) |
| Test runtime | ~80s full sweep |
| Bundled corpora | None yet; cocrucible Neo4j snapshot placeholder in README's Validation Corpus section |
| Cocrucible graph | 27 docs / 798 chunks / 697 entities / 504+ mentions / 715 relations / 13 L1 + 0 L2 |
| Commits ahead of origin | 0 after this handover lands |

## Known remaining issues (T6c gap-L1 quality review)

Honest grading of the 8 fresh gap L1s: 1 decent, 1 borderline, 6 noise.
Three failure shapes, all structural:

1. **`CONCEPT`-typed code entities slip T3.** The ontology extractor
   classified `sample_chunks` and `Referee` as `CONCEPT`, not `MODULE`.
   T3's `_is_code_like_type` keys on type name only. Two function /
   class names surfaced as "blind spots" when they're just
   defined-once-called-many patterns.

2. **Per-entity duplicates within a cycle.** `sample_chunks` surfaced
   twice, `Referee` surfaced twice. Meta dedup is at (a_id, b_id)
   pair level; within-strategy per-entity dedup doesn't exist.

3. **LLM-channel claims refutable by reading the file.** Two L1s
   claimed concrete architectural details were missing when
   `evaluator.py` has the full Advocate/Skeptic/Referee prompts inline
   and the README has the Thompson Sampling math. Grounding marks
   them `unsupported` at 0.9 but persists them anyway because drop
   policy only fires on `contradict`.

All three items are tracked on the open follow-up task `56afbe15`.

## Proposals for improvement — ranked by leverage

### Tier A — high impact, low effort (ship first, as the next PR)

**A1. Python-identifier shape check in `_is_code_like_type`.** 5-line
regex for snake_case or CamelCase identifiers that Python parses as
names. Catches `CONCEPT`-typed code entities without requiring the
ontology generator to classify them correctly. No false-positive risk
on prose corpora where entity names are proper nouns or multi-word
phrases.

**A2. Per-entity dedup in `_gap_cycle`.** Track `seen_entity_ids: set`
in cycle scope, skip repeats. ~15 lines. Fixes the 2× duplication seen
in T6c.

**A3. Bundled cocrucible Neo4j snapshot.** User raised this as a
distribution goal. Commit `cocrucible-snapshot.tar.gz` + a
`docker-compose.bundle.yml` overlay that extracts the snapshot into
the volume on `up`. Fresh clones can try the README recipes without
running a full ingest. ~3 hours.

### Tier B — high impact, medium effort (next task cycle)

**B1. Strategy-specific grounding.** Biggest architectural unlock.
Today's one-size-fits-all chunk grounding fits contradiction + LLM-gap
but not gap-structural, meta, or hub. Refactor:

```
Referee                              (existing — the quality scorer)
  ↓
GroundingStrategy                    (new — per-insight-strategy)
  ├── ChunkGrounding                 (current behavior)
  ├── StructuralGrounding            (gap — verify against graph stats)
  ├── SourceInsightGrounding         (meta — verify against source L1s)
  └── ConnectivityGrounding          (hub — verify against cross-doc stats)
```

Each strategy picks its grounding at construction. ChunkGrounding
stays opt-in; the others default-on because they're Cypher-only, no
LLM. Expected lift: drop path fires for many more cases; gap's
structural claims become verifiable against the numbers the claim
asserts. ~2 days.

**B2. Multi-fire synthesize action in MCTS.** NEXT.md named this the
"synthesis token ceiling." T6a truncated because synthesize fired once
at depth 2 and the next 9 iterations all challenged a frozen answer.
Let the action bandit re-select synthesize at the same depth, or drop
the depth constraint for synthesize. ~1 day.

**B3. Referee benchmark (still outstanding from b174f847 plan).**
Hand-score 30 (insight, score) pairs from `output/explore-cycle-*/`
artifacts plus the Yepis 7-principle ground truth. New CLI
`crucible benchmark-referee --suite insight|answer` reports
correlation, MAE, per-strategy breakdown. Load-bearing for tuning the
referee prompts empirically instead of by feel. ~1-2 days, half
hand-scoring.

### Tier C — high impact, higher effort (separate task)

**C1. Domain plug-in architecture.** README claims "universal
knowledge base builder, no domain-specific config required." Reality:
cocrucible needed a domain preamble, noise patterns, a code-aware
heuristic, and a strategy-disable flag. A plug-in system would let
each domain register: a `Config` extension, a `GroundingStrategy`
bundle (B1), a noise-pattern set, a referee (LLM or symbolic), and a
list of disabled strategies. ~1 week. Commits to the "universal"
claim instead of hedging around it.

**C2. Structural grounding for gap specifically (subset of B1).**
If B1 is too large: add `_ground_gap` that verifies the claimed
`incoming=N, mentions=M` against current graph stats at persist time.
Pass if stats match; drop if wildly off. High hit-rate for gap's
class of hallucination. ~½ day.

### Tier D — polish

**D1. Engine module split.** `insight/engine.py` is 1200 lines. Move
strategies into `insight/strategies/{bridge,outlier,hub,meta,contradiction,gap}.py`.
Shared entity-channel helpers get their own home. ~2-3 days.

**D2. `EXTRACTION_DONE` warning spam.** Neo4j warns on every
`NOT exists(...EXTRACTION_DONE...)` before any such edge exists.
Prime the relationship type at schema setup. ~½ hour.

**D3. MCP server hardening.** `server.py:13` instantiates
`CrucibleGraph` at import; Neo4j-unreachable fails noisily. Lazy
property. ~15 minutes.

**D4. `graph._driver` encapsulation.** Seven callsites reach into the
private driver attribute. Mirror burokrat's `BurokratGraph.session()`
public `@contextmanager`. ~1 hour.

## Architectural tensions still open

**The referee compression problem.** LLM referee's score band runs
roughly 0.4-0.85 with most insights clustering 0.5-0.7. T6b's six
top-scored L2s all landed at exactly 0.700 despite wildly different
quality. The bandit CAN'T reliably distinguish good from bad at this
resolution. This session's work (pseudo-count, per-corpus state,
anti-cluster, dedup, grounding) is plumbing around this central fact.
B3 (referee benchmark) is the diagnostic; a referee prompt rewrite
(Skeptic calibration, explicit novelty grading) or a pluggable
symbolic-first `HybridReferee` per domain are the strategic moves.

**The "grounded" vs "provocative" tension.** T4's grounding favors
insights whose claims are directly chunk-verifiable. But the most
valuable insights are often about the corpus's shape, not its content
— those live above the chunk epistemic level by construction.
Strategy-specific grounding (B1) is a partial solution; the deeper
fix is being honest about which strategies can be machine-verified
and which need human feedback. `apply_human_feedback` is that slot
and was called twice in the original 640-entry reward log. That
number needs to be in the hundreds before anyone declares the referee
trained.

**Bundle-and-distribute.** Making the cocrucible corpus bundled (A3)
turns Crucible from "run it on your own data" to "here's a working
example you can explore, then swap your data in." A1-A3 plus B2
would make the bundled corpus meaningfully richer for new users —
recipes in the README hit a pre-warmed graph with real insights,
grounded entities, a responsive MCTS answer loop.

## Suggested next move for the fresh session

Open cortex task `56afbe15`. Plan-mode → draft (A1, A2, C2 as one
cohesive PR) → implement → validate → commit → push. Ships the
highest-leverage low-effort fixes without opening Tier-B architectural
work before the patch lands. B1 (strategy-specific grounding) is the
right next task after.

Memory anchors from this session:
- `memory/crucible-bandit-replay-drift.md` — empirical case for
  pseudo-count default.
- `memory/burokrat-as-crucible-template.md` — patterns to adopt as
  reference, not as integration target.
- `memory/crucible-structural-ceiling.md` — the referee-compression
  observation that set the thesis for this session's work.

Open follow-up task: `56afbe15`.
