# Crucible: What's Next

Findings from the Yepis LTV-vs-marginalism test case (April 8, 2026).

## What shipped (actions.py only, ~100 lines)

**Fix 1: Two-phase synthesize prompt.** SynthesizeAction now instructs the LLM to EXTRACT generalizable principles from evidence, APPLY each to the query with explicit transfer logic, then SYNTHESIZE. Moved ground truth from 0/7 to 2.5/7. max_tokens 800 -> 1000.

**Fix 2: Query expansion.** SearchAction now generates 4 conceptual facet queries via LLM on first search, vector-searches each, merges results. Reaches semantic neighborhoods the main query embedding misses. Moved ground truth from 2.5/7 to 3.0/7. Top results increased from 10 to 15.

## Three ceilings identified

### 1. Facet quality ceiling
The facet query LLM generates topically adjacent queries, not conceptually orthogonal ones. It found "gig economy ideology in Turkish commentary" but missed "crisis exploitation / alternative destruction" (shock doctrine) because that requires a conceptual leap: "what if platform workers 'choose' platforms because alternatives were destroyed?"

**Fix candidate: Lens/principle pre-computation.** A new insight strategy that extracts transferable principles from the corpus and persists them as embedded insights. MCTS search then finds pre-abstracted concepts, not just raw chunks. The principle "alternatives are manufactured away" would exist before any query needs it. Design note from Eren's Claude conversation: these should be structured concept types feeding back into the ontology, not free-form text nodes.

### 2. Synthesis token ceiling
1000 tokens for EXTRACT+APPLY+SYNTHESIZE across 15 evidence items means ~150 words per principle. More evidence = shallower treatment. Fix 2 actually diluted Fix 1's attention economy focus.

**Fix candidate: Iterative re-synthesis.** Allow SynthesizeAction to fire more than once in the MCTS tree. Currently it fires at depth 2 and the answer is frozen for 21 entity_walk iterations. A re-synthesis after new evidence would deepen treatment. Requires either lifting the max_depth constraint or allowing the action bandit to re-select synthesize at the same depth.

### 3. Single-seed traversal
EntityWalkAction ran 21/25 iterations using one strategy: "find entities in query, walk neighbors." Same neighborhoods explored repeatedly without improving the answer.

**Fix candidate: Traversal bandit.** A second Thompson Sampling bandit layer with competing traversal strategies:
- Neighbor walk (current)
- Shortest path between query entities
- Random walk with restart (PageRank-style)
- Community detection sampling (Louvain clusters)

Bernoulli updates, same as insight bandit. Weight overlay for feedback-driven traversal stays in a separate layer so base graph stays immutable. From Eren's Claude conversation: Thompson Sampling is right for homogeneous corpora (SEC filings), consider EXP3 for heterogeneous ones (Yepis: essays + transcripts + social posts).

## Ground truth principles (evaluation benchmark)

7 transferable principles identified in Yepis corpus for the LTV/marginalism debate:

| # | Principle | Source | Status after Fix 1+2 |
|---|-----------|--------|---------------------|
| 1 | Manufactured consent: voluntary choice is manufactured when choice architecture is controlled | Manufacturing consent essay, Strigoi series | Extracted as "Ideological Obfuscation" |
| 2 | Tekelci kapitalizm: platform IS the market, not a free marketplace | sePuN40Zgfs transcript | Partially applied (monopoly mentioned) |
| 3 | Shock doctrine: alternatives destroyed to make current option look voluntary | Shock doctrine essay, 99 earthquake chunks | Missed (facet quality ceiling) |
| 4 | Attention economy as exploitation: surplus extraction via attention capture | Strigoi series (22-3) | Partially applied (diluted by broader evidence) |
| 5 | Individual vs systemic: satisfaction surveys in rigged system don't prove non-exploitation | w_4dj0gp_9Y, oldjoe-2022 | Extracted as gig freedom vs precarity |
| 6 | Algorithmic radicalization circularity: subjective value is circular when preferences are endogenous | Strigoi series (21-2) | Missed (corpus implies, never states) |
| 7 | Marx's wage nuance: "her isin ucreti ayni olmaz" refutes the equal-wages strawman | oldjoe-2022, Kmcy0FiqQYo | Missed (needle in haystack) |

## Score progression

| Run | Ground Truth | Best MCTS Score | Changes |
|-----|-------------|-----------------|---------|
| Baseline | 0/7 | 0.733 | -- |
| Fix 1 | 2.5/7 | 0.800 | Two-phase synthesize |
| Fix 1+2 | 3.0/7 | 0.733 | + Query expansion |

## Design notes from Eren's Claude conversation (April 6-7)

- OWL ontology generation via MAB before relation extraction: the schema of what counts as a relationship should itself be learned
- Lens insights should feed back into the ontology as first-class concept types
- Weight overlay for feedback-driven traversal must be isolated from base graph (prevents bias amplification)
- "Motivate complexity from observed failure, not theory" -- don't add architecture until you see it fail
- Yepis is heterogeneous (essays + transcripts + social + memories) -- Thompson may over-converge. Hub at 0.742 with 351 observations may be feedback-amplified

## The structural ceiling (2026-04-23)

After Phases 0/1/2 landed and the system was run against its own source as a test corpus, one observation kept coming back:

**Crucible's most important feature is its least used one.**

Every reward signal that drives the bandit ultimately resolves to a single referee score. The bandit math is airtight; the feedback loop from answers back to insights is clean. But the referee is an LLM (Advocate/Skeptic/Referee debate) with the same biases as the corpus the model is judging. Skeptic's recurring "merely restates the obvious" pattern flattens 95% of insight scores into the 0.5–0.85 band — too narrow to separate strategies cleanly. The 0.733→0.800→0.733 progression above is not signal noise from the search algorithm; it is the dynamic range of the referee.

The system has exactly one mechanism that breaks the circular trust: `apply_human_feedback` (engine.py) and the `rate` CLI / `rate_insight` MCP tool, weighted **3×** in the bandit. After 640 bandit observations, that mechanism has been called **twice**. The epistemic bootstrap is built but unplugged.

The deeper observation: **crucible is an epistemic system that outsources its own epistemology.** It trusts an LLM to tell it what's worth knowing. Phases 0.3 (pseudo-count determinism), 1 (Mention/Entity split), and 2 (synthesizer) all materially improved the engineering. None of them moved the referee. They never could; that is a different layer, and it is the layer that decides which strategy gets explored next.

Two structural responses live in the cortex task `b174f847` plan:
- **Phase 6.4** — symbolic referee adapter (e.g. burokrat's `evaluate_provision`) plugged into HybridReferee, so a domain that has crisp predicates can short-circuit the LLM judgment.
- **Phase 6.1** — principle/lens strategy that pre-abstracts transferable principles, so the referee scores an abstracted concept rather than a raw chunk and the dynamic range can stretch.

But the *zero-engineering* response is to put `crucible rate <insight_id> <score>` in front of users on every cycle. Either as a CLI prompt at the end of `explore`, or as an MCP-driven UI that surfaces the top N insights for human triage. The system already supports it. Until that loop is closed, every refactor in the codebase is being measured against the system's own prior.
