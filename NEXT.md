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
