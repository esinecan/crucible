import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_NEO4J_URI", "bolt://localhost:7689")
    )
    neo4j_user: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_NEO4J_USER", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_NEO4J_PASSWORD", "nous-dev")
    )

    embed_dim: int = 768

    eval_model: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_EVAL_MODEL", "deepseek-chat")
    )

    # LLM API credentials. Previously read via os.getenv() at call time in
    # 5 separate modules; consolidated here so callers get one injection
    # point and tests can swap keys without monkeypatching os.environ.
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )

    chunk_max_chars: int = 2000
    chunk_overlap_chars: int = 200

    # MCTS answer mode
    mcts_max_depth: int = 5
    mcts_max_iterations: int = 20
    mcts_uct_c: float = 1.41  # exploration constant (sqrt(2) is standard)

    # Bandit update semantics. "pseudo_count" adds the raw score directly to
    # alpha (and 1-score to beta) and is deterministic across replays. The
    # historical "bernoulli" mode flips a coin biased by the score and bumps
    # one side by 1 — same posterior in expectation but with per-replay drift
    # because random.random() is consumed on every load_from_log entry.
    bandit_update_mode: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_BANDIT_UPDATE_MODE", "pseudo_count")
    )

    # Entity synthesizer cost cap. Each call hits the LLM once per entity, so
    # large dirty sets can rack up tokens; the synthesizer exits cleanly when
    # this many calls have been made and reports remaining-dirty count.
    max_synthesis_calls: int = field(
        default_factory=lambda: int(os.getenv("CRUCIBLE_MAX_SYNTHESIS_CALLS", "500"))
    )

    # Per-corpus state directory. When a corpus_id is supplied to InsightEngine
    # or MCTSEngine, reward logs and bandit snapshots are routed to
    # `<state_dir>/<corpus_id>/` so a Yepis-warmed bandit doesn't steer
    # discovery on a fresh corpus. When no corpus_id is supplied, the engines
    # fall back to the historical CWD-based paths (`./insight_rewards.jsonl`
    # etc.) so existing scripts keep working.
    state_dir: str = field(
        default_factory=lambda: os.getenv(
            "CRUCIBLE_STATE_DIR",
            str(Path.home() / ".crucible"),
        )
    )

    # Strategy escape hatch. Comma-separated names (bridge, outlier, hub,
    # meta, contradiction, gap) excluded from Thompson Sampling selection.
    # The gap strategy fires false positives on code corpora because every
    # called function is "referenced more than mentioned" — a power user who
    # knows their corpus type can disable gap (or any other ill-fitting
    # strategy) without waiting for the bandit to learn it the slow way.
    # An explicit `--strategy gap` CLI flag still overrides this.
    disabled_strategies: list[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv("CRUCIBLE_DISABLED_STRATEGIES", "").split(",")
            if s.strip()
        ]
    )

    # Pre-persist grounding check. When enabled, every L1 insight gets one
    # extra LLM call after the evaluator runs: the model is shown the
    # insight's claim and one of its source chunks and asked whether the
    # source supports, contradicts, or is silent on the claim. Contradicted
    # insights at confidence >= 0.7 are dropped before persist; unsupported
    # insights are persisted with a `context.grounding='unsupported'` marker
    # and their bandit reward is halved. Off by default because it doubles
    # per-insight LLM cost; the cocrucible gap hallucinations would all be
    # caught by it (every gap claim was directly refutable by reading any
    # chunk of the source it referenced).
    grounding_check: bool = field(
        default_factory=lambda: os.getenv(
            "CRUCIBLE_GROUNDING_CHECK", ""
        ).strip().lower() in ("1", "true", "yes", "on")
    )

    domain_context: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_DOMAIN", "")
    )

    @property
    def domain_preamble(self) -> str:
        if self.domain_context:
            return self.domain_context + "\n\n"
        return ""
