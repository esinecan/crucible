import os
from dataclasses import dataclass, field


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

    domain_context: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_DOMAIN", "")
    )

    @property
    def domain_preamble(self) -> str:
        if self.domain_context:
            return self.domain_context + "\n\n"
        return ""
