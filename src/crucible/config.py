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

    chunk_max_chars: int = 2000
    chunk_overlap_chars: int = 200

    # MCTS answer mode
    mcts_max_depth: int = 5
    mcts_max_iterations: int = 20
    mcts_uct_c: float = 1.41  # exploration constant (sqrt(2) is standard)

    domain_context: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_DOMAIN", "")
    )

    @property
    def domain_preamble(self) -> str:
        if self.domain_context:
            return self.domain_context + "\n\n"
        return ""
