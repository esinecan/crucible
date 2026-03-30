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

    ollama_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_EMBED_MODEL", "nomic-embed-text")
    )
    embed_dim: int = 768

    eval_model: str = field(
        default_factory=lambda: os.getenv("CRUCIBLE_EVAL_MODEL", "gemma3:4b")
    )

    chunk_max_chars: int = 2000
    chunk_overlap_chars: int = 200
