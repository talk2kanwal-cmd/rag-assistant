"""Central configuration. Everything can be overridden with environment variables
(see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # --- paths ---
    docs_dir: Path = field(default_factory=lambda: ROOT / _env("DOCS_DIR", "data/docs"))
    chroma_dir: Path = field(
        default_factory=lambda: ROOT / _env("CHROMA_DIR", "data/chroma")
    )
    collection: str = field(
        default_factory=lambda: _env("CHROMA_COLLECTION", "sbp_docs")
    )

    # --- chunking ---
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "900")))
    chunk_overlap: int = field(
        default_factory=lambda: int(_env("CHUNK_OVERLAP", "150"))
    )

    # --- embeddings ---
    # "bge" -> BAAI/bge-small-en-v1.5 (default, ~33M params, runs on CPU)
    # "hashing" -> offline deterministic fallback, ONLY for tests/CI (no semantics)
    embedding_backend: str = field(
        default_factory=lambda: _env("EMBEDDING_BACKEND", "bge")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # --- retrieval ---
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    dense_k: int = field(default_factory=lambda: int(_env("DENSE_K", "12")))
    bm25_k: int = field(default_factory=lambda: int(_env("BM25_K", "12")))
    hybrid: bool = field(
        default_factory=lambda: _env("HYBRID", "true").lower() == "true"
    )

    # --- LLM ---
    # provider: "groq" (free API tier) | "ollama" (fully local) | "fake" (tests)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq"))
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "openai/gpt-oss-20b")
    )
    ollama_model: str = field(
        default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1:8b")
    )
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    temperature: float = field(
        default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.0"))
    )

    # --- eval ---
    # A smaller/faster judge keeps RAGAS within Groq's free-tier rate limits.
    judge_model: str = field(
        default_factory=lambda: _env("JUDGE_MODEL", "openai/gpt-oss-20b")
    )


settings = Settings()
