"""LLM + embedding factories. Swappable via .env so the same pipeline runs on
Groq (free API), Ollama (fully local) or a fake model (tests)."""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings, settings as default_settings


# ----------------------------------------------------------------------------
# Chat models
# ----------------------------------------------------------------------------
def get_llm(cfg: Settings = default_settings, model: str | None = None) -> BaseChatModel:
    provider = cfg.llm_provider.lower()

    if provider == "groq":
        if not cfg.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and put it in .env, or set LLM_PROVIDER=ollama."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or cfg.groq_model,
            temperature=cfg.temperature,
            api_key=cfg.groq_api_key,
            max_retries=3,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or cfg.ollama_model,
            base_url=cfg.ollama_base_url,
            temperature=cfg.temperature,
        )

    if provider == "fake":
        # Echoes the first retrieved source so tests can check citation plumbing.
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=["This is a fake answer. [1]"])

    raise ValueError(f"Unknown LLM_PROVIDER: {cfg.llm_provider!r}")


# ----------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------
class HashingEmbeddings(Embeddings):
    """Deterministic, dependency-free bag-of-words hashing embeddings.

    Purely lexical (no semantics) - used so the test-suite and CI can run
    without downloading a model. Never use this for real retrieval quality.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings(cfg: Settings = default_settings) -> Embeddings:
    backend = cfg.embedding_backend.lower()
    if backend == "hashing":
        return HashingEmbeddings()

    from langchain_huggingface import HuggingFaceEmbeddings

    # bge models recommend normalised embeddings (cosine similarity).
    return HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
