"""Hybrid retrieval: dense (ChromaDB, bge embeddings) + sparse (BM25), fused
with Reciprocal Rank Fusion. Falls back to dense-only when HYBRID=false."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr
from rank_bm25 import BM25Okapi

from .config import Settings, settings as default_settings
from .ingest import get_vectorstore

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HybridRetriever(BaseRetriever):
    """Dense + BM25 with RRF. Built once from the Chroma collection."""

    cfg: Settings = Field(default=default_settings)
    k: int = 5
    _vs: Any = PrivateAttr(default=None)
    _bm25: BM25Okapi | None = PrivateAttr(default=None)
    _docs: list[Document] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self, cfg: Settings = default_settings, k: int | None = None, **kwargs
    ):
        super().__init__(cfg=cfg, k=k or cfg.top_k, **kwargs)
        self._vs = get_vectorstore(cfg)
        if cfg.hybrid:
            self._build_bm25()

    # ---- sparse index -------------------------------------------------------
    def _build_bm25(self) -> None:
        data = self._vs.get(include=["documents", "metadatas"])
        self._docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        if self._docs:
            self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self._docs])

    def refresh(self) -> None:
        """Call after ingesting new documents so BM25 sees them."""
        if self.cfg.hybrid:
            self._build_bm25()

    @property
    def size(self) -> int:
        return len(self._docs) if self._docs else self._vs._collection.count()

    # ---- retrieval ---------------------------------------------------------
    def _dense(self, query: str, k: int) -> list[Document]:
        return self._vs.similarity_search(query, k=k)

    def _sparse(self, query: str, k: int) -> list[Document]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in order if scores[i] > 0]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        dense = self._dense(query, self.cfg.dense_k)
        if not self.cfg.hybrid:
            return dense[: self.k]

        sparse = self._sparse(query, self.cfg.bm25_k)

        # Reciprocal Rank Fusion (Cormack et al. 2009), c=60.
        fused: dict[str, float] = {}
        by_id: dict[str, Document] = {}
        for ranked in (dense, sparse):
            for rank, doc in enumerate(ranked):
                did = (
                    doc.metadata.get("id")
                    or f"{doc.metadata.get('source')}::{doc.metadata.get('chunk')}"
                )
                by_id.setdefault(did, doc)
                fused[did] = fused.get(did, 0.0) + 1.0 / (60 + rank + 1)

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: self.k]
        out = []
        for did, score in top:
            d = by_id[did]
            d.metadata["rrf_score"] = round(score, 5)
            out.append(d)
        return out


def get_retriever(
    cfg: Settings = default_settings, k: int | None = None
) -> HybridRetriever:
    return HybridRetriever(cfg=cfg, k=k)
