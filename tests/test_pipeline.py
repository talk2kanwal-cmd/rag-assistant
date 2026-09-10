"""Offline tests: run with fake LLM + hashing embeddings so CI needs no
network or API key.   pytest -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["LLM_PROVIDER"] = "fake"

from rag.config import Settings  # noqa: E402
from rag.ingest import chunk_documents, ingest_directory, list_sources, load_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "sample_docs"


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    c = Settings()
    c.chroma_dir = tmp_path_factory.mktemp("chroma")
    c.collection = "test"
    return c


def test_loaders_and_chunking(cfg):
    docs = []
    for p in SAMPLES.iterdir():
        docs += load_path(p)
    assert len(docs) == 3
    chunks = chunk_documents(docs, cfg)
    assert chunks
    assert all(len(c.page_content) <= cfg.chunk_size + 50 for c in chunks)
    assert len({c.metadata["id"] for c in chunks}) == len(chunks), (
        "chunk ids must be unique"
    )


def test_ingest_is_idempotent(cfg):
    a = ingest_directory(SAMPLES, cfg)
    b = ingest_directory(SAMPLES, cfg)
    assert a["chunks"] == b["chunks"] > 0
    assert sum(list_sources(cfg).values()) == a["chunks"], (
        "re-ingest must not duplicate chunks"
    )


def test_hybrid_retrieval_finds_right_doc(cfg):
    from rag.retriever import get_retriever

    ingest_directory(SAMPLES, cfg)
    r = get_retriever(cfg, k=3)
    docs = r.invoke("EBITDA margin subscribers 4G")
    assert docs[0].metadata["source"] == "demo_telecom_annual_report_2024.md"


def test_chain_returns_answer_and_citations(cfg):
    from rag.chain import RAGChain

    ingest_directory(SAMPLES, cfg)
    chain = RAGChain(cfg)
    ans = chain.ask("What is the minimum CCTV resolution?")
    assert ans.answer
    assert ans.sources and ans.cited == [1]
    assert ans.cited_sources()[0].n == 1


def test_ground_truth_eval_set_is_compatible():
    import json

    rows = [
        json.loads(l)
        for l in (ROOT / "eval" / "ground_truth.jsonl").read_text().splitlines()
        if l.strip()
    ]
    assert rows
    assert all({"id", "question", "answer", "source"} <= r.keys() for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)


def test_eval_set_is_well_formed():
    import json

    for name in ("sbp_bprd_2024.jsonl", "demo.jsonl"):
        rows = [
            json.loads(l)
            for l in (ROOT / "eval" / name).read_text().splitlines()
            if l.strip()
        ]
        assert all({"id", "question", "reference", "sources"} <= r.keys() for r in rows)
        assert len({r["id"] for r in rows}) == len(rows)
    sbp = [
        l
        for l in (ROOT / "eval" / "sbp_bprd_2024.jsonl").read_text().splitlines()
        if l.strip()
    ]
    assert len(sbp) == 30
