"""Evaluate the RAG pipeline on a question set.

Reports
  * Retrieval:  hit-rate@k (did any retrieved chunk come from the expected
                document?), MRR, and per-question ranks - no LLM needed.
  * Generation: answer accuracy (%) via an LLM judge with a strict
                correct/incorrect rubric, plus RAGAS metrics
                (faithfulness, answer_correctness, context_precision,
                context_recall).

Usage
  python eval/run_eval.py                          # eval/sbp_bprd_2024.jsonl, full run
  python eval/run_eval.py --retrieval-only         # no LLM calls at all
  python eval/run_eval.py --no-ragas               # accuracy judge only
  python eval/run_eval.py --set eval/demo.jsonl    # the fictional sample corpus
  python eval/run_eval.py --k 5 --k 10             # several k values

Results are written to eval/results/<timestamp>.{json,csv,md}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chain import RAGChain  # noqa: E402
from rag.config import settings  # noqa: E402
from rag.llm import get_embeddings, get_llm  # noqa: E402
from rag.retriever import get_retriever  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def load_set(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def source_matches(source: str, expected: list[str]) -> bool:
    s = Path(source).name.lower()
    return any(s.startswith(e.lower()) for e in expected)


def first_hit_rank(docs, expected: list[str] | str | None) -> int | None:
    if isinstance(expected, str):
        expected_names = [expected]
    elif isinstance(expected, list):
        expected_names = expected
    else:
        expected_names = []

    for i, d in enumerate(docs, start=1):
        if source_matches(str(d.metadata.get("source", "")), expected_names):
            return i
    return None


JUDGE_PROMPT = """You are grading a question-answering system against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Is the system answer CORRECT? Rules:
- CORRECT if it states the same key facts (figures, dates, names, conditions) as the reference, \
even if worded differently or with extra correct detail. Citation markers like [1] are fine.
- INCORRECT if it contradicts the reference, misses the main fact, gives a wrong number/date, \
or says the information was not found when the reference has it.

Reply with exactly one word: CORRECT or INCORRECT."""


def judge(llm, question: str, reference: str, answer: str) -> bool:
    out = llm.invoke(
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    )
    text = (out.content if hasattr(out, "content") else str(out)).strip().upper()
    return text.startswith("CORRECT")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--set", default=str(Path(__file__).resolve().parent / "sbp_bprd_2024.jsonl")
    )
    ap.add_argument(
        "--k",
        type=int,
        action="append",
        help="k for hit-rate (repeatable). Default: TOP_K and 10",
    )
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument(
        "--limit", type=int, default=None, help="only evaluate the first N questions"
    )
    args = ap.parse_args()

    ks = sorted(set(args.k or [settings.top_k, 10]))
    max_k = max(ks)
    rows = load_set(Path(args.set))[: args.limit]
    print(f"Eval set: {args.set}  ({len(rows)} questions)")
    print(
        f"Config: embeddings={settings.embedding_model} hybrid={settings.hybrid} chunk={settings.chunk_size}/{settings.chunk_overlap} llm={settings.llm_provider}"
    )

    retriever = get_retriever(settings, k=max_k)
    if retriever.size == 0:
        sys.exit("The index is empty - run `python scripts/ingest.py` first.")

    # ---- retrieval metrics -------------------------------------------------
    per_q = []
    t0 = time.time()
    for r in rows:
        docs = retriever.invoke(r["question"])
        expected = r.get("sources") or r.get("source") or []
        rank = first_hit_rank(docs, expected)
        per_q.append(
            {
                "id": r["id"],
                "question": r["question"],
                "reference": r.get("reference", r.get("answer", "")),
                "expected": expected,
                "rank": rank,
                "retrieved": [str(d.metadata.get("source")) for d in docs],
                "docs": docs,
            }
        )
    retrieval_secs = time.time() - t0

    summary: dict = {
        "n": len(rows),
        "set": Path(args.set).name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "embedding_model": settings.embedding_model
            if settings.embedding_backend != "hashing"
            else "hashing (test only)",
            "hybrid": settings.hybrid,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "llm": {
                "groq": f"groq:{settings.groq_model}",
                "ollama": f"ollama:{settings.ollama_model}",
            }.get(settings.llm_provider, settings.llm_provider),
        },
    }
    for k in ks:
        hits = sum(1 for q in per_q if q["rank"] is not None and q["rank"] <= k)
        summary[f"hit_rate@{k}"] = hits / len(per_q)
    summary["mrr"] = sum(1 / q["rank"] for q in per_q if q["rank"]) / len(per_q)
    summary["retrieval_secs_per_q"] = retrieval_secs / len(per_q)

    print("\nRetrieval")
    for k in ks:
        print(f"  hit-rate@{k}: {summary[f'hit_rate@{k}']:.1%}")
    print(f"  MRR:         {summary['mrr']:.3f}")
    misses = [q for q in per_q if q["rank"] is None or q["rank"] > settings.top_k]
    if misses:
        print(
            f"  misses at k={settings.top_k}: "
            + ", ".join(f"#{q['id']}" for q in misses)
        )

    # ---- generation --------------------------------------------------------
    if not args.retrieval_only:
        chain = RAGChain(
            settings, retriever=get_retriever(settings)
        )  # production top_k
        judge_llm = get_llm(
            settings,
            model=settings.judge_model if settings.llm_provider == "groq" else None,
        )
        print("\nGenerating answers…")
        correct = 0
        for q in per_q:
            ans = chain.ask(q["question"])
            q["answer"] = ans.answer
            q["contexts"] = [s.text for s in ans.sources]
            q["correct"] = judge(judge_llm, q["question"], q["reference"], ans.answer)
            correct += q["correct"]
            print(
                f"  #{q['id']:>2} {'✓' if q['correct'] else '✗'}  {q['question'][:70]}"
            )
            time.sleep(0.3)  # be gentle with free-tier rate limits
        summary["answer_accuracy"] = correct / len(per_q)
        print(
            f"\nAnswer accuracy (LLM judge): {summary['answer_accuracy']:.1%}  ({correct}/{len(per_q)})"
        )

        if not args.no_ragas:
            summary["ragas"] = run_ragas(per_q, judge_llm)

    # ---- write results -----------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    for q in per_q:
        q.pop("docs", None)
    (stem.with_suffix(".json")).write_text(
        json.dumps(
            {"summary": summary, "questions": per_q}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    import csv

    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "question",
                "rank",
                "correct",
                "answer",
                "reference",
                "expected",
                "retrieved",
            ]
        )
        for q in per_q:
            w.writerow(
                [
                    q["id"],
                    q["question"],
                    q["rank"],
                    q.get("correct", ""),
                    q.get("answer", ""),
                    q["reference"],
                    "|".join(q["expected"]),
                    "|".join(q["retrieved"]),
                ]
            )

    (stem.with_suffix(".md")).write_text(render_md(summary), encoding="utf-8")
    print(f"\nWrote {stem}.json / .csv / .md")
    print(render_md(summary))


def run_ragas(per_q: list[dict], judge_llm) -> dict:
    """RAGAS metrics with the same free LLM as judge and local embeddings."""
    try:
        from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
        from ragas.metrics import (
            answer_correctness,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        print(f"RAGAS not available ({e}); skipping. pip install ragas")
        return {}

    print(
        "\nRunning RAGAS (this makes several LLM calls per question; free tiers may take a few minutes)…"
    )
    samples = [
        SingleTurnSample(
            user_input=q["question"],
            response=q["answer"],
            retrieved_contexts=q["contexts"],
            reference=q["reference"],
        )
        for q in per_q
    ]
    try:
        result = evaluate(
            EvaluationDataset(samples=samples),
            metrics=[
                faithfulness,
                answer_correctness,
                context_precision,
                context_recall,
            ],
            llm=judge_llm,
            embeddings=get_embeddings(settings),
            run_config=RunConfig(max_workers=2, timeout=180, max_retries=5),
            raise_exceptions=False,
            show_progress=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"RAGAS failed: {e}")
        return {"error": str(e)}

    df = result.to_pandas()
    out = {}
    for col in [
        "faithfulness",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]:
        if col in df:
            vals = [
                v
                for v in df[col].tolist()
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            ]
            out[col] = sum(vals) / len(vals) if vals else None
    print(
        "RAGAS:", {k: (round(v, 3) if v is not None else None) for k, v in out.items()}
    )
    return out


def render_md(s: dict) -> str:
    lines = [
        f"## Eval results — {s['set']} ({s['n']} questions) — {s['timestamp']}",
        "",
        f"Config: `{json.dumps(s['config'])}`",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in s.items():
        if k.startswith("hit_rate") or k in {"mrr", "answer_accuracy"}:
            lines.append(f"| {k} | {v:.1%} |" if k != "mrr" else f"| {k} | {v:.3f} |")
    for k, v in (s.get("ragas") or {}).items():
        if isinstance(v, (int, float)):
            lines.append(f"| ragas.{k} | {v:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
