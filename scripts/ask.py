"""Ask a question from the terminal.

  python scripts/ask.py "What is the regulatory retail portfolio limit?"
  python scripts/ask.py            # interactive loop
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chain import RAGChain  # noqa: E402
from rag.config import settings  # noqa: E402


def show(ans) -> None:
    print("\n" + ans.answer + "\n")
    print("Sources:")
    for s in ans.cited_sources():
        print(f"  [{s.n}] {s.label}  —  {s.text[:120].replace(chr(10), ' ')}…")


def main() -> None:
    chain = RAGChain(settings)
    if len(sys.argv) > 1:
        show(chain.ask(" ".join(sys.argv[1:])))
        return
    history: list[dict] = []
    print("Ask a question (Ctrl-C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        ans = chain.ask(q, history=history)
        show(ans)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans.answer}]


if __name__ == "__main__":
    main()
