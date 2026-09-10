"""Index every supported file in a directory (default: data/docs).

  python scripts/ingest.py                 # index data/docs
  python scripts/ingest.py path/to/dir     # index another folder
  python scripts/ingest.py --reset         # wipe the collection first
  python scripts/ingest.py --list          # show what is indexed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import settings  # noqa: E402
from rag.ingest import ingest_directory, list_sources, reset_store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default=None)
    ap.add_argument("--reset", action="store_true", help="delete the existing collection first")
    ap.add_argument("--list", action="store_true", help="list indexed sources and exit")
    args = ap.parse_args()

    if args.list:
        for src, n in list_sources(settings).items():
            print(f"{n:5d}  {src}")
        return

    if args.reset:
        reset_store(settings)
        print("Collection reset.")

    t0 = time.time()
    stats = ingest_directory(args.directory, settings)
    print(f"Indexed {stats['files']} files / {stats['pages']} pages / {stats['chunks']} chunks in {time.time()-t0:.1f}s")
    for s in stats["skipped"]:
        print("  skipped:", s)


if __name__ == "__main__":
    main()
