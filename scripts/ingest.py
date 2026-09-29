"""Build (or rebuild) the Chroma knowledge base from data/processed/**/*.md.

Usage:  python scripts/ingest.py
Requires the OPENROUTER_API_KEY environment variable (embeddings are created via OpenRouter).

IMPORTANT: close the running Streamlit app first (Ctrl+C in its terminal), or use the
"Rebuild knowledge base" button inside the app instead. Chroma's on-disk database must not be
written by two processes at the same time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navigator.rag.ingest import ingest  # noqa: E402
from navigator.utils.errors import NavigatorError  # noqa: E402

_T0 = time.time()


def _progress(stage: str, done: int, total: int, msg: str) -> None:
    bar = ""
    if stage in {"embed", "write"} and total:
        pct = int(done / total * 30)
        bar = "[" + "#" * pct + "-" * (30 - pct) + "] "
    print(f"{time.time() - _T0:6.1f}s {stage:>6} {bar}{msg}", flush=True)


def main() -> int:
    print("Rebuilding the knowledge base. If the Streamlit app is running, stop it first.\n", flush=True)
    try:
        report = ingest(progress=_progress)
    except NavigatorError as exc:
        print(f"\nERROR: {exc}\n{exc.user_message}", flush=True)
        print("Tip: run  python scripts/check_openrouter.py  to test the API key, credits and embedding model.", flush=True)
        return 1
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", flush=True)
        return 1
    print(
        f"\nDone in {time.time() - _T0:.1f}s: indexed {report.indexed} chunks from {report.documents} documents "
        f"({report.embedding_dim}-dim {report.embedding_model}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
