"""Load processed markdown documents (with YAML front matter) into LangChain Documents."""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.documents import Document

from navigator.config import PROCESSED_DIR

REQUIRED_META = ("title", "topic", "jurisdiction", "language")


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split '---\\nyaml\\n---\\nbody' into (meta, body). Tolerates files without front matter."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), parts[2].lstrip("\n")


def load_processed_documents(directory: Path | None = None) -> list[Document]:
    directory = directory or PROCESSED_DIR
    docs: list[Document] = []
    for path in sorted(directory.rglob("*.md")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_front_matter(raw)
        if not body.strip():
            continue
        meta.setdefault("title", path.stem.replace("_", " ").title())
        meta.setdefault("topic", "general")
        meta.setdefault("jurisdiction", "federal")
        meta.setdefault("language", "de")
        meta.setdefault("source_url", None)
        meta.setdefault("fetched_at", None)
        meta["file"] = path.name
        # Chroma metadata must be flat scalars; drop None and non-scalars.
        clean = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
        docs.append(Document(page_content=body, metadata=clean))
    return docs
