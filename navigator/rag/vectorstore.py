"""Chroma vector store behind a small factory so it can be swapped later (e.g. Qdrant)."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from navigator.config import get_settings, resolve_path

EMBED_MODEL_META_KEY = "embedding_model"


def get_vectorstore(embeddings: Embeddings) -> Chroma:
    s = get_settings().vectorstore
    persist_dir = resolve_path(s.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=s.collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def collection_count(store: Chroma) -> int:
    try:
        return store._collection.count()  # noqa: SLF001 — Chroma exposes no public count()
    except Exception:
        return 0


def get_all_chunks(store: Chroma) -> tuple[list[str], list[str], list[dict]]:
    """Return (ids, texts, metadatas) for the whole collection — used to build BM25."""
    if collection_count(store) == 0:
        return [], [], []
    res = store.get(include=["documents", "metadatas"])
    return res.get("ids", []), res.get("documents", []), res.get("metadatas", [])


def document_inventory(store: Chroma) -> list[dict]:
    """Group chunks by source document for the Knowledge Base page."""
    _, _, metas = get_all_chunks(store)
    inv: dict[str, dict] = {}
    for m in metas:
        key = m.get("source_url") or m.get("title") or "unknown"
        row = inv.setdefault(
            key,
            {
                "title": m.get("title", "untitled"),
                "source_url": m.get("source_url"),
                "topic": m.get("topic"),
                "jurisdiction": m.get("jurisdiction"),
                "language": m.get("language"),
                "fetched_at": m.get("fetched_at"),
                "chunks": 0,
            },
        )
        row["chunks"] += 1
    return sorted(inv.values(), key=lambda r: (str(r["topic"]), str(r["title"])))
