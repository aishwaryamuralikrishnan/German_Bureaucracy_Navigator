"""Ingestion: processed markdown -> chunks -> embeddings -> Chroma.

Order matters: we embed EVERYTHING first (with progress), and only then reset the collection and
write. If OpenRouter is unreachable, the existing index stays intact instead of being wiped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from navigator.rag.chunking import chunk_documents
from navigator.rag.embeddings import OpenRouterEmbeddings
from navigator.rag.loaders import load_processed_documents
from navigator.rag.retriever import reset_retriever
from navigator.rag.vectorstore import collection_count, get_vectorstore
from navigator.utils.errors import ExternalServiceError
from navigator.utils.logging import get_logger, log_event

log = get_logger(__name__)

# (stage, done, total, message)
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass
class IngestReport:
    documents: int
    chunks: int
    indexed: int
    embedding_model: str
    embedding_dim: int


def ingest(progress: ProgressCallback | None = None, batch_size: int = 32, embeddings=None) -> IngestReport:
    def report(stage: str, done: int, total: int, msg: str) -> None:
        if progress:
            progress(stage, done, total, msg)

    report("load", 0, 1, "Loading processed documents…")
    docs = load_processed_documents()
    if not docs:
        raise RuntimeError("No documents found in data/processed. Add markdown files or run scripts/fetch_sources.py.")
    report("load", 1, 1, f"Loaded {len(docs)} documents")

    report("chunk", 0, 1, "Chunking…")
    chunks = chunk_documents(docs)
    report("chunk", 1, 1, f"Created {len(chunks)} chunks")

    embeddings = embeddings or OpenRouterEmbeddings()
    model_name = getattr(embeddings, "model", type(embeddings).__name__)

    # --- 1. connectivity check: fail fast with a readable reason instead of hanging silently
    report("ping", 0, 1, f"Testing OpenRouter embeddings ({model_name})…")
    if hasattr(embeddings, "ping"):
        dim, secs = embeddings.ping()
        report("ping", 1, 1, f"OpenRouter reachable — {dim}-dim embeddings, {secs}s round trip")
    else:
        dim = len(embeddings.embed_query("ping"))
        report("ping", 1, 1, f"Embedder ready ({dim} dims)")

    # --- 2. embed everything BEFORE touching the existing index
    texts = [c.page_content for c in chunks]
    total = len(texts)
    report("embed", 0, total, f"Embedding {total} chunks in batches of {batch_size}…")

    def _on_batch(done: int, _total: int) -> None:
        report("embed", done, total, f"Embedded {done}/{total} chunks")

    def _on_retry(msg: str) -> None:
        report("embed", 0, total, f"⚠ {msg}")

    try:
        if isinstance(embeddings, OpenRouterEmbeddings):
            embeddings.batch_size = batch_size
            vectors = embeddings.embed_documents(texts, progress=_on_batch, on_retry=_on_retry)
        else:
            vectors = embeddings.embed_documents(texts)
            report("embed", total, total, f"Embedded {total}/{total} chunks")
    except ExternalServiceError:
        report("embed", 0, total, "Embedding failed — existing index left untouched.")
        raise

    # --- 3. replace the collection contents
    store = get_vectorstore(embeddings)
    report("write", 0, total, "Resetting collection…")
    store.reset_collection()
    collection = store._collection  # noqa: SLF001 — needed to write precomputed vectors
    for start in range(0, total, 100):
        end = min(start + 100, total)
        collection.upsert(
            ids=[c.metadata["chunk_id"] for c in chunks[start:end]],
            embeddings=vectors[start:end],
            documents=texts[start:end],
            metadatas=[c.metadata for c in chunks[start:end]],
        )
        report("write", end, total, f"Wrote {end}/{total} chunks to Chroma")

    indexed = collection_count(store)
    reset_retriever()
    log_event(log, "ingest_done", documents=len(docs), chunks=total, indexed=indexed, model=model_name)
    return IngestReport(len(docs), total, indexed, model_name, dim)
