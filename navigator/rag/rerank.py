"""Cross-encoder reranking of the fused candidates.

Providers:
  openrouter — Cohere Rerank served through OpenRouter's /api/v1/rerank (no extra key, ~$0.002 per search)
  local      — sentence-transformers CrossEncoder (large download, offline)
  none       — keep the fused order
Any failure falls back to the fused order so retrieval never breaks because of the reranker.
"""

from __future__ import annotations

import time

import requests

from navigator.config import get_api_key, get_settings
from navigator.core.schemas import RetrievedChunk
from navigator.utils.logging import get_logger, log_event

log = get_logger(__name__)


class NoOpReranker:
    name = "none"

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]


class OpenRouterReranker:
    """Cohere Rerank via OpenRouter. Documents are sent as plain strings; scores are relevance in [0, 1]."""

    def __init__(self, model: str, timeout: float = 20.0) -> None:
        s = get_settings()
        self.model = model
        self.name = f"openrouter:{model}"
        self.url = s.llm.base_url.rstrip("/") + "/rerank"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {get_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/",
            "X-Title": "German Bureaucracy Navigator",
        })

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        payload = {"model": self.model, "query": query, "documents": [c.text[:4000] for c in chunks], "top_n": min(top_k, len(chunks))}
        t0 = time.time()
        try:
            resp = self.session.post(self.url, json=payload, timeout=self.timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            results = resp.json().get("results", [])
            ranked = []
            for r in results:
                idx = int(r["index"])
                if 0 <= idx < len(chunks):
                    rel = round(float(r.get("relevance_score", 0.0)), 4)
                    ranked.append(chunks[idx].model_copy(update={"score": rel, "relevance": rel}))
            if not ranked:
                raise RuntimeError("empty rerank result")
            log_event(log, "rerank", model=self.model, n_in=len(chunks), n_out=len(ranked), seconds=round(time.time() - t0, 2))
            return ranked[:top_k]
        except Exception as exc:  # never let the reranker break retrieval
            log.warning(f"OpenRouter rerank failed, keeping fused order: {exc}")
            return chunks[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # lazy import: heavy dependency

        self.name = f"local:{model_name}"
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self.model.predict([(query, c.text) for c in chunks])
        paired = sorted(zip(chunks, scores), key=lambda p: float(p[1]), reverse=True)
        # sentence-transformers cross-encoders return logits; squash to [0, 1] so the coverage rule can use them
        import math
        return [c.model_copy(update={"score": round(float(s), 4), "relevance": round(1 / (1 + math.exp(-float(s))), 4)}) for c, s in paired[:top_k]]


def get_reranker():
    cfg = get_settings().retrieval.reranker
    if not cfg.get("enabled", False):
        return NoOpReranker()
    provider = str(cfg.get("provider", "openrouter")).lower()
    if provider == "openrouter":
        if not get_api_key():
            return NoOpReranker()
        return OpenRouterReranker(cfg.get("model", "cohere/rerank-4-fast"))
    if provider == "local":
        try:
            return CrossEncoderReranker(cfg.get("model", "BAAI/bge-reranker-v2-m3"))
        except ImportError:
            log.warning("local reranker requested but sentence-transformers is not installed; falling back to no-op")
            return NoOpReranker()
    return NoOpReranker()
