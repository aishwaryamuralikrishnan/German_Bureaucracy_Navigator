"""What an evaluation run cost.

Preferred: OpenRouter's key endpoint reports the credits the key has used so far (USD). Reading it just before and
just after a run gives the run's real cost including everything the token counts miss — the judge, the query
expansion, embeddings and the reranker. Fallback: an estimate from the answering model's recorded tokens with a
small price table, clearly labelled as such.
"""

from __future__ import annotations

from typing import Any

from navigator.config import get_api_key, get_settings
from navigator.utils.logging import get_logger

log = get_logger(__name__)

# USD per 1M tokens (input, output) for the models this project offers; used only for the fallback estimate.
PRICES_PER_M = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1": (2.00, 8.00),
    "openai/gpt-4o": (2.50, 10.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-pro": (1.25, 10.00),
}


def credits_used(timeout: float = 10.0) -> float | None:
    """Total USD credits this API key has used so far, or None when the endpoint cannot be read."""
    try:
        import httpx

        base = get_settings().llm.base_url.rstrip("/")
        r = httpx.get(f"{base}/auth/key", headers={"Authorization": f"Bearer {get_api_key()}"}, timeout=timeout)
        r.raise_for_status()
        usage = (r.json().get("data") or {}).get("usage")
        return float(usage) if usage is not None else None
    except Exception as exc:  # cost is bookkeeping — never let it break a run
        log.warning(f"could not read OpenRouter key usage: {exc}")
        return None


def estimate_from_tokens(e2e: dict[str, Any] | None) -> float | None:
    """Fallback: answering-model tokens × list price. Ignores judge, embeddings and reranker (says so in the label)."""
    if not e2e:
        return None
    tok = e2e.get("tokens") or {}
    price = PRICES_PER_M.get(str(e2e.get("model")))
    if not price or not tok:
        return None
    return round(tok.get("input", 0) / 1e6 * price[0] + tok.get("output", 0) / 1e6 * price[1], 4)


def cost_label(meta: dict[str, Any], e2e: dict[str, Any] | None = None) -> str:
    """One phrase for the page header and the PDF."""
    cost = meta.get("cost_usd")
    if cost is not None:
        if meta.get("cost_source") == "openrouter_key_usage":
            return f"total cost ${cost:.2f} (OpenRouter credits used by this run — agent, judge, embeddings and reranker)"
        return f"cost ≈ ${cost:.2f} (estimated from the answering model's tokens only)"
    est = estimate_from_tokens(e2e)
    return f"cost ≈ ${est:.2f} (estimated from the answering model's tokens only)" if est is not None else "cost n/a"
