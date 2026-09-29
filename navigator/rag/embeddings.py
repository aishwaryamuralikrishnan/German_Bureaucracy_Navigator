"""Embeddings via OpenRouter's OpenAI-compatible /embeddings endpoint.

We implement LangChain's `Embeddings` interface directly (instead of OpenAIEmbeddings)
so we can force `encoding_format="float"`, send plain strings (no tiktoken pre-tokenising),
batch with a progress callback, keep timeouts short, and raise typed, readable errors.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx
from langchain_core.embeddings import Embeddings
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from navigator.config import get_api_key, get_settings
from navigator.utils.errors import ConfigError, ExternalServiceError
from navigator.utils.logging import get_logger, log_event

log = get_logger(__name__)

ProgressCallback = Callable[[int, int], None]

_STATUS_HINTS = {
    401: "the API key was rejected — check OPENROUTER_API_KEY",
    402: "your OpenRouter account has no credits left — top up at https://openrouter.ai/credits",
    403: "access to this model is forbidden for your key",
    404: "the embedding model id was not found on OpenRouter — check embeddings.model in settings.yaml",
    429: "rate limit reached — wait a moment and retry",
}


class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model: str | None = None, batch_size: int | None = None) -> None:
        s = get_settings()
        key = get_api_key()
        if not key:
            raise ConfigError(f"Environment variable {s.llm.api_key_env} is not set.")
        self.model = model or s.embeddings.model
        self.batch_size = batch_size or s.embeddings.batch_size
        self.timeout = float(s.embeddings.get("timeout_seconds", 30))
        self.attempts = int(s.embeddings.get("max_attempts", 3))
        # Idle pooled connections are dropped after 10 s: a connection the server/proxy has silently closed would
        # otherwise make the first request after a pause hang for the whole timeout before the retry succeeds.
        http_client = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=10.0),
        )
        self._client = OpenAI(
            api_key=key,
            base_url=s.llm.base_url,
            timeout=self.timeout,
            max_retries=0,  # we retry ourselves so we can report progress
            http_client=http_client,
            default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "German Bureaucracy Navigator"},
        )

    def _embed_batch(self, texts: list[str], on_retry: Callable[[str], None] | None = None, timeout: float | None = None) -> list[list[float]]:
        last: Exception | None = None
        client = self._client if timeout is None else self._client.with_options(timeout=timeout)
        for attempt in range(1, self.attempts + 1):
            try:
                resp = client.embeddings.create(model=self.model, input=texts, encoding_format="float")
                ordered = sorted(resp.data, key=lambda d: d.index)
                return [d.embedding for d in ordered]
            except (AuthenticationError, NotFoundError) as exc:  # never retry these
                raise ExternalServiceError(f"OpenRouter embeddings failed (HTTP {exc.status_code}): {_STATUS_HINTS.get(exc.status_code, exc.message)}") from exc
            except APIStatusError as exc:
                if exc.status_code == 402:
                    raise ExternalServiceError(f"OpenRouter embeddings failed (HTTP 402): {_STATUS_HINTS[402]}") from exc
                last = exc
                reason = f"HTTP {exc.status_code}: {_STATUS_HINTS.get(exc.status_code, exc.message)}"
            except RateLimitError as exc:
                last = exc
                reason = _STATUS_HINTS[429]
            except APITimeoutError as exc:
                last = exc
                reason = f"no response within {(timeout or self.timeout):.0f}s"
            except APIConnectionError as exc:
                last = exc
                reason = f"connection error ({exc.__class__.__name__}) — check internet/proxy"
            if attempt < self.attempts:
                wait = min(2 ** attempt, 8)
                msg = f"embedding attempt {attempt}/{self.attempts} failed: {reason}; retrying in {wait}s"
                log.warning(msg)
                if on_retry:
                    on_retry(msg)
                time.sleep(wait)
        raise ExternalServiceError(f"OpenRouter embeddings unavailable after {self.attempts} attempts: {last}") from last

    def embed_documents(
        self,
        texts: list[str],
        progress: ProgressCallback | None = None,
        on_retry: Callable[[str], None] | None = None,
    ) -> list[list[float]]:
        out: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            t0 = time.time()
            out.extend(self._embed_batch(batch, on_retry=on_retry))
            log_event(log, "embed_batch", size=len(batch), seconds=round(time.time() - t0, 2))
            if progress:
                progress(min(start + self.batch_size, total), total)
        return out

    def embed_query(self, text: str) -> list[float]:
        # A single short query never needs 30 s; a shorter timeout makes a stalled connection cost 12 s, not 30.
        return self._embed_batch([text], timeout=min(self.timeout, 12.0))[0]

    def ping(self) -> tuple[int, float]:
        """One tiny request; returns (embedding dimension, seconds). Raises ExternalServiceError with a hint."""
        t0 = time.time()
        vec = self._embed_batch(["ping"])[0]
        return len(vec), round(time.time() - t0, 2)
