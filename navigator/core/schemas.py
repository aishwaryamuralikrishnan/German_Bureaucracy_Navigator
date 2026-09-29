"""Shared Pydantic models. Every tool returns a ToolResult serialised as JSON."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """A citation the UI can render as a source card."""

    ref: str = Field(description="Citation label used in the answer, e.g. 'S1'")
    title: str
    url: str | None = None
    section: str | None = None
    snippet: str | None = None
    score: float | None = None
    jurisdiction: str | None = None
    language: str | None = None
    topic: str | None = None
    similarity: float | None = None   # dense cosine similarity to the query (None if BM25-only hit)
    standout: float | None = None     # robust z-score vs. the whole corpus for this query
    relevance: float | None = None    # cross-encoder (reranker) relevance in [0, 1]; None without a reranker
    weak: bool = False                # does not stand out from the corpus (or below an absolute override)


class ToolResult(BaseModel):
    """Uniform envelope for all tool outputs."""

    ok: bool = True
    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    sources: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, default=str)

    @classmethod
    def failure(cls, tool: str, error: str, **data: Any) -> "ToolResult":
        return cls(ok=False, tool=tool, error=error, data=data)


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    title: str
    file: str | None = None        # source markdown file name (data/processed/*.md)
    source_url: str | None = None
    section: str | None = None
    topic: str | None = None
    jurisdiction: str | None = None
    language: str | None = None
    score: float = 0.0
    similarity: float | None = None
    standout: float | None = None
    relevance: float | None = None    # cross-encoder (reranker) relevance in [0, 1]; None without a reranker
    weak: bool = False
