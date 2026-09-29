"""Test doubles: a scripted tool-calling chat model and a deterministic embedder (no network)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ScriptedToolModel(BaseChatModel):
    """Returns pre-scripted AIMessages in order; supports bind_tools and streaming."""

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolModel":
        return self

    def _next(self) -> AIMessage:
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[idx]

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        msg = self._next()
        chunk = AIMessageChunk(
            content=msg.content,
            tool_call_chunks=[
                {"name": tc["name"], "args": __import__("json").dumps(tc["args"]), "id": tc["id"], "index": i}
                for i, tc in enumerate(msg.tool_calls or [])
            ],
        )
        yield ChatGenerationChunk(message=chunk)


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embedder — good enough to exercise Chroma end-to-end."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model = "hash-test"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str], progress=None) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
