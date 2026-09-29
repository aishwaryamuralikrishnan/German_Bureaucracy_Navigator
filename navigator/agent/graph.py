"""LangGraph agent (via LangChain's create_agent) plus a streaming runner for the UI.

The runner turns the graph's stream into simple events the Streamlit layer can render:
    ("tool_start", {"name", "args", "id"})
    ("tool_end",   {"name", "id", "result": ToolResult-dict | raw string, "duration_s"})
    ("token",      {"text"})                # answer tokens as they stream
    ("final",      {"text", "tool_calls": [...], "messages": [...]})
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage

from navigator.agent.llm import get_chat_model
from navigator.agent.prompts import build_system_prompt
from navigator.config import get_settings
from navigator.tools import ALL_TOOLS
from navigator.utils.errors import ExternalServiceError
from navigator.utils.logging import get_logger, log_event

log = get_logger(__name__)

Event = tuple[str, dict[str, Any]]


def build_agent(model_name: str | None, language: str, city: str | None):
    s = get_settings().llm
    llm = get_chat_model(model_name, streaming=True)
    return create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=build_system_prompt(language, city, s.max_tool_calls_per_turn),
        name="german-bureaucracy-navigator",
    )


def _parse_tool_result(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def run_agent_stream(
    agent,
    history: list[BaseMessage],
    user_text: str,
    on_event: Callable[[Event], None] | None = None,
) -> dict[str, Any]:
    """Run one turn. Emits events via `on_event` and returns the final payload."""
    s = get_settings().llm
    messages = [*history, HumanMessage(content=user_text)]
    recursion_limit = 2 * s.max_tool_calls_per_turn + 3

    started: dict[str, float] = {}
    args_by_id: dict[str, dict[str, Any]] = {}
    tool_calls: list[dict[str, Any]] = []
    final_text_parts: list[str] = []
    streamed_any = False
    last_ai: AIMessage | None = None
    new_messages: list[BaseMessage] = []

    def emit(kind: str, payload: dict[str, Any]) -> None:
        if on_event:
            on_event((kind, payload))

    try:
        stream: Iterator = agent.stream(
            {"messages": messages},
            stream_mode=["updates", "messages"],
            config={"recursion_limit": recursion_limit},
        )
        for mode, data in stream:
            if mode == "messages":
                chunk, meta = data
                if isinstance(chunk, AIMessageChunk) and chunk.content and not chunk.tool_call_chunks:
                    text = chunk.content if isinstance(chunk.content, str) else "".join(
                        p.get("text", "") for p in chunk.content if isinstance(p, dict)
                    )
                    if text:
                        streamed_any = True
                        final_text_parts.append(text)
                        emit("token", {"text": text})
            elif mode == "updates":
                for _node, update in (data or {}).items():
                    if not isinstance(update, dict):
                        continue
                    for msg in update.get("messages", []) or []:
                        new_messages.append(msg)
                        if isinstance(msg, AIMessage):
                            last_ai = msg
                            for tc in msg.tool_calls or []:
                                started[tc["id"]] = time.time()
                                args_by_id[tc["id"]] = tc.get("args") or {}
                                emit("tool_start", {"name": tc["name"], "args": tc["args"], "id": tc["id"]})
                        elif isinstance(msg, ToolMessage):
                            duration = round(time.time() - started.pop(msg.tool_call_id, time.time()), 2)
                            result = _parse_tool_result(msg.content)
                            record = {"name": msg.name, "id": msg.tool_call_id, "args": args_by_id.get(msg.tool_call_id, {}),
                                      "result": result, "duration_s": duration}
                            tool_calls.append(record)
                            emit("tool_end", record)
                            log_event(log, "tool_call", tool=msg.name, duration_s=duration, ok=(result or {}).get("ok") if isinstance(result, dict) else None)
    except Exception as exc:  # network / provider errors surface here
        raise ExternalServiceError(f"The language model call failed: {exc}") from exc

    # Prefer the last AI message content (complete), fall back to streamed tokens.
    final_text = ""
    if last_ai is not None:
        c = last_ai.content
        final_text = c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict))
    if not final_text and streamed_any:
        final_text = "".join(final_text_parts)
    if not final_text:
        final_text = "I could not produce an answer. Please try rephrasing your question."

    payload = {"text": final_text, "tool_calls": tool_calls, "messages": new_messages}
    emit("final", payload)
    return payload
