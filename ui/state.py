"""Streamlit session-state helpers: chat history, LangChain message history, rate limiting."""

from __future__ import annotations

import time
from collections import deque

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from navigator.config import get_settings

MAX_HISTORY_TURNS = 8  # human+assistant pairs sent back to the model


def init_state() -> None:
    s = get_settings()
    st.session_state.setdefault("messages", [])          # UI messages (dicts)
    st.session_state.setdefault("lc_history", [])        # LangChain messages for the agent
    st.session_state.setdefault("request_times", deque())
    st.session_state.setdefault("language", s.app.default_language)
    st.session_state.setdefault("city", s.app.default_city)
    st.session_state.setdefault("model", s.llm.models[0])


def clear_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["lc_history"] = []


def rate_limited() -> tuple[bool, int]:
    """Sliding-window limiter per browser session. Returns (limited, seconds_until_next)."""
    cfg = get_settings().app.rate_limit
    now = time.time()
    q: deque = st.session_state["request_times"]
    while q and now - q[0] > cfg.window_seconds:
        q.popleft()
    if len(q) >= cfg.max_requests:
        return True, int(cfg.window_seconds - (now - q[0])) + 1
    q.append(now)
    return False, 0


def history_for_agent() -> list[BaseMessage]:
    hist: list[BaseMessage] = st.session_state["lc_history"]
    return hist[-2 * MAX_HISTORY_TURNS :]


def append_turn(user_text: str, assistant_text: str) -> None:
    st.session_state["lc_history"].extend([HumanMessage(content=user_text), AIMessage(content=assistant_text)])


def record_failed_turn(user_text: str, shown_text: str, kind: str, model_text: str | None = None) -> dict:
    """Close a turn that produced no answer (rate limit, refusal, error) so that the rendered chat and the
    model-facing history stay in step: every user bubble gets an assistant entry in BOTH lists.

    `kind` is one of "rate_limited" | "refused" | "error" and controls how the UI renders the entry.
    `model_text` is what the model sees as its own reply in later turns (defaults to `shown_text`).
    """
    msg = {"role": "assistant", "content": shown_text, "kind": kind}
    st.session_state["messages"].append(msg)
    append_turn(user_text, model_text or shown_text)
    return msg
