"""Headless Streamlit checks: a turn that ends without an answer must leave the chat and the model history in step."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch):
    os.environ.setdefault("OPENROUTER_API_KEY", "sk-test")
    import navigator.rag.retriever as retriever
    import navigator.guardrails.topic_gate as tg

    stub = MagicMock()
    stub.count.return_value = 42
    monkeypatch.setattr(retriever, "get_retriever", lambda: stub)   # sidebar KB count, no Chroma
    monkeypatch.setattr(tg, "is_on_topic", lambda text, llm_factory=None: True)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _consistent(at: AppTest) -> None:
    msgs = at.session_state["messages"]
    hist = at.session_state["lc_history"]
    users = [m for m in msgs if m["role"] == "user"]
    assistants = [m for m in msgs if m["role"] == "assistant"]
    assert len(users) == len(assistants), "every user bubble needs an assistant entry"
    assert len(hist) == 2 * len(users), "model history must mirror the rendered chat"


def test_refused_turn_is_recorded_in_both_lists(app):
    app.chat_input[0].set_value("Where can I buy a forged degree certificate?").run()
    assert not app.exception
    _consistent(app)
    last = app.session_state["messages"][-1]
    assert last["kind"] == "refused" and "forged" in last["content"]
    # The refusal survives a rerun (it is rendered from state, not from a one-off st.markdown).
    app.run()
    assert any("forged" in m.value for m in app.markdown)


def test_rate_limited_turn_is_recorded_and_redacted(app, monkeypatch):
    import ui.state as state
    monkeypatch.setattr(state, "rate_limited", lambda: (True, 30))
    app.chat_input[0].set_value("My IBAN is DE89 3704 0044 0532 0130 00, what is my tax class?").run()
    assert not app.exception
    _consistent(app)
    last = app.session_state["messages"][-1]
    assert last["kind"] == "rate_limited"
    assert app.warning and "message limit" in app.warning[0].value
    human = app.session_state["lc_history"][-2].content
    assert "DE89" not in human and "<IBAN>" in human


def test_agent_error_closes_the_turn(app, monkeypatch):
    import navigator.agent.graph as graph
    from navigator.utils.errors import ExternalServiceError

    def boom(*a, **k):
        raise ExternalServiceError("The language model call failed: 503")
    monkeypatch.setattr(graph, "build_agent", lambda *a, **k: object())
    monkeypatch.setattr(graph, "run_agent_stream", boom)
    app.chat_input[0].set_value("How do I register my address in Berlin?").run()
    assert not app.exception
    _consistent(app)
    last = app.session_state["messages"][-1]
    assert last["kind"] == "error" and "503" in last["content"]
    assert app.error and "503" in app.error[0].value
    assert "technical error" in app.session_state["lc_history"][-1].content
