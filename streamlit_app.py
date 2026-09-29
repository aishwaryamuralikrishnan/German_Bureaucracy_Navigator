"""German Bureaucracy Navigator — Streamlit entry point.

Run:  streamlit run streamlit_app.py   (or right-click → "Run with Streamlit" in VS Code)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import streamlit as st

# Make the `navigator` and `ui` packages importable regardless of the working directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigator.agent.graph import build_agent, run_agent_stream  # noqa: E402
from navigator.config import get_api_key, get_settings  # noqa: E402
from navigator.guardrails.grounding import apply_grounding  # noqa: E402
from navigator.guardrails.output_checks import check_output  # noqa: E402
from navigator.guardrails.pii import redact_pii  # noqa: E402
from navigator.guardrails.pipeline import run_input_guardrails  # noqa: E402
from navigator.rag.ingest import ingest  # noqa: E402
from navigator.rag.retriever import get_retriever  # noqa: E402
from navigator.utils.errors import ConfigError, ExternalServiceError, KnowledgeBaseError, NavigatorError  # noqa: E402
from navigator.utils.logging import get_logger, log_event  # noqa: E402
from ui.components import collect_sources, render_assistant_message, topics_touched, valid_refs, weak_coverage  # noqa: E402
from ui.state import append_turn, clear_conversation, history_for_agent, init_state, rate_limited, record_failed_turn  # noqa: E402

SETTINGS = get_settings()
LOG = get_logger("navigator.app")


@st.cache_resource
def ingest_lock() -> threading.Lock:
    """One lock per server process, shared by all sessions: only one knowledge-base build may run at a time."""
    return threading.Lock()

st.set_page_config(page_title=SETTINGS.app.title, page_icon="🇩🇪", layout="wide")
init_state()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🇩🇪 Navigator")
    st.caption("RAG + tool-calling assistant for newcomers to Germany")

    lang_label = st.radio(
        "Answer language",
        ["Auto-detect", "English", "Deutsch"],
        index={"auto": 0, "en": 1, "de": 2}.get(st.session_state["language"], 0),
        horizontal=True,
    )
    st.session_state["language"] = {"Auto-detect": "auto", "English": "en", "Deutsch": "de"}[lang_label]

    NOT_SET = "(not set)"
    city_options = [NOT_SET] + list(SETTINGS.app.cities)
    current_city = st.session_state["city"] or NOT_SET
    if current_city not in city_options:
        city_options.append(current_city)
    chosen = st.selectbox(
        "Your city", city_options, index=city_options.index(current_city),
        accept_new_options=True,
        help="Optional. Type any German city. Used to prioritise city-specific sources, regional public holidays. "
             "A city you mention in your message always takes precedence.",
    )
    st.session_state["city"] = "" if chosen in (NOT_SET, None) else chosen
    st.session_state["model"] = st.selectbox(
        "Model (OpenRouter)", SETTINGS.llm.models,
        index=SETTINGS.llm.models.index(st.session_state["model"]) if st.session_state["model"] in SETTINGS.llm.models else 0,
    )

    st.divider()
    key_ok = bool(get_api_key())
    st.markdown(f"**API key:** {'✅ found' if key_ok else '❌ missing'}  ")
    kb_count = None
    if key_ok:
        try:
            kb_count = get_retriever().count()
        except NavigatorError as exc:
            st.caption(f"Knowledge base: {exc.user_message}")
    if kb_count is not None:
        st.markdown(f"**Knowledge base:** {kb_count} chunks indexed")

    if st.button("🗑️ Clear conversation", width="stretch"):
        clear_conversation()
        st.rerun()

    st.divider()
    with st.expander("🔒 Privacy & limits"):
        st.markdown(
            "- Your messages are sent to the selected model via **OpenRouter**.\n"
            "- Tax IDs, IBANs, passport numbers, phone numbers and emails are **redacted** before sending.\n"
            "- Currency conversions call the free **Frankfurter** API (ECB rates); only currency codes and amounts are sent.\n"
            "- This is general information, **not legal or tax advice**.\n"
            f"- Rate limit: {SETTINGS.app.rate_limit.max_requests} messages per {SETTINGS.app.rate_limit.window_seconds // 60} minutes."
        )
    st.caption("Tools: knowledge-base search · Blue Card check · deadline calculator · net-salary estimate · currency conversion")

# ----------------------------------------------------------------------------- header
st.title(SETTINGS.app.title)
st.caption(
    "Ask about Anmeldung, residence permits & the EU Blue Card, student visas, tax ID & tax classes, health insurance, "
    "bank accounts, driving licences, recognition of qualifications, family reunification, deadlines and salaries."
)

# ----------------------------------------------------------------------------- preflight
if not key_ok:
    st.error(
        f"Environment variable **{SETTINGS.llm.api_key_env}** is not set. Create a key at https://openrouter.ai/keys, "
        "set it as a system environment variable (restart VS Code afterwards) or put it in a `.env` file next to "
        "`streamlit_app.py`, then reload this page.",
        icon="🔑",
    )
    st.stop()

if kb_count == 0:
    # The Chroma index is not in the repository and a hosted container starts with an empty disk, so the first
    # visitor after a (re)start builds it: 103 chunks, about a minute, a few cents of embeddings. One build at a
    # time — a second visitor arriving mid-build waits for the lock and then finds the index filled.
    st.info("Preparing the knowledge base — this happens once after a restart and takes about a minute.", icon="📭")
    with ingest_lock():
        if get_retriever().count() == 0:
            bar = st.progress(0, text="Starting…")
            status = st.empty()

            def _progress(stage: str, done: int, total: int, msg: str) -> None:
                base = {"load": (0.0, 0.03), "chunk": (0.03, 0.06), "ping": (0.06, 0.10), "embed": (0.10, 0.85), "write": (0.85, 1.0)}
                lo, hi = base.get(stage, (0.1, 0.85))
                frac = lo + (hi - lo) * (done / max(total, 1))
                bar.progress(min(max(frac, 0.0), 1.0), text=msg)
                status.caption(f"{stage}: {msg}")

            try:
                report = ingest(progress=_progress)
                bar.progress(1.0, text="Done")
                st.success(f"Indexed {report.indexed} chunks from {report.documents} documents.")
            except (NavigatorError, RuntimeError) as exc:
                st.error(f"The knowledge base could not be built: {exc}")
                st.stop()
    st.rerun()

# ----------------------------------------------------------------------------- examples
# Always create this container, even when it stays empty. Streamlit addresses elements by their position, and
# during a rerun it keeps the previous run's elements on screen until the new run overwrites them. When the
# example block (2 elements) disappeared after the first answer, the second question's assistant bubble landed on
# the exact position of the first answer's bubble — so while the model was thinking, the first answer's text and
# source cards were still showing inside the new bubble, below the status box. A constant single block here keeps
# every new bubble at a position no earlier run has used.
examples_box = st.container()
if not st.session_state["messages"]:
    with examples_box:
        st.markdown("**Try one of these:**")
        examples = [
            "I moved into my flat on 1 September. When is my Anmeldung deadline and what do I need to bring?",
            "I have a job offer of €49,000 as a software engineer starting January 2026. Do I qualify for the Blue Card?",
            "Was ist der Unterschied zwischen Steuerklasse 1 und 4?",
            "What is my net salary on €65,000 gross in Bayern, tax class 1, no church?",
            "My current gross salary is 18 lakh INR per year. Is a €52,000 offer in Germany enough for the Blue Card?",
        ]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, width="stretch"):
                st.session_state["queued_prompt"] = ex
                st.rerun()

# ----------------------------------------------------------------------------- history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🧭" if msg["role"] == "assistant" else None):
        if msg["role"] == "assistant":
            render_assistant_message(msg)
        else:
            st.markdown(msg["content"])

# ----------------------------------------------------------------------------- input
prompt = st.chat_input("Ask in English or German…")
if not prompt and st.session_state.get("queued_prompt"):
    prompt = st.session_state.pop("queued_prompt")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    limited, wait_s = rate_limited()
    forced_lang = None if st.session_state["language"] == "auto" else st.session_state["language"]

    with st.chat_message("assistant", avatar="🧭"):
        # Every failure path below closes the turn with record_failed_turn(), so the rendered chat and the
        # model-facing history never drift apart (a user bubble without a matching assistant entry).
        if limited:
            text = f"You've reached the message limit. Please wait about {wait_s} seconds."
            msg = record_failed_turn(
                redact_pii(prompt).text, text, "rate_limited",
                model_text="(This message was not processed because the rate limit was reached.)",
            )
            render_assistant_message(msg)
            st.stop()

        outcome = run_input_guardrails(prompt, forced_language=forced_lang)
        if not outcome.allowed:
            msg = record_failed_turn(outcome.text or redact_pii(prompt).text, outcome.user_message, "refused")
            msg["redactions"] = outcome.redactions
            render_assistant_message(msg)
            st.stop()

        answer_box = st.empty()
        tokens: list[str] = []
        live_tool_calls: list[dict] = []

        with st.status("Thinking…", expanded=True) as status:
            if outcome.redactions:
                status.write(f"🔒 Redacted personal data: {', '.join(sorted(set(outcome.redactions)))}")

            def on_event(event):
                kind, payload = event
                if kind == "tool_start":
                    status.update(label=f"Running `{payload['name']}`…")
                    args_preview = ", ".join(f"{k}={str(v)[:40]}" for k, v in (payload.get("args") or {}).items())
                    status.write(f"🛠️ **{payload['name']}**({args_preview})")
                elif kind == "tool_end":
                    res = payload.get("result")
                    ok = res.get("ok", True) if isinstance(res, dict) else True
                    summary = ""
                    if isinstance(res, dict) and res.get("data"):
                        d = res["data"]
                        if payload["name"] == "search_knowledge_base":
                            summary = f"{len(d.get('passages', []))} passages retrieved (coverage {d.get('coverage', 'ok')})"
                        elif payload["name"] == "convert_currency":
                            summary = f"{d.get('from', {}).get('amount', 0):,.0f} {d.get('from', {}).get('currency', '')} ≈ {d.get('to', {}).get('amount', 0):,.0f} {d.get('to', {}).get('currency', '')}"
                        elif payload["name"] == "calculate_deadline":
                            summary = f"deadline {d.get('deadline')}"
                        elif payload["name"] == "estimate_net_salary":
                            summary = f"net ≈ € {d.get('net_monthly_eur', 0):,.0f}/month"
                        elif payload["name"] == "check_blue_card_salary":
                            g, r = d.get("general_threshold", {}), d.get("reduced_threshold", {})
                            summary = f"general {'✅' if g.get('met') else '❌'} · reduced {'✅' if r.get('met') else '❌'}"
                    status.write(f"{'✅' if ok else '❌'} `{payload['name']}` finished in {payload.get('duration_s')} s — {summary or (res.get('error') if isinstance(res, dict) else '')}")
                    live_tool_calls.append(payload)
                    status.update(label="Composing answer…")
                elif kind == "token":
                    tokens.append(payload["text"])
                    answer_box.markdown("".join(tokens) + "▌")

            def _fail(label: str, shown: str) -> None:
                status.update(label=label, state="error", expanded=False)
                answer_box.empty()
                msg = record_failed_turn(
                    outcome.text, shown, "error",
                    model_text="(No answer was produced for this message because of a technical error; the user may retry.)",
                )
                msg["redactions"] = outcome.redactions
                render_assistant_message(msg)
                st.stop()

            try:
                agent = build_agent(st.session_state["model"], outcome.language, st.session_state["city"] or None)
                result = run_agent_stream(agent, history_for_agent(), outcome.text, on_event=on_event)
            except ConfigError as exc:
                _fail("Configuration error", str(exc))
            except (ExternalServiceError, KnowledgeBaseError) as exc:
                _fail("Failed", f"{exc.user_message}\n\nDetails: {exc}")
            except Exception as exc:  # last resort: never leave a half-finished turn on screen
                LOG.exception("unexpected error during turn")
                _fail("Failed", f"Something went wrong while answering. Please try again.\n\nDetails: {type(exc).__name__}: {exc}")

            tool_calls = result["tool_calls"]
            used_search = any(tc["name"] == "search_knowledge_base" for tc in tool_calls)
            is_weak = weak_coverage(tool_calls)
            checked = check_output(
                result["text"],
                used_search,
                topics_touched(tool_calls),
                outcome.language,
                valid_refs=valid_refs(tool_calls) if used_search else None,
                weak_coverage=is_weak,
            )

            # Second opinion: does every concrete claim trace back to a passage or a tool result? If not, the answer is
            # revised so unsupported claims are dropped or labelled before the user sees it (guardrails.grounding_check.revise).
            grounding = None
            original_answer = None
            if tool_calls:  # also when coverage is weak: "general orientation" must not smuggle in numbers
                status.update(label="Checking the answer against its sources…")
                outcome_g = apply_grounding(checked.text, tool_calls, outcome.language)
                grounding = outcome_g.report
                if grounding.checked:
                    if not grounding.unsupported:
                        status.write("🔎 Grounding check: all concrete claims are backed by the retrieved passages or tool results.")
                    elif outcome_g.revised:
                        status.write(f"✂️ Grounding check: {len(grounding.unsupported)} unsupported claim(s) removed or labelled — answer revised.")
                        original_answer = outcome_g.original_text
                        checked.text = outcome_g.text
                    else:
                        status.write(f"🔎 Grounding check: {len(grounding.unsupported)} claim(s) not backed by the sources — flagged below.")
                    if grounding.ignored:
                        status.write(f"🛡️ Grounding guard: {len(grounding.ignored)} flagged claim(s) kept — their figures appear in the retrieved evidence.")
                    if outcome_g.revision_rejected:
                        status.write("🛡️ Grounding guard: the revision would have removed a figure the evidence supports — original answer kept, claims flagged below.")
                    if outcome_g.note:
                        checked.notes.append(outcome_g.note)
            status.update(label="Done", state="complete", expanded=False)

        answer_box.empty()

        assistant_msg = {
            "role": "assistant",
            "content": checked.text,
            "tool_calls": tool_calls,
            "sources": collect_sources(tool_calls),
            "notes": checked.notes,
            "redactions": outcome.redactions,
            "grounding": {"checked": grounding.checked, "unsupported": grounding.unsupported, "revised": original_answer is not None} if grounding else None,
            "original_answer": original_answer,
        }
        render_assistant_message(assistant_msg)
        st.session_state["messages"].append(assistant_msg)
        append_turn(outcome.text, checked.text)
        log_event(
            LOG, "turn_done",
            tools=[{"name": tc["name"], "ok": (tc.get("result") or {}).get("ok") if isinstance(tc.get("result"), dict) else "raw"} for tc in tool_calls],
            sources=len(assistant_msg["sources"]),
            weak_coverage=checked.weak_coverage,
            citations=checked.citations_found,
            grounding_unsupported=len(grounding.unsupported) if grounding and grounding.checked else None,
            revised=original_answer is not None,
        )
