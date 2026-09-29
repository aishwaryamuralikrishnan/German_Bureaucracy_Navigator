"""Reusable Streamlit rendering: source cards, tool-call traces, notes."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from navigator.tools import TOOL_CATEGORIES
from ui.theme import MUTED, PRIMARY

_JURISDICTION_LABEL = {"federal": "Federal", "berlin": "Berlin", "munich": "Munich", "hamburg": "Hamburg",
                       "frankfurt": "Frankfurt", "cologne": "Cologne", "stuttgart": "Stuttgart"}


def collect_sources(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten SourceRef dicts from every search_knowledge_base result, de-duplicated by ref+title."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for call in tool_calls:
        res = call.get("result")
        if not isinstance(res, dict) or call.get("name") != "search_knowledge_base":
            continue
        for src in res.get("sources", []) or []:
            key = (src.get("ref", ""), src.get("title", ""))
            if key not in seen:
                seen.add(key)
                out.append(src)
    return out


def topics_touched(tool_calls: list[dict[str, Any]]) -> set[str]:
    topics: set[str] = set()
    for src in collect_sources(tool_calls):
        if src.get("topic"):
            topics.add(src["topic"])
    for call in tool_calls:
        if call.get("name") in {"check_blue_card_salary"}:
            topics.add("residence_permits")
        if call.get("name") == "estimate_net_salary":
            topics.add("taxes")
    return topics


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for src in sources:
            juris = _JURISDICTION_LABEL.get(src.get("jurisdiction") or "", src.get("jurisdiction") or "")
            lang = (src.get("language") or "").upper()
            score = src.get("score")
            header = f"**[{src.get('ref')}] {src.get('title')}**"
            if src.get("section"):
                header += f" — {src['section']}"
            if src.get("weak"):
                header += "  ⚠️ *weak match*"
            st.markdown(header)
            sim = src.get("similarity")
            zs = src.get("standout")
            rel = src.get("relevance")
            meta_bits = [
                b for b in (
                    juris, lang,
                    f"relevance {rel:.2f}" if isinstance(rel, (int, float)) else None,
                    f"similarity {sim:.2f}" if isinstance(sim, (int, float)) else None,
                    f"standout z {zs:.1f}" if isinstance(zs, (int, float)) else None,
                    f"fused rank score {score:.3f}" if isinstance(score, (int, float)) else None,
                ) if b
            ]
            if src.get("url"):
                meta_bits.append(f"[open source]({src['url']})")
            st.caption(" · ".join(meta_bits))
            if src.get("snippet"):
                st.markdown(
                    f"<div style='border-left:3px solid {PRIMARY};padding:0.2rem 0.8rem;color:{MUTED};font-size:0.9rem'>{_escape(src['snippet'][:700])}{'…' if len(src['snippet']) > 700 else ''}</div>",
                    unsafe_allow_html=True,
                )
            st.divider()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def _render_currency_result(data: dict[str, Any]) -> None:
    src, dst, rate = data.get("from", {}), data.get("to", {}), data.get("rate", {})
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{src.get('currency', '')} ({src.get('name', '')})", f"{src.get('amount', 0):,.2f}")
    c2.metric(f"{dst.get('currency', '')} ({dst.get('name', '')})", f"{dst.get('amount', 0):,.2f}")
    c3.metric(f"1 {src.get('currency', '')} =", f"{rate.get('value', 0):,.4f} {dst.get('currency', '')}", help=f"Inverse: 1 {dst.get('currency','')} = {rate.get('inverse', 0):,.4f} {src.get('currency','')}")
    eq = data.get("equivalents") or {}
    if eq:
        bits = [f"{k.replace('_', ' ')}: {v:,.2f}" for k, v in eq.items()]
        basis = data.get("salary_basis")
        if basis:
            bits.append(f"salary stated as: **{basis}**" if basis != "unknown" else "gross or net: **not stated by the user**")
        st.caption(" · ".join(bits))
    st.caption(f"Rate date {rate.get('date')} · {rate.get('source', '')}")
    st.caption("Nominal exchange-rate equivalent — not a cost-of-living or purchasing-power comparison; no taxes applied.")


def _render_salary_result(data: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Gross / month", f"€ {data.get('gross_monthly_eur', 0):,.0f}")
    c2.metric("Net / month (est.)", f"€ {data.get('net_monthly_eur', 0):,.0f}")
    c3.metric("Net / year (est.)", f"€ {data.get('net_annual_eur', 0):,.0f}")
    bd = data.get("breakdown_annual_eur") or {}
    if bd:
        df = pd.DataFrame({"deduction": list(bd.keys()), "EUR / year": list(bd.values())})
        st.dataframe(df, hide_index=True, width="stretch")


def _render_deadline_result(data: dict[str, Any]) -> None:
    urgency = data.get("urgency")
    icon = {"ok": "🟢", "soon": "🟠", "overdue": "🔴"}.get(urgency, "⚪")
    st.markdown(f"{icon} **{data.get('what')}** — deadline **{data.get('deadline')}** ({data.get('days_remaining')} days)")
    st.caption(f"Legal basis: {data.get('legal_basis')}")


def _render_blue_card_result(data: dict[str, Any]) -> None:
    g = data.get("general_threshold") or {}
    r = data.get("reduced_threshold") or {}
    st.markdown(f"Salary **€ {data.get('salary_eur', 0):,.0f}** ({data.get('year')})")
    c1, c2 = st.columns(2)
    c1.metric("General threshold", f"€ {g.get('eur', 0):,.2f}", delta=f"{g.get('gap_eur', 0):+,.2f}", delta_color="normal")
    c2.metric("Reduced threshold (shortage / recent graduate / IT)", f"€ {r.get('eur', 0):,.2f}", delta=f"{r.get('gap_eur', 0):+,.2f}", delta_color="normal")
    st.markdown(f"{'✅' if g.get('met') else '❌'} general · {'✅' if r.get('met') else '❌'} reduced — {data.get('verdict', '')}")
    st.caption(data.get("legal_basis", ""))


def render_tool_calls(tool_calls: list[dict[str, Any]]) -> None:
    if not tool_calls:
        return
    with st.expander(f"🛠️ Tool calls ({len(tool_calls)})", expanded=False):
        for i, call in enumerate(tool_calls, 1):
            name = call.get("name", "tool")
            category = TOOL_CATEGORIES.get(name, "tool")
            res = call.get("result")
            ok = res.get("ok", True) if isinstance(res, dict) else True
            st.markdown(f"**{i}. `{name}`** · {category} · {call.get('duration_s', 0)} s · {'✅ ok' if ok else '❌ failed'}")
            if call.get("args"):
                with st.popover("arguments"):
                    st.json(call["args"])
            if not isinstance(res, dict):
                st.code(str(res)[:2000])
            elif not ok:
                st.error(res.get("error") or "Tool failed")
            else:
                data = res.get("data", {})
                if name == "convert_currency":
                    _render_currency_result(data)
                elif name == "estimate_net_salary":
                    _render_salary_result(data)
                elif name == "calculate_deadline":
                    _render_deadline_result(data)
                elif name == "check_blue_card_salary":
                    _render_blue_card_result(data)
                elif name == "search_knowledge_base":
                    ret = data.get("retrieval", {})
                    cov = data.get("coverage", "ok")
                    st.caption(
                        f"coverage **{cov}** · {len(data.get('passages', []))} passages · variants: {ret.get('query_variants')} · "
                        f"BM25 hits {ret.get('bm25_hits')} · dense hits {ret.get('dense_hits')} · fused {ret.get('fused_candidates')} · "
                        f"max similarity {ret.get('max_similarity')} · max standout z {ret.get('max_standout')} (weak below {ret.get('weak_threshold')})"
                        + (f" · max reranker relevance {ret.get('max_relevance')} (weak below {ret.get('min_relevance')})" if ret.get('max_relevance') is not None else "")
                        + f" · reranker {ret.get('reranker')}"
                    )
                with st.popover("raw result"):
                    st.json(res)
            for w in res.get("warnings", []) if isinstance(res, dict) else []:
                if any(k in w.lower() for k in ("unavailable", "failed", "curated", "low coverage", "no passages")):
                    st.warning(w, icon="⚠️")
                else:
                    st.caption(f"⚠️ {w}")
            if i < len(tool_calls):
                st.divider()


def weak_coverage(tool_calls: list[dict[str, Any]]) -> bool:
    """True if every knowledge-base search in this turn came back with weak/no coverage."""
    searches = [c for c in tool_calls if c.get("name") == "search_knowledge_base" and isinstance(c.get("result"), dict)]
    if not searches:
        return False
    return all((c["result"].get("data") or {}).get("coverage") in {"weak", "none"} for c in searches)


def valid_refs(tool_calls: list[dict[str, Any]]) -> set[str]:
    return {src.get("ref") for src in collect_sources(tool_calls) if src.get("ref")}


def render_assistant_message(msg: dict[str, Any]) -> None:
    kind = msg.get("kind")
    if kind == "error":
        st.error(msg.get("content", ""), icon="⚠️")
        return
    if kind == "rate_limited":
        st.warning(msg.get("content", ""), icon="⏳")
        return
    st.markdown(msg.get("content", ""))
    for note in msg.get("notes", []) or []:
        if "no specific passage" in note or "keine spezifische" in note:
            st.warning(note, icon="📭")
        elif note.startswith("Not backed by") or note.startswith("Nicht durch"):
            st.warning(note, icon="🔎")
        elif note.startswith("The answer was revised") or note.startswith("Die Antwort wurde überarbeitet"):
            st.warning(note, icon="✂️")
        else:
            st.info(note, icon="ℹ️")
    g = msg.get("grounding")
    if g and g.get("checked") and not g.get("unsupported"):
        st.caption("🔎 Grounding check passed: every concrete claim is backed by a retrieved passage or tool result.")
    if msg.get("original_answer"):
        with st.expander("Original answer before the grounding revision"):
            st.markdown(msg["original_answer"])
    calls = msg.get("tool_calls", []) or []
    searches = [c for c in calls if c.get("name") == "search_knowledge_base"]
    if calls and not searches:
        st.caption("ℹ️ No knowledge-base search was performed for this answer, so there are no sources to show.")
    elif searches and not (msg.get("sources") or []):
        failed = [c for c in searches if not (isinstance(c.get("result"), dict) and c["result"].get("ok", True))]
        if failed:
            st.caption("⚠️ The knowledge-base search failed — see the Tool calls panel for the error.")
        else:
            st.caption("ℹ️ The knowledge-base search returned no passages.")
    if msg.get("redactions"):
        st.caption(f"🔒 Personal data redacted from your message before processing: {', '.join(sorted(set(msg['redactions'])))}")
    render_sources(msg.get("sources", []) or [])
    render_tool_calls(msg.get("tool_calls", []) or [])
