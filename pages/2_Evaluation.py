"""Evaluation page: browse the runs written by scripts/evaluate.py (retrieval metrics, faithfulness, tool selection,
not-covered handling) and drill into individual questions. Runs are started from the terminal, not from here."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigator.eval import store  # noqa: E402
from navigator.eval.cost import cost_label  # noqa: E402
from navigator.eval.report import CONFIG_COLOURS, TOOL_SHORT, build_pdf, claim_kind, claim_verdict, faithfulness_headline, group_tool_stats, tool_matrix  # noqa: E402
from navigator.eval.retrieval_eval import CONFIGS  # noqa: E402
from ui.theme import BORDER, MUTED, SURFACE  # noqa: E402

st.set_page_config(page_title="Evaluation", page_icon="📏", layout="wide")
BLUE, PROBE_BLUE, GOOD, CRITICAL, FORBIDDEN, WARNING = "#2a78d6", "#1a4f8f", "#0ca30c", "#d03b3b", "#7a3fbf", "#fab219"


def table(rows) -> None:
    """The page's one table style: a static table whose cells wrap (long questions, claims and reasons stay fully
    visible) with every cell centred. st.dataframe would truncate long text to one line."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        st.caption("— nothing to show —")
        return
    st.table(df.astype(str).style.set_properties(**{"text-align": "center", "vertical-align": "middle"}).hide(axis="index"))


def fmt_seconds(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "?"
    m, sec = divmod(int(round(x)), 60)
    return f"{m} min {sec:02d} s" if m else f"{sec} s"

runs = store.list_runs()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📏 Evaluation")
    if not runs:
        st.info("No evaluation runs yet.")
    else:
        run = st.selectbox("Run", runs, format_func=lambda r: r.label)
        others = [r for r in runs if r.path != run.path]
        compare = st.selectbox("Compare with (optional)", [None] + others, format_func=lambda r: "—" if r is None else r.label)
        groups = ["knowledge", "mixed", "tool_only", "not_covered"]
        shown_groups = st.multiselect("Show groups", groups, default=groups)
        pdf = run.pdf_path
        if not pdf.exists() and (run.retrieval or run.e2e):
            if st.button("Build PDF report"):
                with st.spinner("Rendering…"):
                    build_pdf(run.path, run.meta, run.retrieval, run.e2e)
                st.rerun()
        if pdf.exists():
            st.download_button("⬇ Download this run as PDF", data=pdf.read_bytes(), file_name=f"evaluation_{run.name}.pdf", mime="application/pdf", type="primary", width="stretch")
    st.caption("Runs are read from `reports/eval/`. Start one from a terminal:\n\n`python scripts\\evaluate.py`\n\nSee README § 12.")

st.title("📏 Evaluation")
# centre the header row of the static tables too (the Styler above handles the cells)
st.markdown("<style>[data-testid='stTable'] th, [data-testid='stTable'] td {text-align:center !important; vertical-align:middle !important}</style>", unsafe_allow_html=True)
if not runs:
    st.markdown(
        "No evaluation runs found in `reports/eval/`.\n\n"
        "Stop the app, then run in a terminal from the project folder:\n\n"
        "```powershell\npython scripts\\evaluate.py\n```\n\n"
        "The run takes about 15 minutes for the default 23-question set and writes the results this page displays."
    )
    st.stop()

retrieval, e2e, meta = run.retrieval, run.e2e, run.meta


def _results(r):
    return [x for x in (r.e2e or {}).get("results", []) if x["group"] in shown_groups]


def _prod(r):
    if not r or not r.retrieval:
        return None
    return next((c for c in r.retrieval["configs"] if c["config"] == "hybrid_rerank"), r.retrieval["configs"][-1])


def _delta(cur, prev, fmt):
    if cur is None or prev is None:
        return None
    d = cur - prev
    return ("+" if d >= 0 else "") + fmt(d)


bits = [f"{meta.get('questions', '?')} questions"]
if retrieval:
    bits.append(f"retrieval {fmt_seconds(retrieval.get('seconds'))}")
if e2e:
    bits += [f"end-to-end {fmt_seconds(e2e.get('seconds'))}", f"agent {e2e.get('model')}", f"judge {e2e.get('judge_model')} ({e2e.get('judge_backend')})"]
bits.append(cost_label(meta, e2e))
st.caption(" · ".join(b for b in bits if b))

# ----------------------------------------------------------------------------- tiles
prod, prod_prev = _prod(run), _prod(compare)
s, s_prev = (e2e or {}).get("summary"), ((compare.e2e if compare else None) or {}).get("summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"Hit-rate@{retrieval['k'] if retrieval else 5} · " + (prod["label"] if prod else "—"), f"{100 * prod['hit_rate']:.0f}%" if prod else "—",
          _delta(prod and prod["hit_rate"], prod_prev and prod_prev["hit_rate"], lambda d: f"{100 * d:.0f} pts"))
c2.metric("MRR", f"{prod['mrr']:.2f}" if prod else "—", _delta(prod and prod["mrr"], prod_prev and prod_prev["mrr"], lambda d: f"{d:.2f}"))
fm = s["faithfulness"]["mean"] if s else None
c3.metric("Mean faithfulness", f"{fm:.2f}" if fm is not None else "—", _delta(fm, s_prev and s_prev["faithfulness"]["mean"], lambda d: f"{d:.2f}"),
          help=(f"probes {s['faithfulness']['probes']} · others {s['faithfulness']['others']}" if s else None))
ts = s["tool_selection"] if s else None
c4.metric("Tool selection correct", f"{ts['correct']} / {ts['total']}" if ts else "—",
          _delta(ts and ts["correct"] / max(ts["total"], 1), s_prev and s_prev["tool_selection"]["correct"] / max(s_prev["tool_selection"]["total"], 1), lambda d: f"{100 * d:.0f} pts"))
nc = s["not_covered"] if s else None
c5.metric("Not-covered questions handled", f"{nc['handled']} / {nc['total']}" if nc else "—",
          help=(f"answers revised by the grounding step: {s.get('revised', 0)} of {s['questions']}" if s else None))
if s and s.get("tool_errors"):
    st.warning(f"{s['tool_errors']} question(s) ended with a failed tool call even after one automatic re-run — the agent answered without that tool's result. "
               "Open the question below to see the error; a not-covered question with a failed search counts as not handled.", icon="⚠️")

# ----------------------------------------------------------------------------- 1 retrieval
st.header("1 · Retrieval evaluation — hit-rate and MRR for four configurations")
if not retrieval:
    st.info("This run has no retrieval half (`--only e2e`).")
else:
    df = pd.DataFrame(retrieval["configs"])
    df["config_label"] = df["config"].map(CONFIGS)
    order = list(CONFIGS.values())
    colour = alt.Color("config_label:N", scale=alt.Scale(domain=order, range=[CONFIG_COLOURS[c] for c in CONFIGS]), legend=alt.Legend(title=None, orient="top"))

    def bar(field, title, fmt):
        base = alt.Chart(df).encode(x=alt.X("config_label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{field}:Q", scale=alt.Scale(domain=[0, 1.1]), title=None), color=colour)
        return (base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=48) + base.mark_text(dy=-8, fontSize=12).encode(text=alt.Text(f"{field}:Q", format=fmt), color=alt.value("#0b0b0b"))).properties(title=title, height=300)

    left, right = st.columns(2)
    left.altair_chart(bar("hit_rate", f"Hit-rate@{retrieval['k']} — expected document in the top {retrieval['k']}", ".0%"), width="stretch", height=300)
    right.altair_chart(bar("mrr", "Mean reciprocal rank — how high the hit sits", ".2f"), width="stretch", height=300)
    if compare and compare.retrieval:
        prev = {c["config"]: c for c in compare.retrieval["configs"]}
        st.caption("Compared run: " + " · ".join(f"{CONFIGS[c['config']]} {100 * prev[c['config']]['hit_rate']:.0f}% / {prev[c['config']]['mrr']:.2f}" for c in retrieval["configs"] if c["config"] in prev))
    _n, _k = len(retrieval["per_question"]), retrieval["k"]
    _skipped = int(meta.get("questions", _n)) - _n
    with st.expander("How to read this"):
        st.markdown(
            f"Retrieval was tested on the **{_n} questions that have a known answering document**"
            + (f" (the {_skipped} not-covered question{'s' if _skipped != 1 else ''} have none, so they are left out here)." if _skipped > 0 else ".")
            + f" All four configurations were given **the same search queries and the same embeddings** (`{retrieval.get('embedding_model')}`); the only thing that "
            "differs is **how the passages are found**: by meaning (*dense*), by keywords (*BM25*), by both combined (*hybrid*), or by both plus a reranker "
            f"(`{retrieval.get('reranker')}`) that reads question and passage together and re-orders the top candidates.\n\n"
            f"**Hit-rate@{_k}** = share of questions where the expected document appears somewhere in the top {_k}. "
            "**MRR** (mean reciprocal rank) rewards a high position: rank 1 scores 1.0, rank 2 scores 0.5, rank 4 scores 0.25, not found scores 0 — averaged over the questions."
        )
    with st.expander(f"Per-question retrieval ranks ({_n} questions)"):
        st.markdown(f"One row per question; each cell is the **position at which the expected document first appeared** in that configuration's top {_k}. "
                    f"**1** = it was the top result. **—** = not in the top {_k} at all (counts 0 for MRR). Where several documents are expected, the best-placed one counts.")
        rows = [{"ID": q["id"], "group": q["group"], "expected document(s)": ", ".join(d.replace(".md", "") for d in q["expected_docs"]),
                 # ranks as strings throughout: a column mixing ints and "—" is not Arrow-serialisable and makes Streamlit log a warning
                 **{CONFIGS[c]: (str(q["ranks"][c]) if q["ranks"].get(c) else "—") for c in CONFIGS}} for q in retrieval["per_question"] if q["group"] in shown_groups]
        table(rows)

if not e2e:
    st.info("This run has no end-to-end half (`--only retrieval`). Faithfulness, tool selection and not-covered handling need `python scripts\\evaluate.py --only e2e --run reports/eval/" + run.name + "`.")
    st.stop()

results = _results(run)

# ----------------------------------------------------------------------------- details of one question (shown under the faithfulness chart after a click)
def render_details(r: dict) -> None:
    f = r.get("faithfulness") or {}
    t = r.get("tools") or {}
    head = f"{r['id']} · {r['group']}" + (" · hard" if r.get("hard") else "") + (" · probe" if r.get("probe") else "") + (" · ✂ revised by the grounding step" if r.get("answer_original") else "")
    st.markdown("---")
    st.subheader(f"Evaluation details — {head}")
    st.markdown(f"**Question:** {r['question']}")
    if r.get("error"):
        st.error(f"Agent error: {r['error']}")
    for _e in r.get("tool_errors") or []:
        st.error(f"Tool failed: {_e} — the agent answered without this tool's result" + (" (already the second attempt)" if r.get("retried") else ""))
    m1, m2 = st.columns(2)
    m1.metric("Faithfulness", f"{f['score']:.2f}" if f.get("score") is not None else "n/a", help=faithfulness_headline(f) if f.get("claims") else None)
    m2.metric("Tool selection", "✔ correct" if t.get("passed") else "✖ failed", help="; ".join(t.get("reasons") or []) or None)

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**Answer as the user would see it**")
        st.markdown(f"<div style='border:1px solid {BORDER};border-radius:8px;padding:12px 14px;font-size:14px;background:{SURFACE}'>{r.get('answer', '').replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        for n in r.get("notes") or []:
            st.caption(f"ℹ️ {n}")
        if r.get("answer_original"):
            with st.expander("Answer before the grounding revision (what the agent originally wrote)"):
                st.markdown(f"<div style='border:1px solid {BORDER};border-radius:8px;padding:12px 14px;font-size:13px;color:{MUTED};background:{SURFACE}'>{r['answer_original'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        st.markdown("**Tool selection check**")
        exp = r.get("expected", {})
        trows = []
        for tc in r.get("tool_calls") or []:
            name = tc.get("name")
            status = "required, called" if name in (exp.get("tools") or []) else ("✖ forbidden" if name in (exp.get("forbidden") or []) else "called (not required)")
            bad = [m for m in (t.get("arg_mismatches") or []) if m.startswith(name + ".")]
            if bad:
                status = "⚠ " + "; ".join(bad)
            if tc.get("error"):
                status = f"✖ tool FAILED: {tc['error']}"
            trows.append({"tool": name, "arguments": ", ".join(f"{k}={v}" for k, v in (tc.get("args") or {}).items()) or "—", "check": status})
        for miss in t.get("missing") or []:
            trows.append({"tool": miss, "arguments": "— not called — expected " + ", ".join(f"{k}={v}" for k, v in (exp.get("args", {}).get(miss) or {}).items()), "check": "✖ missing"})
        table(trows)
    with rcol:
        st.markdown("**Faithfulness measure for this answer**")
        if f.get("claims"):
            _hl = faithfulness_headline(f).replace("Faithfulness: ", "")
            st.info(f"**Score {f['score']:.2f} — {_hl}.** The judge split the answer into statements and checked each one against the sources listed below — "
                    "and only against those. A factual claim is *unsupported* when none of them says it, whether or not it is true in the real world. Statements that only "
                    "refer you to an authority or website (*referral*) or say what the knowledge base does not cover (*meta*) are shown but not scored.")
            table([{"No.": i, "statement in the answer": c["statement"], "kind": claim_kind(c), "verdict": claim_verdict(c), "judge's reason": c.get("reason", "")}
                   for i, c in enumerate(f["claims"], 1)])
        else:
            st.warning(f"No faithfulness score: {f.get('error') or 'judge switched off'}")
        st.markdown("**Sources the answer was checked against** (what the assistant retrieved for this question)")
        srows = [{"ref": s_["ref"], "document · section": f"{s_['title']} — {s_.get('section') or ''}", "weak": "⚠" if s_.get("weak") else ""} for s_ in r.get("sources") or []]
        for tc in r.get("tool_calls") or []:
            if tc.get("name") != "search_knowledge_base" and isinstance(tc.get("result"), dict) and tc["result"].get("ok", True):
                d = tc["result"].get("data") or {}
                srows.append({"ref": "tool", "document · section": f"{tc['name']}({', '.join(f'{k}={v}' for k, v in (tc.get('args') or {}).items())}) → " + ", ".join(f"{k}={v}" for k, v in list(d.items())[:4]), "weak": ""})
        table(srows)
        if exp.get("facts"):
            st.caption("Expected facts from the question set (for your own comparison): " + " · ".join(exp["facts"]))
    with st.expander("Raw tool results (JSON)"):
        st.json({tc.get("name"): tc.get("result") for tc in r.get("tool_calls") or []})


# ----------------------------------------------------------------------------- 2 faithfulness
_backends = e2e.get("judge_backends") or {}
_backend_label = "RAGAS" if set(_backends) <= {"ragas", "none"} else ("built-in judge" if set(_backends) <= {"builtin", "none"} else "RAGAS + built-in judge")
st.header(f"2 · Faithfulness ({_backend_label}) — judge {e2e.get('judge_model')}, answering model {e2e.get('model')}")
if e2e.get("judge_fallback_reason"):
    st.warning(f"RAGAS did not score all questions in this run — {e2e['judge_fallback_reason']}. Scored by: "
               + ", ".join(f"{n} × {b}" for b, n in _backends.items()) + ". The built-in judge runs the same two steps (claims → verdicts) with the same judge model.", icon="⚠️")
elif _backends:
    st.caption("Scored by: " + ", ".join(f"{n} × {b}" for b, n in _backends.items()))
frows = [{"id": r["id"], "label": r["id"] + ("●" if r.get("probe") else ""), "score": r["faithfulness"].get("score"), "probe": r.get("probe", False),
          "kind": "probe question" if r.get("probe") else "faithfulness score",
          "supported": r["faithfulness"].get("supported"), "total": r["faithfulness"].get("total"), "group": r["group"]}
         for r in results if r.get("faithfulness")]
scored = [r for r in frows if r["score"] is not None]
if not scored:
    st.info("No faithfulness scores in this run (judge switched off or failed).")
else:
    fdf = pd.DataFrame(scored).sort_values("score")
    st.markdown("**Score per question** — share of the answer's factual claims supported by the retrieved passages and tool results.  \n"
                "Key: <span style='display:inline-block;width:11px;height:11px;background:" + BLUE + ";border-radius:2px;vertical-align:-1px'></span> faithfulness score &nbsp;·&nbsp; "
                "<span style='display:inline-block;width:11px;height:11px;background:" + PROBE_BLUE + ";border-radius:2px;vertical-align:-1px'></span> probe question, also marked "
                "with **●** after its ID — a question written to tempt the model into adding facts the sources do not contain.", unsafe_allow_html=True)
    sel = alt.selection_point(fields=["id"], name="pick", on="click", clear="dblclick")
    chart = (
        alt.Chart(fdf).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("label:N", sort=list(fdf["label"]), title=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1.05]), title="faithfulness"),
            color=alt.condition(sel, alt.Color("kind:N", scale=alt.Scale(domain=["faithfulness score", "probe question"], range=[BLUE, PROBE_BLUE]), legend=None), alt.value("#9ec5f4")),
            stroke=alt.condition(sel, alt.value("#0b0b0b"), alt.value("transparent")),
            tooltip=["id", alt.Tooltip("score:Q", format=".2f"), "supported", "total", "group"],
        )
        .add_params(sel)
        .properties(height=380)
    )
    event = st.altair_chart(chart, width="stretch", height=380, on_select="rerun", selection_mode="pick")
    picked = [p.get("id") for p in (event.selection.get("pick") or [])] if hasattr(event, "selection") else []
    # react to *changes* of the chart selection only: a click selects, a double-click (empty selection) clears, and a
    # rerun caused by another widget leaves the open details alone
    if picked != st.session_state.get("eval_pick_prev", []):
        st.session_state["eval_selected"] = picked[0] if picked else None
    st.session_state["eval_pick_prev"] = picked
    unscored = [r["id"] for r in frows if r["score"] is None]
    st.caption("Sorted from lowest to highest. Click a bar to open that question's evaluation details right here below the chart; double-click the chart to close them."
               + (f" No score for: {', '.join(unscored)} (judge error or no evidence)." if unscored else ""))
    _sel = st.session_state.get("eval_selected")
    _by_id = {r["id"]: r for r in e2e["results"]}
    if _sel and _sel in _by_id:
        render_details(_by_id[_sel])

# ----------------------------------------------------------------------------- 3 tool selection
st.header("3 · Tool selection accuracy")
stats = [g for g in group_tool_stats(e2e) if g["group"] in shown_groups]
left, right = st.columns([1, 1.3])
with left:
    gdf = pd.DataFrame([{"group": f"{g['group']} ({g['questions']})", "state": st_, "n": n}
                        for g in stats for st_, n in (("correct", g["correct"]), ("failed", g["questions"] - g["correct"])) if n])
    if not gdf.empty:
        gchart = alt.Chart(gdf).mark_bar(cornerRadius=4).encode(
            y=alt.Y("group:N", sort=[f"{g['group']} ({g['questions']})" for g in stats], title=None),
            x=alt.X("n:Q", title="questions", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("state:N", scale=alt.Scale(domain=["correct", "failed"], range=[GOOD, CRITICAL]), legend=alt.Legend(title=None, orient="top")),
            order=alt.Order("state:N", sort="ascending"),
            tooltip=["group", "state", "n"],
        ).properties(height=120 + 45 * len(stats), title=f"Correct vs failed by group — {s['tool_selection']['correct']} of {s['tool_selection']['total']} correct")
        st.altair_chart(gchart, width="stretch", height=120 + 45 * len(stats))
with right:
    fails = [f for f in tool_matrix(e2e) if f["id"] in {r["id"] for r in results}]
    st.markdown("**Which tool went wrong where** — questions with at least one failure")
    if not fails:
        st.success("Every question called exactly the expected tools with the expected arguments.")
    else:
        sym = {"ok": ("✔", GOOD, "#fff"), "missing": ("✖", CRITICAL, "#fff"), "forbidden": ("⛔", FORBIDDEN, "#fff"), "args": ("⚠", WARNING, "#0b0b0b"), "extra": ("·", "#dcdcd6", "#52514e"), "": ("·", "#eeede8", "#c3c2b7")}
        html = ["<table style='border-collapse:collapse;font-size:13px'><tr><th style='text-align:left;padding:4px 8px'>ID</th>"
                + "".join(f"<th style='padding:4px 6px;font-size:11px;color:{MUTED}'>{v}</th>" for v in TOOL_SHORT.values()) + "<th style='text-align:left;padding:4px 8px'>what happened</th></tr>"]
        for f in fails:
            cells = "".join(f"<td style='text-align:center;padding:3px'><span style='display:inline-block;width:26px;height:22px;line-height:22px;border-radius:4px;background:{sym[f['cells'][t]][1]};color:{sym[f['cells'][t]][2]}'>{sym[f['cells'][t]][0]}</span></td>" for t in TOOL_SHORT)
            html.append(f"<tr><td style='padding:3px 8px'><b>{f['id']}</b></td>{cells}<td style='padding:3px 8px;color:{MUTED}'>{'; '.join(f['reasons'])}</td></tr>")
        html.append("</table>")
        st.markdown("".join(html), unsafe_allow_html=True)
        st.caption("✔ expected & called · ✖ expected but not called · ⛔ forbidden but called · ⚠ called with wrong arguments · · not involved")
    with st.expander("Full question × tool matrix"):
        full = tool_matrix(e2e, failing_only=False)
        _word = {"ok": "✔ called", "missing": "✖ not called", "forbidden": "⛔ forbidden, called", "args": "⚠ wrong arguments", "extra": "called (not required)", "": "·"}
        table([{"ID": f["id"], **{TOOL_SHORT[t]: _word.get(f["cells"][t], f["cells"][t]) for t in TOOL_SHORT}, "result": "✔" if not f["reasons"] else "✖ " + "; ".join(f["reasons"])} for f in full if f["id"] in {r["id"] for r in results}])

# ----------------------------------------------------------------------------- 4 not covered
st.header(f"4 · Handling of questions not covered by the knowledge base — {s['not_covered']['handled']} / {s['not_covered']['total']}")
ncrows = [{"ID": r["id"] + (" (hard)" if r.get("hard") else ""), "question": r["question"],
           "weak-coverage notice shown": "yes" if r["coverage"].get("notice_shown") else "no", "citations in answer": r["coverage"].get("citations", 0),
           "result": "✔ handled" if r["coverage"].get("handled") else ("✖ " + r["coverage"]["handled_reason"] if r["coverage"].get("handled_reason") else "✖ answered as if covered")} for r in e2e["results"] if not r["covered"]]
table(ncrows)
st.caption("handled = the app said the knowledge base does not cover the question, cited no passage for it, and the judge found no unsupported **factual** statement in the "
           "answer. The *hard* probe is a topic the knowledge base mentions in passing: there the app may quote that passage, as long as it adds nothing beyond it. "
           "Naming the responsible authority or its website never counts against the answer.")
for _r in e2e["results"]:
    if not _r["covered"] and _r["coverage"].get("handled_note"):
        st.caption(f"ℹ️ {_r['id']}: {_r['coverage']['handled_note']}")

