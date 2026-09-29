"""Reports for an evaluation run: summary.md (terminal + repo) and report.pdf (download from the app)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from navigator.eval.cost import cost_label
from navigator.eval.retrieval_eval import CONFIGS

# Same palette as the Evaluation page (validated for colour-vision deficiency, see docs/dataviz).
CONFIG_COLOURS = {"dense": "#2a78d6", "bm25": "#eb6834", "hybrid": "#1baf7a", "hybrid_rerank": "#eda100"}
BLUE, GOOD, CRITICAL, WARNING, MUTED = "#2a78d6", "#0ca30c", "#d03b3b", "#fab219", "#898781"
TOOL_SHORT = {"search_knowledge_base": "search_kb", "check_blue_card_salary": "blue_card", "calculate_deadline": "deadline",
              "estimate_net_salary": "net_salary", "convert_currency": "currency"}


# ----------------------------------------------------------------------------- helpers

def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f} %"


def _num(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def group_tool_stats(e2e: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for r in e2e.get("results", []):
        g = rows.setdefault(r["group"], {"group": r["group"], "questions": 0, "correct": 0, "failures": []})
        g["questions"] += 1
        if r.get("tools", {}).get("passed"):
            g["correct"] += 1
        else:
            g["failures"].append({"id": r["id"], "reasons": r.get("tools", {}).get("reasons") or (["agent error"] if r.get("error") else ["failed"])})
    order = ["knowledge", "mixed", "tool_only", "not_covered"]
    return [rows[g] for g in order if g in rows] + [v for k, v in rows.items() if k not in order]


def tool_matrix(e2e: dict[str, Any], failing_only: bool = True) -> list[dict[str, Any]]:
    """Question × tool cells: 'ok' expected&called · 'missing' · 'forbidden' · 'args' · 'extra' (called, neutral) · ''."""
    rows = []
    for r in e2e.get("results", []):
        t = r.get("tools", {})
        if failing_only and t.get("passed"):
            continue
        called = set(t.get("called") or [])
        expected = set(r.get("expected", {}).get("tools") or [])
        forbidden = set(r.get("expected", {}).get("forbidden") or [])
        bad_args = {m.split(".")[0] for m in (t.get("arg_mismatches") or [])}
        cells = {}
        for tool in TOOL_SHORT:
            if tool in bad_args:
                cells[tool] = "args"
            elif tool in expected and tool in called:
                cells[tool] = "ok"
            elif tool in expected:
                cells[tool] = "missing"
            elif tool in forbidden and tool in called:
                cells[tool] = "forbidden"
            elif tool in called:
                cells[tool] = "extra"
            else:
                cells[tool] = ""
        if t.get("search_missing") and cells.get("search_knowledge_base") == "":
            cells["search_knowledge_base"] = "missing"
        rows.append({"id": r["id"], "cells": cells, "reasons": t.get("reasons") or []})
    return rows


# ----------------------------------------------------------------------------- summary.md

def claim_kind(c: dict) -> str:
    return c.get("kind") or "fact"


def claim_verdict(c: dict) -> str:
    if claim_kind(c) != "fact":
        return "not scored"
    return "supported" if c.get("verdict") else "UNSUPPORTED"


def faithfulness_headline(f: dict) -> str:
    """'Faithfulness: 3 of 4 factual claims supported · 2 referral/meta statements not scored'."""
    total, supported = f.get("total", 0), f.get("supported", 0)
    unscored = f.get("unscored", 0)
    if total:
        head = f"Faithfulness: {supported} of {total} factual claims supported"
    else:
        head = "Faithfulness: no factual claims — the answer only says what the knowledge base lacks and whom to ask"
    if unscored:
        head += f" · {unscored} referral/meta statement(s) not scored"
    return head


def summary_markdown(meta: dict[str, Any], retrieval: dict[str, Any] | None, e2e: dict[str, Any] | None) -> str:
    lines = [f"# Evaluation run {meta.get('run', '')}", ""]
    lines.append(f"{meta.get('questions', '?')} questions · created {meta.get('created', '')} · {cost_label(meta, e2e)}")
    lines.append("")
    if retrieval:
        lines += [f"## 1 · Retrieval — hit-rate@{retrieval['k']} and MRR ({len(retrieval['per_question'])} questions with an expected document)", "",
                  "| configuration | hit-rate | MRR |", "|---|---:|---:|"]
        for c in retrieval["configs"]:
            lines.append(f"| {c['label']} | {_pct(c['hit_rate'])} | {_num(c['mrr'])} |")
        lines += ["", f"reranker: {retrieval.get('reranker')} · embeddings: {retrieval.get('embedding_model')} · {retrieval.get('seconds')} s", ""]
    if e2e:
        s = e2e["summary"]
        f = s["faithfulness"]
        lines += [f"## 2 · Faithfulness (judge {e2e.get('judge_model')} via {e2e.get('judge_backend')})", "",
                  f"mean **{_num(f['mean'])}** · probes {_num(f['probes'])} · others {_num(f['others'])}", ""]
        if e2e.get("judge_backends"):
            lines.append("scored by: " + ", ".join(f"{n} × {b}" for b, n in e2e["judge_backends"].items()))
        if e2e.get("judge_fallback_reason"):
            lines.append(f"**⚠ RAGAS was not used for all questions — {e2e['judge_fallback_reason']}**")
        lines.append("")
        low = sorted([r for r in e2e["results"] if r.get("faithfulness", {}).get("score") is not None], key=lambda r: r["faithfulness"]["score"])[:5]
        if low:
            lines += ["Lowest scores:", ""] + [f"- {r['id']} {_num(r['faithfulness']['score'])} — {r['faithfulness'].get('total', 0) - r['faithfulness'].get('supported', 0)} unsupported claim(s)" for r in low] + [""]
        ts = s["tool_selection"]
        lines += [f"## 3 · Tool selection — {ts['correct']} / {ts['total']} correct ({_pct(ts['correct'] / max(ts['total'], 1))})", "",
                  "| group | questions | correct | failures |", "|---|---:|---:|---|"]
        for g in group_tool_stats(e2e):
            fails = "; ".join(f"{x['id']} ({', '.join(x['reasons'])})" for x in g["failures"]) or "—"
            lines.append(f"| {g['group']} | {g['questions']} | {g['correct']} | {fails} |")
        nc = s["not_covered"]
        lines += ["", f"## 4 · Not-covered questions handled — {nc['handled']} / {nc['total']}", "",
                  "| id | notice shown | citations | handled |", "|:---:|:---:|:---:|:---:|"]
        for r in e2e["results"]:
            if not r["covered"]:
                c = r["coverage"]
                lines.append(f"| {r['id']}{' (hard)' if r.get('hard') else ''} | {'yes' if c.get('notice_shown') else 'no'} | {c.get('citations', 0)} | {'yes' if c.get('handled') else 'NO'} |")
        tok = e2e.get("tokens", {})
        lines += ["", f"agent model: {e2e.get('model')} · {e2e.get('seconds')} s · tokens in/out {tok.get('input', 0):,} / {tok.get('output', 0):,} · "
                  f"answers revised by the grounding step: {s.get('revised', 0)} / {s['questions']}",
                  f"questions with a failed tool call in the final attempt: {s.get('tool_errors', 0)} / {s['questions']} (re-run once after a failure: {s.get('retried', 0)})", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- charts (matplotlib, Agg)

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    return plt


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    _mpl().close(fig)
    return buf.getvalue()


def chart_retrieval(retrieval: dict[str, Any]) -> bytes:
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
    for ax, key, title, fmt in ((axes[0], "hit_rate", f"Hit-rate@{retrieval['k']}", lambda v: f"{100 * v:.0f}%"), (axes[1], "mrr", "Mean reciprocal rank", lambda v: f"{v:.2f}")):
        labels = [c["label"] for c in retrieval["configs"]]
        vals = [c[key] for c in retrieval["configs"]]
        cols = [CONFIG_COLOURS.get(c["config"], BLUE) for c in retrieval["configs"]]
        bars = ax.bar(labels, vals, color=cols, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, fmt(v), ha="center", va="bottom", fontsize=9, color="#0b0b0b")
        ax.set_ylim(0, 1.12)
        ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
        ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.tight_layout()
    return _png(fig)


def chart_faithfulness(e2e: dict[str, Any]) -> bytes:
    plt = _mpl()
    rows = [r for r in e2e["results"] if r.get("faithfulness", {}).get("score") is not None]
    rows.sort(key=lambda r: r["faithfulness"]["score"])
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    ids = [r["id"] + ("●" if r.get("probe") else "") for r in rows]
    vals = [r["faithfulness"]["score"] for r in rows]
    bars = ax.bar(ids, vals, color=BLUE, width=0.7)
    for b, v in zip(bars, vals):
        if v < 1.0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.set_title("Faithfulness per question (● = probe), sorted ascending", loc="left", fontsize=10)
    fig.tight_layout()
    return _png(fig)


def chart_tools(e2e: dict[str, Any]) -> bytes:
    plt = _mpl()
    stats = group_tool_stats(e2e)
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(stats) + 1.0))
    labels = [f"{g['group']} ({g['questions']})" for g in stats][::-1]
    correct = [g["correct"] for g in stats][::-1]
    failed = [g["questions"] - g["correct"] for g in stats][::-1]
    ax.barh(labels, correct, color=GOOD, height=0.55, label="correct")
    ax.barh(labels, failed, left=correct, color=CRITICAL, height=0.55, label="failed")
    for i, (c, f) in enumerate(zip(correct, failed)):
        ax.text(c + f + 0.15, i, f"{c} ✔" + (f" · {f} ✖" if f else ""), va="center", fontsize=8)
    ax.set_xlim(0, max(1, max(q["questions"] for q in stats)) * 1.25)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    s = e2e["summary"]["tool_selection"]
    ax.set_title(f"Tool selection — {s['correct']} of {s['total']} correct", loc="left", fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    return _png(fig)


# ----------------------------------------------------------------------------- PDF (reportlab)

def build_pdf(run_dir: Path, meta: dict[str, Any], retrieval: dict[str, Any] | None, e2e: dict[str, Any] | None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#52514e"))
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=17, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=10, spaceBefore=6, spaceAfter=2)
    grid = TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("LEADING", (0, 0), (-1, -1), 10), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e1e0d9")),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")), ("VALIGN", (0, 0), (-1, -1), "TOP")])
    # centred variant (not-covered table): text centred in each cell, cells vertically centred
    body_c = ParagraphStyle("body_c", parent=body, alignment=TA_CENTER)
    small_c = ParagraphStyle("small_c", parent=small, alignment=TA_CENTER)
    grid_c = TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("LEADING", (0, 0), (-1, -1), 10), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e1e0d9")),
                         ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])

    def esc(text: Any) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")

    def P(markup: str, st=body) -> Paragraph:
        """markup may contain <b>/<i>; dynamic content must be passed through esc() by the caller."""
        return Paragraph(markup, st)

    def table(rows: list[list[Any]], widths: list[float] | None = None, centre: bool = False) -> Table:
        st_body, st_small = (body_c, small_c) if centre else (body, small)
        t = Table([[P(esc(c), st_small if (isinstance(c, str) and len(c) > 28) else st_body) for c in row] for row in rows], colWidths=widths, repeatRows=1)
        t.setStyle(grid_c if centre else grid)
        return t

    out = run_dir / "report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"Evaluation {meta.get('run', '')}", author="German Bureaucracy Navigator")
    W = A4[0] - 32 * mm
    story: list[Any] = [P(esc(f"German Bureaucracy Navigator — evaluation run {meta.get('run', '')}"), h1),
                        P(esc(f"{meta.get('questions', '?')} questions · created {meta.get('created', '')} · {cost_label(meta, e2e)}"), small), Spacer(1, 6)]

    # headline tiles
    tiles = []
    if retrieval:
        prod = next((c for c in retrieval["configs"] if c["config"] == "hybrid_rerank"), retrieval["configs"][-1])
        tiles += [[f"Hit-rate@{retrieval['k']} ({prod['label']})", _pct(prod["hit_rate"])], ["MRR", _num(prod["mrr"])]]
    if e2e:
        s = e2e["summary"]
        tiles += [["Mean faithfulness", _num(s["faithfulness"]["mean"])],
                  ["Tool selection correct", f"{s['tool_selection']['correct']} / {s['tool_selection']['total']}"],
                  ["Not-covered handled", f"{s['not_covered']['handled']} / {s['not_covered']['total']}"]]
    if tiles:
        t = Table([[P(esc(k), small) for k, _ in tiles], [P(f"<b>{esc(v)}</b>", ParagraphStyle("v", parent=body, fontSize=13)) for _, v in tiles]], colWidths=[W / len(tiles)] * len(tiles))
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")), ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e1e0d9"))]))
        story += [t, Spacer(1, 8)]

    if retrieval:
        story += [P(esc(f"1 · Retrieval evaluation — hit-rate@{retrieval['k']} and MRR"), h2), Image(io.BytesIO(chart_retrieval(retrieval)), width=W, height=W * 2.9 / 8.2),
                  P(esc(f"{len(retrieval['per_question'])} questions with an expected document · reranker {retrieval.get('reranker')} · embeddings {retrieval.get('embedding_model')}. "
                        "All four configurations use the same query variants and embeddings; only the retrieval method differs."), small)]
        rows = [["ID", "expected document(s)"] + [CONFIGS[c] for c in CONFIGS]]
        for q in retrieval["per_question"]:
            rows.append([q["id"], ", ".join(d.replace(".md", "") for d in q["expected_docs"])] + [str(q["ranks"].get(c) or "—") for c in CONFIGS])
        story += [Spacer(1, 4), table(rows, [14 * mm, 70 * mm] + [(W - 84 * mm) / 4] * 4), P("rank = position of the first expected document in the top-k; — = not retrieved (counts 0 for MRR)", small)]

    if e2e:
        s = e2e["summary"]
        story += [PageBreak(), P(esc(f"2 · Faithfulness — judge {e2e.get('judge_model')} ({e2e.get('judge_backend')}), answering model {e2e.get('model')}"), h2),
                  Image(io.BytesIO(chart_faithfulness(e2e)), width=W, height=W * 2.9 / 8.2),
                  P(esc(f"Mean {_num(s['faithfulness']['mean'])} · probes {_num(s['faithfulness']['probes'])} · others {_num(s['faithfulness']['others'])}. "
                        "Faithfulness = share of the answer's factual claims that are supported by the passages and tool results the assistant retrieved for that question. "
                        "Statements that only refer the user to an authority or website, or say what the knowledge base does not cover, are not scored."), small)]
        story += [P("3 · Tool selection accuracy", h2), Image(io.BytesIO(chart_tools(e2e)), width=W, height=W * (0.55 * len(group_tool_stats(e2e)) + 1.0) / 8.2)]
        fails = tool_matrix(e2e)
        if fails:
            rows = [["ID"] + list(TOOL_SHORT.values()) + ["what happened"]]
            # plain text for the two states whose symbols (⚠, ⛔) are missing from the PDF's Helvetica font
            sym = {"ok": "✔", "missing": "✖ not called", "forbidden": "✖ FORBIDDEN, called", "args": "wrong args", "extra": "called", "": "·"}
            for f in fails:
                rows.append([f["id"]] + [sym[f["cells"][t]] for t in TOOL_SHORT] + ["; ".join(f["reasons"])])
            story += [Spacer(1, 4), table(rows, [12 * mm] + [21 * mm] * 5 + [W - 117 * mm])]
        else:
            story.append(P("Every question called exactly the expected tools with the expected arguments.", body))
        story.append(P(esc(f"4 · Handling of questions not covered by the knowledge base — {s['not_covered']['handled']} / {s['not_covered']['total']}"), h2))
        rows = [["ID", "question", "notice shown", "citations", "handled"]]
        for r in e2e["results"]:
            if not r["covered"]:
                c = r["coverage"]
                rows.append([r["id"] + (" (hard)" if r.get("hard") else ""), r["question"], "yes" if c.get("notice_shown") else "no", str(c.get("citations", 0)),
                             "yes" if c.get("handled") else ("NO — " + c["handled_reason"] if c.get("handled_reason") else "NO")])
        story += [table(rows, [18 * mm, W - 88 * mm, 22 * mm, 18 * mm, 30 * mm], centre=True),
                  P("handled = the app said the knowledge base does not cover the question, cited no passage for it, and the judge found no unsupported factual statement "
                    "in the answer. The hard probe is a topic the knowledge base mentions in passing: there the app may quote that passage, as long as it adds nothing "
                    "beyond it. Naming the responsible authority or its website never counts against the answer.", small)]

        story += [PageBreak(), P("5 · Evaluation details for individual test questions", h2)]
        for r in e2e["results"]:
            f = r.get("faithfulness") or {}
            head = f"{r['id']} · {r['group']}" + (" · hard" if r.get("hard") else "") + f" · faithfulness {_num(f.get('score'))} · tools {'✔' if r['tools'].get('passed') else '✖ ' + '; '.join(r['tools'].get('reasons') or [])}" + (" · revised by the grounding step" if r.get("answer_original") else "") + (" · TOOL FAILURE" if r.get("tool_errors") else "")
            block: list[Any] = [P(esc(head), h3), P(f"<b>Question:</b> {esc(r['question'])}", body)]
            if r.get("error"):
                block.append(P(f"<b>Agent error:</b> {esc(r['error'])}", body))
            block.append(P(f"<b>Answer{' (after grounding revision)' if r.get('answer_original') else ''}:</b> {esc(r.get('answer', '')[:2500])}", body))
            if r.get("answer_original"):
                block.append(P(f"<b>Original answer before revision:</b> {esc(r['answer_original'][:1500])}", small))
            calls = r.get("tool_calls") or []
            if calls:
                if any(tc.get("error") for tc in calls):
                    block.append(table([["tool", "arguments", "result"]] + [[TOOL_SHORT.get(tc.get("name"), tc.get("name")), _fmt_args(tc.get("args")), ("FAILED: " + tc["error"]) if tc.get("error") else "ok"] for tc in calls], [30 * mm, 70 * mm, W - 100 * mm]))
                else:
                    block.append(table([["tool", "arguments"]] + [[TOOL_SHORT.get(tc.get("name"), tc.get("name")), _fmt_args(tc.get("args"))] for tc in calls], [30 * mm, W - 30 * mm]))
            if r.get("retried"):
                block.append(P("A tool failed in the first attempt; the question was run again (see notes).", small))
            if f.get("claims"):
                block.append(P(esc(faithfulness_headline(f)), h3))
                block.append(table([["#", "statement", "kind", "verdict", "reason"]] + [[str(i), c["statement"], claim_kind(c), claim_verdict(c), c.get("reason", "")] for i, c in enumerate(f["claims"], 1)], [8 * mm, 62 * mm, 20 * mm, 22 * mm, W - 112 * mm]))
            if r.get("sources"):
                block.append(P(esc("Sources the answer was checked against: " + "; ".join(f"[{s_['ref']}] {s_['title']} — {s_.get('section') or ''}" for s_ in r["sources"])), small))
            story.append(KeepTogether(block[:3]))
            story += block[3:]
            story.append(Spacer(1, 8))

    doc.build(story)
    return out


def _fmt_args(args: Any) -> str:
    if not args:
        return "—"
    return ", ".join(f"{k}={v}" for k, v in args.items())
