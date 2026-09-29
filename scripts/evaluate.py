"""Run the evaluation harness.

    python scripts\\evaluate.py                       # retrieval + end-to-end on data/eval/questions.yaml
    python scripts\\evaluate.py --only retrieval      # cheap: hit-rate@5 and MRR for the four retriever configurations
    python scripts\\evaluate.py --only e2e            # agent + deterministic checks + faithfulness
    python scripts\\evaluate.py --questions data/eval/questions_full.yaml
    python scripts\\evaluate.py --ids Q11,Q18 --no-judge
    python scripts\\evaluate.py --model openai/gpt-4o-mini --judge anthropic/claude-haiku-4.5 --workers 3   # alternative judge

Every run writes reports/eval/<timestamp>/ with meta.json, retrieval.json, e2e.json, summary.md and report.pdf;
the Evaluation page of the app lists these runs. Stop the Streamlit app before running (Chroma is single-writer).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigator.config import get_api_key, get_settings  # noqa: E402
from navigator.eval import store  # noqa: E402
from navigator.eval.cost import credits_used, estimate_from_tokens  # noqa: E402
from navigator.eval.e2e_eval import evaluate_e2e  # noqa: E402
from navigator.eval.faithfulness import FaithfulnessJudge  # noqa: E402
from navigator.eval.questions import DEFAULT_QUESTIONS, load_questions  # noqa: E402
from navigator.eval.report import build_pdf, summary_markdown  # noqa: E402
from navigator.eval.retrieval_eval import evaluate_retrieval  # noqa: E402
from navigator.rag.retriever import get_retriever  # noqa: E402
from navigator.utils.errors import NavigatorError  # noqa: E402


def _fmt_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m} min {sec:02d} s" if m else f"{sec} s"


def _progress(stage: str, i: int, n: int, msg: str) -> None:
    print(f"  [{i:>2}/{n}] {msg}", flush=True)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles: never crash on ✔ / — characters
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["retrieval", "e2e"], help="run only one half (default: both)")
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="question file (default: data/eval/questions.yaml)")
    ap.add_argument("--ids", help="comma-separated question ids to run, e.g. Q11,Q18")
    ap.add_argument("--limit", type=int, help="run only the first N questions (smoke test)")
    ap.add_argument("--model", default=s.llm.models[0], help=f"answering model (default {s.llm.models[0]})")
    ap.add_argument("--judge", default="google/gemini-2.5-flash",
                    help="faithfulness judge model — keep it a different family than --model (default google/gemini-2.5-flash; "
                         "anthropic/claude-haiku-4.5 agreed with it claim-for-claim in testing at about twice the cost)")
    ap.add_argument("--judge-backend", choices=["auto", "ragas", "builtin"], default="auto", help="auto = RAGAS if installed, else built-in")
    ap.add_argument("--no-judge", action="store_true", help="skip faithfulness (deterministic checks only)")
    ap.add_argument("--workers", type=int, default=1, help="questions in flight at once for the e2e half (default 1)")
    ap.add_argument("--k", type=int, default=s.retrieval.final_k, help="top-k for the retrieval metrics")
    ap.add_argument("--run", help="write into this existing run folder instead of a new one (e.g. to add e2e to a retrieval run)")
    ap.add_argument("--no-grounding", action="store_true", help="skip the app's grounding check + revision (measures the raw agent answer)")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    if not get_api_key():
        print(f"✖ {s.llm.api_key_env} is not set.")
        return 2
    try:
        questions = load_questions(args.questions, ids=args.ids.split(",") if args.ids else None, limit=args.limit)
    except Exception as exc:
        print(f"✖ {exc}")
        return 2

    run_dir = Path(args.run) if args.run else store.new_run_dir()
    if args.run and not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    qpath = Path(args.questions).resolve()
    qfile = str(qpath.relative_to(ROOT)) if qpath.is_relative_to(ROOT) else str(qpath)
    meta = store.update_meta(run_dir, run=run_dir.name, question_file=qfile, questions=len(questions), ids=[q.id for q in questions],
                             model=args.model, judge=None if args.no_judge else args.judge)
    print(f"✔ loaded {len(questions)} questions from {args.questions} → run folder {run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}")
    credits_before = credits_used()  # OpenRouter credits used by this key so far; the difference after the run is the run's cost

    retrieval = e2e = None
    try:
        retriever = get_retriever()
        count = retriever.count()
    except NavigatorError as exc:
        print(f"✖ {exc}")
        return 2
    if count == 0:
        print("✖ The knowledge base is empty — run python scripts\\ingest.py first.")
        return 2

    if args.only in (None, "retrieval"):
        scored = [q for q in questions if q.expected_docs]
        print(f"\n— Retrieval: {len(scored)} questions with an expected document · index {count} chunks · embeddings {s.embeddings.model} · reranker {retriever.reranker.name}")
        rep = evaluate_retrieval(questions, retriever, k=args.k, progress=_progress)
        retrieval = rep.to_dict()
        store.write_json(run_dir / "retrieval.json", retrieval)
        print("\n  configuration        hit-rate@%d   MRR" % args.k)
        for c in rep.configs:
            print(f"  {c.label:<20} {100 * c.hit_rate:>6.0f} %     {c.mrr:.2f}")
        print(f"✔ written {run_dir.name}/retrieval.json  ({_fmt_seconds(rep.seconds)})")

    if args.only in (None, "e2e"):
        judge = None
        if not args.no_judge:
            judge = FaithfulnessJudge(model=args.judge, backend=args.judge_backend)
            if judge.fallback_reason:
                print(f"  ⚠ FAITHFULNESS JUDGE: built-in judge will be used, not RAGAS — {judge.fallback_reason}")
            if judge.model == args.model:
                print("  ⚠ judge and answering model are identical — a model is lenient towards its own phrasing; consider --judge <other model>")
        print(f"\n— End-to-end: {len(questions)} questions · agent {args.model} · judge {judge.model + ' via ' + judge.backend if judge else 'off'} · workers {args.workers}")
        t0 = time.time()
        rep2 = evaluate_e2e(questions, args.model, judge, workers=args.workers, progress=_progress, grounding=not args.no_grounding)
        e2e = rep2.to_dict()
        store.write_json(run_dir / "e2e.json", e2e)
        sm = e2e["summary"]
        f = sm["faithfulness"]
        print(f"\n  mean faithfulness {f['mean'] if f['mean'] is not None else 'n/a'} (probes {f['probes'] if f['probes'] is not None else 'n/a'}) · "
              f"tool selection {sm['tool_selection']['correct']}/{sm['tool_selection']['total']} · not-covered handled {sm['not_covered']['handled']}/{sm['not_covered']['total']}"
              + f" · answers revised by the grounding step {sm.get('revised', 0)}/{sm['questions']}"
              + (f" · {sm['errors']} agent error(s)" if sm["errors"] else ""))
        if judge is not None:
            counts = e2e.get("judge_backends", {})
            summary_line = ", ".join(f"{n} question(s) by {b}" for b, n in counts.items())
            if e2e.get("judge_fallback_reason"):
                print(f"  ⚠ FAITHFULNESS JUDGE: {summary_line}. RAGAS was NOT used for all questions — {e2e['judge_fallback_reason']}")
            else:
                print(f"  faithfulness judge: {summary_line}")
        tok = e2e["tokens"]
        print(f"✔ written {run_dir.name}/e2e.json  ({_fmt_seconds(time.time() - t0)}, agent tokens in/out {tok['input']:,}/{tok['output']:,})")

    existing = store.load_run(run_dir)
    retrieval = retrieval or existing.retrieval
    e2e = e2e or existing.e2e

    credits_after = credits_used() if credits_before is not None else None
    if credits_after is not None and credits_after >= credits_before:
        # --run adds a half to an existing run: add to the cost already recorded for it
        prev_cost = float(meta.get("cost_usd") or 0.0) if meta.get("cost_source") == "openrouter_key_usage" else 0.0
        run_cost = round(credits_after - credits_before + prev_cost, 4)
        meta = store.update_meta(run_dir, cost_usd=run_cost, cost_source="openrouter_key_usage")
        print(f"  cost of this run: ${run_cost:.2f} (OpenRouter credits used — agent, judge, embeddings and reranker)")
    else:
        est = estimate_from_tokens(e2e)
        if est is not None:
            meta = store.update_meta(run_dir, cost_usd=est, cost_source="estimate_agent_tokens")
            print(f"  cost of this run: ≈ ${est:.2f} (estimated from the answering model's tokens only — the OpenRouter key usage could not be read)")
    (run_dir / "summary.md").write_text(summary_markdown(meta, retrieval, e2e), encoding="utf-8")
    written = ["summary.md"]
    if not args.no_pdf:
        try:
            build_pdf(run_dir, meta, retrieval, e2e)
            written.append("report.pdf")
        except Exception as exc:  # the PDF is a convenience; never fail the run because of it
            print(f"  ⚠ PDF not written: {exc}")
    print(f"✔ written {run_dir.name}/{' and '.join(written)} — open the Evaluation page in the app to browse the results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
