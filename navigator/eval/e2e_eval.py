"""End-to-end evaluation: run the real agent on each question, then score it.

Deterministic checks (no judge):
    tool selection   every expected tool called, no forbidden tool called, expected arguments present
    coverage         for not-covered questions: the search reported weak/none coverage (any flag for a "hard"
                     probe the KB mentions in passing) and the answer carries no [Sn] citation
LLM-judged:
    faithfulness     share of the answer's *factual* claims supported by the retrieved passages and tool results;
                     statements that only refer the user to an authority/website or say what the knowledge base
                     lacks are labelled referral/meta and not scored (see faithfulness.py)
    handled          a not-covered question is handled when the coverage check passes AND the judge found no
                     unsupported factual claim in the final answer (the "leaks")
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

from navigator.agent.graph import build_agent, run_agent_stream
from navigator.eval.faithfulness import FaithfulnessJudge, FaithfulnessResult
from navigator.eval.questions import EvalQuestion
from navigator.guardrails.grounding import apply_grounding
from navigator.guardrails.output_checks import DISCLAIMERS, check_output
from navigator.utils.errors import NavigatorError
from navigator.utils.logging import get_logger

log = get_logger(__name__)

_CITATION_RE = re.compile(r"\[S\d+\]")
Progress = Callable[[str, int, int, str], None]


# ----------------------------------------------------------------------------- deterministic checks

@dataclass
class ToolCheck:
    passed: bool
    called: list[str]
    missing: list[str] = field(default_factory=list)          # expected but not called
    forbidden_called: list[str] = field(default_factory=list)
    arg_mismatches: list[str] = field(default_factory=list)   # "tool.arg: expected X, got Y"
    search_missing: bool = False                              # coverage_check requires a search

    @property
    def reasons(self) -> list[str]:
        out = [f"missing {t}" for t in self.missing]
        out += [f"forbidden {t} called" for t in self.forbidden_called]
        out += self.arg_mismatches
        if self.search_missing and "missing search_knowledge_base" not in out:
            out.append("knowledge-base search required but not performed")
        return out


def _arg_matches(expected: Any, actual: Any) -> bool:
    if expected == "any":
        return actual is not None
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(actual) - float(expected)) <= max(0.5, abs(float(expected)) * 0.001)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, bool):
        return bool(actual) is expected
    return str(actual).strip().lower() == str(expected).strip().lower()


def check_tools(q: EvalQuestion, tool_calls: list[dict[str, Any]]) -> ToolCheck:
    called = [tc.get("name", "") for tc in tool_calls]
    called_set = set(called)
    missing = [t for t in q.expected_tools if t not in called_set]
    forbidden = [t for t in q.forbidden_tools if t in called_set]
    mismatches: list[str] = []
    for tool, expected_args in q.expected_args.items():
        calls = [tc for tc in tool_calls if tc.get("name") == tool]
        if not calls:
            continue  # already reported as missing
        # at least one call must satisfy every expected argument
        if not any(all(_arg_matches(v, (tc.get("args") or {}).get(k)) for k, v in expected_args.items()) for tc in calls):
            best = calls[-1].get("args") or {}
            for k, v in expected_args.items():
                if not any(_arg_matches(v, (tc.get("args") or {}).get(k)) for tc in calls):
                    mismatches.append(f"{tool}.{k}: expected {v}, got {best.get(k)!r}")
    search_missing = q.coverage_check and "search_knowledge_base" not in called_set
    passed = not missing and not forbidden and not mismatches and not search_missing
    return ToolCheck(passed=passed, called=called, missing=missing, forbidden_called=forbidden, arg_mismatches=mismatches, search_missing=search_missing)


@dataclass
class CoverageCheck:
    coverage: str | None                 # "ok" | "weak" | "none" | None (no search)
    notice_shown: bool
    citations: int
    handled: bool | None                 # only meaningful for covered=False questions
    handled_reason: str | None = None    # why not (deterministic part; run_question adds the judge's part)

    @property
    def label(self) -> str:
        return "no search" if self.coverage is None else self.coverage


def tool_error(tc: dict[str, Any]) -> str | None:
    """Error text of a failed tool call: a ToolResult with ok=False, or LangChain's raw 'Error: …' string when the
    tool raised (create_agent turns exceptions into a plain-text ToolMessage)."""
    res = tc.get("result")
    if isinstance(res, dict):
        if res.get("ok", True):
            return None
        return str(res.get("error") or res.get("message") or "tool reported a failure")
    if isinstance(res, str) and res.strip():
        return res.strip()[:400]
    return "no result recorded"


def tool_errors(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [f"{tc.get('name')}: {err}" for tc in tool_calls if (err := tool_error(tc))]


def search_results(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tc["result"] for tc in tool_calls if tc.get("name") == "search_knowledge_base" and isinstance(tc.get("result"), dict) and tc["result"].get("ok", True)]


def check_coverage(q: EvalQuestion, tool_calls: list[dict[str, Any]], final_text: str, notes: list[str]) -> CoverageCheck:
    results = search_results(tool_calls)
    coverages = [(r.get("data") or {}).get("coverage") for r in results]
    if not coverages:
        coverage = None
    elif all(c in ("weak", "none") for c in coverages):
        coverage = "weak" if "weak" in coverages else "none"
    else:
        coverage = "ok"
    notice = any(("no specific passage" in n) or ("keine spezifische" in n) for n in notes)
    citations = len(_CITATION_RE.findall(final_text))
    handled: bool | None = None
    reason: str | None = None
    if not q.covered:
        problems: list[str] = []
        if coverage is None:
            failed = [tc for tc in tool_calls if tc.get("name") == "search_knowledge_base"]
            problems.append(f"search_knowledge_base failed: {tool_error(failed[-1])}" if failed else "no knowledge-base search")
        elif coverage == "ok" and not q.is_hard:
            # a "hard" probe is one the KB mentions in passing, so an "ok" flag is expected there; for the others
            # an "ok" flag means the weak-coverage detector missed it
            problems.append("search reported coverage 'ok' (weak-coverage detector missed it)")
        if coverage in ("weak", "none") and citations:
            problems.append(f"{citations} [Sn] citation(s) despite weak coverage")
        handled = not problems
        reason = "; ".join(problems) or None
    return CoverageCheck(coverage=coverage, notice_shown=notice, citations=citations, handled=handled, handled_reason=reason)


# ----------------------------------------------------------------------------- evidence for the judge

def evidence_contexts(tool_calls: list[dict[str, Any]]) -> list[str]:
    """What the assistant actually saw: passage texts plus the data of every other successful tool."""
    out: list[str] = []
    for tc in tool_calls:
        res = tc.get("result")
        if not isinstance(res, dict) or not res.get("ok", True):
            continue
        data = res.get("data") or {}
        if tc.get("name") == "search_knowledge_base":
            for p in data.get("passages", []) or []:
                out.append(f"[{p.get('ref')}] {p.get('title')} — {p.get('section') or ''}\n{p.get('text', '')}")
        else:
            import json
            out.append(f"Tool {tc.get('name')} with arguments {json.dumps(tc.get('args') or {}, ensure_ascii=False)} returned: "
                       f"{json.dumps({'data': data, 'warnings': res.get('warnings', [])}, ensure_ascii=False, default=str)}")
    return out


def strip_disclaimer(text: str) -> str:
    for d in DISCLAIMERS.values():
        text = text.replace(f"_{d}_", "").rstrip()
    return text


_LABELLED_RE = re.compile(r"(?im)^[ \t]*(?:[-*•]\s*)?(?:\*\*)?(?:Not from the knowledge base|Nicht aus der Wissensdatenbank):(?:\*\*)?[^\n]*\n?")


def strip_labelled_general_knowledge(text: str) -> str:
    """Drop lines the assistant explicitly marked as not coming from the knowledge base.

    Faithfulness measures what is *presented as* knowledge-base-backed. A sentence the assistant itself labels
    "Not from the knowledge base:" is the honest behaviour we ask for; scoring it as a hallucination would push the
    model to drop the label rather than the claim."""
    return _LABELLED_RE.sub("", text).strip()


def collect_sources(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out, seen = [], set()
    for tc in tool_calls:
        res = tc.get("result")
        if tc.get("name") != "search_knowledge_base" or not isinstance(res, dict):
            continue
        for src in res.get("sources", []) or []:
            key = (src.get("ref"), src.get("title"))
            if key not in seen:
                seen.add(key)
                out.append({k: src.get(k) for k in ("ref", "title", "section", "url", "weak", "similarity")})
    return out


def valid_refs(tool_calls: list[dict[str, Any]]) -> set[str]:
    return {s["ref"] for s in collect_sources(tool_calls) if s.get("ref")}


def weak_coverage(tool_calls: list[dict[str, Any]]) -> bool:
    results = search_results(tool_calls)
    return bool(results) and all((r.get("data") or {}).get("coverage") in ("weak", "none") for r in results)


# ----------------------------------------------------------------------------- one question

@dataclass
class QuestionResult:
    id: str
    group: str
    language: str
    question: str
    covered: bool
    coverage_check: bool
    probe: bool
    hard: bool
    answer: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tools: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    faithfulness: dict[str, Any] = field(default_factory=dict)
    grounding: dict[str, Any] = field(default_factory=dict)      # app-side grounding check + revision, as the user sees it
    answer_original: str | None = None                            # answer before the grounding revision (None if unchanged)
    expected: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None
    tool_errors: list[str] = field(default_factory=list)   # failed tool calls in the final attempt ("name: error")
    retried: bool = False                                  # the agent was run a second time after a tool failure

    def to_dict(self) -> dict:
        return asdict(self)


def _usage(messages: list[Any]) -> dict[str, int]:
    inp = out = 0
    for m in messages:
        u = getattr(m, "usage_metadata", None) or {}
        inp += int(u.get("input_tokens", 0) or 0)
        out += int(u.get("output_tokens", 0) or 0)
    return {"input": inp, "output": out}


def run_question(
    q: EvalQuestion,
    model: str,
    judge: FaithfulnessJudge | None,
    agent_runner: Callable[..., dict[str, Any]] | None = None,
    grounding_llm_factory: Callable[..., Any] | None = None,
    grounding: bool = True,
    retry_on_tool_error: bool = True,
) -> QuestionResult:
    """Run one question through the agent exactly as the app does (minus the input guardrails), then score it.

    The app's grounding check + revision runs too (guardrails.grounding_check), so faithfulness is measured on the
    answer a user would actually see; the pre-revision answer is kept for the report.

    A tool that fails (an embedding/rerank timeout, a raised exception) leaves the agent answering blind; that is an
    infrastructure hiccup, not the behaviour under test, so the question is run once more. If the tool fails again,
    the error is kept and shown in the report (tool_errors), and a not-covered question counts as not handled."""
    res = QuestionResult(
        id=q.id, group=q.group, language=q.language, question=q.question, covered=q.covered,
        coverage_check=q.coverage_check, probe=q.is_probe, hard=q.is_hard,
        expected={"docs": q.expected_docs, "tools": q.expected_tools, "forbidden": q.forbidden_tools, "args": q.expected_args, "facts": q.expected_facts},
    )
    t0 = time.time()

    def _run() -> dict[str, Any]:
        if agent_runner is None:
            agent = build_agent(model, q.language, None)
            return run_agent_stream(agent, [], q.question)
        return agent_runner(q, model)

    try:
        payload = _run()
        first_errors = tool_errors(payload.get("tool_calls", []))
        if first_errors and retry_on_tool_error:
            log.warning(f"{q.id}: tool failure ({'; '.join(first_errors)[:200]}) — running the question again")
            payload = _run()
            res.retried = True
            res.notes.append("A tool failed in the first attempt (" + "; ".join(e[:120] for e in first_errors) + "); the question was run again.")
    except NavigatorError as exc:
        res.error = str(exc)
        res.seconds = round(time.time() - t0, 1)
        res.tools = asdict(check_tools(q, []))
        res.coverage = asdict(check_coverage(q, [], "", []))
        return res

    tool_calls = payload.get("tool_calls", [])
    used_search = any(tc.get("name") == "search_knowledge_base" for tc in tool_calls)
    checked = check_output(
        payload.get("text", ""), used_search, set(), q.language,
        valid_refs=valid_refs(tool_calls) if used_search else None, weak_coverage=weak_coverage(tool_calls),
    )
    res.answer = checked.text
    res.notes = res.notes + list(checked.notes)
    res.tool_errors = tool_errors(tool_calls)
    for e in res.tool_errors:
        res.notes.append(f"Tool failed: {e}")
    if grounding and tool_calls:  # also when coverage is weak — "general orientation" must not carry unsourced numbers
        g = apply_grounding(checked.text, tool_calls, q.language, llm_factory=grounding_llm_factory)
        res.grounding = g.to_dict()
        if g.revised:
            res.answer_original = checked.text
            res.answer = g.text
            checked.text = g.text
        if g.note:
            res.notes.append(g.note)
        if g.report.ignored:
            res.notes.append(f"Grounding guard: {len(g.report.ignored)} claim(s) flagged by the judge were kept because their figures appear in the evidence.")
        if g.revision_rejected:
            res.notes.append(f"Grounding guard: {g.revision_rejected} — original answer kept.")
    res.tool_calls = [{"name": tc.get("name"), "args": tc.get("args"), "ok": tool_error(tc) is None, "error": tool_error(tc),
                       "duration_s": tc.get("duration_s"), "result": tc.get("result")} for tc in tool_calls]
    # the runner only records ToolMessages; recover the arguments from the AIMessages
    _attach_args(res.tool_calls, payload.get("messages", []))
    res.sources = collect_sources(tool_calls)
    res.tokens = _usage(payload.get("messages", []))
    res.tools = asdict(check_tools(q, res.tool_calls))
    res.tools["reasons"] = ToolCheck(**{k: v for k, v in res.tools.items()}).reasons
    res.coverage = asdict(check_coverage(q, tool_calls, checked.text, checked.notes))
    if judge is not None:
        contexts = evidence_contexts(tool_calls)
        judged_text = strip_labelled_general_knowledge(strip_disclaimer(checked.text))
        fr: FaithfulnessResult = judge.score(q.question, judged_text, contexts) if contexts and judged_text else FaithfulnessResult(
            score=None, backend=judge.backend, judge_model=judge.model, error="no evidence retrieved (no successful tool call)")
        res.faithfulness = fr.to_dict()
    # "handled" for a not-covered question also requires that the final answer added no unsupported *fact*:
    # saying "the knowledge base does not cover this" and then listing amounts, documents or procedures from memory
    # is not handling it. Referrals ("ask the Familienkasse", a website) and meta statements are not facts and are
    # exactly what the answer should consist of, so they never count against it.
    if not q.covered:
        leaks = [c for c in res.faithfulness.get("claims", []) if c.get("kind", "fact") == "fact" and not c.get("verdict")]
        if leaks:
            res.coverage["handled"] = False
            msg = f"{len(leaks)} unsupported factual statement(s) added: " + "; ".join(f"“{c['statement'][:90]}”" for c in leaks[:3])
            res.coverage["handled_reason"] = "; ".join(x for x in (res.coverage.get("handled_reason"), msg) if x)
        elif judge is not None and res.coverage.get("handled") and res.faithfulness.get("score") is None:
            res.coverage["handled_note"] = "judge unavailable — factual leaks not checked"
    res.seconds = round(time.time() - t0, 1)
    return res


def _attach_args(tool_calls: list[dict[str, Any]], messages: list[Any]) -> None:
    by_id: dict[str, dict] = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            by_id[tc.get("id")] = tc.get("args") or {}
    ids = [tc.get("id") for tc in tool_calls]
    for tc in tool_calls:
        if not tc.get("args"):
            tc["args"] = by_id.get(tc.get("id"), tc.get("args") or {})
    del ids


# ----------------------------------------------------------------------------- whole run

@dataclass
class E2EReport:
    model: str
    judge_model: str | None
    judge_backend: str | None
    results: list[QuestionResult]
    seconds: float
    judge_fallback_reason: str | None = None

    def judge_backends(self) -> dict[str, int]:
        """How many questions each faithfulness backend actually scored (ragas / builtin / none)."""
        counts: dict[str, int] = {}
        for r in self.results:
            f = r.faithfulness or {}
            key = f.get("backend", "none") if f.get("score") is not None else "none"
            counts[key] = counts.get(key, 0) + 1
        return counts

    # --- aggregates -----------------------------------------------------------
    def tool_accuracy(self) -> tuple[int, int]:
        ok = [r for r in self.results if r.error is None]
        return sum(1 for r in ok if r.tools.get("passed")), len(self.results)

    def not_covered_handled(self) -> tuple[int, int]:
        nc = [r for r in self.results if not r.covered]
        return sum(1 for r in nc if r.coverage.get("handled")), len(nc)

    def mean_faithfulness(self, only: str | None = None) -> float | None:
        rows = [r for r in self.results if r.faithfulness.get("score") is not None]
        if only == "probe":
            rows = [r for r in rows if r.probe]
        elif only == "other":
            rows = [r for r in rows if not r.probe]
        if not rows:
            return None
        return round(sum(r.faithfulness["score"] for r in rows) / len(rows), 4)

    def tokens(self) -> dict[str, int]:
        return {"input": sum(r.tokens.get("input", 0) for r in self.results), "output": sum(r.tokens.get("output", 0) for r in self.results)}

    def to_dict(self) -> dict:
        ta, tn = self.tool_accuracy()
        nh, nn = self.not_covered_handled()
        return {
            "model": self.model, "judge_model": self.judge_model, "judge_backend": self.judge_backend,
            "judge_backends": self.judge_backends(), "judge_fallback_reason": self.judge_fallback_reason,
            "seconds": round(self.seconds, 1), "tokens": self.tokens(),
            "summary": {
                "questions": len(self.results),
                "errors": sum(1 for r in self.results if r.error),
                "tool_selection": {"correct": ta, "total": tn},
                "not_covered": {"handled": nh, "total": nn},
                "revised": sum(1 for r in self.results if r.answer_original),
                "revision_rejected": sum(1 for r in self.results if r.grounding.get("revision_rejected")),
                "guard_ignored_claims": sum(len(r.grounding.get("ignored") or []) for r in self.results),
                "tool_errors": sum(1 for r in self.results if r.tool_errors),
                "retried": sum(1 for r in self.results if r.retried),
                "grounding_checked": sum(1 for r in self.results if r.grounding.get("checked")),
                "faithfulness": {"mean": self.mean_faithfulness(), "probes": self.mean_faithfulness("probe"), "others": self.mean_faithfulness("other")},
            },
            "results": [r.to_dict() for r in self.results],
        }


def evaluate_e2e(
    questions: list[EvalQuestion],
    model: str,
    judge: FaithfulnessJudge | None,
    workers: int = 1,
    progress: Progress | None = None,
    agent_runner: Callable[..., dict[str, Any]] | None = None,
    grounding_llm_factory: Callable[..., Any] | None = None,
    grounding: bool = True,
) -> E2EReport:
    started = time.time()
    results: dict[str, QuestionResult] = {}

    def _one(q: EvalQuestion) -> QuestionResult:
        return run_question(q, model, judge, agent_runner=agent_runner, grounding_llm_factory=grounding_llm_factory, grounding=grounding)

    def _report(i: int, r: QuestionResult) -> None:
        if progress:
            f = r.faithfulness.get("score")
            tools = "✔" if r.tools.get("passed") else "✖ " + "; ".join(r.tools.get("reasons") or [])
            cov = r.coverage.get("coverage") or "no search"
            if not r.covered:
                cov += " (handled)" if r.coverage.get("handled") else " (expected weak)"
            backend = (r.faithfulness or {}).get("backend")
            fstr = f"{f} ({backend})" if f is not None and backend else ("n/a" if f is None else str(f))
            if r.answer_original:
                fstr += " ✂ revised"
            progress("e2e", i, len(questions), f"{r.id} {r.group:<12} tools {tools}  coverage {cov}  faithfulness {fstr}  ({r.seconds} s)" + (f"  ERROR {r.error}" if r.error else ""))

    if workers <= 1:
        for i, q in enumerate(questions, start=1):
            r = _one(q)
            results[q.id] = r
            _report(i, r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, q): q for q in questions}
            for i, fut in enumerate(as_completed(futures), start=1):
                r = fut.result()
                results[r.id] = r
                _report(i, r)

    ordered = [results[q.id] for q in questions]
    return E2EReport(model=model, judge_model=judge.model if judge else None, judge_backend=judge.backend if judge else None,
                     results=ordered, seconds=time.time() - started,
                     judge_fallback_reason=getattr(judge, "fallback_reason", None) if judge else None)
