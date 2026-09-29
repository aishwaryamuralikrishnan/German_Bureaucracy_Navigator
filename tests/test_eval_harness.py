"""Evaluation harness: question set, deterministic checks, faithfulness judge (fake LLM), report + PDF, store."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from navigator.eval import store
from navigator.eval.e2e_eval import (
    check_coverage, check_tools, evaluate_e2e, evidence_contexts, strip_disclaimer,
)
from navigator.eval.faithfulness import FaithfulnessJudge
from navigator.eval.questions import DEFAULT_QUESTIONS, FULL_QUESTIONS, EvalQuestion, QuestionSetError, load_questions, validate_questions
from navigator.eval.report import build_pdf, group_tool_stats, summary_markdown, tool_matrix
from navigator.guardrails.output_checks import DISCLAIMERS
from tests.fakes import ScriptedToolModel


# ----------------------------------------------------------------------------- question set

def test_default_and_full_sets_load_and_validate():
    short = load_questions(DEFAULT_QUESTIONS)
    full = load_questions(FULL_QUESTIONS)
    assert len(short) == 23 and len(full) == 31
    assert {q.id for q in short} <= {q.id for q in full}
    assert {q.id for q in full} - {q.id for q in short} == {"Q09", "Q10", "Q14", "Q15", "Q17", "Q21", "Q23", "Q29"}
    assert sum(1 for q in short if not q.covered) == 3
    assert load_questions(DEFAULT_QUESTIONS, ids=["q11", "Q18"]) and load_questions(DEFAULT_QUESTIONS, limit=2)[1].id == "Q02"
    with pytest.raises(QuestionSetError):
        load_questions(DEFAULT_QUESTIONS, ids=["Q99"])


def test_not_covered_robustness_set_loads():
    """Three fresh not-covered questions (Elterngeld, Hundesteuer in German, pension refund as a hard probe)."""
    qs = load_questions(DEFAULT_QUESTIONS.with_name("questions_not_covered_v2.yaml"))
    validate_questions(qs)
    assert [q.id for q in qs] == ["Q32", "Q33", "Q34"] and all(not q.covered and q.coverage_check for q in qs)
    assert qs[1].language == "de" and qs[2].is_hard and not qs[0].is_hard
    assert not {q.id for q in qs} & {q.id for q in load_questions(FULL_QUESTIONS)}  # IDs do not collide with the main sets
    v3 = load_questions(DEFAULT_QUESTIONS.with_name("questions_not_covered_v3.yaml"))
    validate_questions(v3)
    assert [q.id for q in v3] == ["Q35", "Q36", "Q37"] and v3[1].is_hard and v3[2].language == "de"
    assert "estimate_net_salary" in v3[0].forbidden_tools and all(not q.covered for q in v3)


def test_validation_catches_label_mistakes():
    base = dict(id="X1", question="q", group="mixed", covered=True, expected_docs=["anmeldung.md"], expected_tools=["search_knowledge_base", "calculate_deadline"])
    validate_questions([EvalQuestion(**base)])
    with pytest.raises(QuestionSetError, match="unknown file"):
        validate_questions([EvalQuestion(**{**base, "expected_docs": ["nope.md"]})])
    with pytest.raises(QuestionSetError, match="no argument"):
        validate_questions([EvalQuestion(**{**base, "expected_args": {"calculate_deadline": {"colour": "red"}}})])
    with pytest.raises(QuestionSetError, match="coverage_check"):
        validate_questions([EvalQuestion(**{**base, "expected_tools": ["calculate_deadline"]})])
    with pytest.raises(QuestionSetError, match="both expected and forbidden"):
        validate_questions([EvalQuestion(**{**base, "forbidden_tools": ["calculate_deadline"]})])


# ----------------------------------------------------------------------------- deterministic checks

Q = EvalQuestion(
    id="T1", question="q", group="mixed", covered=True,
    expected_docs=["blue_card.md"], expected_tools=["search_knowledge_base", "check_blue_card_salary", "calculate_deadline"],
    forbidden_tools=["convert_currency"],
    expected_args={"check_blue_card_salary": {"gross_annual_salary_eur": 50000}, "calculate_deadline": {"event": "residence_permit_expiry", "event_date": "2027-01-15", "city": "any"}},
)


def _call(name, args=None, result=None):
    return {"name": name, "args": args or {}, "result": result or {"ok": True, "data": {}}}


def test_check_tools_passes_when_everything_matches():
    tc = check_tools(Q, [_call("search_knowledge_base", {"query": "x"}), _call("check_blue_card_salary", {"gross_annual_salary_eur": 50000.0, "year": 2026}),
                         _call("calculate_deadline", {"event": "Residence_Permit_Expiry", "event_date": "2027-01-15", "city": "Berlin"})])
    assert tc.passed and not tc.reasons


def test_check_tools_reports_missing_forbidden_and_args():
    tc = check_tools(Q, [_call("search_knowledge_base"), _call("check_blue_card_salary", {"gross_annual_salary_eur": 49000}), _call("convert_currency", {"amount": 1})])
    assert not tc.passed
    assert tc.missing == ["calculate_deadline"] and tc.forbidden_called == ["convert_currency"]
    assert tc.arg_mismatches == ["check_blue_card_salary.gross_annual_salary_eur: expected 50000, got 49000"]
    assert any("missing calculate_deadline" in r for r in tc.reasons)


def test_check_tools_requires_search_when_coverage_checked():
    tc = check_tools(EvalQuestion(id="T2", question="q", group="tool_only", covered=True, coverage_check=True, expected_tools=["search_knowledge_base"]), [_call("estimate_net_salary")])
    assert not tc.passed and tc.search_missing
    tc2 = check_tools(EvalQuestion(id="T3", question="q", group="tool_only", covered=True, coverage_check=False, expected_tools=["estimate_net_salary"]), [_call("estimate_net_salary")])
    assert tc2.passed


def test_check_coverage_for_not_covered_question():
    nc = EvalQuestion(id="N1", question="q", group="not_covered", covered=False, expected_tools=["search_knowledge_base"])
    weak = [_call("search_knowledge_base", result={"ok": True, "data": {"coverage": "weak", "passages": []}})]
    ok = check_coverage(nc, weak, "Not covered.", ["The knowledge base has no specific passage on this question…"])
    assert ok.coverage == "weak" and ok.notice_shown and ok.citations == 0 and ok.handled is True
    bad = check_coverage(nc, [_call("search_knowledge_base", result={"ok": True, "data": {"coverage": "ok"}})], "It costs €18.36 [S1].", [])
    assert bad.coverage == "ok" and bad.citations == 1 and bad.handled is False and "detector missed it" in bad.handled_reason
    cited = check_coverage(nc, weak, "It costs €18.36 [S1].", [])
    assert cited.handled is False and "citation(s) despite weak coverage" in cited.handled_reason
    none = check_coverage(nc, [], "…", [])
    assert none.coverage is None and none.label == "no search" and none.handled is False and none.handled_reason == "no knowledge-base search"
    covered = check_coverage(Q, weak, "x", [])
    assert covered.handled is None and covered.handled_reason is None  # not a metric for covered questions
    # a "hard" probe is one the KB mentions in passing: an "ok" flag (and a citation of that passage) is fine there
    hard = EvalQuestion(id="N2", question="q", group="not_covered", covered=False, difficulty="hard", expected_tools=["search_knowledge_base"])
    hok = check_coverage(hard, [_call("search_knowledge_base", result={"ok": True, "data": {"coverage": "ok"}})], "Every household pays it [S1]. The KB does not state the amount.", [])
    assert hok.coverage == "ok" and hok.handled is True and hok.handled_reason is None


def test_evidence_and_disclaimer_helpers():
    calls = [_call("search_knowledge_base", result={"ok": True, "data": {"passages": [{"ref": "S1", "title": "T", "section": "Sec", "text": "Body"}]}}),
             _call("convert_currency", {"amount": 1}, {"ok": True, "data": {"to": {"amount": 2}}, "warnings": ["w"]}),
             _call("calculate_deadline", {}, {"ok": False, "error": "bad"})]
    ev = evidence_contexts(calls)
    assert len(ev) == 2 and ev[0].startswith("[S1] T — Sec") and "convert_currency" in ev[1]
    assert strip_disclaimer(f"Answer.\n\n_{DISCLAIMERS['en']}_") == "Answer."


# ----------------------------------------------------------------------------- faithfulness (built-in backend, fake LLM)

def _judge(*replies: str) -> FaithfulnessJudge:
    model = ScriptedToolModel(script=[AIMessage(content=r) for r in replies])
    return FaithfulnessJudge(model="fake/judge", llm_factory=lambda *a, **k: model)


def test_builtin_faithfulness_scores_claims():
    j = _judge('{"statements": ["Fee is 29 EUR", "Valid for one year"]}',
               '{"verdicts": [{"statement": "Fee is 29 EUR", "verdict": 1, "reason": "in S1"}, {"statement": "Valid for one year", "verdict": 0, "reason": "not in context"}]}',
               '{"kinds": ["fact", "fact"]}')
    assert j.backend == "builtin"
    r = j.score("q", "The fee is 29 EUR. It is valid for one year.", ["[S1] fee EUR 29"])
    assert r.score == 0.5 and r.supported == 1 and len(r.claims) == 2 and r.claims[1].verdict == 0
    assert r.classifier == "llm" and r.leaks == [r.claims[1]]
    d = r.to_dict()
    assert d["total"] == 2 and d["backend"] == "builtin" and d["unscored"] == 0 and d["raw_score"] == 0.5


def test_referral_and_meta_statements_are_not_scored():
    """The honest answer to an uncovered question is all referral/meta — it must score 1.0, not 0.0."""
    j = _judge('{"statements": ["The knowledge base does not cover Kindergeld", "The user should contact the Familienkasse", "Kindergeld is 250 EUR per child"]}',
               '{"verdicts": [{"statement": "The knowledge base does not cover Kindergeld", "verdict": 1, "reason": ""}, {"statement": "The user should contact the Familienkasse", "verdict": 0, "reason": "not in context"}, {"statement": "Kindergeld is 250 EUR per child", "verdict": 0, "reason": "not in context"}]}',
               '{"kinds": ["meta", "referral", "fact"]}')
    r = j.score("Kindergeld?", "…", ["[S1] unrelated"])
    assert [c.kind for c in r.claims] == ["meta", "referral", "fact"]
    assert r.score == 0.0 and r.total == 1 and r.unscored == 2 and r.raw_score == 0.3333 and [c.statement for c in r.leaks] == ["Kindergeld is 250 EUR per child"]
    # only referral + meta → nothing is presented as knowledge-base-backed → 1.0
    j2 = _judge('{"statements": ["The knowledge base does not cover this", "Ask the Familienkasse"]}',
                '{"verdicts": [{"statement": "The knowledge base does not cover this", "verdict": 1, "reason": ""}, {"statement": "Ask the Familienkasse", "verdict": 0, "reason": ""}]}',
                '{"kinds": ["meta", "referral"]}')
    r2 = j2.score("q", "…", ["ctx"])
    assert r2.score == 1.0 and r2.total == 0 and r2.unscored == 2 and r2.to_dict()["statements"] == 2


def test_claim_kind_falls_back_to_regex_when_classifier_fails():
    from navigator.eval.faithfulness import heuristic_kind
    j = _judge('{"statements": ["Contact the Familienkasse for details", "The fee is 29 EUR"]}',
               '{"verdicts": [{"statement": "Contact the Familienkasse for details", "verdict": 0, "reason": ""}, {"statement": "The fee is 29 EUR", "verdict": 1, "reason": ""}]}',
               '{"kinds": ["fact"]}')  # wrong length → heuristic
    r = j.score("q", "…", ["ctx"])
    assert r.classifier == "heuristic" and [c.kind for c in r.claims] == ["referral", "fact"] and r.score == 1.0
    assert heuristic_kind("The knowledge base does not state how long the letter is valid.") == "meta"
    assert heuristic_kind("Die Wissensdatenbank enthält keine Angaben zur Höhe.") == "meta"
    assert heuristic_kind("More information is available on the official website: https://www.arbeitsagentur.de") == "referral"
    assert heuristic_kind("The responsible authority should be consulted for any updates.") == "referral"
    assert heuristic_kind("Wenden Sie sich an die zuständige Ausländerbehörde.") == "referral"
    assert heuristic_kind("Proof of the children's birth is a required document.") == "fact"
    assert heuristic_kind("Apply at the Familienkasse with the birth certificate within 6 months.") == "fact"
    assert heuristic_kind("Every household must pay the Rundfunkbeitrag.") == "fact"


def test_builtin_faithfulness_fails_soft():
    r = _judge("garbage").score("q", "answer", ["ctx"])
    assert r.score is None and r.error
    r2 = _judge('{"statements": []}').score("q", "Hello!", ["ctx"])
    assert r2.score is None and "no statements" in r2.error
    r3 = _judge("x").score("q", "answer", [])
    assert r3.score is None and "empty" in r3.error


# ----------------------------------------------------------------------------- end-to-end run with a fake agent + report

def _fake_runner(q: EvalQuestion, model: str) -> dict:
    """Pretend to be run_agent_stream: search + the expected calculator, one deliberately wrong question."""
    calls = []
    passages = [{"ref": "S1", "title": "Doc", "section": "Sec", "text": "Registration within 14 days."}]
    coverage = "weak" if not q.covered else "ok"
    def _args(tool):  # expected arguments, with "any" replaced by a concrete value
        return {k: ("X" if v == "any" else v) for k, v in (q.expected_args.get(tool) or {}).items()}
    calls.append({"name": "search_knowledge_base", "id": "c1", "args": {"query": q.question[:30], **_args("search_knowledge_base")}, "duration_s": 0.1,
                  "result": {"ok": True, "data": {"coverage": coverage, "passages": passages if q.covered else []},
                             "sources": [{"ref": "S1", "title": "Doc", "section": "Sec"}] if q.covered else []}})
    for tool in q.expected_tools:
        if tool == "search_knowledge_base" or q.id == "Q02":
            continue
        calls.append({"name": tool, "id": tool, "args": _args(tool),
                      "duration_s": 0.1, "result": {"ok": True, "data": {"value": 1}}})
    text = "You must register within 14 days [S1]." if q.covered else "The knowledge base does not cover this; ask the Familienkasse."
    if q.id == "Q01":
        text += " Students may work 120 days."
    return {"text": text, "tool_calls": calls, "messages": []}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path / "eval")
    return store.new_run_dir()


def test_evaluate_e2e_with_fake_agent_and_judge(run_dir):
    questions = load_questions(DEFAULT_QUESTIONS, ids=["Q01", "Q04", "Q07", "Q27", "Q30"])
    # one scripted judge reply pair per question, cycled: everything supported except one claim
    replies = ['{"statements": ["Register within 14 days", "Students may work 120 days"]}',
               '{"verdicts": [{"statement": "Register within 14 days", "verdict": 1, "reason": "S1"}, {"statement": "Students may work 120 days", "verdict": 0, "reason": "absent"}]}',
               '{"kinds": ["fact", "fact"]}'] * 5
    judge = _judge(*replies)
    # app-side grounding check: nothing unsupported for 4 questions, one unsupported claim for Q01 → revised
    grounding_model = ScriptedToolModel(script=[AIMessage(content='{"unsupported": ["Students may work 120 days"]}'),
                                                AIMessage(content="You must register within 14 days [S1]. Not from the knowledge base: the working-day limit is not covered here.")]
                                               + [AIMessage(content='{"unsupported": []}')] * 10)
    rep = evaluate_e2e(questions, "fake/agent", judge, agent_runner=_fake_runner, grounding_llm_factory=lambda *a, **k: grounding_model)
    d = rep.to_dict()
    ids = [r["id"] for r in d["results"]]
    assert ids == ["Q01", "Q04", "Q07", "Q27", "Q30"]
    by = {r["id"]: r for r in d["results"]}
    assert by["Q04"]["tools"]["passed"] is True                 # search + deadline with expected args ("any" city)
    assert by["Q07"]["tools"]["passed"] is True                 # three tools present
    assert by["Q27"]["coverage"]["handled"] is True and by["Q30"]["coverage"]["handled"] is True
    assert d["summary"]["not_covered"] == {"handled": 2, "total": 2}
    assert d["summary"]["tool_selection"]["total"] == 5
    assert by["Q01"]["faithfulness"]["score"] == 0.5 and d["summary"]["faithfulness"]["mean"] == 0.5
    assert by["Q04"]["answer"].endswith("[S1].") and by["Q04"]["sources"][0]["ref"] == "S1"
    # Q01 was revised by the grounding step: the 120-day claim is gone, the original is kept for the report
    assert by["Q01"]["grounding"]["revised"] is True and "120 days" in by["Q01"]["answer_original"] and "120 days" not in by["Q01"]["answer"]
    assert by["Q01"]["answer"].startswith("You must register within 14 days [S1].")
    assert d["summary"]["revised"] == 1 and any(n.startswith("The answer was revised") for n in by["Q01"]["notes"])

    # store + reports
    meta = store.update_meta(run_dir, run=run_dir.name, question_file="data/eval/questions.yaml", questions=5)
    store.write_json(run_dir / "e2e.json", d)
    md = summary_markdown(meta, None, d)
    assert "Tool selection" in md and "Q27" in md
    pdf = build_pdf(run_dir, meta, None, d)
    assert pdf.exists() and pdf.stat().st_size > 5000
    runs = store.list_runs()
    assert len(runs) == 1 and runs[0].e2e["summary"]["questions"] == 5 and runs[0].retrieval is None
    stats = {g["group"]: g for g in group_tool_stats(d)}
    assert stats["mixed"]["questions"] == 2 and stats["not_covered"]["correct"] == 2
    assert all(f["cells"]["search_knowledge_base"] == "ok" for f in tool_matrix(d, failing_only=False))


def test_summary_and_pdf_with_retrieval_only(run_dir):
    retrieval = {"k": 5, "seconds": 3.0, "reranker": "noop", "embedding_model": "hash",
                 "configs": [{"config": c, "label": lab, "questions": 2, "hits": 1, "hit_rate": 0.5, "mrr": 0.25} for c, lab in
                             (("dense", "dense only"), ("bm25", "BM25 only"), ("hybrid", "hybrid (RRF)"), ("hybrid_rerank", "hybrid + reranker"))],
                 "per_question": [{"id": "Q01", "group": "knowledge", "expected_docs": ["a.md"], "ranks": {"dense": 1, "bm25": None, "hybrid": 1, "hybrid_rerank": 1}, "top_docs": {}, "variants": ["q"], "seconds": 1.0}],
                 "skipped": []}
    meta = store.update_meta(run_dir, run=run_dir.name, question_file="x.yaml", questions=1)
    store.write_json(run_dir / "retrieval.json", retrieval)
    assert "hit-rate" in summary_markdown(meta, retrieval, None)
    assert build_pdf(run_dir, meta, retrieval, None).exists()
    assert json.loads((run_dir / "retrieval.json").read_text())["k"] == 5



# ----------------------------------------------------------------------------- grounding revision (fake LLM)

def test_apply_grounding_revises_unsupported_claims():
    from navigator.guardrails.grounding import apply_grounding
    calls = [{"name": "search_knowledge_base", "args": {}, "result": {"ok": True, "data": {"passages": [{"ref": "S1", "title": "T", "section": "S", "text": "Abmeldung ends the Rundfunkbeitrag."}]}}}]
    model = ScriptedToolModel(script=[AIMessage(content='{"unsupported": ["The fee is 18.36 EUR per month"]}'),
                                      AIMessage(content="Every household pays the Rundfunkbeitrag; Abmeldung ends it [S1]. The knowledge base does not state the amount.")])
    out = apply_grounding("Every household pays the Rundfunkbeitrag; Abmeldung ends it [S1]. The fee is 18.36 EUR per month.", calls, "en", llm_factory=lambda *a, **k: model)
    assert out.revised and "18.36" not in out.text and out.original_text and "18.36" in out.original_text
    assert out.note and out.note.startswith("The answer was revised: 1 statement")
    assert out.to_dict()["unsupported"] == ["The fee is 18.36 EUR per month"]


def test_apply_grounding_leaves_grounded_answer_alone():
    from navigator.guardrails.grounding import apply_grounding
    calls = [{"name": "search_knowledge_base", "args": {}, "result": {"ok": True, "data": {"passages": [{"ref": "S1", "title": "T", "section": "S", "text": "x"}]}}}]
    model = ScriptedToolModel(script=[AIMessage(content='{"unsupported": []}')])
    out = apply_grounding("All good [S1].", calls, "en", llm_factory=lambda *a, **k: model)
    assert not out.revised and out.text == "All good [S1]." and out.note is None and out.report.ok


def test_apply_grounding_keeps_answer_when_revision_is_garbage():
    from navigator.guardrails.grounding import apply_grounding
    calls = [{"name": "search_knowledge_base", "args": {}, "result": {"ok": True, "data": {"passages": [{"ref": "S1", "title": "T", "section": "S", "text": "x"}]}}}]
    model = ScriptedToolModel(script=[AIMessage(content='{"unsupported": ["claim"]}'), AIMessage(content="ok")])
    out = apply_grounding("A long enough answer with a claim in it [S1].", calls, "en", llm_factory=lambda *a, **k: model)
    assert not out.revised and out.text.startswith("A long enough") and out.note and out.note.startswith("Not backed by")


# ----------------------------------------------------------------------------- not-covered handling is strict

def test_labelled_general_knowledge_is_not_judged():
    from navigator.eval.e2e_eval import strip_labelled_general_knowledge
    text = "The knowledge base does not cover Kindergeld.\nNot from the knowledge base: ask the Familienkasse.\n**Nicht aus der Wissensdatenbank:** Hinweis.\nConfirm with the authority."
    assert strip_labelled_general_knowledge(text) == "The knowledge base does not cover Kindergeld.\nConfirm with the authority."


def test_not_covered_question_is_not_handled_when_unsupported_facts_remain():
    """Weak flag + no citations used to be enough; an answer that adds '€219 per child' from memory fails, while the
    referral sentence ('ask the Familienkasse') never counts against it."""
    from navigator.eval.e2e_eval import run_question
    q = load_questions(DEFAULT_QUESTIONS, ids=["Q27"])[0]

    def runner(q, model):
        return {"text": "The knowledge base does not cover Kindergeld. The rate is typically around €219 per child. Ask the Familienkasse.",
                "tool_calls": [{"name": "search_knowledge_base", "id": "c1", "args": {"query": "Kindergeld"},
                                "result": {"ok": True, "data": {"coverage": "weak", "passages": [{"ref": "S1", "title": "T", "section": "S", "text": "unrelated"}]}, "sources": []}}],
                "messages": []}
    judge = _judge('{"statements": ["The knowledge base does not cover Kindergeld", "The rate is around 219 EUR per child", "The user should ask the Familienkasse"]}',
                   '{"verdicts": [{"statement": "The knowledge base does not cover Kindergeld", "verdict": 1, "reason": "ok"}, {"statement": "The rate is around 219 EUR per child", "verdict": 0, "reason": "absent"}, {"statement": "The user should ask the Familienkasse", "verdict": 0, "reason": "absent"}]}',
                   '{"kinds": ["meta", "fact", "referral"]}')
    r = run_question(q, "fake", judge, agent_runner=runner, grounding=False)
    assert r.coverage["coverage"] == "weak" and r.coverage["citations"] == 0
    assert r.coverage["handled"] is False and r.coverage["handled_reason"].startswith("1 unsupported factual statement(s) added: “The rate is around 219 EUR per child”")
    assert r.faithfulness["score"] == 0.0 and r.faithfulness["total"] == 1 and r.faithfulness["unscored"] == 2
    # …and with a grounding revision that removes the figure, the same question is handled: only meta + referral remain
    grounding_model = ScriptedToolModel(script=[AIMessage(content='{"unsupported": ["The rate is typically around €219 per child"]}'),
                                                AIMessage(content="The knowledge base does not cover Kindergeld. Ask the Familienkasse (www.arbeitsagentur.de).")])
    judge2 = _judge('{"statements": ["The knowledge base does not cover Kindergeld", "The user should ask the Familienkasse", "The Familienkasse website is www.arbeitsagentur.de"]}',
                    '{"verdicts": [{"statement": "The knowledge base does not cover Kindergeld", "verdict": 1, "reason": "ok"}, {"statement": "The user should ask the Familienkasse", "verdict": 0, "reason": "absent"}, {"statement": "The Familienkasse website is www.arbeitsagentur.de", "verdict": 0, "reason": "absent"}]}',
                    '{"kinds": ["meta", "referral", "referral"]}')
    r2 = run_question(q, "fake", judge2, agent_runner=runner, grounding_llm_factory=lambda *a, **k: grounding_model)
    assert r2.answer_original and "219" not in r2.answer and r2.coverage["handled"] is True and r2.coverage.get("handled_reason") is None
    assert r2.faithfulness["score"] == 1.0 and r2.faithfulness["total"] == 0 and r2.faithfulness["raw_score"] == 0.3333


def test_office_gloss_statements_count_as_referral():
    """RAGAS atomises 'contact the Elterngeldstelle (parental allowance office)' into 'the responsible authority is the
    Elterngeldstelle' + 'the Elterngeldstelle is the parental allowance office' — referral content even if the classifier says fact."""
    j = _judge('{"statements": ["The responsible authority is the Elterngeldstelle in the user\'s area", "The Elterngeldstelle is the parental allowance office", "The Bürgeramt is the citizens\' office", "Elterngeld is paid for 14 months"]}',
               '{"verdicts": [{"verdict": 0, "reason": ""}, {"verdict": 0, "reason": ""}, {"verdict": 0, "reason": ""}, {"verdict": 0, "reason": ""}]}',
               '{"kinds": ["fact", "fact", "fact", "fact"]}')
    r = j.score("Elterngeld?", "…", ["ctx"])
    assert [c.kind for c in r.claims] == ["referral", "referral", "referral", "fact"] and r.score == 0.0 and r.total == 1


def test_failed_search_tool_is_reported_and_question_is_rerun_once():
    from navigator.eval.e2e_eval import run_question
    q = load_questions(DEFAULT_QUESTIONS, ids=["Q27"])[0]
    attempts = {"n": 0}

    def flaky_runner(q, model):
        attempts["n"] += 1
        if attempts["n"] == 1:  # LangChain-style raw error string from a raised exception
            return {"text": "The knowledge base does not cover this. Ask the Familienkasse.", "messages": [],
                    "tool_calls": [{"name": "search_knowledge_base", "id": "c1", "args": {"query": "Kindergeld"}, "result": "Error: ReadTimeout\n Please fix your mistakes."}]}
        return {"text": "The knowledge base does not cover Kindergeld. Ask the Familienkasse.", "messages": [],
                "tool_calls": [{"name": "search_knowledge_base", "id": "c2", "args": {"query": "Kindergeld"},
                                "result": {"ok": True, "data": {"coverage": "weak", "passages": [{"ref": "S1", "title": "T", "section": "S", "text": "unrelated"}]}, "sources": []}}]}
    r = run_question(q, "fake", None, agent_runner=flaky_runner, grounding=False)
    assert attempts["n"] == 2 and r.retried and not r.tool_errors and r.coverage["handled"] is True
    assert any(n.startswith("A tool failed in the first attempt") for n in r.notes)

    def dead_runner(q, model):
        return {"text": "Die Wissensdatenbank enthält keine Informationen.", "messages": [],
                "tool_calls": [{"name": "search_knowledge_base", "id": "c1", "args": {"query": "Hund"}, "result": {"ok": False, "error": "embedding request timed out"}}]}
    r2 = run_question(q, "fake", None, agent_runner=dead_runner, grounding=False)
    assert r2.retried and r2.tool_errors == ["search_knowledge_base: embedding request timed out"]
    assert r2.coverage["coverage"] is None and r2.coverage["handled"] is False and r2.coverage["handled_reason"].startswith("search_knowledge_base failed: embedding request timed out")
    assert r2.tool_calls[0]["ok"] is False and r2.tool_calls[0]["error"] == "embedding request timed out"
    r3 = run_question(q, "fake", None, agent_runner=dead_runner, grounding=False, retry_on_tool_error=False)
    assert not r3.retried


def test_hard_probe_is_handled_when_no_fact_leaks_even_with_ok_coverage():
    """Q30: the KB mentions the Rundfunkbeitrag in passing, so the flag is 'ok' and citing that passage is right;
    the question is handled as long as no unsupported fact (the €18.36) is added."""
    from navigator.eval.e2e_eval import run_question
    q = load_questions(DEFAULT_QUESTIONS, ids=["Q30"])[0]
    assert q.is_hard

    def runner(q, model):
        return {"text": "Every household pays the Rundfunkbeitrag; the fee office writes to you after the Anmeldung [S1]. The knowledge base does not state the amount — see rundfunkbeitrag.de.",
                "tool_calls": [{"name": "search_knowledge_base", "id": "c1", "args": {"query": "Rundfunkbeitrag"},
                                "result": {"ok": True, "data": {"coverage": "ok", "passages": [{"ref": "S1", "title": "Anmeldung", "section": "Consequences", "text": "Rundfunkbeitrag: every household pays the broadcasting fee; the fee office is informed by the registration office."}]},
                                           "sources": [{"ref": "S1", "title": "Anmeldung", "section": "Consequences"}]}}],
                "messages": []}
    judge = _judge('{"statements": ["Every household pays the Rundfunkbeitrag", "The fee office writes after the Anmeldung", "The knowledge base does not state the amount", "Information is on rundfunkbeitrag.de"]}',
                   '{"verdicts": [{"statement": "Every household pays the Rundfunkbeitrag", "verdict": 1, "reason": "S1"}, {"statement": "The fee office writes after the Anmeldung", "verdict": 1, "reason": "S1"}, {"statement": "The knowledge base does not state the amount", "verdict": 1, "reason": ""}, {"statement": "Information is on rundfunkbeitrag.de", "verdict": 0, "reason": "absent"}]}',
                   '{"kinds": ["fact", "fact", "meta", "referral"]}')
    r = run_question(q, "fake", judge, agent_runner=runner, grounding=False)
    assert r.coverage["coverage"] == "ok" and r.coverage["citations"] == 1 and r.coverage["handled"] is True
    assert r.faithfulness["score"] == 1.0 and r.faithfulness["total"] == 2
    # the same answer with the amount from memory is not handled
    judge_leak = _judge('{"statements": ["Every household pays the Rundfunkbeitrag", "The fee is 18.36 EUR per month"]}',
                        '{"verdicts": [{"statement": "Every household pays the Rundfunkbeitrag", "verdict": 1, "reason": "S1"}, {"statement": "The fee is 18.36 EUR per month", "verdict": 0, "reason": "absent"}]}',
                        '{"kinds": ["fact", "fact"]}')
    r2 = run_question(q, "fake", judge_leak, agent_runner=runner, grounding=False)
    assert r2.coverage["handled"] is False and "18.36" in r2.coverage["handled_reason"]


# ----------------------------------------------------------------------------- cost line

def test_cost_label_prefers_measured_credits_and_falls_back_to_estimate():
    from navigator.eval.cost import cost_label, estimate_from_tokens
    e2e = {"model": "openai/gpt-4o-mini", "tokens": {"input": 1_000_000, "output": 100_000}}
    assert estimate_from_tokens(e2e) == 0.21                      # 1M × 0.15 + 0.1M × 0.60
    assert estimate_from_tokens({"model": "unknown/model", "tokens": {"input": 5}}) is None
    assert cost_label({"cost_usd": 0.2857, "cost_source": "openrouter_key_usage"}, e2e).startswith("total cost $0.29 (OpenRouter credits")
    assert cost_label({"cost_usd": 0.21, "cost_source": "estimate_agent_tokens"}, e2e).startswith("cost ≈ $0.21 (estimated")
    assert cost_label({}, e2e).startswith("cost ≈ $0.21")
    assert cost_label({}, None) == "cost n/a"
