"""Retrieval evaluation on a temporary index with fake embeddings, and the Evaluation page rendering a saved run."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from navigator.eval import store
from navigator.eval.questions import DEFAULT_QUESTIONS, load_questions
from navigator.eval.retrieval_eval import CONFIGS, evaluate_retrieval
from navigator.eval.report import build_pdf, summary_markdown

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "2_Evaluation.py")


@pytest.mark.slow
def test_retrieval_eval_on_temp_index(tmp_path, monkeypatch):
    from fakes import HashEmbeddings
    from navigator.config import get_settings
    from navigator.rag import vectorstore as vs
    from navigator.rag.ingest import ingest
    from navigator.rag.rerank import NoOpReranker
    from navigator.rag.retriever import KnowledgeBaseRetriever

    monkeypatch.setattr(vs, "resolve_path", lambda _rel: tmp_path / "chroma")
    get_settings.cache_clear()
    ingest(embeddings=HashEmbeddings())
    retriever = KnowledgeBaseRetriever(embeddings=HashEmbeddings(), query_expansion=False, reranker=NoOpReranker())

    questions = load_questions(DEFAULT_QUESTIONS, ids=["Q02", "Q05", "Q22", "Q27"])   # Q27 has no expected document
    seen = []
    rep = evaluate_retrieval(questions, retriever, k=5, progress=lambda st, i, n, msg: seen.append(msg))
    d = rep.to_dict()
    assert d["skipped"] == ["Q27"] and len(d["per_question"]) == 3 and len(seen) == 3
    assert [c["config"] for c in d["configs"]] == list(CONFIGS)
    for c in d["configs"]:
        assert 0.0 <= c["hit_rate"] <= 1.0 and 0.0 <= c["mrr"] <= c["hit_rate"] + 1e-9 and c["questions"] == 3
    q02 = next(q for q in d["per_question"] if q["id"] == "Q02")
    # the Verpflichtungserklärung passage shares rare keywords with the question → BM25 must find it in the top 5
    assert q02["ranks"]["bm25"] is not None and q02["ranks"]["hybrid"] is not None
    assert all(len(v) <= 5 for v in q02["top_docs"].values()) and q02["variants"] == [questions[0].question]
    # noop reranker → hybrid and hybrid_rerank rank the same documents
    assert q02["ranks"]["hybrid"] == q02["ranks"]["hybrid_rerank"]


def _synthetic_run(root: Path) -> Path:
    run = store.new_run_dir(root)
    meta = store.update_meta(run, run=run.name, question_file="data/eval/questions.yaml", questions=3)
    retrieval = {"k": 5, "seconds": 2.0, "reranker": "cohere/rerank-4-fast", "embedding_model": "openai/text-embedding-3-small",
                 "configs": [{"config": c, "label": lab, "questions": 2, "hits": h, "hit_rate": h / 2, "mrr": m} for (c, lab), h, m in
                             zip(CONFIGS.items(), (1, 1, 2, 2), (0.5, 0.25, 0.75, 1.0))],
                 "per_question": [{"id": "Q02", "group": "knowledge", "expected_docs": ["students_blocked_account.md"], "ranks": {"dense": 2, "bm25": 1, "hybrid": 1, "hybrid_rerank": 1}, "top_docs": {}, "variants": ["v"], "seconds": 1},
                                  {"id": "Q04", "group": "mixed", "expected_docs": ["anmeldung_munich.md"], "ranks": {"dense": None, "bm25": None, "hybrid": 3, "hybrid_rerank": 1}, "top_docs": {}, "variants": ["v"], "seconds": 1}],
                 "skipped": ["Q27"]}
    def res(id_, group, covered, passed, score, claims, handled=None, coverage="ok", reasons=None):
        return {"id": id_, "group": group, "language": "en", "question": f"question {id_}", "covered": covered, "coverage_check": True, "probe": id_ == "Q02", "hard": id_ == "Q30",
                "answer": "Answer text [S1].", "tool_calls": [{"name": "search_knowledge_base", "args": {"query": "q"}, "ok": True, "duration_s": 0.1, "result": {"ok": True, "data": {"coverage": coverage, "passages": []}}}],
                "sources": [{"ref": "S1", "title": "Doc", "section": "Sec", "url": None, "weak": False, "similarity": 0.9}], "notes": [],
                "tools": {"passed": passed, "called": ["search_knowledge_base"], "missing": [] if passed else ["calculate_deadline"], "forbidden_called": [], "arg_mismatches": [], "search_missing": False, "reasons": reasons or ([] if passed else ["missing calculate_deadline"])},
                "coverage": {"coverage": coverage, "notice_shown": not covered, "citations": 1, "handled": handled},
                "faithfulness": {"score": score, "supported": sum(c["verdict"] for c in claims), "total": len(claims), "backend": "builtin", "judge_model": "j", "error": None, "claims": claims},
                "expected": {"docs": [], "tools": ["search_knowledge_base", "calculate_deadline"], "forbidden": [], "args": {}, "facts": ["fact"]}, "tokens": {"input": 10, "output": 5}, "seconds": 3.0, "error": None}
    claims_ok = [{"statement": "s1", "verdict": 1, "reason": "S1"}]
    claims_bad = claims_ok + [{"statement": "valid for one year", "verdict": 0, "reason": "absent"}]
    results = [res("Q02", "knowledge", True, True, 0.5, claims_bad), res("Q04", "mixed", True, False, 1.0, claims_ok), res("Q27", "not_covered", False, True, 1.0, claims_ok, handled=True, coverage="weak")]
    e2e = {"model": "openai/gpt-4o-mini", "judge_model": "anthropic/claude-haiku-4.5", "judge_backend": "builtin", "seconds": 9.0, "tokens": {"input": 30, "output": 15},
           "summary": {"questions": 3, "errors": 0, "tool_selection": {"correct": 2, "total": 3}, "not_covered": {"handled": 1, "total": 1}, "faithfulness": {"mean": 0.8333, "probes": 0.5, "others": 1.0}},
           "results": results}
    store.write_json(run / "retrieval.json", retrieval)
    store.write_json(run / "e2e.json", e2e)
    (run / "summary.md").write_text(summary_markdown(meta, retrieval, e2e), encoding="utf-8")
    build_pdf(run, meta, retrieval, e2e)
    return run


def test_evaluation_page_renders_a_run_and_details(tmp_path, monkeypatch):
    os.environ.setdefault("OPENROUTER_API_KEY", "sk-test")
    root = tmp_path / "eval"
    run = _synthetic_run(root)
    monkeypatch.setattr(store, "RUNS_DIR", root)
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(h.value for h in at.header) + " ".join(c.value for c in at.caption)
    assert "Retrieval evaluation" in text and "Faithfulness" in text and "Tool selection accuracy" in text and "not covered" in text
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["MRR"] == "1.00" and metrics["Tool selection correct"] == "2 / 3" and metrics["Not-covered questions handled"] == "1 / 1"
    assert any("Click a bar" in c.value for c in at.caption)      # details hidden until a bar is clicked
    assert not any("Faithfulness measure for this answer" in m.value for m in at.markdown)
    assert "How to read this" in " ".join(e.label for e in at.expander)
    assert (run / "report.pdf").exists()
    # simulate the click: the page stores the picked id in session state (a rerun without a new click keeps it)
    at.session_state["eval_selected"] = "Q02"
    at.run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown) + " ".join(h.value for h in at.subheader)
    assert "Evaluation details — Q02" in body and "Faithfulness measure for this answer" in body
    assert "Coverage flag" not in {m.label for m in at.metric}
    assert any("1 of 2 factual claims supported" in i.value for i in at.info)


def test_evaluation_page_without_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path / "none")
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    assert not at.exception and any("No evaluation runs" in m.value for m in at.markdown)
