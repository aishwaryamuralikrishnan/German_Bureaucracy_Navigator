"""Grounding check: evidence serialisation, judge parsing, fail-open behaviour — no network."""

from langchain_core.messages import AIMessage

from navigator.guardrails import grounding as g
from tests.fakes import ScriptedToolModel

SEARCH_CALL = {
    "name": "search_knowledge_base",
    "args": {"query": "Verpflichtungserklärung validity"},
    "result": {
        "ok": True,
        "data": {
            "coverage": "ok",
            "passages": [
                {"ref": "S1", "title": "Blocked account", "section": "Ways to prove financing",
                 "relevance": "ok", "text": "A person living in Germany signs a guarantee at their Ausländerbehörde."},
            ],
        },
    },
}
CURRENCY_CALL = {
    "name": "convert_currency",
    "args": {"amount": 2200000, "from_currency": "INR", "period": "annual"},
    "result": {"ok": True, "data": {"to": {"currency": "EUR", "amount": 22000.0}}, "warnings": ["Nominal conversion only."]},
}
FAILED_CALL = {"name": "calculate_deadline", "args": {}, "result": {"ok": False, "error": "bad date"}}


def _judge(reply: str):
    model = ScriptedToolModel(script=[AIMessage(content=reply)])
    return lambda *a, **k: model


def test_evidence_includes_passages_and_tool_data_but_not_failures():
    ev = g.evidence_from_tool_calls([SEARCH_CALL, CURRENCY_CALL, FAILED_CALL])
    assert "[S1] Blocked account" in ev and "signs a guarantee" in ev
    assert "TOOL convert_currency" in ev and "22000.0" in ev
    assert "bad date" not in ev


def test_unsupported_claims_are_reported():
    answer = "The guarantor signs at the Ausländerbehörde [S1]. The letter is typically valid for one year."
    rep = g.check_grounding(answer, [SEARCH_CALL], llm_factory=_judge('{"unsupported": ["typically valid for one year"]}'))
    assert rep.checked and rep.unsupported == ["typically valid for one year"]
    note = rep.note("en")
    assert note and note.startswith("Not backed by") and "one year" in note
    assert rep.note("de").startswith("Nicht durch")


def test_fully_grounded_answer_has_no_note():
    rep = g.check_grounding("The guarantor signs at the Ausländerbehörde [S1].", [SEARCH_CALL],
                            llm_factory=_judge('Sure! {"unsupported": []}'))
    assert rep.checked and rep.ok and rep.note("en") is None


def test_judge_garbage_fails_open():
    rep = g.check_grounding("Some answer.", [SEARCH_CALL], llm_factory=_judge("I cannot help with that"))
    assert not rep.checked and not rep.unsupported and rep.error


def test_judge_exception_fails_open():
    def boom(*a, **k):
        raise RuntimeError("openrouter down")
    rep = g.check_grounding("Some answer.", [SEARCH_CALL], llm_factory=boom)
    assert not rep.checked and "openrouter down" in (rep.error or "")


def test_no_evidence_means_no_check():
    rep = g.check_grounding("Hello!", [], llm_factory=_judge('{"unsupported": ["x"]}'))
    assert not rep.checked and not rep.unsupported


def test_claim_list_is_capped_and_deduplicated():
    claims = g._parse_claims('{"unsupported": ["a", "a", "b", "c", "d", "e", "f"]}', max_claims=4)
    assert claims == ["a", "b", "c", "d"]


# ----------------------------------------------------------------------------- figure guard

def test_figures_normalise_numbers_dates_and_durations():
    from navigator.guardrails.figures import figures, figures_missing_from
    ev = figures('{"eur": 50700.0, "reduced": 45934.2, "deadline": "2027-02-03"} 140 full days or 280 half days; EUR 12,348; '
                 'for up to one year; 9.3 % share; roughly EUR 130–150 per month; pays out EUR 992 each month; 6934.24; 18 months; at least 8 weeks')
    assert {"50700", "45934.2", "2027-02-03", "140", "280", "12348", "1 year", "9.3", "130", "150", "992", "6934.24", "18 month", "8 week"} <= ev
    for ok in ("threshold of €50,700", "€45,934.20", "apply by February 3, 2027", "bis zum 3. Februar 2027", "am 03.02.2027", "€12,348",
               "stay for up to one year", "9.3% employee share", "€130 to €150 per month", "992 euros per month", "6,934.24 EUR", "valid for 18 months", "8 weeks before"):
        assert figures_missing_from(ok, ev) == set(), ok
    assert figures_missing_from("around €219 per child", ev) == {"219"}
    assert "14 month" in figures_missing_from("paid for 14 months", ev)
    assert "2026-11-20" in figures_missing_from("by 20 November 2026", ev)
    assert figures_missing_from("the fee is 18.36 EUR per month", ev) == {"18.36"}
    assert figures_missing_from("costs 5 euros", ev) == set()  # single digits are not figures (they match trivially)


def test_verify_unsupported_ignores_claims_whose_figures_are_in_the_evidence():
    from navigator.guardrails.grounding import verify_unsupported
    evidence = "[S1] general threshold: EUR 50,700 (2026); reduced: EUR 45,934.20. TOOL calculate_deadline: {\"deadline\": \"2026-11-20\"}"
    kept, ignored = verify_unsupported(["below the general threshold of €50,700", "apply by November 20, 2026", "the fee is €29",
                                        "the letter is valid for one year", "you must hold a B1 certificate"], evidence)
    assert ignored == ["below the general threshold of €50,700", "apply by November 20, 2026"]
    assert kept == ["the fee is €29", "the letter is valid for one year", "you must hold a B1 certificate"]


def test_apply_grounding_ignores_false_positive_figures():
    """The judge flags a figure that is in the tool output in another format → no revision at all."""
    from navigator.guardrails.grounding import apply_grounding
    calls = [{"name": "check_blue_card_salary", "args": {"gross_annual_salary_eur": 50000}, "result": {"ok": True, "data": {"general_threshold": {"eur": 50700.0}, "reduced_threshold": {"eur": 45934.2}}}}]
    model = ScriptedToolModel(script=[AIMessage(content='{"unsupported": ["general threshold of €50,700", "reduced threshold of €45,934.20"]}'),
                                      AIMessage(content="SHOULD NOT BE CALLED")])
    out = apply_grounding("Your salary is below the general threshold of €50,700 but above the reduced threshold of €45,934.20.", calls, "en", llm_factory=lambda *a, **k: model)
    assert not out.revised and out.report.ok and out.note is None and len(out.report.ignored) == 2 and model.calls == 1
    assert out.to_dict()["ignored"] == ["general threshold of €50,700", "reduced threshold of €45,934.20"]


def test_apply_grounding_rejects_revision_that_drops_supported_figures():
    """One claim is really unsupported, but the reviser also deletes the 140/280-day rule the passage states → rejected."""
    from navigator.guardrails.grounding import apply_grounding
    calls = [{"name": "search_knowledge_base", "args": {}, "result": {"ok": True, "data": {"passages": [{"ref": "S1", "title": "T", "section": "S", "text": "Students may work 140 full days or 280 half days per year."}]}}}]
    answer = "You may work 140 full days or 280 half days per year [S1]. The fee for the permit is typically EUR 110."
    model = ScriptedToolModel(script=[AIMessage(content='{"unsupported": ["The fee for the permit is typically EUR 110"]}'),
                                      AIMessage(content="You may work without needing permission [S1]. The knowledge base does not state the fee.")])
    out = apply_grounding(answer, calls, "en", llm_factory=lambda *a, **k: model)
    assert not out.revised and out.text == answer and out.revision_rejected and "140" in out.revision_rejected and "280" in out.revision_rejected
    assert out.note and out.note.startswith("Not backed by") and out.to_dict()["revision_rejected"]
    # a revision that keeps the supported figures and only drops the fee is accepted
    model2 = ScriptedToolModel(script=[AIMessage(content='{"unsupported": ["The fee for the permit is typically EUR 110"]}'),
                                       AIMessage(content="You may work 140 full days or 280 half days per year [S1]. The knowledge base does not state the fee.")])
    out2 = apply_grounding(answer, calls, "en", llm_factory=lambda *a, **k: model2)
    assert out2.revised and "110" not in out2.text and "140 full days" in out2.text and out2.revision_rejected is None
