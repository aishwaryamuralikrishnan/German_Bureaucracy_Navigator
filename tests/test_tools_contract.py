"""Every tool must return a JSON ToolResult envelope and validate its inputs."""

import json

from navigator.core.schemas import ToolResult
from navigator.tools import ALL_TOOLS, TOOL_CATEGORIES
from navigator.tools.blue_card import check_blue_card_salary
from navigator.tools.deadlines import calculate_deadline
from navigator.tools.net_salary import estimate_net_salary


def _parse(raw: str) -> ToolResult:
    return ToolResult.model_validate(json.loads(raw))


def test_all_tools_registered_with_categories():
    assert len(ALL_TOOLS) == 5
    for t in ALL_TOOLS:
        assert t.name in TOOL_CATEGORIES
        assert t.description and t.args_schema is not None


def test_blue_card_tool_ok_and_failure_envelopes():
    ok = _parse(check_blue_card_salary.invoke({"gross_annual_salary_eur": 60000, "year": 2026}))
    assert ok.ok and ok.tool == "check_blue_card_salary" and ok.data["general_threshold"]["met"] is True
    bad = _parse(check_blue_card_salary.invoke({"gross_annual_salary_eur": -5, "year": 2026}))
    assert not bad.ok and bad.error


def test_deadline_tool_parses_dates():
    res = _parse(calculate_deadline.invoke({"event": "moved_in", "event_date": "2026-09-01", "city": "Berlin"}))
    assert res.ok and res.data["deadline"] == "2026-09-15"
    bad = _parse(calculate_deadline.invoke({"event": "moved_in", "event_date": "not a date"}))
    assert not bad.ok


def test_net_salary_tool_carries_warnings():
    res = _parse(estimate_net_salary.invoke({"gross_annual_eur": 55000, "tax_class": 1, "federal_state": "Hamburg"}))
    assert res.ok and res.data["net_monthly_eur"] > 0
    assert any("Estimate" in w for w in res.warnings)


def test_tool_names_are_the_expected_five():
    assert {t.name for t in ALL_TOOLS} == {"search_knowledge_base", "check_blue_card_salary", "calculate_deadline", "estimate_net_salary", "convert_currency"}
