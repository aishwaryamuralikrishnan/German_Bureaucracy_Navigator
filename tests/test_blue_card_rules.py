import pytest

from navigator.core.blue_card_rules import available_years, check_blue_card
from navigator.utils.errors import ValidationError


def test_general_threshold_2026():
    r = check_blue_card(50_700, 2026)
    assert r.eligible and r.meets_general and r.meets_reduced and r.gap_to_general_eur == 0.0


def test_between_thresholds_reports_both():
    r = check_blue_card(47_000, 2026)
    assert not r.eligible and not r.meets_general and r.meets_reduced
    assert r.general_threshold_eur == 50_700 and r.reduced_threshold_eur == pytest.approx(45_934.20)
    r2 = check_blue_card(47_000, 2026, is_shortage_occupation=True)
    assert r2.eligible and r2.applicable_threshold_type == "reduced" and r2.requires_ba_approval


def test_45000_is_below_both_thresholds():
    r = check_blue_card(45_000, 2026, is_shortage_occupation=True)
    assert not r.eligible and not r.meets_reduced
    assert r.gap_to_reduced_eur == pytest.approx(-934.20)


def test_it_specialist_without_degree_uses_reduced_threshold():
    r = check_blue_card(46_000, 2026, is_it_specialist_without_degree=True)
    assert r.eligible and r.applicable_threshold_eur == pytest.approx(45_934.20)


def test_years_available_and_unknown_year_rejected():
    assert {2024, 2025, 2026} <= set(available_years())
    with pytest.raises(ValidationError):
        check_blue_card(60_000, 1990)


def test_invalid_salary():
    with pytest.raises(ValidationError):
        check_blue_card(0, 2026)
