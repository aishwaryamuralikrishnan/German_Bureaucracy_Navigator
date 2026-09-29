import pytest

from navigator.core.tax_calc import estimate_net_salary, income_tax_basic, load_params
from navigator.utils.errors import ValidationError


@pytest.fixture(scope="module")
def p():
    return load_params(2026)


def test_no_tax_below_basic_allowance(p):
    assert income_tax_basic(12_348, p) == 0.0
    assert income_tax_basic(5_000, p) == 0.0


@pytest.mark.parametrize("boundary", [17_799, 69_878, 277_825])
def test_tariff_is_continuous_at_zone_boundaries(boundary, p):
    # Adjacent euros across a zone boundary differ by at most the top marginal rate (+1 for rounding).
    assert abs(income_tax_basic(boundary + 1, p) - income_tax_basic(boundary, p)) <= 1.0


def test_tariff_is_monotonic(p):
    prev = 0.0
    for zve in range(0, 400_000, 2_500):
        cur = income_tax_basic(zve, p)
        assert cur >= prev
        prev = cur


def test_net_salary_plausible_for_60k_class_1():
    r = estimate_net_salary(60_000, tax_class=1, federal_state="Berlin", year=2026)
    assert 2_900 <= r.net_monthly <= 3_400
    assert r.pension_insurance == pytest.approx(60_000 * 0.093, rel=1e-3)
    assert r.church_tax == 0.0


def test_class_3_pays_less_than_class_1():
    c1 = estimate_net_salary(60_000, tax_class=1, year=2026)
    c3 = estimate_net_salary(60_000, tax_class=3, year=2026)
    assert c3.net_annual > c1.net_annual


def test_ceilings_cap_contributions():
    r = estimate_net_salary(200_000, tax_class=1, year=2026)
    assert r.pension_insurance == pytest.approx(101_400 * 0.093, rel=1e-3)
    assert r.health_insurance == pytest.approx(69_750 * (0.146 + 0.029) / 2, rel=1e-3)


def test_church_tax_rate_by_state():
    by = estimate_net_salary(60_000, church_member=True, federal_state="Bayern", year=2026)
    be = estimate_net_salary(60_000, church_member=True, federal_state="Berlin", year=2026)
    assert by.church_tax == pytest.approx(by.income_tax * 0.08, rel=1e-3)
    assert be.church_tax == pytest.approx(be.income_tax * 0.09, rel=1e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gross_annual_eur": -1},
        {"gross_annual_eur": 50_000, "tax_class": 7},
        {"gross_annual_eur": 50_000, "federal_state": "Atlantis"},
        {"gross_annual_eur": 50_000, "health_insurance": "private"},  # missing premium
        {"gross_annual_eur": 50_000, "year": 1999},
    ],
)
def test_validation_errors(kwargs):
    with pytest.raises(ValidationError):
        estimate_net_salary(**kwargs)
