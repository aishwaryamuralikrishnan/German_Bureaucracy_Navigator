"""Simplified German net-salary estimator (employee perspective).

This implements the § 32a EStG tariff and 2026 social-insurance rates from
data/reference/tax_params_<year>.yaml. It is an ESTIMATE (typically within a few
percent of payroll) — it ignores Kinderfreibeträge for wage tax, Vorsorgepauschale
edge cases, and company-specific items. Every result carries its assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from navigator.config import REFERENCE_DIR
from navigator.utils.errors import ValidationError

VALID_TAX_CLASSES = {1, 2, 3, 4, 5, 6}
STATES = {
    "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg", "Hessen",
    "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz",
    "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen",
}


@dataclass
class NetSalaryResult:
    year: int
    gross_annual: float
    taxable_income: float
    income_tax: float
    solidarity_surcharge: float
    church_tax: float
    health_insurance: float
    care_insurance: float
    pension_insurance: float
    unemployment_insurance: float
    total_deductions: float
    net_annual: float
    net_monthly: float
    assumptions: list[str] = field(default_factory=list)


@lru_cache(maxsize=4)
def load_params(year: int) -> dict:
    path = REFERENCE_DIR / f"tax_params_{year}.yaml"
    if not path.exists():
        available = sorted(int(p.stem.split("_")[-1]) for p in REFERENCE_DIR.glob("tax_params_*.yaml"))
        raise ValidationError(f"No tax parameters for {year}. Available years: {available}.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def income_tax_basic(zve: float, p: dict) -> float:
    """§ 32a Abs. 1 EStG basic tariff on taxable income (zvE), rounded down to full EUR."""
    t = p["income_tax"]
    zve = float(int(zve))  # taxable income is rounded down to a full euro
    if zve <= t["basic_allowance"]:
        return 0.0
    if zve <= t["zone2_upper"]:
        y = (zve - t["basic_allowance"]) / 10_000
        tax = (t["zone2"]["a"] * y + t["zone2"]["b"]) * y
    elif zve <= t["zone3_upper"]:
        z = (zve - t["zone2_upper"]) / 10_000
        tax = (t["zone3"]["a"] * z + t["zone3"]["b"]) * z + t["zone3"]["c"]
    elif zve <= t["zone4_upper"]:
        tax = t["zone4"]["rate"] * zve - t["zone4"]["minus"]
    else:
        tax = t["zone5"]["rate"] * zve - t["zone5"]["minus"]
    return float(int(tax))


def income_tax_splitting(zve: float, p: dict) -> float:
    """Splitting tariff (tax class III approximation): 2 × tax(zvE / 2)."""
    return 2 * income_tax_basic(zve / 2, p)


def solidarity_surcharge(tax: float, p: dict, splitting: bool) -> float:
    s = p["income_tax"]["solidarity_surcharge"]
    exemption = s["exemption_splitting"] if splitting else s["exemption_single"]
    if tax <= exemption:
        return 0.0
    full = tax * s["rate"]
    phased = (tax - exemption) * s["phase_in_rate"]
    return round(min(full, phased), 2)


def estimate_net_salary(
    gross_annual_eur: float,
    tax_class: int = 1,
    federal_state: str = "Berlin",
    church_member: bool = False,
    has_children: bool = False,
    health_insurance: str = "public",
    private_premium_monthly: float | None = None,
    year: int = 2026,
    age: int = 30,
) -> NetSalaryResult:
    # ---- validation -----------------------------------------------------------------
    if gross_annual_eur <= 0 or gross_annual_eur > 5_000_000:
        raise ValidationError("Gross annual salary must be between 1 and 5,000,000 EUR.")
    if tax_class not in VALID_TAX_CLASSES:
        raise ValidationError("Tax class must be 1–6.")
    if federal_state not in STATES:
        raise ValidationError(f"Unknown federal state '{federal_state}'. Use one of: {sorted(STATES)}.")
    if health_insurance not in {"public", "private"}:
        raise ValidationError("health_insurance must be 'public' or 'private'.")
    if health_insurance == "private" and (private_premium_monthly is None or private_premium_monthly < 0):
        raise ValidationError("private_premium_monthly is required (>= 0) for private health insurance.")

    p = load_params(year)
    si = p["social_insurance"]
    assumptions: list[str] = [
        f"Parameters for {year}; employee share of social contributions only.",
        "Wage-tax approximation via § 32a EStG on gross minus lump sums and deductible contributions.",
        "Child allowances (Kinderfreibeträge) not applied to wage tax; Kindergeld not included.",
    ]

    # ---- social insurance (employee share) --------------------------------------------
    health_base = min(gross_annual_eur, si["health"]["ceiling_annual"])
    pension_base = min(gross_annual_eur, si["pension"]["ceiling_annual"])

    if health_insurance == "public":
        health_rate_employee = (si["health"]["general_rate"] + si["health"]["average_additional_rate"]) / 2
        health = round(health_base * health_rate_employee, 2)
        assumptions.append(
            f"Public health insurance with the average additional contribution "
            f"({si['health']['average_additional_rate'] * 100:.1f} %); your Krankenkasse may differ."
        )
        care_rate_employee = si["care"]["total_rate"] / 2
        if not has_children and age >= 23:
            care_rate_employee += si["care"]["childless_surcharge_employee"]
            assumptions.append("Childless surcharge on care insurance applied (age ≥ 23).")
        if federal_state == "Sachsen":
            care_rate_employee += si["care"]["saxony_employee_extra"]
        care = round(health_base * care_rate_employee, 2)
    else:
        health = round(float(private_premium_monthly) * 12, 2)
        care = 0.0
        assumptions.append("Private health insurance: premium taken as given (incl. care); employer subsidy ignored.")

    pension = round(pension_base * si["pension"]["total_rate"] / 2, 2)
    unemployment = round(pension_base * si["unemployment"]["total_rate"] / 2, 2)

    # ---- taxable income ---------------------------------------------------------------
    t = p["income_tax"]
    # Deductible precautionary expenses (Vorsorgeaufwendungen), simplified:
    # 100 % of pension, health minus 4 % (sick-pay share), care in full.
    deductible = pension + (health * 0.96 if health_insurance == "public" else health) + care
    zve = gross_annual_eur - t["employee_lump_sum"] - t["special_expenses_lump_sum"] - deductible
    if tax_class == 2:
        zve -= t["single_parent_relief"]
        assumptions.append("Tax class II: single-parent relief amount applied.")
    zve = max(zve, 0.0)

    # ---- income tax by class ------------------------------------------------------------
    splitting = False
    if tax_class == 3:
        tax = income_tax_splitting(zve, p)
        splitting = True
        assumptions.append("Tax class III approximated with the splitting tariff (spouse with no income).")
    elif tax_class in (5, 6):
        # Classes V/VI have no basic allowance and (for VI) no lump sums. Approximation:
        zve_56 = zve + t["basic_allowance"] + (t["employee_lump_sum"] if tax_class == 6 else 0)
        tax = income_tax_basic(zve_56, p)
        assumptions.append(f"Tax class {'V' if tax_class == 5 else 'VI'} approximated (no basic allowance).")
    else:
        tax = income_tax_basic(zve, p)

    soli = solidarity_surcharge(tax, p, splitting)
    church = 0.0
    if church_member:
        rate = p["church_tax"]["rate_by_state"].get(federal_state, p["church_tax"]["rate_by_state"]["default"])
        church = round(tax * rate, 2)
        assumptions.append(f"Church tax at {rate * 100:.0f} % of income tax ({federal_state}).")

    total = round(tax + soli + church + health + care + pension + unemployment, 2)
    net = round(gross_annual_eur - total, 2)

    return NetSalaryResult(
        year=year,
        gross_annual=round(gross_annual_eur, 2),
        taxable_income=round(zve, 2),
        income_tax=tax,
        solidarity_surcharge=soli,
        church_tax=church,
        health_insurance=health,
        care_insurance=care,
        pension_insurance=pension,
        unemployment_insurance=unemployment,
        total_deductions=total,
        net_annual=net,
        net_monthly=round(net / 12, 2),
        assumptions=assumptions,
    )
