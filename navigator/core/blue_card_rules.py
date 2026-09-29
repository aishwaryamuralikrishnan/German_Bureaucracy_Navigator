"""EU Blue Card salary-threshold logic (pure, testable, no LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from navigator.config import REFERENCE_DIR
from navigator.utils.errors import ValidationError


@dataclass(frozen=True)
class BlueCardCheck:
    year: int
    salary_eur: float
    general_threshold_eur: float
    reduced_threshold_eur: float
    meets_general: bool
    meets_reduced: bool
    gap_to_general_eur: float        # positive = salary exceeds the threshold
    gap_to_reduced_eur: float
    reduced_case_claimed: bool       # user said shortage occupation / recent graduate / IT specialist
    eligible: bool                   # under the case the user described
    applicable_threshold_type: str   # "general" | "reduced"
    applicable_threshold_eur: float
    legal_basis: str
    notes_general: str
    notes_reduced: str
    requires_ba_approval: bool


@lru_cache(maxsize=1)
def _load_table() -> dict:
    with (REFERENCE_DIR / "blue_card_thresholds.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def available_years() -> list[int]:
    return sorted(int(y) for y in _load_table()["years"].keys())


def check_blue_card(
    gross_annual_salary_eur: float,
    year: int,
    is_shortage_occupation: bool = False,
    is_recent_graduate: bool = False,
    is_it_specialist_without_degree: bool = False,
) -> BlueCardCheck:
    if gross_annual_salary_eur <= 0:
        raise ValidationError("Gross annual salary must be a positive number.")
    if gross_annual_salary_eur > 5_000_000:
        raise ValidationError("Gross annual salary looks unrealistic (> 5,000,000 EUR).")

    table = _load_table()
    years = table["years"]
    if year not in years and str(year) not in years:
        raise ValidationError(f"No Blue Card threshold table for {year}. Available years: {available_years()}.")
    row = years.get(year) or years[str(year)]

    general = float(row["general"])
    reduced = float(row["reduced"])
    reduced_claimed = is_shortage_occupation or is_recent_graduate or is_it_specialist_without_degree
    applicable_type = "reduced" if reduced_claimed else "general"
    applicable = reduced if reduced_claimed else general

    return BlueCardCheck(
        year=year,
        salary_eur=gross_annual_salary_eur,
        general_threshold_eur=general,
        reduced_threshold_eur=reduced,
        meets_general=gross_annual_salary_eur >= general,
        meets_reduced=gross_annual_salary_eur >= reduced,
        gap_to_general_eur=round(gross_annual_salary_eur - general, 2),
        gap_to_reduced_eur=round(gross_annual_salary_eur - reduced, 2),
        reduced_case_claimed=reduced_claimed,
        eligible=gross_annual_salary_eur >= applicable,
        applicable_threshold_type=applicable_type,
        applicable_threshold_eur=applicable,
        legal_basis=table["legal_basis"],
        notes_general=table["notes"]["general"].strip(),
        notes_reduced=table["notes"]["reduced"].strip(),
        requires_ba_approval=reduced_claimed,
    )
