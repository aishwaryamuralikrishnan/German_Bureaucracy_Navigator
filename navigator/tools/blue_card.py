from __future__ import annotations

from datetime import date

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from navigator.core import blue_card_rules
from navigator.core.schemas import ToolResult
from navigator.utils.errors import ValidationError


class BlueCardInput(BaseModel):
    gross_annual_salary_eur: float = Field(description="Gross annual salary offered in EUR")
    year: int = Field(default_factory=lambda: date.today().year, description="Year the job starts")
    is_shortage_occupation: bool = Field(False, description="True for STEM, IT, medicine, nursing, teaching, engineering etc. Leave False if unknown.")
    is_recent_graduate: bool = Field(False, description="True if the degree was completed within the last 3 years. Leave False if unknown.")
    is_it_specialist_without_degree: bool = Field(False, description="True for IT specialists with >= 3 years experience but no degree")


@tool("check_blue_card_salary", args_schema=BlueCardInput)
def check_blue_card_salary(
    gross_annual_salary_eur: float,
    year: int | None = None,
    is_shortage_occupation: bool = False,
    is_recent_graduate: bool = False,
    is_it_specialist_without_degree: bool = False,
) -> str:
    """Check a gross annual salary against BOTH EU Blue Card thresholds for a year (§ 18g AufenthG):
    the general threshold and the reduced threshold for shortage occupations, recent graduates and
    IT specialists without a degree. Always returns both so the answer can explain each case.
    Use whenever the user asks if they qualify for the Blue Card or what the threshold is."""
    year = year or date.today().year
    try:
        r = blue_card_rules.check_blue_card(
            gross_annual_salary_eur, year, is_shortage_occupation, is_recent_graduate, is_it_specialist_without_degree
        )
    except ValidationError as exc:
        return ToolResult.failure("check_blue_card_salary", str(exc)).to_json()

    # A human-readable verdict the model can relay directly.
    if r.meets_general:
        verdict = f"Meets the general threshold ({r.general_threshold_eur:,.2f} EUR) — salary condition satisfied for any qualifying occupation."
    elif r.meets_reduced:
        verdict = (
            f"Below the general threshold ({r.general_threshold_eur:,.2f} EUR) but above the reduced threshold "
            f"({r.reduced_threshold_eur:,.2f} EUR): qualifies ONLY if the job is a shortage occupation, the degree is "
            f"less than 3 years old, or the applicant is an IT specialist with 3+ years of experience"
            + (" — which the user indicated." if r.reduced_case_claimed else " — ask the user whether one of these applies.")
        )
    else:
        verdict = (
            f"Below both thresholds: {abs(r.gap_to_general_eur):,.2f} EUR short of the general threshold "
            f"({r.general_threshold_eur:,.2f}) and {abs(r.gap_to_reduced_eur):,.2f} EUR short of the reduced threshold "
            f"({r.reduced_threshold_eur:,.2f}). Alternatives: negotiate the salary, or apply for the skilled-worker "
            f"residence permit under § 18b AufenthG, which has no fixed salary threshold."
        )

    warnings = []
    if r.eligible and min(abs(r.gap_to_general_eur), abs(r.gap_to_reduced_eur)) < 1500:
        warnings.append("Salary is within 1,500 EUR of a threshold — thresholds rise every January; check the year of the permit decision.")
    if not r.reduced_case_claimed and not r.meets_general:
        warnings.append("Occupation not specified: the reduced threshold may apply if it is a shortage occupation or the degree is recent.")

    return ToolResult(
        tool="check_blue_card_salary",
        data={
            "year": r.year,
            "salary_eur": r.salary_eur,
            "verdict": verdict,
            "eligible_under_described_case": r.eligible,
            "general_threshold": {"eur": r.general_threshold_eur, "met": r.meets_general, "gap_eur": r.gap_to_general_eur, "applies_to": r.notes_general},
            "reduced_threshold": {"eur": r.reduced_threshold_eur, "met": r.meets_reduced, "gap_eur": r.gap_to_reduced_eur, "applies_to": r.notes_reduced},
            "reduced_case_claimed_by_user": r.reduced_case_claimed,
            "requires_federal_employment_agency_approval": r.requires_ba_approval,
            "legal_basis": r.legal_basis,
            "other_requirements": "A recognised university degree (or 3+ years IT experience), a concrete job offer of at least 6 months, and the job must match the qualification.",
        },
        warnings=warnings,
    ).to_json()
