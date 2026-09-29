from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from navigator.core import tax_calc
from navigator.core.schemas import ToolResult
from navigator.utils.errors import ValidationError


class NetSalaryInput(BaseModel):
    gross_annual_eur: float = Field(description="Gross annual salary in EUR")
    tax_class: int = Field(1, ge=1, le=6, description="German wage tax class 1-6 (single = 1, married default = 4)")
    federal_state: str = Field("Berlin", description="Federal state in German spelling, e.g. Berlin, Bayern, Nordrhein-Westfalen")
    church_member: bool = Field(False, description="Pays church tax?")
    has_children: bool = Field(False, description="Has at least one child (affects care insurance)")
    health_insurance: Literal["public", "private"] = Field("public")
    private_premium_monthly: float | None = Field(None, description="Monthly private premium if private")
    year: int = Field(2026)
    age: int = Field(30, ge=16, le=80)


@tool("estimate_net_salary", args_schema=NetSalaryInput)
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
) -> str:
    """Estimate the monthly and annual net salary from a German gross salary, with a full
    breakdown of income tax, solidarity surcharge, church tax and the employee share of
    health, care, pension and unemployment insurance. Use for 'what will I take home' questions."""
    try:
        res = tax_calc.estimate_net_salary(
            gross_annual_eur, tax_class, federal_state, church_member, has_children,
            health_insurance, private_premium_monthly, year, age,
        )
    except ValidationError as exc:
        return ToolResult.failure("estimate_net_salary", str(exc)).to_json()
    return ToolResult(
        tool="estimate_net_salary",
        data={
            "year": res.year,
            "gross_annual_eur": res.gross_annual,
            "gross_monthly_eur": round(res.gross_annual / 12, 2),
            "net_annual_eur": res.net_annual,
            "net_monthly_eur": res.net_monthly,
            "breakdown_annual_eur": {
                "income_tax": res.income_tax,
                "solidarity_surcharge": res.solidarity_surcharge,
                "church_tax": res.church_tax,
                "health_insurance": res.health_insurance,
                "care_insurance": res.care_insurance,
                "pension_insurance": res.pension_insurance,
                "unemployment_insurance": res.unemployment_insurance,
                "total_deductions": res.total_deductions,
            },
            "taxable_income_eur": res.taxable_income,
            "inputs": {"tax_class": tax_class, "federal_state": federal_state, "health_insurance": health_insurance},
        },
        warnings=["Estimate only (±3 %). Use an official Brutto-Netto calculator or your payslip for exact figures."]
        + res.assumptions,
    ).to_json()
