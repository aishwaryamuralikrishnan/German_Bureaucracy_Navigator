from __future__ import annotations

from datetime import date
from typing import Literal

from dateutil import parser as dateparser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from navigator.core import deadline_rules
from navigator.core.schemas import ToolResult
from navigator.utils.errors import ValidationError

EventType = Literal[
    "moved_in",
    "residence_permit_expiry",
    "visa_expiry",
    "child_birth_registration",
    "elterngeld_application",
    "job_start",
]


class DeadlineInput(BaseModel):
    event: EventType = Field(description="Which event the deadline is derived from")
    event_date: str = Field(description="Date of the event in ISO format YYYY-MM-DD")
    city: str | None = Field(None, description="City, used for regional public holidays (e.g. Berlin, Munich)")


@tool("calculate_deadline", args_schema=DeadlineInput)
def calculate_deadline(event: str, event_date: str, city: str | None = None) -> str:
    """Calculate the statutory or recommended deadline that follows an event, e.g. the 14-day
    Anmeldung deadline after moving in, when to apply for a residence permit extension, or the
    Elterngeld application window. Accounts for weekends and regional public holidays."""
    try:
        parsed = dateparser.isoparse(event_date).date()
    except (ValueError, TypeError):
        try:
            parsed = dateparser.parse(event_date, dayfirst=True).date()
        except (ValueError, TypeError, OverflowError):
            return ToolResult.failure("calculate_deadline", f"Could not parse date '{event_date}'. Use YYYY-MM-DD.").to_json()
    try:
        res = deadline_rules.calculate_deadline(event, parsed, city, today=date.today())
    except ValidationError as exc:
        return ToolResult.failure("calculate_deadline", str(exc)).to_json()

    warnings = []
    if res.urgency == "overdue":
        warnings.append("This deadline has already passed. Act immediately and explain the delay to the authority.")
    elif res.urgency == "soon":
        warnings.append("Deadline is within 7 days — book an appointment now.")
    if res.adjusted:
        warnings.append("Deadline fell on a weekend/public holiday and was moved to the next business day.")
    if not res.hard_deadline:
        warnings.append("This is a recommended date, not a statutory deadline.")

    return ToolResult(
        tool="calculate_deadline",
        data={
            "event": res.event,
            "what": res.label,
            "event_date": res.event_date.isoformat(),
            "deadline": res.deadline.isoformat(),
            "days_remaining": res.days_remaining,
            "urgency": res.urgency,
            "is_statutory_deadline": res.hard_deadline,
            "legal_basis": res.legal_basis,
            "description": res.description,
            "federal_state": res.state_code,
        },
        warnings=warnings,
    ).to_json()
