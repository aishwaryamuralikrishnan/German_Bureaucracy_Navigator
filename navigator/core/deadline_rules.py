"""Deadline rules for common bureaucratic events (pure date logic)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import holidays

from navigator.utils.errors import ValidationError

# Map city -> federal state code used by the `holidays` package.
CITY_TO_STATE = {
    # Baden-Württemberg
    "stuttgart": "BW", "karlsruhe": "BW", "mannheim": "BW", "freiburg": "BW", "heidelberg": "BW", "ulm": "BW", "heilbronn": "BW",
    # Bayern
    "munich": "BY", "münchen": "BY", "nuremberg": "BY", "nürnberg": "BY", "augsburg": "BY", "regensburg": "BY", "würzburg": "BY", "erlangen": "BY", "ingolstadt": "BY",
    # Berlin / Brandenburg / Bremen / Hamburg
    "berlin": "BE", "potsdam": "BB", "cottbus": "BB", "bremen": "HB", "bremerhaven": "HB", "hamburg": "HH",
    # Hessen
    "frankfurt": "HE", "frankfurt am main": "HE", "wiesbaden": "HE", "darmstadt": "HE", "kassel": "HE",
    # Mecklenburg-Vorpommern
    "rostock": "MV", "schwerin": "MV", "greifswald": "MV", "neubrandenburg": "MV", "stralsund": "MV",
    # Niedersachsen
    "hannover": "NI", "hanover": "NI", "braunschweig": "NI", "oldenburg": "NI", "osnabrück": "NI", "göttingen": "NI", "wolfsburg": "NI",
    # Nordrhein-Westfalen
    "cologne": "NW", "köln": "NW", "düsseldorf": "NW", "dortmund": "NW", "essen": "NW", "duisburg": "NW", "bochum": "NW",
    "wuppertal": "NW", "bielefeld": "NW", "bonn": "NW", "münster": "NW", "aachen": "NW",
    # Rheinland-Pfalz / Saarland
    "mainz": "RP", "ludwigshafen": "RP", "koblenz": "RP", "trier": "RP", "kaiserslautern": "RP", "saarbrücken": "SL",
    # Sachsen / Sachsen-Anhalt
    "leipzig": "SN", "dresden": "SN", "chemnitz": "SN", "magdeburg": "ST", "halle": "ST",
    # Schleswig-Holstein / Thüringen
    "kiel": "SH", "lübeck": "SH", "flensburg": "SH", "erfurt": "TH", "jena": "TH", "weimar": "TH",
}


@dataclass(frozen=True)
class DeadlineRule:
    key: str
    label_en: str
    days: int                 # offset from the event date (negative = before the event)
    legal_basis: str
    description: str
    hard_deadline: bool       # False = recommended date, not a statutory deadline


RULES: dict[str, DeadlineRule] = {
    "moved_in": DeadlineRule(
        key="moved_in",
        label_en="Register your address (Anmeldung)",
        days=14,
        legal_basis="§ 17 Abs. 1 Bundesmeldegesetz (BMG)",
        description="You must register your new address at the Bürgeramt within two weeks of moving in.",
        hard_deadline=True,
    ),
    "residence_permit_expiry": DeadlineRule(
        key="residence_permit_expiry",
        label_en="Apply for residence permit extension (recommended latest date)",
        days=-56,
        legal_basis="§ 81 Abs. 4 AufenthG (application before expiry keeps your status valid)",
        description=(
            "Apply well before your permit expires; authorities recommend at least 8 weeks. "
            "Applying before expiry triggers a Fiktionsbescheinigung so your stay remains lawful."
        ),
        hard_deadline=False,
    ),
    "visa_expiry": DeadlineRule(
        key="visa_expiry",
        label_en="Apply for the residence permit before your entry visa expires",
        days=-28,
        legal_basis="§ 81 Abs. 4 AufenthG",
        description="Book the Ausländerbehörde appointment early — waiting times of several weeks are common.",
        hard_deadline=False,
    ),
    "child_birth_registration": DeadlineRule(
        key="child_birth_registration",
        label_en="Register the birth at the Standesamt",
        days=7,
        legal_basis="§ 18 Personenstandsgesetz (PStG)",
        description="A birth must be reported to the civil registry office within one week.",
        hard_deadline=True,
    ),
    "elterngeld_application": DeadlineRule(
        key="elterngeld_application",
        label_en="Apply for Elterngeld to receive full retroactive payment",
        days=90,
        legal_basis="§ 7 Abs. 1 BEEG (paid retroactively for at most 3 months)",
        description="Elterngeld is only paid retroactively for the three months before the application.",
        hard_deadline=False,
    ),
    "job_start": DeadlineRule(
        key="job_start",
        label_en="Provide tax ID and health insurance details to your employer",
        days=0,
        legal_basis="§ 39e EStG (ELStAM), § 28a SGB IV (employer registration)",
        description="Your employer needs your Steuer-ID and Krankenkasse by the first payroll run.",
        hard_deadline=False,
    ),
}


@dataclass(frozen=True)
class DeadlineResult:
    event: str
    label: str
    event_date: date
    raw_deadline: date
    deadline: date              # adjusted to next business day if it falls on a weekend/holiday
    adjusted: bool
    days_remaining: int
    urgency: str                # "ok" | "soon" | "overdue"
    hard_deadline: bool
    legal_basis: str
    description: str
    state_code: str | None


def state_for_city(city: str | None) -> str | None:
    if not city:
        return None
    return CITY_TO_STATE.get(city.strip().lower())


def _next_business_day(d: date, state_code: str | None) -> date:
    de_holidays = holidays.Germany(subdiv=state_code, years=[d.year, d.year + 1]) if state_code else holidays.Germany(years=[d.year, d.year + 1])
    while d.weekday() >= 5 or d in de_holidays:
        d += timedelta(days=1)
    return d


def calculate_deadline(event: str, event_date: date, city: str | None = None, today: date | None = None) -> DeadlineResult:
    if event not in RULES:
        raise ValidationError(f"Unknown event '{event}'. Valid events: {sorted(RULES)}.")
    if not isinstance(event_date, date):
        raise ValidationError("event_date must be a date.")
    today = today or date.today()
    if abs((event_date - today).days) > 366 * 5:
        raise ValidationError("event_date must be within five years of today.")

    rule = RULES[event]
    raw = event_date + timedelta(days=rule.days)
    state_code = state_for_city(city)
    # German civil-law convention (§ 193 BGB): a deadline falling on a Saturday, Sunday or
    # public holiday moves to the next business day. We apply it to hard deadlines only.
    deadline = _next_business_day(raw, state_code) if rule.hard_deadline else raw
    days_remaining = (deadline - today).days
    urgency = "overdue" if days_remaining < 0 else "soon" if days_remaining <= 7 else "ok"

    return DeadlineResult(
        event=event,
        label=rule.label_en,
        event_date=event_date,
        raw_deadline=raw,
        deadline=deadline,
        adjusted=deadline != raw,
        days_remaining=days_remaining,
        urgency=urgency,
        hard_deadline=rule.hard_deadline,
        legal_basis=rule.legal_basis,
        description=rule.description,
        state_code=state_code,
    )
