from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from typing import Literal

from dateutil import parser as dateparser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from navigator.clients.frankfurter import FrankfurterClient
from navigator.core.schemas import ToolResult
from navigator.utils.errors import ExternalServiceError, ValidationError

_CODE = re.compile(r"^[A-Za-z]{3}$")

# Common names/symbols people type instead of ISO codes.
_ALIASES = {
    "€": "EUR", "euro": "EUR", "euros": "EUR",
    "$": "USD", "dollar": "USD", "dollars": "USD", "us dollar": "USD",
    "£": "GBP", "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "₹": "INR", "rupee": "INR", "rupees": "INR", "rs": "INR", "inr": "INR",
    "yuan": "CNY", "renminbi": "CNY", "rmb": "CNY", "yen": "JPY", "won": "KRW",
    "lira": "TRY", "real": "BRL", "reais": "BRL", "peso": "MXN", "zloty": "PLN", "franc": "CHF", "rand": "ZAR",
    "rupiah": "IDR", "ringgit": "MYR", "baht": "THB", "shekel": "ILS", "forint": "HUF", "krone": "DKK", "krona": "SEK",
}


class CurrencyInput(BaseModel):
    amount: float = Field(description="Amount to convert (positive number)")
    from_currency: str = Field(description="ISO 4217 code of the amount's currency, e.g. INR, USD, EUR")
    to_currency: str = Field("EUR", description="Target currency code; defaults to EUR")
    period: Literal["one_off", "monthly", "annual"] = Field(
        "one_off", description="If the amount is a salary, say whether it is per month or per year so both figures are returned"
    )
    gross_or_net: Literal["gross", "net", "unknown"] = Field(
        "unknown",
        description="For salaries: whether the USER said the amount is gross (before tax / CTC) or net (take-home). "
                    "Use 'unknown' when they did not say — never guess.",
    )
    on_date: str | None = Field(None, description="Optional historical date YYYY-MM-DD (ECB rates go back to 1999)")


@lru_cache(maxsize=1)
def _client() -> FrankfurterClient:
    return FrankfurterClient()


def normalise_code(value: str) -> str:
    v = (value or "").strip()
    if v.lower() in _ALIASES:
        return _ALIASES[v.lower()]
    if _CODE.match(v):
        return v.upper()
    raise ValidationError(f"'{value}' is not a currency code. Use an ISO code such as EUR, USD, INR, GBP.")


@tool("convert_currency", args_schema=CurrencyInput)
def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str = "EUR",
    period: str = "one_off",
    gross_or_net: str = "unknown",
    on_date: str | None = None,
) -> str:
    """Convert an amount between currencies using European Central Bank reference rates (live via the
    Frankfurter API, no key). Use it whenever the user compares a salary, job offer, rent, blocked-account
    deposit or fee across currencies — e.g. 'is €58,000 more than my 45 lakh INR?' — and pass the result
    on to the Blue Card check or the net-salary estimator when relevant. Returns the rate, its date, the
    converted amount and, for salaries, both monthly and annual figures. The result is a NOMINAL
    exchange-rate equivalent: it does not adjust for cost of living or purchasing power and applies no
    taxes; for salaries pass gross_or_net exactly as the user stated it ('unknown' if they did not)."""
    try:
        if amount <= 0 or amount > 1e12:
            raise ValidationError("Amount must be a positive number.")
        src, dst = normalise_code(from_currency), normalise_code(to_currency)
        on = None
        if on_date:
            try:
                on = dateparser.isoparse(on_date).date()
            except (ValueError, TypeError):
                raise ValidationError(f"Could not parse date '{on_date}'. Use YYYY-MM-DD.")
            if on > date.today():
                raise ValidationError("The date must not be in the future.")
        client = _client()
        supported = client.currencies()
        missing = [c for c in (src, dst) if c not in supported]
        if missing:
            raise ValidationError(
                f"The ECB does not publish a reference rate for {', '.join(missing)}. Supported: {', '.join(sorted(supported))}."
            )
        r = client.rate(src, dst, on)
    except ValidationError as exc:
        return ToolResult.failure("convert_currency", str(exc)).to_json()
    except ExternalServiceError as exc:
        return ToolResult.failure("convert_currency", f"Exchange-rate service unavailable: {exc}. Try again in a moment.").to_json()

    converted = round(amount * r.rate, 2)
    data = {
        "from": {"currency": src, "name": supported.get(src), "amount": round(amount, 2)},
        "to": {"currency": dst, "name": supported.get(dst), "amount": converted},
        "rate": {"value": round(r.rate, 6), "inverse": round(1 / r.rate, 6) if r.rate else None, "date": r.rate_date, "source": "European Central Bank reference rate via frankfurter.dev"},
        "period": period,
        "basis": "nominal exchange-rate equivalent — not adjusted for cost of living or purchasing power; no taxes or deductions applied",
    }
    is_salary = period in ("monthly", "annual")
    if period == "monthly":
        data["equivalents"] = {f"{dst}_per_month": converted, f"{dst}_per_year": round(converted * 12, 2)}
    elif period == "annual":
        data["equivalents"] = {f"{dst}_per_year": converted, f"{dst}_per_month": round(converted / 12, 2)}
    if is_salary:
        data["salary_basis"] = gross_or_net

    warnings = [
        "Nominal conversion at the market rate only: it says nothing about purchasing power or cost of living "
        "(rent, food, insurance differ between countries) and applies no taxes or deductions.",
        "ECB reference rates are mid-market; banks and transfer services apply a spread of typically 0.5–3 %.",
    ]
    if is_salary:
        if gross_or_net == "unknown":
            warnings.append(
                "The user did not say whether this salary is gross or net. Do NOT assume either — say the amounts are "
                "compared as given and ask whether it is gross (CTC) or net take-home."
            )
        elif gross_or_net == "net":
            warnings.append(
                "This is a NET (take-home) amount. Compare it only with a German NET estimate — never with a gross "
                "offer or with the Blue Card salary threshold, which are gross figures."
            )
        else:
            warnings.append(
                "This is a GROSS amount before home-country taxes and deductions; this tool does not compute them and "
                "estimate_net_salary applies German rules only, so do not derive a foreign take-home pay from it."
            )
    if on and r.rate_date != on.isoformat():
        warnings.append(f"No fixing on {on.isoformat()} (weekend/holiday); the rate of {r.rate_date} was used.")
    if not on:
        warnings.append(f"Rate date {r.rate_date}; the ECB publishes one fixing per working day.")
    return ToolResult(tool="convert_currency", data=data, warnings=warnings).to_json()
