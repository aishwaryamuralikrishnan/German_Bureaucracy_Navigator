"""Deterministic extraction of *figures* — numbers, dates and durations — from text, in a form that lets two
texts be compared regardless of formatting.

The grounding step asks a small model which claims the evidence does not support. Small models produce false
positives when the same figure is written differently: "€50,700" in the answer vs 50700.0 in a tool result,
"February 3, 2027" vs "2027-02-03", "one year" vs "1 year". A revision built on a false positive deletes a correct
figure — usually the very number the user asked for. This module provides the check that prevents it: a figure
that appears in the evidence is supported by definition, whatever the judge said.

Normalisation rules
  numbers    thousands separators removed ("50,700" / "50.700" → "50700"), a lone ".0" dropped ("50700.0" →
             "50700"); a single separator followed by three digits is ambiguous ("9.300" = 9300 or 9.3), so both
             readings are kept. Single-digit standalone numbers are ignored (they match trivially).
  dates      ISO, "3 February 2027", "February 3, 2027", "3. Februar 2027", "03.02.2027", "3.2.2027" → 2027-02-03
  durations  "<n> day(s)|week(s)|month(s)|year(s)" and the German equivalents, with number words up to twelve
             ("one year", "zwei Wochen") → "1 year", "2 week"
"""

from __future__ import annotations

import re

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
    "ein": 1, "eine": 1, "einem": 1, "einen": 1, "einer": 1, "eines": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11, "zwölf": 12,
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "mai": 5, "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "okt": 10,
    "nov": 11, "dec": 12, "dez": 12,
}
_UNITS = {
    "day": "day", "days": "day", "tag": "day", "tage": "day", "tagen": "day",
    "week": "week", "weeks": "week", "woche": "week", "wochen": "week",
    "month": "month", "months": "month", "monat": "month", "monate": "month", "monaten": "month",
    "year": "year", "years": "year", "jahr": "year", "jahre": "year", "jahren": "year",
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_UNIT_RE = "|".join(sorted(_UNITS, key=len, reverse=True))
_WORD_RE = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_DOT = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_D_MONTH_Y = re.compile(rf"\b(\d{{1,2}})\.?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)
_MONTH_D_Y = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DURATION = re.compile(rf"\b(\d+|{_WORD_RE})[\s\-]+({_UNIT_RE})\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])\d[\d.,]*\d(?![\w])|(?<![\w.,])\d(?![\w.,])")


def _date(y: str, m: int | str, d: str) -> str | None:
    try:
        yi, mi, di = int(y), int(m), int(d)
    except ValueError:
        return None
    if not (1 <= mi <= 12 and 1 <= di <= 31 and 1900 <= yi <= 2200):
        return None
    return f"{yi:04d}-{mi:02d}-{di:02d}"


def dates(text: str) -> set[str]:
    out: set[str] = set()
    for y, m, d in _ISO_DATE.findall(text):
        if (v := _date(y, m, d)):
            out.add(v)
    for d, m, y in _DMY_DOT.findall(text):
        if (v := _date(y, m, d)):
            out.add(v)
    for d, mon, y in _D_MONTH_Y.findall(text):
        if (v := _date(y, _MONTHS[mon.lower()], d)):
            out.add(v)
    for mon, d, y in _MONTH_D_Y.findall(text):
        if (v := _date(y, _MONTHS[mon.lower()], d)):
            out.add(v)
    return out


def _strip_dates(text: str) -> str:
    """Blank out recognised dates so their digits are not read as numbers ("03.02.2027" is a date, not 3022027)."""
    for rx in (_ISO_DATE, _DMY_DOT, _D_MONTH_Y, _MONTH_D_Y):
        text = rx.sub(" ", text)
    return text


def durations(text: str) -> set[str]:
    out: set[str] = set()
    for n, unit in _DURATION.findall(text):
        num = _NUMBER_WORDS.get(n.lower()) if not n.isdigit() else int(n)
        if num is not None:
            out.add(f"{num} {_UNITS[unit.lower()]}")
    return out


def _readings(token: str) -> set[str]:
    """All plausible normalised readings of one number token."""
    t = token
    out: set[str] = set()
    if "," in t and "." in t:
        # the last separator is the decimal mark
        if t.rfind(",") > t.rfind("."):
            out.add(t.replace(".", "").replace(",", "."))
        else:
            out.add(t.replace(",", ""))
    elif "," in t or "." in t:
        sep = "," if "," in t else "."
        parts = t.split(sep)
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            out.add("".join(parts))                  # 50,700 / 50.700 / 1.234.567 → thousands separators
        if len(parts) == 2:
            out.add(parts[0] + "." + parts[1])          # 9.3 / 9,3 / 6934.24 → decimal (also kept for 9.300)
    else:
        out.add(t)
    norm: set[str] = set()
    for v in out:
        if "." in v:
            v = v.rstrip("0").rstrip(".") if v.endswith("0") or v.endswith(".") else v
        norm.add(v)
    return norm


def numbers(text: str) -> set[str]:
    """Normalised number tokens with at least two significant characters (single digits match trivially)."""
    out: set[str] = set()
    for tok in _NUMBER.findall(_strip_dates(text)):
        for v in _readings(tok):
            if len(v.replace(".", "")) >= 2:
                out.add(v)
    return out


def figures(text: str) -> set[str]:
    """Every figure in the text: numbers, ISO-normalised dates and '<n> <unit>' durations."""
    return numbers(text) | dates(text) | durations(text)


def figures_missing_from(text: str, evidence_figs: set[str]) -> set[str]:
    """Figures of `text` that do not appear in the evidence. Ambiguous readings count as present if any reading is."""
    missing: set[str] = set()
    for tok in _NUMBER.findall(_strip_dates(text)):
        rd = {v for v in _readings(tok) if len(v.replace(".", "")) >= 2}
        if rd and not (rd & evidence_figs):
            missing.add(tok)
    for d in dates(text):
        if d not in evidence_figs:
            missing.add(d)
    for dur in durations(text):
        if dur not in evidence_figs:
            missing.add(dur)
    return missing
