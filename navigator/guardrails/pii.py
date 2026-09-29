"""Regex-based PII redaction for the identifiers newcomers typically paste."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters: more specific patterns first.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("TAX_ID", re.compile(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b")),          # 11-digit Steuer-ID
    ("SOCIAL_SECURITY_NO", re.compile(r"\b\d{2}\s?\d{6}\s?[A-Z]\s?\d{3}\b")),  # Rentenversicherungsnummer
    ("PHONE", re.compile(r"(?<!\d)(?:\+49|0049|0)\s?[1-9]\d{1,4}[\s/-]?\d{3,}[\s-]?\d{2,}(?!\d)")),
    ("PASSPORT", re.compile(r"\b(?:passport|reisepass|pass nr\.?|passnummer)\s*[:#]?\s*([A-Z0-9]{6,10})\b", re.I)),
    ("DOB", re.compile(r"\b(?:geboren am|born on|dob|date of birth|geburtsdatum)\s*[:]?\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", re.I)),
]


@dataclass
class RedactionResult:
    text: str
    redactions: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.redactions)


def redact_pii(text: str) -> RedactionResult:
    redactions: list[str] = []
    out = text
    for label, pattern in PATTERNS:
        def _sub(m: re.Match[str], label=label) -> str:
            redactions.append(label)
            return f"<{label}>"
        out = pattern.sub(_sub, out)
    return RedactionResult(out, redactions)
