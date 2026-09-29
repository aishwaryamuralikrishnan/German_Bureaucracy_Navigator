"""Input guardrail pipeline: validate -> redact PII -> injection check -> language -> topic gate."""

from __future__ import annotations

from dataclasses import dataclass, field

from navigator.config import get_settings
from navigator.guardrails.injection import check_injection
from navigator.guardrails.input_validation import detect_language, validate_input
from navigator.guardrails.pii import redact_pii
from navigator.guardrails.topic_gate import is_on_topic
from navigator.utils.errors import ValidationError

REFUSALS = {
    "prohibited": {
        "en": "I can't help with anything that involves forged documents, hiding information from authorities, or other ways around the law. I'm happy to explain the legitimate route instead.",
        "de": "Bei gefälschten Dokumenten, dem Verschweigen von Informationen gegenüber Behörden oder anderen Umgehungen des Gesetzes kann ich nicht helfen. Gern erkläre ich den regulären Weg.",
    },
    "injection": {
        "en": "That message looks like an attempt to change how I work. I can only help with questions about living and working in Germany.",
        "de": "Diese Nachricht sieht wie ein Versuch aus, meine Arbeitsweise zu ändern. Ich helfe nur bei Fragen zum Leben und Arbeiten in Deutschland.",
    },
    "off_topic": {
        "en": "I'm specialised in German bureaucracy for newcomers — registration, residence permits, taxes, insurance, banking, licences, recognition and family matters. Could you ask something in that area?",
        "de": "Ich bin auf deutsche Bürokratie für Neuankömmlinge spezialisiert — Anmeldung, Aufenthaltstitel, Steuern, Versicherung, Bank, Führerschein, Anerkennung und Familie. Kannst du etwas aus diesem Bereich fragen?",
    },
}


@dataclass
class GuardrailOutcome:
    allowed: bool
    text: str                   # cleaned + redacted text to send to the agent
    language: str               # "de" | "en"
    redactions: list[str] = field(default_factory=list)
    reason: str | None = None   # "prohibited" | "injection" | "off_topic" | "validation"
    user_message: str | None = None


def run_input_guardrails(raw_text: str, forced_language: str | None = None) -> GuardrailOutcome:
    cfg = get_settings().guardrails
    try:
        text = validate_input(raw_text)
    except ValidationError as exc:
        return GuardrailOutcome(False, "", forced_language or "en", reason="validation", user_message=str(exc))

    language = forced_language if forced_language in {"de", "en"} else detect_language(text)

    redactions: list[str] = []
    if cfg.pii_redaction:
        red = redact_pii(text)
        text, redactions = red.text, red.redactions

    if cfg.injection_check:
        verdict = check_injection(text)
        if verdict.flagged:
            return GuardrailOutcome(False, text, language, redactions, verdict.category, REFUSALS[verdict.category][language])

    if not is_on_topic(text):
        return GuardrailOutcome(False, text, language, redactions, "off_topic", REFUSALS["off_topic"][language])

    return GuardrailOutcome(True, text, language, redactions)
