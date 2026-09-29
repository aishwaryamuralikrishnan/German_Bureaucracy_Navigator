"""Basic input hygiene and DE/EN language detection (dependency-free)."""

from __future__ import annotations

import re
import unicodedata

from navigator.config import get_settings
from navigator.utils.errors import ValidationError

_DE_MARKERS = {
    "ich", "und", "nicht", "das", "die", "der", "ist", "wie", "was", "ein", "eine", "mit", "für", "muss", "kann",
    "wo", "wann", "meine", "mein", "brauche", "habe", "bin", "auf", "zu", "im", "bei", "nach", "von", "oder", "auch",
    "anmeldung", "aufenthaltstitel", "steuer", "krankenkasse", "termin", "behörde", "wohnung",
}
_EN_MARKERS = {
    "i", "the", "and", "not", "is", "how", "what", "a", "an", "with", "for", "must", "can", "where", "when", "my",
    "need", "have", "am", "do", "to", "in", "at", "after", "of", "or", "also", "register", "permit", "tax",
    "insurance", "appointment", "office", "apartment", "should", "does",
}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Strip control characters except newline/tab.
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"[ \t]+", " ", text).strip()


def validate_input(text: str) -> str:
    if text is None or not str(text).strip():
        raise ValidationError("Please type a question.")
    cleaned = normalise(str(text))
    limit = get_settings().app.max_input_chars
    if len(cleaned) > limit:
        raise ValidationError(f"Your message is too long ({len(cleaned)} characters). Please keep it under {limit}.")
    if len(cleaned) < 2:
        raise ValidationError("Your message is too short.")
    return cleaned


def detect_language(text: str) -> str:
    """Return 'de' or 'en' using stop-word voting plus umlaut/ß evidence. Defaults to 'en'."""
    tokens = re.findall(r"[a-zA-ZäöüÄÖÜß]+", text.lower())
    de = sum(1 for t in tokens if t in _DE_MARKERS) + 2 * sum(1 for t in tokens if re.search(r"[äöüß]", t))
    en = sum(1 for t in tokens if t in _EN_MARKERS)
    if de == en == 0:
        return "en"
    return "de" if de > en else "en"
