"""Post-generation checks: citations valid, no PII leaked, weak-coverage notice, disclaimers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from navigator.config import get_settings
from navigator.guardrails.pii import redact_pii

_CITATION_RE = re.compile(r"\[(S\d+)\]")

DISCLAIMERS = {
    "en": "This is general information compiled from official sources, not legal or tax advice. Rules change; confirm details with the responsible authority.",
    "de": "Dies sind allgemeine Informationen aus offiziellen Quellen, keine Rechts- oder Steuerberatung. Regeln ändern sich; bitte bei der zuständigen Behörde bestätigen.",
}
WEAK_NOTICE = {
    "en": "The knowledge base has no specific passage on this question, so the answer above is general orientation rather than a sourced statement. Please verify with the responsible authority.",
    "de": "Die Wissensdatenbank enthält keine spezifische Passage zu dieser Frage; die Antwort ist eine allgemeine Orientierung ohne belegte Quelle. Bitte bei der zuständigen Behörde prüfen.",
}


@dataclass
class OutputCheck:
    text: str
    citations_found: bool
    missing_citations: bool
    invalid_citations_removed: int = 0
    pii_removed: int = 0
    weak_coverage: bool = False
    notes: list[str] = field(default_factory=list)


def check_output(
    answer: str,
    used_search: bool,
    topics_touched: set[str],
    language: str,
    valid_refs: set[str] | None = None,
    weak_coverage: bool = False,
) -> OutputCheck:
    cfg = get_settings().guardrails
    notes: list[str] = []

    red = redact_pii(answer)
    text = red.text
    if red.count:
        notes.append(f"Removed {red.count} personal identifier(s) from the answer.")

    # Drop citations that point to passages the search never returned (hallucinated refs).
    removed = 0
    if valid_refs is not None:
        def _keep(m: re.Match[str]) -> str:
            nonlocal removed
            if m.group(1) in valid_refs:
                return m.group(0)
            removed += 1
            return ""
        text = _CITATION_RE.sub(_keep, text)
        text = re.sub(r" +([.,;:])", r"\1", text)
        if removed:
            notes.append(f"Removed {removed} citation(s) that did not match any retrieved passage.")

    # When retrieval had only weak matches, citations are misleading: strip them and say so.
    if weak_coverage and used_search:
        text = _CITATION_RE.sub("", text)
        text = re.sub(r" +([.,;:])", r"\1", text)
        notes.append(WEAK_NOTICE.get(language, WEAK_NOTICE["en"]))

    has_cit = bool(_CITATION_RE.search(text))
    missing = bool(cfg.require_citations and used_search and not has_cit and not weak_coverage)
    if missing:
        notes.append("The answer used the knowledge base but did not cite passages — treat it as general guidance.")

    if topics_touched & set(cfg.disclaimer_topics):
        text = f"{text.rstrip()}\n\n_{DISCLAIMERS.get(language, DISCLAIMERS['en'])}_"

    return OutputCheck(
        text=text, citations_found=has_cit, missing_citations=missing, invalid_citations_removed=removed,
        pii_removed=red.count, weak_coverage=weak_coverage, notes=notes,
    )
