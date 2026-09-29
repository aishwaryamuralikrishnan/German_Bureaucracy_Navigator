"""Heuristic prompt-injection detection (direct, in user input).

Indirect injection (instructions hidden in retrieved documents) is handled in the system
prompt by wrapping tool output as data, plus the output checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS = [
    # "ignore all previous instructions", "ignore your previous instructions", "ignore the rules above", "forget your instructions"
    r"\b(ignore|forget|disregard|override|bypass)\b[^.\n]{0,20}?\b(previous|prior|above|earlier|initial|original|system|your|these|those|all)\b[^.\n]{0,20}?\b(instructions?|prompts?|rules|guidelines|directions|programming)\b",
    r"\b(ignore|forget|disregard)\b (everything|all|what) (above|before|you were told)",
    r"\b(ignore|forget|disregard)\b (the |all |any )?(instructions?|rules|prompts?|guidelines) (above|before|so far|until now)",
    r"\b(ignoriere|vergiss|missachte)\b[^.\n]{0,20}?\b(vorherigen|bisherigen|obigen|deine|alle)\b[^.\n]{0,20}?\b(anweisungen|regeln|vorgaben|instruktionen)\b",
    r"\bnew instructions?\s*:",
    r"disregard (your|the) (system|previous) (prompt|instructions)",
    r"you are now (a|an|the) ",
    r"act as (a|an) (unrestricted|jailbroken|different)",
    r"pretend (you are|to be) ",
    r"\bDAN\b|do anything now",
    r"reveal (your|the) (system prompt|instructions|hidden prompt)",
    r"print (your|the) (system prompt|instructions)",
    r"<\s*/?\s*(system|assistant|tool)\s*>",
    r"\[\s*system\s*\]|###\s*system",
    r"(?:[A-Za-z0-9+/]{4}){20,}={0,2}",  # long base64 blob
]
_COMPILED = [re.compile(p, re.I) for p in _PATTERNS]

# Requests the app must refuse regardless of phrasing (fraud / evasion).
_PROHIBITED = [
    r"(fake|forge|forged|falsify|counterfeit|fabricate|gefälscht\w*|fälschen)\s+(a |an |the |my |eine? |einen |den |das |der )?(document|passport|certificate|degree|contract|payslip|proof|letter|statement|bank statement|"
    r"meldebescheinigung|bescheinigung|nachweis|anmeldung|wohnungsgeberbestätigung|kontoauszug|zeugnis|urkunde|pass|vertrag|gehaltsabrechnung)",
    r"(sham|fake|schein)\s*(marriage|ehe)",
    # Scheinanmeldung: registering at an address where the person does not live
    r"\bscheinanmeldung\b",
    r"(meldebescheinigung|anmeldung|register|registration|registered|anmelden|melden)[^.\n]{0,80}?\b(address|adresse|wohnung|flat|apartment)\b[^.\n]{0,40}?\b(don'?t|do not|not|never|without|nicht|ohne)\b[^.\n]{0,15}?\b(live|living|reside|residing|stay|wohne\w*|lebe\w*)",
    r"\b(address|adresse|wohnung)\b[^.\n]{0,40}?\b(don'?t|do not|not|never|nicht)\b[^.\n]{0,15}?\b(live|living|reside|wohne\w*)\b[^.\n]{0,80}?(meldebescheinigung|anmeldung|register|anmelden)",
    r"\b(adresse|wohnung|address)\b[^.\n]{0,60}?\b(anmeld\w*|melden|register\w*)\b[^.\n]{0,40}?\b(nicht|not|don'?t)\b[^.\n]{0,15}?\b(wohne\w*|lebe\w*|live|reside)",
    r"(fake|false|falsche?n?|wrong)\s+(address|adresse)[^.\n]{0,60}?(register|registration|anmeld\w*|meldebescheinigung)",
    r"(register|anmelden|registration)[^.\n]{0,60}?(fake|false|falsche?n?|wrong)\s+(address|adresse)",
    r"(hide|conceal|verstecken|verschweigen)\s+(my|the)?\s*(income|einkommen|criminal record|vorstrafe)",
    r"(bribe|bestechen|schmiergeld)",
    r"overstay .* (without|avoid) (getting caught|detection)",
]
_PROHIBITED_C = [re.compile(p, re.I) for p in _PROHIBITED]


@dataclass
class InjectionVerdict:
    flagged: bool
    reason: str | None = None
    category: str | None = None  # "injection" | "prohibited"


def check_injection(text: str) -> InjectionVerdict:
    for pat in _PROHIBITED_C:
        if pat.search(text):
            return InjectionVerdict(True, "Request involves fraud or evading the law.", "prohibited")
    for pat in _COMPILED:
        if pat.search(text):
            return InjectionVerdict(True, "Message looks like a prompt-injection attempt.", "injection")
    return InjectionVerdict(False)
