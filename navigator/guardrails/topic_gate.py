"""Off-topic gate: cheap heuristic first, then a small LLM classification when unsure."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from navigator.config import get_settings
from navigator.utils.logging import get_logger

log = get_logger(__name__)

_ON_TOPIC_HINTS = re.compile(
    r"anmeld|regist|resid|permit|visa|blue ?card|blaue karte|aufenthalt|steuer|tax|finanzamt|krankenkasse|"
    r"insurance|versicherung|bank|konto|schufa|führerschein|driv|licen|anerkenn|recogni|degree|famil|spouse|"
    r"ehe|kind|child|elterngeld|kindergeld|bürgeramt|ausländer|immigra|germany|deutschland|berlin|munich|münchen|"
    r"hamburg|frankfurt|köln|cologne|stuttgart|salary|gehalt|netto|brutto|deadline|frist|termin|appointment|"
    r"document|dokument|passport|reisepass|move|umzug|apartment|wohnung|job|arbeit|employ|contract|vertrag|"
    r"rundfunk|gez|standesamt|birth|geburt|pension|rente|unemploy|arbeitslos|integration|sprach|language|"
    r"student|studium|studier|universit|sperrkonto|blocked account|werkstudent|minijob|intern|praktikum|semester|enrol|immatrikul|"
    r"currenc|exchange rate|convert|eur\b|euro|inr|usd|gbp|rupee|dollar|lakh|crore|wechselkurs|umrechn|"
    r"hello|hi\b|hallo|thanks|danke|help|hilfe|what can you|was kannst",
    re.I,
)

_CLASSIFIER_PROMPT = (
    "Classify whether the user message is about living in, moving to, working in, or dealing with public "
    "administration in Germany (registration, residence permits, taxes, insurance, banking, licences, "
    "recognition of qualifications, family matters, salaries, currency, deadlines) or a greeting / question "
    "about the assistant itself. Reply with exactly ON_TOPIC or OFF_TOPIC."
)


def is_on_topic(text: str, llm_factory=None) -> bool:
    cfg = get_settings().guardrails.topic_gate
    if _ON_TOPIC_HINTS.search(text):
        return True
    if not cfg.enabled:
        return True  # permissive when the gate is disabled and heuristics are silent
    try:
        if llm_factory is None:
            from navigator.agent.llm import get_chat_model

            llm_factory = get_chat_model
        llm = llm_factory(get_settings().llm.small_model, temperature=0.0)
        resp = llm.invoke([SystemMessage(content=_CLASSIFIER_PROMPT), HumanMessage(content=text[:1000])])
        verdict = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip().upper()
        return "OFF_TOPIC" not in verdict
    except Exception as exc:  # fail open: never block a user because the classifier is down
        log.warning(f"topic gate failed open: {exc}")
        return True
