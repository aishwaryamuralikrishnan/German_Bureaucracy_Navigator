"""LangChain tools exposed to the agent."""

from navigator.tools.blue_card import check_blue_card_salary
from navigator.tools.currency import convert_currency
from navigator.tools.deadlines import calculate_deadline
from navigator.tools.net_salary import estimate_net_salary
from navigator.tools.search_kb import search_knowledge_base

ALL_TOOLS = [
    search_knowledge_base,
    check_blue_card_salary,
    calculate_deadline,
    estimate_net_salary,
    convert_currency,
]

TOOL_CATEGORIES = {
    "search_knowledge_base": "retrieval",
    "check_blue_card_salary": "calculator",
    "calculate_deadline": "calculator",
    "estimate_net_salary": "calculator",
    "convert_currency": "live API",
}

__all__ = ["ALL_TOOLS", "TOOL_CATEGORIES"]
