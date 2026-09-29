"""Grounding check: flag concrete claims in the answer that no retrieved passage or tool result supports.

The retriever and the citation checks guarantee that *cited* passages exist, but they cannot see a sentence
such as "the letter is typically valid for one year" that the model added from memory without a citation.
This post-check asks a small model to compare the answer with the exact evidence the agent saw in this turn
and to list unsupported factual claims (numbers, durations, fees, deadlines, legal conditions). With
guardrails.grounding_check.revise enabled, a second call rewrites the answer so those claims are removed or
explicitly marked as not from the knowledge base (apply_grounding); otherwise they are only shown as a warning.
Both steps fail open: a failing check or revision leaves the answer untouched.

Figure guard (navigator/guardrails/figures.py). The judge is a small model and produces false positives when a
figure is merely *written differently* in the answer and in the evidence ("€50,700" vs 50700.0, "February 3, 2027"
vs "2027-02-03", "one year" vs "1 year"); a revision built on such a false positive deletes a correct number —
usually the one the user asked for. So (1) a flagged claim whose figures all appear in the evidence is ignored,
and (2) a revision that would drop any figure the evidence contains is rejected and the answer is kept, with the
warning note instead. The guard is deterministic and only ever makes the revision more conservative.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from navigator.config import get_settings
from navigator.guardrails.figures import figures, figures_missing_from
from navigator.utils.logging import get_logger

log = get_logger(__name__)

_JUDGE_PROMPT = """You are a strict fact-checker for a German-bureaucracy assistant.

You receive EVIDENCE (knowledge-base passages and tool results the assistant retrieved in this turn) and the
assistant's ANSWER. List every concrete factual claim in the ANSWER that is NOT supported by the EVIDENCE:
numbers, amounts, fees, percentages, durations, frequencies ("monthly"), validity periods, deadlines, dates,
legal conditions, eligibility ("every household", "regardless of …"), who must pay or do what, procedures and
application steps (where and how to apply), required documents or forms, names of laws, and statements about
what is or is not allowed.

Rules:
- A claim is supported if the EVIDENCE states it or it follows directly from the evidence (simple arithmetic
  on tool results counts as supported).
- Ignore: greetings, disclaimers, restating the user's own numbers, sentences that explicitly say the knowledge
  base does not cover something, sentences explicitly labelled as not from the knowledge base / general
  knowledge, and pure referrals that name WHOM to ask or WHERE to look ("contact the Familienkasse",
  "see bamf.de") without saying anything about the topic itself. A referral that carries a fact ("apply at the
  Familienkasse with the birth certificate") is a claim.
- Do not judge whether the claim is true in the real world — only whether the EVIDENCE supports it.
- Quote each unsupported claim briefly (max 20 words), in the language of the answer.

Reply with JSON only: {"unsupported": ["claim 1", "claim 2"]} — or {"unsupported": []} if everything is supported."""

# Three searches × 5 passages of ~1,800 chars plus tool results exceed 24k; a truncated evidence block makes the
# judge flag claims whose passage it never saw. 60k chars ≈ 15k tokens is well within the small model's window.
_MAX_EVIDENCE_CHARS = 60_000
_MAX_ANSWER_CHARS = 6_000

NOTE_PREFIX = {
    "en": "Not backed by the retrieved sources — please verify with the responsible authority:",
    "de": "Nicht durch die abgerufenen Quellen belegt — bitte bei der zuständigen Behörde prüfen:",
}


@dataclass
class GroundingReport:
    checked: bool
    unsupported: list[str] = field(default_factory=list)
    error: str | None = None
    ignored: list[str] = field(default_factory=list)   # flagged by the judge, but every figure in them is in the evidence

    @property
    def ok(self) -> bool:
        return self.checked and not self.unsupported

    def note(self, language: str, max_shown: int = 4) -> str | None:
        if not self.unsupported:
            return None
        return f"{NOTE_PREFIX.get(language, NOTE_PREFIX['en'])}\n{_bullets(self.unsupported, max_shown)}"


def _bullets(items: list[str], max_shown: int = 4) -> str:
    shown = "\n".join(f"- {c}" for c in items[:max_shown])
    more = len(items) - max_shown
    return shown + (f"\n- … and {more} more" if more > 0 else "")


def evidence_from_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Serialise what the agent actually saw: passage texts and the data of every other successful tool."""
    parts: list[str] = []
    for call in tool_calls:
        res = call.get("result")
        if not isinstance(res, dict) or not res.get("ok", True):
            continue
        data = res.get("data") or {}
        if call.get("name") == "search_knowledge_base":
            for p in data.get("passages", []) or []:
                parts.append(f"[{p.get('ref')}] {p.get('title')} — {p.get('section') or ''}\n{p.get('text', '')}")
        else:
            payload = {"data": data, "warnings": res.get("warnings", [])}
            parts.append(f"TOOL {call.get('name')}({json.dumps(call.get('args') or {}, ensure_ascii=False)}):\n"
                         f"{json.dumps(payload, ensure_ascii=False, default=str)}")
    text = "\n\n".join(parts)
    return text[:_MAX_EVIDENCE_CHARS]


def verify_unsupported(claims: list[str], evidence: str) -> tuple[list[str], list[str]]:
    """Split the judge's list into (kept, ignored): a claim that carries figures, all of which appear in the
    evidence, is a false positive — the evidence supports it by construction. Claims without figures are kept
    (nothing to verify deterministically)."""
    ev = figures(evidence)
    kept, ignored = [], []
    for c in claims:
        if figures(c) and not figures_missing_from(c, ev):
            ignored.append(c)
        else:
            kept.append(c)
    return kept, ignored


def lost_supported_figures(original: str, revised: str, evidence: str) -> set[str]:
    """Figures that the evidence supports and the original answer contained, but the revision no longer does."""
    supported = figures(original) & figures(evidence)
    return supported - figures(revised)


def _parse_claims(raw: str, max_claims: int) -> list[str]:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError("judge returned no JSON object")
    obj = json.loads(m.group(0))
    claims = obj.get("unsupported", [])
    if not isinstance(claims, list):
        raise ValueError("'unsupported' is not a list")
    out: list[str] = []
    for c in claims:
        s = str(c).strip().strip('"').strip()
        if s and s not in out:
            out.append(s[:200])
    return out[:max_claims]


def check_grounding(
    answer: str,
    tool_calls: list[dict[str, Any]],
    llm_factory: Callable[..., Any] | None = None,
    evidence: str | None = None,
) -> GroundingReport:
    """Compare the final answer with the evidence of this turn. Fails open."""
    cfg = get_settings().guardrails.get("grounding_check", {}) or {}
    if not cfg.get("enabled", False):
        return GroundingReport(checked=False)
    if evidence is None:
        evidence = evidence_from_tool_calls(tool_calls)
    if not evidence.strip() or not answer.strip():
        return GroundingReport(checked=False)

    try:
        if llm_factory is None:
            from navigator.agent.llm import get_chat_model

            llm_factory = get_chat_model
        llm = llm_factory(cfg.get("model") or get_settings().llm.small_model, temperature=0.0)
        user = f"EVIDENCE:\n{evidence}\n\nANSWER:\n{answer[:_MAX_ANSWER_CHARS]}"
        resp = llm.invoke([SystemMessage(content=_JUDGE_PROMPT), HumanMessage(content=user)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        claims = _parse_claims(raw, int(cfg.get("max_claims", 12)))
        claims, ignored = verify_unsupported(claims, evidence)
        if ignored:
            log.info(f"grounding guard: {len(ignored)} flagged claim(s) ignored — their figures appear in the evidence: {ignored}")
        return GroundingReport(checked=True, unsupported=claims, ignored=ignored)
    except Exception as exc:  # never block or alter an answer because the judge failed
        log.warning(f"grounding check failed open: {exc}")
        return GroundingReport(checked=False, error=str(exc))


# ----------------------------------------------------------------------------- revision

_REVISE_PROMPT = """You revise an assistant's answer so that it contains only facts the retrieved evidence supports.
You receive the ANSWER and a list of UNSUPPORTED claims — statements the evidence does not contain.

Rewrite the answer:
- Remove every unsupported claim. Numbers, fees, amounts, durations, deadlines and legal conditions that are
  unsupported must disappear entirely — do not replace them with approximations.
- If an unsupported statement is only general orientation (e.g. which authority to ask), you may keep it in its own
  sentence that starts with "Not from the knowledge base:" — without any numbers or dates.
- Keep every supported sentence as it is, including its [S1]-style citations. Do not add anything new.
- Keep the answer's language and formatting. If little remains, say plainly that the knowledge base does not contain
  these details and point to the responsible authority the answer already names.
Return only the revised answer text, nothing else."""

REVISION_NOTE = {
    "en": "The answer was revised: {n} statement(s) that no retrieved passage or tool result supports were removed or marked as not from the knowledge base.",
    "de": "Die Antwort wurde überarbeitet: {n} Aussage(n), die keine abgerufene Passage und kein Tool-Ergebnis belegt, wurden entfernt oder als nicht aus der Wissensdatenbank stammend markiert.",
}


@dataclass
class GroundingOutcome:
    text: str                       # final answer text (revised if a revision happened)
    report: GroundingReport
    revised: bool = False
    original_text: str | None = None
    note: str | None = None         # user-facing note to show under the answer, if any
    revision_rejected: str | None = None   # why a produced revision was not used (guard)

    def to_dict(self) -> dict:
        return {"checked": self.report.checked, "unsupported": self.report.unsupported, "ignored": self.report.ignored,
                "revised": self.revised, "revision_rejected": self.revision_rejected, "error": self.report.error}


def revise_answer(answer: str, unsupported: list[str], llm_factory: Callable[..., Any] | None = None) -> str | None:
    """Ask the small model to drop or label the unsupported claims. Returns None when the revision failed or is unusable."""
    cfg = get_settings().guardrails.get("grounding_check", {}) or {}
    try:
        if llm_factory is None:
            from navigator.agent.llm import get_chat_model

            llm_factory = get_chat_model
        llm = llm_factory(cfg.get("model") or get_settings().llm.small_model, temperature=0.0)
        bullets = "\n".join(f"- {c}" for c in unsupported)
        resp = llm.invoke([SystemMessage(content=_REVISE_PROMPT), HumanMessage(content=f"ANSWER:\n{answer[:_MAX_ANSWER_CHARS]}\n\nUNSUPPORTED:\n{bullets}")])
        text = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip()
        # sanity: a revision must be a real text and must not have grown into something else entirely
        if len(text) < 20 or len(text) > 2 * len(answer) + 500:
            return None
        return text
    except Exception as exc:
        log.warning(f"grounding revision failed open: {exc}")
        return None


def apply_grounding(
    answer: str,
    tool_calls: list[dict[str, Any]],
    language: str = "en",
    llm_factory: Callable[..., Any] | None = None,
) -> GroundingOutcome:
    """Check the answer against the evidence and, when configured, revise it so unsupported claims do not reach the user.

    Used by the app and by the evaluation harness alike, so the evaluation measures what a user would see.
    """
    cfg = get_settings().guardrails.get("grounding_check", {}) or {}
    evidence = evidence_from_tool_calls(tool_calls)
    report = check_grounding(answer, tool_calls, llm_factory=llm_factory, evidence=evidence)
    if not report.checked or not report.unsupported:
        return GroundingOutcome(text=answer, report=report)
    if not cfg.get("revise", False):
        return GroundingOutcome(text=answer, report=report, note=report.note(language))
    revised = revise_answer(answer, report.unsupported, llm_factory=llm_factory)
    if revised is None or revised.strip() == answer.strip():
        return GroundingOutcome(text=answer, report=report, note=report.note(language))
    lost = lost_supported_figures(answer, revised, evidence)
    if lost:
        # the reviser removed more than it was asked to: figures the evidence backs. Keep the original answer and
        # fall back to the warning note — never trade a correct number for a cleaner-looking text.
        reason = "revision dropped figure(s) the evidence supports: " + ", ".join(sorted(lost))
        log.warning(f"grounding guard: {reason}")
        return GroundingOutcome(text=answer, report=report, note=report.note(language), revision_rejected=reason)
    n = len(report.unsupported)
    return GroundingOutcome(text=revised, report=report, revised=True, original_text=answer,
                            note=REVISION_NOTE.get(language, REVISION_NOTE["en"]).format(n=n) + "\n" + _bullets(report.unsupported))
