"""Faithfulness judge: the share of an answer's claims that the retrieved evidence supports.

Two interchangeable backends produce the same result shape:

  ragas    — the RAGAS Faithfulness metric (statement generation + NLI verdicts) driven through
             OpenRouter's OpenAI-compatible endpoint. Default.
  builtin  — the same two-step procedure implemented with LangChain + the app's own prompts.
             Used automatically when RAGAS is not installed or fails, and in the tests (no network).

Per-claim verdicts are kept so the Evaluation page can show *which* sentence was unsupported.

Claim kinds. A faithful answer to a question the knowledge base does not cover consists almost entirely of
sentences that *cannot* be supported by the retrieved passages: "the knowledge base does not cover this" and
"ask the Familienkasse / see bamf.de". Scoring those as hallucinations would make the honest answer look worse
than a confident invented one. So every extracted statement is labelled fact / referral / meta (one cheap judge
call, regex fallback) and only *fact* statements enter the score; referral and meta statements are kept in the
report but marked "not scored". The unfiltered ratio is kept as raw_score for transparency.
"""

from __future__ import annotations

import json
import math
import re
import sys
import types
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from navigator.config import get_api_key, get_settings
from navigator.utils.logging import get_logger

log = get_logger(__name__)

_STATEMENTS_PROMPT = """Break the ANSWER into a list of short, self-contained factual statements (one fact each,
no pronouns, keep numbers and names exactly). Ignore greetings, disclaimers, generic advice such as
"contact the authority", and sentences that merely restate the QUESTION.
Reply with JSON only: {"statements": ["...", "..."]}"""

_KIND_PROMPT = """You label statements extracted from an assistant's answer about German bureaucracy. For each STATEMENT
choose exactly one label:
- "meta"     — it says what the knowledge base / the assistant does or does not cover, state or know ("The knowledge base
               does not state the fee"), or it is a generic disclaimer ("rules change every year", "confirm the details").
- "referral" — it only tells the user WHOM to ask or WHERE to look: names the responsible authority, office, hotline or
               website, says to contact / apply at / check with / verify with it, or gives a link — and states NOTHING
               about the topic itself. This includes statements ABOUT the authority rather than about the topic:
               which body is responsible ("the responsible authority is the Elterngeldstelle"), where it is ("in the
               user's area", "the local Bürgeramt"), what its name means or translates to ("the Bürgeramt is the
               citizens' office", "the Familienkasse is the family benefits office"), or where its website is.
- "fact"     — everything else: any statement about the topic — amounts, fees, rates, durations, deadlines, frequencies,
               conditions, eligibility, who must pay or do what, procedures, required documents or forms, how something
               works, what a law says. A referral that also carries such a fact ("apply at the Familienkasse with the
               birth certificate") is "fact".
Do not judge whether a statement is true. Reply with JSON only: {"kinds": ["fact", "referral", ...]} — exactly one label per
statement, in the given order."""

_VERDICT_PROMPT = """You are a strict natural-language-inference judge. For each STATEMENT decide whether it can be
directly inferred from the CONTEXT. Simple arithmetic on numbers present in the context counts as inferable.
Do not use outside knowledge: a statement that is true in the real world but absent from the context is 0.
Reply with JSON only: {"verdicts": [{"statement": "...", "verdict": 1, "reason": "..."}, ...]} where verdict is
1 (supported by the context) or 0 (not supported)."""


CLAIM_KINDS = ("fact", "referral", "meta")


@dataclass
class Claim:
    statement: str
    verdict: int            # 1 supported, 0 unsupported
    reason: str = ""
    kind: str = "fact"      # fact | referral | meta — only "fact" statements are scored

    @property
    def scored(self) -> bool:
        return self.kind == "fact"


@dataclass
class FaithfulnessResult:
    score: float | None      # share of supported *fact* statements; None when the judge failed or produced no statements
    claims: list[Claim] = field(default_factory=list)
    backend: str = "none"
    judge_model: str = ""
    error: str | None = None
    classifier: str = "none"  # how the kinds were assigned: llm | heuristic | none

    @property
    def facts(self) -> list[Claim]:
        return [c for c in self.claims if c.scored]

    @property
    def supported(self) -> int:
        return sum(c.verdict for c in self.facts)

    @property
    def total(self) -> int:
        return len(self.facts)

    @property
    def unscored(self) -> int:
        return len(self.claims) - len(self.facts)

    @property
    def leaks(self) -> list[Claim]:
        """Fact statements the evidence does not support — what a user would take as knowledge-base-backed but is not."""
        return [c for c in self.facts if not c.verdict]

    @property
    def raw_score(self) -> float | None:
        return _ratio(self.claims)

    def to_dict(self) -> dict:
        return {"score": self.score, "supported": self.supported, "total": self.total, "statements": len(self.claims),
                "unscored": self.unscored, "raw_score": self.raw_score, "classifier": self.classifier, "backend": self.backend,
                "judge_model": self.judge_model, "error": self.error, "claims": [asdict(c) for c in self.claims]}


# ----------------------------------------------------------------------------- claim kinds (regex fallback)

_URL_RE = re.compile(r"https?://|www\.", re.I)
_META_RE = re.compile(
    r"((knowledge base|Wissensdatenbank|retrieved (passages|sources)|available information)\W.{0,60}?"
    r"(does not|doesn't|do not|did not|has no|have no|lacks?|no |not |only |nicht|keine|kein )"
    r"|(does not|doesn't|no |not |nicht|keine)\W.{0,60}?(knowledge base|Wissensdatenbank)"
    r"|not covered|no specific (information|details|passage)|keine (spezifischen )?(Informationen|Angaben)"
    r"|rules?( may| can)? change|regulations?( may| can)? change|(kann|können) sich ändern|subject to change"
    r"|(confirm|verify|check) (this|the details|the current rules|with the))", re.I)
_REFERRAL_RE = re.compile(
    r"\b(contact(ing)?|consult(ed|ing)?|ask(ing)?|reach out|inquir(e|ing)|enquir(e|ing)|check(ing)? with|confirm(ing)? with|verify(ing)? with"
    r"|recommend(ed|s)?|advis(e|ed|able)|visit(ing)?|website|web ?page|official (site|page|source|portal)|homepage|hotline"
    r"|for (more|further|detailed|accurate|up-to-date|current|specific) (information|details|guidance|requirements)|guidance"
    r"|responsible (authority|office)|competent authority|is responsible for|in charge of"
    r"|wenden Sie sich|kontaktieren|erkundigen|nachfragen|informieren Sie sich|Website|Webseite|Auskunft|weitere Informationen"
    r"|zuständige[nrs]? (Behörde|Stelle|Amt)|ist zuständig)\b", re.I)
_FACT_SIGNAL_RE = re.compile(
    r"(\d|€|EUR|Euro|percent|%|\b(month|year|week|day|monthly|yearly|annual|Monat|Jahr|Woche|Tag|monatlich|jährlich)\w*"
    r"|\b(must|required|need(s|ed)? to|have to|eligib|allowed|entitled|obliged|mandatory|document|form|proof|certificate|application"
    r"|müssen|muss|Pflicht|Nachweis|Formular|Antrag|Anspruch|berechtigt|erlaubt)\w*)", re.I)


# Statements about the authority itself — "the responsible authority is X", "X is the citizens' office" — are what the
# statement extractor produces when it atomises "contact the Bürgeramt (citizens' office)". They are referral content
# whatever the classifier says, as long as they carry no fact signal (no number, obligation, document, deadline …).
_GLOSS_RE = re.compile(
    r"(\b(responsible|competent|relevant|zuständige?) (authority|office|body|agency|Behörde|Stelle|Amt)\b.{0,12}\b(is|ist|are)\b"
    r"|\b(is|ist) (the|die|das|der|your|ihre?) (local |responsible |competent |zuständige? )?(authority|office|body|agency|point of contact|Behörde|Stelle|Amt|Ansprechpartner)\b"
    r"|^(the |die |das |der |ihre? |your |a )?[\w\-]*(amt|behörde|kasse|stelle|versicherung|agentur|ministerium|portal|website|office|authority|agency|department|service)\b.{0,30}?\b(is|ist|means|bedeutet|translates|heißt|stands for|refers to|is called|is known as)\b)",
    re.I)


def gloss_referral(statement: str) -> bool:
    """True for statements that only say which authority is responsible / what its name means."""
    s = statement.strip()
    return bool(_GLOSS_RE.search(s)) and not _FACT_SIGNAL_RE.search(s)


def heuristic_kind(statement: str) -> str:
    """Regex stand-in for the judge's classification, used when the judge call fails."""
    s = statement.strip()
    if _META_RE.search(s):
        return "meta"
    if _URL_RE.search(s) and not _FACT_SIGNAL_RE.search(s):
        return "referral"
    if _REFERRAL_RE.search(s) and not _FACT_SIGNAL_RE.search(s):
        return "referral"
    if gloss_referral(s):
        return "referral"
    return "fact"


def _score_facts(claims: list[Claim]) -> float | None:
    """Score over fact statements only. An answer whose statements are all referrals/meta presents no fact as
    knowledge-base-backed, so nothing in it can be unfaithful: 1.0 (not None, which would mean 'judge failed')."""
    if not claims:
        return None
    facts = [c for c in claims if c.scored]
    return 1.0 if not facts else _ratio(facts)


def _shim_ragas_imports() -> None:
    """RAGAS 0.4 imports a Vertex AI class that langchain-community 0.4 no longer ships; provide a stand-in."""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        shim = types.ModuleType(name)
        shim.ChatVertexAI = type("ChatVertexAI", (), {})  # never instantiated
        sys.modules[name] = shim


def describe_error(exc: BaseException) -> str:
    """Timeouts often carry an empty message (str(TimeoutError()) == ''); always show the type and the cause chain."""
    parts = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur).strip()
        parts.append(f"{type(cur).__name__}: {msg}" if msg else type(cur).__name__)
        cur = cur.__cause__ or cur.__context__
    return " <- ".join(parts[:3])


def _parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError("judge returned no JSON object")
    return json.loads(m.group(0))


class FaithfulnessJudge:
    def __init__(self, model: str | None = None, backend: str = "auto", llm_factory: Callable[..., Any] | None = None,
                 timeout: float = 90.0, max_tokens: int = 8192) -> None:
        s = get_settings()
        self.model = model or s.llm.small_model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.requested_backend = backend
        self._llm_factory = llm_factory          # tests inject a fake LangChain chat model here
        self._ragas_metric = None
        self.fallback_reason: str | None = None  # set when RAGAS was wanted but the built-in judge had to take over
        self.backend = self._select_backend()

    # ------------------------------------------------------------------ backends
    def _select_backend(self) -> str:
        if self.requested_backend == "builtin" or self._llm_factory is not None:
            return "builtin"
        try:
            self._ragas_metric = self._build_ragas()
            return "ragas"
        except Exception as exc:
            if self.requested_backend == "ragas":
                raise
            self.fallback_reason = f"RAGAS could not be initialised: {describe_error(exc)}"
            log.warning(f"RAGAS unavailable ({describe_error(exc)}); using the built-in faithfulness judge")
            return "builtin"

    def _build_ragas(self):
        _shim_ragas_imports()
        import httpx
        from openai import OpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness

        s = get_settings().llm
        # Synchronous client on purpose: no event loop to manage from worker threads, and the openai client's own
        # retries cover transient timeouts. Short keep-alive expiry avoids the "first request after a pause hangs
        # for the whole timeout" symptom caused by stale pooled connections.
        http_client = httpx.Client(
            timeout=httpx.Timeout(float(self.timeout), connect=15.0),
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=10.0),
        )
        client = OpenAI(api_key=get_api_key(), base_url=s.base_url, timeout=float(self.timeout), max_retries=2, http_client=http_client,
                        default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "German Bureaucracy Navigator (evaluation judge)"})
        # RAGAS's instructor LLM defaults to max_tokens=1024. The NLI step must repeat every statement verbatim plus a
        # reason, which for a long answer exceeds that and raises IncompleteOutputException (output cut off).
        llm = llm_factory(self.model, provider="openai", client=client, temperature=0.0, max_tokens=self.max_tokens)
        return Faithfulness(llm=llm)

    def _chat_model(self):
        if self._llm_factory is not None:
            return self._llm_factory(self.model, temperature=0.0)
        from navigator.agent.llm import get_chat_model

        return get_chat_model(self.model, temperature=0.0, streaming=False).bind(timeout=float(self.timeout), max_tokens=self.max_tokens)

    # ------------------------------------------------------------------ scoring
    def score(self, question: str, answer: str, contexts: list[str]) -> FaithfulnessResult:
        if not answer.strip() or not any(c.strip() for c in contexts):
            return FaithfulnessResult(score=None, backend=self.backend, judge_model=self.model, error="empty answer or evidence")
        res = self._verdicts(question, answer, contexts)
        if res.claims:
            self.classify(question, res)
        return res

    def classify(self, question: str, res: FaithfulnessResult) -> None:
        """Label every statement fact / referral / meta and rescore over the fact statements. Falls back to regexes."""
        try:
            kinds = self._classify_llm(question, [c.statement for c in res.claims])
            res.classifier = "llm"
        except Exception as exc:
            log.warning(f"claim classification failed ({describe_error(exc)}); using the regex heuristic")
            kinds = [heuristic_kind(c.statement) for c in res.claims]
            res.classifier = "heuristic"
        for c, k in zip(res.claims, kinds):
            # rule override: a "fact" that only says which office is responsible or what its name means is a referral
            c.kind = "referral" if (k == "fact" and gloss_referral(c.statement)) else k
        res.score = _score_facts(res.claims)

    def _classify_llm(self, question: str, statements: list[str]) -> list[str]:
        llm = self._chat_model()
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
        r = llm.invoke([SystemMessage(content=_KIND_PROMPT), HumanMessage(content=f"QUESTION:\n{question}\n\nSTATEMENTS:\n{numbered}")])
        kinds = _parse_json(_text(r)).get("kinds")
        if not isinstance(kinds, list) or len(kinds) != len(statements):
            raise ValueError(f"expected {len(statements)} labels, got {kinds!r}")
        out = []
        for k in kinds:
            k = str(k).strip().lower()
            if k not in CLAIM_KINDS:
                raise ValueError(f"unknown label {k!r}")
            out.append(k)
        return out

    def _verdicts(self, question: str, answer: str, contexts: list[str]) -> FaithfulnessResult:
        if self.backend == "ragas":
            try:
                return self._score_ragas(question, answer, contexts)
            except Exception as exc:
                reason = describe_error(exc)
                if self.requested_backend == "ragas":
                    log.warning(f"RAGAS judge failed: {reason}", exc_info=True)
                    return FaithfulnessResult(score=None, backend="ragas", judge_model=self.model, error=reason)
                # auto mode: RAGAS misbehaves with this model/endpoint → switch to the built-in judge for the rest of the run
                self.fallback_reason = f"RAGAS failed on a live call: {reason}"
                log.warning(f"RAGAS judge failed ({reason}); switching to the built-in judge", exc_info=True)
                self.backend = "builtin"
        try:
            return self._score_builtin(question, answer, contexts)
        except Exception as exc:
            log.warning(f"faithfulness judge failed: {describe_error(exc)}")
            return FaithfulnessResult(score=None, backend=self.backend, judge_model=self.model, error=describe_error(exc))

    def _score_ragas(self, question: str, answer: str, contexts: list[str]) -> FaithfulnessResult:
        """RAGAS's own prompts and output schemas, executed synchronously through its instructor LLM."""
        from ragas.metrics.collections.faithfulness.util import (
            NLIStatementInput, NLIStatementOutput, StatementGeneratorInput, StatementGeneratorOutput,
        )

        metric = self._ragas_metric
        gen_prompt = metric.statement_generator_prompt.to_string(StatementGeneratorInput(question=question, answer=answer))
        statements = list(metric.llm.generate(gen_prompt, StatementGeneratorOutput).statements)
        if not statements:
            return FaithfulnessResult(score=None, backend="ragas", judge_model=self.model, error="no statements extracted")
        nli_prompt = metric.nli_statement_prompt.to_string(NLIStatementInput(context="\n".join(contexts), statements=statements))
        verdicts = metric.llm.generate(nli_prompt, NLIStatementOutput)
        if not verdicts.statements:
            return FaithfulnessResult(score=None, backend="ragas", judge_model=self.model, error="no verdicts returned")
        claims = [Claim(statement=v.statement, verdict=1 if int(v.verdict) else 0, reason=v.reason) for v in verdicts.statements]
        return FaithfulnessResult(score=_ratio(claims), claims=claims, backend="ragas", judge_model=self.model)

    def _score_builtin(self, question: str, answer: str, contexts: list[str]) -> FaithfulnessResult:
        llm = self._chat_model()
        r1 = llm.invoke([SystemMessage(content=_STATEMENTS_PROMPT), HumanMessage(content=f"QUESTION:\n{question}\n\nANSWER:\n{answer}")])
        statements = [str(s).strip() for s in _parse_json(_text(r1)).get("statements", []) if str(s).strip()]
        if not statements:
            return FaithfulnessResult(score=None, backend="builtin", judge_model=self.model, error="no statements extracted")
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
        r2 = llm.invoke([SystemMessage(content=_VERDICT_PROMPT), HumanMessage(content=f"CONTEXT:\n{chr(10).join(contexts)}\n\nSTATEMENTS:\n{numbered}")])
        raw = _parse_json(_text(r2)).get("verdicts", [])
        claims: list[Claim] = []
        for i, s in enumerate(statements):
            v = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
            claims.append(Claim(statement=str(v.get("statement") or s), verdict=1 if int(v.get("verdict", 0)) else 0, reason=str(v.get("reason", ""))))
        return FaithfulnessResult(score=_ratio(claims), claims=claims, backend="builtin", judge_model=self.model)


def _text(msg: Any) -> str:
    c = getattr(msg, "content", msg)
    return c if isinstance(c, str) else json.dumps(c)


def _ratio(claims: list[Claim]) -> float | None:
    if not claims:
        return None
    v = sum(c.verdict for c in claims) / len(claims)
    return None if math.isnan(v) else round(v, 4)
