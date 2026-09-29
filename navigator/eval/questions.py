"""Evaluation question set: loading and validation of data/eval/questions*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from navigator.config import PROCESSED_DIR, PROJECT_ROOT

DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "eval" / "questions.yaml"
FULL_QUESTIONS = PROJECT_ROOT / "data" / "eval" / "questions_full.yaml"

Group = Literal["knowledge", "mixed", "tool_only", "not_covered"]


class EvalQuestion(BaseModel):
    id: str
    question: str
    language: str = "en"
    group: Group
    covered: bool
    coverage_check: bool = True
    expected_docs: list[str] = Field(default_factory=list)
    expected_section: str = ""
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_args: dict[str, dict[str, Any]] = Field(default_factory=dict)
    expected_facts: list[str] = Field(default_factory=list)
    probe: str | None = None
    difficulty: Literal["normal", "hard"] = "normal"
    notes: str = ""

    @field_validator("expected_args", mode="before")
    @classmethod
    def _none_to_dict(cls, v: Any) -> Any:
        return v or {}

    @property
    def is_probe(self) -> bool:
        return self.probe == "faithfulness"

    @property
    def is_hard(self) -> bool:
        return self.difficulty == "hard"


class QuestionSetError(ValueError):
    pass


def load_questions(path: str | Path | None = None, ids: list[str] | None = None, limit: int | None = None) -> list[EvalQuestion]:
    """Load, validate and optionally filter the question set (order preserved)."""
    p = Path(path) if path else DEFAULT_QUESTIONS
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise QuestionSetError(f"Question file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    items = raw.get("questions") or []
    questions = [EvalQuestion.model_validate(item) for item in items]
    validate_questions(questions)
    if ids:
        wanted = {i.strip().upper() for i in ids}
        questions = [q for q in questions if q.id.upper() in wanted]
        missing = wanted - {q.id.upper() for q in questions}
        if missing:
            raise QuestionSetError(f"Unknown question id(s): {', '.join(sorted(missing))}")
    if limit:
        questions = questions[:limit]
    return questions


def validate_questions(questions: list[EvalQuestion]) -> None:
    """Cross-checks that a typo in the YAML would otherwise turn into a silent zero score."""
    from navigator.tools import ALL_TOOLS  # local import: tools pull in the retriever module

    tool_schemas = {t.name: set(t.args_schema.model_fields) for t in ALL_TOOLS}
    known_docs = {p.name for p in PROCESSED_DIR.rglob("*.md")}
    seen: set[str] = set()
    problems: list[str] = []
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        for d in q.expected_docs:
            if d not in known_docs:
                problems.append(f"{q.id}: expected_docs has unknown file {d}")
        for t in q.expected_tools + q.forbidden_tools:
            if t not in tool_schemas:
                problems.append(f"{q.id}: unknown tool {t}")
        overlap = set(q.expected_tools) & set(q.forbidden_tools)
        if overlap:
            problems.append(f"{q.id}: tool(s) both expected and forbidden: {', '.join(sorted(overlap))}")
        for t, args in q.expected_args.items():
            if t not in q.expected_tools:
                problems.append(f"{q.id}: expected_args for {t} but it is not in expected_tools")
            elif t in tool_schemas:
                bad = set(args) - tool_schemas[t]
                if bad:
                    problems.append(f"{q.id}: {t} has no argument(s) {', '.join(sorted(bad))}")
        if q.coverage_check and "search_knowledge_base" not in q.expected_tools:
            problems.append(f"{q.id}: coverage_check is true but search_knowledge_base is not expected")
        if q.covered is False and q.expected_docs:
            problems.append(f"{q.id}: not covered but expected_docs is not empty")
    if problems:
        raise QuestionSetError("Invalid question set:\n  " + "\n  ".join(problems))
