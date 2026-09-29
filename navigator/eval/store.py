"""Evaluation runs on disk: reports/eval/<timestamp>/{meta,retrieval,e2e}.json + summary.md + report.pdf."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from navigator.config import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "reports" / "eval"


@dataclass
class EvalRun:
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] | None = None
    e2e: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        bits = [self.name]
        if self.e2e:
            bits.append(self.e2e.get("model", "").split("/")[-1])
            if self.e2e.get("judge_model"):
                bits.append("judge " + self.e2e["judge_model"].split("/")[-1])
        if self.meta.get("question_file"):
            bits.append(Path(self.meta["question_file"]).name)
        return " · ".join(b for b in bits if b)

    @property
    def pdf_path(self) -> Path:
        return self.path / "report.pdf"


def new_run_dir(root: Path | None = None) -> Path:
    base = root or RUNS_DIR
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    path = base / stamp
    n = 1
    while path.exists():
        n += 1
        path = base / f"{stamp}_{n}"
    path.mkdir()
    return path


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_meta(run_dir: Path, **fields: Any) -> dict[str, Any]:
    meta = _read(run_dir / "meta.json") or {}
    meta.update(fields)
    meta.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    write_json(run_dir / "meta.json", meta)
    return meta


def load_run(path: Path) -> EvalRun:
    return EvalRun(path=path, meta=_read(path / "meta.json") or {}, retrieval=_read(path / "retrieval.json"), e2e=_read(path / "e2e.json"))


def list_runs(root: Path | None = None) -> list[EvalRun]:
    base = root or RUNS_DIR
    if not base.exists():
        return []
    runs = [load_run(p) for p in sorted(base.iterdir(), reverse=True) if p.is_dir() and ((p / "retrieval.json").exists() or (p / "e2e.json").exists())]
    return runs


def latest_run(root: Path | None = None) -> EvalRun | None:
    runs = list_runs(root)
    return runs[0] if runs else None
