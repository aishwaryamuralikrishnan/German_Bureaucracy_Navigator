"""Central configuration: loads config/settings.yaml once and resolves project paths."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"

# A .env file next to the project is optional; the system environment always wins.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(dict):
    """Dict with attribute-style access for nested settings: settings.llm.models."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Settings(value) if isinstance(value, dict) else value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    with SETTINGS_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Settings(raw)


def get_api_key() -> str | None:
    """Return the OpenRouter API key, or None.

    Order: system environment (or .env next to the project), then Streamlit's secrets store — that is how
    Streamlit Community Cloud passes secrets to the app (Settings → Secrets, `OPENROUTER_API_KEY = "..."`).
    """
    env_name = get_settings().llm.api_key_env
    key = os.environ.get(env_name, "").strip()
    if key:
        return key
    try:  # only inside a running Streamlit app; scripts and tests never reach st.secrets
        import streamlit as st

        key = str(st.secrets.get(env_name, "") or "").strip()
    except Exception:
        key = ""
    return key or None


def resolve_path(relative: str) -> Path:
    """Resolve a settings path relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
