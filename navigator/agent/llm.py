"""ChatOpenAI configured for OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from navigator.config import get_api_key, get_settings
from navigator.utils.errors import ConfigError


def get_chat_model(model: str | None = None, temperature: float | None = None, streaming: bool = True) -> ChatOpenAI:
    s = get_settings().llm
    key = get_api_key()
    if not key:
        raise ConfigError(
            f"Environment variable {s.api_key_env} is not set. Create a key at https://openrouter.ai/keys "
            f"and set it as a system environment variable (or in a .env file next to streamlit_app.py)."
        )
    return ChatOpenAI(
        model=model or s.models[0],
        api_key=key,
        base_url=s.base_url,
        temperature=s.temperature if temperature is None else temperature,
        timeout=s.timeout_seconds,
        max_retries=s.max_retries,
        streaming=streaming,
        default_headers={
            # Optional OpenRouter attribution headers (shown in their dashboard).
            "HTTP-Referer": "https://github.com/",
            "X-Title": "German Bureaucracy Navigator",
        },
    )
