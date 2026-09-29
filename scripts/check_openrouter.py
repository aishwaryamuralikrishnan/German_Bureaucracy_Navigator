"""Check the OpenRouter connection: key, credits, chat model and embedding model.

Usage:  python scripts/check_openrouter.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navigator.config import get_api_key, get_settings  # noqa: E402


def main() -> int:
    s = get_settings()
    key = get_api_key()
    if not key:
        print(f"FAIL  {s.llm.api_key_env} is not set in this terminal. Set it (restart VS Code) or create a .env file.")
        return 1
    print(f"ok    API key found ({key[:8]}…{key[-4:]}), base_url {s.llm.base_url}")

    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

    client = OpenAI(api_key=key, base_url=s.llm.base_url, timeout=30, max_retries=0)

    # 1. account / credits (OpenRouter-specific endpoint)
    try:
        import requests

        r = requests.get("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        if r.ok:
            d = r.json().get("data", {})
            print(f"ok    key valid — usage so far ${d.get('usage', 0):.4f}, limit {d.get('limit')}, free tier: {d.get('is_free_tier')}")
        else:
            print(f"WARN  /auth/key returned HTTP {r.status_code}: {r.text[:120]}")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN  could not query /auth/key: {exc}")

    # 2. embeddings
    t = time.time()
    try:
        resp = client.embeddings.create(model=s.embeddings.model, input=["ping"], encoding_format="float")
        print(f"ok    embeddings {s.embeddings.model}: {len(resp.data[0].embedding)} dims in {time.time() - t:.1f}s")
    except APIStatusError as exc:
        hint = {401: "invalid key", 402: "no credits — top up at https://openrouter.ai/credits", 404: "model id not found", 429: "rate limited"}.get(exc.status_code, "")
        print(f"FAIL  embeddings HTTP {exc.status_code} {hint}: {exc.message[:160]}")
    except (APITimeoutError, APIConnectionError) as exc:
        print(f"FAIL  embeddings: {exc.__class__.__name__} — network/proxy problem or service stalled")

    # 2b. rerank endpoint (used by the retriever when retrieval.reranker.provider == openrouter)
    rr = s.retrieval.reranker
    if rr.get("enabled") and str(rr.get("provider", "openrouter")) == "openrouter":
        t = time.time()
        try:
            r = requests.post(
                s.llm.base_url.rstrip("/") + "/rerank",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": rr.model, "query": "Anmeldung deadline", "documents": ["Register within 14 days.", "Chocolate cake recipe."], "top_n": 2},
                timeout=30,
            )
            if r.ok:
                top = r.json()["results"][0]
                print(f"ok    rerank {rr.model}: top index {top['index']} score {top['relevance_score']:.3f} in {time.time() - t:.1f}s")
            else:
                print(f"FAIL  rerank HTTP {r.status_code}: {r.text[:160]} — set retrieval.reranker.provider: none to disable")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  rerank: {exc}")

    # 3. chat model
    t = time.time()
    try:
        resp = client.chat.completions.create(model=s.llm.models[0], messages=[{"role": "user", "content": "Say ok."}], max_tokens=5)
        print(f"ok    chat {s.llm.models[0]}: '{resp.choices[0].message.content.strip()}' in {time.time() - t:.1f}s")
    except APIStatusError as exc:
        print(f"FAIL  chat HTTP {exc.status_code}: {exc.message[:160]}")
    except (APITimeoutError, APIConnectionError) as exc:
        print(f"FAIL  chat: {exc.__class__.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
