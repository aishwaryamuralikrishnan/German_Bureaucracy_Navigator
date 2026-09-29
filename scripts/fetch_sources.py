"""Download the official pages listed in data/sources.yaml and save them as markdown.

Usage:  python scripts/fetch_sources.py
Then:   python scripts/ingest.py
"""

from __future__ import annotations

import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navigator.config import DATA_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

OUT_DIR = PROCESSED_DIR / "fetched"
HEADERS = {"User-Agent": "german-bureaucracy-navigator/0.1 (portfolio project; contact via GitHub)"}


def slugify(url: str) -> str:
    slug = re.sub(r"https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_")
    return slug[:100]


def main() -> int:
    try:
        import trafilatura
    except ImportError:
        print("trafilatura is not installed: pip install trafilatura")
        return 1

    sources = yaml.safe_load((DATA_DIR / "sources.yaml").read_text(encoding="utf-8"))["sources"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    for i, src in enumerate(sources, 1):
        url = src["url"]
        print(f"[{i}/{len(sources)}] {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"   ! download failed: {exc}")
            failed += 1
            continue
        (RAW_DIR / f"{slugify(url)}.html").write_text(resp.text, encoding="utf-8")
        md = trafilatura.extract(resp.text, output_format="markdown", include_links=False, include_tables=True, url=url)
        if not md or len(md) < 300:
            print("   ! extraction returned too little text, skipped")
            failed += 1
            continue
        meta = trafilatura.extract_metadata(resp.text, default_url=url)
        title = (meta.title if meta and meta.title else url).strip().replace('"', "'")
        front = {
            "title": title,
            "source_url": url,
            "topic": src.get("topic", "general"),
            "jurisdiction": src.get("jurisdiction", "federal"),
            "language": src.get("language", "de"),
            "fetched_at": date.today().isoformat(),
        }
        body = f"---\n{yaml.safe_dump(front, allow_unicode=True, sort_keys=False)}---\n\n# {title}\n\n{md}\n"
        (OUT_DIR / f"{slugify(url)}.md").write_text(body, encoding="utf-8")
        ok += 1
        time.sleep(1.0)  # be polite to public servers
    print(f"\nDone: {ok} saved, {failed} failed. Now run: python scripts/ingest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
