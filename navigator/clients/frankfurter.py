"""Frankfurter — free exchange-rate API serving European Central Bank reference rates. No API key.

Docs: https://frankfurter.dev  (v1 endpoints: /v1/latest, /v1/{date}, /v1/currencies)
The ECB publishes rates for ~30 currencies once per working day (around 16:00 CET).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import date

import requests

from navigator.config import get_settings
from navigator.utils.errors import ExternalServiceError, ValidationError
from navigator.utils.logging import get_logger, log_event

log = get_logger(__name__)


@dataclass(frozen=True)
class Rate:
    base: str
    quote: str
    rate: float
    rate_date: str          # ISO date of the ECB fixing actually used
    requested_date: str | None


class FrankfurterClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        cfg = get_settings().get("frankfurter", {}) or {}
        self.base_urls: list[str] = list(cfg.get("base_urls") or ["https://api.frankfurter.dev/v1", "https://api.frankfurter.app"])
        self.timeout = float(cfg.get("timeout_seconds", 10))
        self.ttl = float(cfg.get("cache_ttl_seconds", 6 * 3600))
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "german-bureaucracy-navigator/0.1", "Accept": "application/json"})
        self._cache: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _cached(self, key: str):
        with self._lock:
            item = self._cache.get(key)
            if item and time.time() - item[0] < self.ttl:
                return item[1]
        return None

    def _store(self, key: str, value) -> None:
        with self._lock:
            self._cache[key] = (time.time(), value)

    def _get(self, path: str, params: dict | None = None) -> dict:
        last: Exception | None = None
        for base in self.base_urls:
            url = f"{base.rstrip('/')}/{path.lstrip('/')}"
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.Timeout as exc:
                last = ExternalServiceError(f"{base} timed out"); continue
            except requests.ConnectionError as exc:
                last = ExternalServiceError(f"could not connect to {base}"); continue
            if resp.status_code == 404:
                # Frankfurter answers 404 for unknown currencies / dates — a client error, don't try mirrors
                raise ValidationError(f"Frankfurter has no data for this request ({resp.text[:120]})")
            if resp.status_code >= 400:
                last = ExternalServiceError(f"{base} returned HTTP {resp.status_code}"); continue
            try:
                return resp.json()
            except ValueError:
                last = ExternalServiceError(f"{base} returned malformed JSON"); continue
        raise last or ExternalServiceError("exchange-rate service unavailable")

    # ------------------------------------------------------------------ API
    def currencies(self) -> dict[str, str]:
        """{'EUR': 'Euro', 'INR': 'Indian Rupee', ...} — the currencies the ECB publishes."""
        cached = self._cached("currencies")
        if cached:
            return cached
        data = self._get("currencies")
        if not isinstance(data, dict) or not data:
            raise ExternalServiceError("unexpected currencies payload")
        self._store("currencies", data)
        return data

    def rate(self, base: str, quote: str, on: date | None = None) -> Rate:
        base, quote = base.upper().strip(), quote.upper().strip()
        if base == quote:
            return Rate(base, quote, 1.0, (on or date.today()).isoformat(), on.isoformat() if on else None)
        key = f"rate::{base}::{quote}::{on.isoformat() if on else 'latest'}"
        cached = self._cached(key)
        if cached:
            return cached
        path = on.isoformat() if on else "latest"
        data = self._get(path, params={"base": base, "symbols": quote})
        try:
            value = float(data["rates"][quote])
            fixing = str(data.get("date", path))
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalServiceError("unexpected rate payload from Frankfurter") from exc
        r = Rate(base, quote, value, fixing, on.isoformat() if on else None)
        self._store(key, r)
        log_event(log, "fx_rate", base=base, quote=quote, date=fixing)
        return r
