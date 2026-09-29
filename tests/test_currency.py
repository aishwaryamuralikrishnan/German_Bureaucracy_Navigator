"""Currency tool + Frankfurter client with mocked HTTP — no network."""

import json
from unittest.mock import MagicMock

import pytest
import requests

from navigator.clients.frankfurter import FrankfurterClient
from navigator.tools import currency as cur
from navigator.utils.errors import ExternalServiceError, ValidationError

CURRENCIES = {"EUR": "Euro", "INR": "Indian Rupee", "USD": "US Dollar", "GBP": "British Pound"}


def _resp(status=200, json_data=None):
    r = MagicMock(); r.status_code = status; r.text = json.dumps(json_data) if json_data is not None else ""
    r.json.return_value = json_data
    return r


def _client(responses):
    session = MagicMock(spec=requests.Session); session.headers = {}
    session.get.side_effect = responses
    return FrankfurterClient(session=session)


def test_rate_and_cache():
    c = _client([_resp(200, {"amount": 1, "base": "INR", "date": "2026-09-12", "rates": {"EUR": 0.0102}})])
    r1 = c.rate("inr", "eur"); r2 = c.rate("INR", "EUR")
    assert r1.rate == 0.0102 and r1.rate_date == "2026-09-12" and r1 == r2
    assert c.session.get.call_count == 1


def test_same_currency_short_circuits():
    c = _client([])
    assert c.rate("EUR", "EUR").rate == 1.0


def test_mirror_failover_and_errors():
    good = _resp(200, {"base": "USD", "date": "2026-09-12", "rates": {"EUR": 0.9}})
    c = _client([requests.Timeout(), good])
    assert c.rate("USD", "EUR").rate == 0.9
    with pytest.raises(ValidationError):
        _client([_resp(404, {"message": "not found"})]).rate("XXX", "EUR")
    with pytest.raises(ExternalServiceError):
        _client([_resp(503), _resp(503)]).rate("USD", "EUR")


def test_normalise_code():
    assert cur.normalise_code("rupees") == "INR" and cur.normalise_code("eur") == "EUR" and cur.normalise_code("£") == "GBP"
    with pytest.raises(ValidationError):
        cur.normalise_code("lakh")


def test_tool_converts_annual_salary(monkeypatch):
    fake = _client([_resp(200, CURRENCIES), _resp(200, {"base": "INR", "date": "2026-09-12", "rates": {"EUR": 0.01}})])
    monkeypatch.setattr(cur, "_client", lambda: fake)
    out = json.loads(cur.convert_currency.invoke({"amount": 1800000, "from_currency": "INR", "to_currency": "EUR", "period": "annual"}))
    assert out["ok"] and out["data"]["to"]["amount"] == 18000.0
    assert out["data"]["equivalents"]["EUR_per_month"] == 1500.0
    assert any("spread" in w for w in out["warnings"])
    # A salary of unknown type must not be silently treated as net or gross.
    assert out["data"]["salary_basis"] == "unknown"
    assert any("did not say whether this salary is gross or net" in w for w in out["warnings"])
    assert any("cost of living" in w for w in out["warnings"])


def test_tool_salary_basis_warnings(monkeypatch):
    def fresh():
        return _client([_resp(200, CURRENCIES), _resp(200, {"base": "INR", "date": "2026-09-12", "rates": {"EUR": 0.01}})])
    monkeypatch.setattr(cur, "_client", fresh)
    net = json.loads(cur.convert_currency.invoke({"amount": 100000, "from_currency": "INR", "period": "monthly", "gross_or_net": "net"}))
    assert net["data"]["salary_basis"] == "net" and any("never with a gross offer" in w for w in net["warnings"])
    monkeypatch.setattr(cur, "_client", fresh)
    gross = json.loads(cur.convert_currency.invoke({"amount": 100000, "from_currency": "INR", "period": "monthly", "gross_or_net": "gross"}))
    assert gross["data"]["salary_basis"] == "gross" and any("German rules only" in w for w in gross["warnings"])
    monkeypatch.setattr(cur, "_client", fresh)
    one_off = json.loads(cur.convert_currency.invoke({"amount": 500, "from_currency": "USD"}))
    assert "salary_basis" not in one_off["data"] and not any("gross or net" in w for w in one_off["warnings"])


def test_tool_rejects_unsupported_currency_and_bad_amount(monkeypatch):
    fake = _client([_resp(200, CURRENCIES)])
    monkeypatch.setattr(cur, "_client", lambda: fake)
    out = json.loads(cur.convert_currency.invoke({"amount": 100, "from_currency": "NGN", "to_currency": "EUR"}))
    assert not out["ok"] and "NGN" in out["error"]
    out2 = json.loads(cur.convert_currency.invoke({"amount": -5, "from_currency": "USD"}))
    assert not out2["ok"]
