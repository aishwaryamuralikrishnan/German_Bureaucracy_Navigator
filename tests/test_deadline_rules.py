from datetime import date

import pytest

from navigator.core.deadline_rules import calculate_deadline, state_for_city
from navigator.utils.errors import ValidationError


def test_anmeldung_is_14_days():
    r = calculate_deadline("moved_in", date(2026, 3, 2), "Berlin", today=date(2026, 3, 2))
    assert r.raw_deadline == date(2026, 3, 16)
    assert r.deadline == date(2026, 3, 16)  # Monday, no holiday
    assert r.hard_deadline and r.urgency == "ok"


def test_weekend_and_holiday_roll_forward():
    # 19 Sep 2026 + 14 = 3 Oct 2026 (Saturday AND German Unity Day) -> Monday 5 Oct
    r = calculate_deadline("moved_in", date(2026, 9, 19), "Munich", today=date(2026, 9, 19))
    assert r.deadline == date(2026, 10, 5) and r.adjusted


def test_recommended_dates_are_not_rolled():
    r = calculate_deadline("residence_permit_expiry", date(2026, 12, 31), None, today=date(2026, 9, 1))
    assert r.deadline == date(2026, 12, 31) - __import__("datetime").timedelta(days=56)
    assert not r.hard_deadline


def test_urgency_levels():
    assert calculate_deadline("moved_in", date(2026, 1, 1), None, today=date(2026, 2, 1)).urgency == "overdue"
    assert calculate_deadline("moved_in", date(2026, 1, 1), None, today=date(2026, 1, 12)).urgency == "soon"


def test_state_mapping_and_unknown_event():
    assert state_for_city("berlin") == "BE" and state_for_city("Köln") == "NW" and state_for_city("Atlantis") is None
    with pytest.raises(ValidationError):
        calculate_deadline("wedding", date(2026, 1, 1))
