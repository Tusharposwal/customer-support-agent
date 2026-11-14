"""Offline tests for the in-memory rate limiter. A fake clock means no real waiting."""

import pytest

from support_agent.rate_limit import DAY, MINUTE, RateLimiter, limiter_from_env


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make(per_minute=20, per_day=300):
    clock = FakeClock()
    return RateLimiter(per_minute=per_minute, per_day=per_day, clock=clock), clock


def test_normal_requests_are_allowed():
    limiter, _ = make(per_minute=3, per_day=10)

    assert [limiter.check() for _ in range(3)] == [None, None, None]


def test_minute_limit_blocks_extra_requests():
    limiter, clock = make(per_minute=2, per_day=100)

    assert limiter.check() is None
    clock.advance(10)
    assert limiter.check() is None
    assert limiter.check() == ("minute", 50)  # Oldest request leaves the window in 50s.


def test_minute_window_expires():
    limiter, clock = make(per_minute=2, per_day=100)
    limiter.check()
    limiter.check()
    assert limiter.check() is not None

    clock.advance(MINUTE - 1)
    assert limiter.check() is not None  # Still inside the 60-second window.

    clock.advance(1)
    assert limiter.check() is None  # Exactly 60 seconds later: allowed again.


def test_minute_window_slides_one_request_at_a_time():
    limiter, clock = make(per_minute=2, per_day=100)
    limiter.check()  # t = 0
    clock.advance(30)
    limiter.check()  # t = 30

    clock.advance(30)  # t = 60: only the first request has expired.
    assert limiter.check() is None
    assert limiter.check() == ("minute", 30)


def test_day_limit_blocks_across_minutes():
    limiter, clock = make(per_minute=2, per_day=3)
    limiter.check()
    limiter.check()
    clock.advance(MINUTE)
    assert limiter.check() is None  # 3rd request of the day.

    clock.advance(MINUTE)
    result = limiter.check()
    assert result is not None and result[0] == "day"


def test_day_window_expires():
    limiter, clock = make(per_minute=10, per_day=2)
    limiter.check()
    limiter.check()

    clock.advance(DAY - 1)
    assert limiter.check() == ("day", 1)

    clock.advance(1)
    assert limiter.check() is None  # 24 hours after the first request.


def test_day_limit_reports_time_until_oldest_request_expires():
    limiter, clock = make(per_minute=10, per_day=2)
    limiter.check()  # t = 0
    clock.advance(3600)
    limiter.check()  # t = 1h

    clock.advance(3600)  # t = 2h
    assert limiter.check() == ("day", DAY - 7200)


def test_rejected_requests_are_not_counted():
    limiter, clock = make(per_minute=1, per_day=100)
    limiter.check()
    for _ in range(5):
        assert limiter.check() is not None  # Blocked attempts are not recorded...

    clock.advance(MINUTE)
    assert limiter.check() is None  # ...so they don't extend the wait.


def test_limits_from_environment(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "5")
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_DAY", "50")

    limiter = limiter_from_env()

    assert (limiter.per_minute, limiter.per_day) == (5, 50)


def test_default_limits(monkeypatch):
    monkeypatch.delenv("CHAT_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_DAY", "  ")

    limiter = limiter_from_env()

    assert (limiter.per_minute, limiter.per_day) == (20, 300)


@pytest.mark.parametrize("value", ["0", "-3", "abc", "2.5"])
def test_invalid_environment_values_fail_clearly(monkeypatch, value):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", value)

    with pytest.raises(ValueError, match="CHAT_RATE_LIMIT_PER_MINUTE must be a positive whole number"):
        limiter_from_env()
