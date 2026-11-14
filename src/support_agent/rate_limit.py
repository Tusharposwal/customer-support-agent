"""A tiny in-memory, global rate limiter for the demo deployment.

It remembers when each accepted request happened and allows a new one only if
fewer than `per_minute` requests happened in the last 60 seconds AND fewer
than `per_day` happened in the last 24 hours ("sliding windows").

Limitations (fine for a demo, not for production):
- Global: all visitors share the same limits.
- In memory: counts reset when the server restarts, and are not shared
  between multiple server processes or instances.
"""

import os
import threading
import time
from collections import deque

MINUTE = 60
DAY = 24 * 60 * 60

DEFAULT_PER_MINUTE = 20
DEFAULT_PER_DAY = 300


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, clock=time.monotonic):
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock  # Replaceable in tests, so no real waiting is needed.
        self._times = deque()  # Times of accepted requests in the last 24 hours, oldest first.
        self._lock = threading.Lock()  # FastAPI runs requests in several threads.

    def check(self) -> tuple[str, int] | None:
        """Record a request and return None if it is allowed.

        If it is not allowed, nothing is recorded and the result is
        (limit_name, seconds_to_wait), where limit_name is "minute" or "day".
        """
        with self._lock:
            now = self._clock()

            # Forget requests older than 24 hours.
            while self._times and now - self._times[0] >= DAY:
                self._times.popleft()

            if len(self._times) >= self.per_day:
                return "day", _seconds_until(self._times[0] + DAY, now)

            last_minute = [t for t in self._times if now - t < MINUTE]
            if len(last_minute) >= self.per_minute:
                return "minute", _seconds_until(last_minute[0] + MINUTE, now)

            self._times.append(now)
            return None


def _seconds_until(moment: float, now: float) -> int:
    return max(1, int(moment - now + 0.999))  # Round up, at least 1 second.


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{name} must be a positive whole number, got {raw!r}")
    return value


def limiter_from_env() -> RateLimiter:
    """Build the limiter from CHAT_RATE_LIMIT_PER_MINUTE / CHAT_RATE_LIMIT_PER_DAY."""
    return RateLimiter(
        per_minute=_positive_int_env("CHAT_RATE_LIMIT_PER_MINUTE", DEFAULT_PER_MINUTE),
        per_day=_positive_int_env("CHAT_RATE_LIMIT_PER_DAY", DEFAULT_PER_DAY),
    )
