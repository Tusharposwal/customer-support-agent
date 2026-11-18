"""Timeouts and retries for Cohere chat and embed requests.

The Cohere SDK on its own waits up to 300 seconds per request and retries up to
2 times without telling us, waiting as long as 60 seconds before a retry after
a rate limit. A customer would be stuck waiting for minutes. So every chat and
embed request switches the SDK's retries off and this module decides instead:

* Each operation has a time budget: COHERE_CHAT_TIMEOUT_SECONDS (default 25)
  and COHERE_EMBED_TIMEOUT_SECONDS (default 10). The whole call, including a
  retry and the wait before it, stays within that budget.
* At most one retry, and only for errors that mean "try again shortly": rate
  limit (429), server errors (500, 502, 503, 504) and a connection that could
  not be opened (the request was never sent).
* No retry for a timeout (the budget is already used up), a connection lost
  after it was opened (Cohere may already have processed the request), an
  authentication error, an invalid request, a malformed response or an
  unrecognised error.
* No retry unless at least half the budget is still left after waiting, and no
  retry if the server asks us to wait longer than 2 seconds.

Errors are reported as CohereCallError, whose message contains only the
operation and the error category. Provider error text is never logged or
re-raised, because it can echo request details. Importing this module does not
import the Cohere SDK.
"""

import logging
import math
import os
import time
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

CHAT_TIMEOUT_ENV = "COHERE_CHAT_TIMEOUT_SECONDS"
EMBED_TIMEOUT_ENV = "COHERE_EMBED_TIMEOUT_SECONDS"
DEFAULT_CHAT_TIMEOUT_SECONDS = 25.0
DEFAULT_EMBED_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 120.0

MAX_RETRIES = 1
RETRY_WAIT_SECONDS = 1.0  # when the server doesn't say how long to wait
MAX_RETRY_AFTER_SECONDS = 2.0  # a longer Retry-After means "hand off", not "wait"

RETRYABLE = frozenset({"rate_limit", "server_error", "connect_error"})
_AUTH_STATUSES = {401, 403, 498}

# Replaced in tests so retries don't really wait.
sleep = time.sleep
clock = time.monotonic


class CohereCallError(RuntimeError):
    """A Cohere request failed. The message never contains provider text."""

    def __init__(self, operation: str, category: str, attempts: int = 1):
        super().__init__(f"Cohere {operation} request failed: {category} (attempts: {attempts})")
        self.operation = operation
        self.category = category
        self.attempts = attempts


def close_client(client) -> None:
    """Close a Cohere client's HTTP connection pool.

    Uses the SDK's own context-manager exit (cohere.ClientV2 supports `with`),
    which closes the httpx client it created. Objects without it (test fakes)
    are left alone. Any error is raised to the caller.
    """
    exit_method = getattr(client, "__exit__", None)
    if exit_method is not None:
        exit_method(None, None, None)


def describe_error(error: BaseException) -> str:
    """A log-safe name for an error: our category, or just the exception type."""
    return f"{error.category} ({error.operation})" if isinstance(error, CohereCallError) else type(error).__name__


# --- Configuration -----------------------------------------------------------


def timeout_from_env(name: str, default: float, environ: Mapping[str, str] | None = None) -> float:
    """Read a timeout in seconds. Blank or unset means `default`; anything that
    is not a number above 0 and at most MAX_TIMEOUT_SECONDS raises ValueError.
    The message names the setting but never repeats its value."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number of seconds") from None
    if not math.isfinite(value) or not 0 < value <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"{name} must be above 0 and at most {MAX_TIMEOUT_SECONDS:g}")
    return value


def chat_timeout(environ: Mapping[str, str] | None = None) -> float:
    return timeout_from_env(CHAT_TIMEOUT_ENV, DEFAULT_CHAT_TIMEOUT_SECONDS, environ)


def embed_timeout(environ: Mapping[str, str] | None = None) -> float:
    return timeout_from_env(EMBED_TIMEOUT_ENV, DEFAULT_EMBED_TIMEOUT_SECONDS, environ)


# --- Error classification ----------------------------------------------------


def classify_error(error: BaseException) -> str:
    """Name the kind of failure. Cohere API errors carry `status_code`; network
    problems are httpx (or built-in) exceptions without one."""
    if isinstance(error, CohereCallError):
        return error.category

    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if status == 429:
            return "rate_limit"
        if status in _AUTH_STATUSES:
            return "auth_error"
        if status == 408:
            return "timeout"
        if status >= 500 and status != 501:
            return "server_error"
        if 200 <= status < 300:
            return "malformed_response"  # the SDK could not read a successful response
        if 400 <= status < 500:
            return "invalid_request"
        return "unknown"

    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a Cohere SDK dependency
        httpx = None
    if httpx is not None:
        # httpcore raises these only while opening the TCP connection or doing
        # the TLS handshake, before any HTTP bytes are written: never sent.
        if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
            return "connect_error"
        if isinstance(error, httpx.TimeoutException):
            return "timeout"
        # Read/Write/Close errors and a remote disconnect happen on an open
        # connection. The same ReadError appears whether or not Cohere had
        # received (and processed) the request, so it must not be repeated.
        # (httpcore even turns a failed request write into a read attempt.)
        if isinstance(error, (httpx.NetworkError, httpx.RemoteProtocolError)):
            return "connection_lost"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, ConnectionRefusedError):
        return "connect_error"
    if isinstance(error, ConnectionError):  # reset, aborted, broken pipe
        return "connection_lost"
    return "unknown"


def _retry_after_seconds(error: BaseException) -> float | None:
    """The server's Retry-After in seconds, if it sent a plain number."""
    headers = getattr(error, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    for name, value in headers.items():
        if isinstance(name, str) and name.lower() == "retry-after":
            try:
                seconds = float(str(value).strip())
            except ValueError:
                return None
            return seconds if math.isfinite(seconds) and seconds >= 0 else None
    return None


# --- The call ----------------------------------------------------------------


def call_with_retries(request: Callable[[dict], object], *, operation: str, timeout_seconds: float):
    """Run `request(request_options)` within `timeout_seconds`, retrying at most
    once for a temporary failure. Returns the response or raises CohereCallError.

    `request` receives the options to pass to the SDK, for example
    `lambda options: client.chat(..., request_options=options)`.
    """
    start = clock()
    remaining = timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            return request({"timeout": remaining, "max_retries": 0})
        except Exception as error:
            category = classify_error(error)
            elapsed = clock() - start
            wait = None
            if category in RETRYABLE and attempt <= MAX_RETRIES:
                retry_after = _retry_after_seconds(error)
                wait = RETRY_WAIT_SECONDS if retry_after is None else retry_after
                if wait > MAX_RETRY_AFTER_SECONDS or timeout_seconds - elapsed - wait < timeout_seconds / 2:
                    wait = None  # waiting would take too long: hand off instead

            log = logger.error if category in ("auth_error", "invalid_request") else logger.warning
            log(
                "Cohere %s attempt %d failed: %s (%s, status %s)%s",
                operation, attempt, category, type(error).__name__,
                getattr(error, "status_code", None), "; retrying" if wait is not None else "",
            )
            if wait is None:
                # `from None`: the provider's error (headers, body) must not travel on.
                raise CohereCallError(operation, category, attempt) from None

        sleep(wait)
        remaining = timeout_seconds - (clock() - start)
