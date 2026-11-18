"""Offline tests for logging configuration, request IDs and the /chat outcome log.

The agent is replaced by fakes: no Cohere client, no network, no .env.
"""

import io
import logging
import re

import pytest
from fastapi.testclient import TestClient

from support_agent import agent, api, log_config, startup
from support_agent.answer import HANDOFF_MESSAGE, SupportAnswer
from support_agent.api import app
from support_agent.rate_limit import RateLimiter

QUESTION = "My card 4111-1111 was charged twice, why?"
ANSWER_TEXT = "Refunds take 5 days. Private detail XYZZY."
SOURCE_URL = "https://support.example.com/private-refund-page"
SECRET = "sk-test-SECRET-abcdef123456"
PROVIDER_MESSAGE = "upstream said: invalid api key " + SECRET

GROUNDED = SupportAnswer(
    answer=ANSWER_TEXT,
    sources=[{"id": "KB-007", "title": "Refunds", "url": SOURCE_URL}],
    escalate=False,
    reason="answered",
)

SETTINGS = (
    "COHERE_RERANK_ENABLED", "COHERE_RERANK_MODEL", "COHERE_RERANK_MIN_SCORE",
    "COHERE_RERANK_TIMEOUT_SECONDS", "COHERE_CHAT_TIMEOUT_SECONDS", "COHERE_EMBED_TIMEOUT_SECONDS",
    "CHAT_RATE_LIMIT_PER_MINUTE", "CHAT_RATE_LIMIT_PER_DAY", "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    """Requests in tests that don't run startup get their own limiter."""
    monkeypatch.setattr(api, "chat_limiter", RateLimiter(per_minute=100, per_day=1000))


@pytest.fixture
def env(monkeypatch, semantic_kb):
    """A valid configuration for tests that run the real startup. The Cohere
    SDK is the fake from conftest.py, so startup creates fake clients."""
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COHERE_API_KEY", SECRET)
    return monkeypatch


class FakeAgent:
    def __init__(self, result=GROUNDED, error=None, log_line=None):
        self.result, self.error, self.log_line = result, error, log_line
        self.questions = []

    def __call__(self, question):
        self.questions.append(question)
        if self.log_line:
            logging.getLogger("support_agent.agent").warning(self.log_line)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def use_agent(monkeypatch):
    def install(**kwargs):
        fake = FakeAgent(**kwargs)
        monkeypatch.setattr(agent, "answer_question", fake)
        return fake

    return install


def outcomes(caplog):
    return [record for record in caplog.records if hasattr(record, "chat_outcome")]


def only_outcome(caplog):
    (record,) = outcomes(caplog)
    return record


def post(client=None, question=QUESTION, **kwargs):
    return (client or TestClient(app)).post("/chat", json={"question": question}, **kwargs)


# --- LOG_LEVEL ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "level"),
    [(None, logging.INFO), ("", logging.INFO), ("  ", logging.INFO), ("debug", logging.DEBUG),
     ("INFO", logging.INFO), (" Warning ", logging.WARNING), ("ERROR", logging.ERROR),
     ("critical", logging.CRITICAL)],
)
def test_log_level_values(raw, level):
    environ = {} if raw is None else {"LOG_LEVEL": raw}
    assert log_config.log_level_from_env(environ) == level


@pytest.mark.parametrize("raw", ["verbose", "10", "WARN ING", "trace", SECRET])
def test_invalid_log_level_is_rejected_without_its_value(raw):
    with pytest.raises(ValueError) as raised:
        log_config.log_level_from_env({"LOG_LEVEL": raw})
    assert str(raised.value) == "LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"


def test_invalid_log_level_fails_startup(env):
    env.setenv("LOG_LEVEL", SECRET)

    with pytest.raises(startup.StartupError) as raised:
        with TestClient(app):
            pass
    assert "LOG_LEVEL must be one of" in str(raised.value)
    assert SECRET not in str(raised.value)


# --- configure_logging ----------------------------------------------------------


def test_configure_logging_formats_package_lines(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", stream)
    log_config.configure_logging(logging.INFO)

    token = log_config.request_id_var.set("req-42")
    try:
        logging.getLogger("support_agent.agent").warning("Retrieval failed: %s", "TimeoutError")
    finally:
        log_config.request_id_var.reset(token)
    logging.getLogger("support_agent.api").info("outside a request")

    first, second = stream.getvalue().splitlines()
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} WARNING support_agent\.agent \[req-42\] Retrieval failed: TimeoutError",
        first,
    )
    assert second.endswith("INFO support_agent.api [-] outside a request")


def test_configure_logging_applies_the_level(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", stream)
    log_config.configure_logging(logging.WARNING)

    logging.getLogger("support_agent.api").info("hidden")
    logging.getLogger("support_agent.api").warning("shown")

    assert stream.getvalue().count("\n") == 1 and "shown" in stream.getvalue()
    assert logging.getLogger("support_agent").level == logging.WARNING


def test_configure_logging_is_idempotent_and_leaves_other_loggers_alone():
    root = logging.getLogger()
    uvicorn_loggers = [logging.getLogger(name) for name in ("uvicorn", "uvicorn.error", "uvicorn.access")]
    before = (list(root.handlers), root.level, [(list(l.handlers), l.level, l.propagate) for l in uvicorn_loggers])

    log_config.configure_logging(logging.INFO)
    log_config.configure_logging(logging.DEBUG)

    package = logging.getLogger("support_agent")
    ours = [h for h in package.handlers if getattr(h, "_support_agent_handler", False)]
    assert len(ours) == 1
    assert package.propagate is False  # never printed twice through the root logger
    assert before == (list(root.handlers), root.level, [(list(l.handlers), l.level, l.propagate) for l in uvicorn_loggers])


def test_startup_configures_logging_from_log_level(env, use_agent, capsys):
    env.setenv("LOG_LEVEL", "warning")
    use_agent()

    with TestClient(app) as client:
        post(client)

    assert logging.getLogger("support_agent").level == logging.WARNING
    err = capsys.readouterr().err
    assert "chat_request" not in err  # a 200 outcome is INFO
    assert "Startup checks passed" not in err


def test_startup_logs_ready_and_outcomes_at_default_level(env, use_agent, capsys):
    use_agent()

    with TestClient(app) as client:
        response = post(client)

    lines = capsys.readouterr().err.splitlines()
    assert any(line.endswith("INFO support_agent.api [-] Startup checks passed; ready for requests") for line in lines)
    (outcome_line,) = [line for line in lines if "chat_request" in line]
    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(
        rf"\S+ \S+ INFO support_agent\.api \[{request_id}\] chat_request status=200 escalate=false "
        r"reason=answered sources=1 duration_ms=\d+(\.\d)?",
        outcome_line,
    )


def test_agent_log_lines_carry_the_request_id(env, use_agent, capsys):
    """Lines logged inside the worker thread running /chat get the same ID."""
    use_agent(log_line="Retrieval failed: retrieval_error")

    with TestClient(app) as client:
        response = post(client, headers={"X-Request-ID": "trace-123"})

    assert response.headers["X-Request-ID"] == "trace-123"
    err = capsys.readouterr().err
    assert "WARNING support_agent.agent [trace-123] Retrieval failed: retrieval_error" in err
    assert "INFO support_agent.api [trace-123] chat_request status=200" in err


# --- Request IDs --------------------------------------------------------------


def test_request_id_is_generated_and_returned(use_agent, caplog):
    use_agent()
    caplog.set_level(logging.INFO, logger="support_agent")

    first, second = post(), post()

    ids = [first.headers["X-Request-ID"], second.headers["X-Request-ID"]]
    assert all(re.fullmatch(r"[0-9a-f]{32}", value) for value in ids)
    assert ids[0] != ids[1]
    assert [record.chat_outcome["request_id"] for record in outcomes(caplog)] == ids


@pytest.mark.parametrize("supplied", ["abc", "req-2026.11.17:0001", "A" * 64, "0f8fad5b-d9cb-469f-a165-70867728950e"])
def test_valid_client_request_id_is_used(use_agent, caplog, supplied):
    use_agent()
    caplog.set_level(logging.INFO, logger="support_agent")

    response = post(headers={"X-Request-ID": supplied})

    assert response.headers["X-Request-ID"] == supplied
    assert only_outcome(caplog).chat_outcome["request_id"] == supplied


@pytest.mark.parametrize(
    "supplied",
    ["", "A" * 65, "has space", "semi;colon", "<script>", "quote\"d", "tab\there", "caf\u00e9", QUESTION],
)
def test_unsafe_client_request_id_is_replaced(use_agent, caplog, supplied):
    use_agent()
    caplog.set_level(logging.INFO, logger="support_agent")

    response = post(headers={"X-Request-ID": supplied.encode("latin-1")})

    generated = response.headers["X-Request-ID"]
    assert generated != supplied and re.fullmatch(r"[0-9a-f]{32}", generated)
    assert only_outcome(caplog).chat_outcome["request_id"] == generated
    assert supplied not in caplog.text or supplied == ""


def test_other_endpoints_get_a_request_id_but_no_outcome_log(caplog):
    caplog.set_level(logging.DEBUG, logger="support_agent")
    client = TestClient(app)

    for path in ("/", "/health", "/ready", "/static/app.js"):
        response = client.get(path)
        assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])
    assert outcomes(caplog) == []


def test_request_id_header_is_not_duplicated(use_agent):
    use_agent()
    response = post(headers={"X-Request-ID": "one-id"})
    assert response.headers.get_list("X-Request-ID") == ["one-id"]


# --- One outcome per /chat request -------------------------------------------


def test_successful_answer_outcome(use_agent, caplog):
    use_agent()
    caplog.set_level(logging.INFO, logger="support_agent")

    response = post()

    assert response.status_code == 200
    record = only_outcome(caplog)
    assert record.levelno == logging.INFO and record.name == "support_agent.api"
    fields = record.chat_outcome
    assert (fields["status"], fields["escalate"], fields["reason"], fields["sources"]) == (200, "false", "answered", 1)
    assert isinstance(fields["duration_ms"], float)


@pytest.mark.parametrize(
    "reason",
    ["insufficient_evidence", "no_articles", "no_citations", "below_rerank_threshold", "llm_error",
     "retrieval_error", "configuration_error", "invalid_citations", "agent_error"],
)
def test_handoff_outcomes(use_agent, caplog, reason):
    use_agent(result=SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason=reason))
    caplog.set_level(logging.INFO, logger="support_agent")

    response = post()

    assert response.status_code == 200 and response.json()["reason"] == reason
    record = only_outcome(caplog)
    assert record.levelno == logging.INFO
    assert record.getMessage() == (
        f"chat_request status=200 escalate=true reason={reason} sources=0 "
        f"duration_ms={record.chat_outcome['duration_ms']}"
    )


def test_duration_is_measured_in_milliseconds(use_agent, caplog, monkeypatch):
    use_agent()
    caplog.set_level(logging.INFO, logger="support_agent")
    times = iter([100.0, 101.2345])
    monkeypatch.setattr(api, "clock", lambda: next(times))

    post()

    assert only_outcome(caplog).chat_outcome["duration_ms"] == 1234.5


def test_invalid_request_is_logged_without_calling_agent(use_agent, caplog):
    fake = use_agent()
    caplog.set_level(logging.INFO, logger="support_agent")

    response = post(question="   ")

    assert response.status_code == 422 and fake.questions == []
    record = only_outcome(caplog)
    assert record.levelno == logging.INFO
    assert record.chat_outcome | {"duration_ms": 0} == {
        "request_id": response.headers["X-Request-ID"], "status": 422, "escalate": "-",
        "reason": "invalid_request", "sources": 0, "duration_ms": 0,
    }


def test_wrong_method_is_logged(caplog):
    caplog.set_level(logging.INFO, logger="support_agent")
    response = TestClient(app).get("/chat")
    assert response.status_code == 405
    assert only_outcome(caplog).chat_outcome["reason"] == "method_not_allowed"


@pytest.mark.parametrize(("limits", "limit_name"), [((1, 10), "minute"), ((10, 1), "day")])
def test_rate_limited_outcome(use_agent, caplog, monkeypatch, limits, limit_name):
    fake = use_agent()
    monkeypatch.setattr(api, "chat_limiter", RateLimiter(per_minute=limits[0], per_day=limits[1]))
    caplog.set_level(logging.INFO, logger="support_agent")

    first, second = post(), post()

    assert (first.status_code, second.status_code) == (200, 429)
    assert len(fake.questions) == 1
    allowed, limited = outcomes(caplog)
    assert allowed.chat_outcome["status"] == 200
    assert limited.levelno == logging.WARNING
    assert limited.getMessage().startswith(
        f"chat_request status=429 escalate=- reason=rate_limited sources=0 duration_ms="
    )
    assert limited.getMessage().endswith(f" limit={limit_name}")
    # One record for the 429, not a separate "rate limit reached" warning as well.
    assert len(caplog.records) == 2


def test_not_ready_outcome(use_agent, caplog, monkeypatch):
    fake = use_agent()
    monkeypatch.setattr(api, "chat_limiter", None)
    caplog.set_level(logging.INFO, logger="support_agent")

    assert post().status_code == 503
    record = only_outcome(caplog)
    assert record.levelno == logging.WARNING and record.chat_outcome["reason"] == "not_ready"
    assert fake.questions == []


def test_agent_exception_outcome(use_agent, caplog):
    use_agent(error=RuntimeError(PROVIDER_MESSAGE))
    caplog.set_level(logging.INFO, logger="support_agent")

    response = post()

    assert response.status_code == 500 and response.json()["reason"] == "server_error"
    record = only_outcome(caplog)
    assert record.levelno == logging.ERROR
    assert record.getMessage().endswith(" error=RuntimeError")
    assert record.chat_outcome["reason"] == "server_error" and record.chat_outcome["escalate"] == "true"
    assert len(caplog.records) == 1  # no second "unexpected error" line


def test_exception_outside_the_agent_still_gets_a_safe_response_and_one_log(caplog, monkeypatch):
    """Even a failure the endpoint doesn't catch (here: the limiter) gets the
    safe hand-off body, an X-Request-ID and exactly one outcome line."""

    class BrokenLimiter:
        def check(self):
            raise KeyError(PROVIDER_MESSAGE)

    monkeypatch.setattr(api, "chat_limiter", BrokenLimiter())
    caplog.set_level(logging.INFO, logger="support_agent")

    response = post(headers={"X-Request-ID": "broken-1"})

    assert response.status_code == 500
    assert response.json() == {"answer": HANDOFF_MESSAGE, "sources": [], "escalate": True, "reason": "server_error"}
    assert response.headers["X-Request-ID"] == "broken-1"
    record = only_outcome(caplog)
    assert record.levelno == logging.ERROR
    assert record.chat_outcome["request_id"] == "broken-1"
    assert record.getMessage().endswith(" error=KeyError")


def test_unusual_reason_is_not_logged_verbatim(use_agent, caplog):
    use_agent(result=SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason="Bad reason\nINJECTED"))
    caplog.set_level(logging.INFO, logger="support_agent")

    post()

    assert only_outcome(caplog).chat_outcome["reason"] == "other"
    assert "INJECTED" not in caplog.text


# --- Nothing sensitive in the logs --------------------------------------------


@pytest.mark.parametrize(
    "agent_kwargs",
    [
        {},
        {"error": RuntimeError(PROVIDER_MESSAGE)},
        {"error": ValueError(QUESTION)},
        {"log_line": "Cohere chat attempt 1 failed: auth_error (ApiError, status 401)"},
    ],
)
def test_logs_never_contain_sensitive_text(env, use_agent, caplog, capsys, agent_kwargs):
    """Checked both through caplog and in the formatted output of a real startup."""
    use_agent(**agent_kwargs)
    caplog.set_level(logging.DEBUG)

    post()  # Without startup: records reach caplog through the root logger.
    with TestClient(app) as client:  # With startup: formatted lines on stderr.
        post(client)

    logged = caplog.text + capsys.readouterr().err
    assert "chat_request" in logged
    for text in (QUESTION, "4111", ANSWER_TEXT, "XYZZY", SOURCE_URL, SECRET, PROVIDER_MESSAGE, "upstream said"):
        assert text not in logged
