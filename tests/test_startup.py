"""Offline tests for startup validation, /health and /ready.

Tests that start the app use the fake `cohere` module from conftest.py (the
`semantic_kb` fixture), so the clients created at startup are fakes, and
sockets refuse to connect. .env is never read (see conftest.py).
"""

import json
import os
import socket
import sys

import dotenv
import pytest
from fastapi.testclient import TestClient

from support_agent import agent, api, kb, startup
from support_agent.api import app
from support_agent.rate_limit import DEFAULT_PER_DAY, DEFAULT_PER_MINUTE

FAKE_KEY = "sk-test-SECRET-1234567890"

SETTINGS = (
    "COHERE_API_KEY",
    "COHERE_RERANK_ENABLED",
    "COHERE_RERANK_MODEL",
    "COHERE_RERANK_MIN_SCORE",
    "COHERE_RERANK_TIMEOUT_SECONDS",
    "COHERE_CHAT_TIMEOUT_SECONDS",
    "COHERE_EMBED_TIMEOUT_SECONDS",
    "CHAT_RATE_LIMIT_PER_MINUTE",
    "CHAT_RATE_LIMIT_PER_DAY",
    "LOG_LEVEL",
)

ARTICLE = {"id": "KB-001", "title": "Reset your password", "text": "Select Forgot password.", "url": "https://x/reset"}


def write_kb(tmp_path, content):
    path = tmp_path / "kb.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


@pytest.fixture
def good_kb(tmp_path):
    return write_kb(tmp_path, [ARTICLE])


@pytest.fixture
def env(monkeypatch, good_kb, semantic_kb):
    """A clean, valid configuration in os.environ: only a (fake) API key set,
    everything else at its default, and a valid knowledge base. The Cohere
    SDK is the fake from conftest.py, so startup creates fake clients."""
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COHERE_API_KEY", FAKE_KEY)
    monkeypatch.setattr(kb, "KB_PATH", good_kb)
    return monkeypatch


@pytest.fixture(autouse=True)
def no_leftover_limiter(monkeypatch):
    """Each test starts with no limiter (as before startup) and any limiter a
    startup creates is discarded afterwards."""
    monkeypatch.setattr(api, "chat_limiter", None)


@pytest.fixture
def dotenv_file(env, tmp_path, monkeypatch):
    """Make startup load a temporary .env file with the real python-dotenv
    parser (never the project's .env). Call it with the file's text.

    load_dotenv writes straight into os.environ, so every setting it may have
    added is removed afterwards (the env fixture then restores the originals)."""

    def use(text):
        path = tmp_path / "test.env"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setattr(startup, "load_dotenv", lambda: dotenv.load_dotenv(path))
        return path

    yield use
    for name in SETTINGS:
        os.environ.pop(name, None)


@pytest.fixture
def offline(monkeypatch):
    """Fail loudly if startup or a probe tries to reach Cohere or the network."""

    def no_connect(*args, **kwargs):
        raise AssertionError("A network connection was attempted")

    # Every outgoing HTTP client (httpx included) resolves a host name first.
    monkeypatch.setattr(socket, "getaddrinfo", no_connect)
    monkeypatch.setattr(socket, "create_connection", no_connect)

    calls = []
    monkeypatch.setattr(agent, "answer_question", lambda *a, **k: calls.append(a) or pytest.fail("agent called"))
    return calls


# --- find_problems: one check at a time ---------------------------------------


def valid_environ(**overrides):
    environ = {"COHERE_API_KEY": FAKE_KEY}
    environ.update(overrides)
    return environ


def test_valid_configuration_has_no_problems(good_kb):
    assert startup.find_problems(valid_environ(), good_kb) == []


def test_valid_non_default_configuration(good_kb):
    environ = valid_environ(
        COHERE_RERANK_ENABLED="true",
        COHERE_RERANK_MODEL="rerank-v5.0-pro",
        COHERE_RERANK_MIN_SCORE="0.2",
        COHERE_RERANK_TIMEOUT_SECONDS="3",
        COHERE_CHAT_TIMEOUT_SECONDS="30",
        COHERE_EMBED_TIMEOUT_SECONDS="8",
        CHAT_RATE_LIMIT_PER_MINUTE="5",
        CHAT_RATE_LIMIT_PER_DAY="100",
    )
    assert startup.find_problems(environ, good_kb) == []


def test_blank_optional_settings_use_defaults(good_kb):
    environ = valid_environ(**{name: "  " for name in SETTINGS if name != "COHERE_API_KEY"})
    assert startup.find_problems(environ, good_kb) == []


@pytest.mark.parametrize("environ", [{}, {"COHERE_API_KEY": ""}, {"COHERE_API_KEY": "   "}])
def test_missing_api_key(good_kb, environ):
    assert startup.find_problems(environ, good_kb) == ["COHERE_API_KEY is not set"]


@pytest.mark.parametrize(
    ("name", "value", "rule"),
    [
        ("COHERE_RERANK_ENABLED", "ture", "must be true or false"),
        ("COHERE_RERANK_MIN_SCORE", "high", "must be a number"),
        ("COHERE_RERANK_MIN_SCORE", "1.5", "must be between 0 and 1"),
        ("COHERE_RERANK_MIN_SCORE", "nan", "must be a finite number"),
        ("COHERE_RERANK_TIMEOUT_SECONDS", "0", "must be greater than 0"),
        ("COHERE_RERANK_TIMEOUT_SECONDS", "soon", "must be a number"),
    ],
)
def test_invalid_rerank_settings(good_kb, name, value, rule):
    assert startup.find_problems(valid_environ(**{name: value}), good_kb) == [f"{name} {rule}"]


@pytest.mark.parametrize("name", ["COHERE_CHAT_TIMEOUT_SECONDS", "COHERE_EMBED_TIMEOUT_SECONDS"])
@pytest.mark.parametrize("value", ["fast", "0", "-1", "121", "inf"])
def test_invalid_timeouts(good_kb, name, value):
    (problem,) = startup.find_problems(valid_environ(**{name: value}), good_kb)
    assert problem.startswith(f"{name} must be")


@pytest.mark.parametrize("name", ["CHAT_RATE_LIMIT_PER_MINUTE", "CHAT_RATE_LIMIT_PER_DAY"])
@pytest.mark.parametrize("value", ["0", "-3", "abc", "2.5"])
def test_invalid_rate_limits(good_kb, name, value):
    assert startup.find_problems(valid_environ(**{name: value}), good_kb) == [
        f"{name} must be a positive whole number"
    ]


def test_missing_knowledge_base(tmp_path):
    missing = tmp_path / "nowhere" / "kb.json"
    assert startup.find_problems(valid_environ(), missing) == [
        f"Knowledge base {missing} could not be read (FileNotFoundError)"
    ]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("{not json", "is not valid JSON"),
        ({"id": "KB-001"}, "must contain a JSON list of articles"),
        ([{"id": "KB-001", "title": "T", "text": "x"}], "missing required field(s): url"),
        ([{**ARTICLE, "text": "  "}], "field 'text' must not be empty"),
        ([ARTICLE, ARTICLE], "duplicate id 'KB-001'"),
        ([], "contains no articles"),
    ],
)
def test_malformed_knowledge_base(tmp_path, content, expected):
    path = write_kb(tmp_path, content)
    (problem,) = startup.find_problems(valid_environ(), path)
    assert problem.startswith(f"Knowledge base {path}") and expected in problem


def test_real_knowledge_base_is_valid():
    """The shipped data/kb.json passes the startup check."""
    assert startup.find_problems(valid_environ()) == []


def test_every_problem_is_reported_at_once(tmp_path):
    environ = {"COHERE_RERANK_ENABLED": "ture", "CHAT_RATE_LIMIT_PER_DAY": "0"}
    problems = startup.find_problems(environ, tmp_path / "missing.json")

    assert len(problems) == 4
    assert problems[0] == "COHERE_API_KEY is not set"
    assert problems[1] == "COHERE_RERANK_ENABLED must be true or false"
    assert problems[2] == "CHAT_RATE_LIMIT_PER_DAY must be a positive whole number"
    assert "could not be read" in problems[3]


def test_unexpected_check_failure_reports_only_its_type(monkeypatch, good_kb):
    def broken(environ=None):
        raise KeyError(FAKE_KEY)

    monkeypatch.setattr(startup, "chat_timeout", broken)
    assert startup.find_problems(valid_environ(), good_kb) == ["Unexpected KeyError while checking settings"]


# --- Credential and setting values never appear in errors --------------------


def test_errors_never_contain_setting_values(tmp_path):
    """Even if the key is pasted into the wrong setting, no error repeats it."""
    environ = {name: FAKE_KEY for name in SETTINGS if name != "COHERE_API_KEY"}
    problems = startup.find_problems(environ, tmp_path / "missing.json")

    # Key missing, rerank (first bad setting), 2 timeouts, rate limit (first bad
    # setting), log level, KB.
    assert len(problems) == 7
    assert all(FAKE_KEY not in problem and "SECRET" not in problem for problem in problems)


def test_check_startup_error_never_contains_the_key(env, caplog):
    env.setenv("COHERE_RERANK_ENABLED", FAKE_KEY)

    with pytest.raises(startup.StartupError) as raised:
        startup.check_startup()

    message = str(raised.value)
    assert message == "The support agent cannot start: COHERE_RERANK_ENABLED must be true or false"
    assert FAKE_KEY not in message and FAKE_KEY not in caplog.text
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_check_startup_passes_with_valid_settings(env):
    startup.check_startup()  # does not raise


def test_check_startup_loads_dotenv_first(env, monkeypatch):
    """Local development keeps working: .env is loaded before checking.
    (Here a fake loader sets the key; the real .env is never read.)"""
    env.delenv("COHERE_API_KEY")
    monkeypatch.setattr(startup, "load_dotenv", lambda: env.setenv("COHERE_API_KEY", FAKE_KEY))

    startup.check_startup()


# --- The app lifecycle, /health and /ready -----------------------------------


def test_ready_after_successful_startup(env, offline):
    with TestClient(app) as client:
        response = client.get("/ready")
        health = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert health.status_code == 200 and health.json() == {"status": "ok"}
    assert offline == []  # the agent was never called


def test_startup_fails_on_invalid_configuration(env, offline):
    env.delenv("COHERE_API_KEY")

    with pytest.raises(startup.StartupError, match="COHERE_API_KEY is not set"):
        with TestClient(app):
            pass
    assert app.state.ready is False


def test_startup_fails_on_broken_knowledge_base(env, offline, tmp_path):
    env.setattr(kb, "KB_PATH", write_kb(tmp_path, "{not json"))

    with pytest.raises(startup.StartupError, match="is not valid JSON"):
        with TestClient(app):
            pass


def test_ready_is_503_until_startup_has_run(offline):
    """Without the lifecycle (startup not run or not finished) the app is not ready,
    but it is still alive."""
    app.state.ready = False
    client = TestClient(app)  # no `with`: startup does not run

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_is_503_after_shutdown(env, offline):
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
    assert app.state.ready is False
    assert TestClient(app).get("/ready").status_code == 503


def test_ready_probe_does_no_work(env, offline, monkeypatch):
    """Probes only read the flag set at startup: no re-validation, no KB read,
    no embeddings, no Cohere call."""
    with TestClient(app) as client:
        def fail(*args, **kwargs):
            raise AssertionError("readiness probe did work")

        monkeypatch.setattr(startup, "find_problems", fail)
        monkeypatch.setattr(kb, "load_articles", fail)
        responses = [client.get("/ready") for _ in range(5)]

    assert all(r.status_code == 200 for r in responses)


def test_ready_response_has_no_internal_details(env, offline):
    with TestClient(app) as client:
        ok = client.get("/ready")
    not_ok = TestClient(app).get("/ready")

    for response in (ok, not_ok):
        assert set(response.json()) == {"status"}
        assert FAKE_KEY not in response.text


def test_validation_does_not_import_cohere(env, monkeypatch):
    """Validation itself needs no Cohere SDK: with `cohere` unimportable it
    still passes. (The clients are created after it; see test_lifecycle.py.)"""
    monkeypatch.setitem(sys.modules, "cohere", None)  # `import cohere` -> ImportError

    startup.check_startup()  # does not raise


def test_ready_is_in_api_schema():
    paths = TestClient(app).get("/openapi.json").json()["paths"]
    assert "/ready" in paths and "get" in paths["/ready"]


# --- Rate-limit settings from .env (loaded before the limiter is built) ------


def test_no_settings_use_default_limits(env, offline):
    with TestClient(app):
        limiter = api.chat_limiter
    assert (limiter.per_minute, limiter.per_day) == (DEFAULT_PER_MINUTE, DEFAULT_PER_DAY)


def test_valid_rate_limits_in_dotenv_are_applied(dotenv_file, offline):
    dotenv_file("CHAT_RATE_LIMIT_PER_MINUTE=2\nCHAT_RATE_LIMIT_PER_DAY=50\n")

    with TestClient(app):
        limiter = api.chat_limiter
    assert (limiter.per_minute, limiter.per_day) == (2, 50)


def test_dotenv_rate_limit_is_enforced_on_chat(dotenv_file, monkeypatch):
    """End to end: a limit set only in .env decides when /chat returns 429."""
    dotenv_file("CHAT_RATE_LIMIT_PER_MINUTE=2\n")
    answer = agent.SupportAnswer(answer="Hi", sources=[], escalate=True, reason="no_articles")
    questions = []
    monkeypatch.setattr(agent, "answer_question", lambda q: questions.append(q) or answer)

    with TestClient(app) as client:
        statuses = [client.post("/chat", json={"question": "hello"}).status_code for _ in range(3)]

    assert statuses == [200, 200, 429]
    assert len(questions) == 2


def test_real_environment_wins_over_dotenv(dotenv_file, offline):
    """Production: Render's environment variables are used even if a .env
    file says otherwise (python-dotenv never overrides them)."""
    dotenv_file("CHAT_RATE_LIMIT_PER_MINUTE=2\nCHAT_RATE_LIMIT_PER_DAY=50\n")
    os.environ["CHAT_RATE_LIMIT_PER_MINUTE"] = "7"

    with TestClient(app):
        limiter = api.chat_limiter
    assert (limiter.per_minute, limiter.per_day) == (7, 50)


def test_blank_rate_limit_in_dotenv_uses_default(dotenv_file, offline):
    dotenv_file("CHAT_RATE_LIMIT_PER_MINUTE=\n")

    with TestClient(app):
        limiter = api.chat_limiter
    assert limiter.per_minute == DEFAULT_PER_MINUTE


@pytest.mark.parametrize("name", ["CHAT_RATE_LIMIT_PER_MINUTE", "CHAT_RATE_LIMIT_PER_DAY"])
@pytest.mark.parametrize("value", ["0", "-3", "abc", "2.5"])
def test_invalid_rate_limit_in_dotenv_fails_startup(dotenv_file, offline, name, value):
    """The same rule as in the real environment, and the value is not shown."""
    dotenv_file(f"{name}={value}\n")

    with pytest.raises(startup.StartupError) as raised:
        with TestClient(app):
            pass

    assert str(raised.value) == f"The support agent cannot start: {name} must be a positive whole number"
    assert api.chat_limiter is None and app.state.ready is False


def test_invalid_rate_limit_is_reported_with_other_problems(dotenv_file, offline):
    """No longer an import-time crash: it is listed with the other problems."""
    dotenv_file("CHAT_RATE_LIMIT_PER_DAY=0\nCOHERE_RERANK_ENABLED=ture\n")

    with pytest.raises(startup.StartupError) as raised:
        with TestClient(app):
            pass
    assert "COHERE_RERANK_ENABLED must be true or false" in str(raised.value)
    assert "CHAT_RATE_LIMIT_PER_DAY must be a positive whole number" in str(raised.value)


def test_importing_the_api_does_not_read_rate_limits():
    """An invalid value no longer breaks `import support_agent.api`; startup
    reports it instead. Run in a separate process so this test's import can't
    replace the app other tests use. (Importing the API reads no .env and
    makes no network call.)"""
    import subprocess

    environ = {**os.environ, "CHAT_RATE_LIMIT_PER_MINUTE": "abc"}
    result = subprocess.run(
        [sys.executable, "-c", "import support_agent.api as api; print(api.chat_limiter)"],
        env=environ, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "None"


def test_chat_before_startup_is_503_and_never_calls_agent(offline):
    """Without startup there is no limiter, so /chat refuses instead of
    calling Cohere with no limit."""
    response = TestClient(app).post("/chat", json={"question": "hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": api.NOT_READY_MESSAGE}
    assert offline == []
