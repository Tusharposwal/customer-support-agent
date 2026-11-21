"""Offline tests for the app lifecycle: client creation at startup, cleanup at
shutdown and after a failed startup, and readiness while shutting down.

The Cohere SDK is the fake from conftest.py (`semantic_kb` fixture): clients
are FakeEmbedClient objects that record requests and closing. Sockets refuse
to connect, and .env is never read.
"""

import asyncio
import logging
import socket
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from support_agent import agent, api, log_config, startup
from support_agent.answer import HANDOFF_MESSAGE

SETTINGS = (
    "COHERE_RERANK_ENABLED", "COHERE_RERANK_MODEL", "COHERE_RERANK_MIN_SCORE",
    "COHERE_RERANK_TIMEOUT_SECONDS", "COHERE_CHAT_TIMEOUT_SECONDS", "COHERE_EMBED_TIMEOUT_SECONDS",
    "CHAT_RATE_LIMIT_PER_MINUTE", "CHAT_RATE_LIMIT_PER_DAY", "LOG_LEVEL",
)


@pytest.fixture
def fake_sdk(monkeypatch, semantic_kb):
    """Valid settings, the fake Cohere SDK, no network. Returns the list of
    clients created so far (the embedding client the fixture imported first)."""
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api, "chat_limiter", None)

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    return type(semantic_kb.client).instances


@pytest.fixture
def app_log(caplog):
    """caplog for app log lines. Startup gives the support_agent logger its own
    handler and stops propagation to the root logger (where caplog listens),
    so caplog's handler is attached to it directly (removed again by conftest)."""
    caplog.set_level(logging.DEBUG)
    logging.getLogger("support_agent").addHandler(caplog.handler)
    return caplog


def chat_client():
    """The cached chat client, or None (never creates one)."""
    return agent._default_client() if agent._default_client.cache_info().currsize else None


# --- Startup -----------------------------------------------------------------


def test_startup_order(fake_sdk, monkeypatch, semantic_kb):
    """Validate, configure logging, build the limiter, create the clients,
    and only then report ready."""
    events = []

    def record(name, real):
        def wrapper(*args, **kwargs):
            events.append((name, api.app.state.ready))
            return real(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(startup, "check_startup", record("validate", startup.check_startup))
    monkeypatch.setattr(log_config, "configure_logging", record("logging", log_config.configure_logging))
    monkeypatch.setattr(api, "limiter_from_env", record("limiter", api.limiter_from_env))
    monkeypatch.setattr(agent, "_default_client", record("chat client", agent._default_client))

    with TestClient(api.app) as client:
        events.append(("first request", client.get("/ready").json()["status"]))

    assert events == [
        ("validate", False), ("logging", False), ("limiter", False), ("chat client", False),
        ("first request", "ready"),
    ]


def test_clients_are_created_at_startup_without_any_request(fake_sdk, semantic_kb):
    embedding_client = semantic_kb.client
    assert fake_sdk == [embedding_client]  # only the embedding client so far

    with TestClient(api.app):
        chat = chat_client()
        assert chat is not None and chat is not embedding_client
        assert fake_sdk == [embedding_client, chat]
        assert chat.api_key == "fake-key-for-tests"
        # No embed or other call: creating clients sends nothing, and the
        # knowledge base is embedded only on first use.
        assert embedding_client.calls == [] and chat.calls == []
        assert semantic_kb._document_cache is None


@pytest.mark.parametrize("raw_key", ["key-123", "  key-123", "key-123\n", "\t key-123 \r\n"])
def test_embedding_and_chat_clients_get_the_same_normalized_key(import_semantic_kb, raw_key):
    """Both clients take the key from agent.api_key_from_env, so surrounding
    whitespace (e.g. a pasted newline) is removed for both, as startup
    validation already did."""
    semantic_kb = import_semantic_kb(raw_key)

    assert semantic_kb.API_KEY == "key-123"
    assert semantic_kb.client.api_key == "key-123"
    assert agent._default_client().api_key == "key-123"


@pytest.mark.parametrize("raw_key", [None, "", "   ", "\n"])
def test_embedding_client_without_a_key_fails_like_the_chat_client(import_semantic_kb, raw_key):
    with pytest.raises(agent.ConfigurationError) as raised:
        import_semantic_kb(raw_key)

    # The same error as the chat client and startup; no mention of .env, since
    # the key may come from the deployment environment.
    assert str(raised.value) == "COHERE_API_KEY is not set"
    assert isinstance(raised.value, RuntimeError)  # as before, for existing callers
    with pytest.raises(agent.ConfigurationError, match="^COHERE_API_KEY is not set$"):
        agent.api_key_from_env()


def test_validation_failure_creates_no_clients(fake_sdk, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY")

    with pytest.raises(startup.StartupError):
        with TestClient(api.app):
            pass

    assert chat_client() is None and len(fake_sdk) == 1  # just the fixture's embedding client


def test_first_request_reuses_the_startup_clients(fake_sdk, semantic_kb):
    """The real pipeline runs on the first question; it creates no new client.
    (The fake can embed but can't chat, so the answer is a safe hand-off.)"""
    with TestClient(api.app) as client:
        created_at_startup = list(fake_sdk)
        response = client.post("/chat", json={"question": "How do I reset my password?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": HANDOFF_MESSAGE, "sources": [], "escalate": True, "reason": "llm_error",
        "conversation_id": response.json()["conversation_id"], "message_index": 1,
    }
    assert fake_sdk == created_at_startup  # no client created by the request
    assert len(semantic_kb.client.calls) == 2  # knowledge base + question, embedded on first use


# --- Shutdown and cleanup ---------------------------------------------------


def test_clients_are_closed_after_shutdown(fake_sdk, semantic_kb):
    with TestClient(api.app):
        chat = chat_client()
        assert not chat.closed and not semantic_kb.client.closed

    assert chat.closed and semantic_kb.client.closed
    assert chat_client() is None  # the next startup would create a new one
    assert api.app.state.ready is False


def test_ready_is_false_while_clients_close(fake_sdk, semantic_kb):
    seen = []
    with TestClient(api.app):
        for client in (chat_client(), semantic_kb.client):
            client.on_close = lambda c: seen.append(api.app.state.ready)

    assert seen == [False, False]


def test_partial_startup_failure_closes_what_was_created(fake_sdk, semantic_kb, monkeypatch):
    """The chat client is created, then the embedding client fails: the chat
    client is closed and the original error is what startup reports."""
    def failing_import(name):
        raise RuntimeError("embedding client could not be created")

    monkeypatch.setattr(api.importlib, "import_module", failing_import)

    with pytest.raises(RuntimeError, match="embedding client could not be created"):
        with TestClient(api.app):
            pass

    chat = fake_sdk[-1]
    assert chat is not semantic_kb.client and chat.closed
    assert chat_client() is None
    assert api.app.state.ready is False


def test_cleanup_errors_do_not_hide_the_startup_error(fake_sdk, semantic_kb, monkeypatch, app_log):
    def failing_import(name):
        raise RuntimeError("original startup error")

    def failing_close(client):
        raise OSError("close failed SECRET-DETAIL")

    monkeypatch.setattr(api.importlib, "import_module", failing_import)
    monkeypatch.setattr(agent, "close_client", failing_close)

    with pytest.raises(RuntimeError, match="original startup error"):
        with TestClient(api.app):
            pass

    assert "Shutdown: closing the Cohere chat client failed: OSError" in app_log.text
    assert "SECRET-DETAIL" not in app_log.text


def test_close_failure_at_shutdown_is_logged_not_raised(fake_sdk, semantic_kb, app_log):

    def broken(client):
        raise ValueError("pool already gone")

    with TestClient(api.app) as client:
        assert client.get("/ready").status_code == 200
        chat = chat_client()
        chat.on_close = broken

    assert "closing the Cohere chat client failed: ValueError" in app_log.text
    assert semantic_kb.client.closed  # the other client is still closed
    assert "pool already gone" not in app_log.text


TIMEOUT = 5  # Upper bound for every wait below, so a broken test fails instead of hanging.


def test_shutdown_waits_for_a_running_chat_before_closing(fake_sdk, semantic_kb, monkeypatch):
    """A /chat still using the clients finishes before they are closed.

    Synchronised with events, not sleeps: the worker signals that it is inside
    the request, and shutdown's wait loop signals (through a stand-in for the
    api module's `asyncio`) that it has started waiting for it."""
    order = []
    request_started = threading.Event()
    shutdown_waiting = threading.Event()

    real_sleep = asyncio.sleep

    async def signalling_sleep(seconds):
        shutdown_waiting.set()
        await real_sleep(seconds)

    monkeypatch.setattr(api, "asyncio", types.SimpleNamespace(sleep=signalling_sleep))

    def running_request():
        with api._chat_in_progress():
            request_started.set()
            assert shutdown_waiting.wait(TIMEOUT), "shutdown never started waiting"
            order.append("request finished")

    with TestClient(api.app):
        chat_client().on_close = lambda c: order.append("closed")
        worker = threading.Thread(target=running_request, daemon=True)
        worker.start()
        assert request_started.wait(TIMEOUT), "worker never started"
    # Leaving the block ran shutdown: it waited for the request, then closed.
    worker.join(TIMEOUT)

    assert not worker.is_alive()
    assert shutdown_waiting.is_set()
    assert order == ["request finished", "closed"]
    assert api._chats_in_progress == 0


def test_shutdown_leaves_clients_open_if_a_chat_never_finishes(fake_sdk, semantic_kb, app_log):
    """A request that outlasts the wait: the clients are left open, not
    closed under it. The worker is released only after shutdown returned."""
    agent._default_client()  # an open chat client, as after startup
    request_started = threading.Event()
    release = threading.Event()

    def stuck_request():
        with api._chat_in_progress():
            request_started.set()
            release.wait(TIMEOUT)

    worker = threading.Thread(target=stuck_request, daemon=True)
    worker.start()
    try:
        assert request_started.wait(TIMEOUT), "worker never started"
        started = time.monotonic()
        asyncio.run(api._close_clients(wait_seconds=0.3))
        waited = time.monotonic() - started
        assert worker.is_alive()  # still inside the request when shutdown gave up
    finally:
        release.set()
        worker.join(TIMEOUT)

    assert not worker.is_alive()
    assert waited >= 0.3  # it did wait the full time before giving up
    assert not chat_client().closed and not semantic_kb.client.closed
    assert "1 chat request(s) still running; Cohere clients left open" in app_log.text
    assert api._chats_in_progress == 0


def test_in_flight_count_returns_to_zero_after_errors(monkeypatch):
    def failing_agent(question, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent, "answer_question", failing_agent)
    from support_agent.rate_limit import RateLimiter

    monkeypatch.setattr(api, "chat_limiter", RateLimiter(per_minute=10, per_day=10))
    response = TestClient(api.app).post("/chat", json={"question": "hi"})

    assert response.status_code == 500
    assert api._chats_in_progress == 0


def test_close_without_startup_is_harmless(fake_sdk):
    """Nothing created yet: closing does nothing and raises nothing."""
    assert chat_client() is None
    asyncio.run(api._close_clients(wait_seconds=0))
    assert chat_client() is None


def test_close_client_uses_the_sdk_context_manager_exit():
    from support_agent.cohere_calls import close_client

    class SdkLike:
        exits = []

        def __exit__(self, *exc_info):
            self.exits.append(exc_info)

    client = SdkLike()
    close_client(client)
    close_client(object())  # no __exit__: left alone

    assert SdkLike.exits == [(None, None, None)]


def test_real_sdk_client_close_closes_its_http_pool():
    """Checks against the installed cohere SDK (no request is sent): closing
    goes through its documented `with` support and closes its httpx client."""
    import cohere

    from support_agent.cohere_calls import close_client

    client = cohere.ClientV2(api_key="fake-key-not-real")
    pool = client._client_wrapper.httpx_client.httpx_client
    assert not pool.is_closed

    close_client(client)

    assert pool.is_closed
