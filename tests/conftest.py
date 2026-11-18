import importlib
import logging
import sys
import types
from types import SimpleNamespace

import pytest

import support_agent
from support_agent import agent, cohere_calls, startup


@pytest.fixture(autouse=True)
def startup_never_reads_dotenv(monkeypatch):
    """Startup validation and the chat client (created at startup) call
    load_dotenv(); in tests they must never read the real .env, so settings
    come only from what each test sets."""
    monkeypatch.setattr(startup, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(agent, "load_dotenv", lambda *args, **kwargs: False)


@pytest.fixture(autouse=True)
def no_cached_chat_client():
    """No test inherits a chat client created by another test's startup."""
    agent._default_client.cache_clear()
    yield
    agent._default_client.cache_clear()


@pytest.fixture(autouse=True)
def restore_package_logger():
    """App startup configures the `support_agent` logger (handler, level,
    propagate=False). Undo that after each test, so later tests that read
    logs with caplog (through the root logger) still see records."""
    package_logger = logging.getLogger("support_agent")
    saved = (list(package_logger.handlers), package_logger.level, package_logger.propagate)
    yield
    package_logger.handlers[:] = saved[0]
    package_logger.setLevel(saved[1])
    package_logger.propagate = saved[2]


@pytest.fixture(autouse=True)
def no_real_retry_waits(monkeypatch):
    """Retries never really wait in tests. Returns the list of requested waits."""
    waits = []
    monkeypatch.setattr(cohere_calls, "sleep", waits.append)
    return waits


class FakeEmbedClient:
    """Stands in for cohere.ClientV2. Outcomes: an exception, "ok" (one vector
    per text) or a ready-made response."""

    instances = []  # Every client created, in order (reset by the semantic_kb fixture).

    def __init__(self, api_key):
        self.api_key = api_key
        self.outcomes = []
        self.calls = []
        self.closed = False
        self.on_close = None  # Optional callback, e.g. to record state at close time.
        FakeEmbedClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        # Like cohere.ClientV2: leaving `with` closes the connection pool.
        if self.on_close:
            self.on_close(self)
        self.closed = True

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "ok":
            return SimpleNamespace(embeddings=SimpleNamespace(float=[[1.0, 0.0] for _ in kwargs["texts"]]))
        return outcome


_FRESH_MODULES = ("support_agent.semantic_kb", "support_agent.hybrid_kb")


@pytest.fixture
def import_semantic_kb(monkeypatch):
    """Returns import_fresh(api_key): imports semantic_kb fresh (hybrid_kb is
    re-imported on next use) with a fake `cohere` and a no-op `dotenv`, so no
    client is real and .env is never read. `api_key` is put in COHERE_API_KEY first
    (None removes it). The previous modules, if any, are restored afterwards."""
    fake_cohere = types.ModuleType("cohere")
    fake_cohere.ClientV2 = FakeEmbedClient
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, "cohere", fake_cohere)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setattr(FakeEmbedClient, "instances", [])

    saved_modules = {name: sys.modules.pop(name, None) for name in _FRESH_MODULES}
    saved_attrs = {name: support_agent.__dict__.get(name.rsplit(".", 1)[1]) for name in _FRESH_MODULES}

    def import_fresh(api_key="fake-key-for-tests"):
        if api_key is None:
            monkeypatch.delenv("COHERE_API_KEY", raising=False)
        else:
            monkeypatch.setenv("COHERE_API_KEY", api_key)
        for name in _FRESH_MODULES:
            sys.modules.pop(name, None)
            support_agent.__dict__.pop(name.rsplit(".", 1)[1], None)
        return importlib.import_module("support_agent.semantic_kb")

    try:
        yield import_fresh
    finally:
        for name in _FRESH_MODULES:
            attr = name.rsplit(".", 1)[1]
            sys.modules.pop(name, None)
            support_agent.__dict__.pop(attr, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]
            if saved_attrs[name] is not None:
                setattr(support_agent, attr, saved_attrs[name])


@pytest.fixture
def semantic_kb(import_semantic_kb):
    """semantic_kb imported fresh with the fake SDK and a fake key."""
    module = import_semantic_kb()
    assert isinstance(module.client, FakeEmbedClient)
    return module
