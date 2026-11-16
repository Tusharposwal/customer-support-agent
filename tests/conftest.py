import importlib
import sys
import types
from types import SimpleNamespace

import pytest

import support_agent
from support_agent import cohere_calls


@pytest.fixture(autouse=True)
def no_real_retry_waits(monkeypatch):
    """Retries never really wait in tests. Returns the list of requested waits."""
    waits = []
    monkeypatch.setattr(cohere_calls, "sleep", waits.append)
    return waits


class FakeEmbedClient:
    """Stands in for cohere.ClientV2. Outcomes: an exception, "ok" (one vector
    per text) or a ready-made response."""

    def __init__(self, api_key):
        self.api_key = api_key
        self.outcomes = []
        self.calls = []

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
def semantic_kb(monkeypatch):
    """Import semantic_kb (and hybrid_kb, which uses it) with a fake `cohere`
    and a no-op `dotenv`, so no client is real and .env is never read. The
    previous modules, if any, are restored afterwards."""
    fake_cohere = types.ModuleType("cohere")
    fake_cohere.ClientV2 = FakeEmbedClient
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, "cohere", fake_cohere)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setenv("COHERE_API_KEY", "fake-key-for-tests")

    saved_modules = {name: sys.modules.pop(name, None) for name in _FRESH_MODULES}
    saved_attrs = {name: support_agent.__dict__.get(name.rsplit(".", 1)[1]) for name in _FRESH_MODULES}
    try:
        module = importlib.import_module("support_agent.semantic_kb")
        assert isinstance(module.client, FakeEmbedClient)
        yield module
    finally:
        for name in _FRESH_MODULES:
            attr = name.rsplit(".", 1)[1]
            sys.modules.pop(name, None)
            support_agent.__dict__.pop(attr, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]
            if saved_attrs[name] is not None:
                setattr(support_agent, attr, saved_attrs[name])
