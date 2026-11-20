"""Offline tests for conversations in the API: startup, /chat persistence,
GET /conversations/{id}, failures and logging. The agent is a fake (no Cohere
calls); the database is a temporary SQLite file (conftest.conversation_db)."""

import logging
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from support_agent import agent, api, conversation_store, log_config, startup
from support_agent.answer import CLARIFICATION_MESSAGE, HANDOFF_MESSAGE, SupportAnswer
from support_agent.conversation_store import ConversationStore, StoreError
from support_agent.rate_limit import RateLimiter

client = TestClient(api.app)

SOURCE = {"id": "KB-002", "title": "Refund policy", "url": "https://support.example.com/refunds"}
ANSWERED = SupportAnswer(answer="Refunds are possible within 30 days of purchase.", sources=[SOURCE])
HANDOFF = SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason="insufficient_evidence")
CLARIFY = SupportAnswer(answer=CLARIFICATION_MESSAGE, sources=[], escalate=False, reason="needs_clarification")


@pytest.fixture(autouse=True)
def limiter(monkeypatch):
    limiter = RateLimiter(per_minute=100, per_day=1000)
    monkeypatch.setattr(api, "chat_limiter", limiter)
    return limiter


class FakeAgent:
    """Records each call's question, history and trace; can fill the trace
    like the real agent does."""

    def __init__(self):
        self.calls = []
        self.result = ANSWERED
        self.error = None
        self.rewrite = None  # (retrieval_query, outcome) to put in the trace

    def __call__(self, question, *, history=None, trace=None):
        self.calls.append({"question": question, "history": history})
        if self.rewrite and trace is not None:
            trace["retrieval_query"], trace["rewrite"] = self.rewrite
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def fake_agent(monkeypatch):
    fake = FakeAgent()
    monkeypatch.setattr(agent, "answer_question", fake)
    return fake


def ask(question="Can I get a refund?", conversation_id=None, **kwargs):
    body = {"question": question}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    return client.post("/chat", json=body, **kwargs)


def rows(store, sql, *args):
    with sqlite3.connect(store.path) as connection:
        return connection.execute(sql, args).fetchall()


# --- Startup and readiness ---------------------------------------------------------


def test_invalid_db_path_setting_is_a_startup_problem(tmp_path):
    problems = startup.find_problems({"COHERE_API_KEY": "k", conversation_store.DB_PATH_ENV: str(tmp_path)})
    assert problems == [f"{conversation_store.DB_PATH_ENV} must be a file path, not a folder"]


def test_invalid_rewrite_timeout_is_a_startup_problem():
    problems = startup.find_problems({"COHERE_API_KEY": "k", "COHERE_REWRITE_TIMEOUT_SECONDS": "soon"})
    assert problems == ["COHERE_REWRITE_TIMEOUT_SECONDS must be a number of seconds"]


@pytest.fixture
def offline_startup(monkeypatch, semantic_kb):
    for name in ("CHAT_RATE_LIMIT_PER_MINUTE", "CHAT_RATE_LIMIT_PER_DAY", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api, "chat_limiter", None)
    monkeypatch.setattr(api, "conversations", None)


def test_startup_opens_the_configured_database_before_ready(offline_startup, monkeypatch, tmp_path):
    path = tmp_path / "startup" / "conv.sqlite3"
    monkeypatch.setenv(conversation_store.DB_PATH_ENV, str(path))

    with TestClient(api.app) as started:
        assert started.get("/ready").status_code == 200
        assert api.conversations.path == path
        assert path.exists()


def test_unusable_database_fails_startup_and_never_reports_ready(offline_startup, monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv(conversation_store.DB_PATH_ENV, str(blocker / "conv.sqlite3"))

    with pytest.raises(startup.StartupError, match="Conversation database"):
        with TestClient(api.app):
            pass
    assert api.app.state.ready is False
    assert api.conversations is None


def test_chat_is_503_without_a_store(fake_agent, monkeypatch):
    monkeypatch.setattr(api, "conversations", None)
    response = ask()
    assert response.status_code == 503 and response.json() == {"detail": api.NOT_READY_MESSAGE}
    assert fake_agent.calls == []


# --- A new conversation ---------------------------------------------------------------


def test_first_message_starts_a_conversation_and_is_saved(fake_agent, conversation_db):
    response = ask("  Can I get a refund?  ")

    body = response.json()
    assert response.status_code == 200
    assert conversation_store.is_valid_id(body["conversation_id"])
    assert fake_agent.calls == [{"question": "Can I get a refund?", "history": []}]
    messages = conversation_db.messages(body["conversation_id"])
    assert [(m.role, m.content) for m in messages] == [
        ("customer", "Can I get a refund?"), ("assistant", ANSWERED.answer),
    ]
    assert messages[1].sources == [SOURCE] and messages[1].reason == "answered"


def test_every_new_conversation_gets_its_own_id(fake_agent):
    ids = {ask().json()["conversation_id"] for _ in range(5)}
    assert len(ids) == 5


def test_client_cannot_choose_a_new_conversation_id(fake_agent, conversation_db):
    """A well-formed but unknown ID is not created on demand: 404."""
    chosen = "0123456789abcdef0123456789abcdef"
    response = ask(conversation_id=chosen)
    assert response.status_code == 404
    assert response.json() == {"detail": api.CONVERSATION_NOT_FOUND_MESSAGE}
    assert fake_agent.calls == []
    assert rows(conversation_db, "SELECT count(*) FROM conversations") == [(0,)]


@pytest.mark.parametrize("bad", [
    "", "abc", "0" * 31, "0" * 33, "A" * 32, "g" * 32, "../../etc/passwd", 123, ["x"],
    "0123456789abcdef0123456789abcdef\n",
])
def test_malformed_conversation_ids_are_rejected_with_422(fake_agent, limiter, bad):
    response = client.post("/chat", json={"question": "hi", "conversation_id": bad})
    assert response.status_code == 422
    assert fake_agent.calls == []


def test_null_conversation_id_starts_a_new_conversation(fake_agent):
    response = client.post("/chat", json={"question": "hi", "conversation_id": None})
    assert response.status_code == 200 and conversation_store.is_valid_id(response.json()["conversation_id"])


# --- Follow-ups --------------------------------------------------------------------


def test_follow_up_gets_its_own_history_only(fake_agent, conversation_db):
    first = ask("How can I return my order?").json()["conversation_id"]
    other = ask("Where is my package?").json()["conversation_id"]
    fake_agent.result = HANDOFF
    ask("What if I bought it 35 days ago?", first)

    assert fake_agent.calls[-1] == {
        "question": "What if I bought it 35 days ago?",
        "history": [
            {"role": "customer", "content": "How can I return my order?"},
            {"role": "assistant", "content": ANSWERED.answer},
        ],
    }
    assert len(conversation_db.messages(first)) == 4
    assert [m.content for m in conversation_db.messages(other)] == ["Where is my package?", ANSWERED.answer]


def test_rewritten_query_is_stored_apart_from_the_customer_message(fake_agent, conversation_db):
    conversation_id = ask("How can I return my order?").json()["conversation_id"]
    fake_agent.rewrite = ("Is a return possible 35 days after purchase?", "rewritten")

    response = ask("What if I bought it 35 days ago?", conversation_id)

    assert response.status_code == 200
    assert "35 days after purchase" not in response.text  # never shown to the customer
    assert rows(conversation_db, "SELECT content, retrieval_query FROM messages WHERE role='customer' ORDER BY id") == [
        ("How can I return my order?", None),
        ("What if I bought it 35 days ago?", "Is a return possible 35 days after purchase?"),
    ]


def test_clarification_is_returned_and_saved(fake_agent, conversation_db):
    conversation_id = ask().json()["conversation_id"]
    fake_agent.result = CLARIFY

    body = ask("Can I do that too?", conversation_id).json()

    assert body == {
        "answer": CLARIFICATION_MESSAGE, "sources": [], "escalate": False,
        "reason": "needs_clarification", "conversation_id": conversation_id,
    }
    assert conversation_db.messages(conversation_id)[-1].reason == "needs_clarification"


def test_full_conversation_is_refused_before_the_agent_runs(fake_agent, monkeypatch, tmp_path, limiter):
    store = ConversationStore(tmp_path / "small.sqlite3", max_messages=4)
    store.initialize()
    monkeypatch.setattr(api, "conversations", store)
    conversation_id = ask("q1").json()["conversation_id"]
    assert ask("q2", conversation_id).status_code == 200
    calls = len(fake_agent.calls)

    response = ask("q3", conversation_id)

    assert response.status_code == 409
    assert response.json() == {"detail": api.CONVERSATION_FULL_MESSAGE}
    assert len(fake_agent.calls) == calls
    assert [m.content for m in store.messages(conversation_id) if m.role == "customer"] == ["q1", "q2"]


def test_refused_conversations_do_not_use_quota(fake_agent, monkeypatch):
    clock_limiter = RateLimiter(per_minute=1, per_day=1)
    monkeypatch.setattr(api, "chat_limiter", clock_limiter)
    for _ in range(3):
        assert ask(conversation_id="f" * 32).status_code == 404
    assert ask().status_code == 200


def test_conversation_filled_by_a_parallel_request_is_409_not_truncated(fake_agent, monkeypatch, tmp_path):
    store = ConversationStore(tmp_path / "small.sqlite3", max_messages=4)
    store.initialize()
    monkeypatch.setattr(api, "conversations", store)
    conversation_id = ask("q1").json()["conversation_id"]

    def answer_while_another_request_fills_it(question, **kwargs):
        store.add_exchange(conversation_id, "parallel", conversation_store.AssistantReply("a", [], True, "x"))
        return ANSWERED

    monkeypatch.setattr(agent, "answer_question", answer_while_another_request_fills_it)
    response = ask("q2", conversation_id)

    assert response.status_code == 409
    assert [m.content for m in store.messages(conversation_id) if m.role == "customer"] == ["q1", "parallel"]


def test_conversation_expired_during_the_request_is_404(fake_agent, monkeypatch, conversation_db):
    conversation_id = ask().json()["conversation_id"]

    def answer_then_expire(question, **kwargs):
        conversation_db.clock = lambda: 10**12  # far in the future: expired
        return ANSWERED

    monkeypatch.setattr(agent, "answer_question", answer_then_expire)
    assert ask("again", conversation_id).status_code == 404


# --- Storage failures -----------------------------------------------------------------


class BrokenStore(ConversationStore):
    def __init__(self, path, fail_on):
        super().__init__(path)
        self.fail_on = fail_on

    def ping(self):
        if "ping" in self.fail_on:
            raise StoreError("Conversation database error: OperationalError")
        super().ping()

    def messages(self, conversation_id):
        if "read" in self.fail_on:
            raise StoreError("Conversation database error: OperationalError")
        return super().messages(conversation_id)

    def add_exchange(self, *args, **kwargs):
        if "write" in self.fail_on:
            raise StoreError("Conversation database error: OperationalError")
        return super().add_exchange(*args, **kwargs)


def use_broken(monkeypatch, tmp_path, *fail_on):
    store = BrokenStore(tmp_path / "broken.sqlite3", set(fail_on))
    store.fail_on = set()
    store.initialize()
    store.fail_on = set(fail_on)
    monkeypatch.setattr(api, "conversations", store)
    return store


def test_unavailable_storage_is_detected_before_the_agent_runs(fake_agent, monkeypatch, tmp_path):
    use_broken(monkeypatch, tmp_path, "ping")
    response = ask()
    assert response.status_code == 503
    assert response.json() == {"detail": api.STORAGE_UNAVAILABLE_MESSAGE}
    assert fake_agent.calls == []


def test_unreadable_conversation_is_503_before_the_agent_runs(fake_agent, monkeypatch, tmp_path):
    store = use_broken(monkeypatch, tmp_path)
    conversation_id = ask().json()["conversation_id"]
    store.fail_on = {"read"}
    assert ask("again", conversation_id).status_code == 503
    assert len(fake_agent.calls) == 1


def test_failed_save_never_returns_the_answer(fake_agent, monkeypatch, tmp_path):
    """The answer was generated but couldn't be saved: the customer gets a
    503 and no answer text, so nothing is shown that the conversation lacks."""
    store = use_broken(monkeypatch, tmp_path, "write")

    response = ask()

    assert response.status_code == 503
    assert response.json() == {"detail": api.STORAGE_UNAVAILABLE_MESSAGE}
    assert ANSWERED.answer not in response.text and "conversation_id" not in response.text
    assert rows(store, "SELECT count(*) FROM messages") == [(0,)]


def test_agent_error_is_saved_as_a_safe_hand_off(fake_agent, conversation_db):
    fake_agent.error = RuntimeError("SECRET internal detail")

    response = ask()

    assert response.status_code == 500
    body = response.json()
    assert body["answer"] == HANDOFF_MESSAGE and body["reason"] == "server_error" and body["escalate"] is True
    assert "SECRET" not in response.text
    stored = conversation_db.messages(body["conversation_id"])
    assert stored[1].content == HANDOFF_MESSAGE and stored[1].reason == "server_error"


def test_purge_failure_does_not_affect_the_saved_answer(fake_agent, monkeypatch, conversation_db, caplog):
    def broken_purge():
        raise StoreError("Conversation database error: OperationalError")

    monkeypatch.setattr(conversation_db, "purge_if_due", broken_purge)
    caplog.set_level(logging.WARNING)
    response = ask()
    assert response.status_code == 200
    assert "Removing expired conversations failed" in caplog.text


# --- GET /conversations/{id} ---------------------------------------------------------


def test_get_conversation_restores_messages_in_order(fake_agent):
    conversation_id = ask("<b>first</b>").json()["conversation_id"]
    fake_agent.result = HANDOFF
    ask("second", conversation_id)

    response = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == conversation_id and body["message_limit"] == 40
    assert [(m["role"], m["content"], m["sources"], m["escalate"], m["reason"]) for m in body["messages"]] == [
        ("customer", "<b>first</b>", [], None, None),
        ("assistant", ANSWERED.answer, [SOURCE], False, "answered"),
        ("customer", "second", [], None, None),
        ("assistant", HANDOFF_MESSAGE, [], True, "insufficient_evidence"),
    ]
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", m["created_at"]) for m in body["messages"])


def test_get_conversation_never_returns_the_rewritten_query(fake_agent):
    conversation_id = ask().json()["conversation_id"]
    fake_agent.rewrite = ("INTERNAL REWRITE", "rewritten")
    ask("follow-up", conversation_id)
    assert "INTERNAL REWRITE" not in client.get(f"/conversations/{conversation_id}").text


def test_get_unknown_conversation_is_404():
    response = client.get(f"/conversations/{'a' * 32}")
    assert response.status_code == 404 and response.json() == {"detail": api.CONVERSATION_NOT_FOUND_MESSAGE}


@pytest.mark.parametrize("bad", ["abc", "A" * 32, "a" * 33, "a" * 31 + "%00"])
def test_get_malformed_id_is_422(bad):
    assert client.get(f"/conversations/{bad}").status_code == 422


def test_get_conversation_without_store_or_with_broken_store(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "conversations", None)
    assert client.get(f"/conversations/{'a' * 32}").status_code == 503
    use_broken(monkeypatch, tmp_path, "read")
    response = client.get(f"/conversations/{'a' * 32}")
    assert response.status_code == 503 and "OperationalError" not in response.text


def test_get_conversation_makes_no_agent_call(fake_agent):
    conversation_id = ask().json()["conversation_id"]
    client.get(f"/conversations/{conversation_id}")
    assert len(fake_agent.calls) == 1


def test_endpoints_are_in_the_api_schema():
    paths = client.get("/openapi.json").json()["paths"]
    assert "get" in paths["/conversations/{conversation_id}"]


# --- Logging never contains the conversation ID ---------------------------------------


def test_outcome_log_has_a_hash_turn_and_rewrite_but_never_the_id(fake_agent, caplog):
    caplog.set_level(logging.DEBUG)
    conversation_id = ask().json()["conversation_id"]
    fake_agent.rewrite = ("standalone", "rewritten")
    ask("follow-up", conversation_id)

    outcomes = [r.chat_outcome for r in caplog.records if hasattr(r, "chat_outcome")]
    assert [(o["turn"], o["rewrite"]) for o in outcomes] == [(1, "none"), (2, "rewritten")]
    assert {o["conversation"] for o in outcomes} == {conversation_store.log_hash(conversation_id)}
    assert conversation_id not in caplog.text


def test_refused_requests_log_a_reason_without_the_id(fake_agent, caplog):
    caplog.set_level(logging.DEBUG)
    unknown = "b" * 32
    ask(conversation_id=unknown)
    (outcome,) = [r.chat_outcome for r in caplog.records if hasattr(r, "chat_outcome")]
    assert outcome["reason"] == "conversation_not_found" and outcome["status"] == 404
    assert unknown not in caplog.text


@pytest.mark.parametrize("path, expected", [
    ("/conversations/0123456789abcdef0123456789abcdef", "/conversations/<redacted>"),
    ("/conversations/0123456789abcdef0123456789abcdef?x=1", "/conversations/<redacted>?x=1"),
    ("/conversations/anything-else", "/conversations/<redacted>"),
    ("/chat", "/chat"),
])
def test_access_log_redacts_conversation_ids(path, expected):
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", path, "1.1", 200), None,
    )
    log_config.ConversationIdRedactor().filter(record)
    assert record.getMessage() == f'127.0.0.1:5000 - "GET {expected} HTTP/1.1" 200'


def test_configure_logging_installs_the_access_log_filter_once():
    access = logging.getLogger(log_config.ACCESS_LOGGER)
    saved = list(access.filters)
    try:
        log_config.configure_logging(logging.INFO)
        log_config.configure_logging(logging.INFO)
        assert sum(isinstance(f, log_config.ConversationIdRedactor) for f in access.filters) == 1
    finally:
        access.filters[:] = saved
