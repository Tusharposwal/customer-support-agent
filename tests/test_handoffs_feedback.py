"""Offline tests for hand-off records (handoffs.py) and reply feedback
(feedback.py, POST /feedback), and how /chat and GET /conversations use them.
Temporary SQLite files only; the agent is a fake. No Cohere calls."""

import logging
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from support_agent import agent, api, feedback, handoffs
from support_agent import conversation_store as cs
from support_agent.answer import CANCELLATION_AVAILABLE, HANDOFF_MESSAGE, SupportAnswer, cancellation_message
from support_agent.conversation_store import AssistantReply, ConversationNotFound, ConversationStore
from support_agent.feedback import FeedbackStore, MessageNotFound
from support_agent.handoffs import HandoffStore
from support_agent.rate_limit import RateLimiter

SOURCE = {"id": "KB-001", "title": "Reset your password", "url": "https://support.example.com/password-reset"}
ANSWER = AssistantReply("Use Forgot password.", [SOURCE], False, "answered")
ESCALATED = AssistantReply(HANDOFF_MESSAGE, [], True, "insufficient_evidence")
DAY = 24 * 3600


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def path(tmp_path):
    return tmp_path / "db" / "conversations.sqlite3"


def open_stores(path, clock):
    store = ConversationStore(path, clock=clock)
    store.initialize()
    records, ratings = HandoffStore(store), FeedbackStore(store)
    records.initialize()
    ratings.initialize()
    return store, records, ratings


@pytest.fixture
def stores(path, clock):
    return open_stores(path, clock)


def table(path, sql, *args):
    with sqlite3.connect(path) as connection:
        return connection.execute(sql, args).fetchall()


def exchange(store, records, reply=ESCALATED, conversation_id=None):
    """Save a question and reply as /chat does (with the hand-off record when escalated)."""
    create = conversation_id is None
    conversation_id = conversation_id or cs.new_conversation_id()
    record = records.recorder(conversation_id, reply.reason) if reply.escalate else None
    store.add_exchange(conversation_id, "question", reply, create=create, in_transaction=record)
    return conversation_id


# --- Hand-off records ----------------------------------------------------------------------


@pytest.mark.parametrize("reason, expected", [
    ("insufficient_evidence", ("insufficient_evidence", "no_answer")),
    ("below_rerank_threshold", ("below_rerank_threshold", "no_answer")),
    ("llm_error", ("llm_error", "service_problem")),
    ("server_error", ("server_error", "service_problem")),
    ("invalid_tool_call", ("invalid_tool_call", "service_problem")),
    ("COHERE_API_KEY=secret", ("other", "other")),  # Never stored as given.
    ("", ("other", "other")),
    (None, ("other", "other")),
    (42, ("other", "other")),
])
def test_escalation_reasons_are_validated(reason, expected):
    assert handoffs.classify(reason) == expected


def test_every_reason_the_app_escalates_with_is_known():
    import re
    from pathlib import Path

    source = "\n".join(p.read_text(encoding="utf-8") for p in Path(handoffs.__file__).parent.glob("*.py"))
    used = set(re.findall(r'_(?:escalation|handoff)\("([a-z_]+)"\)', source)) | {"server_error"}
    assert used <= set(handoffs.REASONS), used - set(handoffs.REASONS)


def test_handoff_is_created_with_the_escalated_reply_and_persists(stores, path, clock):
    store, records, _ = stores
    conversation_id = exchange(store, records)

    rows = table(path, "SELECT conversation_id, status, category, first_reason, last_reason, escalations, "
                       "first_message_id FROM handoffs")
    (reply_id,) = table(path, "SELECT max(id) FROM messages WHERE role = 'assistant'")[0]
    assert rows == [(conversation_id, "open", "no_answer", "insufficient_evidence", "insufficient_evidence", 1, reply_id)]

    _, reopened, _ = open_stores(path, clock)  # New instances: like a restarted process.
    assert reopened.open_handoff(conversation_id) == handoffs.Handoff(
        "open", "no_answer", "insufficient_evidence", "insufficient_evidence", 1, clock.now, clock.now)


def test_a_second_escalation_updates_the_open_handoff(stores, path, clock):
    store, records, _ = stores
    conversation_id = exchange(store, records)
    clock.now += 60
    exchange(store, records, AssistantReply(HANDOFF_MESSAGE, [], True, "llm_error"), conversation_id)

    assert len(table(path, "SELECT * FROM handoffs")) == 1
    handoff = records.open_handoff(conversation_id)
    assert (handoff.first_reason, handoff.last_reason, handoff.escalations) == ("insufficient_evidence", "llm_error", 2)
    assert handoff.category == "no_answer"  # From the first escalation.
    assert handoff.updated_at == handoff.created_at + 60


def test_the_database_itself_refuses_a_second_open_handoff(stores, path):
    store, records, _ = stores
    conversation_id = exchange(store, records)
    with sqlite3.connect(path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO handoffs (conversation_id, status, category, first_reason, last_reason, escalations, "
            "first_message_id, created_at, updated_at) VALUES (?, 'open', 'other', 'other', 'other', 1, 1, 0, 0)",
            (conversation_id,))


def test_the_database_refuses_an_unknown_category(stores, path):
    store, records, _ = stores
    conversation_id = exchange(store, records)
    with sqlite3.connect(path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE handoffs SET category = 'urgent' WHERE conversation_id = ?", (conversation_id,))


def test_concurrent_escalations_keep_one_open_handoff(stores, path, clock):
    store, records, _ = stores
    conversation_id = exchange(store, records)
    barrier = threading.Barrier(6)
    errors = []

    def escalate():
        local = ConversationStore(path, clock=clock)
        barrier.wait()
        try:
            exchange(local, HandoffStore(local), conversation_id=conversation_id)
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=escalate) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []
    assert table(path, "SELECT count(*), max(escalations) FROM handoffs") == [(1, 7)]


def test_answers_create_no_handoff_and_handoffs_belong_to_their_conversation(stores, path):
    store, records, _ = stores
    answered = exchange(store, records, ANSWER)
    escalated = exchange(store, records)
    assert records.open_handoff(answered) is None
    assert records.open_handoff(escalated) is not None
    assert records.open_handoff(cs.new_conversation_id()) is None and records.open_handoff("bad") is None


def test_handoff_failure_saves_nothing(stores, path):
    store, _, _ = stores

    def broken(connection, now):
        connection.execute("INSERT INTO no_such_table VALUES (1)")

    with pytest.raises(cs.StoreError):
        store.add_exchange(cs.new_conversation_id(), "q", ESCALATED, create=True, in_transaction=broken)
    assert table(path, "SELECT count(*) FROM messages") == [(0,)]


def test_handoffs_are_deleted_with_their_conversation(stores, path, clock):
    store, records, _ = stores
    exchange(store, records)
    clock.now += cs.RETENTION_SECONDS + 1
    store.purge_expired()
    assert table(path, "SELECT * FROM handoffs") == []


# --- Feedback ------------------------------------------------------------------------------


def test_feedback_is_stored_for_an_assistant_reply_and_persists(stores, path, clock):
    store, records, ratings = stores
    conversation_id = exchange(store, records, ANSWER)

    result = ratings.submit(conversation_id, 1, "positive")

    assert result == feedback.FeedbackResult(1, "positive", changed=True)
    _, _, reopened = open_stores(path, clock)
    assert reopened.ratings(conversation_id) == {1: "positive"}
    (reply_id,) = table(path, "SELECT max(id) FROM messages")[0]
    assert table(path, "SELECT message_id, conversation_id, rating FROM feedback") == [
        (reply_id, conversation_id, "positive")]


def test_duplicate_feedback_changes_nothing_and_a_new_rating_replaces_it(stores, path, clock):
    store, records, ratings = stores
    conversation_id = exchange(store, records, ANSWER)
    ratings.submit(conversation_id, 1, "positive")
    clock.now += 5
    assert ratings.submit(conversation_id, 1, "positive").changed is False
    assert table(path, "SELECT rating, created_at, updated_at FROM feedback") == [("positive", clock.now - 5, clock.now - 5)]
    assert ratings.submit(conversation_id, 1, "negative").changed is True
    assert table(path, "SELECT rating, updated_at FROM feedback") == [("negative", clock.now)]


@pytest.mark.parametrize("index", [0, 2, 3, 40, 10_000])  # 0 is the customer's message; 2+ don't exist.
def test_only_an_existing_assistant_reply_can_be_rated(stores, index):
    store, records, ratings = stores
    conversation_id = exchange(store, records, ANSWER)
    with pytest.raises(MessageNotFound):
        ratings.submit(conversation_id, index, "positive")


@pytest.mark.parametrize("index", [-1, True, "1", 1.0, None])
def test_malformed_message_indexes_are_rejected(stores, index):
    store, records, ratings = stores
    conversation_id = exchange(store, records, ANSWER)
    with pytest.raises(MessageNotFound):
        ratings.submit(conversation_id, index, "positive")


def test_a_reply_of_another_conversation_cannot_be_rated(stores, path):
    store, records, ratings = stores
    short = exchange(store, records, ANSWER)  # 2 messages
    long = exchange(store, records, ANSWER)
    exchange(store, records, ANSWER, long)  # 4 messages: index 3 exists only here
    with pytest.raises(MessageNotFound):
        ratings.submit(short, 3, "positive")
    for conversation_id in (cs.new_conversation_id(), "not-an-id"):
        with pytest.raises(ConversationNotFound):
            ratings.submit(conversation_id, 1, "positive")
    assert table(path, "SELECT * FROM feedback") == []


def test_feedback_on_an_expired_conversation_is_refused(stores, clock):
    store, records, ratings = stores
    conversation_id = exchange(store, records, ANSWER)
    clock.now += cs.RETENTION_SECONDS + 1
    with pytest.raises(ConversationNotFound):
        ratings.submit(conversation_id, 1, "positive")


def test_only_two_ratings_exist(stores):
    store, records, ratings = stores
    conversation_id = exchange(store, records, ANSWER)
    with pytest.raises(ValueError):
        ratings.submit(conversation_id, 1, "meh")


def test_concurrent_duplicate_feedback_stores_one_row(stores, path, clock):
    store, records, _ = stores
    conversation_id = exchange(store, records, ANSWER)
    barrier = threading.Barrier(6)
    results = []

    def rate():
        local = FeedbackStore(ConversationStore(path, clock=clock))
        barrier.wait()
        results.append(local.submit(conversation_id, 1, "negative"))

    threads = [threading.Thread(target=rate) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(r.changed for r in results) == [False] * 5 + [True]
    assert table(path, "SELECT count(*) FROM feedback") == [(1,)]


def test_ratings_map_to_message_positions(stores):
    store, records, ratings = stores
    conversation_id = exchange(store, records, ANSWER)
    exchange(store, records, ESCALATED, conversation_id)
    ratings.submit(conversation_id, 3, "negative")
    ratings.submit(conversation_id, 1, "positive")
    assert ratings.ratings(conversation_id) == {1: "positive", 3: "negative"}
    assert ratings.ratings(cs.new_conversation_id()) == {} and ratings.ratings("bad") == {}


# --- The API -------------------------------------------------------------------------------


http = TestClient(api.app)


class FakeAgent:
    def __init__(self):
        self.result = SupportAnswer("Use Forgot password.", [SOURCE], False, "answered")
        self.error = None

    def __call__(self, question, **kwargs):
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def fake_agent(monkeypatch):
    monkeypatch.setattr(api, "chat_limiter", RateLimiter(per_minute=100, per_day=1000))
    fake = FakeAgent()
    monkeypatch.setattr(agent, "answer_question", fake)
    return fake


def ask(question="How do I reset my password?", conversation_id=None):
    body = {"question": question} if conversation_id is None else {"question": question, "conversation_id": conversation_id}
    return http.post("/chat", json=body)


def rate(conversation_id, index, rating="positive", **extra):
    return http.post("/feedback", json={"conversation_id": conversation_id, "message_index": index, "rating": rating,
                                        **extra})


def test_the_handoff_message_claims_no_human_contact():
    lowered = HANDOFF_MESSAGE.lower()
    assert "recorded" in lowered
    for claim in ("follow up", "team member", "passed your question", "assigned", "contact"):
        assert claim not in lowered


def test_chat_escalation_opens_a_handoff_shown_on_the_conversation(fake_agent, conversation_db):
    fake_agent.result = SupportAnswer(HANDOFF_MESSAGE, [], True, "insufficient_evidence")
    body = ask().json()
    assert body["escalate"] is True and body["message_index"] == 1
    assert "handoff" not in body  # The /chat response shape is unchanged apart from message_index.

    restored = http.get(f"/conversations/{body['conversation_id']}").json()
    handoff = restored["handoff"]
    assert set(handoff) == {"status", "category", "created_at", "updated_at"}  # No internal reasons or row IDs.
    assert (handoff["status"], handoff["category"]) == ("open", "no_answer")

    fake_agent.result = SupportAnswer(HANDOFF_MESSAGE, [], True, "llm_error")
    ask("again", body["conversation_id"])
    assert table(conversation_db.path, "SELECT count(*), max(escalations) FROM handoffs") == [(1, 2)]


def test_answers_open_no_handoff(fake_agent, conversation_db):
    body = ask().json()
    assert http.get(f"/conversations/{body['conversation_id']}").json()["handoff"] is None
    assert table(conversation_db.path, "SELECT * FROM handoffs") == []


def test_server_error_and_unknown_reasons_are_recorded_safely(fake_agent, conversation_db):
    fake_agent.error = RuntimeError("boom with a secret")
    crashed = ask()
    assert crashed.status_code == 500 and crashed.json()["message_index"] == 1
    fake_agent.error = None
    fake_agent.result = SupportAnswer(HANDOFF_MESSAGE, [], True, "SELECT * FROM users")
    odd = ask().json()
    rows = dict(table(conversation_db.path, "SELECT conversation_id, first_reason || '/' || category FROM handoffs"))
    assert rows[crashed.json()["conversation_id"]] == "server_error/service_problem"
    assert rows[odd["conversation_id"]] == "other/other"


def test_handoff_storage_failure_saves_nothing(fake_agent, conversation_db, monkeypatch):
    fake_agent.result = SupportAnswer(HANDOFF_MESSAGE, [], True, "insufficient_evidence")

    def broken(conversation_id, reason):
        def record(connection, now):
            connection.execute("INSERT INTO no_such_table VALUES (1)")
        return record

    monkeypatch.setattr(api.handoff_store, "recorder", broken)
    assert ask().status_code == 503
    assert table(conversation_db.path, "SELECT count(*) FROM messages") == [(0,)]


def test_chat_and_conversations_need_the_new_stores(fake_agent, monkeypatch):
    conversation_id = ask().json()["conversation_id"]
    monkeypatch.setattr(api, "handoff_store", None)
    assert ask().status_code == 503
    assert http.get(f"/conversations/{conversation_id}").status_code == 503
    monkeypatch.setattr(api, "feedback_store", None)
    assert rate(conversation_id, 1).json() == {"detail": api.NOT_READY_MESSAGE}


def test_feedback_endpoint_records_and_restores(fake_agent, conversation_db):
    body = ask().json()
    conversation_id, index = body["conversation_id"], body["message_index"]

    first = rate(conversation_id, index)
    assert first.status_code == 200 and first.json() == {"message_index": 1, "rating": "positive", "changed": True}
    assert rate(conversation_id, index).json()["changed"] is False
    assert rate(conversation_id, index, "negative").json() == {"message_index": 1, "rating": "negative", "changed": True}

    messages = http.get(f"/conversations/{conversation_id}").json()["messages"]
    assert [(m["message_index"], m["feedback"]) for m in messages] == [(0, None), (1, "negative")]
    assert table(conversation_db.path, "SELECT count(*) FROM feedback") == [(1,)]


def test_feedback_endpoint_refuses_other_messages_and_conversations(fake_agent):
    mine = ask().json()["conversation_id"]
    theirs = ask().json()["conversation_id"]
    ask("follow-up", theirs)

    assert rate(mine, 0).status_code == 404  # A customer message.
    assert rate(mine, 3).json() == {"detail": api.MESSAGE_NOT_FOUND_MESSAGE}  # Exists only in theirs.
    unknown = rate(cs.new_conversation_id(), 1)
    assert unknown.status_code == 404 and unknown.json() == {"detail": api.CONVERSATION_NOT_FOUND_MESSAGE}


@pytest.mark.parametrize("body", [
    {"conversation_id": None, "message_index": 1, "rating": "positive"},  # None: left out
    {"conversation_id": "x", "message_index": 1, "rating": "positive"},
    {"message_index": -1, "rating": "positive"},
    {"message_index": "1", "rating": "positive"},
    {"message_index": True, "rating": "positive"},
    {"message_index": 1.5, "rating": "positive"},
    {"message_index": 10_001, "rating": "positive"},
    {"message_index": 1, "rating": "meh"},
    {"message_index": 1, "rating": "positive", "comment": "my phone is 555-0100"},
    {"message_index": 1, "rating": "positive", "email": "someone@example.com"},
])
def test_feedback_endpoint_rejects_malformed_requests_and_extra_data(fake_agent, conversation_db, body):
    conversation_id = ask().json()["conversation_id"]
    if "conversation_id" not in body:
        body = {"conversation_id": conversation_id, **body}
    elif body["conversation_id"] is None:
        body = {k: v for k, v in body.items() if k != "conversation_id"}
    assert http.post("/feedback", json=body).status_code == 422
    assert table(conversation_db.path, "SELECT count(*) FROM feedback") == [(0,)]


def test_feedback_storage_failure_is_a_503(fake_agent, monkeypatch):
    conversation_id = ask().json()["conversation_id"]

    def broken(*args):
        raise cs.StoreError("Conversation database error: OperationalError")

    monkeypatch.setattr(api.feedback_store, "submit", broken)
    assert rate(conversation_id, 1).json() == {"detail": api.STORAGE_UNAVAILABLE_MESSAGE}


def test_feedback_logs_never_contain_the_conversation_id(fake_agent, caplog):
    conversation_id = ask().json()["conversation_id"]
    with caplog.at_level(logging.INFO, logger="support_agent"):
        rate(conversation_id, 1)
        rate(conversation_id, 0)
    assert "feedback outcome=recorded rating=positive" in caplog.text
    assert "outcome=message_not_found" in caplog.text
    assert conversation_id not in caplog.text


def test_cancellation_offers_are_unchanged(monkeypatch, conversation_db):
    monkeypatch.setattr(api, "chat_limiter", RateLimiter(per_minute=100, per_day=1000))

    def offer(question, *, trace=None, **kwargs):
        trace["cancellable_order_id"] = "ORD-1001"
        return SupportAnswer(cancellation_message("ORD-1001"), [], False, CANCELLATION_AVAILABLE)

    monkeypatch.setattr(agent, "answer_question", offer)
    body = ask("Please cancel ORD-1001").json()
    assert body["pending_cancellation"]["order_id"] == "ORD-1001" and body["message_index"] == 1
    assert table(conversation_db.path, "SELECT count(*) FROM handoffs") == [(0,)]
    restored = http.get(f"/conversations/{body['conversation_id']}").json()
    assert restored["messages"][1]["cancellation"]["status"] == "pending" and restored["handoff"] is None
