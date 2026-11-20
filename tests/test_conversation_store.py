"""Offline tests for the SQLite conversation store (no network, temporary files only)."""

import sqlite3
import threading

import pytest

from support_agent import conversation_store as cs
from support_agent.conversation_store import (
    AssistantReply,
    ConversationFull,
    ConversationNotFound,
    ConversationStore,
    StoreError,
)

ANSWER = AssistantReply(
    answer="Use Forgot password.",
    sources=[{"id": "KB-001", "title": "Reset your password", "url": "https://support.example.com/password-reset"}],
    escalate=False,
    reason="answered",
)
HANDOFF = AssistantReply(answer="handed off", sources=[], escalate=True, reason="insufficient_evidence")
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
def store(tmp_path, clock):
    store = ConversationStore(tmp_path / "db" / "conversations.sqlite3", clock=clock)
    store.initialize()
    return store


def start(store, question="How do I reset my password?", reply=ANSWER, **kwargs):
    conversation_id = cs.new_conversation_id()
    store.add_exchange(conversation_id, question, reply, create=True, **kwargs)
    return conversation_id


# --- IDs and settings --------------------------------------------------------------


def test_new_ids_are_random_128_bit_hex():
    ids = {cs.new_conversation_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(cs.is_valid_id(i) for i in ids)


@pytest.mark.parametrize("value", [
    "", "abc", "A" * 32, "g" * 32, "0" * 31, "0" * 33, "0" * 32 + "\n", " " + "0" * 31, None, 123,
    "../" + "0" * 29, "0" * 16 + "' OR 1=1 --",
])
def test_invalid_ids_are_rejected(value):
    assert not cs.is_valid_id(value)


def test_log_hash_is_short_stable_and_not_the_id():
    conversation_id = cs.new_conversation_id()
    label = cs.log_hash(conversation_id)
    assert label == cs.log_hash(conversation_id) and len(label) == 12
    assert label not in conversation_id


def test_db_path_setting(tmp_path):
    assert cs.db_path_from_env({}) == cs.DEFAULT_DB_PATH
    assert cs.db_path_from_env({cs.DB_PATH_ENV: "  "}) == cs.DEFAULT_DB_PATH
    assert cs.db_path_from_env({cs.DB_PATH_ENV: str(tmp_path / "x.db")}) == tmp_path / "x.db"
    assert not (tmp_path / "x.db").exists()  # checking creates nothing


@pytest.mark.parametrize("value", [":memory:", "file:x?mode=memory"])
def test_db_path_rejects_in_memory_databases(value):
    with pytest.raises(ValueError, match=f"^{cs.DB_PATH_ENV} must be a file path$"):
        cs.db_path_from_env({cs.DB_PATH_ENV: value})


def test_db_path_rejects_a_folder_without_repeating_it(tmp_path):
    with pytest.raises(ValueError) as raised:
        cs.db_path_from_env({cs.DB_PATH_ENV: str(tmp_path)})
    assert str(tmp_path) not in str(raised.value)


def test_default_path_is_git_ignored():
    from pathlib import Path

    gitignore = (Path(cs.__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")
    assert "var/" in gitignore.splitlines()


def test_message_limit_must_be_even():
    with pytest.raises(ValueError):
        ConversationStore("x.db", max_messages=3)


# --- Setup -------------------------------------------------------------------------


def test_initialize_creates_folder_tables_and_wal(tmp_path, clock):
    path = tmp_path / "a" / "b" / "c.sqlite3"
    ConversationStore(path, clock=clock).initialize()

    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert {"conversations", "messages"} <= tables
    assert mode == "wal"


def test_initialize_is_idempotent_and_keeps_data(tmp_path, clock):
    path = tmp_path / "c.sqlite3"
    first = ConversationStore(path, clock=clock)
    first.initialize()
    conversation_id = start(first)

    second = ConversationStore(path, clock=clock)
    second.initialize()

    assert len(second.messages(conversation_id)) == 2


def test_initialize_fails_with_store_error_for_an_unusable_path(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder", encoding="utf-8")
    with pytest.raises(StoreError) as raised:
        ConversationStore(blocker / "db.sqlite3").initialize()
    assert str(blocker) not in str(raised.value)


def test_a_corrupt_file_is_a_store_error(tmp_path):
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"this is not a database" * 100)
    with pytest.raises(StoreError, match="DatabaseError"):
        ConversationStore(path).initialize()


def test_ping(store, tmp_path):
    store.ping()
    with pytest.raises(StoreError):
        ConversationStore(tmp_path / "missing-folder" / "x" / "db.sqlite3").ping()


# --- Saving and reading -------------------------------------------------------------


def test_exchange_round_trip(store, clock):
    conversation_id = start(store, retrieval_query="reset password")

    customer, assistant = store.messages(conversation_id)

    assert customer == cs.Message(role="customer", content="How do I reset my password?", created_at=clock.now)
    assert assistant == cs.Message(
        role="assistant", content=ANSWER.answer, created_at=clock.now,
        sources=ANSWER.sources, escalate=False, reason="answered",
    )


def test_text_is_stored_exactly(store):
    question = "  <script>alert('x')</script> \"quotes\" ünïcødé 🙂\nsecond line  "
    conversation_id = start(store, question=question)
    assert store.messages(conversation_id)[0].content == question


def test_messages_keep_their_order(store, clock):
    conversation_id = start(store, question="first")
    clock.now += 1
    store.add_exchange(conversation_id, "second", HANDOFF)

    messages = store.messages(conversation_id)

    assert [(m.role, m.content) for m in messages] == [
        ("customer", "first"), ("assistant", ANSWER.answer), ("customer", "second"), ("assistant", "handed off"),
    ]
    assert messages[3].escalate is True and messages[3].sources == []


def test_rewritten_query_is_stored_separately_from_the_customer_text(store):
    conversation_id = start(store, question="what if it's 35 days?", retrieval_query="return after 35 days")

    assert store.messages(conversation_id)[0].content == "what if it's 35 days?"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT retrieval_query FROM messages WHERE role='customer'").fetchone() == (
            "return after 35 days",
        )


def test_conversations_are_isolated(store):
    a = start(store, question="question A")
    b = start(store, question="question B")
    store.add_exchange(a, "follow-up A", ANSWER)

    assert [m.content for m in store.messages(a) if m.role == "customer"] == ["question A", "follow-up A"]
    assert [m.content for m in store.messages(b) if m.role == "customer"] == ["question B"]


@pytest.mark.parametrize("conversation_id", ["0" * 32, "not-an-id", "' OR 1=1 --"])
def test_unknown_or_invalid_ids_are_not_found(store, conversation_id):
    start(store)
    with pytest.raises(ConversationNotFound):
        store.messages(conversation_id)
    with pytest.raises(ConversationNotFound):
        store.add_exchange(conversation_id, "hi", ANSWER)


def test_create_with_an_existing_id_fails_and_changes_nothing(store):
    conversation_id = start(store)
    with pytest.raises(StoreError, match="IntegrityError"):
        store.add_exchange(conversation_id, "again", ANSWER, create=True)
    assert len(store.messages(conversation_id)) == 2


def test_a_failed_save_stores_neither_message(store, monkeypatch):
    """The customer message and the reply are saved together or not at all."""
    conversation_id = start(store)
    bad = AssistantReply(answer=None, sources=[], escalate=True, reason="x")  # NOT NULL violation

    with pytest.raises(StoreError):
        store.add_exchange(conversation_id, "lost question", bad)

    assert [m.content for m in store.messages(conversation_id)] == ["How do I reset my password?", ANSWER.answer]


def test_store_errors_never_contain_message_text(store):
    conversation_id = start(store)
    with pytest.raises(StoreError) as raised:
        store.add_exchange(conversation_id, "SECRET-QUESTION", AssistantReply(None, [], True, "x"))
    assert "SECRET" not in str(raised.value)


# --- Message limit ------------------------------------------------------------------


def test_full_conversation_refuses_more_messages_without_dropping_any(tmp_path, clock):
    store = ConversationStore(tmp_path / "db.sqlite3", clock=clock, max_messages=4)
    store.initialize()
    conversation_id = start(store, question="q1")
    assert store.add_exchange(conversation_id, "q2", ANSWER) == 4

    with pytest.raises(ConversationFull):
        store.add_exchange(conversation_id, "q3", ANSWER)

    assert [m.content for m in store.messages(conversation_id) if m.role == "customer"] == ["q1", "q2"]
    assert store.has_room(2) and not store.has_room(4)


def test_default_limit_is_40_messages(store):
    conversation_id = start(store)
    for n in range(19):
        store.add_exchange(conversation_id, f"q{n}", ANSWER)
    assert len(store.messages(conversation_id)) == 40
    with pytest.raises(ConversationFull):
        store.add_exchange(conversation_id, "one too many", ANSWER)


def test_concurrent_writers_never_exceed_the_limit(tmp_path):
    """Many threads add to one conversation at once: each write is checked
    inside its own transaction, so the limit holds exactly."""
    store = ConversationStore(tmp_path / "db.sqlite3", max_messages=10)
    store.initialize()
    conversation_id = start(store)
    results = []

    def writer(n):
        try:
            store.add_exchange(conversation_id, f"q{n}", ANSWER)
            results.append("saved")
        except ConversationFull:
            results.append("full")

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert results.count("saved") == 4 and results.count("full") == 8
    messages = store.messages(conversation_id)
    assert len(messages) == 10
    # Every customer message is immediately followed by its own reply.
    assert [m.role for m in messages] == ["customer", "assistant"] * 5


def test_concurrent_conversations_in_threads(store):
    errors = []

    def worker():
        try:
            conversation_id = start(store, question="mine")
            for _ in range(3):
                store.add_exchange(conversation_id, "mine", ANSWER)
            assert len(store.messages(conversation_id)) == 8
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert errors == []


# --- Retention ----------------------------------------------------------------------


def test_conversation_expires_30_days_after_its_last_message(store, clock):
    conversation_id = start(store)
    clock.now += 29 * DAY
    store.add_exchange(conversation_id, "still here", ANSWER)  # activity resets the clock
    clock.now += 29 * DAY
    assert len(store.messages(conversation_id)) == 4

    clock.now += 1 * DAY + 1
    with pytest.raises(ConversationNotFound):
        store.messages(conversation_id)  # expired: missing even before cleanup
    with pytest.raises(ConversationNotFound):
        store.add_exchange(conversation_id, "too late", ANSWER)


def test_purge_deletes_only_expired_conversations_and_their_messages(store, clock):
    old = start(store)
    clock.now += 20 * DAY
    active = start(store)
    clock.now += 10 * DAY + 1  # old is 30 days + 1 s idle; active 10 days

    assert store.purge_expired() == 1

    assert len(store.messages(active)) == 2
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM messages WHERE conversation_id=?", (old,)).fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM conversations").fetchone() == (1,)


def test_purge_does_not_touch_a_conversation_used_during_the_retention_window(store, clock):
    conversation_id = start(store)
    clock.now += 30 * DAY  # exactly at the limit: still live
    store.add_exchange(conversation_id, "just in time", ANSWER)
    store.purge_expired()
    assert len(store.messages(conversation_id)) == 4


def test_purge_if_due_runs_at_most_once_an_hour(store, clock):
    assert store.purge_if_due() is None  # initialize() just purged
    clock.now += cs.PURGE_INTERVAL_SECONDS
    assert store.purge_if_due() == 0
    assert store.purge_if_due() is None


def test_initialize_purges_expired_conversations(tmp_path, clock):
    path = tmp_path / "db.sqlite3"
    store = ConversationStore(path, clock=clock)
    store.initialize()
    conversation_id = start(store)
    clock.now += 31 * DAY

    ConversationStore(path, clock=clock).initialize()

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM conversations WHERE id=?", (conversation_id,)).fetchone() == (0,)
