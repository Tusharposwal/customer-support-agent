"""Offline tests for persistent cancellations with explicit confirmation
(cancellations.py, POST /cancellations/confirm, and how /chat and the tools
use them). Temporary SQLite files only; every Cohere client is a fake and no
request is ever sent."""

import json
import logging
import socket
import sqlite3
import threading
from dataclasses import replace
from datetime import date

import cohere
import pytest
from fastapi.testclient import TestClient

from support_agent import agent, api, cancellations, tools
from support_agent import conversation_store as cs
from support_agent.answer import CANCELLATION_AVAILABLE, SupportAnswer, cancellation_message
from support_agent.cancellations import (
    ActionExpired,
    ActionNotFound,
    CancellationStore,
    OrderNotCancellable,
)
from support_agent.conversation_store import AssistantReply, ConversationFull, ConversationStore
from support_agent.demo_orders import DEMO_ORDERS_PATH, OrderBook, load_orders
from support_agent.kb import load_articles
from support_agent.query_rewrite import NOT_NEEDED, Rewrite
from support_agent.rate_limit import RateLimiter
from support_agent.rerank import RerankConfig

BOOK = load_orders()
MAYA = "maya.fernandes@example.com"  # Owner of ORD-1001 (processing) in the demo data.
ORDER = "ORD-1001"
CANCEL_Q = f"Please cancel {ORDER}, my email is {MAYA}"
TTL = 15 * 60
OFFER = AssistantReply(answer=cancellation_message(ORDER), sources=[], escalate=False, reason=CANCELLATION_AVAILABLE)
TODAY = date(2026, 11, 20)
KB = {a["id"]: a for a in load_articles()}


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
    """A ConversationStore and CancellationStore on `path`: every instance
    opens its own connections, like separate processes or threads would."""
    store = ConversationStore(path, clock=clock)
    store.initialize()
    actions = CancellationStore(store, ttl_seconds=TTL)
    actions.initialize()
    return store, actions


@pytest.fixture
def stores(path, clock):
    return open_stores(path, clock)


def propose(store, actions, order_id=ORDER, conversation_id=None):
    """Save a cancellation offer with its pending action, as /chat does."""
    create = conversation_id is None
    conversation_id = conversation_id or cs.new_conversation_id()
    record, created = actions.proposer(conversation_id, order_id)
    store.add_exchange(conversation_id, CANCEL_Q, OFFER, create=create, in_transaction=record)
    return conversation_id, created()


def processing(order_id):
    return "processing"


def table(path, sql, *args):
    with sqlite3.connect(path) as connection:
        return connection.execute(sql, args).fetchall()


# --- Creating an action -----------------------------------------------------------------


def test_action_is_saved_with_the_reply_and_has_a_random_id_and_expiry(stores, path, clock):
    store, actions = stores
    conversation_id, pending = propose(store, actions)

    assert cs.is_valid_id(pending.action_id)  # 32 hex characters (128 bits)
    assert pending.order_id == ORDER and pending.expires_at == clock.now + TTL
    rows = table(path, "SELECT conversation_id, order_id, expires_at, confirmed_at FROM cancellation_actions")
    assert rows == [(conversation_id, ORDER, clock.now + TTL, None)]
    assert len({propose(store, actions)[1].action_id for _ in range(20)}) == 20


def test_action_id_is_stored_only_as_a_hash(stores, path):
    store, actions = stores
    _, pending = propose(store, actions)
    with sqlite3.connect(path) as connection:
        dump = "\n".join(connection.iterdump())
    assert pending.action_id not in dump
    assert cancellations._hash(pending.action_id) in dump


def test_no_action_without_the_reply_and_no_reply_without_the_action(stores, path, clock):
    store, actions = stores
    full = ConversationStore(path, clock=clock, max_messages=2)
    conversation_id, _ = propose(full, CancellationStore(full))
    record, _created = actions.proposer(conversation_id, ORDER)
    with pytest.raises(ConversationFull):
        full.add_exchange(conversation_id, "again", OFFER, in_transaction=record)
    assert len(table(path, "SELECT * FROM cancellation_actions")) == 1  # Only the first.

    def failing(connection, now):
        raise RuntimeError("action could not be recorded")

    with pytest.raises(RuntimeError):
        store.add_exchange(cs.new_conversation_id(), CANCEL_Q, OFFER, create=True, in_transaction=failing)
    assert len(table(path, "SELECT * FROM conversations")) == 1  # Rolled back entirely.


# --- Confirming --------------------------------------------------------------------------


def test_successful_confirmation_cancels_the_order(stores, path, clock):
    store, actions = stores
    conversation_id, pending = propose(store, actions)
    assert actions.is_cancelled(ORDER) is False

    clock.now += 60
    confirmation = actions.confirm(conversation_id, pending.action_id, processing)

    assert confirmation == cancellations.Confirmation(ORDER, clock.now, already_confirmed=False)
    assert actions.is_cancelled(ORDER) is True
    assert actions.is_cancelled("ORD-1002") is False
    assert table(path, "SELECT order_id, cancelled_at FROM order_cancellations") == [(ORDER, clock.now)]
    assert table(path, "SELECT confirmed_at FROM cancellation_actions") == [(clock.now,)]


@pytest.mark.parametrize("elapsed", [TTL, TTL + 1, 10 * TTL])
def test_expired_action_cancels_nothing(stores, clock, elapsed):
    store, actions = stores
    conversation_id, pending = propose(store, actions)
    clock.now += elapsed
    with pytest.raises(ActionExpired):
        actions.confirm(conversation_id, pending.action_id, processing)
    assert actions.is_cancelled(ORDER) is False


def test_confirmation_just_before_expiry_works(stores, clock):
    store, actions = stores
    conversation_id, pending = propose(store, actions)
    clock.now += TTL - 1
    assert actions.confirm(conversation_id, pending.action_id, processing).already_confirmed is False


@pytest.mark.parametrize("action_id", [
    cs.new_conversation_id(),  # Well-formed but unknown.
    "", "yes", "0" * 31, "A" * 32, "g" * 32, None, 12345,
])
def test_invalid_action_ids_are_not_found(stores, action_id):
    store, actions = stores
    conversation_id, _ = propose(store, actions)
    with pytest.raises(ActionNotFound):
        actions.confirm(conversation_id, action_id, processing)
    assert actions.is_cancelled(ORDER) is False


def test_action_from_another_conversation_is_not_found(stores):
    store, actions = stores
    owner, pending = propose(store, actions)
    other, _ = propose(store, actions, order_id="ORD-1002")
    for conversation_id in (other, cs.new_conversation_id(), "not-an-id"):
        with pytest.raises(ActionNotFound):
            actions.confirm(conversation_id, pending.action_id, processing)
    assert actions.is_cancelled(ORDER) is False
    # The rightful conversation can still confirm it.
    assert actions.confirm(owner, pending.action_id, processing).order_id == ORDER


def test_action_of_an_expired_conversation_is_not_found(stores, clock):
    store, actions = stores
    conversation_id, pending = propose(store, actions)
    clock.now += cs.RETENTION_SECONDS + 1  # Conversation expired (the action long before).
    with pytest.raises(ActionNotFound):
        actions.confirm(conversation_id, pending.action_id, processing)


@pytest.mark.parametrize("status", ["shipped", "delivered", "cancelled", None])
def test_order_status_changed_before_confirmation(stores, status):
    store, actions = stores
    conversation_id, pending = propose(store, actions)
    with pytest.raises(OrderNotCancellable):
        actions.confirm(conversation_id, pending.action_id, lambda order_id: status)
    assert actions.is_cancelled(ORDER) is False
    # Nothing was marked confirmed: if the order could be cancelled again, it can be confirmed.
    assert actions.confirm(conversation_id, pending.action_id, processing).already_confirmed is False


def test_status_is_checked_for_the_actions_own_order(stores):
    store, actions = stores
    conversation_id, pending = propose(store, actions, order_id="ORD-1002")
    asked = []
    actions.confirm(conversation_id, pending.action_id, lambda order_id: asked.append(order_id) or "processing")
    assert asked == ["ORD-1002"]


def test_duplicate_confirmation_is_idempotent(stores, path, clock):
    store, actions = stores
    conversation_id, pending = propose(store, actions)
    first = actions.confirm(conversation_id, pending.action_id, processing)

    clock.now += 2 * TTL  # Even after the expiry and a status change, a repeat just reports the result.
    again = actions.confirm(conversation_id, pending.action_id, lambda order_id: "cancelled")

    assert again == replace(first, already_confirmed=True)
    assert len(table(path, "SELECT * FROM order_cancellations")) == 1


def test_a_second_action_for_a_cancelled_order_cancels_nothing_more(stores, path):
    store, actions = stores
    conversation_id, first = propose(store, actions)
    _, second = propose(store, actions, conversation_id=conversation_id)
    actions.confirm(conversation_id, first.action_id, processing)
    with pytest.raises(OrderNotCancellable):
        actions.confirm(conversation_id, second.action_id, processing)
    assert table(path, "SELECT action_hash FROM order_cancellations") == [(cancellations._hash(first.action_id),)]


def run_concurrently(path, clock, attempts):
    """Run each attempt(actions) in its own thread with its own stores
    (separate connections), all released at once. Returns the outcomes."""
    barrier = threading.Barrier(len(attempts))
    outcomes = [None] * len(attempts)

    def worker(index, attempt):
        actions = CancellationStore(ConversationStore(path, clock=clock), ttl_seconds=TTL)  # Already initialised.
        barrier.wait()
        try:
            outcomes[index] = attempt(actions)
        except Exception as error:
            outcomes[index] = error

    threads = [threading.Thread(target=worker, args=(i, a)) for i, a in enumerate(attempts)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return outcomes


def test_concurrent_confirmations_of_one_action_cancel_exactly_once(stores, path, clock):
    store, actions = stores
    conversation_id, pending = propose(store, actions)

    outcomes = run_concurrently(
        path, clock, [lambda a: a.confirm(conversation_id, pending.action_id, processing)] * 8
    )

    assert all(isinstance(o, cancellations.Confirmation) for o in outcomes), outcomes
    assert sorted(o.already_confirmed for o in outcomes) == [False] + [True] * 7
    assert len({o.cancelled_at for o in outcomes}) == 1
    assert len(table(path, "SELECT * FROM order_cancellations")) == 1


def test_concurrent_confirmations_of_different_actions_for_one_order(stores, path, clock):
    store, actions = stores
    pairs = [propose(store, actions) for _ in range(6)]  # Six conversations, same order.

    outcomes = run_concurrently(
        path, clock, [lambda a, c=c, p=p: a.confirm(c, p.action_id, processing) for c, p in pairs]
    )

    assert sum(isinstance(o, cancellations.Confirmation) for o in outcomes) == 1, outcomes
    assert sum(isinstance(o, OrderNotCancellable) for o in outcomes) == 5
    assert len(table(path, "SELECT * FROM order_cancellations")) == 1


def test_actions_and_cancellations_persist_across_connections_and_instances(path, clock):
    store, actions = open_stores(path, clock)
    conversation_id, pending = propose(store, actions)
    del store, actions

    _, confirming = open_stores(path, clock)  # Like a restarted process: new instances.
    assert confirming.confirm(conversation_id, pending.action_id, processing).already_confirmed is False

    _, reading = open_stores(path, clock)
    assert reading.is_cancelled(ORDER) is True
    assert reading.confirm(conversation_id, pending.action_id, processing).already_confirmed is True


def test_purging_a_conversation_removes_its_actions_but_keeps_the_cancellation(stores, path, clock):
    store, actions = stores
    confirmed_in, done = propose(store, actions)
    actions.confirm(confirmed_in, done.action_id, processing)
    propose(store, actions, order_id="ORD-1002")
    clock.now += cs.RETENTION_SECONDS + 1
    assert store.purge_expired() == 2
    assert table(path, "SELECT * FROM cancellation_actions") == []
    assert actions.is_cancelled(ORDER) is True


def test_initialize_is_repeatable(stores):
    _, actions = stores
    actions.initialize()


def test_ttl_must_be_positive(stores):
    with pytest.raises(ValueError):
        CancellationStore(stores[0], ttl_seconds=0)


def test_database_failure_is_a_store_error(tmp_path, clock):
    store = ConversationStore(tmp_path / "missing" / "db.sqlite3", clock=clock)  # Never initialised.
    actions = CancellationStore(store)
    with pytest.raises(cs.StoreError):
        actions.is_cancelled(ORDER)
    with pytest.raises(cs.StoreError):
        actions.confirm(cs.new_conversation_id(), cs.new_conversation_id(), processing)


# --- Tools read recorded cancellations ----------------------------------------------------


def tool(name, is_cancelled, order_id=ORDER):
    context = tools.ToolContext.for_question(BOOK, CANCEL_Q, None, TODAY, is_cancelled=is_cancelled)
    arguments = json.dumps({"order_id": order_id, "email": MAYA})
    return tools.execute(tools.ToolCall("c", name, arguments), context)


def test_tools_show_a_confirmed_cancellation():
    assert tool("get_order_status", lambda o: False).data["order_status"] == "processing"
    assert tool("get_order_status", lambda o: o == ORDER).data["order_status"] == "cancelled"
    assert tool("request_order_cancellation", lambda o: False).status == tools.CONFIRMATION_REQUIRED
    assert tool("request_order_cancellation", lambda o: True).status == tools.NOT_CANCELLABLE
    refund = tool("check_refund_eligibility", lambda o: True).data
    assert refund["order_status"] == "cancelled" and refund["items"][0]["decision"] == "not_applicable"


def test_the_order_file_is_never_changed():
    before = DEMO_ORDERS_PATH.read_bytes()
    tool("get_order_status", lambda o: True)
    assert DEMO_ORDERS_PATH.read_bytes() == before


def test_a_failing_cancellation_lookup_is_a_tool_error_not_a_guess():
    def broken(order_id):
        raise cs.StoreError("Conversation database error: OperationalError")

    assert tool("request_order_cancellation", broken).status == tools.ERROR


def test_default_context_has_no_cancellations():
    context = tools.ToolContext.for_question(BOOK, CANCEL_Q, None, TODAY)
    assert context.is_cancelled(ORDER) is False


# --- The API ------------------------------------------------------------------------------


http = TestClient(api.app)


@pytest.fixture
def limiter(monkeypatch):
    monkeypatch.setattr(api, "chat_limiter", RateLimiter(per_minute=100, per_day=1000))


@pytest.fixture
def offer_agent(monkeypatch, limiter):
    """A fake agent that always gives the server-written cancellation offer,
    as the real one does after the tool said the order can be cancelled."""
    calls = []

    def fake(question, *, history=None, trace=None, is_cancelled=None, **kwargs):
        calls.append({"question": question, "history": history, "is_cancelled": is_cancelled})
        trace["cancellable_order_id"] = ORDER
        return SupportAnswer(cancellation_message(ORDER), [], False, CANCELLATION_AVAILABLE)

    monkeypatch.setattr(agent, "answer_question", fake)
    return calls


def confirm(conversation_id, action_id):
    return http.post("/cancellations/confirm", json={"conversation_id": conversation_id, "action_id": action_id})


def test_chat_returns_the_pending_action_only_to_the_browser(offer_agent, conversation_db):
    body = http.post("/chat", json={"question": CANCEL_Q}).json()

    pending = body["pending_cancellation"]
    assert set(pending) == {"action_id", "order_id", "expires_at"}
    assert cs.is_valid_id(pending["action_id"]) and pending["order_id"] == ORDER
    assert body["answer"] == cancellation_message(ORDER) and body["reason"] == CANCELLATION_AVAILABLE

    # Not in the stored messages or the conversation view (which feed later prompts).
    conversation = http.get(f"/conversations/{body['conversation_id']}").json()
    assert pending["action_id"] not in json.dumps(conversation)
    with sqlite3.connect(conversation_db.path) as connection:
        assert pending["action_id"] not in "\n".join(connection.iterdump())

    # The next question's history (what the rewriter and model can see) doesn't contain it.
    http.post("/chat", json={"question": "yes", "conversation_id": body["conversation_id"]})
    assert pending["action_id"] not in json.dumps(offer_agent[-1]["history"])
    assert offer_agent[-1]["is_cancelled"] == api.cancellation_store.is_cancelled


def test_chat_creates_no_action_for_other_replies(monkeypatch, limiter, conversation_db):
    def fake(question, *, trace=None, **kwargs):
        trace["cancellable_order_id"] = ORDER  # Ignored: the reply is not the cancellation offer.
        return SupportAnswer("Orders can be cancelled while processing.", [
            {"id": "KB-024", "title": KB["KB-024"]["title"], "url": KB["KB-024"]["url"]}], False, "answered")

    monkeypatch.setattr(agent, "answer_question", fake)
    body = http.post("/chat", json={"question": CANCEL_Q}).json()
    assert "pending_cancellation" not in body
    assert table(conversation_db.path, "SELECT * FROM cancellation_actions") == []


def test_confirm_endpoint_success_and_repeat(offer_agent):
    body = http.post("/chat", json={"question": CANCEL_Q}).json()
    action_id = body["pending_cancellation"]["action_id"]

    first = confirm(body["conversation_id"], action_id)
    assert first.status_code == 200
    assert first.json()["status"] == "cancelled" and first.json()["order_id"] == ORDER
    assert first.json()["already_confirmed"] is False
    assert first.json()["message"] == f"Order {ORDER} has been cancelled."

    second = confirm(body["conversation_id"], action_id)
    assert second.status_code == 200 and second.json() == {**first.json(), "already_confirmed": True}
    assert api.cancellation_store.is_cancelled(ORDER)


def test_confirm_endpoint_failures(offer_agent, conversation_db, monkeypatch):
    first = http.post("/chat", json={"question": CANCEL_Q}).json()
    other = http.post("/chat", json={"question": CANCEL_Q}).json()
    action_id = first["pending_cancellation"]["action_id"]

    unknown = confirm(first["conversation_id"], cs.new_conversation_id())
    wrong_conversation = confirm(other["conversation_id"], action_id)
    assert unknown.status_code == wrong_conversation.status_code == 404
    assert unknown.json() == wrong_conversation.json() == {"detail": api.CANCELLATION_NOT_FOUND_MESSAGE}

    for bad in ({"conversation_id": first["conversation_id"]},
                {"conversation_id": first["conversation_id"], "action_id": "yes"},
                {"conversation_id": "x", "action_id": action_id}):
        assert http.post("/cancellations/confirm", json=bad).status_code == 422

    shipped = OrderBook([replace(BOOK.find(ORDER, MAYA), status="shipped")])
    monkeypatch.setattr(agent, "_default_order_book", lambda: shipped)
    changed = confirm(first["conversation_id"], action_id)
    assert changed.status_code == 409 and changed.json() == {"detail": api.ORDER_NOT_CANCELLABLE_MESSAGE}

    monkeypatch.setattr(conversation_db, "clock", lambda: 10**10)  # Far later: action expired (conversation not).
    monkeypatch.setattr(conversation_db, "retention_seconds", 10**11)
    expired = confirm(other["conversation_id"], other["pending_cancellation"]["action_id"])
    assert expired.status_code == 410 and expired.json() == {"detail": api.CANCELLATION_EXPIRED_MESSAGE}
    assert not api.cancellation_store.is_cancelled(ORDER)


def test_confirm_endpoint_unavailable(offer_agent, monkeypatch):
    body = http.post("/chat", json={"question": CANCEL_Q}).json()
    ids = (body["conversation_id"], body["pending_cancellation"]["action_id"])

    def no_orders():
        raise OSError("disk")

    monkeypatch.setattr(agent, "_default_order_book", no_orders)
    assert confirm(*ids).json() == {"detail": api.ORDERS_UNAVAILABLE_MESSAGE}
    monkeypatch.setattr(agent, "_default_order_book", lambda: BOOK)

    def broken(*args):
        raise cs.StoreError("Conversation database error: OperationalError")

    monkeypatch.setattr(api.cancellation_store, "confirm", broken)
    assert confirm(*ids).status_code == 503
    monkeypatch.setattr(api, "cancellation_store", None)
    assert confirm(*ids).json() == {"detail": api.NOT_READY_MESSAGE}
    assert http.post("/chat", json={"question": CANCEL_Q}).status_code == 503


def test_saving_the_action_failing_saves_nothing(offer_agent, conversation_db, monkeypatch):
    def failing_proposer(conversation_id, order_id):
        def record(connection, now):
            connection.execute("INSERT INTO no_such_table VALUES (1)")

        return record, lambda: pytest.fail("no action was created")

    monkeypatch.setattr(api.cancellation_store, "proposer", failing_proposer)
    response = http.post("/chat", json={"question": CANCEL_Q})
    assert response.status_code == 503
    assert table(conversation_db.path, "SELECT * FROM messages") == []


def test_confirm_logs_never_contain_ids_or_order_numbers(offer_agent, caplog):
    body = http.post("/chat", json={"question": CANCEL_Q}).json()
    action_id = body["pending_cancellation"]["action_id"]
    with caplog.at_level(logging.INFO, logger="support_agent"):
        confirm(body["conversation_id"], action_id)
        confirm(body["conversation_id"], cs.new_conversation_id())
    text = caplog.text
    assert "cancellation_confirm outcome=cancelled" in text and "outcome=not_found" in text
    for secret in (action_id, body["conversation_id"], ORDER):
        assert secret not in text


# --- No cancellation without explicit confirmation ------------------------------------------


def tool_call(order_id=ORDER):
    message = {"role": "assistant", "tool_plan": "Check the order.", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "request_order_cancellation", "arguments": json.dumps({"order_id": order_id, "email": MAYA})}}]}
    return cohere.ChatResponse.model_validate({"id": "r", "finish_reason": "TOOL_CALL", "message": message})


def text_reply(text, source="KB-024"):
    message = {"role": "assistant", "content": [{"type": "text", "text": text}], "citations": [
        {"start": 0, "end": len(text), "text": text, "type": "TEXT_CONTENT",
         "sources": [{"type": "document", "id": source}]}]}
    return cohere.ChatResponse.model_validate({"id": "r", "finish_reason": "COMPLETE", "message": message})


class RecordingClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(json.dumps(kwargs, default=str))
        return self.outcomes.pop(0)


@pytest.fixture
def real_agent(monkeypatch, limiter):
    """The real agent behind the API, with a fake Cohere client, keyword-free
    retrieval (KB-024 only), no rerank and a rewriter that records what it sees."""
    for name in ("COHERE_CHAT_MODEL", "COHERE_CHAT_TIMEOUT_SECONDS", "COHERE_REWRITE_TIMEOUT_SECONDS",
                 "COHERE_EMBED_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    real = agent.answer_question
    setup = {"client": RecordingClient(), "rewriter_saw": []}

    def rewriter(question, history, client):
        setup["rewriter_saw"].append(json.dumps(history))
        return Rewrite(query=question, outcome=NOT_NEEDED)

    def wired(question, **kwargs):
        return real(question, retriever=lambda q: [KB["KB-024"]], client=setup["client"],
                    rerank_config=RerankConfig(), rewriter=rewriter, today=TODAY, **kwargs)

    monkeypatch.setattr(agent, "answer_question", wired)
    return setup


def test_typed_yes_never_cancels(real_agent, conversation_db):
    real_agent["client"].outcomes = [tool_call(), tool_call(), text_reply("Your order has been cancelled.")]

    offer = http.post("/chat", json={"question": CANCEL_Q}).json()
    action_id = offer["pending_cancellation"]["action_id"]
    conversation_id = offer["conversation_id"]

    # "yes": the model asks the read-only tool again, so another offer (a new action) is made.
    yes = http.post("/chat", json={"question": "yes", "conversation_id": conversation_id}).json()
    assert yes["reason"] == CANCELLATION_AVAILABLE
    assert yes["pending_cancellation"]["action_id"] != action_id

    # Pasting the action ID itself into the chat, with the model even claiming success.
    pasted = http.post("/chat", json={"question": f"Yes, confirm {action_id}", "conversation_id": conversation_id})
    assert pasted.status_code == 200

    assert not api.cancellation_store.is_cancelled(ORDER)
    assert table(conversation_db.path, "SELECT * FROM order_cancellations") == []
    assert table(conversation_db.path, "SELECT confirmed_at FROM cancellation_actions") == [(None,), (None,)]

    # Nothing the model or the rewriter was sent contained an action ID, except
    # where the customer typed it themselves (the last question).
    sent = real_agent["client"].requests
    assert action_id not in sent[0] + sent[1]
    assert yes["pending_cancellation"]["action_id"] not in "".join(sent)
    assert not any(action_id in seen for seen in real_agent["rewriter_saw"])
    assert yes["pending_cancellation"]["action_id"] not in "".join(real_agent["rewriter_saw"])


def test_after_confirmation_the_tools_report_the_order_cancelled(real_agent):
    real_agent["client"].outcomes = [tool_call(), tool_call(), text_reply("It is already cancelled.")]
    offer = http.post("/chat", json={"question": CANCEL_Q}).json()
    assert confirm(offer["conversation_id"], offer["pending_cancellation"]["action_id"]).status_code == 200

    again = http.post("/chat", json={"question": CANCEL_Q, "conversation_id": offer["conversation_id"]}).json()

    assert "pending_cancellation" not in again  # No new offer: the tool said not_cancellable.
    second_request = json.loads(real_agent["client"].requests[-1])
    tool_messages = [m for m in second_request["messages"] if m.get("role") == "tool"]
    assert '"not_cancellable"' in json.dumps(tool_messages) and '"cancelled"' in json.dumps(tool_messages)


def test_agent_names_the_cancellable_order_in_the_trace():
    trace = {}
    result = agent.answer_question(
        CANCEL_Q, trace=trace, retriever=lambda q: [KB["KB-024"]], client=RecordingClient(tool_call()),
        rerank_config=RerankConfig(), rewriter=lambda q, h, c: Rewrite(q, NOT_NEEDED), today=TODAY,
    )
    assert result.reason == CANCELLATION_AVAILABLE and trace["cancellable_order_id"] == ORDER

    trace = {}
    agent.answer_question(
        CANCEL_Q, trace=trace, retriever=lambda q: [KB["KB-024"]],
        client=RecordingClient(tool_call(), text_reply("It is already cancelled.")),
        rerank_config=RerankConfig(), rewriter=lambda q, h, c: Rewrite(q, NOT_NEEDED), today=TODAY,
        is_cancelled=lambda order_id: True,
    )
    assert "cancellable_order_id" not in trace


# --- Restoring offers after a reload (Milestone 3.5) ----------------------------------------


def test_offers_report_pending_cancelled_and_expired(stores, clock):
    store, actions = stores
    conversation_id, first = propose(store, actions)
    clock.now += 10
    propose(store, actions, conversation_id=conversation_id)
    clock.now += 10
    propose(store, actions, order_id="ORD-1002", conversation_id=conversation_id)
    assert [o.status for o in actions.offers(conversation_id)] == ["pending"] * 3

    actions.confirm(conversation_id, first.action_id, processing)
    offers = actions.offers(conversation_id)
    # Both ORD-1001 offers now say cancelled (an older offer must not look confirmable).
    assert [(o.order_id, o.status) for o in offers] == [
        ("ORD-1001", "cancelled"), ("ORD-1001", "cancelled"), ("ORD-1002", "pending")]
    assert offers[0].cancelled_at == offers[1].cancelled_at == clock.now

    clock.now += TTL
    assert [o.status for o in actions.offers(conversation_id)] == ["cancelled", "cancelled", "expired"]
    assert actions.offers(cs.new_conversation_id()) == [] and actions.offers("bad") == []


def test_restored_conversation_shows_the_offer_status_without_the_action_id(offer_agent, conversation_db):
    body = http.post("/chat", json={"question": CANCEL_Q}).json()
    conversation_id, pending = body["conversation_id"], body["pending_cancellation"]

    restored = http.get(f"/conversations/{conversation_id}")
    customer, offer = restored.json()["messages"]
    assert customer["cancellation"] is None
    assert offer["cancellation"] == {
        "order_id": ORDER, "status": "pending", "expires_at": pending["expires_at"], "cancelled_at": None}
    assert pending["action_id"] not in restored.text
    assert cancellations._hash(pending["action_id"]) not in restored.text

    confirm(conversation_id, pending["action_id"])
    offer = http.get(f"/conversations/{conversation_id}").json()["messages"][1]
    assert offer["cancellation"]["status"] == "cancelled" and offer["cancellation"]["cancelled_at"]


def test_restored_offer_expires(offer_agent, conversation_db, monkeypatch):
    conversation_id = http.post("/chat", json={"question": CANCEL_Q}).json()["conversation_id"]
    created = conversation_db.clock()
    monkeypatch.setattr(conversation_db, "clock", lambda: created + cancellations.ACTION_TTL_SECONDS + 1)
    offer = http.get(f"/conversations/{conversation_id}").json()["messages"][1]
    assert offer["cancellation"]["status"] == "expired"


def test_restored_ordinary_replies_have_no_cancellation(monkeypatch, limiter):
    monkeypatch.setattr(agent, "answer_question", lambda q, **k: SupportAnswer(
        "Orders can be cancelled while processing.",
        [{"id": "KB-024", "title": KB["KB-024"]["title"], "url": KB["KB-024"]["url"]}], False, "answered"))
    conversation_id = http.post("/chat", json={"question": "How do I cancel?"}).json()["conversation_id"]
    messages = http.get(f"/conversations/{conversation_id}").json()["messages"]
    assert [m["cancellation"] for m in messages] == [None, None]


def test_restoring_fails_safely_when_offers_cannot_be_read(offer_agent, monkeypatch):
    conversation_id = http.post("/chat", json={"question": CANCEL_Q}).json()["conversation_id"]

    def broken(conversation_id):
        raise cs.StoreError("Conversation database error: OperationalError")

    monkeypatch.setattr(api.cancellation_store, "offers", broken)
    response = http.get(f"/conversations/{conversation_id}")
    assert response.status_code == 503 and response.json() == {"detail": api.STORAGE_UNAVAILABLE_MESSAGE}


def test_offer_wording_asks_for_confirmation_and_not_the_orders_page():
    text = cancellation_message(ORDER)
    assert ORDER in text and "has not been cancelled yet" in text
    assert "Confirm cancellation" in text and "Keep order" in text
    assert "Orders" not in text and "Cancel order" not in text
    assert "yes" not in text.lower().split()  # Never asks the customer to type a confirmation.
