"""End-to-end reliability scenarios (Phase 5, Milestone 5).

Each test drives the real application over HTTP: the real agent, real hybrid
retrieval (keyword + semantic, with the conftest's fake embedding client),
the real query rewriter, the real tool registry and order checks, and real
SQLite storage for conversations, cancellations, hand-offs and feedback.
Only the Cohere chat client is scripted. Sockets are blocked, so nothing can
reach the network: these tests prove orchestration, not live model quality.
"""

import json
import socket
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import cohere
import httpx
import pytest
from fastapi.testclient import TestClient

from support_agent import agent, api
from support_agent import conversation_store as cs
from support_agent.answer import CANCELLATION_AVAILABLE, HANDOFF_MESSAGE, NEEDS_ORDER_DETAILS_MESSAGE
from support_agent.rate_limit import RateLimiter
from support_agent.rerank import RerankConfig

TODAY = date(2026, 11, 20)
PRIYA = "priya.nair@example.com"  # ORD-1003, shipped
MAYA = "maya.fernandes@example.com"  # ORD-1001 processing, ORD-1005 delivered

http = TestClient(api.app)


# --- A scripted Cohere chat client ------------------------------------------------------


def _response(finish, message):
    return cohere.ChatResponse.model_validate({"id": "scripted", "finish_reason": finish, "message": message})


def text(answer, *sources):
    message = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    if sources:
        message["citations"] = [{"start": 0, "end": len(answer), "text": answer, "type": "TEXT_CONTENT",
                                 "sources": [{"type": kind, "id": sid, **({"tool_output": {}} if kind == "tool" else {})}
                                             for kind, sid in sources]}]
    return _response("COMPLETE", message)


def cite_first_article(answer):
    """An answer citing whichever article retrieval put first."""
    return lambda request: text(answer, ("document", request["documents"][0]["id"]))


def cite_tool(answer, call_id="call_1"):
    return lambda request: text(answer, ("tool", f"{call_id}:0"))


def tool_call(name, **arguments):
    return _response("TOOL_CALL", {"role": "assistant", "tool_plan": "Look it up.", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]})


def rewrite(standalone, clarify=False):
    return text(json.dumps({"standalone_query": standalone, "needs_clarification": clarify}))


class ApiFailure(Exception):
    status_code = 503


class ScriptedCohere:
    """chat(): rewrite requests (they set response_format) take the next
    `rewrites` outcome; answer requests the next `answers` outcome. An outcome
    is a response, an exception to raise, or a function of the request.
    `default` answers when the queue is empty (used by concurrent tests).
    Thread-safe; records every request."""

    def __init__(self):
        self.answers, self.rewrites, self.requests = [], [], []
        self.default = None
        self._lock = threading.Lock()

    def chat(self, **request):
        with self._lock:
            self.requests.append(request)
            queue = self.rewrites if "response_format" in request else self.answers
            outcome = queue.pop(0) if queue else self.default
        if outcome is None:
            raise AssertionError("unexpected Cohere request")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome(request) if callable(outcome) else outcome

    def answer_requests(self):
        return [r for r in self.requests if "response_format" not in r]


@pytest.fixture
def cohere_fake(semantic_kb, monkeypatch):
    """The real agent behind the API, with the scripted chat client, real
    hybrid retrieval (fake embeddings), the real rewriter, rerank off and a
    fixed date. No network."""
    for name in ("COHERE_CHAT_MODEL", "COHERE_CHAT_TIMEOUT_SECONDS", "COHERE_EMBED_TIMEOUT_SECONDS",
                 "COHERE_REWRITE_TIMEOUT_SECONDS", "COHERE_REWRITE_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(api, "chat_limiter", RateLimiter(per_minute=1000, per_day=10000))
    fake = ScriptedCohere()
    real = agent.answer_question

    def wired(question, **kwargs):
        return real(question, client=fake, rerank_config=RerankConfig(), today=TODAY, **kwargs)

    monkeypatch.setattr(agent, "answer_question", wired)
    return fake


def chat(question, conversation_id=None):
    body = {"question": question} if conversation_id is None else {"question": question, "conversation_id": conversation_id}
    return http.post("/chat", json=body)


def restore(conversation_id):
    return http.get(f"/conversations/{conversation_id}").json()


def rows(store, sql, *args):
    with sqlite3.connect(store.path) as connection:
        return connection.execute(sql, args).fetchall()


# --- Answers and hand-offs ------------------------------------------------------------------


def test_normal_knowledge_base_question(cohere_fake, semantic_kb, conversation_db):
    cohere_fake.answers = [cite_first_article("Use Forgot password on the sign-in page.")]

    body = chat("How do I reset my password?").json()

    assert body["reason"] == "answered" and body["escalate"] is False and body["message_index"] == 1
    request = cohere_fake.requests[0]
    assert body["sources"][0]["id"] == request["documents"][0]["id"]
    assert "tools" not in request  # No order number: no tools offered.
    assert semantic_kb.client.calls  # Hybrid retrieval really embedded the question.
    assert restore(body["conversation_id"])["handoff"] is None


def test_unsupported_question_is_a_recorded_handoff_that_survives_a_reload(cohere_fake, conversation_db):
    cohere_fake.answers = [text("INSUFFICIENT_EVIDENCE")]

    body = chat("Can you book me a flight to Paris?").json()

    assert (body["answer"], body["escalate"], body["reason"]) == (HANDOFF_MESSAGE, True, "insufficient_evidence")
    restored = restore(body["conversation_id"])
    assert restored["handoff"]["status"] == "open" and restored["handoff"]["category"] == "no_answer"
    assert restored["messages"][1]["escalate"] is True
    assert http.post("/feedback", json={"conversation_id": body["conversation_id"], "message_index": 1,
                                        "rating": "negative"}).status_code == 200
    assert restore(body["conversation_id"])["messages"][1]["feedback"] == "negative"


def test_follow_up_uses_conversation_memory_for_retrieval_only(cohere_fake, conversation_db):
    cohere_fake.answers = [cite_first_article("Start a return from Orders."),
                           cite_first_article("Returns are accepted within 30 days.")]
    first = chat("How can I return an item?").json()
    cohere_fake.rewrites = [rewrite("Can I return an item 35 days after purchase?")]

    second = chat("What if I bought it 35 days ago?", first["conversation_id"]).json()

    assert second["reason"] == "answered" and second["message_index"] == 3
    rewrite_request = next(r for r in cohere_fake.requests if "response_format" in r)
    assert "How can I return an item?" in json.dumps(rewrite_request["messages"])  # History reached the rewriter.
    answer_request = cohere_fake.answer_requests()[-1]
    assert "35 days after purchase" in json.dumps(answer_request["messages"])  # Standalone query passed along.
    stored = rows(conversation_db, "SELECT content, retrieval_query FROM messages WHERE role = 'customer' ORDER BY id")
    assert stored[1] == ("What if I bought it 35 days ago?", "Can I return an item 35 days after purchase?")


def test_a_rewrite_that_invents_facts_falls_back_to_the_message(cohere_fake, conversation_db):
    cohere_fake.answers = [cite_first_article("Start a return from Orders."), cite_first_article("See the policy.")]
    first = chat("How can I return an item?").json()
    cohere_fake.rewrites = [rewrite("Can I return an item 90 days after purchase?")]  # 90 never written.

    chat("What about a late return?", first["conversation_id"])

    stored = rows(conversation_db, "SELECT retrieval_query FROM messages WHERE role = 'customer' ORDER BY id")
    assert stored[1] == ("What about a late return?",)


# --- Orders -----------------------------------------------------------------------------------


def tool_documents(request):
    return [m["content"][0]["document"]["data"] for m in request["messages"] if m.get("role") == "tool"]


def test_order_lookup(cohere_fake):
    cohere_fake.answers = [tool_call("get_order_status", order_id="ORD-1003", email=PRIYA),
                           cite_tool("Your order is on its way.")]

    body = chat(f"Where is my order ORD-1003? My email is {PRIYA}.").json()

    assert body["reason"] == "answered"
    assert {"id": "order:ORD-1003:get_order_status", "title": "Your order ORD-1003 (order status)",
            "url": ""} in body["sources"]
    final = cohere_fake.answer_requests()[-1]
    assert final["tool_choice"] == "NONE"
    (document,) = tool_documents(final)
    assert document["status"] == "ok" and document["order_status"] == "shipped"
    assert PRIYA not in json.dumps(tool_documents(final))  # Results never carry the email.


def test_refund_eligibility(cohere_fake):
    cohere_fake.answers = [tool_call("check_refund_eligibility", order_id="ORD-1005", email=MAYA, item_name="Linen Shirt"),
                           cite_tool("Yes, the Linen Shirt can be returned.")]

    body = chat(f"Can I return the Linen Shirt from ORD-1005? {MAYA}").json()

    assert body["reason"] == "answered"
    (document,) = tool_documents(cohere_fake.answer_requests()[-1])
    assert json.loads(document["items"])[0]["decision"] == "eligible"


def test_someone_elses_order_reveals_nothing(cohere_fake):
    cohere_fake.answers = [tool_call("get_order_status", order_id="ORD-1003", email=MAYA),
                           cite_tool("I couldn't find that order.")]

    chat(f"Where is ORD-1003? My email is {MAYA}")

    (document,) = tool_documents(cohere_fake.answer_requests()[-1])
    assert document == {"status": "not_found"}


def test_missing_email_asks_for_details_without_a_handoff(cohere_fake, conversation_db):
    """Milestone 5 fix: a call that leaves out the email asks the customer for
    it; it used to hand off and open a service_problem hand-off record."""
    cohere_fake.answers = [tool_call("request_order_cancellation", order_id="ORD-1001")]

    body = chat("Can I cancel ORD-1001?").json()

    assert (body["answer"], body["escalate"]) == (NEEDS_ORDER_DETAILS_MESSAGE, False)
    assert rows(conversation_db, "SELECT count(*) FROM handoffs") == [(0,)]


# --- Cancellation ---------------------------------------------------------------------------


def test_cancellation_offer_confirmation_and_restoration(cohere_fake, conversation_db):
    cohere_fake.answers = [tool_call("request_order_cancellation", order_id="ORD-1001", email=MAYA)]
    offer = chat(f"Please cancel ORD-1001, my email is {MAYA}").json()
    conversation_id = offer["conversation_id"]
    action_id = offer["pending_cancellation"]["action_id"]
    assert offer["reason"] == CANCELLATION_AVAILABLE
    assert restore(conversation_id)["messages"][1]["cancellation"]["status"] == "pending"

    confirmations = [http.post("/cancellations/confirm",
                               json={"conversation_id": conversation_id, "action_id": action_id}) for _ in range(2)]
    assert [c.json()["already_confirmed"] for c in confirmations] == [False, True]
    assert restore(conversation_id)["messages"][1]["cancellation"]["status"] == "cancelled"

    # The tools now see the cancellation recorded in SQLite.
    cohere_fake.rewrites = [rewrite("What is the status of order ORD-1001?")]
    cohere_fake.answers = [tool_call("get_order_status", order_id="ORD-1001", email=MAYA),
                           cite_tool("It has been cancelled.")]
    chat("What's its status now?", conversation_id)
    (document,) = tool_documents(cohere_fake.answer_requests()[-1])
    assert document["order_status"] == "cancelled"
    # No action ID reached the model or the rewriter at any point.
    assert action_id not in json.dumps(cohere_fake.requests, default=str)
    assert rows(conversation_db, "SELECT count(*) FROM order_cancellations") == [(1,)]


# --- Failures ---------------------------------------------------------------------------------


@pytest.mark.parametrize("outcomes, reason", [
    ([httpx.ReadTimeout("timed out")], "llm_error"),  # A timeout is not retried.
    ([ApiFailure("busy"), ApiFailure("busy")], "llm_error"),  # 503, retried once, fails again.
    ([_response("COMPLETE", {"role": "assistant", "content": []})], "malformed_response"),
    ([_response("MAX_TOKENS", {"role": "assistant", "content": [{"type": "text", "text": "Half an ans"}]})],
     "incomplete_response"),
])
def test_cohere_failures_are_recorded_handoffs(cohere_fake, conversation_db, outcomes, reason):
    cohere_fake.answers = list(outcomes)

    body = chat("How do I reset my password?").json()

    assert (body["answer"], body["sources"], body["escalate"], body["reason"]) == (HANDOFF_MESSAGE, [], True, reason)
    assert "timed out" not in json.dumps(body) and "busy" not in json.dumps(body)
    assert restore(body["conversation_id"])["handoff"]["status"] == "open"


def test_retry_then_success_is_one_answer(cohere_fake):
    cohere_fake.answers = [ApiFailure("busy"), cite_first_article("Use Forgot password.")]
    assert chat("How do I reset my password?").json()["reason"] == "answered"
    assert len(cohere_fake.answer_requests()) == 2


def test_database_down_before_answering_makes_no_cohere_call(cohere_fake, conversation_db, monkeypatch):
    def broken(*args, **kwargs):
        raise cs.StoreError("Conversation database error: OperationalError")

    monkeypatch.setattr(conversation_db, "ping", broken)
    response = chat("How do I reset my password?")
    assert response.status_code == 503 and response.json() == {"detail": api.STORAGE_UNAVAILABLE_MESSAGE}
    assert cohere_fake.requests == []


def test_database_failing_while_saving_shows_and_records_nothing(cohere_fake, conversation_db, monkeypatch):
    cohere_fake.answers = [text("INSUFFICIENT_EVIDENCE")]

    def broken(*args, **kwargs):
        raise cs.StoreError("Conversation database error: OperationalError")

    monkeypatch.setattr(conversation_db, "add_exchange", broken)
    response = chat("Can you book me a flight?")
    assert response.status_code == 503 and HANDOFF_MESSAGE not in response.text
    assert rows(conversation_db, "SELECT count(*) FROM messages") == [(0,)]
    assert rows(conversation_db, "SELECT count(*) FROM handoffs") == [(0,)]


def test_failure_outside_the_endpoint_never_claims_a_record(cohere_fake, monkeypatch):
    """Milestone 5 fix: the middleware's last-resort 500 used HANDOFF_MESSAGE
    ("I've recorded your question...") although nothing was saved."""
    class BrokenLimiter:
        def check(self):
            raise RuntimeError("limiter bug")

    monkeypatch.setattr(api, "chat_limiter", BrokenLimiter())
    response = chat("How do I reset my password?")
    body = response.json()
    assert response.status_code == 500 and body["answer"] == api.UNSAVED_ERROR_MESSAGE
    assert "recorded" not in body["answer"] and "conversation_id" not in body and "message_index" not in body
    assert "limiter bug" not in response.text


# --- Concurrency and duplicates ---------------------------------------------------------------


def test_concurrent_chats_each_get_their_own_saved_conversation(cohere_fake, conversation_db):
    cohere_fake.default = cite_first_article("Use Forgot password.")
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda i: chat(f"How do I reset my password? ({i})"), range(8)))
    assert [r.status_code for r in responses] == [200] * 8
    assert len({r.json()["conversation_id"] for r in responses}) == 8
    assert rows(conversation_db, "SELECT count(*) FROM messages") == [(16,)]


def test_concurrent_follow_ups_never_overfill_a_conversation(cohere_fake, conversation_db, monkeypatch):
    cohere_fake.default = cite_first_article("Use Forgot password.")
    conversation_id = chat("How do I reset my password?").json()["conversation_id"]
    monkeypatch.setattr(conversation_db, "max_messages", 6)  # Room for two more exchanges.
    # Rewrites get no usable JSON (each falls back to the message itself);
    # answers cite the first article.
    cohere_fake.default = lambda request: (
        text("not json") if "response_format" in request else text("Use the app.", ("document", request["documents"][0]["id"])))

    def follow_up(i):
        return chat(f"And on mobile? ({i})", conversation_id).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(pool.map(follow_up, range(6)))
    assert statuses.count(200) == 2 and set(statuses) <= {200, 409}
    assert rows(conversation_db, "SELECT count(*) FROM messages") == [(6,)]


def test_concurrent_duplicate_confirmations_and_feedback(cohere_fake, conversation_db):
    cohere_fake.answers = [tool_call("request_order_cancellation", order_id="ORD-1001", email=MAYA)]
    offer = chat(f"Please cancel ORD-1001, my email is {MAYA}").json()
    ids = {"conversation_id": offer["conversation_id"], "action_id": offer["pending_cancellation"]["action_id"]}
    rating = {"conversation_id": offer["conversation_id"], "message_index": 1, "rating": "positive"}

    with ThreadPoolExecutor(max_workers=8) as pool:
        confirms = list(pool.map(lambda _: http.post("/cancellations/confirm", json=ids).json(), range(8)))
        ratings = list(pool.map(lambda _: http.post("/feedback", json=rating).json(), range(8)))

    assert sorted(c["already_confirmed"] for c in confirms) == [False] + [True] * 7
    assert sorted(r["changed"] for r in ratings) == [False] * 7 + [True]
    assert rows(conversation_db, "SELECT count(*) FROM order_cancellations") == [(1,)]
    assert rows(conversation_db, "SELECT count(*) FROM feedback") == [(1,)]
