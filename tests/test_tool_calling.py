"""Offline tests for the Cohere tool-call cycle (answer.generate_answer_with_tools)
and its use in the agent. Every Cohere client is a fake; responses are built
with the installed SDK's own pydantic models, so attribute names are real.
No request is ever sent."""

import copy
import json
import logging
import socket
from datetime import date

import cohere
import pytest

from support_agent import agent, answer, cohere_calls, tools
from support_agent.agent import answer_question
from support_agent.answer import (
    CANCELLATION_AVAILABLE,
    HANDOFF_MESSAGE,
    NEEDS_ORDER_DETAILS,
    NEEDS_ORDER_DETAILS_MESSAGE,
    SupportAnswer,
    ToolRound,
    cancellation_message,
    generate_answer_with_tools,
    tool_source,
)
from support_agent.demo_orders import load_orders
from support_agent.kb import load_articles
from support_agent.query_rewrite import NOT_NEEDED, Rewrite
from support_agent.rerank import RerankConfig

TODAY = date(2026, 11, 20)
BOOK = load_orders()
KB = {a["id"]: a for a in load_articles()}
ARTICLES = [KB["KB-003"], KB["KB-024"]]
OFF = RerankConfig()
PRIYA = "priya.nair@example.com"
MAYA = "maya.fernandes@example.com"
STATUS_Q = f"Where is my order ORD-1003? My email is {PRIYA}."
CANCEL_Q = f"Please cancel ORD-1001, email {MAYA}"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    for name in ("COHERE_CHAT_MODEL", "COHERE_CHAT_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def executions(monkeypatch):
    """Counts every tools.execute_all call (the only way tools run)."""
    calls = []
    real = tools.execute_all

    def counting(tool_calls, context):
        calls.append([c.id for c in tool_calls])
        return real(tool_calls, context)

    monkeypatch.setattr(tools, "execute_all", counting)
    return calls


# --- Fake Cohere responses, built with the SDK's own models ------------------------------


def tool_call_response(*calls, plan="I will look up the order."):
    """calls: (id, name, arguments) where arguments is a dict (JSON-encoded) or raw text."""
    message = {"role": "assistant", "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": name, "arguments": args if isinstance(args, str) or args is None else json.dumps(args)}}
        for cid, name, args in calls
    ]}
    if plan is not None:
        message["tool_plan"] = plan
    return cohere.ChatResponse.model_validate({"id": "r1", "finish_reason": "TOOL_CALL", "message": message})


def final_response(text, *sources, finish_reason="COMPLETE"):
    """sources: ("tool" | "document", id)."""
    message = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    if sources:
        message["citations"] = [{"start": 0, "end": len(text), "text": text, "type": "TEXT_CONTENT", "sources": [
            {"type": "tool", "id": sid, "tool_output": {}} if kind == "tool" else {"type": "document", "id": sid}
            for kind, sid in sources
        ]}]
    return cohere.ChatResponse.model_validate({"id": "r2", "finish_reason": finish_reason, "message": message})


class FakeClient:
    """chat() returns (or raises) the next outcome and records a deep copy of its arguments."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def context(*customer_messages):
    return tools.ToolContext(BOOK, tuple(customer_messages), TODAY)


def generate(client, question=STATUS_Q, articles=ARTICLES, **kwargs):
    return generate_answer_with_tools(question, articles, client, context(question), **kwargs)


STATUS_ARGS = {"order_id": "ORD-1003", "email": PRIYA}
STATUS_SOURCE = {"id": "order:ORD-1003:get_order_status", "title": "Your order ORD-1003 (order status)", "url": ""}


def assert_handoff(result, reason):
    assert (result.answer, result.sources, result.escalate, result.reason) == (HANDOFF_MESSAGE, [], True, reason)


# --- The cycle ---------------------------------------------------------------------------


def test_one_successful_tool_call(executions):
    client = FakeClient(
        tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
        final_response("Your order is in transit with Northwind Parcel.", ("tool", "call_1:0")),
    )

    round_ = generate(client)

    assert round_.answer == SupportAnswer("Your order is in transit with Northwind Parcel.", [STATUS_SOURCE], False, "answered")
    assert [(r.call_id, r.status) for r in round_.results] == [("call_1", "ok")]
    assert executions == [["call_1"]]


def test_message_structure_of_both_calls():
    client = FakeClient(
        tool_call_response(("call_1", "get_order_status", STATUS_ARGS), plan="Look up ORD-1003."),
        final_response("In transit.", ("tool", "call_1:0")),
    )
    generate(client)
    first, second = client.calls

    # First call: articles and tools, no tool_choice.
    assert [d["id"] for d in first["documents"]] == ["KB-003", "KB-024"]
    assert first["tools"] == tools.definitions() and "tool_choice" not in first
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert first["messages"][0]["content"] == answer.TOOL_SYSTEM_PROMPT
    assert first["messages"][1] == {"role": "user", "content": STATUS_Q}

    # Second call: the same start, the model's own tool-call message unchanged, one tool message.
    assert second["tool_choice"] == "NONE"
    assert second["documents"] == first["documents"] and second["tools"] == first["tools"]
    system, user, assistant_msg, tool_msg = second["messages"]
    assert [system, user] == first["messages"]
    assert assistant_msg == {
        "role": "assistant", "tool_plan": "Look up ORD-1003.",
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_order_status", "arguments": json.dumps(STATUS_ARGS)}}],
    }
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
    (content,) = tool_msg["content"]
    assert content["type"] == "document" and content["document"]["id"] == "tool-1-get_order_status"
    assert content["document"]["data"]["order_status"] == "shipped"
    # Tool results are kept apart from the article documents.
    assert all(not d["id"].startswith("tool-") for d in second["documents"])
    # Every chat call disables SDK retries (the shared policy).
    assert all(c["request_options"]["max_retries"] == 0 for c in client.calls)


def test_second_call_messages_validate_against_sdk_types():
    client = FakeClient(
        tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
        final_response("In transit.", ("tool", "call_1:0")),
    )
    generate(client)
    _, _, assistant_msg, tool_msg = client.calls[1]["messages"]
    parsed = cohere.AssistantChatMessageV2.model_validate(assistant_msg)
    assert parsed.tool_calls[0].id == "call_1" and parsed.tool_plan == "I will look up the order."
    assert cohere.ToolChatMessageV2.model_validate(tool_msg).tool_call_id == "call_1"


def test_multiple_calls_in_one_round_keep_their_ids(executions):
    refund_args = {"order_id": "ORD-1005", "email": MAYA}
    question = f"{STATUS_Q} Also can I return ORD-1005? email {MAYA}"
    client = FakeClient(
        tool_call_response(("a1", "get_order_status", STATUS_ARGS), ("b2", "check_refund_eligibility", refund_args)),
        final_response("In transit, and yes.", ("tool", "a1:0"), ("tool", "b2:0")),
    )

    round_ = generate(client, question)

    tool_msgs = client.calls[1]["messages"][3:]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a1", "b2"]
    assert [m["content"][0]["document"]["id"] for m in tool_msgs] == ["tool-1-get_order_status", "tool-2-check_refund_eligibility"]
    assert json.loads(tool_msgs[1]["content"][0]["document"]["data"]["items"])[0]["decision"] == "eligible"
    assert [s["id"] for s in round_.answer.sources] == ["order:ORD-1003:get_order_status", "order:ORD-1005:check_refund_eligibility"]
    assert executions == [["a1", "b2"]]


def test_more_than_three_calls_run_three_and_tell_the_model(executions):
    calls = [(f"c{n}", "get_order_status", STATUS_ARGS) for n in range(1, 5)]
    client = FakeClient(tool_call_response(*calls), final_response("In transit.", ("tool", "c1:0")))

    round_ = generate(client)

    assert [r.status for r in round_.results] == ["ok", "ok", "ok", "limit_exceeded"]
    assert [m["tool_call_id"] for m in client.calls[1]["messages"][3:]] == ["c1", "c2", "c3", "c4"]
    assert round_.answer.reason == "answered"


def test_article_and_tool_citations_together():
    client = FakeClient(
        tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
        final_response("In transit; updates can take 24 hours.", ("tool", "call_1:0"), ("document", "KB-003")),
    )
    round_ = generate(client)
    article = {"id": "KB-003", "title": KB["KB-003"]["title"], "url": KB["KB-003"]["url"]}
    assert round_.answer.sources == [STATUS_SOURCE, article]


@pytest.mark.parametrize("cited", ["tool-1-get_order_status", "call_1"])
def test_tool_citations_by_document_id_or_call_id(cited):
    client = FakeClient(tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
                        final_response("In transit.", ("tool", cited)))
    assert generate(client).answer.sources == [STATUS_SOURCE]


@pytest.mark.parametrize("cited", [
    ("tool", "call_0:0"),  # a call from an earlier question
    ("tool", "tool-1-get_order_status-old"),
    ("tool", None),
    ("document", "KB-015"),  # an article that was not supplied
])
def test_citations_outside_this_question_are_rejected(cited):
    client = FakeClient(tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
                        final_response("In transit.", ("tool", "call_1:0"), cited))
    assert_handoff(generate(client).answer, "invalid_citations")


@pytest.mark.parametrize("final, reason", [
    (final_response("In transit."), "no_citations"),
    (final_response("INSUFFICIENT_EVIDENCE", ("tool", "call_1:0")), "insufficient_evidence"),
    (final_response("In tra", ("tool", "call_1:0"), finish_reason="MAX_TOKENS"), "incomplete_response"),
    (final_response("x", ("tool", "call_1:0"), finish_reason="ERROR"), "llm_error"),
    (final_response("x", ("tool", "call_1:0"), finish_reason="TOOL_CALL"), "malformed_response"),
    (final_response("x", ("tool", "call_1:0"), finish_reason="SOMETHING_NEW"), "malformed_response"),
    (tool_call_response(("call_2", "get_order_status", STATUS_ARGS)), "malformed_response"),  # asks for more tools
])
def test_bad_final_answers_are_never_shown(final, reason, executions):
    client = FakeClient(tool_call_response(("call_1", "get_order_status", STATUS_ARGS)), final)
    assert_handoff(generate(client).answer, reason)
    assert len(executions) == 1  # the extra tool request was not executed


def test_final_answer_with_more_tool_calls_is_rejected_even_if_complete():
    final = final_response("In transit.", ("tool", "call_1:0"))
    final.message.tool_calls = tool_call_response(("call_9", "get_order_status", STATUS_ARGS)).message.tool_calls
    client = FakeClient(tool_call_response(("call_1", "get_order_status", STATUS_ARGS)), final)
    assert_handoff(generate(client).answer, "malformed_response")


# --- Unknown, invalid and malformed tool calls -------------------------------------------


@pytest.mark.parametrize("call, reason", [
    (("call_1", "delete_order", STATUS_ARGS), "invalid_tool_call"),
    (("call_1", "get_order_status", "not json"), "invalid_tool_call"),
    (("call_1", "get_order_status", {**STATUS_ARGS, "admin": True}), "invalid_tool_call"),
    (("call_1", "get_order_status", None), "invalid_tool_call"),
])
def test_unknown_or_invalid_calls_hand_off_without_a_second_call(call, reason):
    client = FakeClient(tool_call_response(call))
    round_ = generate(client)
    assert_handoff(round_.answer, reason)
    assert len(client.calls) == 1


def malformed(mutate):
    response = tool_call_response(("call_1", "get_order_status", STATUS_ARGS), ("call_2", "get_order_status", STATUS_ARGS))
    mutate(response.message)
    return response


@pytest.mark.parametrize("response", [
    malformed(lambda m: setattr(m, "tool_calls", None)),
    malformed(lambda m: setattr(m, "tool_calls", [])),
    malformed(lambda m: setattr(m.tool_calls[0], "id", None)),
    malformed(lambda m: setattr(m.tool_calls[1], "id", "  ")),
    malformed(lambda m: setattr(m.tool_calls[1], "id", "call_1")),  # duplicate ID
    malformed(lambda m: setattr(m, "tool_calls", "get_order_status")),
], ids=["none", "empty", "missing-id", "blank-id", "duplicate-id", "not-a-list"])
def test_malformed_tool_call_responses_execute_nothing(response, executions):
    client = FakeClient(response)
    assert_handoff(generate(client).answer, "malformed_response")
    assert executions == [] and len(client.calls) == 1


def test_missing_function_is_an_unknown_tool():
    response = tool_call_response(("call_1", "get_order_status", STATUS_ARGS))
    response.message.tool_calls[0].function = None
    assert_handoff(generate(FakeClient(response)).answer, "invalid_tool_call")


def test_tool_errors_hand_off_safely(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("SECRET database detail")

    monkeypatch.setattr(tools, "check_refund", broken)
    caplog.set_level(logging.DEBUG)
    client = FakeClient(tool_call_response(("call_1", "check_refund_eligibility", {"order_id": "ORD-1005", "email": MAYA})))

    round_ = generate(client, f"Refund for ORD-1005? {MAYA}")

    assert_handoff(round_.answer, "tool_error")
    assert len(client.calls) == 1 and "SECRET" not in caplog.text


# --- Server-written outcomes ---------------------------------------------------------------


def test_missing_details_get_a_fixed_request(executions):
    client = FakeClient(tool_call_response(("call_1", "get_order_status", STATUS_ARGS)))
    round_ = generate(client, "Where is ORD-1003?")  # no email written by the customer
    assert round_.answer == SupportAnswer(NEEDS_ORDER_DETAILS_MESSAGE, [], False, NEEDS_ORDER_DETAILS)
    assert len(client.calls) == 1


def test_cancellable_order_gets_a_fixed_reply_and_nothing_changes():
    snapshot = copy.deepcopy(BOOK._orders)
    client = FakeClient(tool_call_response(("call_1", "request_order_cancellation", {"order_id": "ORD-1001", "email": MAYA})))

    round_ = generate(client, CANCEL_Q)

    assert round_.answer.reason == CANCELLATION_AVAILABLE and round_.answer.escalate is False
    assert round_.answer.answer == cancellation_message("ORD-1001")
    assert "has not been cancelled" in round_.answer.answer
    assert [s["id"] for s in round_.answer.sources] == ["KB-024", "order:ORD-1001:request_order_cancellation"]
    assert len(client.calls) == 1  # no model wording for a state change
    assert BOOK._orders == snapshot and BOOK.find("ORD-1001", MAYA).status == "processing"


def test_not_cancellable_order_goes_to_the_model():
    client = FakeClient(
        tool_call_response(("call_1", "request_order_cancellation", STATUS_ARGS)),
        final_response("It has shipped, so it can't be cancelled.", ("tool", "call_1:0"), ("document", "KB-024")),
    )
    round_ = generate(client, f"Cancel ORD-1003 {PRIYA}")
    assert round_.answer.reason == "answered" and len(client.calls) == 2


# --- No tool call: the ordinary rules ----------------------------------------------------


def test_direct_answer_with_tools_offered_uses_the_article_rules(executions):
    client = FakeClient(final_response("Track it from Orders.", ("document", "KB-003")))
    round_ = generate(client)
    assert round_.answer.reason == "answered" and [s["id"] for s in round_.answer.sources] == ["KB-003"]
    assert executions == [] and round_.results == []


def test_direct_answer_citing_a_tool_is_not_accepted():
    client = FakeClient(final_response("In transit.", ("tool", "call_1:0")))
    assert_handoff(generate(client).answer, "no_citations")


# --- Time budget and retries ---------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_both_calls_share_one_budget(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(cohere_calls, "clock", clock)

    class SlowClient(FakeClient):
        def chat(self, **kwargs):
            clock.now += 12.0  # each call takes 12 s
            return super().chat(**kwargs)

    client = SlowClient(tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
                        final_response("In transit.", ("tool", "call_1:0")))
    generate(client)
    assert [c["request_options"]["timeout"] for c in client.calls] == [25.0, 13.0]


def test_too_little_time_left_hands_off_without_a_second_call(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(cohere_calls, "clock", clock)

    class SlowClient(FakeClient):
        def chat(self, **kwargs):
            clock.now += 21.0
            return super().chat(**kwargs)

    client = SlowClient(tool_call_response(("call_1", "get_order_status", STATUS_ARGS)))
    assert_handoff(generate(client).answer, "llm_error")
    assert len(client.calls) == 1


class ServerError(Exception):
    status_code = 503


def test_final_call_retry_never_reruns_tools(executions, no_real_retry_waits):
    client = FakeClient(
        tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
        ServerError(),
        final_response("In transit.", ("tool", "call_1:0")),
    )
    round_ = generate(client)
    assert round_.answer.reason == "answered"
    assert len(client.calls) == 3 and executions == [["call_1"]]
    assert client.calls[1]["messages"] == client.calls[2]["messages"]  # the same request, retried


def test_first_call_failure_hands_off(executions):
    class AuthError(Exception):
        status_code = 401

    assert_handoff(generate(FakeClient(AuthError())).answer, "llm_error")
    assert executions == []


def test_empty_question():
    assert_handoff(generate(FakeClient(), "  ").answer, "empty_question")


# --- The agent ----------------------------------------------------------------------------


def ask(question, client, history=None, **kwargs):
    kwargs.setdefault("retriever", lambda q: ARTICLES)
    kwargs.setdefault("rewriter", lambda q, h, c: Rewrite(q, NOT_NEEDED))
    return answer_question(question, history=history, client=client, rerank_config=OFF, today=TODAY,
                           order_book=BOOK, **kwargs)


@pytest.mark.parametrize("question, history, offered", [
    ("Where is my order?", None, False),
    ("ORD 1003 status", None, False),
    ("XORD-1003", None, False),
    ("Status of ord-1003 please", None, True),
    ("and the email is priya.nair@example.com", [{"role": "customer", "content": "Order ORD-1003"}], True),
    ("what about it?", [{"role": "assistant", "content": "Is it ORD-1003?"}], False),  # assistant text doesn't count
])
def test_tools_are_offered_only_for_a_customer_written_order_number(question, history, offered):
    assert agent.mentions_order_number(question, history) is offered


def test_questions_without_an_order_number_keep_the_existing_flow():
    """No tools, no tool generator, the original prompt and an unchanged trace."""
    client = FakeClient(final_response("Track it from Orders.", ("document", "KB-003")))
    trace = {}

    result = ask("How do I track my order?", client, trace=trace,
                 tool_generator=lambda *a, **k: pytest.fail("tools offered"))

    assert result.reason == "answered"
    (call,) = client.calls
    assert "tools" not in call and call["messages"][0]["content"] == answer.SYSTEM_PROMPT
    assert trace == {"rewrite": NOT_NEEDED, "retrieval_query": "How do I track my order?"}


def test_agent_runs_the_tool_cycle_and_checks_the_result():
    client = FakeClient(
        tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
        final_response("Your order is in transit.", ("tool", "call_1:0")),
    )
    trace = {}
    result = ask(STATUS_Q, client, trace=trace)
    assert result == SupportAnswer("Your order is in transit.", [STATUS_SOURCE], False, "answered")
    assert trace["tools_offered"] is True and trace["tool_calls"] == 1


def test_agent_tool_cycle_with_details_from_earlier_turns():
    history = [{"role": "customer", "content": "My order ORD-1003 hasn't arrived"},
               {"role": "assistant", "content": "Which email did you use?"}]
    client = FakeClient(
        tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
        final_response("It is in transit.", ("tool", "call_1:0")),
    )
    assert ask(PRIYA, client, history=history).reason == "answered"


def test_agent_passes_the_fixed_outcomes():
    details = ask("Where is ORD-1003?", FakeClient(tool_call_response(("c", "get_order_status", STATUS_ARGS))))
    assert details.reason == NEEDS_ORDER_DETAILS and details.escalate is False
    cancel = ask(CANCEL_Q, FakeClient(tool_call_response(
        ("c", "request_order_cancellation", {"order_id": "ORD-1001", "email": MAYA}))))
    assert cancel.reason == CANCELLATION_AVAILABLE and cancel.answer == cancellation_message("ORD-1001")


def _round(answer_, results=()):
    return lambda *a, **k: ToolRound(answer_, list(results))


@pytest.mark.parametrize("fake, reason", [
    # Server-written outcomes without the tool result that justifies them.
    (SupportAnswer(NEEDS_ORDER_DETAILS_MESSAGE, [], False, NEEDS_ORDER_DETAILS), "agent_error"),
    (SupportAnswer(cancellation_message("ORD-1001"), [STATUS_SOURCE], False, CANCELLATION_AVAILABLE), "agent_error"),
    # A tool source that no result of this question produced.
    (SupportAnswer("In transit.", [STATUS_SOURCE], False, "answered"), "invalid_citations"),
])
def test_gate_rejects_tool_outcomes_without_matching_results(fake, reason):
    result = ask(STATUS_Q, FakeClient(), tool_generator=_round(fake))
    assert (result.answer, result.escalate, result.reason) == (HANDOFF_MESSAGE, True, reason)


def test_gate_rejects_altered_server_wording():
    ctx = context(CANCEL_Q)
    (result,) = tools.execute_all([tools.ToolCall("c", "request_order_cancellation",
                                                  json.dumps({"order_id": "ORD-1001", "email": MAYA}))], ctx)
    altered = SupportAnswer("Done! Order ORD-1001 has been cancelled.", [tool_source(result)], False, CANCELLATION_AVAILABLE)
    assert ask(CANCEL_Q, FakeClient(), tool_generator=_round(altered, [result])).reason == "agent_error"


def test_tool_source_from_an_earlier_question_is_rejected_by_the_gate():
    earlier = tools.execute_all([tools.ToolCall("old", "get_order_status", json.dumps(STATUS_ARGS))], context(STATUS_Q))
    reused = SupportAnswer("In transit.", [tool_source(earlier[0])], False, "answered")
    # This question's round produced no results, so the earlier source is not allowed.
    assert ask(STATUS_Q, FakeClient(), tool_generator=_round(reused, [])).reason == "invalid_citations"


def test_tools_work_without_any_retrieved_article():
    client = FakeClient(
        tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
        final_response("In transit.", ("tool", "call_1:0")),
    )
    result = ask(STATUS_Q, client, retriever=lambda q: [])
    assert result.reason == "answered" and client.calls[0]["documents"] == []


def test_order_book_failure_falls_back_to_articles(monkeypatch):
    monkeypatch.setattr(agent, "_default_order_book", lambda: (_ for _ in ()).throw(OSError("disk")))
    client = FakeClient(final_response("Track it from Orders.", ("document", "KB-003")))
    result = answer_question(STATUS_Q, client=client, retriever=lambda q: ARTICLES, rerank_config=OFF)
    assert result.reason == "answered" and "tools" not in client.calls[0]


def test_tool_generator_crash_or_junk_is_a_safe_hand_off():
    for broken in (lambda *a, **k: 1 / 0, lambda *a, **k: "not a round"):
        result = ask(STATUS_Q, FakeClient(), tool_generator=broken)
        assert (result.answer, result.escalate, result.reason) == (HANDOFF_MESSAGE, True, "agent_error")


def test_logs_never_contain_order_details(caplog):
    caplog.set_level(logging.DEBUG)
    client = FakeClient(
        tool_call_response(("call_1", "get_order_status", STATUS_ARGS)),
        final_response("In transit.", ("tool", "call_1:0")),
    )
    ask(STATUS_Q, client)
    assert "ORD-1003" not in caplog.text and "@" not in caplog.text


def test_api_serialises_tool_sources_without_a_link(monkeypatch):
    """The existing response model accepts a tool source (empty url); no API change."""
    from fastapi.testclient import TestClient

    from support_agent import api
    from support_agent.rate_limit import RateLimiter

    monkeypatch.setattr(api, "chat_limiter", RateLimiter(per_minute=10, per_day=10))
    monkeypatch.setattr(agent, "answer_question",
                        lambda q, **k: SupportAnswer("In transit.", [STATUS_SOURCE], False, "answered"))
    body = TestClient(api.app).post("/chat", json={"question": STATUS_Q}).json()
    assert body["sources"] == [STATUS_SOURCE] and body["reason"] == "answered"
