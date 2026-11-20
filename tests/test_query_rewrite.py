"""Offline tests for follow-up rewriting (query_rewrite) and its use in the
agent. Every Cohere client here is a fake: no network, no API key needed."""

import json
import logging
from types import SimpleNamespace

import pytest

from support_agent import agent, query_rewrite
from support_agent.agent import answer_question
from support_agent.answer import CLARIFICATION_MESSAGE, HANDOFF_MESSAGE, SupportAnswer, user_message
from support_agent.query_rewrite import (
    FAILED,
    NEEDS_CLARIFICATION,
    NOT_NEEDED,
    REWRITTEN,
    UNCHANGED,
    Rewrite,
    rewrite_question,
)
from support_agent.rerank import RerankConfig

HISTORY = [
    {"role": "customer", "content": "How can I return my order?"},
    {"role": "assistant", "content": "You can return items within 30 days of purchase."},
]
FOLLOW_UP = "What if I bought it 35 days ago?"


def chat_response(text, finish_reason="COMPLETE"):
    return SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], citations=[]),
    )


def output(query, needs_clarification=False):
    return json.dumps({"standalone_query": query, "needs_clarification": needs_clarification})


class FakeClient:
    """Fake ClientV2: each chat() call pops the next outcome (response or exception)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in ("COHERE_REWRITE_MODEL", "COHERE_REWRITE_TIMEOUT_SECONDS", "COHERE_CHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)


# --- Settings ----------------------------------------------------------------------


def test_default_settings():
    assert query_rewrite.rewrite_timeout({}) == 8.0
    assert query_rewrite.rewrite_model({}) == "command-a-03-2025"


def test_model_falls_back_to_the_chat_model():
    assert query_rewrite.rewrite_model({"COHERE_CHAT_MODEL": "chat-x"}) == "chat-x"
    assert query_rewrite.rewrite_model({"COHERE_CHAT_MODEL": "chat-x", "COHERE_REWRITE_MODEL": "rw"}) == "rw"


@pytest.mark.parametrize("value", ["-7", "abc", "nan", "999"])
def test_invalid_timeout_is_rejected_without_its_value(value):
    with pytest.raises(ValueError) as raised:
        query_rewrite.rewrite_timeout({"COHERE_REWRITE_TIMEOUT_SECONDS": value})
    assert "COHERE_REWRITE_TIMEOUT_SECONDS" in str(raised.value) and value not in str(raised.value)


# --- No history: no call -------------------------------------------------------------


@pytest.mark.parametrize("history", [None, []])
def test_first_message_makes_no_call(history):
    client = FakeClient()
    assert rewrite_question("Hi", history, client) == Rewrite("Hi", NOT_NEEDED)
    assert client.calls == []


# --- The request ----------------------------------------------------------------------


def test_request_shape_and_policy(monkeypatch):
    monkeypatch.setenv("COHERE_REWRITE_MODEL", "rewrite-model")
    monkeypatch.setenv("COHERE_REWRITE_TIMEOUT_SECONDS", "6")
    client = FakeClient(chat_response(output("Can I return an order 35 days after purchase?")))

    rewrite_question(FOLLOW_UP, HISTORY, client)

    (call,) = client.calls
    assert call["model"] == "rewrite-model"
    assert call["temperature"] == 0 and call["max_tokens"] == query_rewrite.MAX_TOKENS
    assert call["response_format"]["type"] == "json_object"
    assert call["request_options"] == {"timeout": 6.0, "max_retries": 0}  # shared retry policy
    assert "documents" not in call  # no KB evidence goes to the rewriter
    system, user = call["messages"]
    assert system == {"role": "system", "content": query_rewrite.SYSTEM_PROMPT}
    assert "Customer: How can I return my order?" in user["content"]
    assert "Assistant (context only, may be wrong): You can return" in user["content"]
    assert user["content"].endswith(f"Customer's latest message: {FOLLOW_UP}")


def test_temporary_failure_is_retried_once():
    class ServerError(Exception):
        status_code = 503

    client = FakeClient(ServerError(), chat_response(output("Can I return an order 35 days after purchase?")))
    assert rewrite_question(FOLLOW_UP, HISTORY, client).outcome == REWRITTEN
    assert len(client.calls) == 2


def test_excerpt_is_bounded_and_keeps_the_latest_messages():
    history = [{"role": "customer", "content": f"message {n} " + "x" * 900} for n in range(20)]
    excerpt = query_rewrite.conversation_excerpt(history)
    lines = excerpt.splitlines()
    assert len(lines) <= query_rewrite.HISTORY_MESSAGES
    assert len(excerpt) <= query_rewrite.HISTORY_CHARS + len(lines)
    assert all(len(line) <= len("Customer: ") + query_rewrite.MESSAGE_CHARS for line in lines)
    assert lines[-1].startswith("Customer: message 19 ")


def test_excerpt_does_not_change_the_history():
    history = [dict(m) for m in HISTORY]
    query_rewrite.conversation_excerpt(history)
    assert history == HISTORY


# --- Checking the output ---------------------------------------------------------------


def test_rewritten_question():
    client = FakeClient(chat_response(output("  Can I return an order   35 days after purchase? ")))
    assert rewrite_question(FOLLOW_UP, HISTORY, client) == Rewrite(
        "Can I return an order 35 days after purchase?", REWRITTEN
    )


def test_standalone_message_is_unchanged():
    question = "How much is express shipping?"
    client = FakeClient(chat_response(output("how much is express shipping?")))
    assert rewrite_question(question, HISTORY, client) == Rewrite(question, UNCHANGED)


def test_needs_clarification():
    client = FakeClient(chat_response(output("", needs_clarification=True)))
    assert rewrite_question("Can I do that too?", HISTORY, client) == Rewrite("Can I do that too?", NEEDS_CLARIFICATION)


@pytest.mark.parametrize("text", [
    "not json",
    "[]",
    json.dumps({"standalone_query": "x"}),
    json.dumps({"standalone_query": 5, "needs_clarification": False}),
    json.dumps({"standalone_query": "x", "needs_clarification": "no"}),
    output("   "),
    output("q" * (query_rewrite.MAX_QUERY_CHARS + 1)),
    output("Can I return an order 30 days after purchase?"),  # 30 came from the assistant, not the customer
    output("Can I get a refund of $49 after 35 days?"),  # invented amount
])
def test_unusable_output_falls_back_to_the_message(text, caplog):
    caplog.set_level(logging.WARNING)
    client = FakeClient(chat_response(text))
    assert rewrite_question(FOLLOW_UP, HISTORY, client) == Rewrite(FOLLOW_UP, FAILED)
    assert text not in caplog.text or not text.strip()  # model output is never logged


@pytest.mark.parametrize("finish_reason", ["MAX_TOKENS", "ERROR", None])
def test_incomplete_generation_falls_back(finish_reason):
    client = FakeClient(chat_response(output("Can I return it?"), finish_reason=finish_reason))
    assert rewrite_question(FOLLOW_UP, HISTORY, client).outcome == FAILED


def test_numbers_from_earlier_customer_messages_are_allowed():
    history = [{"role": "customer", "content": "I ordered 2 lamps on order 12345"}]
    client = FakeClient(chat_response(output("Can I return 2 lamps from order 12345?")))
    assert rewrite_question("Can I return them?", history, client).outcome == REWRITTEN


@pytest.mark.parametrize("error", [RuntimeError("SECRET provider text"), TimeoutError()])
def test_api_failure_falls_back_and_logs_no_details(error, caplog):
    caplog.set_level(logging.WARNING)
    assert rewrite_question(FOLLOW_UP, HISTORY, FakeClient(error)) == Rewrite(FOLLOW_UP, FAILED)
    assert "SECRET" not in caplog.text and FOLLOW_UP not in caplog.text


def test_invalid_timeout_setting_falls_back(monkeypatch):
    monkeypatch.setenv("COHERE_REWRITE_TIMEOUT_SECONDS", "never")
    client = FakeClient()
    assert rewrite_question(FOLLOW_UP, HISTORY, client).outcome == FAILED
    assert client.calls == []


# --- The answer generator's user message ----------------------------------------------


def test_user_message_is_unchanged_without_a_rewrite():
    assert user_message("How do I reset my password?") == "How do I reset my password?"
    assert user_message("Same", "Same") == "Same"


def test_user_message_with_a_rewrite_keeps_the_original_and_labels_the_rewrite():
    text = user_message(FOLLOW_UP, "Can I return an order 35 days after purchase?")
    assert text.startswith(f"Customer's latest message: {FOLLOW_UP}")
    assert "not a source of facts" in text and "35 days after purchase" in text


# --- The agent --------------------------------------------------------------------


ARTICLES = [
    {"id": "KB-002", "title": "Refund policy", "text": "Refunds within 30 days of purchase.",
     "url": "https://support.example.com/refunds"},
]
SOURCE = {"id": "KB-002", "title": "Refund policy", "url": "https://support.example.com/refunds"}
OFF = RerankConfig()


class Recorder:
    def __init__(self, rewrite=None, result=None):
        self.rewrite = rewrite
        self.result = result or SupportAnswer(answer="Within 30 days.", sources=[SOURCE])
        self.rewriter_calls, self.retrieved, self.generated = [], [], []

    def rewriter(self, question, history, client):
        self.rewriter_calls.append((question, history))
        return self.rewrite

    def retriever(self, query):
        self.retrieved.append(query)
        return ARTICLES

    def generator(self, question, articles, client, **kwargs):
        self.generated.append((question, kwargs))
        return self.result


def run(recorder, question=FOLLOW_UP, history=HISTORY, trace=None, **kwargs):
    kwargs.setdefault("rewriter", recorder.rewriter)
    kwargs.setdefault("rerank_config", OFF)
    return answer_question(
        question, history=history, trace=trace, retriever=recorder.retriever,
        generator=recorder.generator, client=object(), **kwargs,
    )


def test_first_message_pipeline_is_unchanged_and_makes_no_rewrite_call():
    """No history: the real rewriter returns at once (no client call) and the
    generator is called with exactly (question, articles, client), as before."""
    client = FakeClient()  # would fail if chat() were called
    recorder = Recorder()
    trace = {}

    result = answer_question(
        "How do I get a refund?", trace=trace, retriever=recorder.retriever,
        generator=recorder.generator, client=client, rerank_config=OFF,
    )

    assert result.reason == "answered"
    assert client.calls == []
    assert recorder.retrieved == ["How do I get a refund?"]
    assert recorder.generated == [("How do I get a refund?", {})]
    assert trace == {"rewrite": NOT_NEEDED, "retrieval_query": "How do I get a refund?"}


def test_follow_up_retrieves_with_the_rewrite_and_answers_the_original():
    standalone = "Can I return an order 35 days after purchase?"
    recorder = Recorder(rewrite=Rewrite(standalone, REWRITTEN))
    trace = {}

    result = run(recorder, trace=trace)

    assert recorder.rewriter_calls == [(FOLLOW_UP, HISTORY)]
    assert recorder.retrieved == [standalone]
    # The generator gets the original message plus the standalone query, never the history.
    assert recorder.generated == [(FOLLOW_UP, {"standalone_query": standalone})]
    assert result.sources == [SOURCE]
    assert trace == {"rewrite": REWRITTEN, "retrieval_query": standalone}


def test_rerank_uses_the_rewritten_query():
    standalone = "Can I return an order 35 days after purchase?"
    seen = []

    def reranker(query, articles, client, config):
        seen.append(query)
        return articles

    run(Recorder(rewrite=Rewrite(standalone, REWRITTEN)), reranker=reranker, rerank_config=RerankConfig(enabled=True))
    assert seen == [standalone]


@pytest.mark.parametrize("outcome", [UNCHANGED, FAILED])
def test_unchanged_or_failed_rewrite_uses_the_message(outcome):
    recorder = Recorder(rewrite=Rewrite(FOLLOW_UP, outcome))
    trace = {}
    run(recorder, trace=trace)
    assert recorder.retrieved == [FOLLOW_UP]
    assert recorder.generated == [(FOLLOW_UP, {})]
    assert trace["rewrite"] == outcome


def test_rewriter_crash_or_junk_falls_back(caplog):
    def crash(*args):
        raise RuntimeError("boom")

    for rewriter in (crash, lambda *a: "not a Rewrite", lambda *a: Rewrite("  ", REWRITTEN)):
        recorder = Recorder()
        trace = {}
        result = run(recorder, trace=trace, rewriter=rewriter)
        assert result.reason == "answered"
        assert recorder.retrieved == [FOLLOW_UP] and trace["rewrite"] == FAILED


def test_clarification_skips_retrieval_and_generation():
    recorder = Recorder(rewrite=Rewrite("Can I do that?", NEEDS_CLARIFICATION))
    trace = {}

    result = run(recorder, question="Can I do that?", trace=trace)

    assert result == SupportAnswer(
        answer=CLARIFICATION_MESSAGE, sources=[], escalate=False, reason="needs_clarification"
    )
    assert recorder.retrieved == [] and recorder.generated == []
    assert trace == {"rewrite": NEEDS_CLARIFICATION}


@pytest.mark.parametrize("fake", [
    SupportAnswer(answer=CLARIFICATION_MESSAGE, sources=[], escalate=False, reason="needs_clarification"),
    SupportAnswer(answer="Anything", sources=[], escalate=False, reason="needs_clarification"),
    SupportAnswer(answer=CLARIFICATION_MESSAGE, sources=[SOURCE], escalate=False, reason="needs_clarification"),
])
def test_generator_can_never_return_a_clarification(fake):
    """The gate accepts a clarification only from the agent itself, so a
    generator can't use the reason to get an uncited answer through."""
    result = run(Recorder(rewrite=Rewrite(FOLLOW_UP, UNCHANGED), result=fake))
    assert (result.answer, result.escalate, result.reason) == (HANDOFF_MESSAGE, True, "agent_error")


@pytest.mark.parametrize("fake", [
    SupportAnswer(answer=CLARIFICATION_MESSAGE, sources=[], escalate=False, reason="needs_clarification"),
    SupportAnswer(answer="Other text", sources=[], escalate=False, reason="needs_clarification"),
    SupportAnswer(answer=CLARIFICATION_MESSAGE, sources=[SOURCE], escalate=False, reason="needs_clarification"),
])
def test_gate_only_accepts_the_exact_clarification(fake):
    allowed = agent._checked_result(fake, [], clarification_allowed=True)
    if fake.answer == CLARIFICATION_MESSAGE and not fake.sources:
        assert allowed.reason == "needs_clarification" and allowed.escalate is False
    else:
        assert allowed.reason == "agent_error"
    assert agent._checked_result(fake, ARTICLES).reason == "agent_error"


def test_answers_after_a_rewrite_still_need_valid_citations():
    bad = SupportAnswer(answer="Yes.", sources=[{"id": "KB-999", "title": "x", "url": "y"}])
    recorder = Recorder(rewrite=Rewrite("standalone question", REWRITTEN), result=bad)
    assert run(recorder).reason == "invalid_citations"


def test_real_generator_receives_original_and_standalone(monkeypatch):
    """End to end with fakes: the history never reaches the answer call, and
    the rewrite call (with the history) never carries documents."""
    standalone = "Can I return an order 35 days after purchase?"
    answer = chat_response("Within 30 days.")
    answer.message.citations = [SimpleNamespace(type="TEXT_CONTENT", sources=[SimpleNamespace(type="document", id="KB-002")])]
    client = FakeClient(chat_response(output(standalone)), answer)

    result = answer_question(
        FOLLOW_UP, history=HISTORY, retriever=lambda q: ARTICLES, client=client, rerank_config=OFF,
    )

    assert result.reason == "answered" and result.sources == [SOURCE]
    rewrite_call, answer_call = client.calls
    assert "documents" not in rewrite_call
    assert answer_call["documents"][0]["id"] == "KB-002"
    user_text = answer_call["messages"][1]["content"]
    assert FOLLOW_UP in user_text and standalone in user_text
    assert HISTORY[1]["content"] not in user_text and HISTORY[0]["content"] not in user_text
