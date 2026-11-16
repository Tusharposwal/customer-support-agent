"""Offline tests for Cohere timeouts, retries and error handling (cohere_calls).

Errors are the installed SDK's real exception classes (and httpx's), raised by
fake clients, so classification matches what a real failure looks like. No
request ever leaves the machine.
"""

import logging
from types import SimpleNamespace

import httpx
import pytest
from cohere.core.api_error import ApiError
from cohere.core.parse_error import ParsingError
from cohere.errors import (
    BadRequestError,
    ForbiddenError,
    GatewayTimeoutError,
    InternalServerError,
    InvalidTokenError,
    NotFoundError,
    ServiceUnavailableError,
    TooManyRequestsError,
    UnauthorizedError,
    UnprocessableEntityError,
)

from support_agent import cohere_calls
from support_agent.agent import answer_question
from support_agent.answer import HANDOFF_MESSAGE, SupportAnswer, generate_answer
from support_agent.cohere_calls import (
    CohereCallError,
    call_with_retries,
    chat_timeout,
    classify_error,
    embed_timeout,
)
from support_agent.rerank import RerankConfig

SECRET = "sk-live-SECRET-123"
BODY = {"message": f"invalid api key {SECRET}; request was: my card number 4111"}

ARTICLES = [
    {"id": "KB-001", "title": "Reset your password", "text": "Select Forgot password.", "url": "https://x/reset"},
]
QUESTION = "How do I reset my password?"


def timeout_error():
    return httpx.ReadTimeout(f"timed out {SECRET}")


def connect_error():
    """The connection could not be opened: the request was never sent."""
    return httpx.ConnectError(f"connection refused {SECRET}")


def connection_lost():
    """An open connection failed: Cohere may have processed the request."""
    return httpx.ReadError(f"connection reset {SECRET}")


def rate_limit(retry_after=None):
    return TooManyRequestsError(body=BODY, headers=None if retry_after is None else {"Retry-After": retry_after})


def server_error():
    return ServiceUnavailableError(body=BODY)


def auth_error():
    return UnauthorizedError(body=BODY)


class Clock:
    """A fake clock; each request can advance it to simulate a slow failure."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(cohere_calls, "clock", fake)
    return fake


class Script:
    """`request(options)` that plays a list of outcomes: an exception to raise,
    a number of seconds to spend before failing with a server error, or a
    response to return."""

    def __init__(self, *outcomes, clock=None, seconds_per_call=0.0):
        self.outcomes = list(outcomes)
        self.options = []
        self.clock = clock
        self.seconds_per_call = seconds_per_call

    def __call__(self, options):
        self.options.append(dict(options))
        if self.clock is not None:
            self.clock.now += self.seconds_per_call
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


# --- Classification ----------------------------------------------------------


@pytest.mark.parametrize(
    "error, category",
    [
        (TooManyRequestsError(body=BODY), "rate_limit"),
        (UnauthorizedError(body=BODY), "auth_error"),
        (ForbiddenError(body=BODY), "auth_error"),
        (InvalidTokenError(body=BODY), "auth_error"),
        (BadRequestError(body=BODY), "invalid_request"),
        (NotFoundError(body=BODY), "invalid_request"),
        (UnprocessableEntityError(body=BODY), "invalid_request"),
        (InternalServerError(body=BODY), "server_error"),
        (ApiError(status_code=502, body=BODY), "server_error"),
        (ServiceUnavailableError(body=BODY), "server_error"),
        (GatewayTimeoutError(body=BODY), "server_error"),
        (ApiError(status_code=501, body=BODY), "unknown"),
        (ApiError(status_code=408, body=BODY), "timeout"),
        (ApiError(status_code=200, body="<html>"), "malformed_response"),
        (ParsingError(status_code=200, body={}), "malformed_response"),
        (httpx.ReadTimeout("x"), "timeout"),
        (httpx.WriteTimeout("x"), "timeout"),
        (httpx.PoolTimeout("x"), "timeout"),
        (TimeoutError(), "timeout"),
        (httpx.ProxyError("x"), "unknown"),
        (httpx.LocalProtocolError("x"), "unknown"),
        (ValueError("x"), "unknown"),
    ],
    ids=lambda value: value if isinstance(value, str) else type(value).__name__,
)
def test_errors_are_classified(error, category):
    assert classify_error(error) == category


# --- Retry eligibility and limits --------------------------------------------


def test_success_needs_one_request_with_sdk_retries_off():
    request = Script("response")

    assert call_with_retries(request, operation="chat", timeout_seconds=25) == "response"
    assert request.options == [{"timeout": 25, "max_retries": 0}]


@pytest.mark.parametrize("first_error", [rate_limit, server_error, connect_error],
                         ids=["rate-limit", "server-error", "connect-error"])
def test_temporary_failure_is_retried_once(first_error, no_real_retry_waits):
    request = Script(first_error(), "response")

    assert call_with_retries(request, operation="chat", timeout_seconds=25) == "response"
    assert len(request.options) == 2
    assert no_real_retry_waits == [cohere_calls.RETRY_WAIT_SECONDS]


@pytest.mark.parametrize(
    "error, category",
    [
        (timeout_error(), "timeout"),
        (auth_error(), "auth_error"),
        (BadRequestError(body=BODY), "invalid_request"),
        (ParsingError(status_code=200, body={}), "malformed_response"),
        (ValueError(SECRET), "unknown"),
    ],
    ids=["timeout", "auth", "invalid-request", "malformed", "unknown"],
)
def test_permanent_failures_and_timeouts_are_not_retried(error, category, no_real_retry_waits):
    request = Script(error, "never reached")

    with pytest.raises(CohereCallError) as raised:
        call_with_retries(request, operation="chat", timeout_seconds=25)

    assert raised.value.category == category
    assert raised.value.attempts == 1
    assert len(request.options) == 1
    assert no_real_retry_waits == []


def test_never_more_than_one_retry():
    request = Script(server_error(), server_error(), server_error(), "never reached")

    with pytest.raises(CohereCallError) as raised:
        call_with_retries(request, operation="embed", timeout_seconds=10)

    assert (raised.value.category, raised.value.attempts) == ("server_error", 2)
    assert len(request.options) == 2


@pytest.mark.parametrize("header, wait", [("1", 1.0), ("0.5", 0.5), (" 2 ", 2.0), ("0", 0.0), ("soon", 1.0)])
def test_short_retry_after_is_respected(header, wait, no_real_retry_waits):
    request = Script(rate_limit(header), "response")

    assert call_with_retries(request, operation="chat", timeout_seconds=25) == "response"
    assert no_real_retry_waits == [wait]


@pytest.mark.parametrize("header", ["3", "60"])
def test_long_retry_after_hands_off_instead_of_waiting(header, no_real_retry_waits):
    request = Script(rate_limit(header), "never reached")

    with pytest.raises(CohereCallError) as raised:
        call_with_retries(request, operation="chat", timeout_seconds=25)

    assert raised.value.category == "rate_limit"
    assert len(request.options) == 1
    assert no_real_retry_waits == []


def test_retry_after_as_a_date_is_treated_as_not_given(no_real_retry_waits):
    request = Script(rate_limit("Wed, 21 Oct 2026 07:28:00 GMT"), "response")

    assert call_with_retries(request, operation="chat", timeout_seconds=25) == "response"
    assert no_real_retry_waits == [cohere_calls.RETRY_WAIT_SECONDS]


# --- Connection failures: never sent (retry) vs maybe processed (no retry) ---

NEVER_SENT = [
    httpx.ConnectError("connection refused"),      # TCP connect or TLS handshake failed
    httpx.ConnectTimeout("connect timed out"),     # no connection within the timeout
    ConnectionRefusedError(),
]
MAYBE_PROCESSED = [
    httpx.ReadError("connection reset"),           # seen with and without a received request
    httpx.WriteError("broken pipe"),
    httpx.CloseError("close failed"),
    httpx.RemoteProtocolError("Server disconnected without sending a response."),
    ConnectionResetError(),
    ConnectionAbortedError(),
    BrokenPipeError(),
]


def _name(error):
    return type(error).__name__


@pytest.mark.parametrize("error", NEVER_SENT, ids=_name)
def test_failure_before_sending_is_a_connect_error(error):
    assert classify_error(error) == "connect_error"


@pytest.mark.parametrize("error", MAYBE_PROCESSED, ids=_name)
def test_failure_on_an_open_connection_is_connection_lost(error):
    assert classify_error(error) == "connection_lost"


@pytest.mark.parametrize("error", NEVER_SENT, ids=_name)
def test_request_that_was_never_sent_is_retried_once(error, no_real_retry_waits):
    request = Script(error, "response")

    assert call_with_retries(request, operation="chat", timeout_seconds=25) == "response"
    assert len(request.options) == 2
    assert no_real_retry_waits == [cohere_calls.RETRY_WAIT_SECONDS]


@pytest.mark.parametrize("error", MAYBE_PROCESSED, ids=_name)
def test_request_that_may_have_been_processed_is_not_retried(error, no_real_retry_waits):
    request = Script(error, "never reached")

    with pytest.raises(CohereCallError) as raised:
        call_with_retries(request, operation="chat", timeout_seconds=25)

    assert (raised.value.category, raised.value.attempts) == ("connection_lost", 1)
    assert len(request.options) == 1
    assert no_real_retry_waits == []


@pytest.mark.parametrize(
    "error, transport_attempts",
    [(error, 2) for error in NEVER_SENT[:2]] + [(error, 1) for error in MAYBE_PROCESSED[:4]],
    ids=_name,
)
def test_real_sdk_client_sends_the_request_at_most_as_often_as_the_policy_allows(error, transport_attempts):
    """The installed cohere.ClientV2 over an in-memory httpx transport (no
    sockets): the SDK passes the httpx error through unchanged, and its own
    hidden retry (it normally repeats RemoteProtocolError) stays off."""
    import cohere

    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        raise error

    client = cohere.ClientV2(api_key="fake-key", httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert_handoff(generate_answer(QUESTION, ARTICLES, client), "llm_error")
    assert attempts == ["/v2/chat"] * transport_attempts


# --- Total time stays within the budget --------------------------------------


def test_retry_gets_only_the_time_left_in_the_budget(clock):
    request = Script(server_error(), "response", clock=clock, seconds_per_call=3.0)

    call_with_retries(request, operation="chat", timeout_seconds=25)

    # 3 s for the failed attempt; the wait is faked, so 25 - 3 = 22 s remain.
    assert [options["timeout"] for options in request.options] == [25, 22.0]


def test_slow_failure_is_not_retried(clock):
    """After 13 s and a 1 s wait, less than half of a 25 s budget is left."""
    request = Script(server_error(), "never reached", clock=clock, seconds_per_call=13.0)

    with pytest.raises(CohereCallError):
        call_with_retries(request, operation="chat", timeout_seconds=25)

    assert len(request.options) == 1


def test_worst_case_never_exceeds_the_budget(clock, monkeypatch):
    """Every attempt uses its whole timeout and the waits really advance the clock."""
    monkeypatch.setattr(cohere_calls, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))

    def request(options):
        clock.now += options["timeout"] if len(calls) else 4.0  # first: a fast 503 after 4 s
        calls.append(options)
        raise server_error()

    calls = []
    start = clock.now
    with pytest.raises(CohereCallError):
        call_with_retries(request, operation="chat", timeout_seconds=25)

    assert len(calls) == 2
    assert clock.now - start == pytest.approx(25)


# --- Nothing secret escapes --------------------------------------------------


def test_error_and_logs_contain_no_provider_text(caplog):
    caplog.set_level(logging.DEBUG)
    request = Script(auth_error())

    with pytest.raises(CohereCallError) as raised:
        call_with_retries(request, operation="chat", timeout_seconds=25)

    error = raised.value
    assert str(error) == "Cohere chat request failed: auth_error (attempts: 1)"
    assert error.__cause__ is None and error.__suppress_context__  # provider error not chained
    assert SECRET not in caplog.text and "4111" not in caplog.text
    assert "auth_error" in caplog.text and "UnauthorizedError" in caplog.text


def test_authentication_failure_is_logged_as_an_error(caplog):
    with pytest.raises(CohereCallError):
        call_with_retries(Script(auth_error()), operation="chat", timeout_seconds=25)

    assert [record.levelname for record in caplog.records] == ["ERROR"]


# --- Configuration -----------------------------------------------------------


def test_default_timeouts():
    assert chat_timeout({}) == 25.0
    assert embed_timeout({}) == 10.0
    assert chat_timeout({"COHERE_CHAT_TIMEOUT_SECONDS": "  "}) == 25.0


def test_timeouts_can_be_configured():
    environ = {"COHERE_CHAT_TIMEOUT_SECONDS": "12.5", "COHERE_EMBED_TIMEOUT_SECONDS": " 4 "}

    assert (chat_timeout(environ), embed_timeout(environ)) == (12.5, 4.0)


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "inf", "nan", "121"])
def test_invalid_timeout_is_rejected(raw):
    with pytest.raises(ValueError, match="COHERE_CHAT_TIMEOUT_SECONDS"):
        chat_timeout({"COHERE_CHAT_TIMEOUT_SECONDS": raw})


# --- Chat: through generate_answer and the agent -----------------------------


def chat_response():
    return SimpleNamespace(
        finish_reason="COMPLETE",
        message=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Select Forgot password.")],
            citations=[SimpleNamespace(type="TEXT_CONTENT", sources=[SimpleNamespace(type="document", id="KB-001")])],
        ),
    )


class ChatClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def assert_handoff(result, reason):
    assert result == SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason=reason)


def test_chat_request_has_explicit_timeout_and_no_sdk_retries(monkeypatch):
    monkeypatch.setenv("COHERE_CHAT_TIMEOUT_SECONDS", "8")
    client = ChatClient(chat_response())

    assert generate_answer(QUESTION, ARTICLES, client).reason == "answered"
    assert client.calls[0]["request_options"] == {"timeout": 8.0, "max_retries": 0}


@pytest.mark.parametrize("first_error", [rate_limit, server_error, connect_error],
                         ids=["rate-limit", "server-error", "connect-error"])
def test_chat_recovers_from_one_temporary_failure(first_error):
    client = ChatClient(first_error(), chat_response())

    result = generate_answer(QUESTION, ARTICLES, client)

    assert result.reason == "answered"
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "outcomes, attempts",
    [
        ((timeout_error(),), 1),
        ((auth_error(),), 1),
        ((BadRequestError(body=BODY),), 1),
        ((rate_limit(), rate_limit()), 2),
        ((server_error(), server_error()), 2),
        ((connect_error(), connect_error()), 2),
        ((connection_lost(),), 1),
        ((ParsingError(status_code=200, body={}),), 1),
    ],
    ids=["timeout", "auth", "invalid-request", "rate-limit", "server-error", "connect-error",
         "connection-lost", "malformed"],
)
def test_chat_failure_is_a_safe_llm_error_hand_off(outcomes, attempts, caplog):
    caplog.set_level(logging.DEBUG)
    client = ChatClient(*outcomes)

    result = generate_answer(QUESTION, ARTICLES, client)

    assert_handoff(result, "llm_error")
    assert len(client.calls) == attempts
    assert SECRET not in caplog.text and "4111" not in caplog.text


def test_agent_chat_failure_is_safe_end_to_end(caplog):
    caplog.set_level(logging.DEBUG)
    client = ChatClient(auth_error())

    result = answer_question(QUESTION, retriever=lambda q: ARTICLES, client=client, rerank_config=RerankConfig())

    assert_handoff(result, "llm_error")
    assert SECRET not in result.answer and SECRET not in caplog.text


@pytest.mark.parametrize("name", ["COHERE_CHAT_TIMEOUT_SECONDS", "COHERE_EMBED_TIMEOUT_SECONDS"])
def test_invalid_timeout_setting_is_a_configuration_error(name, monkeypatch):
    monkeypatch.setenv(name, "fast")
    client = ChatClient()

    def retriever(question):
        raise AssertionError("must not retrieve")

    result = answer_question(QUESTION, retriever=retriever, client=client, rerank_config=RerankConfig())

    assert_handoff(result, "configuration_error")
    assert client.calls == []


def test_embedding_failure_is_a_retrieval_error_and_logs_the_category(caplog):
    def retriever(question):
        raise CohereCallError("embed", "rate_limit", attempts=2)

    result = answer_question(QUESTION, retriever=retriever, client=ChatClient(), rerank_config=RerankConfig())

    assert_handoff(result, "retrieval_error")
    assert "rate_limit (embed)" in caplog.text


# --- Embeddings: semantic_kb with a fake Cohere SDK --------------------------
# The `semantic_kb` fixture (fake `cohere`, no-op `dotenv`) is in conftest.py.


def test_embed_requests_have_explicit_timeout_and_no_sdk_retries(semantic_kb, monkeypatch):
    monkeypatch.setenv("COHERE_EMBED_TIMEOUT_SECONDS", "4")

    assert semantic_kb.retrieve_semantic(QUESTION)

    calls = semantic_kb.client.calls
    assert [call["input_type"] for call in calls] == ["search_query", "search_document"]
    assert all(call["request_options"] == {"timeout": 4.0, "max_retries": 0} for call in calls)


def test_query_embedding_recovers_from_one_temporary_failure(semantic_kb):
    semantic_kb.client.outcomes = [server_error(), "ok", "ok"]

    assert semantic_kb.retrieve_semantic(QUESTION)
    assert len(semantic_kb.client.calls) == 3  # failed query, retried query, documents


@pytest.mark.parametrize(
    "outcomes, category",
    [
        ([timeout_error()], "timeout"),
        ([auth_error()], "auth_error"),
        ([rate_limit(), rate_limit()], "rate_limit"),
        ([connect_error(), connect_error()], "connect_error"),
        ([httpx.RemoteProtocolError(SECRET)], "connection_lost"),
        ([SimpleNamespace()], "malformed_response"),
        ([SimpleNamespace(embeddings=SimpleNamespace(float=[]))], "malformed_response"),
        ([SimpleNamespace(embeddings=SimpleNamespace(float=[None]))], "malformed_response"),
    ],
    ids=["timeout", "auth", "rate-limit", "connect-error", "connection-lost", "no-embeddings", "too-few",
         "not-a-vector"],
)
def test_query_embedding_failures_raise_a_safe_error(semantic_kb, outcomes, category, caplog):
    semantic_kb.client.outcomes = list(outcomes)

    with pytest.raises(CohereCallError) as raised:
        semantic_kb.retrieve_semantic(QUESTION)

    assert raised.value.category == category
    assert SECRET not in str(raised.value) and SECRET not in caplog.text


def test_short_document_embeddings_are_rejected_and_not_cached(semantic_kb):
    """Fewer vectors than articles used to drop articles silently (zip), and the
    result was cached for the life of the process."""
    short = SimpleNamespace(embeddings=SimpleNamespace(float=[[1.0, 0.0]]))
    semantic_kb.client.outcomes = ["ok", short]  # query fine, documents short

    with pytest.raises(CohereCallError, match="malformed_response"):
        semantic_kb.retrieve_semantic(QUESTION)

    # The next question tries the document embedding again and succeeds.
    assert semantic_kb.retrieve_semantic(QUESTION)
    assert [call["input_type"] for call in semantic_kb.client.calls] == [
        "search_query", "search_document", "search_query", "search_document",
    ]
