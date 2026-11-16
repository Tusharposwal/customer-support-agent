"""Offline tests for the optional rerank step in answer_question.
No Cohere import, no API calls."""

import logging
from types import SimpleNamespace

import httpx
import pytest

from support_agent.agent import answer_question
from support_agent.answer import HANDOFF_MESSAGE, SupportAnswer
from support_agent.rerank import RerankConfig

ARTICLES = [
    {"id": "KB-001", "title": "Reset your password", "text": "Select Forgot password.",
     "url": "https://x/reset", "score": 0.032},
    {"id": "KB-002", "title": "Refund eligibility", "text": "Refunds within 30 days.",
     "url": "https://x/refunds", "score": 0.031},
    {"id": "KB-003", "title": "Track an order", "text": "Open Orders to track.",
     "url": "https://x/track", "score": 0.016},
]
QUESTION = "How do I reset my password?"
SECRET = "sk-secret-123"
ENABLED = RerankConfig(enabled=True)


def response(*pairs):
    return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in pairs])


class FakeClient:
    """Rerank and chat; the chat part fails the test if reached unexpectedly."""

    def __init__(self, rerank_result=None, rerank_error=None, chat_response=None):
        self.rerank_result = rerank_result
        self.rerank_error = rerank_error
        self.chat_response = chat_response
        self.rerank_calls = []
        self.chat_calls = []

    def rerank(self, **kwargs):
        self.rerank_calls.append(kwargs)
        if self.rerank_error:
            raise self.rerank_error
        return self.rerank_result

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.chat_response is None:
            raise AssertionError("chat must not be called in this test")
        return self.chat_response


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, question, articles, client):
        self.calls.append(articles)
        first = articles[0]  # cite a supplied article, as a real answer must
        source = {"id": first["id"], "title": first["title"], "url": first["url"]}
        return SupportAnswer(answer="ok", sources=[source], escalate=False, reason="answered")

    @property
    def ids(self):
        return [a["id"] for a in self.calls[-1]]


def ask(client, config, *, articles=ARTICLES, **kwargs):
    generator = kwargs.pop("generator", RecordingGenerator())
    result = answer_question(
        QUESTION, retriever=lambda q: articles, client=client,
        generator=generator, rerank_config=config, **kwargs,
    )
    return result, generator


def assert_handoff(result, reason):
    assert result.escalate is True
    assert result.reason == reason
    assert result.answer == HANDOFF_MESSAGE
    assert result.sources == []


@pytest.fixture(autouse=True)
def rerank_settings_unset(monkeypatch):
    for name in ("COHERE_RERANK_ENABLED", "COHERE_RERANK_MODEL",
                 "COHERE_RERANK_MIN_SCORE", "COHERE_RERANK_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)


# --- Disabled ----------------------------------------------------------------


def test_disabled_config_never_reranks_and_passes_hybrid_articles_unchanged():
    client = FakeClient(rerank_error=AssertionError("must not rerank"))

    result, generator = ask(client, RerankConfig())

    assert result.reason == "answered"
    assert client.rerank_calls == []
    assert generator.calls == [ARTICLES]
    assert generator.calls[0] is ARTICLES


def test_disabled_by_default_when_no_settings_exist():
    def reranker(*args):
        raise AssertionError("must not rerank")

    result, generator = ask(FakeClient(), None, reranker=reranker)

    assert result.reason == "answered"
    assert generator.calls[0] is ARTICLES


def test_settings_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("COHERE_RERANK_ENABLED", "true")
    monkeypatch.setenv("COHERE_RERANK_MODEL", "rerank-test")
    client = FakeClient(response((2, 0.9), (0, 0.5), (1, 0.1)))

    result, generator = ask(client, None)

    assert client.rerank_calls[0]["model"] == "rerank-test"
    assert generator.ids == ["KB-003", "KB-001", "KB-002"]


@pytest.mark.parametrize("name, value", [
    ("COHERE_RERANK_ENABLED", "ture"),
    ("COHERE_RERANK_MIN_SCORE", "2"),
    ("COHERE_RERANK_TIMEOUT_SECONDS", "0"),
])
def test_invalid_settings_hand_off_before_retrieval(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    retrieved = []

    result = answer_question(
        QUESTION, retriever=lambda q: retrieved.append(q) or ARTICLES,
        client=FakeClient(), generator=RecordingGenerator(),
    )

    assert_handoff(result, "configuration_error")
    assert retrieved == []


# --- Enabled -----------------------------------------------------------------


def test_enabled_passes_reranked_articles_to_the_generator():
    client = FakeClient(response((1, 0.2), (2, 0.95), (0, 0.5)))

    result, generator = ask(client, ENABLED)

    assert result.reason == "answered"
    assert generator.ids == ["KB-003", "KB-001", "KB-002"]
    assert [a["rerank_score"] for a in generator.calls[0]] == [0.95, 0.5, 0.2]
    assert client.rerank_calls[0]["query"] == QUESTION
    assert client.rerank_calls[0]["request_options"] == {"max_retries": 0, "timeout": 5.0}


def test_threshold_passes_only_articles_at_or_above_it():
    client = FakeClient(response((0, 0.7), (1, 0.3), (2, 0.1)))

    result, generator = ask(client, RerankConfig(enabled=True, min_score=0.3))

    assert generator.ids == ["KB-001", "KB-002"]


def test_nothing_above_threshold_hands_off_without_a_chat_call(caplog):
    client = FakeClient(response((0, 0.2), (1, 0.1), (2, 0.05)))
    generator = RecordingGenerator()

    with caplog.at_level(logging.INFO, logger="support_agent.agent"):
        result, _ = ask(client, RerankConfig(enabled=True, min_score=0.5), generator=generator)

    assert_handoff(result, "below_rerank_threshold")
    assert generator.calls == []
    assert client.chat_calls == []
    assert QUESTION not in caplog.text  # no customer text in logs
    assert "KB-001" in caplog.text


def test_no_retrieved_articles_skips_reranking():
    client = FakeClient(rerank_error=AssertionError("must not rerank"))

    result, _ = ask(client, ENABLED, articles=[])

    assert_handoff(result, "no_articles")
    assert client.rerank_calls == []


# --- Failures fail open ------------------------------------------------------


class FakeApiError(Exception):
    def __init__(self, status_code, body):
        super().__init__(f"status_code: {status_code}, body: {body}")
        self.status_code = status_code
        self.body = body


@pytest.mark.parametrize("client", [
    FakeClient(rerank_error=FakeApiError(401, f"invalid api key {SECRET}")),
    FakeClient(rerank_error=FakeApiError(429, f"rate limited {SECRET}")),
    FakeClient(rerank_error=FakeApiError(503, f"unavailable {SECRET}")),
    FakeClient(rerank_error=httpx.ReadTimeout(f"timed out {SECRET}")),
    FakeClient(rerank_result=response((0, 0.9))),                  # malformed: articles missing
    FakeClient(rerank_result=SimpleNamespace(results="garbage")),  # malformed
], ids=["401", "429", "503", "timeout", "missing-articles", "garbage"])
def test_rerank_failure_falls_back_to_hybrid_articles(client, caplog):
    with caplog.at_level(logging.INFO):
        result, generator = ask(client, RerankConfig(enabled=True, min_score=0.5))

    assert result.reason == "answered"          # not converted into a hand-off
    assert generator.calls[0] is ARTICLES       # original hybrid articles and order
    assert len(client.rerank_calls) == 1        # no retries
    assert "Rerank failed, using hybrid order: RerankError" in caplog.text
    assert SECRET not in caplog.text
    assert QUESTION not in caplog.text


def test_unexpected_reranker_exception_also_fails_open():
    def reranker(*args):
        raise KeyError("bug")

    result, generator = ask(FakeClient(), ENABLED, reranker=reranker)

    assert result.reason == "answered"
    assert generator.calls[0] is ARTICLES


def test_empty_result_without_threshold_is_treated_as_a_failure():
    result, generator = ask(FakeClient(), ENABLED, reranker=lambda *args: [])

    assert result.reason == "answered"
    assert generator.calls[0] is ARTICLES


def test_fail_open_still_goes_through_the_citation_check():
    """Fail open keeps the existing safety net: the real citation check."""
    from support_agent.answer import generate_answer

    chat = SimpleNamespace(
        finish_reason="COMPLETE",
        message=SimpleNamespace(content=[SimpleNamespace(type="text", text="Some answer")], citations=[]),
    )
    client = FakeClient(rerank_error=FakeApiError(500, SECRET), chat_response=chat)

    result = answer_question(QUESTION, retriever=lambda q: ARTICLES, client=client,
                             generator=generate_answer, rerank_config=ENABLED)

    assert_handoff(result, "no_citations")
    assert SECRET not in result.answer


# --- Citation compatibility through the agent --------------------------------


def chat_citing(*doc_ids):
    return SimpleNamespace(
        finish_reason="COMPLETE",
        message=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Select Forgot password.")],
            citations=[
                SimpleNamespace(type="TEXT_CONTENT", sources=[SimpleNamespace(type="document", id=i)])
                for i in doc_ids
            ],
        ),
    )


def test_agent_with_reranking_produces_normal_sources():
    from support_agent.answer import generate_answer

    client = FakeClient(response((0, 0.9), (1, 0.6), (2, 0.1)), chat_response=chat_citing("KB-001"))

    result = answer_question(QUESTION, retriever=lambda q: ARTICLES, client=client,
                             generator=generate_answer,
                             rerank_config=RerankConfig(enabled=True, min_score=0.5))

    assert result.reason == "answered"
    assert result.sources == [{"id": "KB-001", "title": "Reset your password", "url": "https://x/reset"}]
    assert [d["id"] for d in client.chat_calls[0]["documents"]] == ["KB-001", "KB-002"]


def test_agent_rejects_citation_of_an_article_the_threshold_removed():
    from support_agent.answer import generate_answer

    client = FakeClient(response((0, 0.9), (1, 0.2), (2, 0.1)), chat_response=chat_citing("KB-003"))

    result = answer_question(QUESTION, retriever=lambda q: ARTICLES, client=client,
                             generator=generate_answer,
                             rerank_config=RerankConfig(enabled=True, min_score=0.5))

    assert_handoff(result, "invalid_citations")
