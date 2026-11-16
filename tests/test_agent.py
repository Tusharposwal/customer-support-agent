"""Offline tests for the support agent workflow. No Cohere API calls are made."""

import subprocess
import sys

import cohere
import pytest
from cohere import (
    AssistantMessageResponse,
    Citation,
    DocumentSource,
    TextAssistantMessageResponseContentItem,
    V2ChatResponse,
)

from support_agent import agent
from support_agent.agent import answer_question, main
from support_agent.answer import HANDOFF_MESSAGE, INSUFFICIENT_EVIDENCE, SupportAnswer

ARTICLES = [
    {
        "id": "KB-001",
        "title": "Reset your password",
        "text": "Open the sign-in page and select Forgot password.",
        "url": "https://support.example.com/password-reset",
        "score": 0.03,
    },
    {
        "id": "KB-002",
        "title": "Refund eligibility",
        "text": "Refund requests are accepted within 30 days of purchase.",
        "url": "https://support.example.com/refunds",
        "score": 0.01,
    },
]

QUESTION = "How do I reset my password?"
ANSWER_TEXT = "Open the sign-in page and select Forgot password."


def make_response(text=ANSWER_TEXT, cited_ids=("KB-001",)):
    """Fake chat response built from the installed SDK's real classes."""
    citations = [
        Citation(
            start=0,
            end=len(text),
            text=text,
            sources=[DocumentSource(type="document", id=doc_id, document={"id": doc_id})],
            type="TEXT_CONTENT",
        )
        for doc_id in cited_ids
    ]
    return V2ChatResponse(
        id="fake-response",
        finish_reason="COMPLETE",
        message=AssistantMessageResponse(
            role="assistant",
            content=[TextAssistantMessageResponseContentItem(type="text", text=text)],
            citations=citations or None,
        ),
    )


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeRetriever:
    def __init__(self, articles=None, error=None):
        self.articles = ARTICLES if articles is None else articles
        self.error = error
        self.questions = []

    def __call__(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.articles


@pytest.fixture(autouse=True)
def reset_default_client():
    agent._default_client.cache_clear()
    yield
    agent._default_client.cache_clear()


@pytest.fixture(autouse=True)
def rerank_settings_unset(monkeypatch):
    """These tests cover the default (reranking off) flow; see test_agent_rerank.py."""
    for name in (
        "COHERE_RERANK_ENABLED",
        "COHERE_RERANK_MODEL",
        "COHERE_RERANK_MIN_SCORE",
        "COHERE_RERANK_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def assert_handoff(result, reason):
    assert result.escalate is True
    assert result.reason == reason
    assert result.answer == HANDOFF_MESSAGE
    assert result.sources == []


def test_successful_retrieval_and_grounded_answer():
    retriever = FakeRetriever()
    client = FakeClient(make_response())

    result = answer_question(QUESTION, retriever=retriever, client=client)

    assert result.escalate is False
    assert result.reason == "answered"
    assert result.answer == ANSWER_TEXT
    assert result.sources == [
        {
            "id": "KB-001",
            "title": "Reset your password",
            "url": "https://support.example.com/password-reset",
        }
    ]
    assert retriever.questions == [QUESTION]
    assert [doc["id"] for doc in client.calls[0]["documents"]] == ["KB-001", "KB-002"]


def test_generator_receives_retrieved_articles():
    received = {}
    source = {"id": "KB-001", "title": "Reset your password", "url": "https://support.example.com/password-reset"}
    expected = SupportAnswer(answer="ok", sources=[source], escalate=False, reason="answered")
    client = FakeClient()

    def fake_generator(question, articles, client_arg):
        received.update(question=question, articles=articles, client=client_arg)
        return expected

    result = answer_question(
        QUESTION, retriever=FakeRetriever(), client=client, generator=fake_generator
    )

    assert result is expected
    assert received == {"question": QUESTION, "articles": ARTICLES, "client": client}


def test_no_retrieved_articles_hands_off_without_api_call():
    client = FakeClient(make_response())

    result = answer_question(QUESTION, retriever=FakeRetriever(articles=[]), client=client)

    assert_handoff(result, "no_articles")
    assert client.calls == []


def test_insufficient_evidence_escalates():
    client = FakeClient(make_response(text=INSUFFICIENT_EVIDENCE, cited_ids=()))

    result = answer_question(QUESTION, retriever=FakeRetriever(), client=client)

    assert_handoff(result, "insufficient_evidence")


def test_api_error_gives_safe_handoff():
    client = FakeClient(error=RuntimeError("401 invalid api key sk-secret"))

    result = answer_question(QUESTION, retriever=FakeRetriever(), client=client)

    assert_handoff(result, "llm_error")
    assert "secret" not in result.answer


@pytest.mark.parametrize("question", ["", "   \n", None])
def test_empty_question_does_no_work(question):
    retriever = FakeRetriever()
    client = FakeClient(make_response())

    result = answer_question(question, retriever=retriever, client=client)

    assert_handoff(result, "empty_question")
    assert retriever.questions == []
    assert client.calls == []


def test_retrieval_error_gives_safe_handoff():
    retriever = FakeRetriever(error=RuntimeError("embedding service down"))

    result = answer_question(QUESTION, retriever=retriever, client=FakeClient())

    assert_handoff(result, "retrieval_error")


def test_unexpected_generator_error_gives_safe_handoff():
    def broken_generator(question, articles, client):
        raise ValueError("internal detail")

    result = answer_question(
        QUESTION, retriever=FakeRetriever(), client=FakeClient(), generator=broken_generator
    )

    assert_handoff(result, "agent_error")


def test_missing_api_key_gives_configuration_handoff(monkeypatch):
    monkeypatch.setattr(agent, "load_dotenv", lambda: None)  # Ignore the real .env file.
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    retriever = FakeRetriever()

    result = answer_question(QUESTION, retriever=retriever)

    assert_handoff(result, "configuration_error")
    assert retriever.questions == []


def test_default_client_is_created_lazily_once(monkeypatch):
    created = []

    class FakeClientV2(FakeClient):
        def __init__(self, api_key):
            super().__init__(make_response())
            created.append(api_key)

    monkeypatch.setattr(agent, "load_dotenv", lambda: None)
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setattr(cohere, "ClientV2", FakeClientV2)

    assert created == []
    answer_question(QUESTION, retriever=FakeRetriever())
    answer_question(QUESTION, retriever=FakeRetriever())

    assert created == ["test-key"]


def test_importing_agent_creates_no_cohere_client():
    code = (
        "import sys, support_agent.agent; "
        "assert 'support_agent.semantic_kb' not in sys.modules; "
        "assert 'cohere' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_cli_without_question_prints_usage(capsys):
    assert main([]) == 2
    assert "Usage" in capsys.readouterr().err


def test_cli_prints_answer_and_sources(monkeypatch, capsys):
    result = SupportAnswer(
        answer=ANSWER_TEXT,
        sources=[{"id": "KB-001", "title": "Reset your password", "url": "https://x/reset"}],
    )
    monkeypatch.setattr(agent, "answer_question", lambda question: result)

    assert main(["How", "do", "I", "reset?"]) == 0

    output = capsys.readouterr().out
    assert ANSWER_TEXT in output
    assert "Reset your password (KB-001): https://x/reset" in output
    assert "escalate=False" in output
