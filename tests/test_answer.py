"""Offline tests for answer generation. No Cohere API calls are made.

Fake responses are built from the installed Cohere SDK's own response classes,
so they have the same shape as a real `ClientV2.chat` response.
"""

import pytest
from cohere import (
    AssistantMessageResponse,
    Citation,
    DocumentSource,
    TextAssistantMessageResponseContentItem,
    V2ChatResponse,
)
from cohere.types.source import ToolSource

from support_agent.answer import (
    HANDOFF_MESSAGE,
    INSUFFICIENT_EVIDENCE,
    SupportAnswer,
    generate_answer,
)

ARTICLES = [
    {
        "id": "KB-001",
        "title": "Reset your password",
        "text": "Open the sign-in page and select Forgot password.",
        "url": "https://support.example.com/password-reset",
        "score": 0.9,
    },
    {
        "id": "KB-002",
        "title": "Refund eligibility",
        "text": "Refund requests are accepted within 30 days of purchase.",
        "url": "https://support.example.com/refunds",
        "score": 0.4,
    },
]

QUESTION = "How do I reset my password?"
ANSWER_TEXT = "Open the sign-in page and select Forgot password."


def make_response(text=ANSWER_TEXT, cited_ids=("KB-001",), finish_reason="COMPLETE"):
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
        finish_reason=finish_reason,
        message=AssistantMessageResponse(
            role="assistant",
            content=[TextAssistantMessageResponseContentItem(type="text", text=text)],
            citations=citations or None,
        ),
    )


class FakeClient:
    """Records chat() calls and returns a canned response or raises an error."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def assert_escalated(result, reason):
    assert isinstance(result, SupportAnswer)
    assert result.escalate is True
    assert result.reason == reason
    assert result.answer == HANDOFF_MESSAGE
    assert result.sources == []


def test_valid_answer_with_citations():
    client = FakeClient(make_response())

    result = generate_answer(QUESTION, ARTICLES, client)

    assert result == SupportAnswer(
        answer=ANSWER_TEXT,
        sources=[
            {
                "id": "KB-001",
                "title": "Reset your password",
                "url": "https://support.example.com/password-reset",
            }
        ],
        escalate=False,
        reason="answered",
    )


def test_articles_are_sent_as_documents():
    client = FakeClient(make_response())

    generate_answer(QUESTION, ARTICLES, client, model="test-model")

    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"][-1] == {"role": "user", "content": QUESTION}
    assert call["documents"][0] == {
        "id": "KB-001",
        "data": {
            "title": "Reset your password",
            "text": "Open the sign-in page and select Forgot password.",
            "url": "https://support.example.com/password-reset",
        },
    }
    assert [doc["id"] for doc in call["documents"]] == ["KB-001", "KB-002"]


def test_model_can_be_set_from_environment(monkeypatch):
    monkeypatch.setenv("COHERE_CHAT_MODEL", "env-model")
    client = FakeClient(make_response())

    generate_answer(QUESTION, ARTICLES, client)

    assert client.calls[0]["model"] == "env-model"


def test_no_citations_causes_escalation():
    client = FakeClient(make_response(cited_ids=()))

    assert_escalated(generate_answer(QUESTION, ARTICLES, client), "no_citations")


def test_insufficient_evidence_causes_escalation():
    client = FakeClient(make_response(text=INSUFFICIENT_EVIDENCE, cited_ids=()))

    assert_escalated(generate_answer(QUESTION, ARTICLES, client), "insufficient_evidence")


def test_insufficient_evidence_wins_even_with_citations():
    client = FakeClient(make_response(text=f"{INSUFFICIENT_EVIDENCE}.", cited_ids=("KB-001",)))

    assert_escalated(generate_answer(QUESTION, ARTICLES, client), "insufficient_evidence")


def test_api_exception_causes_escalation():
    client = FakeClient(error=RuntimeError("secret internal error detail"))

    result = generate_answer(QUESTION, ARTICLES, client)

    assert_escalated(result, "llm_error")
    assert "secret" not in result.answer


@pytest.mark.parametrize("articles", [[], None])
def test_no_articles_means_no_api_call(articles):
    client = FakeClient(make_response())

    assert_escalated(generate_answer(QUESTION, articles, client), "no_articles")
    assert client.calls == []


def test_empty_question_means_no_api_call():
    client = FakeClient(make_response())

    assert_escalated(generate_answer("   ", ARTICLES, client), "empty_question")
    assert client.calls == []


def test_unknown_citation_ids_are_not_returned():
    client = FakeClient(make_response(cited_ids=("KB-999", "KB-001")))

    result = generate_answer(QUESTION, ARTICLES, client)

    assert result.escalate is False
    assert [source["id"] for source in result.sources] == ["KB-001"]


def test_only_unknown_citation_ids_causes_escalation():
    client = FakeClient(make_response(cited_ids=("KB-999",)))

    assert_escalated(generate_answer(QUESTION, ARTICLES, client), "no_citations")


def test_tool_sources_are_not_treated_as_documents():
    response = make_response(cited_ids=())
    response.message.citations = [
        Citation(start=0, end=5, text="Open", sources=[ToolSource(type="tool", id="KB-001")])
    ]

    assert_escalated(generate_answer(QUESTION, ARTICLES, FakeClient(response)), "no_citations")


def test_duplicate_citations_are_listed_once():
    client = FakeClient(make_response(cited_ids=("KB-002", "KB-001", "KB-002")))

    result = generate_answer(QUESTION, ARTICLES, client)

    assert [source["id"] for source in result.sources] == ["KB-002", "KB-001"]


@pytest.mark.parametrize(
    "response",
    [
        None,
        object(),
        V2ChatResponse(
            id="x",
            finish_reason="COMPLETE",
            message=AssistantMessageResponse(role="assistant", content=None),
        ),
        V2ChatResponse(
            id="x",
            finish_reason="COMPLETE",
            message=AssistantMessageResponse(
                role="assistant",
                content=[TextAssistantMessageResponseContentItem(type="text", text="   ")],
            ),
        ),
    ],
    ids=["none", "plain-object", "no-content", "blank-text"],
)
def test_malformed_response_causes_safe_escalation(response):
    assert_escalated(
        generate_answer(QUESTION, ARTICLES, FakeClient(response)), "malformed_response"
    )


def test_truncated_answer_causes_escalation():
    client = FakeClient(make_response(finish_reason="MAX_TOKENS"))

    assert_escalated(generate_answer(QUESTION, ARTICLES, client), "incomplete_response")
