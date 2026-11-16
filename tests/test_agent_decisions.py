"""Offline tests for answer_question's decisions: answer, or hand off because
the evidence is insufficient, or hand off because something failed.
No Cohere import, no API calls."""

from types import SimpleNamespace

import pytest

from support_agent.agent import answer_question
from support_agent.answer import HANDOFF_MESSAGE, SupportAnswer, generate_answer
from support_agent.rerank import RerankConfig

ARTICLES = [
    {"id": "KB-001", "title": "Reset your password", "text": "Select Forgot password.", "url": "https://x/reset"},
    {"id": "KB-002", "title": "Refund eligibility", "text": "Refunds within 30 days.", "url": "https://x/refunds"},
]
QUESTION = "How do I reset my password?"
SECRET = "sk-secret-123"
OFF = RerankConfig()


def source(article):
    return {"id": article["id"], "title": article["title"], "url": article["url"]}


def chat_response(text="Select Forgot password.", cited=("KB-001",), finish="COMPLETE"):
    return SimpleNamespace(
        finish_reason=finish,
        message=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            citations=[
                SimpleNamespace(type="TEXT_CONTENT", sources=[SimpleNamespace(type="document", id=i)])
                for i in cited
            ],
        ),
    )


class FakeClient:
    def __init__(self, chat=None, chat_error=None, rerank=None, rerank_error=None):
        self.chat_result, self.chat_error = chat, chat_error
        self.rerank_result, self.rerank_error = rerank, rerank_error
        self.chat_calls, self.rerank_calls = [], []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.chat_error:
            raise self.chat_error
        return self.chat_result

    def rerank(self, **kwargs):
        self.rerank_calls.append(kwargs)
        if self.rerank_error:
            raise self.rerank_error
        return self.rerank_result


def rerank_response(*pairs):
    return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in pairs])


def ask(*, client=None, articles=ARTICLES, retriever=None, generator=generate_answer, config=OFF, **kwargs):
    return answer_question(
        kwargs.pop("question", QUESTION),
        retriever=retriever or (lambda q: articles),
        client=client or FakeClient(chat=chat_response()),
        generator=generator,
        rerank_config=config,
        **kwargs,
    )


def assert_handoff(result, reason):
    assert result == SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason=reason)


class FakeApiError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status_code: {status_code}, body: invalid api key {SECRET}")
        self.status_code = status_code


def _broken_retriever(question):
    raise RuntimeError(f"embedding service down {SECRET}")


# --- Every request outcome, end to end ---------------------------------------

OUTCOMES = {
    # name: (ask() arguments, expected reason or "answered")
    "empty question": (dict(question="   "), "empty_question"),
    "no articles": (dict(articles=[]), "no_articles"),
    "retrieval fails": (dict(retriever=_broken_retriever), "retrieval_error"),
    "rerank removes all": (
        dict(config=RerankConfig(enabled=True, min_score=0.5),
             client=FakeClient(chat=chat_response(), rerank=rerank_response((0, 0.2), (1, 0.1)))),
        "below_rerank_threshold",
    ),
    "rerank API fails (fail open)": (
        dict(config=RerankConfig(enabled=True, min_score=0.5),
             client=FakeClient(chat=chat_response(), rerank_error=FakeApiError(503))),
        "answered",
    ),
    "insufficient evidence": (
        dict(client=FakeClient(chat=chat_response(text="INSUFFICIENT_EVIDENCE", cited=()))),
        "insufficient_evidence",
    ),
    "no citations": (dict(client=FakeClient(chat=chat_response(cited=()))), "no_citations"),
    "citation not supplied": (
        dict(client=FakeClient(chat=chat_response(cited=("KB-001", "KB-999")))), "invalid_citations",
    ),
    "LLM API fails": (dict(client=FakeClient(chat_error=FakeApiError(500))), "llm_error"),
    "LLM generation error": (dict(client=FakeClient(chat=chat_response(finish="ERROR"))), "llm_error"),
    "LLM output truncated": (
        dict(client=FakeClient(chat=chat_response(finish="MAX_TOKENS"))), "incomplete_response",
    ),
    "LLM output malformed": (dict(client=FakeClient(chat=SimpleNamespace())), "malformed_response"),
    "successful answer": (dict(), "answered"),
}


@pytest.mark.parametrize("name", OUTCOMES)
def test_every_outcome_gives_the_documented_decision(name, caplog):
    arguments, expected = OUTCOMES[name]

    result = ask(**arguments)

    if expected == "answered":
        assert result.escalate is False
        assert result.reason == "answered"
        assert result.sources == [source(ARTICLES[0])]
    else:
        assert_handoff(result, expected)
    assert SECRET not in result.answer
    assert SECRET not in caplog.text


# --- Retrieved articles are checked before use -------------------------------


@pytest.mark.parametrize("articles", [
    [{"id": "KB-001", "title": "Reset", "text": "x"}],                                    # missing url
    [{"id": "KB-001", "title": "Reset", "text": "x", "url": "  "}],                       # blank url
    [{"id": 1, "title": "Reset", "text": "x", "url": "https://x"}],                       # non-string id
    [ARTICLES[0], dict(ARTICLES[0], text="a different body")],                            # duplicate id
    ["KB-001"],                                                                           # not a dict
    "KB-001",                                                                             # not a list
], ids=["missing-field", "blank-field", "non-string-id", "duplicate-id", "not-dict", "not-list"])
def test_malformed_retrieval_results_are_a_retrieval_error(articles):
    client = FakeClient(chat=chat_response())

    assert_handoff(ask(client=client, articles=articles), "retrieval_error")
    assert client.chat_calls == []


def test_extra_article_fields_are_allowed():
    articles = [dict(ARTICLES[0], score=0.03, keyword_rank=1, semantic_rank=None), ARTICLES[1]]

    assert ask(articles=articles).reason == "answered"


# --- Reranker output must come from the retrieved articles -------------------


@pytest.mark.parametrize("reranked", [
    [dict(ARTICLES[0], id="KB-999")],                 # an article that was never retrieved
    [ARTICLES[0], ARTICLES[0]],                       # duplicate
    [{"id": "KB-001"}],                               # fields missing
    None,
], ids=["foreign", "duplicate", "missing-fields", "none"])
def test_invalid_reranker_output_fails_open_to_the_hybrid_articles(reranked):
    received = []

    def generator(question, articles, client):
        received.append(articles)
        return generate_answer(question, articles, client)

    result = ask(generator=generator, reranker=lambda *args: reranked, config=RerankConfig(enabled=True))

    assert result.reason == "answered"
    assert received == [ARTICLES]


# --- Final answer gate (independent of answer.py) ----------------------------


def _returns(result):
    return lambda question, articles, client: result


def test_valid_generator_answer_is_returned_unchanged():
    answer = SupportAnswer(answer="Select Forgot password.", sources=[source(ARTICLES[0])])

    assert ask(generator=_returns(answer)) is answer


@pytest.mark.parametrize("sources", [
    [{"id": "KB-999", "title": "Unknown", "url": "https://evil"}],                  # never supplied
    [source(ARTICLES[0]), {"id": "KB-999", "title": "Unknown", "url": "https://x"}],
    [dict(source(ARTICLES[0]), url="https://evil")],                                # wrong URL
    [dict(source(ARTICLES[0]), title="Different title")],                           # wrong title
    ["KB-001"],                                                                     # not a dict
    [{"id": ["KB-001"], "title": "x", "url": "y"}],                                 # unhashable id
], ids=["foreign", "one-foreign", "wrong-url", "wrong-title", "not-dict", "list-id"])
def test_answer_citing_anything_not_supplied_is_handed_off(sources):
    result = ask(generator=_returns(SupportAnswer(answer="An answer", sources=sources)))

    assert_handoff(result, "invalid_citations")


def test_source_removed_by_the_rerank_threshold_cannot_be_cited():
    client = FakeClient(rerank=rerank_response((0, 0.9), (1, 0.1)))
    answer = SupportAnswer(answer="Refunds within 30 days.", sources=[source(ARTICLES[1])])

    result = ask(client=client, generator=_returns(answer),
                 config=RerankConfig(enabled=True, min_score=0.5))

    assert_handoff(result, "invalid_citations")


def test_answer_without_sources_is_handed_off():
    assert_handoff(ask(generator=_returns(SupportAnswer(answer="An answer", sources=[]))), "no_citations")


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_answer_text_is_handed_off(text):
    result = ask(generator=_returns(SupportAnswer(answer=text, sources=[source(ARTICLES[0])])))

    assert_handoff(result, "malformed_response")


def test_hand_off_always_uses_the_standard_message_and_no_sources():
    leaky = SupportAnswer(answer=f"Error 401: {SECRET}", sources=[source(ARTICLES[0])],
                          escalate=True, reason="llm_error")

    assert_handoff(ask(generator=_returns(leaky)), "llm_error")


@pytest.mark.parametrize("reason", ["", None])
def test_hand_off_without_a_reason_becomes_agent_error(reason):
    result = ask(generator=_returns(SupportAnswer(answer="x", escalate=True, reason=reason)))

    assert_handoff(result, "agent_error")


@pytest.mark.parametrize("result", [None, "an answer", {"answer": "x"}])
def test_generator_returning_the_wrong_type_is_an_agent_error(result):
    assert_handoff(ask(generator=_returns(result)), "agent_error")
