"""Offline tests for the whole answer_question pipeline (V2 Phase 3, Milestone 3):

question -> configuration -> retrieval (keyword + semantic, hybrid) -> optional
rerank -> answer generation -> final answer check -> SupportAnswer.

Every stage is made to fail in a controlled way; the customer always gets
either a checked answer or the standard hand-off, and nothing internal leaks.
The real hybrid retrieval runs against a temporary kb.json with a fake Cohere
SDK (see the `semantic_kb` fixture in conftest.py). No network, no .env.
"""

import json
import logging
from types import SimpleNamespace

import pytest

from support_agent import agent, kb
from support_agent.agent import answer_question
from support_agent.answer import HANDOFF_MESSAGE, SupportAnswer, generate_answer
from support_agent.rerank import RerankConfig

SECRET = "sk-SECRET-pipeline-42"
QUESTION = "How do I reset my password?"
OFF = RerankConfig()
ARTICLES = [
    {"id": "KB-001", "title": "Reset your password", "text": "Select Forgot password.", "url": "https://x/reset"},
    {"id": "KB-002", "title": "Refund eligibility", "text": "Refunds within 30 days.", "url": "https://x/refunds"},
]


def source(article):
    return {"id": article["id"], "title": article["title"], "url": article["url"]}


def chat_response(*, finish="COMPLETE", text="Select Forgot password.", citations=None):
    if citations is None:
        citations = [SimpleNamespace(type="TEXT_CONTENT", sources=[SimpleNamespace(type="document", id="KB-001")])]
    return SimpleNamespace(
        finish_reason=finish,
        message=SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], citations=citations),
    )


class ChatClient:
    def __init__(self, response=None):
        self.response = response if response is not None else chat_response()
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def assert_handoff(result, reason):
    assert result == SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason=reason)


def ask(**kwargs):
    kwargs.setdefault("retriever", lambda question: ARTICLES)
    kwargs.setdefault("client", ChatClient())
    kwargs.setdefault("rerank_config", OFF)
    return answer_question(kwargs.pop("question", QUESTION), **kwargs)


def _boom(*args, **kwargs):
    raise RuntimeError(f"internal failure at /srv/app/agent.py line 99, key={SECRET}")


# --- An unexpected exception at any stage -------------------------------------


STAGE_FAILURES = {
    # stage: (ask() arguments, expected reason)
    "client creation": (dict(client=None), "configuration_error"),
    "retrieval": (dict(retriever=_boom), "retrieval_error"),
    "rerank (fails open)": (dict(reranker=_boom, rerank_config=RerankConfig(enabled=True)), "answered"),
    "answer generation": (dict(generator=_boom), "agent_error"),
}


@pytest.mark.parametrize("stage", STAGE_FAILURES)
def test_unexpected_exception_at_any_stage_gives_a_safe_result(stage, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(agent, "_default_client", _boom)
    arguments, expected = STAGE_FAILURES[stage]

    result = ask(**arguments)

    if expected == "answered":
        assert (result.escalate, result.reason, result.sources) == (False, "answered", [source(ARTICLES[0])])
    else:
        assert_handoff(result, expected)
    for text in (result.answer, caplog.text):
        assert SECRET not in text and "/srv/app" not in text and "Traceback" not in text


@pytest.mark.parametrize("returned", [None, [], ()], ids=["none", "empty-list", "empty-tuple"])
def test_retrieval_returning_nothing_is_a_hand_off_not_an_answer(returned):
    client = ChatClient()

    assert_handoff(ask(retriever=lambda question: returned, client=client), "no_articles")
    assert client.calls == []


@pytest.mark.parametrize("returned", [42, "KB-001", {"id": "KB-001"}, [None], iter(ARTICLES)],
                         ids=["int", "string", "dict", "list-of-none", "iterator"])
def test_retrieval_returning_the_wrong_type_is_a_retrieval_error(returned):
    client = ChatClient()

    assert_handoff(ask(retriever=lambda question: returned, client=client), "retrieval_error")
    assert client.calls == []


# --- Malformed chat responses: one category, never an internal error --------


@pytest.mark.parametrize(
    "response",
    [
        chat_response(citations=5),                                                      # not a list
        chat_response(citations=[SimpleNamespace(type="TEXT_CONTENT", sources=7)]),      # sources not a list
        chat_response(citations=[SimpleNamespace(
            type="TEXT_CONTENT", sources=[SimpleNamespace(type="document", id=["KB-001"])])]),  # unhashable id
        chat_response(finish=["COMPLETE"]),                                               # unhashable finish reason
        SimpleNamespace(finish_reason="COMPLETE", message=SimpleNamespace(content=5, citations=[])),
    ],
    ids=["citations-int", "sources-int", "id-list", "finish-reason-list", "content-int"],
)
def test_unreadable_chat_response_is_a_malformed_response(response):
    """These used to escape generate_answer as TypeError and end up as agent_error."""
    assert_handoff(generate_answer(QUESTION, ARTICLES, ChatClient(response)), "malformed_response")
    assert_handoff(ask(client=ChatClient(response)), "malformed_response")


def test_unexpected_finish_reason_is_logged_shortened(caplog):
    long_reason = "X" * 500

    ask(client=ChatClient(chat_response(finish=long_reason)))

    assert long_reason not in caplog.text


# --- The final answer check cannot be bypassed --------------------------------


def _returns(result):
    return lambda question, articles, client: result


@pytest.mark.parametrize("sources", [5, 3.5, object()], ids=["int", "float", "object"])
def test_sources_that_are_not_a_list_are_handed_off(sources):
    """A non-iterable `sources` used to make answer_question itself raise."""
    assert_handoff(ask(generator=_returns(SupportAnswer(answer="x", sources=sources))), "invalid_citations")


@pytest.mark.parametrize("reason", ["llm_error", "", None, "insufficient_evidence"])
def test_answer_with_a_reason_other_than_answered_is_handed_off(reason):
    answer = SupportAnswer(answer="Select Forgot password.", sources=[source(ARTICLES[0])], reason=reason)

    assert_handoff(ask(generator=_returns(answer)), "agent_error")


@pytest.mark.parametrize("escalate", [None, 0, "no", "yes"])
def test_escalate_flag_must_be_a_boolean(escalate):
    answer = SupportAnswer(answer="Select Forgot password.", sources=[source(ARTICLES[0])], escalate=escalate)

    assert_handoff(ask(generator=_returns(answer)), "agent_error")


def test_a_crash_inside_the_final_check_is_still_a_safe_hand_off(caplog):
    class Exploding(SupportAnswer):
        @property
        def sources(self):
            raise RuntimeError(SECRET)

        @sources.setter
        def sources(self, value):
            pass

    result = ask(generator=_returns(Exploding(answer="x")))

    assert_handoff(result, "agent_error")
    assert SECRET not in caplog.text


# --- Reranking may re-order and remove, never change -------------------------


@pytest.mark.parametrize("field", ["title", "text", "url"])
def test_reranker_that_changes_an_article_fails_open_to_the_retrieved_articles(field):
    received = []

    def generator(question, articles, client):
        received.append(articles)
        return generate_answer(question, articles, client)

    def changing_reranker(question, articles, client, config):
        return [dict(articles[0], **{field: "Refunds are always granted."})]

    result = ask(generator=generator, reranker=changing_reranker, rerank_config=RerankConfig(enabled=True))

    assert result.reason == "answered"
    assert received == [ARTICLES]


def test_valid_rerank_order_and_extra_score_field_are_used():
    received = []

    def generator(question, articles, client):
        received.append([article["id"] for article in articles])
        return generate_answer(question, articles, client)

    def reranker(question, articles, client, config):
        return [dict(articles[1], rerank_score=0.9), dict(articles[0], rerank_score=0.8)]

    assert ask(generator=generator, reranker=reranker, rerank_config=RerankConfig(enabled=True)).reason == "answered"
    assert received == [["KB-002", "KB-001"]]


# --- The real hybrid retrieval: knowledge-base problems and freshness --------


@pytest.fixture
def kb_file(tmp_path, monkeypatch):
    """A temporary kb.json used by both keyword and semantic retrieval."""
    path = tmp_path / "kb.json"
    monkeypatch.setattr(kb, "KB_PATH", path)

    def write(articles):
        path.write_text(articles if isinstance(articles, str) else json.dumps(articles), encoding="utf-8")

    write(ARTICLES)
    return write


def test_real_hybrid_pipeline_answers_from_the_knowledge_base(semantic_kb, kb_file):
    result = ask(retriever=None)

    assert result.reason == "answered"
    assert result.sources == [source(ARTICLES[0])]


@pytest.mark.parametrize(
    "content",
    ["{not json", json.dumps({"id": "KB-001"}), json.dumps([{"id": "KB-001", "title": "Reset"}]), None],
    ids=["invalid-json", "not-a-list", "missing-fields", "missing-file"],
)
def test_broken_knowledge_base_is_a_retrieval_error(semantic_kb, kb_file, content, caplog):
    if content is None:
        kb.KB_PATH.unlink()
    else:
        kb_file(content)
    client = ChatClient()

    result = ask(retriever=None, client=client)

    assert_handoff(result, "retrieval_error")
    assert client.calls == []
    assert str(kb.KB_PATH) not in caplog.text  # file paths stay out of the logs


def test_embedding_outage_is_a_retrieval_error_not_an_empty_answer(semantic_kb, kb_file):
    from support_agent.cohere_calls import CohereCallError

    # Two failures: the one retry (Milestone 2 policy) also fails.
    semantic_kb.client.outcomes = [CohereCallError("embed", "server_error")] * 2
    client = ChatClient()

    assert_handoff(ask(retriever=None, client=client), "retrieval_error")
    assert client.calls == []


def test_semantic_results_follow_knowledge_base_edits(semantic_kb, kb_file):
    """Semantic retrieval used to keep the first-loaded articles for the life of
    the process, so after an edit it returned the old text and URL while
    keyword retrieval returned the new ones."""
    semantic_kb.retrieve_semantic(QUESTION)
    edited = [dict(ARTICLES[0], text="Open Settings, then Security.", url="https://x/new-reset"), ARTICLES[1]]
    kb_file(edited)

    results = semantic_kb.retrieve_semantic(QUESTION)

    by_id = {article["id"]: article for article in results}
    assert by_id["KB-001"]["text"] == "Open Settings, then Security."
    assert by_id["KB-001"]["url"] == "https://x/new-reset"


def test_answer_after_an_edit_cites_the_current_article(semantic_kb, kb_file):
    ask(retriever=None)
    kb_file([dict(ARTICLES[0], url="https://x/new-reset"), ARTICLES[1]])

    result = ask(retriever=None)

    assert result.sources == [{"id": "KB-001", "title": "Reset your password", "url": "https://x/new-reset"}]


def test_articles_are_embedded_again_only_when_their_text_changes(semantic_kb, kb_file):
    def document_embeds():
        return [call for call in semantic_kb.client.calls if call["input_type"] == "search_document"]

    semantic_kb.retrieve_semantic(QUESTION)
    semantic_kb.retrieve_semantic(QUESTION)
    assert len(document_embeds()) == 1                      # cached

    kb_file([dict(ARTICLES[0], url="https://x/moved"), ARTICLES[1]])
    semantic_kb.retrieve_semantic(QUESTION)
    assert len(document_embeds()) == 1                      # URL isn't embedded: still cached

    kb_file([dict(ARTICLES[0], text="New steps."), ARTICLES[1]])
    semantic_kb.retrieve_semantic(QUESTION)
    assert len(document_embeds()) == 2                      # text changed: embedded again
    assert document_embeds()[-1]["texts"][0] == "Reset your password\nNew steps."


def test_empty_knowledge_base_makes_no_document_embedding_request(semantic_kb, kb_file):
    kb_file([])

    assert semantic_kb.retrieve_semantic(QUESTION) == []
    assert [call["input_type"] for call in semantic_kb.client.calls] == ["search_query"]
    assert_handoff(ask(retriever=None), "no_articles")
