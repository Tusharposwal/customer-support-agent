"""Offline tests for support_agent.rerank. No Cohere import, no API calls."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from support_agent.answer import generate_answer
from support_agent.rerank import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    RerankConfig,
    RerankError,
    config_from_env,
    document_text,
    rerank_articles,
)

ARTICLES = [
    {"id": "KB-001", "title": "Reset your password", "text": "Select Forgot password.",
     "url": "https://x/reset", "score": 0.032, "keyword_rank": 1, "semantic_rank": 2},
    {"id": "KB-002", "title": "Refund eligibility", "text": "Refunds within 30 days.",
     "url": "https://x/refunds", "score": 0.031, "keyword_rank": None, "semantic_rank": 1},
    {"id": "KB-003", "title": "Track an order", "text": "Open Orders to track.",
     "url": "https://x/track", "score": 0.016, "keyword_rank": None, "semantic_rank": 3},
]
QUESTION = "How do I reset my password?"
ENABLED = RerankConfig(enabled=True)


def response(*pairs):
    """Fake rerank response: pairs of (index, relevance_score)."""
    return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in pairs])


class FakeRerankClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


# --- Configuration -----------------------------------------------------------


def test_defaults_are_off_with_no_threshold():
    config = config_from_env({})

    assert config == RerankConfig()
    assert config.enabled is False
    assert config.model == DEFAULT_MODEL == "rerank-v5.0-pro"
    assert config.min_score is None
    assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS == 5.0


def test_blank_values_mean_defaults():
    blank = {name: "  " for name in (
        "COHERE_RERANK_ENABLED", "COHERE_RERANK_MODEL",
        "COHERE_RERANK_MIN_SCORE", "COHERE_RERANK_TIMEOUT_SECONDS",
    )}
    assert config_from_env(blank) == RerankConfig()


@pytest.mark.parametrize("raw, expected", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True), (" Yes ", True),
    ("false", False), ("0", False), ("no", False), ("False", False),
])
def test_enabled_flag_spellings(raw, expected):
    assert config_from_env({"COHERE_RERANK_ENABLED": raw}).enabled is expected


def test_all_settings_are_read():
    config = config_from_env({
        "COHERE_RERANK_ENABLED": "true",
        "COHERE_RERANK_MODEL": "rerank-v3.5",
        "COHERE_RERANK_MIN_SCORE": "0.25",
        "COHERE_RERANK_TIMEOUT_SECONDS": "2.5",
    })
    assert config == RerankConfig(enabled=True, model="rerank-v3.5", min_score=0.25, timeout_seconds=2.5)


@pytest.mark.parametrize("score", ["0", "1", "0.5"])
def test_min_score_bounds_are_inclusive(score):
    assert config_from_env({"COHERE_RERANK_MIN_SCORE": score}).min_score == float(score)


@pytest.mark.parametrize("env", [
    {"COHERE_RERANK_ENABLED": "ture"},
    {"COHERE_RERANK_ENABLED": "on"},
    {"COHERE_RERANK_MIN_SCORE": "high"},
    {"COHERE_RERANK_MIN_SCORE": "-0.1"},
    {"COHERE_RERANK_MIN_SCORE": "1.01"},
    {"COHERE_RERANK_MIN_SCORE": "nan"},
    {"COHERE_RERANK_TIMEOUT_SECONDS": "0"},
    {"COHERE_RERANK_TIMEOUT_SECONDS": "-5"},
    {"COHERE_RERANK_TIMEOUT_SECONDS": "soon"},
    {"COHERE_RERANK_TIMEOUT_SECONDS": "inf"},
])
def test_invalid_settings_raise(env):
    with pytest.raises(ValueError):
        config_from_env(env)


def test_reads_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("COHERE_RERANK_ENABLED", "1")
    monkeypatch.setenv("COHERE_RERANK_MIN_SCORE", "0.4")
    config = config_from_env()
    assert config.enabled is True and config.min_score == 0.4


# --- Request -----------------------------------------------------------------


def test_document_text_matches_the_experiment_format():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
    import rerank_experiment as rx

    assert rx.DOCUMENT_FORMAT == "title-text-v1"
    for article in ARTICLES:
        assert document_text(article) == rx.document_string(article)


def test_request_has_model_question_documents_timeout_and_no_retries():
    client = FakeRerankClient(response((0, 0.9), (1, 0.2), (2, 0.1)))
    config = RerankConfig(enabled=True, model="rerank-test", timeout_seconds=3.0)

    rerank_articles(QUESTION, ARTICLES, client, config)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "rerank-test"
    assert call["query"] == QUESTION
    assert call["documents"] == [
        "Title: Reset your password\nText: Select Forgot password.",
        "Title: Refund eligibility\nText: Refunds within 30 days.",
        "Title: Track an order\nText: Open Orders to track.",
    ]
    assert call["request_options"] == {"max_retries": 0, "timeout": 3.0}
    assert "top_n" not in call  # every article must come back


# --- Ordering ----------------------------------------------------------------


def test_results_are_sorted_by_score_and_mapped_to_the_right_articles():
    client = FakeRerankClient(response((1, 0.2), (2, 0.95), (0, 0.5)))

    result = rerank_articles(QUESTION, ARTICLES, client, ENABLED)

    assert [a["id"] for a in result] == ["KB-003", "KB-001", "KB-002"]
    assert [a["rerank_score"] for a in result] == [0.95, 0.5, 0.2]


def test_every_original_field_is_kept_and_input_is_not_modified():
    before = [dict(a) for a in ARTICLES]
    client = FakeRerankClient(response((2, 0.9), (0, 0.5), (1, 0.1)))

    result = rerank_articles(QUESTION, ARTICLES, client, ENABLED)

    assert ARTICLES == before
    by_id = {a["id"]: a for a in result}
    for original in ARTICLES:
        copy = by_id[original["id"]]
        assert {k: v for k, v in copy.items() if k != "rerank_score"} == original
        assert copy is not original


def test_equal_scores_keep_the_api_order():
    client = FakeRerankClient(response((2, 0.5), (0, 0.5), (1, 0.5)))
    result = rerank_articles(QUESTION, ARTICLES, client, ENABLED)
    assert [a["id"] for a in result] == ["KB-003", "KB-001", "KB-002"]


# --- Threshold ---------------------------------------------------------------


def test_threshold_keeps_scores_at_or_above_it():
    client = FakeRerankClient(response((0, 0.6), (1, 0.4), (2, 0.39999)))
    config = RerankConfig(enabled=True, min_score=0.4)

    result = rerank_articles(QUESTION, ARTICLES, client, config)

    assert [(a["id"], a["rerank_score"]) for a in result] == [("KB-001", 0.6), ("KB-002", 0.4)]


def test_threshold_can_remove_every_article():
    client = FakeRerankClient(response((0, 0.1), (1, 0.05), (2, 0.01)))
    assert rerank_articles(QUESTION, ARTICLES, client, RerankConfig(enabled=True, min_score=0.5)) == []


def test_no_threshold_never_removes_articles():
    client = FakeRerankClient(response((0, 0.0), (1, 0.0), (2, 0.0)))
    assert len(rerank_articles(QUESTION, ARTICLES, client, ENABLED)) == 3


# --- Empty input and failures ------------------------------------------------


def test_no_articles_makes_no_request():
    client = FakeRerankClient(error=AssertionError("must not be called"))
    assert rerank_articles(QUESTION, [], client, ENABLED) == []
    assert client.calls == []


class FakeApiError(Exception):
    def __init__(self, status_code, body):
        super().__init__(f"status_code: {status_code}, body: {body}")
        self.status_code = status_code
        self.body = body


@pytest.mark.parametrize("error", [
    FakeApiError(401, "invalid api key sk-secret-123"),
    FakeApiError(429, "rate limited"),
    FakeApiError(500, "internal error"),
    httpx.ReadTimeout("timed out"),
    httpx.ConnectError("connection refused"),
])
def test_api_failures_and_timeouts_raise_rerank_error_after_one_attempt(error):
    client = FakeRerankClient(error=error)

    with pytest.raises(RerankError) as raised:
        rerank_articles(QUESTION, ARTICLES, client, ENABLED)

    assert len(client.calls) == 1
    assert str(raised.value) == f"Rerank request failed: {type(error).__name__}"
    assert "secret" not in str(raised.value)


@pytest.mark.parametrize("bad_response", [
    None,
    SimpleNamespace(),
    SimpleNamespace(results=None),
    response(),                                      # empty results
    response((0, 0.9), (1, 0.5)),                    # an article missing
    response((0, 0.9), (0, 0.5), (1, 0.1)),          # duplicate index
    response((0, 0.9), (1, 0.5), (3, 0.1)),          # index out of range
    response((0, 0.9), (1, 0.5), (-1, 0.1)),         # negative index
    response((0, 0.9), (1, 0.5), (2, 0.1), (2, 0.1)),
    response((0, 0.9), (1, 0.5), (True, 0.1)),       # bool is not an index
    response((0, 0.9), (1, 0.5), ("2", 0.1)),
    response((0, 0.9), (1, 0.5), (2, None)),
    response((0, 0.9), (1, 0.5), (2, "0.1")),
    response((0, 0.9), (1, 0.5), (2, float("nan"))),
])
def test_malformed_responses_raise_rerank_error(bad_response):
    with pytest.raises(RerankError):
        rerank_articles(QUESTION, ARTICLES, FakeRerankClient(bad_response), ENABLED)


# --- Citation compatibility --------------------------------------------------


class FakeChatClient(FakeRerankClient):
    def __init__(self, rerank_result, cited_ids):
        super().__init__(rerank_result)
        self.cited_ids = cited_ids
        self.chat_calls = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        citations = [
            SimpleNamespace(type="TEXT_CONTENT", sources=[SimpleNamespace(type="document", id=doc_id)])
            for doc_id in self.cited_ids
        ]
        return SimpleNamespace(
            finish_reason="COMPLETE",
            message=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Select Forgot password.")],
                citations=citations,
            ),
        )


def test_reranked_articles_work_with_the_real_answer_generator():
    client = FakeChatClient(response((0, 0.9), (1, 0.2), (2, 0.1)), cited_ids=["KB-001"])
    articles = rerank_articles(QUESTION, ARTICLES, client, ENABLED)

    result = generate_answer(QUESTION, articles, client)

    assert result.reason == "answered"
    assert result.sources == [{"id": "KB-001", "title": "Reset your password", "url": "https://x/reset"}]
    documents = client.chat_calls[0]["documents"]
    assert [d["id"] for d in documents] == ["KB-001", "KB-002", "KB-003"]
    assert all(set(d["data"]) == {"title", "text", "url"} for d in documents)  # no scores sent


def test_citation_of_an_article_removed_by_the_threshold_is_rejected():
    client = FakeChatClient(response((0, 0.9), (1, 0.2), (2, 0.1)), cited_ids=["KB-002"])
    articles = rerank_articles(QUESTION, ARTICLES, client, RerankConfig(enabled=True, min_score=0.5))

    result = generate_answer(QUESTION, articles, client)

    assert [d["id"] for d in client.chat_calls[0]["documents"]] == ["KB-001"]
    assert result.escalate is True
    assert result.reason == "no_citations"


# --- Import cost -------------------------------------------------------------


def test_importing_rerank_imports_no_cohere():
    code = "import sys, support_agent.rerank; assert 'cohere' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
