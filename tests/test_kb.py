import json
from pathlib import Path

import pytest

from support_agent import kb
from support_agent.kb import load_articles, retrieve

# The keyword-retrieval tests below were written (and the title-match rule was
# tuned) against the 3-article KB v1. They keep running on that frozen copy so
# they still test the same behaviour; KB v2 checks are in tests/test_kb_v2.py.
V1_KB_PATH = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "kb_v1.json"


@pytest.fixture(autouse=True)
def v1_knowledge_base(monkeypatch):
    monkeypatch.setattr(kb, "KB_PATH", V1_KB_PATH)


def _valid_article(**overrides):
    article = {
        "id": "KB-T01",
        "title": "Test article",
        "text": "Test article text.",
        "url": "https://support.example.com/test",
    }
    article.update(overrides)
    return article


def _write_kb(tmp_path, content):
    """Write a temporary KB file; non-strings are serialized as JSON."""
    path = tmp_path / "kb.json"
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    return path


def test_load_articles():
    articles = load_articles()
    assert len(articles) == 3


def test_password_query():
    results = retrieve("How do I reset my password?")
    assert results
    assert results[0]["id"] == "KB-001"


def test_refund_query():
    results = retrieve("Can I get a refund?")
    assert results
    assert results[0]["id"] == "KB-002"


def test_order_tracking_query():
    results = retrieve("How can I track my shipment?")
    assert results
    assert results[0]["id"] == "KB-003"


def test_unrelated_query():
    results = retrieve("What is the weather on Mars?")
    assert results == []


def test_login_problem():
    results = retrieve("I can't log into my account")
    assert results
    assert results[0]["id"] == "KB-001"


def test_email_change_is_not_supported():
    results = retrieve("How do I change my email address?")
    assert results == []


def test_forgot_password_query():
    results = retrieve("I forgot my password and cannot sign in")
    assert results
    assert results[0]["id"] == "KB-001"


def test_where_is_my_order_ranks_tracking_first():
    results = retrieve("Where is my order?")
    assert results
    assert results[0]["id"] == "KB-003"


def test_refund_order_number_query():
    results = retrieve("Where do I enter my order number for a refund?")
    assert results
    assert results[0]["id"] == "KB-002"


@pytest.mark.parametrize(
    "query",
    [
        "How do I update my email?",
        "How do I delete my account?",
        "How do I change my phone number?",
        "How do I change my shipping address?",
    ],
)
def test_unsupported_account_questions(query):
    assert retrieve(query) == []


def test_load_valid_temporary_kb(tmp_path):
    articles = [_valid_article(), _valid_article(id="KB-T02")]
    path = _write_kb(tmp_path, articles)

    assert load_articles(path) == articles


def test_invalid_json_is_rejected(tmp_path):
    path = _write_kb(tmp_path, '[{"id": "KB-T01",')

    with pytest.raises(ValueError, match="not valid JSON") as error:
        load_articles(path)

    assert str(path) in str(error.value)
    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def test_top_level_must_be_list(tmp_path):
    path = _write_kb(tmp_path, {"articles": [_valid_article()]})

    with pytest.raises(ValueError, match="must contain a JSON list"):
        load_articles(path)


def test_article_must_be_object(tmp_path):
    path = _write_kb(tmp_path, [_valid_article(), "not an article"])

    with pytest.raises(ValueError, match=r"article #1 must be a JSON object"):
        load_articles(path)


@pytest.mark.parametrize("field", ["id", "title", "text", "url"])
def test_missing_required_field_is_rejected(tmp_path, field):
    article = _valid_article()
    del article[field]
    path = _write_kb(tmp_path, [article])

    with pytest.raises(ValueError, match=f"missing required field\\(s\\): {field}"):
        load_articles(path)


@pytest.mark.parametrize("field", ["id", "title", "text", "url"])
@pytest.mark.parametrize("value", ["", "   \n\t"])
def test_empty_field_is_rejected(tmp_path, field, value):
    path = _write_kb(tmp_path, [_valid_article(**{field: value})])

    with pytest.raises(ValueError, match=f"field '{field}' must not be empty"):
        load_articles(path)


@pytest.mark.parametrize("field", ["id", "title", "text", "url"])
@pytest.mark.parametrize("value", [None, 1, ["text"]])
def test_non_string_field_is_rejected(tmp_path, field, value):
    path = _write_kb(tmp_path, [_valid_article(**{field: value})])

    with pytest.raises(ValueError, match=f"field '{field}' must be a string"):
        load_articles(path)


def test_duplicate_ids_are_rejected(tmp_path):
    path = _write_kb(tmp_path, [_valid_article(), _valid_article(title="Other")])

    with pytest.raises(ValueError, match="article #1 has duplicate id 'KB-T01'"):
        load_articles(path)
