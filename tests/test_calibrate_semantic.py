"""Offline tests for evals/calibrate_semantic.py. No Cohere calls are made."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

import support_agent
from support_agent.kb import load_articles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "evals" / "calibrate_semantic.py"

_spec = importlib.util.spec_from_file_location("calibrate_semantic", SCRIPT_PATH)
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)

ARTICLES = [
    {"id": "KB-A", "title": "Alpha", "text": "About alpha.", "url": "https://x/a"},
    {"id": "KB-B", "title": "Beta", "text": "About beta.", "url": "https://x/b"},
]
CASES = [
    {"query": "alpha question", "expected_id": "KB-A"},
    {"query": "beta question", "expected_id": "KB-B"},
    {"query": "unrelated question", "expected_id": None},
]
DOC_VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
QUESTION_VECTORS = [[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.3, 0.3, 0.9]]


def _fixture(question_vectors=QUESTION_VECTORS):
    return {
        "version": cal.FIXTURE_VERSION,
        "model": "fake-model",
        "document_input_type": "search_document",
        "query_input_type": "search_query",
        "dimensions": 3,
        "kb_sha256": cal.kb_hash(ARTICLES),
        "questions_sha256": cal.questions_hash(CASES),
        "documents": [{"id": a["id"], "embedding": v} for a, v in zip(ARTICLES, DOC_VECTORS)],
        "questions": [{"query": c["query"], "embedding": v} for c, v in zip(CASES, question_vectors)],
    }


def _write(tmp_path, fixture):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path


@pytest.fixture
def block_cohere(monkeypatch):
    """Make any attempt to import semantic_kb or cohere fail loudly."""
    monkeypatch.setitem(sys.modules, "support_agent.semantic_kb", None)
    monkeypatch.setitem(sys.modules, "cohere", None)
    monkeypatch.delattr(support_agent, "semantic_kb", raising=False)


# --- Scoring ------------------------------------------------------------------


def test_cosine_similarity():
    assert cal.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cal.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cal.cosine_similarity([0, 0], [1, 1]) == 0.0


def test_score_rows_use_expected_score_or_max_score():
    rows = cal.score_questions(_fixture(), CASES)

    alpha, beta, unrelated = rows
    assert alpha["top_id"] == "KB-A"
    assert alpha["key_score"] == alpha["scores"]["KB-A"]
    assert beta["key_score"] == beta["scores"]["KB-B"]
    assert unrelated["key_score"] == max(unrelated["scores"].values())
    assert alpha["margin"] == pytest.approx(alpha["scores"]["KB-A"] - alpha["scores"]["KB-B"])


def test_threshold_metrics_count_each_error_type():
    rows = cal.score_questions(_fixture(), CASES)
    unrelated_max = rows[2]["key_score"]

    at_zero = cal.threshold_metrics(rows, 0.0)
    assert (at_zero["false_rejections"], at_zero["false_acceptances"], at_zero["wrong_article"]) == (0, 1, 2)

    above_all = cal.threshold_metrics(rows, 1.01)
    assert (above_all["false_rejections"], above_all["false_acceptances"]) == (2, 0)

    just_above_unrelated = cal.threshold_metrics(rows, unrelated_max + 1e-9)
    assert just_above_unrelated["false_acceptances"] == 0
    assert just_above_unrelated["false_rejections"] == 0


def test_clean_separation_gives_interval_and_midpoint():
    s = cal.separation(cal.score_questions(_fixture(), CASES))

    assert s["clean"] is True
    assert s["lowest_answerable"] > s["highest_unsupported"]
    assert s["midpoint"] == pytest.approx((s["lowest_answerable"] + s["highest_unsupported"]) / 2)


def test_overlap_is_reported_without_midpoint():
    # The unrelated question now points almost exactly at KB-A.
    vectors = QUESTION_VECTORS[:2] + [[1.0, 0.05, 0.0]]
    s = cal.separation(cal.score_questions(_fixture(vectors), CASES))

    assert s["clean"] is False
    assert s["midpoint"] is None


# --- Fixture validation -------------------------------------------------------


def test_valid_fixture_loads_without_cohere(tmp_path, block_cohere):
    fixture = cal.load_fixture(ARTICLES, CASES, _write(tmp_path, _fixture()))
    assert fixture["dimensions"] == 3


def test_missing_fixture_is_rejected(tmp_path):
    with pytest.raises(cal.FixtureError, match="not found"):
        cal.load_fixture(ARTICLES, CASES, tmp_path / "missing.json")


def test_changed_kb_is_rejected(tmp_path):
    changed = [dict(ARTICLES[0], text="Edited."), ARTICLES[1]]
    with pytest.raises(cal.FixtureError, match="Knowledge base changed"):
        cal.load_fixture(changed, CASES, _write(tmp_path, _fixture()))


def test_changed_questions_are_rejected(tmp_path):
    changed = CASES + [{"query": "new question", "expected_id": None}]
    with pytest.raises(cal.FixtureError, match="Question list changed"):
        cal.load_fixture(ARTICLES, changed, _write(tmp_path, _fixture()))


def test_wrong_dimensions_are_rejected(tmp_path):
    fixture = _fixture()
    fixture["questions"][0]["embedding"] = [1.0, 0.0]
    with pytest.raises(cal.FixtureError, match="dimensions"):
        cal.load_fixture(ARTICLES, CASES, _write(tmp_path, fixture))


def test_offline_main_reports_without_cohere(tmp_path, monkeypatch, block_cohere, capsys):
    monkeypatch.setattr(cal, "load_articles", lambda: ARTICLES)
    monkeypatch.setattr(cal, "load_cases", lambda path: CASES)
    monkeypatch.setattr(cal, "FIXTURE_PATH", _write(tmp_path, _fixture()))

    assert cal.main([]) == 0
    output = capsys.readouterr().out
    assert "CLEAN separation" in output
    assert "Threshold trade-offs" in output


# --- Generation (fake Cohere client) ------------------------------------------


class _FakeClient:
    def __init__(self):
        self.calls = 0

    def embed(self, **kwargs):
        self.calls += 1
        vectors = DOC_VECTORS if kwargs["input_type"] == "search_document" else QUESTION_VECTORS
        return types.SimpleNamespace(embeddings=types.SimpleNamespace(float=vectors[: len(kwargs["texts"])]))


def _fake_semantic_kb(monkeypatch):
    client = _FakeClient()
    module = types.ModuleType("support_agent.semantic_kb")
    module.client = client
    module.MODEL = "fake-model"

    def _embedded_articles():
        response = module.client.embed(
            texts=[cal.document_text(a) for a in ARTICLES],
            model=module.MODEL,
            input_type="search_document",
            embedding_types=["float"],
        )
        return [{**a, "embedding": v} for a, v in zip(ARTICLES, response.embeddings.float)]

    module._embedded_articles = _embedded_articles
    monkeypatch.setitem(sys.modules, "support_agent.semantic_kb", module)
    monkeypatch.setattr(support_agent, "semantic_kb", module, raising=False)
    return client


def test_generate_makes_exactly_two_calls_and_round_trips(tmp_path, monkeypatch):
    client = _fake_semantic_kb(monkeypatch)
    path = tmp_path / "fixtures" / "fixture.json"

    fixture = cal.generate_fixture(ARTICLES, CASES, path)

    assert client.calls == 2
    assert [c["input_type"] for c in fixture["embed_calls"]] == ["search_document", "search_query"]
    assert fixture["model"] == "fake-model"
    assert fixture["dimensions"] == 3
    assert cal.load_fixture(ARTICLES, CASES, path)["questions_sha256"] == cal.questions_hash(CASES)


def test_generate_refuses_to_overwrite(tmp_path, monkeypatch):
    client = _fake_semantic_kb(monkeypatch)
    path = _write(tmp_path, _fixture())

    with pytest.raises(cal.FixtureError, match="already exists"):
        cal.generate_fixture(ARTICLES, CASES, path)
    assert client.calls == 0


# --- The real fixture ---------------------------------------------------------


@pytest.mark.skipif(not cal.FIXTURE_PATH.exists(), reason="semantic fixture not generated")
def test_saved_fixture_matches_current_kb_and_dataset(block_cohere):
    cases = cal.load_cases()
    fixture = cal.load_fixture(load_articles(), cases)

    assert len(fixture["documents"]) == 3
    assert len(fixture["questions"]) == len(cases) == 17
    assert [c["texts"] for c in fixture["embed_calls"]] == [3, 17]
