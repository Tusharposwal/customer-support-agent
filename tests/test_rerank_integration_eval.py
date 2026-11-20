"""Offline tests for evals/evaluate_rerank_integration.py. No Cohere calls."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS_DIR))

import calibrate_semantic as cal  # noqa: E402
import evaluate_rerank_integration as ev  # noqa: E402
import rerank_experiment as rx  # noqa: E402
from support_agent.kb import load_articles  # noqa: E402
from support_agent.rerank import RerankConfig, rerank_articles  # noqa: E402

ARTICLES = load_articles(cal.HISTORICAL_KB_PATH)
ORIGINAL = cal.load_cases(cal.DATASET_PATH)
HOLDOUT = cal.load_cases(rx.HOLDOUT_DATASET_PATH)


@pytest.fixture(scope="module", autouse=True)
def v1_knowledge_base():
    """The saved fixtures were made from KB v1; the real hybrid retriever's
    keyword half must read the same articles (set before any module fixture)."""
    with cal.historical_kb():
        yield


@pytest.fixture(scope="module")
def document_fixture():
    return cal.load_fixture(ARTICLES, ORIGINAL, cal.FIXTURE_PATH)


@pytest.fixture(scope="module")
def original_rerank():
    return rx.load_results(ARTICLES, ORIGINAL, rx.RESULTS_PATH, "original")


@pytest.fixture(scope="module")
def original_results(document_fixture, original_rerank):
    return ev.evaluate(ARTICLES, ORIGINAL, document_fixture, original_rerank, thresholds=(0.0, 0.3, 1.0))


# --- Replay client -------------------------------------------------------------


def test_replay_returns_saved_scores_for_documents_in_any_order(original_rerank):
    client = ev.ReplayClient(original_rerank, ARTICLES)
    query = ORIGINAL[0]["query"]
    saved = {i["article_id"]: i["relevance_score"] for i in original_rerank["results"][query]["results"]}
    reordered = list(reversed(ARTICLES))

    result = rerank_articles(query, reordered, client, RerankConfig(enabled=True, model=original_rerank["model"]))

    assert {a["id"]: a["rerank_score"] for a in result} == saved
    assert [a["rerank_score"] for a in result] == sorted(saved.values(), reverse=True)


@pytest.mark.parametrize("change", [
    {"model": "other-model"},
    {"query": "a question with no saved result"},
    {"documents": ["Title: changed\nText: changed"] * 3},
    {"top_n": 1},
])
def test_replay_refuses_anything_it_cannot_reproduce(original_rerank, change):
    client = ev.ReplayClient(original_rerank, ARTICLES)
    call = {"model": original_rerank["model"], "query": ORIGINAL[0]["query"],
            "documents": [rx.document_string(a) for a in ARTICLES], **change}
    with pytest.raises(ev.EvaluationError):
        client.rerank(**call)


def test_replay_client_refuses_chat(original_rerank):
    with pytest.raises(ev.EvaluationError):
        ev.ReplayClient(original_rerank, ARTICLES).chat(model="x")


def test_replay_failure_is_reported_not_hidden_by_fail_open(original_rerank):
    """The agent fails open on rerank errors; the evaluation must stop instead."""
    client = ev.ReplayClient(original_rerank, ARTICLES)
    wrong_model = RerankConfig(enabled=True, model="not-the-saved-model")

    with pytest.raises(ev.EvaluationError, match="Replay failed"):
        ev.run_configuration(ORIGINAL[:1], lambda q: ARTICLES, client, wrong_model)


# --- Evaluation through the real agent -----------------------------------------


def test_every_configuration_is_reported(original_results):
    assert list(original_results) == [
        "Hybrid only", "Hybrid + rerank", "Rerank, min 0.00", "Rerank, min 0.30", "Rerank, min 1.00",
    ]


def test_hybrid_always_passes_all_three_articles(original_results):
    assert set(original_results["Hybrid only"]["candidates"]) == {3}


def test_hybrid_only_matches_the_experiment_hybrid_row(document_fixture, original_rerank, original_results):
    expected = rx.compare_methods(ARTICLES, ORIGINAL, document_fixture, original_rerank)["Hybrid"]
    got = original_results["Hybrid only"]["metrics"]
    assert {k: got[k] for k in expected} == expected
    assert original_results["Hybrid only"]["reasons"] == {ev.SENT: len(ORIGINAL)}


def test_rerank_without_threshold_matches_the_saved_rerank_ranking(original_rerank, original_results):
    expected = rx.ranked_ids(rx.rerank_rows(original_rerank, ORIGINAL))
    assert original_results["Hybrid + rerank"]["returned"] == expected


def test_threshold_zero_equals_no_threshold_and_one_hands_off_everything(original_results):
    assert original_results["Rerank, min 0.00"]["returned"] == original_results["Hybrid + rerank"]["returned"]
    everything_cut = original_results["Rerank, min 1.00"]
    assert everything_cut["reasons"] == {"below_rerank_threshold": len(ORIGINAL)}
    assert everything_cut["metrics"]["false_reject"] == 10
    assert everything_cut["metrics"]["abstain"] == 7


def test_threshold_results_match_the_saved_scores(original_rerank, original_results):
    for case in ORIGINAL:
        saved = original_rerank["results"][case["query"]]["results"]
        kept = sorted((i for i in saved if i["relevance_score"] >= 0.3),
                      key=lambda i: i["relevance_score"], reverse=True)
        assert original_results["Rerank, min 0.30"]["returned"][case["query"]] == [i["article_id"] for i in kept]


def test_report_prints_each_configuration(original_results, capsys):
    ev.print_report("original", ORIGINAL, original_results, "rerank-v5.0-pro")
    out = capsys.readouterr().out
    assert "ORIGINAL (17 questions: 10 answerable, 7 unsupported)" in out
    assert "Hybrid only" in out and "Rerank, min 0.30" in out


# --- Holdout embeddings: generation (fake client) and validation ---------------


class FakeEmbedClient:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(embeddings=SimpleNamespace(float=self.vectors))


def _vectors(document_fixture, count=len(HOLDOUT)):
    return [[0.01 * (i + 1)] * document_fixture["dimensions"] for i in range(count)]


def test_generation_makes_exactly_one_batched_query_call(tmp_path, document_fixture):
    client = FakeEmbedClient(_vectors(document_fixture))
    path = tmp_path / "holdout.json"

    fixture = ev.generate_holdout_embeddings(client, "embed-v4.0", ARTICLES, HOLDOUT, document_fixture, path)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["texts"] == [case["query"] for case in HOLDOUT]
    assert call["model"] == "embed-v4.0"
    assert call["input_type"] == "search_query"
    assert call["embedding_types"] == ["float"]
    assert call["request_options"] == {"max_retries": 0}
    assert fixture["embed_calls"] == [{"model": "embed-v4.0", "input_type": "search_query", "texts": 20}]

    combined = ev.load_holdout_embeddings(ARTICLES, HOLDOUT, document_fixture, path)
    assert combined["documents"] is document_fixture["documents"]
    assert len(combined["questions"]) == 20


def test_generation_refuses_to_overwrite(tmp_path, document_fixture):
    path = tmp_path / "holdout.json"
    path.write_text("{}", encoding="utf-8")
    client = FakeEmbedClient(_vectors(document_fixture))

    with pytest.raises(cal.FixtureError, match="already exists"):
        ev.generate_holdout_embeddings(client, "embed-v4.0", ARTICLES, HOLDOUT, document_fixture, path)
    assert client.calls == []


def test_generation_refuses_a_different_model_before_calling(tmp_path, document_fixture):
    client = FakeEmbedClient(_vectors(document_fixture))
    with pytest.raises(cal.FixtureError, match="differs"):
        ev.generate_holdout_embeddings(client, "embed-other", ARTICLES, HOLDOUT, document_fixture,
                                       tmp_path / "h.json")
    assert client.calls == []


def test_generation_refuses_the_wrong_question_count_before_calling(tmp_path, document_fixture):
    client = FakeEmbedClient(_vectors(document_fixture))
    with pytest.raises(cal.FixtureError):
        ev.generate_holdout_embeddings(client, "embed-v4.0", ARTICLES, HOLDOUT[:5], document_fixture,
                                       tmp_path / "h.json")
    assert client.calls == []


@pytest.mark.parametrize("vectors", ["short", "wrong-dims"])
def test_generation_rejects_bad_vectors_and_writes_nothing(tmp_path, document_fixture, vectors):
    bad = _vectors(document_fixture, 19) if vectors == "short" else [[0.1, 0.2]] * 20
    path = tmp_path / "h.json"
    with pytest.raises(cal.FixtureError):
        ev.generate_holdout_embeddings(FakeEmbedClient(bad), "embed-v4.0", ARTICLES, HOLDOUT,
                                       document_fixture, path)
    assert not path.exists()


@pytest.mark.parametrize("mutate, message", [
    (lambda f: f.update(version=99), "version"),
    (lambda f: f.update(dataset="original"), "not a holdout"),
    (lambda f: f.update(kb_sha256="0"), "Knowledge base"),
    (lambda f: f.update(questions_sha256="0"), "questions changed"),
    (lambda f: f.update(model="embed-other"), "settings"),
    (lambda f: f["questions"].reverse(), "dataset order"),
    (lambda f: f["questions"][0].update(embedding=[0.1]), "dimensions"),
])
def test_loading_rejects_stale_or_inconsistent_fixtures(tmp_path, document_fixture, mutate, message):
    path = tmp_path / "h.json"
    ev.generate_holdout_embeddings(FakeEmbedClient(_vectors(document_fixture)), "embed-v4.0",
                                   ARTICLES, HOLDOUT, document_fixture, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(cal.FixtureError, match=message):
        ev.load_holdout_embeddings(ARTICLES, HOLDOUT, document_fixture, path)


def test_missing_holdout_fixture_points_to_the_generate_option(tmp_path, document_fixture):
    with pytest.raises(cal.FixtureError, match="--embed-holdout"):
        ev.load_holdout_embeddings(ARTICLES, HOLDOUT, document_fixture, tmp_path / "missing.json")


@pytest.mark.skipif(not ev.HOLDOUT_EMBEDDINGS_PATH.exists(), reason="holdout embeddings not generated")
def test_saved_holdout_fixture_is_valid_and_used_one_call(document_fixture):
    combined = ev.load_holdout_embeddings(ARTICLES, HOLDOUT, document_fixture)
    saved = json.loads(ev.HOLDOUT_EMBEDDINGS_PATH.read_text(encoding="utf-8"))

    assert len(combined["questions"]) == 20
    assert saved["embed_calls"] == [{"model": "embed-v4.0", "input_type": "search_query", "texts": 20}]


# --- Command line ----------------------------------------------------------------


def test_cli_rejects_thresholds_outside_zero_to_one():
    with pytest.raises(SystemExit):
        ev.main(["--thresholds", "1.5"])


def test_offline_run_imports_no_cohere_and_makes_no_calls():
    code = (
        "import sys; sys.path.insert(0, r'%s'); import evaluate_rerank_integration as ev; "
        "code = ev.main(['--dataset', 'original', '--thresholds', '0.3']); "
        "assert code == 0, code; "
        "assert 'cohere' not in sys.modules; "
        "assert 'support_agent.semantic_kb' not in sys.modules" % EVALS_DIR
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert "Hybrid + rerank" in completed.stdout
