"""Offline tests for evals/rerank_experiment.py. No Cohere imports or calls."""

import json
import sys
import types
from pathlib import Path

import pytest

import support_agent

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS_DIR))

import calibrate_semantic as cal  # noqa: E402
import rerank_experiment as rx  # noqa: E402

ARTICLES = [
    {"id": "KB-A", "title": "Alpha", "text": "About alpha.", "url": "https://x/a"},
    {"id": "KB-B", "title": "Beta", "text": "About beta.", "url": "https://x/b"},
    {"id": "KB-C", "title": "Gamma", "text": "About gamma.", "url": "https://x/c"},
]
CASES = [
    {"query": "alpha question", "expected_id": "KB-A"},
    {"query": "gamma question", "expected_id": "KB-C"},
    {"query": "unrelated question", "expected_id": None},
]
# Per query: (index, score) pairs in the order the API returns them.
RERANK = {
    "alpha question": [(0, 0.9), (2, 0.2), (1, 0.1)],
    "gamma question": [(2, 0.8), (0, 0.3), (1, 0.05)],
    "unrelated question": [(1, 0.04), (0, 0.02), (2, 0.01)],
}


class FakeApiError(Exception):
    def __init__(self, status_code, body):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


class FakeClient:
    """Records every rerank request; `errors` maps call number -> exception."""

    def __init__(self, errors=None, unavailable_models=()):
        self.requests = []
        self.errors = errors or {}
        self.unavailable_models = set(unavailable_models)

    def rerank(self, **kwargs):
        self.requests.append(kwargs)
        number = len(self.requests)
        if number in self.errors:
            raise self.errors[number]
        if kwargs["model"] in self.unavailable_models:
            raise FakeApiError(404, {"message": f"model '{kwargs['model']}' not found"})
        return types.SimpleNamespace(
            results=[
                types.SimpleNamespace(index=i, relevance_score=s) for i, s in RERANK[kwargs["query"]]
            ],
            meta=types.SimpleNamespace(billed_units=types.SimpleNamespace(search_units=1)),
        )


def _clock():
    ticks = iter(range(1000))
    return lambda: next(ticks) * 0.25  # every call takes 250 ms


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(rx, "EXPECTED_ARTICLES", 3)
    monkeypatch.setattr(rx, "EXPECTED_QUESTIONS", 3)
    monkeypatch.setattr(rx, "MAX_SUCCESSFUL_CALLS", 3)


@pytest.fixture
def block_cohere(monkeypatch):
    monkeypatch.setitem(sys.modules, "support_agent.semantic_kb", None)
    monkeypatch.setitem(sys.modules, "cohere", None)
    monkeypatch.delattr(support_agent, "semantic_kb", raising=False)


def _run(client, path, configured=rx.PRIMARY_MODEL, cases=CASES):
    return rx.run(client, ARTICLES, cases, configured, path, clock=_clock(), log=lambda *_: None)


# --- Requests and mapping -----------------------------------------------------


def test_document_string_has_title_and_text_but_no_url():
    text = rx.document_string(ARTICLES[0])
    assert text == "Title: Alpha\nText: About alpha."
    assert "https://" not in text


def test_request_construction(tmp_path, small_limits):
    client = FakeClient()
    _run(client, tmp_path / "r.json")

    assert len(client.requests) == 3
    first = client.requests[0]
    assert first == {
        "model": rx.PRIMARY_MODEL,
        "query": "alpha question",
        "documents": [rx.document_string(a) for a in ARTICLES],
        "request_options": {"max_retries": 0},
    }


def test_indices_map_back_to_article_ids(tmp_path, small_limits):
    path = tmp_path / "r.json"
    _run(FakeClient(), path)
    entry = json.loads(path.read_text())["results"]["gamma question"]

    assert [(r["index"], r["article_id"]) for r in entry["results"]] == [
        (2, "KB-C"), (0, "KB-A"), (1, "KB-B"),
    ]
    row = rx.rerank_rows(rx.load_results(ARTICLES, CASES, path), CASES)[1]
    assert row["scores"] == {"KB-C": 0.8, "KB-A": 0.3, "KB-B": 0.05}
    assert row["top_id"] == "KB-C"


def test_saved_metadata(tmp_path, small_limits):
    path = tmp_path / "r.json"
    _run(FakeClient(), path)
    data = json.loads(path.read_text())

    assert data["model"] == rx.PRIMARY_MODEL
    assert data["document_format"] == rx.DOCUMENT_FORMAT
    assert data["kb_sha256"] == cal.kb_hash(ARTICLES)
    assert data["questions_sha256"] == cal.questions_hash(CASES)
    assert data["dataset_sha256"] == rx.dataset_hash(CASES)
    entry = data["results"]["alpha question"]
    assert entry["expected_id"] == "KB-A"
    assert entry["search_units"] == 1
    assert entry["latency_ms"] == 250.0
    assert entry["completed_at"]


# --- Model selection and fallback ---------------------------------------------


def test_primary_model_is_default(monkeypatch):
    monkeypatch.delenv(rx.MODEL_ENV, raising=False)
    assert rx.configured_model() == "rerank-v5.0-pro"
    monkeypatch.setenv(rx.MODEL_ENV, "custom-model")
    assert rx.configured_model() == "custom-model"


def test_fallback_when_primary_unavailable(tmp_path, small_limits):
    client = FakeClient(unavailable_models={rx.PRIMARY_MODEL})
    stats = _run(client, tmp_path / "r.json")

    assert [r["model"] for r in client.requests] == [rx.PRIMARY_MODEL] + [rx.FALLBACK_MODEL] * 3
    assert stats == {"successful": 3, "failed": 1, "fallback_used": True, "model": rx.FALLBACK_MODEL}
    assert json.loads((tmp_path / "r.json").read_text())["model"] == rx.FALLBACK_MODEL


def test_stops_when_fallback_also_unavailable(tmp_path, small_limits):
    client = FakeClient(unavailable_models={rx.PRIMARY_MODEL, rx.FALLBACK_MODEL})

    with pytest.raises(FakeApiError):
        _run(client, tmp_path / "r.json")
    assert len(client.requests) == 2
    assert not (tmp_path / "r.json").exists()


@pytest.mark.parametrize(
    "error",
    [
        FakeApiError(429, {"message": "rate limit"}),
        FakeApiError(401, {"message": "invalid api token"}),
        FakeApiError(500, {"message": "model server error"}),
        FakeApiError(400, {"message": "too many documents"}),
        RuntimeError("network down"),
    ],
)
def test_no_fallback_for_unrelated_errors(tmp_path, small_limits, error):
    client = FakeClient(errors={1: error})

    with pytest.raises(type(error)):
        _run(client, tmp_path / "r.json")
    assert len(client.requests) == 1


def test_no_fallback_for_non_primary_model(tmp_path, small_limits):
    client = FakeClient(unavailable_models={"custom-model"})
    with pytest.raises(FakeApiError):
        _run(client, tmp_path / "r.json", configured="custom-model")
    assert len(client.requests) == 1


def test_no_fallback_after_primary_already_succeeded(tmp_path, small_limits):
    error = FakeApiError(404, {"message": "model not found"})
    client = FakeClient(errors={2: error})

    with pytest.raises(FakeApiError):
        _run(client, tmp_path / "r.json")
    assert [r["model"] for r in client.requests] == [rx.PRIMARY_MODEL] * 2


def test_saved_model_is_locked(tmp_path, small_limits):
    path = tmp_path / "r.json"
    _run(FakeClient(unavailable_models={rx.PRIMARY_MODEL}), path, cases=CASES)
    saved = rx.load_results(ARTICLES, CASES, path)

    assert rx.choose_model(saved, rx.PRIMARY_MODEL) == (rx.FALLBACK_MODEL, False)
    with pytest.raises(rx.ResultsError, match="mixing models"):
        rx.choose_model(saved, "custom-model")


# --- Saved results: validation, resume, limits ---------------------------------


def test_each_result_saved_immediately(tmp_path, small_limits):
    path = tmp_path / "r.json"
    client = FakeClient(errors={2: FakeApiError(429, {"message": "rate limit"})})

    with pytest.raises(FakeApiError):
        _run(client, path)
    assert list(json.loads(path.read_text())["results"]) == ["alpha question"]


def test_resume_only_requests_missing_questions(tmp_path, small_limits):
    path = tmp_path / "r.json"
    with pytest.raises(FakeApiError):
        _run(FakeClient(errors={2: FakeApiError(429, {})}), path)
    first_entry = json.loads(path.read_text())["results"]["alpha question"]

    client = FakeClient()
    stats = _run(client, path)
    assert [r["query"] for r in client.requests] == ["gamma question", "unrelated question"]
    assert stats["successful"] == 2
    assert json.loads(path.read_text())["results"]["alpha question"] == first_entry

    again = FakeClient()
    _run(again, path)
    assert again.requests == []


def test_call_limit_is_checked_before_any_request(tmp_path, small_limits, monkeypatch):
    monkeypatch.setattr(rx, "MAX_SUCCESSFUL_CALLS", 2)
    client = FakeClient()

    with pytest.raises(rx.ResultsError, match="limit is 2"):
        _run(client, tmp_path / "r.json")
    assert client.requests == []


def test_dataset_and_kb_sizes_are_enforced(small_limits):
    with pytest.raises(rx.ResultsError, match="Expected 3 questions"):
        rx.preflight(ARTICLES, CASES[:2], None, rx.PRIMARY_MODEL)
    with pytest.raises(rx.ResultsError, match="Expected 3 articles"):
        rx.preflight(ARTICLES[:2], CASES, None, rx.PRIMARY_MODEL)


def test_real_limits_match_the_approval():
    assert (rx.EXPECTED_QUESTIONS, rx.EXPECTED_ARTICLES, rx.MAX_SUCCESSFUL_CALLS) == (17, 3, 17)
    assert rx.REQUEST_OPTIONS == {"max_retries": 0}


@pytest.fixture
def saved_path(tmp_path, small_limits):
    path = tmp_path / "r.json"
    _run(FakeClient(), path)
    return path


def test_kb_hash_mismatch_is_rejected(saved_path):
    changed = [dict(ARTICLES[0], text="Edited."), *ARTICLES[1:]]
    with pytest.raises(rx.ResultsError, match="Knowledge base changed"):
        rx.load_results(changed, CASES, saved_path)


def test_question_hash_mismatch_is_rejected(saved_path):
    changed = CASES + [{"query": "new", "expected_id": None}]
    with pytest.raises(rx.ResultsError, match="Question list changed"):
        rx.load_results(ARTICLES, changed, saved_path)


def _corrupt(path, mutate):
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))


def test_wrong_article_mapping_is_rejected(saved_path):
    _corrupt(saved_path, lambda d: d["results"]["alpha question"]["results"][0].update(article_id="KB-B"))
    with pytest.raises(rx.ResultsError, match="wrong article"):
        rx.load_results(ARTICLES, CASES, saved_path)


def test_missing_index_is_rejected(saved_path):
    _corrupt(saved_path, lambda d: d["results"]["alpha question"]["results"].pop())
    with pytest.raises(rx.ResultsError, match="exactly once"):
        rx.load_results(ARTICLES, CASES, saved_path)


def test_mixed_models_are_rejected(saved_path):
    _corrupt(saved_path, lambda d: d["results"]["alpha question"].update(model="other"))
    with pytest.raises(rx.ResultsError, match="different model"):
        rx.load_results(ARTICLES, CASES, saved_path)


def test_invalid_results_are_never_rerequested(saved_path):
    _corrupt(saved_path, lambda d: d["results"]["alpha question"]["results"].pop())
    client = FakeClient()
    with pytest.raises(rx.ResultsError):
        _run(client, saved_path)
    assert client.requests == []


# --- Offline evaluation -------------------------------------------------------


def test_method_metrics_definitions():
    returned = {"alpha question": ["KB-A", "KB-B"], "gamma question": ["KB-A", "KB-C"], "unrelated question": []}
    m = rx.method_metrics(CASES, returned)
    assert (m["top1"], m["recall3"], m["wrong_article"]) == (1, 2, 2)
    assert (m["false_accept"], m["abstain"]) == (0, 1)


def _embedding_fixture():
    vectors = {"KB-A": [1, 0, 0], "KB-B": [0, 1, 0], "KB-C": [0, 0, 1]}
    questions = {"alpha question": [0.9, 0.1, 0], "gamma question": [0.1, 0, 0.9], "unrelated question": [0.3, 0.3, 0.3]}
    return {
        "documents": [{"id": k, "embedding": v} for k, v in vectors.items()],
        "questions": [{"query": c["query"], "embedding": questions[c["query"]]} for c in CASES],
    }


def test_offline_comparison_uses_production_hybrid(saved_path, block_cohere, monkeypatch):
    import support_agent.kb as kb

    monkeypatch.setattr(kb, "load_articles", lambda path=None: ARTICLES)
    # Only the "Keyword" row is stubbed; hybrid runs the real kb.retrieve on ARTICLES.
    monkeypatch.setattr(rx, "retrieve", lambda q, top_k=3: [])
    data = rx.load_results(ARTICLES, CASES, saved_path)

    results = rx.compare_methods(ARTICLES, CASES, _embedding_fixture(), data)

    assert results["Semantic"]["top1"] == 2
    assert results["Hybrid"]["top1"] == 2  # real RRF over semantic-only ranks
    assert results["Hybrid"]["false_accept"] == 1
    assert results["Rerank"]["top1"] == 2
    # The real semantic module was never imported, and the block is restored.
    assert sys.modules["support_agent.semantic_kb"] is None
    assert "support_agent.hybrid_kb" not in sys.modules


def test_usage_summary(saved_path):
    usage = rx.usage_summary(rx.load_results(ARTICLES, CASES, saved_path))
    assert usage == {"calls": 3, "model": rx.PRIMARY_MODEL, "search_units": 3,
                     "avg_latency_ms": 250.0, "max_latency_ms": 250.0}


# --- The real saved results ---------------------------------------------------


@pytest.mark.skipif(not rx.RESULTS_PATH.exists(), reason="rerank results not generated")
def test_saved_rerank_results_are_complete_and_valid(block_cohere):
    from support_agent.kb import load_articles

    cases = cal.load_cases()
    data = rx.load_results(load_articles(), cases)

    assert len(data["results"]) == len(cases) == 17
    assert data["model"] in (rx.PRIMARY_MODEL, rx.FALLBACK_MODEL)
    assert all(len(entry["results"]) == 3 for entry in data["results"].values())


# --- Dataset selection (original vs holdout) ----------------------------------


def test_dataset_configs_are_separate():
    original, holdout = rx.DATASETS["original"], rx.DATASETS["holdout"]

    assert rx.DEFAULT_DATASET == "original"
    assert (original.queries_path, original.results_path) == (cal.DATASET_PATH, rx.RESULTS_PATH)
    assert (original.expected_questions, original.max_calls) == (17, 17)
    assert holdout.queries_path.name == "rerank_holdout_queries.json"
    assert holdout.results_path.name == "rerank_holdout_results.json"
    assert (holdout.expected_questions, holdout.max_calls) == (20, 20)
    assert original.results_path != holdout.results_path


def _run_dataset(client, path, dataset, max_calls=3):
    return rx.run(
        client, ARTICLES, CASES, rx.PRIMARY_MODEL, path,
        dataset=dataset, expected_questions=3, max_calls=max_calls,
        clock=_clock(), log=lambda *_: None,
    )


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """main() with temporary datasets and no .env loading."""
    import dotenv

    for name in ("original", "holdout"):
        queries = tmp_path / f"{name}_queries.json"
        queries.write_text(json.dumps(CASES), encoding="utf-8")
        monkeypatch.setitem(rx.DATASETS, name, rx.DatasetConfig(
            name, queries, tmp_path / f"{name}_results.json", 3, 3,
        ))
    monkeypatch.setattr(rx, "load_articles", lambda: ARTICLES)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv(rx.MODEL_ENV, raising=False)
    return tmp_path


def test_cli_defaults_to_original_dataset(cli, capsys, small_limits):
    assert rx.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dataset=original" in out
    assert "results=original_results.json" in out
    assert "max successful calls=3" in out


def test_cli_selects_holdout_dataset_and_limit(cli, capsys, small_limits):
    assert rx.main(["--dry-run", "--dataset", "holdout"]) == 0
    out = capsys.readouterr().out
    assert "dataset=holdout" in out
    assert "results=holdout_results.json" in out
    assert "to request=3" in out


def test_cli_holdout_limit_is_enforced_before_any_call(cli, capsys, small_limits, monkeypatch):
    # Holdout limit below its 3 questions: refused in preflight, before a client exists.
    config = rx.DATASETS["holdout"]
    monkeypatch.setitem(rx.DATASETS, "holdout", rx.DatasetConfig(
        config.name, config.queries_path, config.results_path, 3, 2,
    ))
    assert rx.main(["--run", "--dataset", "holdout"]) == 1
    assert "limit is 2" in capsys.readouterr().err
    assert not (cli / "holdout_results.json").exists()


def test_dry_run_never_writes_results(cli, small_limits):
    rx.main(["--dry-run", "--dataset", "holdout"])
    assert not (cli / "holdout_results.json").exists()


def test_preflight_uses_per_dataset_limits():
    twenty = [{"query": f"q{i}", "expected_id": None} for i in range(20)]

    plan = rx.preflight(ARTICLES, twenty, None, rx.PRIMARY_MODEL, expected_questions=20, max_calls=20)
    assert plan["max_calls"] == 20 and len(plan["missing"]) == 20

    with pytest.raises(rx.ResultsError, match="Expected 17 questions"):
        rx.preflight(ARTICLES, twenty, None, rx.PRIMARY_MODEL)  # original defaults
    with pytest.raises(rx.ResultsError, match="limit is 17"):
        rx.preflight(ARTICLES, twenty, None, rx.PRIMARY_MODEL, expected_questions=20, max_calls=17)


def test_run_stops_at_dataset_call_limit(tmp_path):
    client = FakeClient()
    with pytest.raises(rx.ResultsError, match="limit is 2"):
        _run_dataset(client, tmp_path / "r.json", "holdout", max_calls=2)
    assert client.requests == []


def test_holdout_results_record_their_dataset(tmp_path):
    path = tmp_path / "r.json"
    _run_dataset(FakeClient(), path, "holdout")
    assert json.loads(path.read_text())["dataset"] == "holdout"


def test_original_and_holdout_results_cannot_be_mixed(tmp_path):
    holdout_path = tmp_path / "holdout.json"
    _run_dataset(FakeClient(), holdout_path, "holdout")

    with pytest.raises(rx.ResultsError, match="holds 'holdout' results, not 'original'"):
        rx.load_results(ARTICLES, CASES, holdout_path, "original")
    client = FakeClient()
    with pytest.raises(rx.ResultsError):
        _run_dataset(client, holdout_path, "original")
    assert client.requests == []


def test_legacy_file_without_dataset_field_is_original_only(tmp_path):
    path = tmp_path / "legacy.json"
    _run_dataset(FakeClient(), path, "original")
    _corrupt(path, lambda d: d.pop("dataset"))

    assert rx.load_results(ARTICLES, CASES, path, "original") is not None
    with pytest.raises(rx.ResultsError, match="not 'holdout'"):
        rx.load_results(ARTICLES, CASES, path, "holdout")


@pytest.mark.parametrize("dataset", ["original", "holdout"])
def test_hash_validation_for_both_datasets(tmp_path, dataset):
    path = tmp_path / "r.json"
    _run_dataset(FakeClient(), path, dataset)

    with pytest.raises(rx.ResultsError, match="Knowledge base changed"):
        rx.load_results([dict(ARTICLES[0], title="Edited"), *ARTICLES[1:]], CASES, path, dataset)
    with pytest.raises(rx.ResultsError, match="Question list changed"):
        rx.load_results(ARTICLES, CASES[:2] + [{"query": "other", "expected_id": None}], path, dataset)


@pytest.mark.parametrize("dataset", ["original", "holdout"])
def test_resume_for_both_datasets(tmp_path, dataset):
    path = tmp_path / "r.json"
    with pytest.raises(FakeApiError):
        _run_dataset(FakeClient(errors={2: FakeApiError(429, {})}), path, dataset)

    client = FakeClient()
    assert _run_dataset(client, path, dataset)["successful"] == 2
    assert [r["query"] for r in client.requests] == ["gamma question", "unrelated question"]

    again = FakeClient()
    _run_dataset(again, path, dataset)
    assert again.requests == []


def test_comparison_without_embeddings_shows_keyword_and_rerank_only(saved_path, block_cohere):
    data = rx.load_results(ARTICLES, CASES, saved_path)
    results = rx.compare_methods(ARTICLES, CASES, None, data)

    assert list(results) == ["Keyword", "Rerank"]
    assert "support_agent.hybrid_kb" not in sys.modules


def test_real_holdout_dataset_passes_preflight():
    from support_agent.kb import load_articles

    config = rx.DATASETS["holdout"]
    cases = cal.load_cases(config.queries_path)
    plan = rx.preflight(load_articles(), cases, None, rx.PRIMARY_MODEL,
                        expected_questions=config.expected_questions, max_calls=config.max_calls)

    assert (plan["questions"], len(plan["missing"]), plan["max_calls"]) == (20, 20, 20)


@pytest.mark.skipif(not rx.RESULTS_PATH.exists(), reason="rerank results not generated")
def test_real_original_results_are_rejected_as_holdout(block_cohere):
    from support_agent.kb import load_articles

    with pytest.raises(rx.ResultsError, match="not 'holdout'"):
        rx.load_results(load_articles(), cal.load_cases(), rx.RESULTS_PATH, "holdout")
