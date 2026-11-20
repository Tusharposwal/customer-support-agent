"""Offline checks for KB v2, its evaluation dataset and evals/evaluate_kb_v2.py.

No Cohere calls: the live recording path is tested with fakes only.
"""

import copy
import json
import re
import sys
from pathlib import Path

import pytest

import support_agent
from support_agent import kb
from support_agent.answer import SupportAnswer
from support_agent.kb import load_articles

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS_DIR))

import calibrate_semantic as cal  # noqa: E402
import evaluate_kb_v2 as ev  # noqa: E402

ARTICLES = load_articles()
CASES = ev.load_cases()
BY_ID = {article["id"]: article for article in ARTICLES}


@pytest.fixture
def block_cohere(monkeypatch):
    """Any attempt to import the Cohere SDK or the embedding module fails."""
    monkeypatch.setitem(sys.modules, "support_agent.semantic_kb", None)
    monkeypatch.setitem(sys.modules, "support_agent.hybrid_kb", None)
    monkeypatch.setitem(sys.modules, "cohere", None)
    monkeypatch.delattr(support_agent, "semantic_kb", raising=False)
    monkeypatch.delattr(support_agent, "hybrid_kb", raising=False)


# --- The knowledge base ----------------------------------------------------------


def test_kb_size_and_categories():
    assert 25 <= len(ARTICLES) <= 35
    ev.validate_kb(ARTICLES)
    counts = {c: sum(a["category"] == c for a in ARTICLES) for c in ev.CATEGORIES}
    assert all(count >= 2 for count in counts.values()), counts


def test_original_articles_keep_id_title_url_and_text():
    """KB-001..003 only gained sentences: nothing that was true in v1 changed."""
    for old in load_articles(cal.HISTORICAL_KB_PATH):
        new = BY_ID[old["id"]]
        assert (new["title"], new["url"]) == (old["title"], old["url"])
        assert new["text"].startswith(old["text"])


@pytest.mark.parametrize("pattern, value", [
    (r"within (\d+) days of purchase", "30"),                          # refunds, returns, cancel
    (r"within (\d+) days of delivery", "7"),                           # damaged / non-returnable
    (r"at least (\d+) characters", "10"),                              # password rules
    (r"pending charges? [^.]*?(\d+-\d+) business days", "3-5"),         # authorization holds
    (r"[Oo]rder confirmation emails arrive within (\d+) minutes", "15"),
    (r"[Pp]rocessing[^.]*?within (\d+) hour|[Pp]rocessing[^.]*?about (\d+) hour", "1"),
])
def test_shared_policy_numbers_are_consistent(pattern, value):
    """A policy stated in several articles must use the same number everywhere."""
    found = {}
    for article in ARTICLES:
        for match in re.finditer(pattern, article["text"]):
            found.setdefault(next(g for g in match.groups() if g), set()).add(article["id"])
    assert sum(len(ids) for ids in found.values()) >= 2, "pattern should match several articles"
    assert set(found) == {value}, found


@pytest.mark.parametrize("terms", [
    ("installment", "instalment"), ("customs", "duties", "import tax"), ("time slot",),
    ("locker", "pickup point"), ("gift wrap",), ("student",), ("refurbished",),
    ("holiday return", "extended return"), ("physical store", "retail store"),
    ("export", "copy of your data"),
])
def test_unsupported_topics_stay_absent_from_the_kb(terms):
    """The unsupported labels rely on the KB saying nothing about these topics."""
    text = " ".join(f"{a['title']} {a['text']}".lower() for a in ARTICLES)
    assert not [term for term in terms if re.search(rf"\b{re.escape(term)}", text)]


def test_v1_snapshot_matches_every_saved_fixture():
    """evals/fixtures/kb_v1.json is the exact KB the saved experiments used."""
    snapshot_hash = cal.kb_hash(load_articles(cal.HISTORICAL_KB_PATH))
    fixtures = EVALS_DIR / "fixtures"
    for name in ("semantic_embeddings", "semantic_holdout_embeddings", "rerank_results", "rerank_holdout_results"):
        data = json.loads((fixtures / f"{name}.json").read_text(encoding="utf-8"))
        assert data["kb_sha256"] == snapshot_hash, name


def test_historical_kb_switches_and_restores_the_kb_path():
    before = kb.KB_PATH
    with cal.historical_kb():
        assert [a["id"] for a in load_articles()] == ["KB-001", "KB-002", "KB-003"]
    assert kb.KB_PATH == before
    with pytest.raises(RuntimeError), cal.historical_kb():
        raise RuntimeError
    assert kb.KB_PATH == before


# --- The dataset -------------------------------------------------------------------


def test_real_dataset_is_valid():
    ev.validate_cases(CASES, ARTICLES)


def test_dataset_size_and_label_distribution():
    labels = {label: sum(c["label"] == label for c in CASES) for label in ev.LABELS}
    assert 50 <= len(CASES) <= 80
    assert labels["answerable"] >= 40 and labels["unsupported"] >= 10 and labels["ambiguous"] >= 8


def test_every_category_and_article_is_exercised():
    answerable = [c for c in CASES if c["label"] == "answerable"]
    for category in ev.CATEGORIES:
        assert sum(c["category"] == category for c in answerable) >= 4, category
    assert {i for c in answerable for i in c["expected_ids"]} == set(BY_ID)


def test_edge_cases_and_unsupported_scopes_are_present():
    tags = [t for c in CASES for t in c["tags"]]
    for tag in ("typo", "multi_intent", "negation", "boundary", "informal", "adversarial"):
        assert tag in tags, tag
    for tag in ev.SCOPE_TAGS:
        assert tags.count(tag) >= 3, tag


def test_no_overlap_with_the_v1_question_sets():
    v1 = set()
    for name in ("retrieval_queries.json", "rerank_holdout_queries.json"):
        v1 |= {ev.normalise_query(c["query"]) for c in json.loads((EVALS_DIR / name).read_text(encoding="utf-8"))}
    assert not v1 & {ev.normalise_query(c["query"]) for c in CASES}


def _first(label):
    return next(i for i, c in enumerate(CASES) if c["label"] == label)


@pytest.mark.parametrize("label, change, message", [
    ("answerable", lambda c: c.update(expected_ids=["KB-999"]), "unknown article"),
    ("answerable", lambda c: c.update(expected_ids=[]), "must not be empty"),
    ("answerable", lambda c: c.update(also_relevant_ids=list(c["expected_ids"])), "both expected"),
    ("answerable", lambda c: c.update(answer_facts=[]), "answer_facts"),
    ("answerable", lambda c: c.update(answer_facts=[[" "]]), "answer_facts"),
    ("answerable", lambda c: c.update(tags=["multi_intent"]), "multi_intent"),
    ("answerable", lambda c: c.update(tags=["near_miss"]), "scope tags"),
    ("answerable", lambda c: c.update(category="none"), "needs a category"),
    ("answerable", lambda c: c.update(label="maybe"), "label must be"),
    ("answerable", lambda c: c.update(id="Q1"), "V2-001"),
    ("answerable", lambda c: c.update(query="  "), "non-empty"),
    ("answerable", lambda c: c.update(tags=["made_up"]), "tags"),
    ("answerable", lambda c: c.pop("answer_facts"), "missing"),
    ("unsupported", lambda c: c.update(note=""), "note"),
    ("unsupported", lambda c: c.update(tags=[]), "exactly one"),
    ("unsupported", lambda c: c.update(expected_ids=["KB-001"]), "unexpected"),
    ("unsupported", lambda c: c.update(related_ids=["KB-999"]), "unknown article"),
    ("ambiguous", lambda c: c.update(tags=["near_miss"]), "scope tags"),
])
def test_validation_rejects_bad_labels(label, change, message):
    cases = copy.deepcopy(CASES)
    change(cases[_first(label)])
    with pytest.raises(ev.DatasetError, match=message):
        ev.validate_cases(cases, ARTICLES)


def test_validation_rejects_duplicates():
    cases = copy.deepcopy(CASES)
    cases[1]["id"] = cases[0]["id"]
    with pytest.raises(ev.DatasetError, match="duplicate id"):
        ev.validate_cases(cases, ARTICLES)

    cases = copy.deepcopy(CASES)
    cases[1]["query"] = cases[0]["query"].upper() + "!"
    with pytest.raises(ev.DatasetError, match="duplicate query"):
        ev.validate_cases(cases, ARTICLES)


@pytest.mark.parametrize("change, message", [
    (lambda a: a[0].update(id="X-1"), "KB-001"),
    (lambda a: a[0].pop("category"), "category"),
    (lambda a: a[0].update(category="misc"), "category"),
    (lambda a: a[1].update(url=a[0]["url"]), "duplicate url"),
    (lambda a: a[1].update(title=a[0]["title"].upper()), "duplicate title"),
])
def test_kb_validation_rejects_bad_articles(change, message):
    articles = copy.deepcopy(ARTICLES)
    change(articles)
    with pytest.raises(ev.DatasetError, match=message):
        ev.validate_kb(articles)


# --- Retrieval metrics -----------------------------------------------------------


def _case(case_id, label="answerable", expected=(), also=(), tags=(), category="account"):
    case = {"id": case_id, "query": case_id, "label": label, "category": category, "tags": list(tags)}
    if label == "answerable":
        case.update(expected_ids=list(expected), also_relevant_ids=list(also), answer_facts=[["x"]])
    else:
        case.update(related_ids=[], note="n")
    return case


def test_retrieval_metrics_by_hand():
    cases = [
        _case("A", expected=["KB-1"]),                                  # rank 1
        _case("B", expected=["KB-1"], also=["KB-2"]),                   # also-relevant at 1, expected at 3
        _case("C", expected=["KB-1", "KB-2"], tags=["multi_intent"]),   # half covered
        _case("D", expected=["KB-1"]),                                  # nothing returned
        _case("U1", "unsupported", tags=["near_miss"]),
        _case("U2", "unsupported", tags=["out_of_domain"]),
        _case("M", "ambiguous"),
    ]
    returned = {"A": ["KB-1", "KB-9"], "B": ["KB-2", "KB-9", "KB-1"], "C": ["KB-9", "KB-2"],
                "D": [], "U1": ["KB-5"], "U2": [], "M": ["KB-1"]}

    m = ev.retrieval_metrics(cases, returned, k=3)
    a = m["answerable"]

    assert (a["questions"], a["top1"], a["hit"], a["covered"], a["no_result"]) == (4, 2, 3, 2, 1)
    assert a["recall"] == pytest.approx((1 + 1 + 0.5 + 0) / 4)
    assert a["precision"] == pytest.approx((1 / 3 + 2 / 3 + 1 / 3 + 0) / 4)
    assert a["returned_precision"] == pytest.approx((1 / 2 + 2 / 3 + 1 / 2) / 3)  # D returned nothing
    assert a["mrr"] == pytest.approx((1 + 1 + 1 / 2 + 0) / 4)
    assert m["by_tag"]["multi_intent"]["covered"] == 0
    assert [miss[0] for miss in m["misses"]] == ["D"]
    assert m["unsupported"]["abstained"] == 1
    assert m["unsupported"]["by_scope"]["near_miss"] == {"questions": 1, "abstained": 0}
    assert m["ambiguous"] == {"questions": 1, "returned_any": 1}


def test_retrieval_metrics_cut_at_k():
    cases = [_case("A", expected=["KB-1"])]
    assert ev.retrieval_metrics(cases, {"A": ["KB-9", "KB-1"]}, k=1)["answerable"]["hit"] == 0
    assert ev.retrieval_metrics(cases, {"A": ["KB-9", "KB-1"]}, k=2)["answerable"]["hit"] == 1


def test_offline_keyword_run_makes_no_cohere_calls(block_cohere, capsys):
    assert ev.main([]) == 0
    output = capsys.readouterr().out
    assert "RETRIEVAL (keyword, K=3)" in output
    assert f"Questions: {len(CASES)}" in output


# --- Answer metrics --------------------------------------------------------------


@pytest.mark.parametrize("answer, alternatives, expected", [
    ("Refunds take 5 business days.", ["5 business days"], True),
    ("It takes 5–8 business days", ["5-8"], True),            # en dash
    ("Express costs $12.99.", ["$12.99"], True),
    ("Please see the NOTE below", ["not"], False),                   # whole words only
    ("You cannot return it", ["cannot", "can't"], True),
    ("You can’t return it", ["can't"], True),                  # curly apostrophe
    ("Within 15 minutes", ["5 minutes"], False),
])
def test_fact_present(answer, alternatives, expected):
    assert ev.fact_present(answer, alternatives) is expected


def _source(article_id):
    article = BY_ID[article_id]
    return {"id": article_id, "title": article["title"], "url": article["url"]}


def _answered(text, *ids):
    return {"answer": text, "sources": [_source(i) for i in ids], "escalate": False, "reason": "answered"}


HANDOFF = {"answer": "handoff", "sources": [], "escalate": True, "reason": "insufficient_evidence"}


def test_answer_metrics_by_hand():
    cases = [
        {**_case("A", expected=["KB-001"]), "answer_facts": [["60 minutes"]]},
        {**_case("B", expected=["KB-002"]), "answer_facts": [["30 days"], ["order number"]]},
        {**_case("C", expected=["KB-003"]), "answer_facts": [["24 hours"]]},
        {**_case("D", expected=["KB-004"]), "answer_facts": [["24 hours"]]},
        _case("U1", "unsupported", tags=["near_miss"]),
        _case("U2", "unsupported", tags=["out_of_domain"]),
        _case("M", "ambiguous"),
    ]
    bad_title = _answered("24 hours", "KB-004")
    bad_title["sources"][0]["title"] = "Something else"
    results = {
        "A": _answered("The link lasts 60 minutes.", "KB-001"),      # correct
        "B": _answered("Within 30 days.", "KB-002"),                 # one fact missing
        "C": _answered("Updates appear within 24 hours.", "KB-009"),  # wrong article cited
        "D": bad_title,                                             # invalid citation
        "U1": _answered("Yes.", "KB-018"),                           # false answer
        "U2": HANDOFF,
        "M": _answered("Maybe.", "KB-001"),                          # needs review
    }

    m = ev.answer_metrics(cases, results, ARTICLES)
    a = m["answerable"]

    assert (a["questions"], a["answered"], a["handed_off"]) == (4, 4, 0)
    assert (a["valid_citations"], a["supporting_cited"], a["all_facts"], a["correct_proxy"]) == (3, 2, 3, 1)
    assert m["mean_fact_recall"] == pytest.approx((1 + 0.5 + 1 + 1) / 4)
    assert m["unsupported"]["abstained"] == 1 and m["false_answers"] == ["U1"]
    assert m["review"] == ["M"]
    assert m["reasons"]["unsupported"]["insufficient_evidence"] == 1


def test_answer_metrics_count_handoffs_of_answerable_questions():
    cases = [_case("A", expected=["KB-001"])]
    m = ev.answer_metrics(cases, {"A": HANDOFF}, ARTICLES)
    assert (m["answerable"]["handed_off"], m["answerable"]["answered"], m["mean_fact_recall"]) == (1, 0, 0.0)


def _answers_file(tmp_path, **overrides):
    data = {
        "version": ev.ANSWERS_VERSION, "recorded_on": "2026-01-01", "chat_model": "m", "rerank": {},
        "dataset_sha256": ev.sha256_json(CASES), "kb_sha256": ev.sha256_json(ARTICLES),
        "results": {c["id"]: HANDOFF for c in CASES},
    }
    data.update(overrides)
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_answers_accepts_a_matching_file(tmp_path):
    assert len(ev.load_answers(_answers_file(tmp_path), CASES, ARTICLES)["results"]) == len(CASES)


@pytest.mark.parametrize("overrides, message", [
    ({"version": 99}, "version"),
    ({"dataset_sha256": "x"}, "question set changed"),
    ({"kb_sha256": "x"}, "knowledge base changed"),
    ({"results": {}}, "missing"),
])
def test_load_answers_rejects_stale_or_partial_files(tmp_path, overrides, message):
    with pytest.raises(ev.DatasetError, match=message):
        ev.load_answers(_answers_file(tmp_path, **overrides), CASES, ARTICLES)


def test_main_scores_a_recorded_file_offline(tmp_path, block_cohere, capsys):
    assert ev.main(["--answers", str(_answers_file(tmp_path))]) == 0
    output = capsys.readouterr().out
    assert "ANSWERS" in output and "RETRIEVAL" not in output
    assert f"Handed off (false)       {sum(c['label'] == 'answerable' for c in CASES)}" in output


def test_record_answers_refuses_to_overwrite(tmp_path, monkeypatch):
    from support_agent import agent

    monkeypatch.setattr(agent, "answer_question", lambda q: pytest.fail("must not run"))
    path = tmp_path / "answers.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ev.DatasetError, match="already exists"):
        ev.record_answers(CASES, ARTICLES, path)


def test_record_answers_with_a_fake_agent_round_trips(tmp_path, monkeypatch):
    from support_agent import agent, rerank

    asked = []

    def fake_answer(question):
        asked.append(question)
        return SupportAnswer(answer="ok", sources=[_source("KB-001")], escalate=False, reason="answered")

    monkeypatch.setattr(agent, "answer_question", fake_answer)
    monkeypatch.setattr(agent, "_default_client", lambda: object())
    monkeypatch.setattr(rerank, "config_from_env", lambda: rerank.RerankConfig())
    path = tmp_path / "out" / "answers.json"

    ev.record_answers(CASES, ARTICLES, path)

    assert asked == [c["query"] for c in CASES]
    data = ev.load_answers(path, CASES, ARTICLES)
    assert data["rerank"]["enabled"] is False
    assert data["results"][CASES[0]["id"]]["sources"] == [_source("KB-001")]
