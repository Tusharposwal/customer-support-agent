"""Offline checks for the tool-selection evaluation (evals/evaluate_tool_selection.py
and evals/tool_selection_eval.json). No Cohere calls: the pipeline runs with
a scripted fake client and sockets blocked."""

import copy
import json
import socket
import sys
from pathlib import Path

import pytest

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS_DIR))

import evaluate_tool_selection as ev  # noqa: E402

CASES = ev.load_cases()
BY_ID = {c["id"]: c for c in CASES}


def case(**overrides):
    base = {"id": "TS-900", "category": "order_status", "scenario": "faithful",
            "question": "Where is ORD-1003? priya.nair@example.com", "expected_tools": ["get_order_status"],
            "script": {"answer": "x"}, "expected_outcome": {"offered": True, "reason": "answered", "statuses": []}}
    return {**base, **overrides}


# --- The dataset ---------------------------------------------------------------------


def test_dataset_covers_every_required_category():
    counts = {cat: sum(c["category"] == cat for c in CASES) for cat in ev.CATEGORIES}
    assert all(n >= 4 for n in counts.values()), counts
    assert {c["scenario"] for c in CASES} == {"faithful", "misbehaving"}
    assert {name for c in CASES for name in c["expected_tools"]} == set(ev.TOOL_NAMES)
    assert sum(not c["expected_tools"] for c in CASES) >= 10


def test_dataset_is_separate_from_the_kb_and_uses_fictional_data_only():
    assert ev.DATASET_PATH.parent.name == "evals"
    text = ev.DATASET_PATH.read_text(encoding="utf-8")
    assert "@" in text and all(e.lower().endswith("@example.com") for e in ev._EMAIL.findall(text))


@pytest.mark.parametrize("change, message", [
    (lambda c: c.update(expected_tools=["issue_refund"]), "registry tools"),
    (lambda c: c.update(acceptable=[["delete_order"]]), "registry tools"),
    (lambda c: c.update(category="other"), "unknown category"),
    (lambda c: c.update(question="Where is ORD-9999? priya.nair@example.com"), "not a demo order"),
    (lambda c: c.update(question="Where is ORD-1003? priya@gmail.com"), "fictional"),
    (lambda c: c.pop("script"), "missing"),
    (lambda c: c.update(extra=1), "unexpected"),
    (lambda c: c.update(script={"calls": [], "answer": "x"}), "exactly one"),
    (lambda c: c.update(script={"calls": [{"name": "get_order_status"}]}), "name and arguments"),
    (lambda c: c.update(expected_outcome={"offered": False, "reason": "answered", "statuses": []}), "contradicts"),
    (lambda c: c.update(expected_outcome={"offered": True, "reason": "great", "statuses": []}), "expected_outcome"),
    (lambda c: c.update(history=[{"role": "system", "content": "x"}]), "history"),
    (lambda c: c.update(id="X-1"), "id"),
])
def test_validation_rejects_bad_cases(change, message):
    bad = case()
    change(bad)
    with pytest.raises(ev.DatasetError, match=message):
        ev.validate_cases([bad])


def test_validation_rejects_duplicate_ids_and_empty_sets():
    with pytest.raises(ev.DatasetError, match="duplicate"):
        ev.validate_cases([case(), case()])
    with pytest.raises(ev.DatasetError):
        ev.validate_cases([])


# --- Selection metrics ---------------------------------------------------------------------


@pytest.mark.parametrize("expected, acceptable, selected, bucket", [
    (["get_order_status"], [], ["get_order_status"], "correct_tool"),
    ([], [], [], "correct_no_tool"),
    (["get_order_status"], [], ["check_refund_eligibility"], "incorrect_tool"),
    (["get_order_status"], [], [], "missing_tool"),
    ([], [], ["get_order_status"], "unnecessary_tool"),
    # Order and repeats don't matter.
    (["get_order_status", "check_refund_eligibility"], [], ["check_refund_eligibility", "get_order_status",
                                                           "get_order_status"], "correct_tool"),
    # Partial or extra selections are incorrect.
    (["get_order_status", "check_refund_eligibility"], [], ["get_order_status"], "incorrect_tool"),
    (["get_order_status"], [], ["get_order_status", "request_order_cancellation"], "incorrect_tool"),
    # Acceptable alternatives count as correct, including "no tool".
    ([], [["get_order_status"]], ["get_order_status"], "correct_tool"),
    (["get_order_status"], [[]], [], "correct_no_tool"),
    ([], [["get_order_status"]], ["check_refund_eligibility"], "unnecessary_tool"),
    # A tool that doesn't exist is never correct.
    (["check_refund_eligibility"], [], ["issue_refund"], "incorrect_tool"),
])
def test_classify(expected, acceptable, selected, bucket):
    assert ev.classify(case(expected_tools=expected, acceptable=acceptable), selected) == bucket


def test_score_selections_counts_and_confusion_matrix():
    cases = [case(id="TS-901"), case(id="TS-902"), case(id="TS-903", expected_tools=[]),
             case(id="TS-904", expected_tools=[])]
    metrics = ev.score_selections(cases, {"TS-901": ["get_order_status"], "TS-902": [],
                                          "TS-903": [], "TS-904": ["get_order_status"]})
    assert metrics["buckets"] == {"correct_tool": 1, "correct_no_tool": 1, "incorrect_tool": 0,
                                  "missing_tool": 1, "unnecessary_tool": 1}
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion"] == [
        {"expected": "get_order_status", "selected": "get_order_status", "count": 1},
        {"expected": "get_order_status", "selected": "none", "count": 1},
        {"expected": "none", "selected": "get_order_status", "count": 1},
        {"expected": "none", "selected": "none", "count": 1},
    ]
    assert metrics["by_category"]["order_status"]["missing_tool"] == 1


def test_score_selections_rejects_missing_or_malformed_selections():
    with pytest.raises(ev.DatasetError, match="no selection"):
        ev.score_selections([case()], {})
    with pytest.raises(ev.DatasetError, match="list of tool names"):
        ev.score_selections([case()], {"TS-900": "get_order_status"})


def test_perfect_selections_score_100_percent():
    perfect = {c["id"]: c["expected_tools"] for c in CASES}
    metrics = ev.score_selections(CASES, perfect)
    assert metrics["accuracy"] == 1.0
    assert all(row["expected"] == row["selected"] for row in metrics["confusion"])


def test_selections_file(tmp_path):
    path = tmp_path / "selections.json"
    path.write_text(json.dumps({"version": 1, "source": "test", "selections": {"TS-001": []}}), encoding="utf-8")
    selections, source = ev.load_selections(path, CASES)
    assert source == "test" and selections == {"TS-001": []}
    path.write_text(json.dumps({"version": 1, "selections": {"TS-999": []}}), encoding="utf-8")
    with pytest.raises(ev.DatasetError, match="unknown case"):
        ev.load_selections(path, CASES)
    path.write_text(json.dumps({"selections": {}}), encoding="utf-8")
    with pytest.raises(ev.DatasetError, match="version 1"):
        ev.load_selections(path, CASES)


# --- The offer rule and the baseline ---------------------------------------------------------


def test_offer_rule_never_hides_a_needed_tool():
    metrics = ev.offer_metrics(CASES)
    assert metrics["not_offered_but_needed"] == 0
    assert sum(metrics.values()) == len(CASES)


@pytest.mark.parametrize("question, expected", [
    ("Where is ORD-1003? priya.nair@example.com", ["get_order_status"]),
    ("Where is ORD-1003?", []),  # No email: no tool.
    ("Where is my order? priya.nair@example.com", []),  # No order number: no tool.
    ("Cancel ORD-1001, Maya.Fernandes@Example.com", ["request_order_cancellation"]),  # Any case of email.
    ("Can I return ORD-1005? maya.fernandes@example.com", ["check_refund_eligibility"]),
])
def test_baseline_rules(question, expected):
    assert ev.baseline_select(case(question=question)) == expected


def test_baseline_is_deterministic():
    first = [ev.baseline_select(c) for c in CASES]
    assert first == [ev.baseline_select(c) for c in copy.deepcopy(CASES)]


# --- Orchestration with scripted decisions -----------------------------------------------------


@pytest.fixture(scope="module")
def orchestration():
    return ev.run_orchestration(CASES)


def test_every_case_reproduces_its_expected_outcome(orchestration):
    failed = [r for r in orchestration["per_case"] if not r["passed"]]
    assert not failed, failed
    assert orchestration["safety_violations"] == 0
    assert orchestration["second_requests_without_tool_choice_none"] == 0


def test_invalid_calls_and_missing_details_are_handled_safely(orchestration):
    for key in ("invalid_calls", "missing_details"):
        group = orchestration[key]
        assert group["cases"] >= 5 and group["handled_safely"] == group["cases"]


def test_no_tool_runs_when_tools_are_not_offered(orchestration):
    for row in orchestration["per_case"]:
        if not row["observed"]["offered"]:
            assert row["executed_tools"] == [] and row["chat_requests"] == 1


def test_misbehaving_decisions_never_reach_order_data_unsafely(orchestration):
    rows = {r["id"]: r for r in orchestration["per_case"]}
    assert rows["TS-029"]["observed"]["reason"] == "needs_order_details"  # Invented email refused.
    assert rows["TS-041"]["observed"]["statuses"] == ["details_not_provided"]  # Swapped order refused.
    assert rows["TS-043"]["observed"]["reason"] == "invalid_tool_call"  # Unknown tool.
    assert rows["TS-047"]["executed_tools"] == []  # Duplicate call IDs: nothing ran.
    assert rows["TS-046"]["observed"]["statuses"][-1] == "limit_exceeded"


def test_safety_check_flags_order_data_without_details():
    no_email = case(question="Where is ORD-1003?")
    assert ev.safety_violations(no_email, [("get_order_status", "ok")])
    assert not ev.safety_violations(no_email, [("get_order_status", "details_not_provided")])
    assert not ev.safety_violations(case(), [("get_order_status", "ok")])


def test_scripted_client_cannot_call_tools_it_was_not_offered():
    client = ev.ScriptedClient({"calls": [{"name": "get_order_status", "arguments": {}}]})
    response = client.chat(messages=[], documents=[{"id": "KB-003", "data": {}}])
    assert response.finish_reason == "COMPLETE"
    assert response.message.citations[0].sources[0].id == "KB-003"


def test_the_evaluation_never_opens_a_connection(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket, "getaddrinfo", fail)
    result = ev.run_orchestration(CASES[:3])
    assert result["passed"] == 3
    # The guard is removed afterwards and the test's own patch is back.
    assert socket.create_connection is fail


def test_a_wrong_expectation_is_reported_as_a_failure():
    wrong = copy.deepcopy(BY_ID["TS-001"])
    wrong["expected_outcome"]["reason"] = "cancellation_available"
    result = ev.run_orchestration([wrong])
    assert result["passed"] == 0 and result["per_case"][0]["observed"]["reason"] == "answered"


# --- The command line -------------------------------------------------------------------------


def test_main_reports_and_writes_a_new_file_only(tmp_path, capsys):
    output = tmp_path / "results.json"
    assert ev.main(["--output", str(output)]) == 0
    printed = capsys.readouterr().out
    assert "not the model" in printed and "not model accuracy" in printed
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["orchestration"]["passed"] == len(CASES)
    assert data["dataset_sha256"] == ev.sha256_json(CASES)
    with pytest.raises(FileExistsError):
        ev.main(["--output", str(output)])


def test_main_scores_recorded_selections(tmp_path, capsys):
    path = tmp_path / "selections.json"
    path.write_text(json.dumps({"version": 1, "source": "hand-made", "selections": {
        c["id"]: [] for c in CASES}}), encoding="utf-8")
    ev.main(["--selections", str(path)])
    assert "Recorded selections (hand-made)" in capsys.readouterr().out


def test_results_are_reproducible():
    first, second = ev.evaluate(CASES), ev.evaluate(CASES)
    assert first == second
