"""Offline checks for the multi-turn dataset and evals/evaluate_conversations.py.
No Cohere calls: the live recording path runs with a fake client only."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import support_agent
from support_agent import query_rewrite
from support_agent.kb import load_articles

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS_DIR))

import evaluate_conversations as ev  # noqa: E402

ARTICLES = load_articles()
CASES = ev.load_cases()


@pytest.fixture
def block_cohere(monkeypatch):
    for name in ("support_agent.semantic_kb", "support_agent.hybrid_kb", "cohere"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delattr(support_agent, "semantic_kb", raising=False)
    monkeypatch.delattr(support_agent, "hybrid_kb", raising=False)


# --- The dataset ---------------------------------------------------------------------


def test_real_dataset_is_valid():
    ev.validate_cases(CASES, ARTICLES)


def test_label_distribution():
    labels = {label: sum(c["label"] == label for c in CASES) for label in ev.LABELS}
    assert len(CASES) >= 30
    assert labels["follow_up"] >= 15 and labels["needs_clarification"] >= 4
    assert labels["standalone"] >= 3 and labels["topic_switch"] >= 3


def test_follow_ups_cover_most_categories_and_numbers():
    follow_ups = [c for c in CASES if c["label"] == "follow_up"]
    assert len({c["category"] for c in follow_ups}) >= 8
    assert sum("number" in c["tags"] for c in follow_ups) >= 4


@pytest.mark.parametrize("case", [c for c in CASES if c["label"] != "needs_clarification"], ids=lambda c: c["id"])
def test_reference_questions_pass_the_production_rewrite_checks(case):
    """The guard in query_rewrite must accept every hand-written reference:
    a check that rejected good rewrites would silently disable rewriting."""
    output = json.dumps({"standalone_query": case["reference_query"], "needs_clarification": False})
    rewrite = query_rewrite.check_output(output, case["message"], case["history"])
    assert rewrite.outcome in (query_rewrite.REWRITTEN, query_rewrite.UNCHANGED)


@pytest.mark.parametrize("case", [c for c in CASES if "number" in c["tags"]], ids=lambda c: c["id"])
def test_the_rewrite_check_rejects_an_invented_number(case):
    output = json.dumps({"standalone_query": case["reference_query"] + " within 999 days", "needs_clarification": False})
    with pytest.raises(ValueError, match="number"):
        query_rewrite.check_output(output, case["message"], case["history"])


def _first(label):
    return next(i for i, c in enumerate(CASES) if c["label"] == label)


@pytest.mark.parametrize("label, change, message", [
    ("follow_up", lambda c: c.update(id="X-1"), "C-001"),
    ("follow_up", lambda c: c.update(label="other"), "label"),
    ("follow_up", lambda c: c.update(message=" "), "message"),
    ("follow_up", lambda c: c.update(tags=["nope"]), "tags"),
    ("follow_up", lambda c: c.update(history=[]), "pairs"),
    ("follow_up", lambda c: c["history"].reverse(), "alternate"),
    ("follow_up", lambda c: c.update(expected_ids=["KB-999"]), "unknown article"),
    ("follow_up", lambda c: c.update(expected_ids=[]), "expected_ids"),
    ("follow_up", lambda c: c.update(required_terms=[]), "required_terms"),
    ("follow_up", lambda c: c.update(required_terms=[["zebra"]]), "lacks required"),
    ("follow_up", lambda c: c.update(forbidden_terms=["return"]), "forbidden"),
    ("follow_up", lambda c: c.update(reference_query=c["reference_query"] + " in 77 days"), "never wrote"),
    ("follow_up", lambda c: c.update(reference_query="x" * 301), "at most"),
    ("standalone", lambda c: c.update(reference_query=c["message"] + " now"), "must equal"),
    ("needs_clarification", lambda c: c.update(note=""), "note"),
    ("needs_clarification", lambda c: c.update(reference_query="q"), "unexpected"),
])
def test_validation_rejects_bad_cases(label, change, message):
    cases = copy.deepcopy(CASES)
    change(cases[_first(label)])
    with pytest.raises(ev.DatasetError, match=message):
        ev.validate_cases(cases, ARTICLES)


def test_validation_rejects_duplicate_ids():
    cases = copy.deepcopy(CASES)
    cases[1]["id"] = cases[0]["id"]
    with pytest.raises(ev.DatasetError, match="duplicate"):
        ev.validate_cases(cases, ARTICLES)


# --- Metrics by hand -----------------------------------------------------------------


def _case(case_id, label="follow_up", message="m", reference="ref", expected=("KB-001",), required=(("ref",),),
          forbidden=(), customer="c"):
    case = {"id": case_id, "label": label, "category": "account", "tags": [],
            "history": [{"role": "customer", "content": customer}, {"role": "assistant", "content": "a"}],
            "message": message}
    if label != "needs_clarification":
        case.update(reference_query=reference, expected_ids=list(expected),
                    required_terms=[list(r) for r in required], forbidden_terms=list(forbidden))
    else:
        case["note"] = "n"
    return case


def fake_retriever(results):
    """query -> article ids, as retrieve(query, top_k) would return."""
    return lambda query, top_k: [{"id": i} for i in results.get(query, [])][:top_k]


def test_retrieval_comparison_by_hand():
    cases = [
        _case("C-001", message="it?", reference="refund 35", customer="refund"),
        _case("C-002", label="standalone", message="track", reference="track"),
        _case("C-003", label="needs_clarification"),
    ]
    retriever = fake_retriever({"refund 35": ["KB-001"], "refund it?": ["KB-001"], "track": ["KB-001"],
                                "c track": ["KB-009"]})

    m = ev.retrieval_comparison(cases, retriever, k=3)

    assert (m["message"]["hits"], m["naive"]["hits"], m["reference"]["hits"]) == (1, 1, 2)
    assert m["message"]["misses"] == ["C-001"] and m["naive"]["misses"] == ["C-002"]
    assert m["reference"]["questions"] == 2  # clarification cases are not retrieval questions


def test_rewrite_metrics_by_hand():
    cases = [
        _case("C-001", required=(("refund",), ("35",)), forbidden=("password",), customer="return it 35 days"),
        _case("C-002", required=(("track",),), forbidden=("refund",)),
        _case("C-003", required=(("invoice",),)),
        _case("C-004", label="needs_clarification"),
        _case("C-005", label="needs_clarification"),
    ]
    results = {
        "C-001": {"query": "refund after 35 days", "outcome": "rewritten"},  # good
        "C-002": {"query": "track my refund in 12 days", "outcome": "rewritten"},  # leak + invented number
        "C-003": {"query": "c", "outcome": "needs_clarification"},  # false alarm
        "C-004": {"query": "x", "outcome": "needs_clarification"},  # correct
        "C-005": {"query": "x", "outcome": "unchanged"},  # missed
    }
    retriever = fake_retriever({"refund after 35 days": ["KB-001"]})

    m = ev.rewrite_metrics(cases, results, retriever, k=3)

    assert m["outcomes"] == {"rewritten": 2, "needs_clarification": 2, "unchanged": 1}
    assert m["clarification"] == {"expected": 2, "asked": 2, "correct": 1, "false_alarm": ["C-003"]}
    # C-002 has its required term; the stale topic and the number are counted separately.
    assert m["terms"] == {"questions": 2, "all_required": 2, "forbidden_leaks": ["C-002"], "invented_numbers": ["C-002"]}
    assert m["retrieval"] == {"questions": 2, "hits": 1, "misses": ["C-002"],
                              "baselines": {"message": 0, "naive": 0, "reference": 0}}
    assert m["requests"] is None  # these results have no request/latency fields (version 1 shape)


# --- Recorded rewrites -----------------------------------------------------------------


def _rewrites_file(tmp_path, **overrides):
    data = {"version": ev.REWRITES_VERSION, "recorded_on": "2026-01-01", "rewrite_model": "m",
            "dataset_sha256": ev.sha256_json(CASES),
            "results": {c["id"]: {"query": c["message"], "outcome": "failed"} for c in CASES}}
    data.update(overrides)
    path = tmp_path / "rewrites.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("overrides, message", [
    ({"version": 99}, "version"),
    ({"dataset_sha256": "x"}, "changed"),
    ({"results": {}}, "missing"),
    ({"results": {c["id"]: {"query": 1, "outcome": "x"} for c in CASES}}, "query and an outcome"),
])
def test_load_rewrites_rejects_stale_or_broken_files(tmp_path, overrides, message):
    with pytest.raises(ev.DatasetError, match=message):
        ev.load_rewrites(_rewrites_file(tmp_path, **overrides), CASES)


def test_main_scores_a_recorded_file_offline(tmp_path, block_cohere, capsys):
    assert ev.main(["--rewrites", str(_rewrites_file(tmp_path))]) == 0
    output = capsys.readouterr().out
    assert "RECORDED REWRITES" in output and f"failed={len(CASES)}" in output


def test_main_offline_comparison_makes_no_cohere_calls(block_cohere, capsys):
    assert ev.main([]) == 0
    output = capsys.readouterr().out
    assert f"Conversations: {len(CASES)}" in output
    assert "upper bound, not a measurement of the rewriter" in output


def test_record_rewrites_refuses_to_overwrite(tmp_path):
    path = tmp_path / "rewrites.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ev.DatasetError, match="already exists"):
        ev.record_rewrites(CASES, path)


# --- Controlled recording (all offline: fake clients, no key, no network) ----------------


class FakeChatClient:
    """Stands in for cohere.ClientV2. Each chat() call pops the next outcome:
    an exception to raise, or the text the model "returns"."""

    def __init__(self, *outcomes, default="clarify"):
        self.outcomes = list(outcomes)
        self.default = default
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else self.default
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "clarify":
            outcome = json.dumps({"standalone_query": "", "needs_clarification": True})
        return SimpleNamespace(finish_reason="COMPLETE",
                               message=SimpleNamespace(content=[SimpleNamespace(type="text", text=outcome)]))


class StatusError(Exception):
    """Looks like a Cohere API error with an HTTP status (retryable for 429/5xx)."""

    def __init__(self, status_code):
        super().__init__(f"provider text with SECRET-DETAIL (status {status_code})")
        self.status_code = status_code


@pytest.fixture
def never_read_dotenv(monkeypatch):
    """Any load_dotenv / dotenv lookup fails the test; so does the app's client."""
    import dotenv

    from support_agent import agent

    def forbidden(*args, **kwargs):
        pytest.fail(".env lookup attempted")

    monkeypatch.setattr(agent, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "find_dotenv", forbidden)
    monkeypatch.setattr(agent, "_default_client", forbidden)


@pytest.fixture
def fake_sdk(monkeypatch):
    """A fake `cohere` module whose ClientV2 records the key it was given."""
    created = []

    class ClientV2(FakeChatClient):
        def __init__(self, api_key):
            super().__init__()
            self.api_key = api_key
            created.append(self)

    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=ClientV2))
    return created


@pytest.fixture
def ticking_clock(monkeypatch):
    """perf_counter stand-in: every reading is 0.25 s after the previous one,
    so each timed request measures exactly 250 ms."""
    ticks = iter(range(10_000))
    monkeypatch.setattr(ev, "clock", lambda: next(ticks) * 0.25)


def test_recording_client_uses_only_the_environment(never_read_dotenv, fake_sdk, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "  env-key-123\n")
    client = ev.recording_client()
    assert client.api_key == "env-key-123" and fake_sdk == [client]


def test_record_rewrites_never_reads_dotenv(never_read_dotenv, fake_sdk, monkeypatch, tmp_path):
    monkeypatch.setenv("COHERE_API_KEY", "env-key-123")
    ev.record_rewrites(CASES, tmp_path / "rewrites.json")
    assert len(fake_sdk) == 1 and len(fake_sdk[0].calls) == len(CASES)


@pytest.mark.parametrize("value", [None, "", "   ", "\n"])
def test_missing_key_fails_safely_before_any_request(never_read_dotenv, fake_sdk, monkeypatch, tmp_path, value):
    if value is None:
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("COHERE_API_KEY", value)
    path = tmp_path / "out" / "rewrites.json"

    with pytest.raises(ev.RecordingError) as raised:
        ev.record_rewrites(CASES, path)

    assert "COHERE_API_KEY is not set" in str(raised.value) and "does not read .env" in str(raised.value)
    assert fake_sdk == [] and not path.parent.exists()


def test_main_reports_a_missing_key_without_a_traceback(never_read_dotenv, fake_sdk, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    assert ev.main(["--record-rewrites", str(tmp_path / "r.json")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: COHERE_API_KEY is not set") and "Traceback" not in err


def test_every_request_disables_sdk_retries(tmp_path):
    client = FakeChatClient()
    ev.record_rewrites(CASES, tmp_path / "r.json", client=client)
    assert {call["request_options"]["max_retries"] for call in client.calls} == {0}


@pytest.mark.parametrize("error", [StatusError(429), StatusError(503), ConnectionRefusedError()])
def test_a_retryable_failure_is_not_retried(tmp_path, error, no_real_retry_waits, caplog):
    """Errors the production policy would retry once are recorded instead:
    one request for that case, no wait, and the next case still runs."""
    client = FakeChatClient(error)

    data = ev.record_rewrites(CASES, tmp_path / "r.json", client=client)

    first = data["results"][CASES[0]["id"]]
    assert len(client.calls) == len(CASES)  # exactly one request per case, failures included
    assert no_real_retry_waits == []
    assert first["outcome"] == "failed" and first["requests"] == 1
    assert first["error"] in ("rate_limit", "server_error", "connect_error")
    assert first["query"] == CASES[0]["message"]
    assert "SECRET-DETAIL" not in json.dumps(data) and "SECRET-DETAIL" not in caplog.text


def test_production_retry_policy_is_restored_afterwards(tmp_path):
    from support_agent import cohere_calls

    before = cohere_calls.MAX_RETRIES
    ev.record_rewrites(CASES[:2], tmp_path / "a.json", client=FakeChatClient())
    assert cohere_calls.MAX_RETRIES == before == 1

    class Abort(BaseException):
        """Like Ctrl+C: not an Exception, so nothing in the rewriter catches it."""

    with pytest.raises(Abort):
        ev.record_rewrites(CASES[:2], tmp_path / "b.json", client=FakeChatClient(Abort()))
    assert not (tmp_path / "b.json").exists()  # an aborted run leaves no partial file
    assert cohere_calls.MAX_RETRIES == before


def test_single_request_client_blocks_a_second_request():
    """The guard holds even if a retry happened anyway."""
    inner = FakeChatClient()
    guarded = ev.SingleRequestClient(inner)
    options = {"request_options": {"timeout": 8, "max_retries": 0}}

    guarded.chat(**options)
    with pytest.raises(RuntimeError, match="blocked"):
        guarded.chat(**options)

    assert len(inner.calls) == 1 and guarded.requests == 1 and guarded.blocked == 1
    guarded.start_case()
    guarded.chat(**options)
    assert len(inner.calls) == 2


def test_single_request_client_refuses_sdk_retries_without_sending():
    inner = FakeChatClient()
    guarded = ev.SingleRequestClient(inner)
    with pytest.raises(RuntimeError, match="retries"):
        guarded.chat(request_options={"timeout": 8, "max_retries": 2})
    assert inner.calls == [] and guarded.requests == 0


def test_each_case_records_latency_and_its_request(tmp_path, ticking_clock):
    data = ev.record_rewrites(CASES, tmp_path / "r.json", client=FakeChatClient())

    for result in data["results"].values():
        assert result["requests"] == 1 and result["latency_ms"] == 250.0 and result["error"] is None
    assert data["version"] == 2 and data["max_retries"] == 0


def test_successes_and_failures_are_recorded_distinctly(tmp_path, ticking_clock):
    standalone = CASES[0]["reference_query"]
    client = FakeChatClient(
        json.dumps({"standalone_query": standalone, "needs_clarification": False}),  # rewritten
        StatusError(500),  # request failed
        "not json",  # request succeeded, output rejected
        TimeoutError(),  # request failed
    )

    data = ev.record_rewrites(CASES[:5], tmp_path / "r.json", client=client)
    r = [data["results"][c["id"]] for c in CASES[:5]]

    assert (r[0]["outcome"], r[0]["query"], r[0]["error"]) == ("rewritten", standalone, None)
    assert (r[1]["outcome"], r[1]["error"], r[1]["latency_ms"]) == ("failed", "server_error", 250.0)
    assert (r[2]["outcome"], r[2]["error"], r[2]["query"]) == ("failed", None, CASES[2]["message"])
    assert (r[3]["outcome"], r[3]["error"]) == ("failed", "timeout")
    assert r[4]["outcome"] == "needs_clarification"
    assert all(x["requests"] == 1 for x in r)

    q = ev.request_metrics(CASES[:5], data["results"])
    assert (q["attempted"], q["succeeded"], q["failed"]) == (5, 3, 2)
    assert q["errors"] == ["server_error", "timeout"]
    assert q["latency_ms"] == {"mean": 250.0, "median": 250.0, "max": 250.0, "measured": 5}


def test_results_contain_no_credentials(never_read_dotenv, fake_sdk, monkeypatch, tmp_path):
    monkeypatch.setenv("COHERE_API_KEY", "KEY-MUST-NOT-APPEAR")
    path = tmp_path / "r.json"
    ev.record_rewrites(CASES, path)
    assert "KEY-MUST-NOT-APPEAR" not in path.read_text(encoding="utf-8")


def test_recorded_file_round_trips_into_offline_scoring(tmp_path, block_cohere, capsys, ticking_clock):
    path = tmp_path / "results" / "rewrites.json"
    ev.record_rewrites(CASES, path, client=FakeChatClient(default="not json"))

    assert ev.main(["--rewrites", str(path)]) == 0  # cohere is blocked: no client can be made
    output = capsys.readouterr().out
    assert f"Requests: attempted {len(CASES)}, succeeded {len(CASES)}, failed 0" in output
    assert "Latency (ms): mean 250.0, median 250.0, max 250.0" in output
    for mode in ("recorded rewrites", "message", "naive", "reference"):
        assert mode in output


def test_offline_scoring_never_creates_a_client(tmp_path, block_cohere, never_read_dotenv, monkeypatch, capsys):
    monkeypatch.setattr(ev, "recording_client", lambda *a: pytest.fail("client created"))
    assert ev.main(["--rewrites", str(_rewrites_file(tmp_path))]) == 0
    assert "not recorded (version 1 file)" in capsys.readouterr().out


def test_version_1_files_still_score(tmp_path):
    data = ev.load_rewrites(_rewrites_file(tmp_path, version=1), CASES)
    assert ev.request_metrics(CASES, data["results"]) is None


@pytest.mark.parametrize("make_existing", [lambda p: p.write_text("{}", encoding="utf-8"), lambda p: p.mkdir()])
def test_existing_output_is_never_overwritten_and_no_request_is_made(tmp_path, make_existing):
    path = tmp_path / "rewrites.json"
    make_existing(path)
    client = FakeChatClient()
    with pytest.raises(ev.DatasetError, match="already exists"):
        ev.record_rewrites(CASES, path, client=client)
    assert client.calls == []


def test_a_file_created_during_recording_is_not_overwritten(tmp_path):
    path = tmp_path / "rewrites.json"

    class AppearsMidway(FakeChatClient):
        def chat(self, **kwargs):
            if not path.exists():
                path.write_text("SOMEONE ELSE'S RESULTS", encoding="utf-8")
            return super().chat(**kwargs)

    with pytest.raises(ev.DatasetError, match="already exists"):
        ev.record_rewrites(CASES, path, client=AppearsMidway())
    assert path.read_text(encoding="utf-8") == "SOMEONE ELSE'S RESULTS"
