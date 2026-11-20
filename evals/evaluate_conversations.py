"""Evaluate follow-up handling on the multi-turn dataset (conversation_eval.json).

Offline by default (no Cohere calls):

    python evals/evaluate_conversations.py
        Validates the dataset, then compares keyword retrieval for each
        conversation's last message used three ways:
          message    the message alone (what retrieval saw before conversations)
          naive      the earlier customer messages + the message, concatenated
          reference  the hand-written standalone question (an upper bound for
                     what a perfect rewrite could give keyword retrieval)
        This measures how much context matters for retrieval. It does NOT
        measure the rewriter: the reference questions were written by hand.

    python evals/evaluate_conversations.py --rewrites FILE
        Scores rewrites recorded earlier (offline): clarification decisions,
        required/forbidden terms, numbers the customer never wrote, and keyword
        retrieval with the recorded queries.

LIVE (calls Cohere; run only with explicit approval):

    python evals/evaluate_conversations.py --record-rewrites FILE
        Runs query_rewrite.rewrite_question on every case and saves the
        results to a new FILE (never overwrites), with each request's latency.
        Controlled so the number of requests is known in advance:
          * The API key comes only from the COHERE_API_KEY environment
            variable. Recording never calls load_dotenv() and never reads .env.
          * No retries: at most one Cohere request per case (32 for 32 cases).
            A failed request is recorded as a failure.
        The app's own client and retry policy are not changed.
"""

import argparse
import json
import re
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

from evaluate_kb_v2 import ARTICLE_ID, DatasetError, fact_present, sha256_json  # noqa: E402

from support_agent.kb import load_articles  # noqa: E402
from support_agent.query_rewrite import (  # noqa: E402
    FAILED,
    MAX_QUERY_CHARS,
    NEEDS_CLARIFICATION,
    REWRITTEN,
    _numbers,
)

DATASET_PATH = PROJECT_ROOT / "evals" / "conversation_eval.json"
DEFAULT_K = 3
REWRITES_VERSION = 2  # 2 added per-case requests, latency_ms and error.
SUPPORTED_REWRITES_VERSIONS = (1, 2)
LABELS = ("follow_up", "standalone", "topic_switch", "needs_clarification")
TAGS = ("pronoun", "ellipsis", "number", "informal", "topic_change", "short", "multi_intent")
CASE_ID = re.compile(r"^C-\d{3}$")
MODES = ("message", "naive", "reference")


# --- Dataset ---------------------------------------------------------------------


def customer_text(case: dict) -> str:
    return " ".join([m["content"] for m in case["history"] if m["role"] == "customer"] + [case["message"]])


def validate_cases(cases, articles: list[dict]) -> None:
    """Raise DatasetError naming the first problem found."""
    if not isinstance(cases, list) or not cases:
        raise DatasetError("The dataset must be a non-empty list.")
    kb_ids = {a["id"] for a in articles}
    seen = set()
    for case in cases:
        where = case.get("id", "?") if isinstance(case, dict) else "?"
        if not isinstance(case, dict):
            raise DatasetError(f"{where}: each case must be an object")
        if not isinstance(case.get("id"), str) or not CASE_ID.match(case["id"]):
            raise DatasetError(f"{where}: id must look like C-001")
        if case["id"] in seen:
            raise DatasetError(f"{where}: duplicate id")
        seen.add(case["id"])
        if case.get("label") not in LABELS:
            raise DatasetError(f"{where}: label must be one of {', '.join(LABELS)}")
        if not isinstance(case.get("message"), str) or not case["message"].strip():
            raise DatasetError(f"{where}: message must be non-empty text")
        if not isinstance(case.get("tags"), list) or any(t not in TAGS for t in case["tags"]):
            raise DatasetError(f"{where}: tags must be a list of {', '.join(TAGS)}")

        history = case.get("history")
        if not isinstance(history, list) or not history or len(history) % 2:
            raise DatasetError(f"{where}: history must be customer/assistant pairs")
        for index, message in enumerate(history):
            role = "customer" if index % 2 == 0 else "assistant"
            if not isinstance(message, dict) or message.get("role") != role:
                raise DatasetError(f"{where}: history must alternate customer, assistant, ...")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise DatasetError(f"{where}: history messages must be non-empty text")

        if case["label"] == "needs_clarification":
            if not isinstance(case.get("note"), str) or not case["note"].strip():
                raise DatasetError(f"{where}: needs_clarification cases need a note")
            unexpected = {"reference_query", "expected_ids", "required_terms", "forbidden_terms"} & set(case)
            if unexpected:
                raise DatasetError(f"{where}: unexpected fields {sorted(unexpected)}")
            continue

        reference = case.get("reference_query")
        if not isinstance(reference, str) or not reference.strip() or len(reference) > MAX_QUERY_CHARS:
            raise DatasetError(f"{where}: reference_query must be non-empty and at most {MAX_QUERY_CHARS} characters")
        expected = case.get("expected_ids")
        if not isinstance(expected, list) or not expected or len(set(expected)) != len(expected):
            raise DatasetError(f"{where}: expected_ids must be a non-empty list without duplicates")
        for article_id in expected:
            if not isinstance(article_id, str) or not ARTICLE_ID.match(article_id) or article_id not in kb_ids:
                raise DatasetError(f"{where}: unknown article {article_id!r}")
        required = case.get("required_terms")
        if not isinstance(required, list) or not required or any(
            not isinstance(group, list) or not group or any(not isinstance(t, str) or not t.strip() for t in group)
            for group in required
        ):
            raise DatasetError(f"{where}: required_terms must be a non-empty list of non-empty alternative lists")
        forbidden = case.get("forbidden_terms")
        if not isinstance(forbidden, list) or any(not isinstance(t, str) or not t.strip() for t in forbidden):
            raise DatasetError(f"{where}: forbidden_terms must be a list of non-empty strings")
        # The reference must pass the same checks a real rewrite has to pass.
        missing = [group for group in required if not _terms_present(reference, group)]
        if missing:
            raise DatasetError(f"{where}: reference_query lacks required terms {missing}")
        if any(_terms_present(reference, [term]) for term in forbidden):
            raise DatasetError(f"{where}: reference_query contains a forbidden term")
        if not _numbers(reference) <= _numbers(customer_text(case)):
            raise DatasetError(f"{where}: reference_query contains a number the customer never wrote")
        if case["label"] == "standalone" and reference != case["message"]:
            raise DatasetError(f"{where}: a standalone case's reference_query must equal its message")


def _terms_present(text: str, alternatives: list[str]) -> bool:
    """Whole-word match, except that a term may be a word prefix ("delet" matches "deleting")."""
    lowered = " ".join(text.lower().split())
    return any(re.search(rf"(?<!\w){re.escape(alt.lower())}", lowered) for alt in alternatives)


def load_cases(path: Path = DATASET_PATH) -> list[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DatasetError(f"Could not read {path}: {type(error).__name__}") from None


# --- Retrieval comparison --------------------------------------------------------


def queries_for(case: dict, mode: str) -> str:
    if mode == "message":
        return case["message"]
    if mode == "naive":
        return customer_text(case)
    return case["reference_query"]


def retrieval_hits(cases: list[dict], retriever, k: int, queries: dict[str, str]) -> dict[str, bool]:
    """Case id -> whether any expected article is in the top k for that query."""
    return {
        case["id"]: bool(set(case["expected_ids"]) & {a["id"] for a in retriever(queries[case["id"]], top_k=k)[:k]})
        for case in cases if case["id"] in queries
    }


def retrieval_comparison(cases: list[dict], retriever, k: int) -> dict:
    answerable = [c for c in cases if c["label"] != "needs_clarification"]
    result = {}
    for mode in MODES:
        hits = retrieval_hits(answerable, retriever, k, {c["id"]: queries_for(c, mode) for c in answerable})
        by_label = {}
        for case in answerable:
            entry = by_label.setdefault(case["label"], {"questions": 0, "hits": 0})
            entry["questions"] += 1
            entry["hits"] += hits[case["id"]]
        result[mode] = {"questions": len(answerable), "hits": sum(hits.values()), "by_label": by_label,
                        "misses": sorted(i for i, hit in hits.items() if not hit)}
    return result


# --- Recorded rewrites -----------------------------------------------------------


def load_rewrites(path: Path, cases: list[dict]) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") not in SUPPORTED_REWRITES_VERSIONS:
        raise DatasetError(f"Unsupported rewrites file version: {data.get('version')!r}")
    if data.get("dataset_sha256") != sha256_json(cases):
        raise DatasetError("The conversation set changed since these rewrites were recorded.")
    results = data.get("results")
    missing = [c["id"] for c in cases if not isinstance(results, dict) or c["id"] not in results]
    if missing:
        raise DatasetError(f"Rewrites missing for: {', '.join(missing[:5])}")
    for case_id, result in results.items():
        if not isinstance(result, dict) or not isinstance(result.get("query"), str) or not isinstance(result.get("outcome"), str):
            raise DatasetError(f"{case_id}: each result needs a query and an outcome")
    return data


def rewrite_metrics(cases: list[dict], results: dict[str, dict], retriever, k: int) -> dict:
    """Score recorded rewrites against the hand labels."""
    outcomes = {}
    clarify = {"expected": 0, "asked": 0, "correct": 0, "false_alarm": []}
    terms = {"questions": 0, "all_required": 0, "forbidden_leaks": [], "invented_numbers": []}
    for case in cases:
        result = results[case["id"]]
        outcomes[result["outcome"]] = outcomes.get(result["outcome"], 0) + 1
        asked = result["outcome"] == NEEDS_CLARIFICATION
        clarify["asked"] += asked
        if case["label"] == "needs_clarification":
            clarify["expected"] += 1
            clarify["correct"] += asked
            continue
        if asked:
            clarify["false_alarm"].append(case["id"])
            continue
        query = result["query"]
        terms["questions"] += 1
        terms["all_required"] += all(_terms_present(query, group) for group in case["required_terms"])
        if any(_terms_present(query, [term]) for term in case["forbidden_terms"]):
            terms["forbidden_leaks"].append(case["id"])
        if not _numbers(query) <= _numbers(customer_text(case)):
            terms["invented_numbers"].append(case["id"])

    scored = [c for c in cases if c["label"] != "needs_clarification" and results[c["id"]]["outcome"] != NEEDS_CLARIFICATION]
    hits = retrieval_hits(scored, retriever, k, {c["id"]: results[c["id"]]["query"] for c in scored})
    # The same cases with the baseline queries, so the recorded rewrites are
    # compared like for like (cases where it asked for clarification are excluded).
    baselines = {
        mode: sum(retrieval_hits(scored, retriever, k, {c["id"]: queries_for(c, mode) for c in scored}).values())
        for mode in MODES
    }
    return {
        "outcomes": outcomes,
        "clarification": clarify,
        "terms": terms,
        "retrieval": {"questions": len(scored), "hits": sum(hits.values()),
                      "misses": sorted(i for i, hit in hits.items() if not hit),
                      "baselines": baselines},
        "requests": request_metrics(cases, results),
    }


def request_metrics(cases: list[dict], results: dict[str, dict]) -> dict | None:
    """Requests and latency from a version-2 recording (None for version 1,
    which did not record them)."""
    recorded = [results[c["id"]] for c in cases]
    if not all("requests" in r for r in recorded):
        return None
    attempted = sum(r["requests"] for r in recorded)
    failed = sum(1 for r in recorded if r["requests"] and r.get("error"))
    latencies = [r["latency_ms"] for r in recorded if isinstance(r.get("latency_ms"), (int, float))]
    return {
        "attempted": attempted,
        "succeeded": attempted - failed,
        "failed": failed,
        "errors": sorted({r["error"] for r in recorded if r.get("error")}),
        "latency_ms": None if not latencies else {
            "mean": round(statistics.fmean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "max": round(max(latencies), 1),
            "measured": len(latencies),
        },
    }


clock = time.perf_counter  # Replaceable in tests.


class RecordingError(DatasetError):
    """Recording can't start (e.g. no API key). Never contains a credential."""


def recording_client(environ=None):
    """A Cohere client for recording, keyed only from the process environment.

    Unlike the app's agent._default_client(), this never calls load_dotenv(),
    so .env is never read. The app's client is not touched.
    """
    from support_agent.agent import ConfigurationError, api_key_from_env

    try:
        api_key = api_key_from_env(environ)  # Reads os.environ only; strips; never echoes the key.
    except ConfigurationError:
        raise RecordingError(
            "COHERE_API_KEY is not set in this process's environment. Recording does not "
            "read .env; set the variable in the shell that runs this command."
        ) from None

    import cohere

    return cohere.ClientV2(api_key=api_key)


class SingleRequestClient:
    """Wraps the Cohere client for one case at a time: lets exactly one chat
    request through, times it, and remembers the failure category (never the
    provider's message). A second request for the same case is refused before
    it reaches the SDK, so even a retry can't send it."""

    def __init__(self, client):
        self._client = client
        self.start_case()

    def start_case(self) -> None:
        self.requests = 0
        self.latency_ms = None
        self.error = None
        self.blocked = 0

    def chat(self, **kwargs):
        if self.requests:
            self.blocked += 1
            raise RuntimeError("second request for this case blocked")
        options = kwargs.get("request_options") or {}
        if options.get("max_retries") != 0:  # Refused before anything is sent.
            raise RuntimeError("SDK retries must be disabled for recording")
        self.requests += 1
        from support_agent.cohere_calls import classify_error

        started = clock()
        try:
            return self._client.chat(**kwargs)
        except Exception as error:
            self.error = classify_error(error)
            raise
        finally:
            self.latency_ms = round((clock() - started) * 1000, 1)


@contextmanager
def no_retries():
    """Turn off the one retry call_with_retries allows, for this process only,
    and restore it afterwards. (The SDK's own retries are already off: every
    request is sent with max_retries=0.)"""
    from support_agent import cohere_calls

    saved = cohere_calls.MAX_RETRIES
    cohere_calls.MAX_RETRIES = 0
    try:
        yield
    finally:
        cohere_calls.MAX_RETRIES = saved


def record_rewrites(cases: list[dict], path: Path, *, client=None) -> dict:
    """LIVE: run the production rewriter once per case and save the results.

    At most one Cohere request per case, no retries, no .env. `client` is for
    tests; by default a client is made from COHERE_API_KEY in the environment.
    """
    path = Path(path)
    if path.exists():
        raise DatasetError(f"{path} already exists; choose a new file name.")

    from support_agent.query_rewrite import rewrite_model, rewrite_question

    guarded = SingleRequestClient(client if client is not None else recording_client())
    results = {}
    with no_retries():
        for number, case in enumerate(cases, start=1):
            guarded.start_case()
            rewrite = rewrite_question(case["message"], case["history"], guarded)
            results[case["id"]] = {
                "query": rewrite.query,
                "outcome": rewrite.outcome,
                "requests": guarded.requests,  # 0 or 1
                "latency_ms": guarded.latency_ms,  # the request alone; None if none was sent
                "error": guarded.error,  # failure category, e.g. "rate_limit"; None if the request succeeded
            }
            if guarded.blocked:  # Can't happen with retries off; recorded rather than hidden.
                results[case["id"]]["blocked_requests"] = guarded.blocked
            shown = f" ({guarded.latency_ms} ms)" if guarded.latency_ms is not None else ""
            print(f"[{number}/{len(cases)}] {case['id']} {rewrite.outcome}{shown}")

    data = {
        "version": REWRITES_VERSION,
        "recorded_on": date.today().isoformat(),
        "rewrite_model": rewrite_model(),
        "max_retries": 0,
        "dataset_sha256": sha256_json(cases),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as file:  # "x": fails if the file appeared meanwhile.
            file.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    except FileExistsError:
        raise DatasetError(f"{path} already exists; choose a new file name.") from None
    return data


# --- Reports ---------------------------------------------------------------------


def _pct(part: float, whole: float) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else f"{part}/0"


def print_summary(cases: list[dict]) -> None:
    labels = {label: sum(c["label"] == label for c in cases) for label in LABELS}
    print(f"Conversations: {len(cases)}  " + "  ".join(f"{k}={v}" for k, v in labels.items()))


def print_comparison(metrics: dict, k: int) -> None:
    print(f"\nKEYWORD RETRIEVAL, Expected@{k} (any expected article in the top {k})")
    labels = sorted({label for m in metrics.values() for label in m["by_label"]})
    print(f"  {'query used':<12}{'all':>14}" + "".join(f"{label:>18}" for label in labels))
    for mode, m in metrics.items():
        row = f"  {mode:<12}{_pct(m['hits'], m['questions']):>14}"
        row += "".join(f"{_pct(m['by_label'][l]['hits'], m['by_label'][l]['questions']):>18}" for l in labels)
        print(row)
    print("  'reference' uses hand-written questions: an upper bound, not a measurement of the rewriter.")
    for mode, m in metrics.items():
        if m["misses"]:
            print(f"  misses ({mode}): {', '.join(m['misses'])}")


def print_rewrite_report(metrics: dict, meta: dict, k: int) -> None:
    print(f"\nRECORDED REWRITES ({meta.get('rewrite_model', '?')}, {meta.get('recorded_on', '?')})")
    print("  Outcomes: " + ", ".join(f"{name}={count}" for name, count in sorted(metrics["outcomes"].items())))
    c = metrics["clarification"]
    print(f"  Clarification asked when needed   {_pct(c['correct'], c['expected'])}")
    print(f"  Clarification asked unnecessarily {len(c['false_alarm'])}" + (f" ({', '.join(c['false_alarm'])})" if c["false_alarm"] else ""))
    t = metrics["terms"]
    print(f"  All required terms present        {_pct(t['all_required'], t['questions'])}")
    print(f"  Forbidden (stale-topic) terms     {len(t['forbidden_leaks'])}" + (f" ({', '.join(t['forbidden_leaks'])})" if t["forbidden_leaks"] else ""))
    print(f"  Numbers the customer never wrote  {len(t['invented_numbers'])}")
    r = metrics["retrieval"]
    print(f"  Keyword Expected@{k} on the {r['questions']} answerable cases not sent to clarification:")
    print(f"    recorded rewrites  {_pct(r['hits'], r['questions'])}")
    for mode, hits in r["baselines"].items():
        print(f"    {mode:<18} {_pct(hits, r['questions'])}")
    q = metrics["requests"]
    if q is None:
        print("  Requests and latency: not recorded (version 1 file)")
    else:
        print(f"  Requests: attempted {q['attempted']}, succeeded {q['succeeded']}, failed {q['failed']}"
              + (f" ({', '.join(q['errors'])})" if q["errors"] else ""))
        lat = q["latency_ms"]
        if lat:
            print(f"  Latency (ms): mean {lat['mean']}, median {lat['median']}, max {lat['max']} ({lat['measured']} requests)")
    if metrics["outcomes"].get(FAILED):
        print(f"  Note: {metrics['outcomes'][FAILED]} rewrite(s) failed and used the message as written.")
    if not metrics["outcomes"].get(REWRITTEN):
        print("  Note: no case was rewritten.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="cut-off for Expected@K (default 3)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--rewrites", type=Path, help="score a recorded rewrites file (offline)")
    action.add_argument("--record-rewrites", type=Path, help="LIVE: record rewrites to a new file")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least 1")

    from support_agent.kb import retrieve

    try:
        cases = load_cases()
        validate_cases(cases, load_articles())
        print_summary(cases)
        if args.record_rewrites:
            record_rewrites(cases, args.record_rewrites)
            print(f"Saved rewrites to {args.record_rewrites}")
            return 0
        if args.rewrites:
            data = load_rewrites(args.rewrites, cases)
            print_rewrite_report(rewrite_metrics(cases, data["results"], retrieve, args.k), data, args.k)
            return 0
        print_comparison(retrieval_comparison(cases, retrieve, args.k), args.k)
    except DatasetError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
