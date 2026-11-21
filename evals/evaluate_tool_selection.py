"""Evaluate tool selection and tool-call handling (tool_selection_eval.json).

Offline only: no Cohere call is ever made (sockets are blocked while the
pipeline runs). Three separate measurements, never mixed:

  1. The offer rule (production code): tools are offered only when the
     customer has written an order number (agent.mentions_order_number).
     This is the real rule, measured for real.

  2. A deterministic keyword BASELINE (this file only, not used by the app):
     a few hand-written rules that pick tools from the question. It is
     scored with the selection metrics and a confusion matrix, as a
     reference point that a real model should beat. It is NOT the model.

  3. Orchestration with SCRIPTED model decisions: each case's `script` is fed
     to the real agent (answer_question -> generate_answer_with_tools ->
     tools registry) through a fake Cohere client. This checks what the
     pipeline DOES with a given decision (which tools run, their statuses,
     the outcome), including deliberately wrong decisions. The decisions are
     written by hand, so this says nothing about how well a real model
     chooses tools.

    python evals/evaluate_tool_selection.py
    python evals/evaluate_tool_selection.py --output evals/results/tool_selection_<date>.json
    python evals/evaluate_tool_selection.py --selections FILE
        Score recorded selections ({"version": 1, "source": "...",
        "selections": {"TS-001": ["get_order_status"], ...}}), e.g. from a
        future, separately approved live run, with the same metrics as the
        baseline. Nothing here records live selections.
"""

import argparse
import json
import logging
import re
import socket
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

from evaluate_kb_v2 import DatasetError, sha256_json  # noqa: E402

from support_agent import agent, answer, tools  # noqa: E402
from support_agent.demo_orders import load_orders  # noqa: E402
from support_agent.kb import retrieve  # noqa: E402
from support_agent.query_rewrite import NOT_NEEDED, Rewrite  # noqa: E402
from support_agent.rerank import RerankConfig  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "evals" / "tool_selection_eval.json"
TODAY = date(2026, 11, 20)  # Fixed, so order ages (and refund decisions) never drift.
TOOL_NAMES = tuple(tools.REGISTRY)
NONE = "none"

CATEGORIES = (
    "order_status", "refund", "cancellation", "general_kb", "order_number_unrelated",
    "missing_details", "ambiguous_multi_intent", "adversarial", "invalid_model_output",
)
SCENARIOS = ("faithful", "misbehaving")
OUTCOME_REASONS = (
    "answered", "cancellation_available", "needs_order_details",
    "invalid_tool_call", "malformed_response", "tool_error", "no_articles", "no_citations",
)
ALL_STATUSES = (
    tools.OK, tools.CONFIRMATION_REQUIRED, tools.NOT_CANCELLABLE, tools.NOT_FOUND,
    tools.DETAILS_NOT_PROVIDED, tools.INVALID_ARGUMENTS, tools.UNKNOWN_ITEM, tools.UNKNOWN_TOOL,
    tools.LIMIT_EXCEEDED, tools.ERROR,
)
# Statuses that reveal something about a real order: allowed only when the
# customer wrote both the order number and the email.
DATA_STATUSES = {tools.OK, tools.CONFIRMATION_REQUIRED, tools.NOT_CANCELLABLE, tools.UNKNOWN_ITEM}
# Safe ends for a question that lacks the customer's details.
SAFE_MISSING_DETAILS_REASONS = {"answered", "needs_order_details", "invalid_tool_call", "malformed_response",
                                "no_articles", "no_citations"}
SAFE_INVALID_CALL_REASONS = {"invalid_tool_call", "malformed_response"}

_CASE_FIELDS = {"id", "category", "scenario", "question", "expected_tools", "script", "expected_outcome"}
_OPTIONAL_FIELDS = {"acceptable", "history", "note"}
_CASE_ID = re.compile(r"^TS-\d{3}$")
_ORDER_NUMBER = re.compile(r"(?<![A-Za-z0-9-])ORD-\d{4}(?!\d)", re.IGNORECASE)
# Any written email address, in any case (customers write "Luc.Tremblay@Example.com").
_EMAIL = re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)


# --- The dataset ---------------------------------------------------------------------


def label(selection) -> str:
    """One label for a selection: tool names, sorted and joined, or "none"."""
    names = sorted(set(selection))
    return "+".join(names) if names else NONE


def validate_cases(cases, book=None) -> None:
    """Raise DatasetError if any case breaks the dataset rules."""
    book = book or load_orders()
    if not isinstance(cases, list) or not cases:
        raise DatasetError("the dataset has no cases")
    seen = set()
    for case in cases:
        where = case.get("id", "?") if isinstance(case, dict) else "?"
        if not isinstance(case, dict):
            raise DatasetError("every case must be an object")
        missing = _CASE_FIELDS - set(case)
        extra = set(case) - _CASE_FIELDS - _OPTIONAL_FIELDS
        if missing or extra:
            raise DatasetError(f"{where}: missing {sorted(missing)} / unexpected {sorted(extra)}")
        if not isinstance(case["id"], str) or not _CASE_ID.fullmatch(case["id"]) or case["id"] in seen:
            raise DatasetError(f"{where}: invalid or duplicate id")
        seen.add(case["id"])
        if case["category"] not in CATEGORIES or case["scenario"] not in SCENARIOS:
            raise DatasetError(f"{where}: unknown category or scenario")
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise DatasetError(f"{where}: empty question")
        for selection in [case["expected_tools"], *case.get("acceptable", [])]:
            if not isinstance(selection, list) or any(name not in TOOL_NAMES for name in selection):
                raise DatasetError(f"{where}: selections must list registry tools")
        for message in case.get("history", []):
            if not isinstance(message, dict) or message.get("role") not in ("customer", "assistant") \
                    or not isinstance(message.get("content"), str):
                raise DatasetError(f"{where}: invalid history message")
        _validate_script(case, where)
        _validate_fictional(case, book, where)
        outcome = case["expected_outcome"]
        if set(outcome) != {"offered", "reason", "statuses"} or not isinstance(outcome["offered"], bool) \
                or outcome["reason"] not in OUTCOME_REASONS \
                or not all(status in ALL_STATUSES for status in outcome["statuses"]):
            raise DatasetError(f"{where}: invalid expected_outcome")
        if outcome["offered"] != offers_tools(case):
            raise DatasetError(f"{where}: expected_outcome.offered contradicts the order numbers written")


def _validate_script(case: dict, where: str) -> None:
    script = case["script"]
    if not isinstance(script, dict) or set(script) - {"calls", "answer", "duplicate_call_ids"} \
            or (("calls" in script) == ("answer" in script)):
        raise DatasetError(f"{where}: script needs exactly one of 'calls' or 'answer'")
    for call in script.get("calls", []):
        if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["name"], str):
            raise DatasetError(f"{where}: each scripted call needs a name and arguments")
    if "answer" in script and not isinstance(script["answer"], str):
        raise DatasetError(f"{where}: scripted answer must be text")


def _validate_fictional(case: dict, book, where: str) -> None:
    """Only demo order numbers and example.com emails anywhere in the case."""
    text = json.dumps(case, ensure_ascii=False)
    for order_id in _ORDER_NUMBER.findall(text):
        if order_id.upper() != "ORD-1234" and book.status(order_id.upper()) is None:  # 1234: the fixed reply's example
            raise DatasetError(f"{where}: {order_id} is not a demo order")
    for email in _EMAIL.findall(text):
        if not email.lower().endswith("@example.com"):
            raise DatasetError(f"{where}: emails must be fictional (@example.com)")


def load_cases(path: Path = DATASET_PATH) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise DatasetError("unsupported dataset version")
    cases = data.get("cases")
    validate_cases(cases)
    return cases


def customer_messages(case: dict) -> list[str]:
    return [m["content"] for m in case.get("history", []) if m["role"] == "customer"] + [case["question"]]


def offers_tools(case: dict) -> bool:
    """The production offer rule, applied to this case."""
    return agent.mentions_order_number(case["question"], case.get("history"))


# --- Selection metrics -----------------------------------------------------------------

SELECTION_BUCKETS = ("correct_tool", "correct_no_tool", "incorrect_tool", "missing_tool", "unnecessary_tool")


def classify(case: dict, selection) -> str:
    """Which bucket a selection falls in. Order and repeats are ignored; a
    selection listed in `acceptable` counts as correct."""
    chosen = label(selection)
    allowed = {label(case["expected_tools"])} | {label(s) for s in case.get("acceptable", [])}
    if chosen in allowed:
        return "correct_no_tool" if chosen == NONE else "correct_tool"
    expected_none = not case["expected_tools"]
    if chosen == NONE:
        return "missing_tool"
    return "unnecessary_tool" if expected_none else "incorrect_tool"


def score_selections(cases: list[dict], selections: dict) -> dict:
    """Selection metrics for {case id: [tool names]}. Every case needs a
    selection. Names outside the registry are kept as they are (always wrong)."""
    missing = [c["id"] for c in cases if c["id"] not in selections]
    if missing:
        raise DatasetError(f"no selection for {len(missing)} case(s), e.g. {missing[0]}")
    per_case, buckets, matrix = [], Counter(), Counter()
    by_category: dict[str, Counter] = {}
    for case in cases:
        selection = selections[case["id"]]
        if not isinstance(selection, list) or not all(isinstance(name, str) for name in selection):
            raise DatasetError(f"{case['id']}: a selection must be a list of tool names")
        bucket = classify(case, selection)
        buckets[bucket] += 1
        by_category.setdefault(case["category"], Counter())[bucket] += 1
        matrix[(label(case["expected_tools"]), label(selection))] += 1
        per_case.append({"id": case["id"], "category": case["category"], "expected": label(case["expected_tools"]),
                         "selected": label(selection), "bucket": bucket})
    total = len(cases)
    correct = buckets["correct_tool"] + buckets["correct_no_tool"]
    return {
        "cases": total,
        "buckets": {name: buckets[name] for name in SELECTION_BUCKETS},
        "accuracy": correct / total if total else 0.0,
        "by_category": {cat: {name: counts[name] for name in SELECTION_BUCKETS}
                        for cat, counts in sorted(by_category.items())},
        "confusion": [{"expected": e, "selected": s, "count": n} for (e, s), n in sorted(matrix.items())],
        "per_case": per_case,
    }


# --- 1. The production offer rule ---------------------------------------------------------


def offer_metrics(cases: list[dict]) -> dict:
    """How the offer rule relates to whether a tool is actually needed.
    'needed' = the ideal selection uses a tool."""
    counts = Counter()
    for case in cases:
        needed = bool(case["expected_tools"])
        counts[("offered" if offers_tools(case) else "not_offered", "needed" if needed else "not_needed")] += 1
    return {
        "offered_and_needed": counts[("offered", "needed")],
        "offered_not_needed": counts[("offered", "not_needed")],
        "not_offered_but_needed": counts[("not_offered", "needed")],
        "not_offered_not_needed": counts[("not_offered", "not_needed")],
    }


# --- 2. The deterministic keyword baseline (eval only) ---------------------------------------

_BASELINE_RULES = (
    ("request_order_cancellation", re.compile(r"\bcancel|\bstop my order", re.IGNORECASE)),
    ("check_refund_eligibility", re.compile(
        r"\b(refund|return|money back|broken|damaged|defective|wrong item|missing)", re.IGNORECASE)),
    ("get_order_status", re.compile(r"\b(where|status|track|shipped|ship|arriv|deliver|when)", re.IGNORECASE)),
)


def baseline_select(case: dict) -> list[str]:
    """Hand-written rules, not a model: no order number or no email in the
    customer's messages -> no tool (the tool descriptions require both);
    otherwise every tool whose keywords appear."""
    texts = customer_messages(case)
    if not any(_ORDER_NUMBER.search(t) for t in texts) or not any(_EMAIL.search(t) for t in texts):
        return []
    text = " ".join(texts)
    return [name for name, pattern in _BASELINE_RULES if pattern.search(text)]


# --- 3. Orchestration with scripted model decisions ------------------------------------------


def _chat_response(payload: dict):
    import cohere  # The installed SDK's own models, so attribute names are real. No client is created.

    return cohere.ChatResponse.model_validate(payload)


class ScriptedClient:
    """A fake Cohere client that plays one case's scripted decision.

    First request: if tools were offered and the script has calls, a
    TOOL_CALL response with them (call_1, call_2, ...; the same ID twice
    when `duplicate_call_ids`). Otherwise (no tools offered, or a scripted
    answer) a text answer citing the first supplied article: a model can't
    call tools it wasn't given. Second request (after tool results): a text
    answer citing every citable tool result, or the first article.
    Records each request's shape, never sends anything."""

    def __init__(self, script: dict):
        self.script = script
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append({
            "tools": [t["function"]["name"] for t in kwargs.get("tools") or []],
            "documents": [d["id"] for d in kwargs.get("documents") or []],
            "tool_choice": kwargs.get("tool_choice"),
            "tool_messages": sum(1 for m in kwargs["messages"] if isinstance(m, dict) and m.get("role") == "tool"),
        })
        if len(self.requests) == 1 and kwargs.get("tools") and "calls" in self.script:
            return self._tool_call()
        return self._answer(kwargs)

    def _tool_call(self):
        calls = []
        for index, call in enumerate(self.script["calls"], start=1):
            arguments = call["arguments"] if isinstance(call["arguments"], str) else json.dumps(call["arguments"])
            call_id = "call_1" if self.script.get("duplicate_call_ids") else f"call_{index}"
            calls.append({"id": call_id, "type": "function", "function": {"name": call["name"], "arguments": arguments}})
        return _chat_response({"id": "scripted", "finish_reason": "TOOL_CALL", "message": {
            "role": "assistant", "tool_plan": "Scripted decision.", "tool_calls": calls}})

    def _answer(self, kwargs):
        sources = []
        for message in kwargs["messages"]:
            if isinstance(message, dict) and message.get("role") == "tool":
                data = message["content"][0]["document"]["data"]
                if data.get("status") not in answer._UNCITABLE:
                    sources.append({"type": "tool", "id": message["tool_call_id"] + ":0", "tool_output": {}})
        documents = kwargs.get("documents") or []
        if not sources and documents:
            sources.append({"type": "document", "id": documents[0]["id"]})
        text = self.script.get("answer") or "Here is what I found."
        message = {"role": "assistant", "content": [{"type": "text", "text": text}]}
        if sources:
            message["citations"] = [{"start": 0, "end": len(text), "text": text, "type": "TEXT_CONTENT",
                                     "sources": sources}]
        return _chat_response({"id": "scripted", "finish_reason": "COMPLETE", "message": message})


@contextmanager
def no_network():
    """Fail loudly if anything tries to open a connection."""
    def blocked(*args, **kwargs):
        raise RuntimeError("network access attempted during the tool-selection evaluation")

    saved = socket.getaddrinfo, socket.create_connection
    socket.getaddrinfo = socket.create_connection = blocked
    try:
        yield
    finally:
        socket.getaddrinfo, socket.create_connection = saved


def safety_violations(case: dict, executed: list[tuple[str, str]]) -> list[str]:
    """Invariants that must hold whatever the model does: a status that
    reveals order data only when the customer wrote both the order number
    and an email. (The tools check this; this re-checks it independently.)"""
    texts = customer_messages(case)
    wrote_order = any(_ORDER_NUMBER.search(t) for t in texts)
    wrote_email = any(_EMAIL.search(t) for t in texts)
    return [f"{name} returned {status} without the customer's details"
            for name, status in executed if status in DATA_STATUSES and not (wrote_order and wrote_email)]


def run_case(case: dict, book) -> dict:
    """Run one case through the real agent with its scripted decision."""
    client = ScriptedClient(case["script"])
    rounds = []

    def capturing_tool_generator(*args, **kwargs):
        rounds.append(answer.generate_answer_with_tools(*args, **kwargs))
        return rounds[-1]

    trace = {}
    result = agent.answer_question(
        case["question"], history=case.get("history"), trace=trace, retriever=retrieve, client=client,
        rerank_config=RerankConfig(), rewriter=lambda question, history, client: Rewrite(question, NOT_NEEDED),
        tool_generator=capturing_tool_generator, order_book=book, today=TODAY,
    )
    executed = [(r.name, r.status) for r in rounds[0].results] if rounds else []
    observed = {"offered": bool(trace.get("tools_offered")), "reason": result.reason,
                "statuses": [status for _, status in executed]}
    expected = case["expected_outcome"]
    return {
        "id": case["id"], "category": case["category"], "scenario": case["scenario"],
        "observed": observed, "executed_tools": [name for name, _ in executed],
        "passed": observed == expected,
        "violations": safety_violations(case, executed),
        "chat_requests": len(client.requests),
        "second_request_tool_choice": client.requests[1]["tool_choice"] if len(client.requests) > 1 else None,
    }


def run_orchestration(cases: list[dict]) -> dict:
    book = load_orders()
    with no_network():
        results = [run_case(case, book) for case in cases]
    by_id = {c["id"]: c for c in cases}

    def group(predicate):
        chosen = [r for r in results if predicate(by_id[r["id"]], r)]
        return {"cases": len(chosen), "passed": sum(r["passed"] for r in chosen)}

    invalid = [r for r in results if r["category"] == "invalid_model_output"
               or any(s in (tools.UNKNOWN_TOOL, tools.INVALID_ARGUMENTS) for s in r["observed"]["statuses"])]
    missing_details = [r for r in results if r["category"] == "missing_details"
                       or tools.DETAILS_NOT_PROVIDED in r["observed"]["statuses"]]
    return {
        "cases": len(results),
        "passed": sum(r["passed"] for r in results),
        "faithful": group(lambda c, r: c["scenario"] == "faithful"),
        "misbehaving": group(lambda c, r: c["scenario"] == "misbehaving"),
        "invalid_calls": {
            "cases": len(invalid),
            "handled_safely": sum(r["observed"]["reason"] in SAFE_INVALID_CALL_REASONS
                                  or tools.LIMIT_EXCEEDED in r["observed"]["statuses"] for r in invalid),
        },
        "missing_details": {
            "cases": len(missing_details),
            "handled_safely": sum(r["observed"]["reason"] in SAFE_MISSING_DETAILS_REASONS and not r["violations"]
                                  for r in missing_details),
        },
        "safety_violations": sum(len(r["violations"]) for r in results),
        "second_requests_without_tool_choice_none": sum(
            1 for r in results if r["chat_requests"] > 1 and r["second_request_tool_choice"] != "NONE"),
        "per_case": results,
    }


# --- Recorded selections -----------------------------------------------------------------


def load_selections(path: Path, cases: list[dict]) -> tuple[dict, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("selections"), dict):
        raise DatasetError("a selections file needs version 1 and a 'selections' object")
    unknown = set(data["selections"]) - {c["id"] for c in cases}
    if unknown:
        raise DatasetError(f"selections for unknown case(s), e.g. {sorted(unknown)[0]}")
    return data["selections"], str(data.get("source", "unspecified"))


# --- Report ------------------------------------------------------------------------------


def evaluate(cases: list[dict]) -> dict:
    return {
        "dataset_sha256": sha256_json(cases),
        "today": TODAY.isoformat(),
        "offer_rule": offer_metrics(cases),
        "baseline": score_selections(cases, {c["id"]: baseline_select(c) for c in cases}),
        "orchestration": run_orchestration(cases),
    }


def _short(name: str) -> str:
    return {"get_order_status": "status", "check_refund_eligibility": "refund",
            "request_order_cancellation": "cancel"}.get(name, name)


def print_selection_report(title: str, metrics: dict) -> None:
    b = metrics["buckets"]
    print(f"\n{title}: {metrics['cases']} cases, accuracy {metrics['accuracy']:.1%}")
    print(f"  correct tool {b['correct_tool']} | correct no-tool {b['correct_no_tool']} | "
          f"incorrect tool {b['incorrect_tool']} | missing tool {b['missing_tool']} | "
          f"unnecessary tool {b['unnecessary_tool']}")
    labels = sorted({row["expected"] for row in metrics["confusion"]} | {row["selected"] for row in metrics["confusion"]})
    names = {lab: "+".join(_short(part) for part in lab.split("+")) for lab in labels}
    counts = {(row["expected"], row["selected"]): row["count"] for row in metrics["confusion"]}
    width = max(len(n) for n in names.values()) + 2
    print("  confusion (rows: expected, columns: selected)")
    print("  " + " " * width + "".join(names[lab].rjust(width) for lab in labels))
    for expected in labels:
        if any(counts.get((expected, s)) for s in labels):
            print("  " + names[expected].ljust(width) + "".join(str(counts.get((expected, s), "")).rjust(width)
                                                             for s in labels))
    wrong = [row for row in metrics["per_case"] if row["bucket"] not in ("correct_tool", "correct_no_tool")]
    for row in wrong:
        print(f"  {row['id']} {row['bucket']}: expected {row['expected']}, selected {row['selected']}")


def print_report(results: dict) -> None:
    offer = results["offer_rule"]
    print("Tool-selection evaluation (offline; no Cohere calls)")
    print(f"dataset sha256 {results['dataset_sha256'][:12]}, fixed date {results['today']}")
    print("\n1. Offer rule (production: tools offered only when an order number is written)")
    print(f"  offered & tool needed {offer['offered_and_needed']} | offered, no tool needed "
          f"{offer['offered_not_needed']} | not offered, tool needed {offer['not_offered_but_needed']} | "
          f"not offered, none needed {offer['not_offered_not_needed']}")
    print_selection_report("2. Keyword BASELINE (eval-only rules, not the model)", results["baseline"])
    o = results["orchestration"]
    print("\n3. Orchestration with SCRIPTED model decisions (tests the pipeline, not model accuracy)")
    print(f"  expected outcome reproduced: {o['passed']}/{o['cases']} "
          f"(faithful {o['faithful']['passed']}/{o['faithful']['cases']}, "
          f"misbehaving {o['misbehaving']['passed']}/{o['misbehaving']['cases']})")
    print(f"  invalid tool calls handled safely: {o['invalid_calls']['handled_safely']}/{o['invalid_calls']['cases']}")
    print(f"  missing customer details handled safely: "
          f"{o['missing_details']['handled_safely']}/{o['missing_details']['cases']}")
    print(f"  safety violations (order data without both details): {o['safety_violations']}")
    print(f"  second requests without tool_choice NONE: {o['second_requests_without_tool_choice_none']}")
    for row in o["per_case"]:
        if not row["passed"]:
            print(f"  {row['id']} FAILED: observed {row['observed']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="also write the results as JSON to a NEW file")
    parser.add_argument("--selections", type=Path, help="score recorded selections from FILE (offline)")
    args = parser.parse_args(argv)

    cases = load_cases()
    # The pipeline logs every tool status and rejected call; the report shows
    # them already. Quiet the app's logger for this run only, then restore it.
    package_logger = logging.getLogger("support_agent")
    saved_level = package_logger.level
    package_logger.setLevel(logging.ERROR)
    try:
        results = evaluate(cases)
    finally:
        package_logger.setLevel(saved_level)
    if args.selections:
        selections, source = load_selections(args.selections, cases)
        results["recorded"] = {"source": source, **score_selections(cases, selections)}
    print_report(results)
    if "recorded" in results:
        print_selection_report(f"Recorded selections ({results['recorded']['source']})", results["recorded"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "x", encoding="utf-8") as handle:  # Never overwrites an earlier run.
            json.dump(results, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"\nWrote {args.output}")
    o = results["orchestration"]
    return 0 if o["passed"] == o["cases"] and o["safety_violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
