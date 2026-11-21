"""Offline tests for the tool registry (tools.py). Real demo data, a fixed
date, no network and no Cohere client."""

import copy
import json
import logging
import socket
import sys
from datetime import date

import pytest

from support_agent import demo_orders, refund_policy, tools
from support_agent.demo_orders import load_orders
from support_agent.tools import ToolCall, ToolContext, execute, execute_all

TODAY = date(2026, 11, 20)
BOOK = load_orders()
MAYA = "maya.fernandes@example.com"
KENJI = "kenji.sato@example.com"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)


def context(*customer_messages, history=None):
    if history is not None:
        return ToolContext.for_question(BOOK, customer_messages[0], history, TODAY)
    return ToolContext(orders=BOOK, customer_messages=tuple(customer_messages), today=TODAY)


def call(name, call_id="call-1", **arguments):
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def said(order_id, email):
    """A customer message containing both details."""
    return f"My order is {order_id} and my email is {email}."


def run(name, order_id, email, **extra):
    return execute(call(name, order_id=order_id, email=email, **extra), context(said(order_id, email)))


# --- Definitions -------------------------------------------------------------------


def test_definitions_come_from_the_registry():
    defs = tools.definitions()
    assert [d["function"]["name"] for d in defs] == list(tools.REGISTRY) == [
        "get_order_status", "check_refund_eligibility", "request_order_cancellation",
    ]
    for d, spec in zip(defs, tools.REGISTRY.values()):
        params = d["function"]["parameters"]
        assert params["additionalProperties"] is False
        assert params["required"] == list(spec.required)
        assert set(params["properties"]) == set(spec.required) | set(spec.optional)
        assert d["type"] == "function" and d["function"]["description"]


def test_definitions_validate_against_the_installed_sdk_models():
    """Checked with the SDK's own pydantic models; no request is sent."""
    from cohere import ToolV2

    for definition in tools.definitions():
        parsed = ToolV2.model_validate(definition)
        assert parsed.function.name == definition["function"]["name"]
        assert parsed.function.parameters == definition["function"]["parameters"]


def test_schemas_match_the_python_rules():
    refund = tools.REGISTRY["check_refund_eligibility"].properties
    assert refund["reason"]["enum"] == list(refund_policy.REASONS)
    assert tools._ORDER_ID_SCHEMA["pattern"] == f"^{demo_orders.ORDER_ID.pattern}$"
    assert "NOT cancel" in tools.REGISTRY["request_order_cancellation"].description


def test_tool_result_documents_fit_the_sdk_tool_message():
    from cohere import ToolChatMessageV2

    result = run("get_order_status", "ORD-1003", "priya.nair@example.com")
    message = ToolChatMessageV2.model_validate({
        "role": "tool", "tool_call_id": result.call_id,
        "content": [{"type": "document", "document": result.as_document()}],
    })
    document = message.content[0].document
    assert document.id == "tool-1-get_order_status"
    assert all(isinstance(v, str) for v in document.data.values())
    assert json.loads(document.data["items"]) == [{"name": "Wireless Earbuds", "quantity": 1}]


# --- Successful calls --------------------------------------------------------------


def test_get_order_status_shipped():
    result = run("get_order_status", "ORD-1003", "priya.nair@example.com")
    assert result.status == "ok" and result.name == "get_order_status"
    assert result.data == {
        "status": "ok", "order_id": "ORD-1003", "order_status": "shipped",
        "ordered_on": "2026-11-18", "ordered_days_ago": 2, "delivered_on": None,
        "shipping": {"method": "standard", "country": "US", "carrier": "Northwind Parcel",
                     "tracking_status": "In transit", "estimated_delivery_on": "2026-11-23"},
        "items": [{"name": "Wireless Earbuds", "quantity": 1}],
        "article_ids": ["KB-003", "KB-026"],
    }


def test_get_order_status_delivered_dates():
    data = run("get_order_status", "ORD-1005", MAYA).data
    assert (data["ordered_on"], data["delivered_on"], data["order_status"]) == ("2026-11-10", "2026-11-15", "delivered")
    assert data["shipping"]["estimated_delivery_on"] is None


def test_refund_tool_returns_exactly_the_policy_decision():
    order = BOOK.find("ORD-1010", KENJI)
    expected = refund_policy.check_refund(order)
    data = run("check_refund_eligibility", "ORD-1010", KENJI).data
    assert data["decided_by"] == "refund_policy"
    assert [(i["item_name"], i["decision"], i["rule"], i["exclusion"], tuple(i["article_ids"])) for i in data["items"]] == [
        (d.item_name, d.decision, d.rule, d.exclusion, d.article_ids) for d in expected.items
    ]


def test_refund_tool_with_reason_and_item():
    data = run("check_refund_eligibility", "ORD-1010", KENJI, reason="damaged", item_name="hydrating face cream").data
    assert [(i["item_name"], i["decision"], i["rule"]) for i in data["items"]] == [
        ("Hydrating Face Cream", "eligible", "problem_reported_in_time")
    ]
    assert data["reason"] == "damaged"


def test_refund_tool_unknown_item_lists_only_the_owners_items():
    result = run("check_refund_eligibility", "ORD-1005", MAYA, item_name="Gold Watch")
    assert result.status == "unknown_item"
    assert result.data["items"] == [{"name": "Linen Shirt", "quantity": 2}]


# --- Ownership and grounding --------------------------------------------------------


@pytest.mark.parametrize("name", list(tools.REGISTRY))
@pytest.mark.parametrize("order_id, email", [
    ("ORD-1003", MAYA),  # a real customer, but not this order's
    ("ORD-1003", "nobody@example.com"),
    ("ORD-9999", "priya.nair@example.com"),  # unknown order
])
def test_wrong_owner_or_unknown_order_reveals_nothing(name, order_id, email):
    result = run(name, order_id, email)
    assert result.status == "not_found" and result.data == {"status": "not_found"}


@pytest.mark.parametrize("name", list(tools.REGISTRY))
def test_details_must_come_from_the_customer(name):
    args = {"order_id": "ORD-1003", "email": "priya.nair@example.com"}
    cases = {
        ("order_id", "email"): context("Where is my order?"),
        ("email",): context("My order is ORD-1003"),
        ("order_id",): context("My email is priya.nair@example.com"),
    }
    for missing, ctx in cases.items():
        result = execute(call(name, **args), ctx)
        assert result.status == "details_not_provided" and result.data["missing"] == list(missing)


def test_details_in_assistant_messages_do_not_count():
    history = [
        {"role": "customer", "content": "Where is my order?"},
        {"role": "assistant", "content": "Is it ORD-1003 for priya.nair@example.com?"},
    ]
    ctx = context("yes, that one", history=history)
    assert ctx.customer_messages == ("Where is my order?", "yes, that one")
    result = execute(call("get_order_status", order_id="ORD-1003", email="priya.nair@example.com"), ctx)
    assert result.status == "details_not_provided"


def test_details_from_earlier_customer_messages_count():
    history = [
        {"role": "customer", "content": "My order ORD-1003 hasn't arrived"},
        {"role": "assistant", "content": "Which email did you use?"},
    ]
    ctx = context("priya.nair@example.com", history=history)
    assert execute(call("get_order_status", order_id="ORD-1003", email="priya.nair@example.com"), ctx).status == "ok"


@pytest.mark.parametrize("message, grounded", [
    ("order ord-1003, email PRIYA.NAIR@EXAMPLE.COM.", True),  # case and trailing full stop
    ("(ORD-1003) <priya.nair@example.com>", True),
    ("ORD-10031 priya.nair@example.com", False),  # longer number
    ("XORD-1003 priya.nair@example.com", False),
    ("ORD-1003 xpriya.nair@example.com", False),  # longer address
    ("ORD-1003 priya.nair@example.com.au", False),
    ("ORD 1003 priya.nair@example.com", False),  # not written as the order number
])
def test_grounding_matches_whole_tokens_only(message, grounded):
    result = execute(call("get_order_status", order_id="ORD-1003", email="priya.nair@example.com"), context(message))
    assert (result.status == "ok") is grounded


def test_normalised_arguments_still_match_the_order():
    result = execute(call("get_order_status", order_id=" ord-1003 ", email=" Priya.Nair@Example.COM "),
                     context(said("ORD-1003", "priya.nair@example.com")))
    assert result.status == "ok"


# --- Invalid arguments ---------------------------------------------------------------


GOOD = {"order_id": "ORD-1003", "email": "priya.nair@example.com"}


@pytest.mark.parametrize("raw, fields, unexpected", [
    (None, ["arguments"], 0),
    ("", ["arguments"], 0),
    ("not json", ["arguments"], 0),
    ("[]", ["arguments"], 0),
    ('"ORD-1003"', ["arguments"], 0),
    (json.dumps({**GOOD, "order_id": "1003"}), ["order_id"], 0),
    (json.dumps({**GOOD, "order_id": 1003}), ["order_id"], 0),
    (json.dumps({**GOOD, "order_id": None}), ["order_id"], 0),
    (json.dumps({**GOOD, "email": "not-an-email"}), ["email"], 0),
    (json.dumps({**GOOD, "email": "a" * 250 + "@example.com"}), ["email"], 0),
    (json.dumps({**GOOD, "status": "cancelled"}), [], 1),
    (json.dumps({**GOOD, "x": 1, "y": 2}), [], 2),
    ('{"order_id": "ORD-1003", "order_id": "ORD-1001", "email": "priya.nair@example.com"}', ["arguments"], 0),
    ("x" * (tools.MAX_ARGUMENT_CHARS + 1), ["arguments"], 0),
    ("[" * 999 + "]" * 999, ["arguments"], 0),  # deep nesting under the size limit
])
def test_invalid_arguments_run_nothing(raw, fields, unexpected, monkeypatch):
    monkeypatch.setattr(BOOK, "find", lambda *a: pytest.fail("lookup ran"))
    result = execute(ToolCall("c1", "get_order_status", raw), context(said("ORD-1003", "priya.nair@example.com")))
    assert result.status == "invalid_arguments"
    assert result.data == {"status": "invalid_arguments", "invalid_fields": fields, "unexpected_fields": unexpected}


@pytest.mark.parametrize("raw, missing", [
    ("{}", ["email", "order_id"]),
    (json.dumps({"order_id": "ORD-1003"}), ["email"]),
    (json.dumps({"email": "priya.nair@example.com"}), ["order_id"]),
])
def test_left_out_customer_details_ask_for_them_and_run_nothing(raw, missing, monkeypatch):
    """Milestone 5: a well-formed call that leaves out the order number or
    email (the customer hasn't given it) is "details not provided", so the
    customer is asked for them. It used to be invalid_arguments, which handed
    the customer off and opened a service_problem hand-off record."""
    monkeypatch.setattr(BOOK, "find", lambda *a: pytest.fail("lookup ran"))
    result = execute(ToolCall("c1", "get_order_status", raw), context(said("ORD-1003", "priya.nair@example.com")))
    assert result.data == {"status": "details_not_provided", "missing": missing}


@pytest.mark.parametrize("raw, fields, unexpected", [
    # Anything else wrong as well still makes the whole call invalid.
    (json.dumps({"order_id": "1003"}), ["email", "order_id"], 0),
    (json.dumps({"order_id": "ORD-1003", "admin": True}), ["email"], 1),
])
def test_missing_details_with_other_problems_are_invalid(raw, fields, unexpected):
    result = execute(ToolCall("c1", "get_order_status", raw), context(said("ORD-1003", "priya.nair@example.com")))
    assert result.data == {"status": "invalid_arguments", "invalid_fields": fields, "unexpected_fields": unexpected}


@pytest.mark.parametrize("extra, field", [
    ({"reason": "changed_my_mind"}, "reason"),
    ({"reason": ""}, "reason"),
    ({"item_name": ""}, "item_name"),
    ({"item_name": "x" * 101}, "item_name"),
    ({"item_name": ["Linen Shirt"]}, "item_name"),
])
def test_invalid_optional_refund_arguments(extra, field):
    result = run("check_refund_eligibility", "ORD-1005", MAYA, **extra)
    assert result.status == "invalid_arguments" and result.data["invalid_fields"] == [field]


def test_invalid_argument_results_never_echo_values():
    raw = json.dumps({"order_id": "SECRET-ORDER", "email": "secret@evil.test", "SECRET-KEY": "x"})
    result = execute(ToolCall("c1", "get_order_status", raw), context("anything"))
    assert "SECRET" not in json.dumps(result.data) and "evil" not in json.dumps(result.data)


@pytest.mark.parametrize("name", ["cancel_order", "", "GET_ORDER_STATUS", None, 42])
def test_unknown_tools_run_nothing(name, monkeypatch):
    monkeypatch.setattr(BOOK, "find", lambda *a: pytest.fail("lookup ran"))
    result = execute(ToolCall("c1", name, json.dumps(GOOD)), context(said(*GOOD.values())))
    assert (result.status, result.name, result.data) == ("unknown_tool", "unknown", {"status": "unknown_tool"})


# --- Cancellation is read-only ------------------------------------------------------


def test_processing_order_needs_confirmation():
    result = run("request_order_cancellation", "ORD-1001", MAYA)
    assert result.status == "confirmation_required"
    assert result.data["items"] == [{"name": "Quilted Rain Jacket", "quantity": 1}, {"name": "Merino Scarf", "quantity": 1}]
    assert result.data["article_ids"] == ["KB-024"] and "NOT been cancelled" in result.data["note"]


@pytest.mark.parametrize("order_id, email, status", [
    ("ORD-1003", "priya.nair@example.com", "shipped"),
    ("ORD-1005", MAYA, "delivered"),
    ("ORD-1011", "jonas.weber@example.com", "cancelled"),
])
def test_orders_past_processing_are_not_cancellable(order_id, email, status):
    result = run("request_order_cancellation", order_id, email)
    assert result.data == {"status": "not_cancellable", "order_id": order_id, "order_status": status,
                           "article_ids": ["KB-024"]}


def test_cancellation_request_changes_nothing():
    snapshot = copy.deepcopy(BOOK._orders)
    for _ in range(3):
        assert run("request_order_cancellation", "ORD-1001", MAYA).status == "confirmation_required"
    assert BOOK._orders == snapshot
    assert BOOK.find("ORD-1001", MAYA).status == "processing"


def test_no_code_path_can_change_an_order():
    """Orders are frozen, and neither module offers a way to change them."""
    order = BOOK.find("ORD-1001", MAYA)
    with pytest.raises(AttributeError):
        order.status = "cancelled"
    writers = [n for n in dir(tools) + dir(demo_orders.OrderBook)
               if any(w in n.lower() for w in ("cancel_order", "set_status", "update", "save", "write"))]
    assert writers == []


# --- Errors, logging, privacy -------------------------------------------------------


def test_unexpected_errors_are_generic_and_not_logged_in_detail(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("SECRET internal detail at /srv/app.py line 12")

    monkeypatch.setattr(tools, "check_refund", broken)
    caplog.set_level(logging.DEBUG)

    result = run("check_refund_eligibility", "ORD-1005", MAYA)

    assert result.data == {"status": "error"}
    assert "SECRET" not in caplog.text and "/srv" not in caplog.text
    assert "check_refund_eligibility: error (RuntimeError)" in caplog.text


def test_logs_never_contain_arguments(caplog):
    caplog.set_level(logging.DEBUG)
    for name in tools.REGISTRY:
        run(name, "ORD-1005", MAYA)
        run(name, "ORD-1005", "wrong@example.com")
    execute(ToolCall("c", "nope", json.dumps(GOOD)), context("x"))
    execute(ToolCall("c", "get_order_status", '{"order_id": "ORD-1005", "email": 5}'), context("x"))
    assert "ORD-1005" not in caplog.text and "@" not in caplog.text and "nope" not in caplog.text
    assert "Tool get_order_status: ok" in caplog.text


def test_results_never_contain_an_email():
    texts = []
    for raw in json.loads(demo_orders.DEMO_ORDERS_PATH.read_text(encoding="utf-8"))["orders"]:
        for name in tools.REGISTRY:
            texts.append(json.dumps(run(name, raw["order_id"], raw["email"]).data))
            texts.append(json.dumps(run(name, raw["order_id"], raw["email"], **(
                {"reason": "damaged"} if name == "check_refund_eligibility" else {})).data))
    assert texts and not any("@" in text for text in texts)


def test_results_are_deterministic():
    first = [run(n, "ORD-1010", KENJI) for n in tools.REGISTRY]
    again = [run(n, "ORD-1010", KENJI) for n in tools.REGISTRY]
    assert first == again


# --- execute_all and the call limit ------------------------------------------------


def test_three_calls_run_and_later_ones_are_limited(monkeypatch):
    ran = []
    real_find = BOOK.find
    monkeypatch.setattr(BOOK, "find", lambda *a: ran.append(a) or real_find(*a))
    calls = [call("get_order_status", f"id-{n}", order_id="ORD-1005", email=MAYA) for n in range(1, 6)]

    results = execute_all(calls, context(said("ORD-1005", MAYA)))

    assert [r.call_id for r in results] == ["id-1", "id-2", "id-3", "id-4", "id-5"]
    assert [r.status for r in results] == ["ok", "ok", "ok", "limit_exceeded", "limit_exceeded"]
    assert len(ran) == tools.MAX_TOOL_CALLS == 3
    assert [r.document_id for r in results] == [f"tool-{n}-get_order_status" for n in range(1, 6)]


def test_mixed_calls_each_get_their_own_result():
    ctx = context(said("ORD-1001", MAYA), said("ORD-1005", MAYA))
    results = execute_all([
        call("get_order_status", "a", order_id="ORD-1005", email=MAYA),
        ToolCall("b", "delete_everything", "{}"),
        call("request_order_cancellation", "c", order_id="ORD-1001", email=MAYA),
    ], ctx)
    assert [(r.call_id, r.name, r.status) for r in results] == [
        ("a", "get_order_status", "ok"), ("b", "unknown", "unknown_tool"),
        ("c", "request_order_cancellation", "confirmation_required"),
    ]
    assert len({r.document_id for r in results}) == 3


def test_document_ids_never_clash_with_kb_ids():
    import re

    results = execute_all([call(n, str(i), order_id="ORD-1005", email=MAYA) for i, n in enumerate(tools.REGISTRY)],
                          context(said("ORD-1005", MAYA)))
    assert not any(re.fullmatch(r"KB-\d{3}", r.document_id) for r in results)


def test_empty_call_list():
    assert execute_all([], context("hi")) == []




SUBPROCESS_CHECK = """
import datetime, json, sys
sys.modules["cohere"] = None  # any import of the SDK fails
from support_agent import demo_orders, tools
ctx = tools.ToolContext(demo_orders.load_orders(), ("ORD-1005 maya.fernandes@example.com",), datetime.date(2026, 11, 20))
args = json.dumps({"order_id": "ORD-1005", "email": "maya.fernandes@example.com"})
assert tools.execute(tools.ToolCall("c", "get_order_status", args), ctx).status == "ok"
print("OK")
"""


def test_registry_never_imports_cohere():
    """In a fresh interpreter with the SDK blocked, the registry still works."""
    import subprocess

    result = subprocess.run([sys.executable, "-c", SUBPROCESS_CHECK], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and result.stdout.strip() == "OK", result.stderr
