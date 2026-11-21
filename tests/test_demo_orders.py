"""Offline tests for the fictional demo orders (data/demo_orders.json, demo_orders.py)."""

import copy
import json
import socket
import sys
from datetime import date

import pytest

from support_agent import demo_orders
from support_agent.demo_orders import DemoDataError, load_orders

RAW = json.loads(demo_orders.DEMO_ORDERS_PATH.read_text(encoding="utf-8"))
BOOK = load_orders()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network and no Cohere SDK: these modules must need neither."""
    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setitem(sys.modules, "cohere", None)


def raw_order(order_id):
    return next(o for o in RAW["orders"] if o["order_id"] == order_id)


# --- Loading -----------------------------------------------------------------------


def test_all_demo_orders_load():
    assert len(BOOK) == len(RAW["orders"]) == 12
    for raw in RAW["orders"]:
        order = BOOK.find(raw["order_id"], raw["email"])
        assert order is not None and order.order_id == raw["order_id"]
        assert [i.name for i in order.items] == [i["name"] for i in raw["items"]]


def test_every_status_and_scenario_is_covered():
    statuses = {o["status"] for o in RAW["orders"]}
    assert statuses == set(demo_orders.STATUSES)
    items = [i for o in RAW["orders"] for i in o["items"]]
    assert {i["category"] for i in items} == set(demo_orders.CATEGORIES)
    assert any(i["final_sale"] for i in items) and any(i["personalized"] for i in items)
    assert any(i["category"] == "beauty" and i["opened"] for i in items)
    assert {o["shipping"]["country"] for o in RAW["orders"]} == {"US", "CA"}
    # At least one customer has two orders (for cross-order checks).
    emails = [o["email"] for o in RAW["orders"]]
    assert len(set(emails)) < len(emails)


def test_data_is_clearly_fictional():
    assert "fictional" in RAW["notice"].lower()
    assert all(o["email"].endswith("@example.com") for o in RAW["orders"])


def test_dates_are_relative_and_never_stale():
    order = BOOK.find("ORD-1005", "maya.fernandes@example.com")
    for today in (date(2026, 11, 20), date(2031, 2, 3)):
        assert (today - order.order_date(today)).days == 10
        assert (today - order.delivery_date(today)).days == 5
    assert BOOK.find("ORD-1003", "priya.nair@example.com").delivery_date(date(2026, 1, 1)) is None


# --- Validation --------------------------------------------------------------------


def broken(change):
    data = copy.deepcopy(RAW)
    change(data)
    return data


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(version=2), "version"),
    (lambda d: d.pop("notice"), "missing field"),
    (lambda d: d.update(extra=1), "unexpected field"),
    (lambda d: d.update(orders=[]), "non-empty"),
    (lambda d: d["orders"][0].pop("email"), "missing field"),
    (lambda d: d["orders"][0].update(secret="x"), "unexpected field"),
    (lambda d: d["orders"][0].update(order_id="1001"), "ORD-1001"),
    (lambda d: d["orders"][0].update(order_id="ord-1001"), "ORD-1001"),
    (lambda d: d["orders"][1].update(order_id="ORD-1001"), "duplicate order_id"),
    (lambda d: d["orders"][0].update(email="Maya@Example.com"), "lowercase email"),
    (lambda d: d["orders"][0].update(email="maya@gmail.com"), "example.com"),
    (lambda d: d["orders"][0].update(status="lost"), "status"),
    (lambda d: d["orders"][0].update(ordered_days_ago=-1), "whole number"),
    (lambda d: d["orders"][0].update(ordered_days_ago=1.5), "whole number"),
    (lambda d: d["orders"][0].update(ordered_days_ago=True), "whole number"),
    (lambda d: d["orders"][0].update(items=[]), "non-empty list"),
    (lambda d: d["orders"][0]["items"][0].pop("final_sale"), "missing field"),
    (lambda d: d["orders"][0]["items"][0].update(category="food"), "category"),
    (lambda d: d["orders"][0]["items"][0].update(quantity=0), "at least 1"),
    (lambda d: d["orders"][0]["items"][0].update(opened="no"), "true or false"),
    (lambda d: d["orders"][0]["items"][0].update(name=" Jacket"), "surrounding spaces"),
    (lambda d: d["orders"][0]["items"][1].update(name="quilted rain jacket"), "unique"),
    (lambda d: d["orders"][0]["shipping"].update(country="UK"), "country"),
    (lambda d: d["orders"][0]["shipping"].update(method="drone"), "method"),
    # Status consistency.
    (lambda d: d["orders"][0].update(delivered_days_ago=0), "only a delivered order"),
    (lambda d: d["orders"][0]["shipping"].update(carrier="X"), "no shipping details"),
    (lambda d: d["orders"][2]["shipping"].update(carrier=None), "shipped order needs"),
    (lambda d: d["orders"][4].update(delivered_days_ago=None), "delivered order needs"),
    (lambda d: d["orders"][4].update(delivered_days_ago=11), "no earlier than the order"),
    (lambda d: d["orders"][4]["shipping"].update(tracking_status="In transit"), "tracking_status Delivered"),
])
def test_invalid_data_is_rejected(change, message):
    with pytest.raises(DemoDataError, match=message):
        demo_orders.validate(broken(change))


def test_missing_or_malformed_file_is_rejected(tmp_path):
    with pytest.raises(DemoDataError, match="could not be read"):
        load_orders(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DemoDataError, match="not valid JSON"):
        load_orders(bad)


def test_validation_errors_never_contain_customer_emails():
    with pytest.raises(DemoDataError) as raised:
        demo_orders.validate(broken(lambda d: d["orders"][0].update(email="maya@gmail.com")))
    assert "maya" not in str(raised.value)


# --- Lookup ------------------------------------------------------------------------


def test_lookup_needs_the_matching_email():
    assert BOOK.find("ORD-1003", "priya.nair@example.com").status == "shipped"


@pytest.mark.parametrize("order_id, email", [
    (" ord-1003 ", "  Priya.Nair@EXAMPLE.com\n"),
    ("ORD-1003", "PRIYA.NAIR@example.com"),
])
def test_lookup_ignores_case_and_surrounding_spaces(order_id, email):
    assert BOOK.find(order_id, email).order_id == "ORD-1003"


@pytest.mark.parametrize("order_id, email", [
    ("ORD-1003", "maya.fernandes@example.com"),  # another real customer's email
    ("ORD-1003", "priya.nair@example.co"),  # near miss
    ("ORD-1003", ""),
    ("ORD-9999", "priya.nair@example.com"),  # unknown order
    ("ORD-1003x", "priya.nair@example.com"),
    ("", ""),
    (None, "priya.nair@example.com"),
    ("ORD-1003", None),
    (1003, "priya.nair@example.com"),
    ("ORD-1003", ["priya.nair@example.com"]),
])
def test_wrong_email_unknown_or_malformed_input_reveals_nothing(order_id, email):
    """Every failure is the same None: no way to learn whether an order exists."""
    assert BOOK.find(order_id, email) is None


def test_a_customer_sees_only_their_own_orders():
    """Maya owns ORD-1001 and ORD-1005; her email opens no one else's order."""
    maya = "maya.fernandes@example.com"
    opened = {raw["order_id"] for raw in RAW["orders"] if BOOK.find(raw["order_id"], maya)}
    assert opened == {"ORD-1001", "ORD-1005"}


def test_order_item_lookup():
    order = BOOK.find("ORD-1010", "kenji.sato@example.com")
    assert order.item("  hydrating   FACE cream ").opened is True
    assert order.item("Missing thing") is None


def test_lookup_is_deterministic():
    assert all(BOOK.find("ORD-1005", "maya.fernandes@example.com") == load_orders().find(
        "ORD-1005", "maya.fernandes@example.com") for _ in range(3))
