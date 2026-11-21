"""Offline tests for the deterministic refund policy (refund_policy.py)."""

import re
import socket
import sys
from dataclasses import replace

import pytest

from support_agent import refund_policy as rp
from support_agent.demo_orders import Item, load_orders
from support_agent.kb import load_articles
from support_agent.refund_policy import (
    ELIGIBLE,
    NEEDS_REVIEW,
    NOT_APPLICABLE,
    NOT_ELIGIBLE,
    check_refund,
    decide_item,
)

BOOK = load_orders()
KB = {a["id"]: a for a in load_articles()}

EMAILS = {
    "ORD-1001": "maya.fernandes@example.com", "ORD-1002": "jonas.weber@example.com",
    "ORD-1003": "priya.nair@example.com", "ORD-1004": "luc.tremblay@example.com",
    "ORD-1005": "maya.fernandes@example.com", "ORD-1006": "sam.okafor@example.com",
    "ORD-1007": "sam.okafor@example.com", "ORD-1008": "elena.rossi@example.com",
    "ORD-1009": "elena.rossi@example.com", "ORD-1010": "kenji.sato@example.com",
    "ORD-1011": "jonas.weber@example.com", "ORD-1012": "amara.diallo@example.com",
}


def order(order_id):
    return BOOK.find(order_id, EMAILS[order_id])


PLAIN = Item("Plain Tee", "apparel", 1, opened=False, final_sale=False, personalized=False)


def delivered(days_since_order, days_since_delivery, item=PLAIN):
    """A delivered order with chosen ages, built from a real demo order."""
    return replace(order("ORD-1005"), ordered_days_ago=days_since_order,
                   delivered_days_ago=days_since_delivery, items=(item,))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setitem(sys.modules, "cohere", None)


# --- The demo scenarios ----------------------------------------------------------------


@pytest.mark.parametrize("order_id, item, reason, decision, rule", [
    ("ORD-1001", None, "standard", NOT_APPLICABLE, "not_shipped_cancel_instead"),
    ("ORD-1003", None, "damaged", NOT_APPLICABLE, "not_delivered_yet"),
    ("ORD-1011", None, "standard", NOT_APPLICABLE, "order_cancelled"),
    ("ORD-1005", None, "standard", ELIGIBLE, "within_return_window"),  # 10 days
    ("ORD-1006", None, "standard", ELIGIBLE, "within_return_window"),  # exactly 30 days
    ("ORD-1007", None, "standard", NEEDS_REVIEW, "outside_return_window"),  # 31 days
    ("ORD-1008", None, "damaged", ELIGIBLE, "problem_reported_in_time"),  # delivered exactly 7 days ago
    ("ORD-1009", None, "damaged", ELIGIBLE, "within_return_window_free_return_shipping"),  # 8 days, order 11
    ("ORD-1010", "$50 Gift Card", "standard", NOT_ELIGIBLE, "excluded_item"),
    ("ORD-1010", "Photo Editing Course (Download)", "standard", NOT_ELIGIBLE, "excluded_item"),
    ("ORD-1010", "Engraved Silver Bracelet", "standard", NOT_ELIGIBLE, "excluded_item"),
    ("ORD-1010", "Hydrating Face Cream", "standard", NOT_ELIGIBLE, "excluded_item"),  # opened beauty
    ("ORD-1010", "Tinted Lip Balm", "standard", ELIGIBLE, "within_return_window"),  # unopened beauty
    ("ORD-1010", "Striped Summer Dress", "standard", NOT_ELIGIBLE, "excluded_item"),  # final sale
    ("ORD-1010", "Hydrating Face Cream", "damaged", ELIGIBLE, "problem_reported_in_time"),  # KB-016
    ("ORD-1012", "Cast Iron Skillet", "standard", NEEDS_REVIEW, "outside_return_window"),
    ("ORD-1012", "Clearance Oven Mitts", "defective", NOT_ELIGIBLE, "excluded_item_problem_reported_late"),
    ("ORD-1012", "Cast Iron Skillet", "missing_item", NEEDS_REVIEW, "problem_reported_late"),
])
def test_demo_scenarios(order_id, item, reason, decision, rule):
    result = check_refund(order(order_id), reason, item)
    assert {(d.decision, d.rule) for d in result.items} == {(decision, rule)}


def test_whole_order_check_decides_each_item():
    result = check_refund(order("ORD-1010"))
    assert {d.item_name: d.decision for d in result.items} == {
        "$50 Gift Card": NOT_ELIGIBLE, "Photo Editing Course (Download)": NOT_ELIGIBLE,
        "Engraved Silver Bracelet": NOT_ELIGIBLE, "Hydrating Face Cream": NOT_ELIGIBLE,
        "Tinted Lip Balm": ELIGIBLE, "Striped Summer Dress": NOT_ELIGIBLE,
    }
    assert {d.item_name: d.exclusion for d in result.items if d.exclusion} == {
        "$50 Gift Card": "gift_card", "Photo Editing Course (Download)": "digital_download",
        "Engraved Silver Bracelet": "personalized", "Hydrating Face Cream": "opened_beauty",
        "Striped Summer Dress": "final_sale",
    }
    assert (result.order_id, result.status, result.days_since_order, result.days_since_delivery) == (
        "ORD-1010", "delivered", 6, 2)


# --- Boundaries ------------------------------------------------------------------------


@pytest.mark.parametrize("days, decision", [(0, ELIGIBLE), (29, ELIGIBLE), (30, ELIGIBLE), (31, NEEDS_REVIEW), (90, NEEDS_REVIEW)])
def test_return_window_boundary(days, decision):
    assert decide_item(delivered(days, 0), PLAIN).decision == decision


@pytest.mark.parametrize("reason", rp.PROBLEM_REASONS)
@pytest.mark.parametrize("days, rule", [
    (0, "problem_reported_in_time"), (7, "problem_reported_in_time"),
    (8, "within_return_window_free_return_shipping"),
])
def test_problem_report_window_boundary(reason, days, rule):
    result = decide_item(delivered(20, days), PLAIN, reason)
    if reason == "missing_item" and days > 7:
        assert (result.decision, result.rule) == (NEEDS_REVIEW, "problem_reported_late")
    else:
        assert (result.decision, result.rule) == (ELIGIBLE, rule)


def test_late_problem_report_after_the_return_window_needs_review():
    assert decide_item(delivered(31, 8), PLAIN, "damaged").rule == "outside_return_window"
    assert decide_item(delivered(31, 8), PLAIN, "damaged").decision == NEEDS_REVIEW


def test_problem_reported_in_time_is_eligible_even_long_after_ordering():
    """KB-028 counts from delivery, not from the order."""
    assert decide_item(delivered(60, 7), PLAIN, "wrong_item").decision == ELIGIBLE


@pytest.mark.parametrize("item", [
    replace(PLAIN, category="gift_card"),
    replace(PLAIN, category="digital"),
    replace(PLAIN, personalized=True),
    replace(PLAIN, category="beauty", opened=True),
    replace(PLAIN, final_sale=True),
])
def test_every_exclusion_at_both_windows(item):
    assert decide_item(delivered(1, 0, item), item).decision == NOT_ELIGIBLE  # even on day one
    assert decide_item(delivered(10, 7, item), item, "damaged").decision == ELIGIBLE  # KB-016 cover
    late = decide_item(delivered(10, 8, item), item, "damaged")
    assert (late.decision, late.rule) == (NOT_ELIGIBLE, "excluded_item_problem_reported_late")


def test_unopened_beauty_is_not_excluded():
    item = replace(PLAIN, category="beauty", opened=False)
    assert rp.exclusion_for(item) is None


def test_eligible_standard_returns_carry_the_kb_015_condition():
    assert decide_item(delivered(5, 1), PLAIN).conditions == (rp.STANDARD_RETURN_CONDITION,)


@pytest.mark.parametrize("status, rule", [
    ("processing", "not_shipped_cancel_instead"), ("shipped", "not_delivered_yet"), ("cancelled", "order_cancelled"),
])
@pytest.mark.parametrize("reason", rp.REASONS)
def test_undelivered_or_cancelled_orders_are_never_a_return(status, rule, reason):
    o = replace(order("ORD-1005"), status=status, delivered_days_ago=None)
    assert (decide_item(o, PLAIN, reason).decision, decide_item(o, PLAIN, reason).rule) == (NOT_APPLICABLE, rule)


def test_unknown_reason_and_item_are_rejected():
    with pytest.raises(ValueError, match="reason"):
        check_refund(order("ORD-1005"), "changed_my_mind")
    with pytest.raises(ValueError, match="reason"):
        decide_item(order("ORD-1005"), PLAIN, "")
    with pytest.raises(rp.UnknownItemError):
        check_refund(order("ORD-1005"), "standard", "Gold Watch")


# --- Consistency with the knowledge base ------------------------------------------------


def _numbers(article_id, pattern):
    return {int(n) for n in re.findall(pattern, KB[article_id]["text"])}


@pytest.mark.parametrize("article_id", ["KB-002", "KB-015", "KB-024"])
def test_return_window_matches_the_kb(article_id):
    assert _numbers(article_id, r"within (\d+) days of purchase") == {rp.RETURN_WINDOW_DAYS}


@pytest.mark.parametrize("article_id", ["KB-016", "KB-028"])
def test_problem_report_window_matches_the_kb(article_id):
    assert _numbers(article_id, r"within (\d+) days of delivery") == {rp.PROBLEM_REPORT_WINDOW_DAYS}


@pytest.mark.parametrize("key, phrase", rp.EXCLUSIONS.items())
def test_every_exclusion_is_stated_in_kb_016(key, phrase):
    assert phrase.lower() in KB["KB-016"]["text"].lower()


def test_kb_016_lists_no_exclusion_the_code_misses():
    listed = KB["KB-016"]["text"].split("cannot be returned or refunded:", 1)[1].split(".", 1)[0]
    parts = [p.strip().removeprefix("and ").strip() for p in listed.split(",")]
    covered = [p for p in parts if any(phrase.lower() in p.lower() for phrase in rp.EXCLUSIONS.values())]
    assert covered == parts, f"KB-016 lists exclusions the code does not handle: {set(parts) - set(covered)}"


def test_kb_states_the_other_rules_the_code_relies_on():
    assert "require review" in KB["KB-002"]["text"] and "reviewed" in KB["KB-015"]["text"]
    assert "order date" in KB["KB-002"]["text"]  # the 30 days count from the order date
    assert "only while its status is Processing" in KB["KB-024"]["text"]
    assert "refunded in full" in KB["KB-024"]["text"]
    assert "Return shipping is free if the item arrived damaged, defective or was the wrong item" in KB["KB-015"]["text"]
    assert "unused, in their original packaging" in KB["KB-015"]["text"]


def test_every_cited_article_exists_and_matches_its_rule():
    decisions = [
        decide_item(o, item, reason)
        for oid in EMAILS for o in [order(oid)] for item in o.items for reason in rp.REASONS
    ]
    cited = {a for d in decisions for a in d.article_ids}
    assert cited <= set(KB)
    assert cited == {"KB-002", "KB-015", "KB-016", "KB-024", "KB-028"}
    assert all(d.article_ids and d.decision in rp.DECISIONS for d in decisions)


# --- Determinism -----------------------------------------------------------------------


def test_results_are_deterministic():
    first = [check_refund(order(oid), reason) for oid in EMAILS for reason in rp.REASONS]
    again = [check_refund(load_orders().find(oid, EMAILS[oid]), reason) for oid in EMAILS for reason in rp.REASONS]
    assert first == again


def test_policy_does_not_read_the_clock(monkeypatch):
    import datetime as real_datetime

    class Exploding(real_datetime.date):
        @classmethod
        def today(cls):
            raise AssertionError("refund policy read the clock")

    monkeypatch.setattr(real_datetime, "date", Exploding)
    check_refund(order("ORD-1005"))
