"""Refund eligibility for the demo orders, decided by Python, never by a model.

Every rule comes from an existing knowledge-base article; nothing here adds
policy. Decisions depend only on the order and the reason given (no clock,
no randomness, no network), so the same input always gives the same answer.

Policy sources:
    KB-002  Refund eligibility: within 30 days of purchase, counted from the
            order date; later requests are reviewed by a representative;
            gift cards and digital downloads can't be refunded.
    KB-015  Return an item: most items within 30 days of purchase, unused, in
            original packaging with tags; later returns are reviewed; free
            return shipping for damaged, defective or wrong items.
    KB-016  Items that cannot be returned: gift cards, digital downloads,
            personalized or engraved items, opened personal care and beauty
            products, Final sale items. If one arrives damaged, defective or
            wrong, it is still covered when reported within 7 days of delivery.
    KB-024  Cancel an order: only while Processing; cancelled items are
            refunded in full automatically; a shipped order is returned after
            delivery instead.
    KB-028  Damaged, wrong or missing items: report within 7 days of delivery.

Decisions:
    eligible        the policy allows a refund (conditions may apply)
    not_eligible    the policy excludes it
    needs_review    outside the normal rules: a support representative decides
    not_applicable  no refund request applies yet (not delivered) or any more
                    (cancelled orders are refunded automatically)
"""

from dataclasses import dataclass

from support_agent.demo_orders import Item, Order

# Policy numbers, as stated in the articles named (tests check the KB text).
RETURN_WINDOW_DAYS = 30  # "within 30 days of purchase" (KB-002, KB-015, KB-024)
PROBLEM_REPORT_WINDOW_DAYS = 7  # "within 7 days of delivery" (KB-016, KB-028)

ELIGIBLE = "eligible"
NOT_ELIGIBLE = "not_eligible"
NEEDS_REVIEW = "needs_review"
NOT_APPLICABLE = "not_applicable"
DECISIONS = (ELIGIBLE, NOT_ELIGIBLE, NEEDS_REVIEW, NOT_APPLICABLE)

STANDARD = "standard"  # The customer doesn't want the item (no problem with it).
PROBLEM_REASONS = ("damaged", "defective", "wrong_item", "missing_item")  # KB-028
REASONS = (STANDARD,) + PROBLEM_REASONS

# Why an item can't be returned, with the KB-016 wording each comes from.
EXCLUSIONS = {
    "gift_card": "gift cards",
    "digital_download": "digital downloads",
    "personalized": "personalized or engraved items",
    "opened_beauty": "opened personal care and beauty products",
    "final_sale": "Final sale",
}

# The condition KB-015 sets on ordinary returns (shown, not checked: the
# system can't know whether an item is unused).
STANDARD_RETURN_CONDITION = "unused, in its original packaging, with any tags attached"


class UnknownItemError(LookupError):
    """The order has no item with that name."""


@dataclass(frozen=True)
class RefundDecision:
    item_name: str
    decision: str
    rule: str  # A stable identifier, e.g. "within_return_window".
    article_ids: tuple[str, ...]  # The KB articles that state the rule.
    exclusion: str | None = None  # Key into EXCLUSIONS, when one applies.
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RefundCheck:
    order_id: str
    status: str
    reason: str
    days_since_order: int
    days_since_delivery: int | None
    items: tuple[RefundDecision, ...]


def exclusion_for(item: Item) -> str | None:
    """The KB-016 exclusion that applies to this item, or None."""
    if item.category == "gift_card":
        return "gift_card"
    if item.category == "digital":
        return "digital_download"
    if item.personalized:
        return "personalized"
    if item.category == "beauty" and item.opened:
        return "opened_beauty"
    if item.final_sale:
        return "final_sale"
    return None


def decide_item(order: Order, item: Item, reason: str = STANDARD) -> RefundDecision:
    """Apply the policy to one item of one order."""
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {', '.join(REASONS)}")
    name = item.name

    # Not a return at all: the order was cancelled, or hasn't arrived yet (KB-024).
    if order.status == "cancelled":
        return RefundDecision(name, NOT_APPLICABLE, "order_cancelled", ("KB-024",))
    if order.status == "processing":
        return RefundDecision(name, NOT_APPLICABLE, "not_shipped_cancel_instead", ("KB-024",))
    if order.status == "shipped":
        return RefundDecision(name, NOT_APPLICABLE, "not_delivered_yet", ("KB-024", "KB-015"))

    exclusion = exclusion_for(item)
    days_since_order = order.ordered_days_ago
    days_since_delivery = order.delivered_days_ago

    if reason in PROBLEM_REASONS:
        if days_since_delivery <= PROBLEM_REPORT_WINDOW_DAYS:
            # KB-028; KB-016: covered even for items that can't otherwise be returned.
            articles = ("KB-028", "KB-016") if exclusion else ("KB-028",)
            return RefundDecision(name, ELIGIBLE, "problem_reported_in_time", articles, exclusion)
        if exclusion:
            # KB-016 covers excluded items only when reported within 7 days.
            return RefundDecision(name, NOT_ELIGIBLE, "excluded_item_problem_reported_late",
                                  ("KB-016", "KB-028"), exclusion)
        if reason == "missing_item":
            # No article says what happens after 7 days; a person decides.
            return RefundDecision(name, NEEDS_REVIEW, "problem_reported_late", ("KB-028",))
        # A damaged, defective or wrong item can still be returned within the
        # 30-day window, with free return shipping (KB-015).
        if days_since_order <= RETURN_WINDOW_DAYS:
            return RefundDecision(name, ELIGIBLE, "within_return_window_free_return_shipping",
                                  ("KB-015", "KB-028"))
        return RefundDecision(name, NEEDS_REVIEW, "outside_return_window", ("KB-002", "KB-015"))

    if exclusion:
        # KB-002 repeats the gift card and digital download exclusions.
        articles = ("KB-016", "KB-002") if exclusion in ("gift_card", "digital_download") else ("KB-016",)
        return RefundDecision(name, NOT_ELIGIBLE, "excluded_item", articles, exclusion)
    if days_since_order <= RETURN_WINDOW_DAYS:
        return RefundDecision(name, ELIGIBLE, "within_return_window", ("KB-002", "KB-015"),
                              conditions=(STANDARD_RETURN_CONDITION,))
    return RefundDecision(name, NEEDS_REVIEW, "outside_return_window", ("KB-002", "KB-015"))


def check_refund(order: Order, reason: str = STANDARD, item_name: str | None = None) -> RefundCheck:
    """Decide eligibility for one named item, or for every item in the order.

    Raises ValueError for an unknown reason and UnknownItemError if
    `item_name` is not in the order.
    """
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {', '.join(REASONS)}")
    if item_name is None:
        items = order.items
    else:
        item = order.item(item_name)
        if item is None:
            raise UnknownItemError("The order has no item with that name")
        items = (item,)
    return RefundCheck(
        order_id=order.order_id,
        status=order.status,
        reason=reason,
        days_since_order=order.ordered_days_ago,
        days_since_delivery=order.delivered_days_ago,
        items=tuple(decide_item(order, item, reason) for item in items),
    )
