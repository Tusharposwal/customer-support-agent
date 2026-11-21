"""Fictional demo orders (data/demo_orders.json) and how they are looked up.

These orders are made up for the demo: no real customers, orders or carriers,
and no connection to any order-management system. Everything here is plain
Python with no network access and no Cohere calls.

Dates are stored as "days before today" (ordered_days_ago,
delivered_days_ago), so an order delivered "5 days ago" is always 5 days old
and the scenarios never go stale. Calendar dates are computed only for
display, from a `today` the caller passes in.

Looking up an order needs both its order ID and the email address it was
placed with (the same check the Track order page uses for guests, KB-003).
An unknown order, a wrong email and a malformed ID all give the same answer
(None), so a lookup never reveals whether an order ID exists or whose it is.
"""

import hmac
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ORDERS_PATH = PROJECT_ROOT / "data" / "demo_orders.json"
DATA_VERSION = 1

ORDER_ID = re.compile(r"ORD-\d{4}")
EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")
STATUSES = ("processing", "shipped", "delivered", "cancelled")
SHIPPING_METHODS = ("standard", "express")
COUNTRIES = ("US", "CA")  # KB-013: we ship to the United States and Canada.
CATEGORIES = ("apparel", "accessories", "home", "electronics", "beauty", "gift_card", "digital")

_ORDER_FIELDS = {"order_id", "email", "status", "ordered_days_ago", "delivered_days_ago", "shipping", "items"}
_SHIPPING_FIELDS = {"method", "country", "carrier", "tracking_status", "estimated_delivery_in_days"}
_ITEM_FIELDS = {"name", "category", "quantity", "opened", "final_sale", "personalized"}


class DemoDataError(ValueError):
    """The demo order file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Item:
    name: str
    category: str
    quantity: int
    opened: bool
    final_sale: bool
    personalized: bool


@dataclass(frozen=True)
class Shipping:
    method: str
    country: str
    carrier: str | None
    tracking_status: str | None
    estimated_delivery_in_days: int | None


@dataclass(frozen=True)
class Order:
    order_id: str
    email: str
    status: str
    ordered_days_ago: int
    delivered_days_ago: int | None
    shipping: Shipping
    items: tuple[Item, ...]

    def order_date(self, today: date) -> date:
        return today - timedelta(days=self.ordered_days_ago)

    def delivery_date(self, today: date) -> date | None:
        return None if self.delivered_days_ago is None else today - timedelta(days=self.delivered_days_ago)

    def item(self, name: str) -> Item | None:
        """The item with this name (case and surrounding spaces ignored), or None."""
        wanted = " ".join(str(name).split()).casefold()
        return next((i for i in self.items if i.name.casefold() == wanted), None)


# --- Validation ------------------------------------------------------------------


def _fields(value, expected: set[str], where: str) -> dict:
    if not isinstance(value, dict):
        raise DemoDataError(f"{where}: must be an object")
    missing, extra = expected - set(value), set(value) - expected
    if missing:
        raise DemoDataError(f"{where}: missing field(s) {', '.join(sorted(missing))}")
    if extra:
        raise DemoDataError(f"{where}: unexpected field(s) {', '.join(sorted(extra))}")
    return value


def _text(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DemoDataError(f"{where}: must be non-empty text without surrounding spaces")
    return value


def _days(value, where: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DemoDataError(f"{where}: must be a whole number of days, 0 or more")
    return value


def _flag(value, where: str) -> bool:
    if not isinstance(value, bool):
        raise DemoDataError(f"{where}: must be true or false")
    return value


def _choice(value, choices, where: str) -> str:
    if value not in choices:
        raise DemoDataError(f"{where}: must be one of {', '.join(choices)}")
    return value


def _item(raw, where: str) -> Item:
    raw = _fields(raw, _ITEM_FIELDS, where)
    quantity = raw["quantity"]
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
        raise DemoDataError(f"{where}.quantity: must be a whole number of at least 1")
    return Item(
        name=_text(raw["name"], f"{where}.name"),
        category=_choice(raw["category"], CATEGORIES, f"{where}.category"),
        quantity=quantity,
        opened=_flag(raw["opened"], f"{where}.opened"),
        final_sale=_flag(raw["final_sale"], f"{where}.final_sale"),
        personalized=_flag(raw["personalized"], f"{where}.personalized"),
    )


def _shipping(raw, where: str) -> Shipping:
    raw = _fields(raw, _SHIPPING_FIELDS, where)
    carrier, tracking = raw["carrier"], raw["tracking_status"]
    return Shipping(
        method=_choice(raw["method"], SHIPPING_METHODS, f"{where}.method"),
        country=_choice(raw["country"], COUNTRIES, f"{where}.country"),
        carrier=None if carrier is None else _text(carrier, f"{where}.carrier"),
        tracking_status=None if tracking is None else _text(tracking, f"{where}.tracking_status"),
        estimated_delivery_in_days=_days(
            raw["estimated_delivery_in_days"], f"{where}.estimated_delivery_in_days", optional=True
        ),
    )


def _order(raw, where: str) -> Order:
    raw = _fields(raw, _ORDER_FIELDS, where)
    order_id = raw["order_id"]
    if not isinstance(order_id, str) or not ORDER_ID.fullmatch(order_id):
        raise DemoDataError(f"{where}.order_id: must look like ORD-1001")
    where = order_id
    email = raw["email"]
    if not isinstance(email, str) or not EMAIL.fullmatch(email):
        raise DemoDataError(f"{where}.email: must be a lowercase email address")
    if not email.endswith("@example.com"):
        raise DemoDataError(f"{where}.email: demo data must use the reserved example.com domain")
    status = _choice(raw["status"], STATUSES, f"{where}.status")
    ordered = _days(raw["ordered_days_ago"], f"{where}.ordered_days_ago")
    delivered = _days(raw["delivered_days_ago"], f"{where}.delivered_days_ago", optional=True)
    shipping = _shipping(raw["shipping"], f"{where}.shipping")
    items = raw["items"]
    if not isinstance(items, list) or not items:
        raise DemoDataError(f"{where}.items: must be a non-empty list")
    parsed_items = tuple(_item(item, f"{where}.items[{n}]") for n, item in enumerate(items))
    names = [i.name.casefold() for i in parsed_items]
    if len(set(names)) != len(names):
        raise DemoDataError(f"{where}.items: item names must be unique within an order")

    # The status decides which delivery details must (and must not) be present.
    if status == "delivered":
        if delivered is None or delivered > ordered:
            raise DemoDataError(f"{where}: a delivered order needs delivered_days_ago, no earlier than the order")
        if shipping.carrier is None or shipping.tracking_status != "Delivered":
            raise DemoDataError(f"{where}: a delivered order needs a carrier and tracking_status Delivered")
        if shipping.estimated_delivery_in_days is not None:
            raise DemoDataError(f"{where}: a delivered order has no estimated delivery")
    elif delivered is not None:
        raise DemoDataError(f"{where}: only a delivered order has delivered_days_ago")
    if status == "shipped" and (
        shipping.carrier is None or shipping.tracking_status is None or shipping.estimated_delivery_in_days is None
    ):
        raise DemoDataError(f"{where}: a shipped order needs a carrier, tracking status and estimated delivery")
    if status in ("processing", "cancelled") and (
        shipping.carrier is not None or shipping.tracking_status is not None
        or shipping.estimated_delivery_in_days is not None
    ):
        raise DemoDataError(f"{where}: a {status} order has not shipped, so it has no shipping details")

    return Order(order_id, email, status, ordered, delivered, shipping, parsed_items)


def validate(data) -> list[Order]:
    """Check the whole file. Raises DemoDataError naming the first problem."""
    data = _fields(data, {"version", "notice", "orders"}, "demo orders file")
    if data["version"] != DATA_VERSION:
        raise DemoDataError(f"demo orders file: unsupported version {data['version']!r}")
    _text(data["notice"], "demo orders file.notice")
    if not isinstance(data["orders"], list) or not data["orders"]:
        raise DemoDataError("demo orders file: orders must be a non-empty list")
    orders = [_order(raw, f"orders[{n}]") for n, raw in enumerate(data["orders"])]
    ids = [o.order_id for o in orders]
    if len(set(ids)) != len(ids):
        raise DemoDataError("demo orders file: duplicate order_id")
    return orders


# --- Loading and lookup ----------------------------------------------------------


class OrderBook:
    """The demo orders, looked up only by order ID *and* email."""

    def __init__(self, orders: list[Order]):
        self._orders = {order.order_id: order for order in orders}

    def __len__(self) -> int:
        return len(self._orders)

    def find(self, order_id, email) -> Order | None:
        """The order if both the ID and the email match it; otherwise None.

        Unknown IDs, wrong emails and malformed input all return None, so the
        caller can't tell them apart. Spaces around either value, the case of
        the email and the case of "ord-" are ignored.
        """
        if not isinstance(order_id, str) or not isinstance(email, str):
            return None
        order = self._orders.get(order_id.strip().upper())
        supplied = email.strip().casefold().encode("utf-8")
        expected = order.email.encode("utf-8") if order is not None else b"\0"
        # compare_digest runs even for unknown IDs, so timing reveals nothing either.
        matches = hmac.compare_digest(supplied, expected)
        return order if order is not None and matches else None

    def status(self, order_id: str) -> str | None:
        """The order's status in the data, or None if there is no such order.

        For server-side checks of an order the customer has already proved
        is theirs (cancellations.confirm). Tools never use it: they look
        orders up only with `find`, by ID and email."""
        order = self._orders.get(order_id) if isinstance(order_id, str) else None
        return None if order is None else order.status


def load_orders(path: Path | str | None = None) -> OrderBook:
    """Load and validate the demo orders. Raises DemoDataError on any problem."""
    path = DEMO_ORDERS_PATH if path is None else Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DemoDataError(f"Demo orders {path} could not be read ({type(error).__name__})") from None
    except ValueError:
        raise DemoDataError(f"Demo orders {path} is not valid JSON") from None
    return OrderBook(validate(data))
