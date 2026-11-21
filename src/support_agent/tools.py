"""The tool registry: the Python functions the model may ask to run.

A *tool definition* is what the model is shown: a name, a description and a
JSON Schema for its arguments (`definitions()`, sent to Cohere as `tools=`).
*Tool execution* is what happens when the model asks for one: this module
checks the request and runs the matching Python function (`execute`,
`execute_all`). Both come from the same `ToolSpec` entries in `REGISTRY`, so
what the model is told and what the code accepts can't drift apart.

The three tools only read data:

    get_order_status            status, shipping and items of one order
    check_refund_eligibility    the decision made by refund_policy.py
    request_order_cancellation  whether an order can be cancelled; it only
                                proposes a cancellation for the customer to
                                confirm and never changes anything

Rules every call goes through, in order:

1. The tool must exist (else `unknown_tool`; nothing runs).
2. The arguments must be a JSON object of at most MAX_ARGUMENT_CHARS
   characters, with no duplicate or unknown keys, every required key, and
   valid values (else `invalid_arguments`, naming fields but never values).
3. Grounding: the order number and the email must both be given (an
   argument the model left out because the customer never gave it) and
   must both appear in the customer's own messages in this conversation
   (else `details_not_provided`, and the customer is asked for them). Assistant replies don't count, so the model can't
   invent an order or be talked into using someone else's.
4. Ownership: the order is looked up only by ID *and* email
   (OrderBook.find); an unknown order and a wrong email look the same
   (`not_found`).
5. The tool runs. Any unexpected error becomes a generic `error`.

Every call gets a result (a JSON object with a `status`); `execute` never
raises. Results never contain the customer's email. Log lines carry the tool
name, the status and an error type only: never arguments, order numbers,
emails or error text. At most MAX_TOOL_CALLS calls run per question; any
further call gets `limit_exceeded` without running.

No network, no Cohere client and no order changes happen here. A
cancellation the customer confirmed (cancellations.py, through its own
endpoint) is *read* here: `ToolContext.is_cancelled` makes such an order
appear as cancelled, because data/demo_orders.json itself is never changed.
"""

import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from support_agent.demo_orders import EMAIL, ORDER_ID, Order, OrderBook
from support_agent.refund_policy import REASONS, STANDARD, UnknownItemError, check_refund

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 3
MAX_ARGUMENT_CHARS = 2000
MAX_EMAIL_CHARS = 254
MAX_ITEM_NAME_CHARS = 100

# Result statuses (plain identifiers: shown to the model and logged).
OK = "ok"
CONFIRMATION_REQUIRED = "confirmation_required"
NOT_CANCELLABLE = "not_cancellable"
NOT_FOUND = "not_found"
DETAILS_NOT_PROVIDED = "details_not_provided"
INVALID_ARGUMENTS = "invalid_arguments"
UNKNOWN_ITEM = "unknown_item"
UNKNOWN_TOOL = "unknown_tool"
LIMIT_EXCEEDED = "limit_exceeded"
ERROR = "error"

# KB articles a tool result relates to, so an answer can cite the policy too.
_STATUS_ARTICLES = {
    "processing": ("KB-003", "KB-024"),
    "shipped": ("KB-003", "KB-026"),
    "delivered": ("KB-003", "KB-027"),
    "cancelled": ("KB-024",),
}
_CANCELLATION_ARTICLES = ("KB-024",)


# --- Data structures ----------------------------------------------------------------


@dataclass(frozen=True)
class ToolContext:
    """What a tool may use: the orders, today's date and the customer's own
    messages in this conversation (the only evidence of order details).

    `is_cancelled(order_id)` says whether a confirmed cancellation of the
    order has been recorded (cancellations.CancellationStore.is_cancelled).
    Default: none ever has."""

    orders: OrderBook
    customer_messages: tuple[str, ...]
    today: date
    is_cancelled: Callable[[str], bool] = field(default=lambda order_id: False)

    @classmethod
    def for_question(cls, orders: OrderBook, question: str, history: Sequence[Mapping] | None, today: date,
                     is_cancelled: Callable[[str], bool] | None = None):
        """Customer messages from `history` ([{"role", "content"}], as the API
        passes it) plus the current question. Assistant messages are dropped."""
        earlier = tuple(
            str(m.get("content", "")) for m in (history or ()) if m.get("role") == "customer"
        )
        context = cls(orders=orders, customer_messages=earlier + (question,), today=today)
        return context if is_cancelled is None else replace(context, is_cancelled=is_cancelled)


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: str | None  # The JSON text from the model (SDK: Optional[str]).


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str  # The tool that ran, or "unknown" for a name not in the registry.
    status: str
    data: dict  # JSON-safe; always includes "status". Never the email.
    document_id: str  # Unique within a question; never clashes with KB-### IDs.

    def as_document(self) -> dict:
        """The result as a Cohere tool-result document: {"id", "data"} with
        text values (nested lists and objects are JSON-encoded)."""
        values = {
            key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            for key, value in self.data.items()
        }
        return {"id": self.document_id, "data": values}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict  # JSON Schema of each argument
    required: tuple[str, ...]
    handler: Callable[[dict, ToolContext], dict]
    optional: tuple[str, ...] = field(default=())

    def definition(self) -> dict:
        """The Cohere `tools=` entry (ToolV2) for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                    "additionalProperties": False,
                },
            },
        }


class _InvalidArguments(Exception):
    def __init__(self, fields: Iterable[str] = (), unexpected: int = 0):
        super().__init__("invalid arguments")
        self.fields = sorted(set(fields))
        self.unexpected = unexpected


class _MissingDetails(Exception):
    """Well-formed arguments that leave out the order number and/or email."""

    def __init__(self, fields: Iterable[str]):
        super().__init__("missing details")
        self.fields = sorted(set(fields))


# Required arguments that only the customer can supply. Leaving them out is
# "details not provided" (the customer is asked for them), not a malformed call.
_CUSTOMER_DETAILS = {"order_id", "email"}


# --- Argument parsing and validation -------------------------------------------------


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _normalise_order_id(value) -> str | None:
    if not isinstance(value, str):
        return None
    order_id = value.strip().upper()
    return order_id if ORDER_ID.fullmatch(order_id) else None


def _normalise_email(value) -> str | None:
    if not isinstance(value, str):
        return None
    email = value.strip().casefold()
    return email if len(email) <= MAX_EMAIL_CHARS and EMAIL.fullmatch(email) else None


def _normalise_reason(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() in REASONS else None


def _normalise_item_name(value) -> str | None:
    if not isinstance(value, str):
        return None
    name = " ".join(value.split())
    return name if 1 <= len(name) <= MAX_ITEM_NAME_CHARS else None


_NORMALISERS = {
    "order_id": _normalise_order_id,
    "email": _normalise_email,
    "reason": _normalise_reason,
    "item_name": _normalise_item_name,
}


def parse_arguments(raw, spec: ToolSpec) -> dict:
    """Parse and validate the model's JSON arguments for `spec`.
    Returns normalised values; raises _InvalidArguments otherwise."""
    if not isinstance(raw, str) or len(raw) > MAX_ARGUMENT_CHARS:
        raise _InvalidArguments(["arguments"])
    try:
        parsed = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (ValueError, RecursionError):  # Not JSON, a duplicate key, or absurdly deep nesting.
        raise _InvalidArguments(["arguments"]) from None
    if not isinstance(parsed, dict):
        raise _InvalidArguments(["arguments"])

    allowed = set(spec.required) | set(spec.optional)
    unexpected = len(set(parsed) - allowed)
    absent = [name for name in spec.required if name not in parsed]
    invalid = []
    values = {}
    for name in allowed & set(parsed):
        value = _NORMALISERS[name](parsed[name])
        if value is None:
            invalid.append(name)
        else:
            values[name] = value
    if invalid or unexpected or (set(absent) - _CUSTOMER_DETAILS):
        raise _InvalidArguments(invalid + absent, unexpected)
    if absent:
        # Only the customer's own details are missing (the model left them out
        # because the customer hasn't given them): ask for them, like an
        # ungrounded value. Nothing is looked up.
        raise _MissingDetails(absent)
    return values


# --- Grounding: the details must come from the customer --------------------------------


def _appears(pattern: str, texts: Iterable[str]) -> bool:
    regex = re.compile(pattern, re.IGNORECASE)
    return any(regex.search(text) for text in texts)


def ungrounded_fields(arguments: dict, context: ToolContext) -> list[str]:
    """The order details the customer did not write themselves in this
    conversation (assistant messages are not in the context at all)."""
    missing = []
    order_id = re.escape(arguments["order_id"])
    # Whole token: "ORD-1001" but not "ORD-10011" or "XORD-1001".
    if not _appears(rf"(?<![A-Za-z0-9-]){order_id}(?![0-9])", context.customer_messages):
        missing.append("order_id")
    email = re.escape(arguments["email"])
    # Whole address: not a part of a longer address ("a@example.com.au", "xa@example.com").
    if not _appears(rf"(?<![\w.%+-]){email}(?![\w-]|\.[\w-])", context.customer_messages):
        missing.append("email")
    return missing


# --- The tools ---------------------------------------------------------------------


def _items(order: Order) -> list[dict]:
    return [{"name": item.name, "quantity": item.quantity} for item in order.items]


def _iso(day: date | None) -> str | None:
    return None if day is None else day.isoformat()


def _get_order_status(arguments: dict, context: ToolContext, order: Order) -> dict:
    shipping = order.shipping
    estimated = shipping.estimated_delivery_in_days
    return {
        "status": OK,
        "order_id": order.order_id,
        "order_status": order.status,
        "ordered_on": _iso(order.order_date(context.today)),
        "ordered_days_ago": order.ordered_days_ago,
        "delivered_on": _iso(order.delivery_date(context.today)),
        "shipping": {
            "method": shipping.method,
            "country": shipping.country,
            "carrier": shipping.carrier,
            "tracking_status": shipping.tracking_status,
            "estimated_delivery_on": None if estimated is None else _iso(context.today + timedelta(days=estimated)),
        },
        "items": _items(order),
        "article_ids": list(_STATUS_ARTICLES[order.status]),
    }


def _check_refund_eligibility(arguments: dict, context: ToolContext, order: Order) -> dict:
    try:
        check = check_refund(order, arguments.get("reason", STANDARD), arguments.get("item_name"))
    except UnknownItemError:
        # The order is the customer's own (checked already), so listing its items is safe.
        return {"status": UNKNOWN_ITEM, "order_id": order.order_id, "items": _items(order)}
    return {
        "status": OK,
        "order_id": check.order_id,
        "order_status": check.status,
        "reason": check.reason,
        "days_since_order": check.days_since_order,
        "days_since_delivery": check.days_since_delivery,
        "items": [
            {
                "item_name": d.item_name,
                "decision": d.decision,
                "rule": d.rule,
                "exclusion": d.exclusion,
                "conditions": list(d.conditions),
                "article_ids": list(d.article_ids),
            }
            for d in check.items
        ],
        "decided_by": "refund_policy",
    }


def _request_order_cancellation(arguments: dict, context: ToolContext, order: Order) -> dict:
    """Read-only: says whether the order can be cancelled. Nothing is cancelled
    or recorded here; a later, separate customer confirmation does that."""
    if order.status != "processing":  # KB-024: only while Processing.
        return {
            "status": NOT_CANCELLABLE,
            "order_id": order.order_id,
            "order_status": order.status,
            "article_ids": list(_CANCELLATION_ARTICLES),
        }
    return {
        "status": CONFIRMATION_REQUIRED,
        "order_id": order.order_id,
        "order_status": order.status,
        "items": _items(order),
        "article_ids": list(_CANCELLATION_ARTICLES),
        "note": "The order has NOT been cancelled. Only the customer's own confirmation can cancel it.",
    }


def _owned_order(handler):
    """Wrap a tool so it runs only for an order the customer owns and named."""

    def run(arguments: dict, context: ToolContext) -> dict:
        missing = ungrounded_fields(arguments, context)
        if missing:
            return {"status": DETAILS_NOT_PROVIDED, "missing": missing}
        order = context.orders.find(arguments["order_id"], arguments["email"])
        if order is None:
            return {"status": NOT_FOUND}  # Unknown order and wrong email look the same.
        if order.status == "processing" and context.is_cancelled(order.order_id):
            # Confirmed by the customer and recorded in SQLite; the JSON file is
            # read-only. (Only a processing order can ever be cancelled.)
            order = replace(order, status="cancelled")
        return handler(arguments, context, order)

    return run


_ORDER_ID_SCHEMA = {
    "type": "string",
    "pattern": r"^ORD-\d{4}$",
    "description": "The order number exactly as the customer wrote it, e.g. ORD-1001.",
}
_EMAIL_SCHEMA = {
    "type": "string",
    "maxLength": MAX_EMAIL_CHARS,
    "description": "The email address the customer gave for this order.",
}

REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="get_order_status",
            description=(
                "Look up the status, shipping details and items of one order. Use only when the "
                "customer has given both their order number and the email address used for the order."
            ),
            properties={"order_id": _ORDER_ID_SCHEMA, "email": _EMAIL_SCHEMA},
            required=("order_id", "email"),
            handler=_owned_order(_get_order_status),
        ),
        ToolSpec(
            name="check_refund_eligibility",
            description=(
                "Check whether items in an order can be refunded, using the store's refund rules. "
                "The decision is made by those rules: report it as given and do not change it. "
                "Use only when the customer has given their order number and email address."
            ),
            properties={
                "order_id": _ORDER_ID_SCHEMA,
                "email": _EMAIL_SCHEMA,
                "reason": {
                    "type": "string",
                    "enum": list(REASONS),
                    "description": (
                        "Why the customer wants a refund: standard (they no longer want it), damaged, "
                        "defective, wrong_item or missing_item. Default: standard."
                    ),
                },
                "item_name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_ITEM_NAME_CHARS,
                    "description": "One item's name, if the customer asks about a single item.",
                },
            },
            required=("order_id", "email"),
            optional=("reason", "item_name"),
            handler=_owned_order(_check_refund_eligibility),
        ),
        ToolSpec(
            name="request_order_cancellation",
            description=(
                "Check whether an order can still be cancelled and prepare a cancellation for the "
                "customer to confirm. This does NOT cancel the order: only the customer's own "
                "confirmation can. Use only when the customer asks to cancel and has given their "
                "order number and email address."
            ),
            properties={"order_id": _ORDER_ID_SCHEMA, "email": _EMAIL_SCHEMA},
            required=("order_id", "email"),
            handler=_owned_order(_request_order_cancellation),
        ),
    )
}


def definitions() -> list[dict]:
    """The `tools=` list for Cohere's ClientV2.chat, built from REGISTRY."""
    return [spec.definition() for spec in REGISTRY.values()]


# --- Execution ---------------------------------------------------------------------


def _result(call_id, name: str, data: dict, index: int) -> ToolResult:
    call_id = call_id if isinstance(call_id, str) else ""
    return ToolResult(call_id=call_id, name=name, status=data["status"], data=data,
                      document_id=f"tool-{index}-{name}")


def execute(call: ToolCall, context: ToolContext, *, index: int = 1) -> ToolResult:
    """Run one tool call. Never raises; every outcome is a ToolResult."""
    spec = REGISTRY.get(call.name) if isinstance(call.name, str) else None
    if spec is None:
        logger.warning("Tool call rejected: unknown tool")
        return _result(call.id, "unknown", {"status": UNKNOWN_TOOL}, index)

    try:
        arguments = parse_arguments(call.arguments, spec)
    except _MissingDetails as missing:
        logger.info("Tool %s: %s", spec.name, DETAILS_NOT_PROVIDED)
        return _result(call.id, spec.name, {"status": DETAILS_NOT_PROVIDED, "missing": missing.fields}, index)
    except _InvalidArguments as invalid:
        logger.info("Tool %s: %s", spec.name, INVALID_ARGUMENTS)
        data = {"status": INVALID_ARGUMENTS, "invalid_fields": invalid.fields,
                "unexpected_fields": invalid.unexpected}
        return _result(call.id, spec.name, data, index)

    try:
        data = spec.handler(arguments, context)
    except Exception as error:  # A bug or data problem: never shown or logged in detail.
        logger.warning("Tool %s: %s (%s)", spec.name, ERROR, type(error).__name__)
        return _result(call.id, spec.name, {"status": ERROR}, index)

    logger.info("Tool %s: %s", spec.name, data["status"])
    return _result(call.id, spec.name, data, index)


def execute_all(calls: Sequence[ToolCall], context: ToolContext) -> list[ToolResult]:
    """Run the calls of one model turn, in order. The first MAX_TOOL_CALLS run;
    every later one gets `limit_exceeded` without running. One result per call."""
    results = []
    for index, call in enumerate(calls, start=1):
        if index > MAX_TOOL_CALLS:
            name = call.name if isinstance(call.name, str) and call.name in REGISTRY else "unknown"
            logger.warning("Tool %s: %s", name, LIMIT_EXCEEDED)
            results.append(_result(call.id, name, {"status": LIMIT_EXCEEDED}, index))
        else:
            results.append(execute(call, context, index=index))
    return results
