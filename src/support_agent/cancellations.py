"""Persistent order cancellations, with explicit customer confirmation.

The flow:

1. The model asks the read-only `request_order_cancellation` tool about an
   order the customer owns. If the order is still processing, the agent gives
   a fixed, server-written reply ("can be cancelled; it has NOT been
   cancelled").
2. When /chat saves that reply, a *pending action* is saved with it, in the
   same transaction (`proposer`): a random 128-bit ID, the conversation, the
   order number and an expiry (ACTION_TTL_SECONDS). The ID goes to the
   customer's browser only, in the /chat response. It is never in a message,
   a prompt, a tool result or a log line, so the model never sees it and
   can't produce or repeat it.
3. The order is cancelled only by POST /cancellations/confirm with the
   conversation ID *and* the action ID (`confirm`). Nothing a customer types
   into the chat can cancel anything: "yes" is just another question.

`confirm` checks, in one write transaction (BEGIN IMMEDIATE, so concurrent
confirmations queue up instead of racing):

* the action exists and belongs to this conversation, which is still live
  (an unknown ID and someone else's conversation look the same: ActionNotFound);
* the action was already confirmed: the original result is returned again
  (idempotent; nothing is written twice);
* the action has not expired (ActionExpired);
* the order's current status is still "processing" and it has no recorded
  cancellation (OrderNotCancellable);
* then the cancellation is recorded. `order_cancellations.order_id` is the
  primary key, so an order can be cancelled at most once, whatever happens.

data/demo_orders.json is never written: a recorded cancellation overrides the
order's status when tools read it (`is_cancelled`, used via
tools.ToolContext). Action IDs are stored only as SHA-256 hashes, so a copy of
the database can't be used to confirm anything.

Pending actions are deleted with their conversation (30 days after its last
message). Recorded cancellations are order state and are kept. All of this
lives in the conversation database, so on Render's free plan it is lost on
every restart, like the conversations.
"""

import hashlib
import logging
import secrets
import sqlite3
from dataclasses import dataclass

from support_agent.conversation_store import ConversationStore, is_valid_id

logger = logging.getLogger(__name__)

ACTION_TTL_SECONDS = 15 * 60
CANCELLABLE_STATUS = "processing"  # KB-024: only while Processing.

# What the page is told about an offer when it restores a conversation.
PENDING = "pending"  # Not confirmed, not expired: it can still be confirmed.
CANCELLED = "cancelled"  # The order has been cancelled (by this offer or another one).
EXPIRED = "expired"  # Not confirmed in time; nothing was cancelled.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cancellation_actions (
    id_hash          TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    order_id         TEXT NOT NULL,
    created_at       REAL NOT NULL,
    expires_at       REAL NOT NULL,
    confirmed_at     REAL
);
CREATE INDEX IF NOT EXISTS cancellation_actions_by_conversation
    ON cancellation_actions (conversation_id);
CREATE TABLE IF NOT EXISTS order_cancellations (
    order_id      TEXT PRIMARY KEY,
    action_hash   TEXT NOT NULL UNIQUE,
    cancelled_at  REAL NOT NULL
);
"""


class ActionNotFound(LookupError):
    """No such action in this (live) conversation."""


class ActionExpired(Exception):
    """The action exists but was not confirmed in time."""


class OrderNotCancellable(Exception):
    """The order's status changed since the action was created."""


@dataclass(frozen=True)
class PendingCancellation:
    action_id: str  # The secret; returned once, to the customer's browser only.
    order_id: str
    expires_at: float


@dataclass(frozen=True)
class OfferStatus:
    """An offer's state, for restoring the page. Never contains the action
    ID or its hash. `created_at` equals the `created_at` of the reply that
    made the offer: both are saved in one transaction with the same time."""

    created_at: float
    order_id: str
    status: str  # PENDING, CANCELLED or EXPIRED
    expires_at: float
    cancelled_at: float | None


@dataclass(frozen=True)
class Confirmation:
    order_id: str
    cancelled_at: float
    already_confirmed: bool  # True when this repeats an earlier confirmation.


def new_action_id() -> str:
    """128 random bits from the OS's secure generator, as 32 hex characters."""
    return secrets.token_hex(16)


def _hash(action_id: str) -> str:
    return hashlib.sha256(action_id.encode("ascii")).hexdigest()


class CancellationStore:
    """Pending actions and recorded cancellations, in the conversation
    database. Uses the conversation store's connections and error handling
    (StoreError on any database failure; never IDs or values in the message)."""

    def __init__(self, conversations: ConversationStore, *, ttl_seconds: float = ACTION_TTL_SECONDS):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.conversations = conversations
        self.ttl_seconds = ttl_seconds

    def initialize(self) -> None:
        """Create the tables if needed (after ConversationStore.initialize)."""
        self.conversations._run(lambda connection: connection.executescript(_SCHEMA))

    # --- Creating an action ----------------------------------------------------------

    def proposer(self, conversation_id: str, order_id: str):
        """Returns (record, created): pass `record` as add_exchange's
        `in_transaction`; once the exchange is saved, `created()` is the
        PendingCancellation. If saving fails, no action exists."""
        action_id = new_action_id()
        made: list[PendingCancellation] = []

        def record(connection: sqlite3.Connection, now: float) -> None:
            expires_at = now + self.ttl_seconds
            connection.execute(
                "INSERT INTO cancellation_actions (id_hash, conversation_id, order_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (_hash(action_id), conversation_id, order_id, now, expires_at),
            )
            made.append(PendingCancellation(action_id=action_id, order_id=order_id, expires_at=expires_at))

        def created() -> PendingCancellation:
            return made[0]

        return record, created

    # --- Reading ---------------------------------------------------------------------

    def is_cancelled(self, order_id: str) -> bool:
        """Whether a cancellation of this order has been recorded."""
        def read(connection):
            row = connection.execute("SELECT 1 FROM order_cancellations WHERE order_id = ?", (order_id,)).fetchone()
            return row is not None

        return self.conversations._run(read)

    def offers(self, conversation_id: str) -> list[OfferStatus]:
        """The state of every cancellation offered in this conversation,
        oldest first. Reads only; the caller has already checked the
        conversation exists. An offer counts as CANCELLED when its order has
        been cancelled at all, so an older offer for the same order doesn't
        look confirmable."""
        if not is_valid_id(conversation_id):
            return []
        now = self.conversations.clock()

        def read(connection):
            return connection.execute(
                "SELECT a.created_at, a.order_id, a.expires_at, a.confirmed_at, c.cancelled_at "
                "FROM cancellation_actions AS a "
                "LEFT JOIN order_cancellations AS c ON c.order_id = a.order_id "
                "WHERE a.conversation_id = ? ORDER BY a.created_at",
                (conversation_id,),
            ).fetchall()

        offers = []
        for created_at, order_id, expires_at, confirmed_at, order_cancelled_at in self.conversations._run(read):
            cancelled_at = confirmed_at if confirmed_at is not None else order_cancelled_at
            if cancelled_at is not None:
                status = CANCELLED
            elif now >= expires_at:
                status = EXPIRED
            else:
                status = PENDING
            offers.append(OfferStatus(created_at, order_id, status, expires_at, cancelled_at))
        return offers

    # --- Confirming ------------------------------------------------------------------

    def confirm(self, conversation_id: str, action_id: str, order_status) -> Confirmation:
        """Cancel the action's order, exactly once.

        `order_status(order_id)` returns the order's current status in the
        order data (or None if it no longer exists). Raises ActionNotFound,
        ActionExpired, OrderNotCancellable or StoreError; see the module
        docstring for the order of the checks.
        """
        if not is_valid_id(conversation_id) or not is_valid_id(action_id):
            raise ActionNotFound()
        store = self.conversations
        action_hash = _hash(action_id)

        def write(connection):
            def transaction():
                now = store.clock()
                if not store._live(connection, conversation_id, store._cutoff(now)):
                    raise ActionNotFound()
                row = connection.execute(
                    "SELECT order_id, expires_at, confirmed_at FROM cancellation_actions "
                    "WHERE id_hash = ? AND conversation_id = ?",
                    (action_hash, conversation_id),
                ).fetchone()
                if row is None:
                    raise ActionNotFound()  # Unknown, or another conversation's: indistinguishable.
                order_id, expires_at, confirmed_at = row
                if confirmed_at is not None:
                    return Confirmation(order_id, confirmed_at, already_confirmed=True)
                if now >= expires_at:
                    raise ActionExpired()
                recorded = connection.execute(
                    "SELECT 1 FROM order_cancellations WHERE order_id = ?", (order_id,)
                ).fetchone()
                if recorded is not None or order_status(order_id) != CANCELLABLE_STATUS:
                    raise OrderNotCancellable()
                connection.execute(
                    "INSERT INTO order_cancellations (order_id, action_hash, cancelled_at) VALUES (?, ?, ?)",
                    (order_id, action_hash, now),
                )
                connection.execute(
                    "UPDATE cancellation_actions SET confirmed_at = ? WHERE id_hash = ?", (now, action_hash)
                )
                return Confirmation(order_id, now, already_confirmed=False)

            return store._write(connection, transaction)

        return store._run(write)
