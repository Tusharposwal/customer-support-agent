"""Hand-off records: which conversations need a human.

When the agent escalates (any reply with escalate=True: the standard
hand-off message), /chat saves a hand-off record together with the reply, in
the same transaction (`recorder`). A record says *that* a conversation needs
human help and why, in safe terms:

    reason    the agent's escalation reason, if it is one of REASONS
              ("insufficient_evidence", "llm_error", ...); anything else is
              stored as "other". Never free text, question text or error text.
    category  "no_answer" (the help articles didn't support an answer),
              "service_problem" (something failed) or "other".

A conversation has at most one *open* record. A later escalation in the same
conversation updates it (last reason, count, time) instead of adding another.
A partial unique index enforces this even if the code were wrong.

IMPORTANT: these are internal records only. This project is not connected to
any human-support system: nobody is notified, assigned or contacted, and
nothing ever closes a record. The customer is told only that their question
was recorded for the support team, never that a person is on it. A real
deployment would have a support tool read (and close) these records.

Records are deleted with their conversation (30 days after its last message).
"""

from dataclasses import dataclass

from support_agent.conversation_store import ConversationStore, is_valid_id

OPEN = "open"

NO_ANSWER = "no_answer"
SERVICE_PROBLEM = "service_problem"
OTHER = "other"
CATEGORIES = (NO_ANSWER, SERVICE_PROBLEM, OTHER)

# Every reason the agent, the answer generator and the API escalate with.
REASONS = {
    "insufficient_evidence": NO_ANSWER,
    "no_articles": NO_ANSWER,
    "no_citations": NO_ANSWER,
    "invalid_citations": NO_ANSWER,
    "below_rerank_threshold": NO_ANSWER,
    "incomplete_response": NO_ANSWER,
    "empty_question": NO_ANSWER,
    "llm_error": SERVICE_PROBLEM,
    "retrieval_error": SERVICE_PROBLEM,
    "configuration_error": SERVICE_PROBLEM,
    "agent_error": SERVICE_PROBLEM,
    "server_error": SERVICE_PROBLEM,
    "malformed_response": SERVICE_PROBLEM,
    "tool_error": SERVICE_PROBLEM,
    "invalid_tool_call": SERVICE_PROBLEM,
}

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS handoffs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    status           TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    category         TEXT NOT NULL CHECK (category IN ({", ".join(repr(c) for c in CATEGORIES)})),
    first_reason     TEXT NOT NULL,
    last_reason      TEXT NOT NULL,
    escalations      INTEGER NOT NULL CHECK (escalations >= 1),
    first_message_id INTEGER NOT NULL,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_handoff_per_conversation
    ON handoffs (conversation_id) WHERE status = 'open';
"""


@dataclass(frozen=True)
class Handoff:
    status: str
    category: str
    first_reason: str
    last_reason: str
    escalations: int
    created_at: float
    updated_at: float


def classify(reason) -> tuple[str, str]:
    """(reason, category) to store: a known reason and its category, or
    ("other", "other") for anything else (unknown, empty, not a string)."""
    if isinstance(reason, str) and reason in REASONS:
        return reason, REASONS[reason]
    return OTHER, OTHER


class HandoffStore:
    """Hand-off records in the conversation database (same connections and
    StoreError handling as ConversationStore)."""

    def __init__(self, conversations: ConversationStore):
        self.conversations = conversations

    def initialize(self) -> None:
        self.conversations._run(lambda connection: connection.executescript(_SCHEMA))

    def recorder(self, conversation_id: str, reason):
        """A callback for add_exchange's `in_transaction`: opens the
        conversation's hand-off record, or updates the open one. Runs after
        the escalated reply was inserted, so that reply is the conversation's
        latest assistant message."""
        reason, category = classify(reason)

        def record(connection, now: float) -> None:
            updated = connection.execute(
                "UPDATE handoffs SET last_reason = ?, escalations = escalations + 1, updated_at = ? "
                "WHERE conversation_id = ? AND status = 'open'",
                (reason, now, conversation_id),
            ).rowcount
            if updated:
                return
            (message_id,) = connection.execute(
                "SELECT max(id) FROM messages WHERE conversation_id = ? AND role = 'assistant'",
                (conversation_id,),
            ).fetchone()
            connection.execute(
                "INSERT INTO handoffs (conversation_id, status, category, first_reason, last_reason, "
                "escalations, first_message_id, created_at, updated_at) VALUES (?, 'open', ?, ?, ?, 1, ?, ?, ?)",
                (conversation_id, category, reason, reason, message_id, now, now),
            )

        return record

    def open_handoff(self, conversation_id: str) -> Handoff | None:
        """The conversation's open hand-off record, if any."""
        if not is_valid_id(conversation_id):
            return None

        def read(connection):
            return connection.execute(
                "SELECT status, category, first_reason, last_reason, escalations, created_at, updated_at "
                "FROM handoffs WHERE conversation_id = ? AND status = 'open'",
                (conversation_id,),
            ).fetchone()

        row = self.conversations._run(read)
        return None if row is None else Handoff(*row)
