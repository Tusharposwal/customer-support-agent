"""Customer feedback on assistant replies: helpful or not.

A rating is "positive" or "negative" and nothing else: no comment box, no
name, no email. It belongs to one assistant reply, identified by its
position in the conversation (`message_index`, 0 = the first message). The
page knows that position; database row IDs are never exposed, because they
count messages across all conversations and could be guessed.

`submit` checks, in one write transaction:
* the conversation exists and is live (else ConversationNotFound): holding
  the conversation ID is what proves it is the customer's;
* the message at that position exists in *this* conversation and is an
  assistant reply (else MessageNotFound): a customer message, a position
  past the end or another conversation's message can't be rated;
* then it saves the rating. There is one rating per reply (the reply's
  message ID is the primary key): sending the same rating again changes
  nothing, and a different rating replaces the earlier one (a correction).

Ratings are deleted with their conversation (30 days after its last message).
"""

from dataclasses import dataclass

from support_agent.conversation_store import ConversationNotFound, ConversationStore, is_valid_id

POSITIVE = "positive"
NEGATIVE = "negative"
RATINGS = (POSITIVE, NEGATIVE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    message_id       INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    rating           TEXT NOT NULL CHECK (rating IN ('positive', 'negative')),
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_by_conversation ON feedback (conversation_id);
"""


class MessageNotFound(LookupError):
    """No assistant reply at that position in this conversation."""


@dataclass(frozen=True)
class FeedbackResult:
    message_index: int
    rating: str
    changed: bool  # False when the same rating was already saved (nothing written)


class FeedbackStore:
    def __init__(self, conversations: ConversationStore):
        self.conversations = conversations

    def initialize(self) -> None:
        self.conversations._run(lambda connection: connection.executescript(_SCHEMA))

    def submit(self, conversation_id: str, message_index: int, rating: str) -> FeedbackResult:
        if rating not in RATINGS:
            raise ValueError("rating must be positive or negative")
        if not is_valid_id(conversation_id):
            raise ConversationNotFound()
        if not isinstance(message_index, int) or isinstance(message_index, bool) or message_index < 0:
            raise MessageNotFound()
        store = self.conversations

        def write(connection):
            def transaction():
                now = store.clock()
                if not store._live(connection, conversation_id, store._cutoff(now)):
                    raise ConversationNotFound()
                row = connection.execute(
                    "SELECT id, role FROM messages WHERE conversation_id = ? ORDER BY id LIMIT 1 OFFSET ?",
                    (conversation_id, message_index),
                ).fetchone()
                if row is None or row[1] != "assistant":
                    raise MessageNotFound()
                message_id = row[0]
                existing = connection.execute(
                    "SELECT rating FROM feedback WHERE message_id = ?", (message_id,)
                ).fetchone()
                if existing is not None and existing[0] == rating:
                    return FeedbackResult(message_index, rating, changed=False)
                if existing is None:
                    connection.execute(
                        "INSERT INTO feedback (message_id, conversation_id, rating, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (message_id, conversation_id, rating, now, now),
                    )
                else:
                    connection.execute(
                        "UPDATE feedback SET rating = ?, updated_at = ? WHERE message_id = ?",
                        (rating, now, message_id),
                    )
                return FeedbackResult(message_index, rating, changed=True)

            return store._write(connection, transaction)

        return store._run(write)

    def ratings(self, conversation_id: str) -> dict[int, str]:
        """{message_index: rating} for the conversation's rated replies."""
        if not is_valid_id(conversation_id):
            return {}

        def read(connection):
            return connection.execute(
                "SELECT (SELECT count(*) FROM messages AS m "
                "        WHERE m.conversation_id = f.conversation_id AND m.id < f.message_id), f.rating "
                "FROM feedback AS f WHERE f.conversation_id = ?",
                (conversation_id,),
            ).fetchall()

        return dict(self.conversations._run(read))
