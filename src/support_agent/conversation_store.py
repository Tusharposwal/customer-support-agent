"""SQLite storage for chat conversations.

A conversation is a list of messages: each customer message followed by the
assistant's reply. Conversations are identified by a random 128-bit ID made by
the server (32 lowercase hex characters). Whoever holds the ID can read the
conversation, so the ID is a secret: it is never logged (see `log_hash`).

Rules:
* A conversation holds at most MAX_MESSAGES messages (customer + assistant).
  When it is full, new messages are refused (ConversationFull); nothing is
  dropped or moved to another conversation.
* A conversation expires RETENTION_DAYS after its last message. Expired
  conversations are treated as missing at once and deleted by `purge_expired`.
  Only conversations idle for the whole retention period are deleted, and
  every write re-checks expiry inside its transaction, so cleanup never
  removes a conversation that is still in use.
* A customer message and its reply are saved together in one transaction:
  either both are stored or neither is.

Thread safety: FastAPI runs /chat in worker threads. Every operation opens
its own connection and closes it afterwards, so no connection is shared
between threads. SQLite's own locking (WAL mode, `BEGIN IMMEDIATE` for
writes, a busy timeout) serialises concurrent writers.

Settings:
    CONVERSATION_DB_PATH   database file (default: var/conversations.sqlite3
                           in the project folder). On Render's free plan the
                           file is temporary: it is lost on every restart,
                           redeploy and spin-down.
"""

import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH_ENV = "CONVERSATION_DB_PATH"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "var" / "conversations.sqlite3"

MAX_MESSAGES = 40
RETENTION_DAYS = 30
RETENTION_SECONDS = RETENTION_DAYS * 24 * 3600
PURGE_INTERVAL_SECONDS = 3600.0  # How often requests trigger a cleanup.
BUSY_TIMEOUT_SECONDS = 5.0

CONVERSATION_ID = re.compile(r"[0-9a-f]{32}")
ROLES = ("customer", "assistant")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('customer', 'assistant')),
    content          TEXT NOT NULL,
    created_at       REAL NOT NULL,
    retrieval_query  TEXT,
    sources          TEXT,
    escalate         INTEGER,
    reason           TEXT
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id);
CREATE INDEX IF NOT EXISTS conversations_by_update ON conversations (updated_at);
"""


class StoreError(RuntimeError):
    """The database could not be used. The message never contains message text."""


class ConversationNotFound(LookupError):
    """No such conversation, or it has expired."""


class ConversationFull(Exception):
    """The conversation has no room for another customer message and reply."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    created_at: float
    sources: list = field(default_factory=list)  # assistant only
    escalate: bool | None = None  # assistant only
    reason: str | None = None  # assistant only


@dataclass(frozen=True)
class AssistantReply:
    answer: str
    sources: list
    escalate: bool
    reason: str


def new_conversation_id() -> str:
    """128 random bits from the OS's secure generator."""
    return secrets.token_hex(16)


def is_valid_id(value) -> bool:
    return isinstance(value, str) and CONVERSATION_ID.fullmatch(value) is not None


def log_hash(conversation_id: str) -> str:
    """A short, one-way label for log lines. Lets lines of one conversation be
    grouped without revealing the ID (a 128-bit random secret can't be found
    from its hash)."""
    return hashlib.sha256(conversation_id.encode("ascii")).hexdigest()[:12]


def db_path_from_env(environ: Mapping[str, str] | None = None) -> Path:
    """Read CONVERSATION_DB_PATH. Blank or unset means DEFAULT_DB_PATH.
    Raises ValueError (naming the setting, never its value) if it can't be a
    database file. Creates nothing."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(DB_PATH_ENV) or "").strip()
    if not raw:
        return DEFAULT_DB_PATH
    if raw == ":memory:" or raw.startswith("file:"):
        # Each operation opens its own connection, so an in-memory database
        # would be empty every time.
        raise ValueError(f"{DB_PATH_ENV} must be a file path")
    path = Path(raw)
    if path.is_dir():
        raise ValueError(f"{DB_PATH_ENV} must be a file path, not a folder")
    return path


class ConversationStore:
    def __init__(
        self,
        path: Path | str,
        *,
        clock=time.time,
        max_messages: int = MAX_MESSAGES,
        retention_seconds: float = RETENTION_SECONDS,
    ):
        if max_messages < 2 or max_messages % 2:
            raise ValueError("max_messages must be an even number of at least 2")
        self.path = Path(path)
        self.clock = clock
        self.max_messages = max_messages
        self.retention_seconds = retention_seconds
        self._purge_lock = threading.Lock()
        self._last_purge: float | None = None

    # --- Connections -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: no implicit transactions; writes use BEGIN IMMEDIATE.
        connection = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _run(self, operation):
        """Run `operation(connection)` on a fresh connection. Database errors
        become StoreError (type name only; never SQL values or message text)."""
        try:
            with closing(self._connect()) as connection:
                return operation(connection)
        except (ConversationNotFound, ConversationFull):
            raise
        except (sqlite3.Error, OSError) as error:
            raise StoreError(f"Conversation database error: {type(error).__name__}") from None

    @staticmethod
    def _write(connection: sqlite3.Connection, operation):
        """Run `operation` in a write transaction (committed, or rolled back on any error)."""
        connection.execute("BEGIN IMMEDIATE")
        try:
            result = operation()
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        return result

    # --- Setup ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the folder, the tables and indexes if needed, switch to WAL
        mode and remove expired conversations. Raises StoreError on failure."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise StoreError(f"Conversation database folder could not be created: {type(error).__name__}") from None

        def setup(connection):
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            connection.execute("SELECT count(*) FROM conversations").fetchone()

        self._run(setup)
        self.purge_expired()

    def ping(self) -> None:
        """Raise StoreError unless the database can be read."""
        self._run(lambda connection: connection.execute("SELECT 1 FROM conversations LIMIT 1").fetchall())

    # --- Reading ---------------------------------------------------------------

    def _cutoff(self, now: float) -> float:
        return now - self.retention_seconds

    @staticmethod
    def _live(connection, conversation_id: str, cutoff: float) -> bool:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND updated_at >= ?", (conversation_id, cutoff)
        ).fetchone()
        return row is not None

    def messages(self, conversation_id: str) -> list[Message]:
        """All messages of a live conversation, oldest first.
        Raises ConversationNotFound for an unknown, invalid or expired ID."""
        if not is_valid_id(conversation_id):
            raise ConversationNotFound()
        cutoff = self._cutoff(self.clock())

        def read(connection):
            if not self._live(connection, conversation_id, cutoff):
                raise ConversationNotFound()
            rows = connection.execute(
                "SELECT role, content, created_at, sources, escalate, reason FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
            return [_message(row) for row in rows]

        return self._run(read)

    def has_room(self, message_count: int) -> bool:
        """Whether a conversation with `message_count` messages can take
        another customer message and reply."""
        return message_count + 2 <= self.max_messages

    # --- Writing ---------------------------------------------------------------

    def add_exchange(
        self,
        conversation_id: str,
        question: str,
        reply: AssistantReply,
        *,
        retrieval_query: str | None = None,
        create: bool = False,
    ) -> int:
        """Save a customer message and the assistant's reply together.

        With `create=True` the conversation is created in the same transaction
        (the ID must be new). Otherwise it must exist and not have expired, and
        it must have room for both messages. Returns the number of messages
        the conversation now holds.
        """
        if not is_valid_id(conversation_id):
            raise ConversationNotFound()
        now = self.clock()
        sources = json.dumps(list(reply.sources), ensure_ascii=False)

        def write(connection):
            def exchange():
                if create:
                    connection.execute(
                        "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                        (conversation_id, now, now),
                    )
                    count = 0
                else:
                    if not self._live(connection, conversation_id, self._cutoff(now)):
                        raise ConversationNotFound()
                    (count,) = connection.execute(
                        "SELECT count(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
                    ).fetchone()
                    if not self.has_room(count):
                        raise ConversationFull()
                    connection.execute(
                        "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
                    )
                connection.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at, retrieval_query) "
                    "VALUES (?, 'customer', ?, ?, ?)",
                    (conversation_id, question, now, retrieval_query),
                )
                connection.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at, sources, escalate, reason) "
                    "VALUES (?, 'assistant', ?, ?, ?, ?, ?)",
                    (conversation_id, reply.answer, now, sources, int(bool(reply.escalate)), reply.reason),
                )
                return count + 2

            return self._write(connection, exchange)

        return self._run(write)

    # --- Retention -------------------------------------------------------------

    def purge_expired(self) -> int:
        """Delete conversations (and their messages) idle for longer than the
        retention period. Returns how many were deleted."""
        now = self.clock()
        cutoff = self._cutoff(now)

        def purge(connection):
            return self._write(
                connection,
                lambda: connection.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,)).rowcount,
            )

        deleted = self._run(purge)
        self._last_purge = now
        return deleted

    def purge_if_due(self) -> int | None:
        """Run `purge_expired` if the last cleanup was more than
        PURGE_INTERVAL_SECONDS ago. Only one thread purges at a time; others
        skip. Returns the number deleted, or None if no cleanup ran."""
        if not self._purge_lock.acquire(blocking=False):
            return None
        try:
            if self._last_purge is not None and self.clock() - self._last_purge < PURGE_INTERVAL_SECONDS:
                return None
            return self.purge_expired()
        finally:
            self._purge_lock.release()


def _message(row) -> Message:
    role, content, created_at, sources, escalate, reason = row
    if role == "assistant":
        return Message(
            role=role, content=content, created_at=created_at,
            sources=json.loads(sources) if sources else [], escalate=bool(escalate), reason=reason,
        )
    return Message(role=role, content=content, created_at=created_at)


def store_from_env(environ: Mapping[str, str] | None = None) -> ConversationStore:
    return ConversationStore(db_path_from_env(environ))
