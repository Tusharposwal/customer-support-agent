"""HTTP API for the customer support agent.

Run locally:
    uvicorn support_agent.api:app --reload
or:
    python -m support_agent.api
"""

import asyncio
import importlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi import Path as PathParameter
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from support_agent import agent, cancellations, conversation_store, feedback, handoffs, log_config, startup
from support_agent.answer import CANCELLATION_AVAILABLE, HANDOFF_MESSAGE
from support_agent.demo_orders import ORDER_ID
from support_agent.conversation_store import (
    AssistantReply,
    ConversationFull,
    ConversationNotFound,
    StoreError,
)
from support_agent.rate_limit import RateLimiter, limiter_from_env

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
# A client-supplied ID is used only if it is short and plain (letters, digits
# and . _ : -), so it can't inject line breaks or junk into the logs.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")
# Where the /chat endpoint leaves its outcome for the request middleware.
OUTCOME_SCOPE_KEY = "support_agent.chat_outcome"
# Reasons and error names are logged only if they look like identifiers.
_SAFE_REASON = re.compile(r"[a-z0-9_]{1,40}")
_SAFE_ERROR_NAME = re.compile(r"[A-Za-z0-9_.]{1,80}")
_LOG_HASH = re.compile(r"[0-9a-f]{12}")
_REASON_BY_STATUS = {404: "not_found", 405: "method_not_allowed", 422: "invalid_request"}

clock = time.perf_counter  # Replaceable in tests.

MAX_QUESTION_LENGTH = 1000
STATIC_DIR = Path(__file__).parent / "static"

# One global limiter for POST /chat (protects the Cohere bill). /health, /ready
# and the page itself are not limited. It is created at startup (see lifespan),
# after .env has been loaded, so CHAT_RATE_LIMIT_* in a local .env applies.
chat_limiter: RateLimiter | None = None
# Conversation storage (SQLite), also created at startup. Until then /chat and
# GET /conversations/{id} answer 503: nothing is answered without being saved.
conversations: conversation_store.ConversationStore | None = None
# Pending and confirmed cancellations, in the same database (cancellations.py).
cancellation_store: cancellations.CancellationStore | None = None
# Hand-off records (internal only: no human-support system is connected) and
# reply ratings, also in the same database (handoffs.py, feedback.py).
handoff_store: handoffs.HandoffStore | None = None
feedback_store: feedback.FeedbackStore | None = None

NOT_READY_MESSAGE = "The assistant is starting up. Please try again in a moment."
CONVERSATION_NOT_FOUND_MESSAGE = (
    "This conversation is no longer available. Please start a new conversation."
)
CONVERSATION_FULL_MESSAGE = (
    "This conversation has reached its message limit. Please start a new conversation."
)
STORAGE_UNAVAILABLE_MESSAGE = (
    "We can't save conversations right now, so your message was not sent. "
    "Please try again in a moment."
)
CONVERSATION_ID_PATTERN = r"^[0-9a-f]{32}$"
ACTION_ID_PATTERN = CONVERSATION_ID_PATTERN  # Also 128 random bits as 32 hex characters.

CANCELLATION_NOT_FOUND_MESSAGE = (
    "This cancellation request was not found, so nothing was cancelled. "
    "Please ask the assistant again if you still want to cancel."
)
CANCELLATION_EXPIRED_MESSAGE = (
    "This cancellation request has expired, so nothing was cancelled. "
    "Please ask the assistant again if you still want to cancel."
)
ORDER_NOT_CANCELLABLE_MESSAGE = (
    "This order can no longer be cancelled because its status has changed. Nothing was cancelled."
)
MESSAGE_NOT_FOUND_MESSAGE = "There is no assistant reply to rate at that position in this conversation."
ORDERS_UNAVAILABLE_MESSAGE = "We can't check orders right now, so nothing was cancelled. Please try again later."

LIMIT_MESSAGES = {
    "minute": "We're receiving a lot of questions right now. Please wait a minute and try again.",
    "day": (
        "The assistant has reached its daily question limit. "
        "Please try again tomorrow or contact our support team."
    ),
}


# /chat requests currently inside the agent (they may be using the Cohere
# clients). Shutdown waits for them before closing the clients.
_chats_in_progress = 0
_chats_lock = threading.Lock()
# How long shutdown waits for running /chat requests. Uvicorn normally lets
# them finish before shutdown starts, so this matters only if it gave up on them.
SHUTDOWN_WAIT_SECONDS = 10.0


@contextmanager
def _chat_in_progress():
    global _chats_in_progress
    with _chats_lock:
        _chats_in_progress += 1
    try:
        yield
    finally:
        with _chats_lock:
            _chats_in_progress -= 1


def _create_clients() -> None:
    """Create the Cohere chat and embedding clients now instead of on the
    first question. Creating a client sends no request; the knowledge base is
    still embedded on first use (that needs a Cohere call)."""
    agent._default_client()
    importlib.import_module("support_agent.semantic_kb")  # Creates the embedding client.


async def _close_clients(wait_seconds: float) -> None:
    """Close the Cohere clients this process created. Never raises, so it
    can't hide the error that ended startup or the server."""
    deadline = time.monotonic() + wait_seconds
    while _chats_in_progress and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    if _chats_in_progress:
        # Closing now would break those requests; the OS frees the
        # connections when the process exits.
        logger.warning("Shutdown: %d chat request(s) still running; Cohere clients left open", _chats_in_progress)
        return

    semantic_kb = sys.modules.get("support_agent.semantic_kb")  # Only if it was imported.
    closers = [("chat", agent.close_default_client)]
    if semantic_kb is not None:
        closers.append(("embedding", semantic_kb.close))
    for name, close in closers:
        try:
            close()
        except Exception as error:
            logger.warning("Shutdown: closing the Cohere %s client failed: %s", name, type(error).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: validate the configuration, set up logging and the rate
    limiter, open the conversation database, create the Cohere clients, then
    report ready. Shutdown: report not ready, then close the clients.

    If validation fails, StartupError is raised: Uvicorn prints it and exits
    ("Application startup failed"), so a broken deploy never goes live. If a
    later step fails, whatever was already created is closed and the original
    error is raised. No Cohere request is made in either direction.

    The conversation database is created if needed (and expired
    conversations removed), with the cancellation, hand-off and feedback
    tables. If it can't be opened, startup fails with StartupError: the app
    never runs without somewhere to save conversations.
    """
    global chat_limiter, conversations, cancellation_store, handoff_store, feedback_store
    app.state.ready = False
    startup.check_startup()  # Loads .env, then validates (CHAT_RATE_LIMIT_*, LOG_LEVEL included).
    log_config.configure_logging(log_config.log_level_from_env())
    chat_limiter = limiter_from_env()  # Same settings that were just validated.
    store = conversation_store.store_from_env()
    actions = cancellations.CancellationStore(store)
    handoff_records = handoffs.HandoffStore(store)
    ratings = feedback.FeedbackStore(store)
    try:
        store.initialize()
        actions.initialize()
        handoff_records.initialize()
        ratings.initialize()
    except conversation_store.StoreError as error:
        raise startup.StartupError(f"The support agent cannot start: {error}") from None
    conversations = store
    cancellation_store = actions
    handoff_store = handoff_records
    feedback_store = ratings
    try:
        _create_clients()
        app.state.ready = True
        logger.info("Startup checks passed; ready for requests")
        yield
    finally:
        app.state.ready = False  # Shutting down (or startup failed): not ready.
        await _close_clients(SHUTDOWN_WAIT_SECONDS)


app = FastAPI(
    title="Customer Support Agent API",
    description="Answers customer questions from the knowledge base, or hands off to a human.",
    version="0.1.0",
    lifespan=lifespan,
)

# The chat page's CSS and JavaScript, served at /static/...
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Request IDs and the /chat outcome log ------------------------------------


def _request_id(scope) -> str:
    """The client's X-Request-ID if it is safe to log, otherwise a new one."""
    for name, value in scope.get("headers", []):
        if name == b"x-request-id":
            supplied = value.decode("latin-1")
            if _VALID_REQUEST_ID.fullmatch(supplied):
                return supplied
            break
    return uuid.uuid4().hex


def _safe(value, pattern: re.Pattern, fallback: str) -> str:
    return value if isinstance(value, str) and pattern.fullmatch(value) else fallback


def _log_chat_outcome(status: int, outcome: dict, duration_ms: float) -> None:
    """One line per /chat request. Never the question, answer, URLs or error text."""
    escalate = outcome.get("escalate")
    fields = {
        "request_id": log_config.request_id_var.get(),
        "status": status,
        "escalate": "-" if escalate is None else str(bool(escalate)).lower(),
        "reason": _safe(outcome.get("reason"), _SAFE_REASON, "other")
        if "reason" in outcome
        else _REASON_BY_STATUS.get(status, f"http_{status}"),
        "sources": outcome.get("sources", 0),
        "duration_ms": duration_ms,
    }
    # Conversation fields: a one-way hash of the ID (never the ID itself, which
    # grants access to the conversation), the turn number and how the
    # question was rewritten for retrieval.
    if "conversation" in outcome:
        fields["conversation"] = _safe(outcome["conversation"], _LOG_HASH, "-")
    if "turn" in outcome:
        turn = outcome["turn"]
        fields["turn"] = turn if isinstance(turn, int) and not isinstance(turn, bool) else "-"
    if "rewrite" in outcome:
        fields["rewrite"] = _safe(outcome["rewrite"], _SAFE_REASON, "other")
    if "limit" in outcome:
        fields["limit"] = outcome["limit"]
    if "error" in outcome:
        fields["error"] = _safe(outcome["error"], _SAFE_ERROR_NAME, "Exception")

    # 429 and 503 are expected pushback (warning); other 5xx are failures (error).
    if status in (429, 503):
        level = logging.WARNING
    elif status >= 500:
        level = logging.ERROR
    else:
        level = logging.INFO
    # request_id is already in the line prefix, so it isn't repeated here.
    text = " ".join(f"{key}={value}" for key, value in fields.items() if key != "request_id")
    logger.log(level, "chat_request %s", text, extra={"chat_outcome": fields})


# The middleware's last-resort body, for a /chat failure outside the
# endpoint's own error handling. Nothing was saved then (no conversation, no
# hand-off record), so it must not use HANDOFF_MESSAGE, which says the
# question was recorded. It has no conversation_id and no message_index:
# that is how a client can tell it was not saved.
UNSAVED_ERROR_MESSAGE = (
    "Something went wrong on our side, so your message was not saved. Please try again in a moment."
)
_SAFE_500_BODY = json.dumps(
    {"answer": UNSAVED_ERROR_MESSAGE, "sources": [], "escalate": True, "reason": "server_error"}
).encode()


class RequestContextMiddleware:
    """For every HTTP request: pick a request ID, make it available to log
    lines (log_config.request_id_var) and return it in X-Request-ID.

    For /chat, also log exactly one outcome line when the request ends,
    including requests rejected before the agent runs (422, 429, 503) and
    unexpected exceptions, which get the usual safe 500 hand-off body.
    A plain ASGI middleware (not BaseHTTPMiddleware), so the endpoint's
    response is passed through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        token = log_config.request_id_var.set(request_id)
        is_chat = scope.get("path") == "/chat"
        outcome: dict = {}
        scope[OUTCOME_SCOPE_KEY] = outcome
        status = None
        start = clock()

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            try:
                await self.app(scope, receive, send_with_id)
            except Exception as error:
                if not is_chat or status is not None:
                    raise  # Not /chat, or the response already started: the server handles it.
                outcome.update(reason="server_error", escalate=True, sources=0, error=type(error).__name__)
                await send_with_id({
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(_SAFE_500_BODY)).encode())],
                })
                await send_with_id({"type": "http.response.body", "body": _SAFE_500_BODY})
        finally:
            if is_chat:
                if status is None:  # No response was sent at all.
                    status = 500
                _log_chat_outcome(status, outcome, round((clock() - start) * 1000, 1))
            log_config.request_id_var.reset(token)


app.add_middleware(RequestContextMiddleware)


def _record_outcome(http_request: Request, **fields) -> None:
    """Tell the middleware how this /chat request ended (logged once, there)."""
    outcome = http_request.scope.get(OUTCOME_SCOPE_KEY)
    if outcome is not None:
        outcome.update(fields)


class ChatRequest(BaseModel):
    # Leading/trailing spaces are removed first, so "   " counts as empty.
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_LENGTH),
    ]
    # Omitted (or null) for the first message: the server starts a new
    # conversation and returns its ID. Only IDs this server generates have
    # this shape; anything else is rejected with 422 before any lookup.
    conversation_id: Annotated[str, StringConstraints(pattern=CONVERSATION_ID_PATTERN)] | None = None


class Source(BaseModel):
    id: str
    title: str
    url: str


class PendingCancellation(BaseModel):
    """Returned only with the "cancellation_available" reply. `action_id` is
    a secret for the customer's browser: it is never stored in a message,
    shown to the model or logged, and only POST /cancellations/confirm uses it."""

    action_id: str
    order_id: str
    expires_at: str  # ISO 8601, UTC


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    escalate: bool
    reason: str
    conversation_id: str
    # The reply's position in the conversation (0 = first message): what
    # POST /feedback takes. Set once the reply is saved.
    message_index: int | None = None
    pending_cancellation: PendingCancellation | None = None


class ConfirmCancellationRequest(BaseModel):
    # Both IDs travel in the body, never the URL, so they can't end up in
    # access logs. Anything that isn't 32 lowercase hex characters is a 422.
    conversation_id: Annotated[str, StringConstraints(pattern=CONVERSATION_ID_PATTERN)]
    action_id: Annotated[str, StringConstraints(pattern=ACTION_ID_PATTERN)]


class ConfirmCancellationResponse(BaseModel):
    status: str  # "cancelled"
    order_id: str
    cancelled_at: str  # ISO 8601, UTC
    already_confirmed: bool  # True when this repeats an earlier confirmation
    message: str


class CancellationStatus(BaseModel):
    """On a restored "can be cancelled" reply: what became of the offer.
    Never the action ID or its hash: a page that still holds the ID (it keeps
    it in sessionStorage) can confirm a pending offer; nobody else can."""

    order_id: str
    status: str  # "pending", "cancelled" or "expired"
    expires_at: str  # ISO 8601, UTC; same value /chat returned
    cancelled_at: str | None = None


class ConversationMessage(BaseModel):
    role: str  # "customer" or "assistant"
    content: str
    created_at: str  # ISO 8601, UTC
    message_index: int  # Position in the conversation (0 = first); POST /feedback takes it.
    # Assistant messages only (empty / null for the customer's).
    sources: list[Source] = []
    escalate: bool | None = None
    reason: str | None = None
    cancellation: CancellationStatus | None = None  # Only on a cancellation offer.
    feedback: str | None = None  # "positive" / "negative" once the customer rated the reply.


class HandoffStatus(BaseModel):
    """The conversation needs a human. An internal record only: no human
    support system is connected, so nobody has been notified or assigned."""

    status: str  # "open"
    category: str  # "no_answer", "service_problem" or "other"
    created_at: str  # ISO 8601, UTC
    updated_at: str


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage]
    message_limit: int
    handoff: HandoffStatus | None = None


class FeedbackRequest(BaseModel):
    # Unknown fields (a comment, a name, an email...) are refused (422), not
    # silently dropped: only the rating is ever collected.
    model_config = ConfigDict(extra="forbid")

    conversation_id: Annotated[str, StringConstraints(pattern=CONVERSATION_ID_PATTERN)]
    message_index: Annotated[int, Field(strict=True, ge=0, le=10_000)]
    rating: Literal["positive", "negative"]  # Nothing else is collected: no comment, no contact details.


class FeedbackResponse(BaseModel):
    message_index: int
    rating: str
    changed: bool  # False when this rating was already saved (nothing changed)


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


def _detail(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": message})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The customer-facing chat page. It calls POST /chat from the browser."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Liveness check. Does not call Cohere."""
    return {"status": "ok"}


@app.get(
    "/ready",
    responses={503: {"description": "Startup has not completed, or the server is shutting down."}},
)
def ready(request: Request):
    """Readiness check: 200 only once startup has completed, i.e. the
    configuration passed validation and the rate limiter and Cohere clients
    were created (see `lifespan`); 503 before that and once shutdown begins.

    Reads a flag set by `lifespan`, so it is instant and makes no Cohere call
    or embedding request. It doesn't check that Cohere accepts the key or is
    reachable.
    """
    if getattr(request.app.state, "ready", False):
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "not_ready"})


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    responses={
        404: {"description": "No such conversation, or it has expired."},
        503: {"description": "Startup has not completed, or conversations can't be read right now."},
    },
)
def get_conversation(
    conversation_id: Annotated[str, PathParameter(pattern=CONVERSATION_ID_PATTERN)],
):
    """The messages of a conversation, oldest first, so the chat page can
    show them again after a reload. The ID is the only key: anyone who has it
    can read the conversation. Makes no Cohere call. Rewritten search
    queries are internal and not returned.

    A "can be cancelled" reply carries `cancellation`: the order, whether the
    offer is pending, cancelled or expired, and when it expires. Action IDs
    are never returned (the server keeps only their hashes).

    Each message has its `message_index`, and a rated reply its `feedback`.
    `handoff` is the open hand-off record, if the agent escalated: an
    internal record only (no human-support system is connected)."""
    store = conversations
    actions = cancellation_store
    records = handoff_store
    ratings = feedback_store
    if store is None or actions is None or records is None or ratings is None:
        return _detail(503, NOT_READY_MESSAGE)
    try:
        messages = store.messages(conversation_id)
        offers = actions.offers(conversation_id)
        handoff = records.open_handoff(conversation_id)
        rated = ratings.ratings(conversation_id)
    except ConversationNotFound:
        return _detail(404, CONVERSATION_NOT_FOUND_MESSAGE)
    except StoreError as error:
        logger.warning("Reading a conversation failed: %s", error)
        return _detail(503, STORAGE_UNAVAILABLE_MESSAGE)
    # An offer and the reply that made it were saved in one transaction with
    # the same timestamp, which is how they are matched here.
    offer_by_time = {offer.created_at: offer for offer in offers}

    def cancellation(message):
        offer = offer_by_time.get(message.created_at) if message.reason == CANCELLATION_AVAILABLE else None
        if offer is None:
            return None
        return CancellationStatus(
            order_id=offer.order_id, status=offer.status, expires_at=_iso(offer.expires_at),
            cancelled_at=None if offer.cancelled_at is None else _iso(offer.cancelled_at),
        )

    return ConversationResponse(
        conversation_id=conversation_id,
        messages=[
            ConversationMessage(
                role=m.role, content=m.content, created_at=_iso(m.created_at), message_index=index,
                sources=m.sources, escalate=m.escalate, reason=m.reason, cancellation=cancellation(m),
                feedback=rated.get(index),
            )
            for index, m in enumerate(messages)
        ],
        message_limit=store.max_messages,
        handoff=None if handoff is None else HandoffStatus(
            status=handoff.status, category=handoff.category,
            created_at=_iso(handoff.created_at), updated_at=_iso(handoff.updated_at),
        ),
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    # `pending_cancellation` appears only with a "can be cancelled" reply;
    # every other response keeps exactly its earlier shape.
    response_model_exclude_none=True,
    responses={
        404: {"description": "The conversation does not exist or has expired."},
        409: {"description": "The conversation has reached its message limit."},
        429: {"description": "Request limit reached; see the Retry-After header."},
        503: {"description": "Startup has not completed, or conversations can't be saved right now."},
    },
)
def chat(request: ChatRequest, http_request: Request):
    """Answer a customer question using the existing support agent.

    Without `conversation_id`, a new conversation is started. With one, the
    question is a follow-up: earlier messages are used only to work out what
    the customer means (see query_rewrite), never as evidence.

    Every reply is saved, together with the question, before it is returned.
    If saving fails, the reply is not returned (503): the customer is never
    shown a message the conversation did not keep.
    """
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so the
    # blocking Cohere and SQLite calls do not freeze the server. Outcomes are
    # recorded with _record_outcome and logged once by RequestContextMiddleware.

    limiter = chat_limiter
    store = conversations
    actions = cancellation_store
    records = handoff_store
    if limiter is None or store is None or actions is None or records is None:
        # Startup (validation, limiter, database) hasn't run: never call Cohere
        # unlimited, or answer without being able to save.
        _record_outcome(http_request, reason="not_ready")
        return _detail(503, NOT_READY_MESSAGE)

    # The conversation is checked first: an unknown, expired or full one is
    # refused before it uses quota or reaches Cohere.
    is_new = request.conversation_id is None
    conversation_id = conversation_store.new_conversation_id() if is_new else request.conversation_id
    _record_outcome(http_request, conversation=conversation_store.log_hash(conversation_id))
    try:
        if is_new:
            store.ping()  # Don't spend a Cohere call on an answer that couldn't be saved.
            history = []
        else:
            history = store.messages(conversation_id)
    except ConversationNotFound:
        _record_outcome(http_request, reason="conversation_not_found")
        return _detail(404, CONVERSATION_NOT_FOUND_MESSAGE)
    except StoreError as error:
        logger.warning("Conversation storage unavailable: %s", error)
        _record_outcome(http_request, reason="storage_unavailable", error=type(error).__name__)
        return _detail(503, STORAGE_UNAVAILABLE_MESSAGE)
    if not store.has_room(len(history)):
        _record_outcome(http_request, reason="conversation_full")
        return _detail(409, CONVERSATION_FULL_MESSAGE)

    # Checked after validation, so invalid (422) requests don't use up the quota.
    limited = limiter.check()
    if limited is not None:
        limit_name, retry_after = limited
        _record_outcome(http_request, reason="rate_limited", limit=limit_name)
        return JSONResponse(
            status_code=429,
            content={"detail": LIMIT_MESSAGES[limit_name]},
            headers={"Retry-After": str(retry_after)},
        )

    trace: dict = {}  # Filled in by the agent: the retrieval query and the rewrite outcome.
    status_code = 200
    try:
        with _chat_in_progress():
            result = agent.answer_question(
                request.question,
                history=[{"role": m.role, "content": m.content} for m in history],
                trace=trace,
                is_cancelled=actions.is_cancelled,  # Tools see confirmed cancellations.
            )
        response = ChatResponse(
            answer=result.answer,
            sources=result.sources,
            escalate=result.escalate,
            reason=result.reason,
            conversation_id=conversation_id,
        )
    except Exception as error:
        # Never send exception details or stack traces to the customer.
        _record_outcome(http_request, error=type(error).__name__)
        status_code = 500
        response = ChatResponse(
            answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason="server_error",
            conversation_id=conversation_id,
        )

    # The customer's message and the reply are saved together; the rewritten
    # query is stored in its own column, apart from the customer's text.
    reply = AssistantReply(
        answer=response.answer,
        sources=[source.model_dump() for source in response.sources],
        escalate=response.escalate,
        reason=response.reason,
    )
    # A "can be cancelled" reply is saved with a pending cancellation, in the
    # same transaction: the reply is never stored without its action.
    record_action = created_action = None
    cancellable = trace.get("cancellable_order_id")
    if response.reason == CANCELLATION_AVAILABLE and isinstance(cancellable, str) and ORDER_ID.fullmatch(cancellable):
        record_action, created_action = actions.proposer(conversation_id, cancellable)
    # Likewise, an escalated reply is saved with the conversation's hand-off
    # record (opened, or updated if one is already open).
    if response.escalate:
        record_action = records.recorder(conversation_id, response.reason)
    try:
        message_count = store.add_exchange(
            conversation_id, request.question, reply,
            retrieval_query=trace.get("retrieval_query"), create=is_new, in_transaction=record_action,
        )
    except ConversationNotFound:  # Expired between reading and saving.
        _record_outcome(http_request, reason="conversation_not_found")
        return _detail(404, CONVERSATION_NOT_FOUND_MESSAGE)
    except ConversationFull:  # Another request filled it in the meantime.
        _record_outcome(http_request, reason="conversation_full")
        return _detail(409, CONVERSATION_FULL_MESSAGE)
    except StoreError as error:
        logger.warning("Saving the conversation failed: %s", error)
        _record_outcome(http_request, reason="storage_unavailable", error=type(error).__name__)
        return _detail(503, STORAGE_UNAVAILABLE_MESSAGE)

    try:
        store.purge_if_due()
    except StoreError as error:  # Retried later; the answer is already saved.
        logger.warning("Removing expired conversations failed: %s", error)

    _record_outcome(
        http_request, reason=response.reason, escalate=response.escalate, sources=len(response.sources),
        turn=message_count // 2, rewrite=trace.get("rewrite", "none"),
    )
    response.message_index = message_count - 1  # The reply is the conversation's last message.
    if created_action is not None:
        pending = created_action()
        response.pending_cancellation = PendingCancellation(
            action_id=pending.action_id, order_id=pending.order_id, expires_at=_iso(pending.expires_at),
        )
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=response.model_dump(exclude_none=True))
    return response


@app.post(
    "/cancellations/confirm",
    response_model=ConfirmCancellationResponse,
    responses={
        404: {"description": "No such cancellation request in this conversation."},
        409: {"description": "The order can no longer be cancelled."},
        410: {"description": "The cancellation request has expired."},
        503: {"description": "Startup has not completed, or orders or storage can't be used right now."},
    },
)
def confirm_cancellation(request: ConfirmCancellationRequest):
    """Cancel an order: the customer's explicit confirmation of a pending
    cancellation that /chat returned (`pending_cancellation`).

    The only way anything is cancelled. Needs the conversation ID and the
    action ID together; an unknown action and another conversation's action
    get the same 404. Confirming again returns the same result with
    `already_confirmed: true` and changes nothing. Makes no Cohere call, so
    it is not rate limited. See cancellations.py.
    """
    actions = cancellation_store
    if actions is None:
        return _detail(503, NOT_READY_MESSAGE)
    label = conversation_store.log_hash(request.conversation_id)

    def log(outcome: str) -> None:
        # Never the action ID, the conversation ID or the order number.
        logger.info("cancellation_confirm outcome=%s conversation=%s", outcome, label)

    try:
        order_status = agent._default_order_book().status
    except Exception as error:
        logger.warning("Demo orders could not be loaded: %s", type(error).__name__)
        log("orders_unavailable")
        return _detail(503, ORDERS_UNAVAILABLE_MESSAGE)
    try:
        confirmation = actions.confirm(request.conversation_id, request.action_id, order_status)
    except cancellations.ActionNotFound:
        log("not_found")
        return _detail(404, CANCELLATION_NOT_FOUND_MESSAGE)
    except cancellations.ActionExpired:
        log("expired")
        return _detail(410, CANCELLATION_EXPIRED_MESSAGE)
    except cancellations.OrderNotCancellable:
        log("not_cancellable")
        return _detail(409, ORDER_NOT_CANCELLABLE_MESSAGE)
    except StoreError as error:
        logger.warning("Confirming a cancellation failed: %s", error)
        log("storage_unavailable")
        return _detail(503, STORAGE_UNAVAILABLE_MESSAGE)
    log("already_confirmed" if confirmation.already_confirmed else "cancelled")
    return ConfirmCancellationResponse(
        status="cancelled",
        order_id=confirmation.order_id,
        cancelled_at=_iso(confirmation.cancelled_at),
        already_confirmed=confirmation.already_confirmed,
        message=f"Order {confirmation.order_id} has been cancelled.",
    )


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={
        404: {"description": "No such conversation, or no assistant reply at that position."},
        503: {"description": "Startup has not completed, or feedback can't be saved right now."},
    },
)
def submit_feedback(request: FeedbackRequest):
    """Rate one assistant reply as helpful ("positive") or not ("negative").

    The reply is named by its position in the conversation, and only a reply
    of *this* conversation can be rated. One rating per reply: the same
    rating again changes nothing (`changed: false`), a different one replaces
    it. Only the rating is stored. Makes no Cohere call. See feedback.py.
    """
    ratings = feedback_store
    if ratings is None:
        return _detail(503, NOT_READY_MESSAGE)
    label = conversation_store.log_hash(request.conversation_id)
    try:
        result = ratings.submit(request.conversation_id, request.message_index, request.rating)
    except ConversationNotFound:
        logger.info("feedback outcome=conversation_not_found conversation=%s", label)
        return _detail(404, CONVERSATION_NOT_FOUND_MESSAGE)
    except feedback.MessageNotFound:
        logger.info("feedback outcome=message_not_found conversation=%s", label)
        return _detail(404, MESSAGE_NOT_FOUND_MESSAGE)
    except StoreError as error:
        logger.warning("Saving feedback failed: %s", error)
        return _detail(503, STORAGE_UNAVAILABLE_MESSAGE)
    logger.info("feedback outcome=%s rating=%s conversation=%s",
                "recorded" if result.changed else "unchanged", result.rating, label)
    return FeedbackResponse(message_index=result.message_index, rating=result.rating, changed=result.changed)


def main() -> None:
    """Start the API server. HOST and PORT can be set as environment variables."""
    import uvicorn

    uvicorn.run(
        "support_agent.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
