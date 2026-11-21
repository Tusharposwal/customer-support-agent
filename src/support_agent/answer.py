"""Grounded answer generation with Cohere Chat.

The model only sees the retrieved KB articles (passed as Cohere `documents`).
An answer is shown to the customer only if the model finished normally, cites
at least one of those articles and cites nothing else. In every other case we
hand the question to a human.
"""

import logging
import os
import re
from dataclasses import dataclass, field

from support_agent.cohere_calls import call_with_retries, chat_timeout, describe_error

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "command-a-03-2025"
MAX_TOKENS = 300
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

# The model may write the marker as markdown ("INSUFFICIENT\_EVIDENCE"), with a
# space, or in another case. Any of these means "the documents don't answer it".
_INSUFFICIENT_PATTERN = re.compile(r"insufficient[\s_\\-]*evidence", re.IGNORECASE)

# Only these finish reasons mean the model finished its answer normally.
_COMPLETE_FINISH_REASONS = {"COMPLETE", "STOP_SEQUENCE"}
# The model stopped because of a failure on Cohere's side.
_FAILED_FINISH_REASONS = {"ERROR", "TIMEOUT"}

# Sent when a follow-up can't be understood from the conversation. Not a
# hand-off and not an answer: no sources, nothing escalated.
CLARIFICATION_MESSAGE = (
    "I'm not sure what you're referring to. Could you tell me a bit more, "
    "for example which order, payment or account setting your question is about?"
)

# Never claims that a person has been contacted or assigned: hand-offs are
# recorded internally (handoffs.py), and no human-support system is connected.
HANDOFF_MESSAGE = (
    "I'm not able to answer this confidently from our help articles, so I've "
    "recorded your question as needing help from our support team."
)

SYSTEM_PROMPT = f"""You are a customer support assistant.
Answer the customer's question using ONLY the information in the provided documents.
Do not use outside knowledge and do not guess.
If the documents do not contain enough information to answer the question,
reply with exactly: {INSUFFICIENT_EVIDENCE}
Keep answers short, clear and polite."""


@dataclass
class SupportAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)  # each: {"id", "title", "url"}
    escalate: bool = False
    reason: str = "answered"


def _escalation(reason: str) -> SupportAnswer:
    """Safe hand-off response. Never includes model output or error details."""
    return SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason=reason)


def _to_document(article: dict) -> dict:
    return {
        "id": article["id"],
        "data": {
            "title": article["title"],
            "text": article["text"],
            "url": article["url"],
        },
    }


def _extract_text(response) -> str | None:
    """Join the text parts of `response.message.content`; None if malformed."""
    content = getattr(getattr(response, "message", None), "content", None)
    if not isinstance(content, (list, tuple)):
        return None

    parts = [
        item.text
        for item in content
        if getattr(item, "type", None) == "text"
        and isinstance(getattr(item, "text", None), str)
    ]
    text = "".join(parts).strip()
    return text or None


def _extract_cited_ids(response) -> list[str]:
    """Return document IDs cited in the answer text, in order, without duplicates."""
    citations = getattr(getattr(response, "message", None), "citations", None) or []
    cited_ids = []

    for citation in citations:
        # Citations on the model's hidden "thinking" are not evidence for the answer.
        if getattr(citation, "type", None) not in (None, "TEXT_CONTENT"):
            continue

        for source in getattr(citation, "sources", None) or []:
            source_id = getattr(source, "id", None)
            if getattr(source, "type", None) == "document" and source_id not in cited_ids:
                cited_ids.append(source_id)

    return cited_ids


def user_message(question: str, standalone_query: str | None = None) -> str:
    """The user turn sent to the model. A first message (or one that needed
    no rewriting) is sent exactly as before. For a rewritten follow-up, the
    standalone question is added so the model knows what is being asked; it
    is labelled as not being a source of facts. Earlier assistant replies are
    never sent here."""
    if not standalone_query or standalone_query == question:
        return question
    return (
        f"Customer's latest message: {question}\n\n"
        "The same question rewritten to stand on its own, using the earlier "
        f"conversation (this is not a source of facts): {standalone_query}"
    )


def generate_answer(
    question: str,
    articles: list[dict],
    client,
    model: str | None = None,
    *,
    standalone_query: str | None = None,
) -> SupportAnswer:
    """Answer `question` from `articles` using a Cohere ClientV2-compatible `client`.

    `standalone_query` is the rewritten form of a follow-up (see
    query_rewrite); it only clarifies the question. The documents remain the
    only allowed source of facts and the citation checks are unchanged.
    """
    if not question or not question.strip():
        return _escalation("empty_question")

    if not articles:
        return _escalation("no_articles")

    model = model or os.getenv("COHERE_CHAT_MODEL") or DEFAULT_MODEL
    # Built before the API call: a bad article is a data problem, not an LLM error.
    articles_by_id = {article["id"]: article for article in articles}
    documents = [_to_document(article) for article in articles]

    timeout_seconds = chat_timeout()

    def request(request_options):
        return client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(question, standalone_query)},
            ],
            documents=documents,
            temperature=0,
            max_tokens=MAX_TOKENS,
            request_options=request_options,
        )

    try:
        # Bounded time, at most one retry for a temporary failure (see cohere_calls).
        response = call_with_retries(request, operation="chat", timeout_seconds=timeout_seconds)
    except Exception as error:  # Any API/network failure -> human hand-off.
        logger.warning("Cohere chat call failed: %s", describe_error(error))
        return _escalation("llm_error")

    try:
        return _read_response(response, articles_by_id)
    except Exception as error:
        # A response with an unexpected shape (e.g. citations that aren't a list)
        # is a malformed response, the same as one with no text.
        logger.warning("Cohere chat response could not be read: %s", type(error).__name__)
        return _escalation("malformed_response")


def _read_response(response, articles_by_id: dict[str, dict]) -> SupportAnswer:
    """Turn a chat response into an answer or a hand-off."""
    # Checked first: text from an unfinished generation must never be shown.
    finish_reason = getattr(response, "finish_reason", None)
    if finish_reason == "MAX_TOKENS":
        return _escalation("incomplete_response")
    if finish_reason in _FAILED_FINISH_REASONS:
        logger.warning("Cohere chat generation stopped early: %s", finish_reason)
        return _escalation("llm_error")
    if finish_reason not in _COMPLETE_FINISH_REASONS:
        logger.warning("Cohere chat response has unexpected finish reason: %.40r", finish_reason)
        return _escalation("malformed_response")

    text = _extract_text(response)
    if text is None:
        logger.warning("Cohere chat response had no usable text content")
        return _escalation("malformed_response")

    if _INSUFFICIENT_PATTERN.search(text):
        return _escalation("insufficient_evidence")

    cited_ids = _extract_cited_ids(response)
    if any(article_id not in articles_by_id for article_id in cited_ids):
        # Text attributed to a document we never supplied is not grounded in
        # our knowledge base, so the whole answer is rejected.
        logger.warning("Cohere chat response cited a document that was not supplied")
        return _escalation("invalid_citations")

    if not cited_ids:
        return _escalation("no_citations")

    sources = [
        {
            "id": article_id,
            "title": articles_by_id[article_id]["title"],
            "url": articles_by_id[article_id]["url"],
        }
        for article_id in cited_ids
    ]
    return SupportAnswer(answer=text, sources=sources, escalate=False, reason="answered")


# --- Tool calling (Phase 5, Milestone 3.3) -------------------------------------------
#
# Used only when the agent offers the order tools (the customer has written an
# order number). The cycle, at most two chat calls:
#
#   1. chat(messages, documents=<KB articles>, tools=<registry definitions>)
#      - finish_reason COMPLETE: an ordinary answer, checked exactly like
#        generate_answer's (articles only).
#      - finish_reason TOOL_CALL: the calls are checked and run once through
#        tools.execute_all (at most 3; one result per call ID).
#   2. chat(messages + [the model's own tool-call message, unchanged]
#                    + [one tool message per call ID],
#           documents=<same KB articles>, tools=<same>, tool_choice="NONE")
#      - must be a finished answer citing supplied articles and/or this
#        question's tool results.
#
# Some outcomes are written by the server instead of the model: a request
# for the order number and email, and the "can be cancelled" reply, which
# must never be worded by a model because it is about a state change.
# Both chat calls share one time budget (COHERE_CHAT_TIMEOUT_SECONDS), so a
# question takes no longer than before.

from support_agent import cohere_calls, tools  # noqa: E402

TOOL_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You also have order tools. Use one only when the customer asks about a
specific order and has written both the order number and the email address
used for the order. Tool results are the only source of facts about orders:
report statuses, dates and refund decisions exactly as the tools give them,
and never decide eligibility yourself. You cannot cancel orders: never say
you have cancelled an order or that one will be cancelled. Say an order is
cancelled only when a tool result gives its status as cancelled."""

NEEDS_ORDER_DETAILS = "needs_order_details"
CANCELLATION_AVAILABLE = "cancellation_available"
NEEDS_ORDER_DETAILS_MESSAGE = (
    "To look up an order, please send both your order number (for example ORD-1234) "
    "and the email address you used when you placed the order."
)
# The second call needs at least this much of the shared budget left.
MIN_FINAL_CALL_SECONDS = 5.0

# Tool results that can't be shown to the customer as a source.
_UNCITABLE = {tools.LIMIT_EXCEEDED, tools.UNKNOWN_TOOL, tools.INVALID_ARGUMENTS, tools.ERROR,
              tools.DETAILS_NOT_PROVIDED}
_TOOL_LABELS = {
    "get_order_status": "order status",
    "check_refund_eligibility": "refund check",
    "request_order_cancellation": "cancellation check",
}


def cancellation_message(order_id: str) -> str:
    """Server-written reply when an order can still be cancelled (KB-024).
    Nothing has been cancelled. It never asks the customer to type a
    confirmation: only POST /cancellations/confirm cancels (cancellations.py),
    using the action ID /chat returns alongside this reply."""
    return (
        f"Order {order_id} is still processing, so it can be cancelled, but it has not been "
        "cancelled yet. Cancelling needs your confirmation: choose Confirm cancellation below "
        "to cancel it, or Keep order to leave it as it is."
    )


def tool_source(result) -> dict:
    """How a tool result appears in `sources`: the customer's own order, no link."""
    order_id = result.data.get("order_id")
    label = _TOOL_LABELS.get(result.name, "order lookup")
    if isinstance(order_id, str):
        return {"id": f"order:{order_id}:{result.name}", "title": f"Your order {order_id} ({label})", "url": ""}
    return {"id": f"tool:{result.name}", "title": f"Order lookup ({label})", "url": ""}


def citable_tool_sources(results) -> list[dict]:
    """The tool sources an answer about this question may show."""
    return [tool_source(r) for r in results if r.status not in _UNCITABLE]


@dataclass
class ToolRound:
    """The outcome of a tool-enabled answer, with this question's tool results
    (the only tool results any citation may refer to)."""

    answer: SupportAnswer
    results: list = field(default_factory=list)


def _article_source(article: dict) -> dict:
    return {"id": article["id"], "title": article["title"], "url": article["url"]}


def _tool_calls(response) -> tuple[list, dict]:
    """The requested calls, and the model's own tool-call message to send back
    unchanged (its tool plan and call IDs). Raises ValueError if malformed."""
    message = getattr(response, "message", None)
    raw_calls = getattr(message, "tool_calls", None)
    if not isinstance(raw_calls, (list, tuple)) or not raw_calls:
        raise ValueError("TOOL_CALL without tool calls")
    calls, echoed = [], []
    for raw in raw_calls:
        call_id = getattr(raw, "id", None)
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("tool call without an ID")
        function = getattr(raw, "function", None)
        name, arguments = getattr(function, "name", None), getattr(function, "arguments", None)
        calls.append(tools.ToolCall(id=call_id, name=name, arguments=arguments))
        echoed.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
    if len({c.id for c in calls}) != len(calls):
        raise ValueError("duplicate tool call IDs")
    assistant = {"role": "assistant", "tool_calls": echoed}
    plan = getattr(message, "tool_plan", None)
    if isinstance(plan, str) and plan:
        assistant["tool_plan"] = plan
    return calls, assistant


def _tool_message(result) -> dict:
    return {
        "role": "tool",
        "tool_call_id": result.call_id,
        "content": [{"type": "document", "document": result.as_document()}],
    }


def _citations(response) -> list[tuple[str, str]]:
    """(source type, id) for every source cited in the answer text, in order,
    without duplicates. Only "document" and "tool" sources are kept."""
    citations = getattr(getattr(response, "message", None), "citations", None) or []
    cited = []
    for citation in citations:
        if getattr(citation, "type", None) not in (None, "TEXT_CONTENT"):
            continue  # Citations on hidden reasoning are not evidence.
        for source in getattr(citation, "sources", None) or []:
            entry = (getattr(source, "type", None), getattr(source, "id", None))
            if entry[0] in ("document", "tool") and entry not in cited:
                cited.append(entry)
    return cited


def _matching_result(source_id, results):
    """The tool result a citation refers to: by our document ID, the call ID,
    or "<call ID>:<n>". Only results produced in this question can match."""
    if not isinstance(source_id, str):
        return None
    for result in results:
        if result.status in _UNCITABLE:
            continue
        if source_id in (result.document_id, result.call_id) or re.fullmatch(
            re.escape(result.call_id) + r":\d+", source_id
        ):
            return result
    return None


def _read_final_response(response, articles_by_id: dict[str, dict], results) -> SupportAnswer:
    """The answer after tool results: finished, no further tool calls, and
    every citation is a supplied article or one of this question's results."""
    finish_reason = getattr(response, "finish_reason", None)
    if finish_reason == "MAX_TOKENS":
        return _escalation("incomplete_response")
    if finish_reason in _FAILED_FINISH_REASONS:
        logger.warning("Cohere chat generation stopped early: %s", finish_reason)
        return _escalation("llm_error")
    if finish_reason not in _COMPLETE_FINISH_REASONS:
        logger.warning("Final tool answer has unexpected finish reason: %.40r", finish_reason)
        return _escalation("malformed_response")
    if getattr(getattr(response, "message", None), "tool_calls", None):
        logger.warning("Model asked for more tools after tool_choice NONE")
        return _escalation("malformed_response")

    text = _extract_text(response)
    if text is None:
        return _escalation("malformed_response")
    if _INSUFFICIENT_PATTERN.search(text):
        return _escalation("insufficient_evidence")

    sources = []
    for source_type, source_id in _citations(response):
        result = _matching_result(source_id, results)
        if result is not None and source_id not in articles_by_id:
            source = tool_source(result)
        elif source_type == "document" and source_id in articles_by_id:
            source = _article_source(articles_by_id[source_id])
        else:
            # An article we never supplied, or a tool result from no call in
            # this question (e.g. an earlier turn): not grounded.
            logger.warning("Final tool answer cited a source that was not supplied")
            return _escalation("invalid_citations")
        if source not in sources:
            sources.append(source)
    if not sources:
        return _escalation("no_citations")
    return SupportAnswer(answer=text, sources=sources, escalate=False, reason="answered")


def generate_answer_with_tools(
    question: str,
    articles: list[dict],
    client,
    tool_context,
    *,
    model: str | None = None,
    standalone_query: str | None = None,
) -> ToolRound:
    """Answer with the KB articles and the order tools. Never raises for API
    or model problems: those become hand-offs, as in generate_answer."""
    if not question or not question.strip():
        return ToolRound(_escalation("empty_question"))

    model = model or os.getenv("COHERE_CHAT_MODEL") or DEFAULT_MODEL
    articles_by_id = {article["id"]: article for article in articles}
    documents = [_to_document(article) for article in articles]
    definitions = tools.definitions()
    budget = chat_timeout()
    started = cohere_calls.clock()
    messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_message(question, standalone_query)},
    ]

    def first_call(request_options):
        return client.chat(
            model=model, messages=messages, documents=documents, tools=definitions,
            temperature=0, max_tokens=MAX_TOKENS, request_options=request_options,
        )

    try:
        response = call_with_retries(first_call, operation="chat", timeout_seconds=budget)
    except Exception as error:
        logger.warning("Cohere chat call failed: %s", describe_error(error))
        return ToolRound(_escalation("llm_error"))

    if getattr(response, "finish_reason", None) != "TOOL_CALL":
        try:  # No tools requested: the ordinary answer rules, articles only.
            return ToolRound(_read_response(response, articles_by_id))
        except Exception as error:
            logger.warning("Cohere chat response could not be read: %s", type(error).__name__)
            return ToolRound(_escalation("malformed_response"))

    try:
        calls, assistant_message = _tool_calls(response)
    except Exception as error:
        logger.warning("Malformed tool-call response: %s", type(error).__name__)
        return ToolRound(_escalation("malformed_response"))

    results = tools.execute_all(calls, tool_context)  # The only execution, ever, for this question.
    statuses = {result.status for result in results}
    if statuses & {tools.UNKNOWN_TOOL, tools.INVALID_ARGUMENTS}:
        return ToolRound(_escalation("invalid_tool_call"), results)
    if tools.ERROR in statuses:
        return ToolRound(_escalation("tool_error"), results)
    if tools.DETAILS_NOT_PROVIDED in statuses:
        reply = SupportAnswer(answer=NEEDS_ORDER_DETAILS_MESSAGE, sources=[], escalate=False,
                              reason=NEEDS_ORDER_DETAILS)
        return ToolRound(reply, results)
    proposal = next((r for r in results if r.status == tools.CONFIRMATION_REQUIRED), None)
    if proposal is not None:
        sources = [_article_source(articles_by_id["KB-024"])] if "KB-024" in articles_by_id else []
        sources.append(tool_source(proposal))
        reply = SupportAnswer(answer=cancellation_message(proposal.data["order_id"]), sources=sources,
                              escalate=False, reason=CANCELLATION_AVAILABLE)
        return ToolRound(reply, results)

    remaining = budget - (cohere_calls.clock() - started)
    if remaining < MIN_FINAL_CALL_SECONDS:
        logger.warning("Not enough time left for the final answer after tool calls")
        return ToolRound(_escalation("llm_error"), results)

    final_messages = messages + [assistant_message] + [_tool_message(result) for result in results]

    def final_call(request_options):
        return client.chat(
            model=model, messages=final_messages, documents=documents, tools=definitions,
            tool_choice="NONE", temperature=0, max_tokens=MAX_TOKENS, request_options=request_options,
        )

    try:
        # Retrying the chat request is safe: the tools already ran and never run again.
        response = call_with_retries(final_call, operation="chat", timeout_seconds=remaining)
    except Exception as error:
        logger.warning("Cohere chat call failed: %s", describe_error(error))
        return ToolRound(_escalation("llm_error"), results)

    try:
        return ToolRound(_read_final_response(response, articles_by_id, results), results)
    except Exception as error:
        logger.warning("Final tool answer could not be read: %s", type(error).__name__)
        return ToolRound(_escalation("malformed_response"), results)
