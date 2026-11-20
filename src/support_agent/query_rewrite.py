"""Turn a follow-up message into a standalone question for retrieval.

    Customer: How can I return my order?
    Assistant: ...
    Customer: What if I bought it 35 days ago?
        -> "Is a return or refund possible 35 days after purchase?"

The rewritten question is used only to *search* the knowledge base (and to
tell the answer generator what the customer is asking). It is never evidence:
answers still come only from the retrieved articles, with the usual citation
checks. The customer's own message is stored and shown unchanged.

Rules:
* No history (the first message of a conversation): no Cohere call; the
  message is used as it is.
* Otherwise one Cohere chat call (COHERE_REWRITE_MODEL, default: the answer
  model) within COHERE_REWRITE_TIMEOUT_SECONDS (default 8), with the shared
  retry policy from cohere_calls.
* Only a bounded excerpt of the conversation is sent: the last
  HISTORY_MESSAGES messages, each cut to MESSAGE_CHARS characters, at most
  HISTORY_CHARS in total. Earlier assistant replies are labelled as context
  that may be wrong. Nothing is removed from the stored conversation.
* The model's output is checked: valid JSON of the expected shape, a
  non-empty question of at most MAX_QUERY_CHARS characters, and no number
  the customer never wrote (a rewrite must not add facts). Anything else,
  and any API failure, falls back to the customer's message as it is: a
  failed rewrite is never a reason to refuse an answer.
* If the model reports that the message can't be understood from the
  conversation, the outcome is "needs_clarification" and the agent asks the
  customer to say more, without retrieving or answering.
"""

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from support_agent.cohere_calls import call_with_retries, describe_error, timeout_from_env

logger = logging.getLogger(__name__)

REWRITE_MODEL_ENV = "COHERE_REWRITE_MODEL"
REWRITE_TIMEOUT_ENV = "COHERE_REWRITE_TIMEOUT_SECONDS"
DEFAULT_REWRITE_TIMEOUT_SECONDS = 8.0

HISTORY_MESSAGES = 6
MESSAGE_CHARS = 500
HISTORY_CHARS = 3000
MAX_QUERY_CHARS = 300
MAX_TOKENS = 200

# Outcomes (also logged, so they are plain identifiers).
NOT_NEEDED = "not_needed"  # No history: no call made.
REWRITTEN = "rewritten"  # A checked standalone question differing from the message.
UNCHANGED = "unchanged"  # The message was already standalone.
NEEDS_CLARIFICATION = "needs_clarification"
FAILED = "failed"  # API error or unusable output: the message is used as it is.

SYSTEM_PROMPT = """You prepare search queries for a customer support help centre.
You are given the recent conversation and the customer's latest message.
Rewrite the latest message as ONE standalone question that can be understood
without the conversation, so it can be used to search help articles.

Rules:
- Use the conversation only to resolve references such as "it", "that" or "the same".
- Keep the customer's meaning. Do not answer the question.
- Do not add facts, numbers, dates, products or details the customer did not write.
- Earlier assistant replies are context only and may be wrong; never copy facts from them.
- If the latest message is already standalone, return it unchanged.
- If the latest message depends on the conversation but it is unclear what the
  customer means, set needs_clarification to true.
- If the customer changes topic, rewrite only the new question.

Reply with JSON only, in exactly this form:
{"standalone_query": "<question>", "needs_clarification": false}"""

RESPONSE_FORMAT = {
    "type": "json_object",
    "json_schema": {
        "type": "object",
        "properties": {
            "standalone_query": {"type": "string"},
            "needs_clarification": {"type": "boolean"},
        },
        "required": ["standalone_query", "needs_clarification"],
    },
}

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_COMPLETE_FINISH_REASONS = {"COMPLETE", "STOP_SEQUENCE"}


@dataclass(frozen=True)
class Rewrite:
    query: str  # What to retrieve with (the original message unless REWRITTEN).
    outcome: str


def rewrite_timeout(environ: Mapping[str, str] | None = None) -> float:
    return timeout_from_env(REWRITE_TIMEOUT_ENV, DEFAULT_REWRITE_TIMEOUT_SECONDS, environ)


def rewrite_model(environ: Mapping[str, str] | None = None) -> str:
    from support_agent.answer import DEFAULT_MODEL

    environ = os.environ if environ is None else environ
    return (
        (environ.get(REWRITE_MODEL_ENV) or "").strip()
        or (environ.get("COHERE_CHAT_MODEL") or "").strip()
        or DEFAULT_MODEL
    )


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def conversation_excerpt(history: Sequence[Mapping]) -> str:
    """The last few messages, labelled and size-bounded, oldest first."""
    lines = []
    total = 0
    for message in reversed(list(history)[-HISTORY_MESSAGES:]):
        label = "Customer" if message.get("role") == "customer" else "Assistant (context only, may be wrong)"
        line = f"{label}: {_clip(message.get('content', ''), MESSAGE_CHARS)}"
        if total + len(line) > HISTORY_CHARS:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(reversed(lines))


def build_prompt(question: str, history: Sequence[Mapping]) -> str:
    return (
        "Conversation so far:\n"
        f"{conversation_excerpt(history)}\n\n"
        f"Customer's latest message: {_clip(question, MESSAGE_CHARS * 2)}"
    )


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def check_output(text: str, question: str, history: Sequence[Mapping]) -> Rewrite:
    """Validate the model's JSON. Raises ValueError for anything unusable."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("not an object")
    needs_clarification = data.get("needs_clarification")
    query = data.get("standalone_query")
    if not isinstance(needs_clarification, bool) or not isinstance(query, str):
        raise ValueError("wrong field types")
    if needs_clarification:
        return Rewrite(query=question, outcome=NEEDS_CLARIFICATION)

    query = " ".join(query.split())
    if not query:
        raise ValueError("empty query")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError("query too long")
    customer_text = " ".join(
        [question] + [str(m.get("content", "")) for m in history if m.get("role") == "customer"]
    )
    if not _numbers(query) <= _numbers(customer_text):
        raise ValueError("query contains a number the customer did not write")
    if query.casefold() == " ".join(question.split()).casefold():
        return Rewrite(query=question, outcome=UNCHANGED)
    return Rewrite(query=query, outcome=REWRITTEN)


def _response_text(response) -> str:
    finish_reason = getattr(response, "finish_reason", None)
    if finish_reason not in _COMPLETE_FINISH_REASONS:
        raise ValueError("generation did not complete")
    content = getattr(getattr(response, "message", None), "content", None)
    if not isinstance(content, (list, tuple)):
        raise ValueError("no content")
    text = "".join(
        item.text for item in content
        if getattr(item, "type", None) == "text" and isinstance(getattr(item, "text", None), str)
    ).strip()
    if not text:
        raise ValueError("empty content")
    return text


def rewrite_question(question: str, history: Sequence[Mapping] | None, client, *, model: str | None = None) -> Rewrite:
    """Return the query to retrieve with. Never raises: on any failure the
    customer's message is used as it is (outcome FAILED)."""
    if not history:
        return Rewrite(query=question, outcome=NOT_NEEDED)

    try:
        timeout_seconds = rewrite_timeout()
        model = model or rewrite_model()
        prompt = build_prompt(question, history)

        def request(request_options):
            return client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=RESPONSE_FORMAT,
                temperature=0,
                max_tokens=MAX_TOKENS,
                request_options=request_options,
            )

        response = call_with_retries(request, operation="rewrite", timeout_seconds=timeout_seconds)
    except Exception as error:
        logger.warning("Query rewrite failed, using the message as written: %s", describe_error(error))
        return Rewrite(query=question, outcome=FAILED)

    try:
        return check_output(_response_text(response), question, history)
    except Exception as error:
        # The reason names the rule only; the model's text is never logged.
        reason = str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else type(error).__name__
        logger.warning("Query rewrite output rejected, using the message as written: %s", reason)
        return Rewrite(query=question, outcome=FAILED)
