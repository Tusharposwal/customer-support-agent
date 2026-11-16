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

HANDOFF_MESSAGE = (
    "I'm not able to answer this confidently from our help articles. "
    "I've passed your question to our support team, and a team member "
    "will follow up with you."
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


def generate_answer(question: str, articles: list[dict], client, model: str | None = None) -> SupportAnswer:
    """Answer `question` from `articles` using a Cohere ClientV2-compatible `client`."""
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
                {"role": "user", "content": question},
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
