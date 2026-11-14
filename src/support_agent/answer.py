"""Grounded answer generation with Cohere Chat.

The model only sees the retrieved KB articles (passed as Cohere `documents`).
An answer is shown to the customer only if the model cites at least one of
those articles. In every other case we hand the question to a human.
"""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "command-a-03-2025"
MAX_TOKENS = 300
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

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
    articles_by_id = {article["id"]: article for article in articles}

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            documents=[_to_document(article) for article in articles],
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
    except Exception as error:  # Any API/network failure -> human hand-off.
        logger.warning("Cohere chat call failed: %s", type(error).__name__)
        return _escalation("llm_error")

    text = _extract_text(response)
    if text is None:
        logger.warning("Cohere chat response had no usable text content")
        return _escalation("malformed_response")

    if getattr(response, "finish_reason", None) == "MAX_TOKENS":
        return _escalation("incomplete_response")

    if INSUFFICIENT_EVIDENCE in text:
        return _escalation("insufficient_evidence")

    sources = [
        {
            "id": article_id,
            "title": articles_by_id[article_id]["title"],
            "url": articles_by_id[article_id]["url"],
        }
        for article_id in _extract_cited_ids(response)
        if article_id in articles_by_id  # Ignore IDs we did not supply.
    ]

    if not sources:
        return _escalation("no_citations")

    return SupportAnswer(answer=text, sources=sources, escalate=False, reason="answered")
