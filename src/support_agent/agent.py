"""Customer support agent: retrieve KB articles, then generate a grounded answer.

Usage:
    python -m support_agent.agent "How do I reset my password?"
"""

import logging
import os
import re
import sys
from collections.abc import Mapping
from datetime import date
from functools import lru_cache

from dotenv import load_dotenv

from support_agent import query_rewrite, tools
from support_agent.answer import (
    CANCELLATION_AVAILABLE,
    CLARIFICATION_MESSAGE,
    HANDOFF_MESSAGE,
    NEEDS_ORDER_DETAILS,
    NEEDS_ORDER_DETAILS_MESSAGE,
    SupportAnswer,
    ToolRound,
    cancellation_message,
    citable_tool_sources,
    generate_answer,
    generate_answer_with_tools,
)
from support_agent.demo_orders import OrderBook, load_orders
from support_agent.cohere_calls import chat_timeout, close_client, describe_error, embed_timeout
from support_agent.rerank import RerankConfig, config_from_env, rerank_articles

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Required configuration (such as the API key) is missing."""


ARTICLE_FIELDS = ("id", "title", "text", "url")


NEEDS_CLARIFICATION = "needs_clarification"


def _handoff(reason: str) -> SupportAnswer:
    """Safe customer-facing hand-off. Never contains error details."""
    return SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason=reason)


def _clarification() -> SupportAnswer:
    """Ask the customer to say more. Not escalated, no sources, fixed text."""
    return SupportAnswer(answer=CLARIFICATION_MESSAGE, sources=[], escalate=False, reason=NEEDS_CLARIFICATION)


def _valid_articles(articles) -> bool:
    """Every article has non-empty text fields and a unique ID."""
    if not isinstance(articles, (list, tuple)):
        return False
    seen = set()
    for article in articles:
        if not isinstance(article, dict) or any(
            not isinstance(article.get(name), str) or not article[name].strip() for name in ARTICLE_FIELDS
        ):
            return False
        if article["id"] in seen:
            return False
        seen.add(article["id"])
    return True


def _checked_result(
    result, articles: list[dict], *, clarification_allowed: bool = False, tool_results=()
) -> SupportAnswer:
    """Last gate before the customer sees anything.

    A hand-off always uses the standard message with no sources. An answer must
    have text and at least one source, and every source must be an article that
    was given to the generator (same ID, title and URL).

    A clarification request (reason "needs_clarification") passes only when
    the agent itself asked for one (`clarification_allowed`), and only as the
    fixed message with no sources and no escalation. It is never accepted from
    the answer generator, so it can't be used to skip the citation checks.

    `tool_results` are the tool results produced while answering *this*
    question. A source may be one of them (as answer.tool_source shows it);
    tool results from any other question are never in the list. The two
    server-written tool outcomes pass only in their exact form:
    "needs_order_details" (fixed text, no sources) after a tool reported
    missing details, and "cancellation_available" (fixed text naming the
    order) after the cancellation tool said the order can be cancelled.
    """
    if not isinstance(result, SupportAnswer):
        logger.warning("Answer generator returned %s, not SupportAnswer", type(result).__name__)
        return _handoff("agent_error")

    if not isinstance(result.escalate, bool):
        logger.warning("Answer generator returned a non-boolean escalate flag")
        return _handoff("agent_error")

    if result.escalate:
        reason = result.reason if isinstance(result.reason, str) and result.reason else "agent_error"
        return _handoff(reason)

    if result.reason == NEEDS_CLARIFICATION:
        if clarification_allowed and result.answer == CLARIFICATION_MESSAGE and result.sources == []:
            return _clarification()
        logger.warning("Unexpected clarification result")
        return _handoff("agent_error")

    allowed_tool_sources = citable_tool_sources(tool_results)

    if result.reason == NEEDS_ORDER_DETAILS:
        if (
            any(r.status == tools.DETAILS_NOT_PROVIDED for r in tool_results)
            and result.answer == NEEDS_ORDER_DETAILS_MESSAGE and result.sources == []
        ):
            return SupportAnswer(answer=NEEDS_ORDER_DETAILS_MESSAGE, sources=[], escalate=False,
                                 reason=NEEDS_ORDER_DETAILS)
        logger.warning("Unexpected order-details request")
        return _handoff("agent_error")

    if result.reason == CANCELLATION_AVAILABLE:
        proposals = [r for r in tool_results if r.status == tools.CONFIRMATION_REQUIRED]
        texts = {cancellation_message(r.data["order_id"]) for r in proposals}
        if not (result.answer in texts and isinstance(result.sources, list) and result.sources):
            logger.warning("Unexpected cancellation reply")
            return _handoff("agent_error")
        # Its sources are checked below like any answer's.
    elif result.reason != "answered":
        # An answer labelled as a failure (or not labelled) is contradictory.
        logger.warning("Answer generator returned an answer with an unexpected reason")
        return _handoff("agent_error")
    if not isinstance(result.answer, str) or not result.answer.strip():
        logger.warning("Answer generator returned an empty answer")
        return _handoff("malformed_response")
    if not isinstance(result.sources, (list, tuple)):
        logger.warning("Answer generator returned sources that are not a list")
        return _handoff("invalid_citations")
    if not result.sources:
        logger.warning("Answer generator returned an answer without sources")
        return _handoff("no_citations")

    supplied = {article["id"]: article for article in articles}
    for source in result.sources:
        if isinstance(source, dict) and source in allowed_tool_sources:
            continue  # One of this question's own tool results.
        source_id = source.get("id") if isinstance(source, dict) else None
        article = supplied.get(source_id) if isinstance(source_id, str) else None
        if article is None or source.get("title") != article["title"] or source.get("url") != article["url"]:
            logger.warning("Answer cited an article that was not given to the generator")
            return _handoff("invalid_citations")

    return result


def _hybrid_retrieve(question: str) -> list[dict]:
    # Imported here, not at the top: semantic_kb creates a Cohere client
    # when it is imported, and we only want that when a question is asked.
    from support_agent.hybrid_kb import retrieve_hybrid

    return retrieve_hybrid(question)


# The order tools are offered only when the customer has written an order
# number: the registry refuses any call whose order number the customer didn't
# write (grounding), so offering them without one could only fail. Every other
# question keeps exactly the request and flow it had before tools existed.
_ORDER_NUMBER = re.compile(r"(?<![A-Za-z0-9-])ORD-\d{4}(?!\d)", re.IGNORECASE)


def mentions_order_number(question: str, history: list[dict] | None) -> bool:
    texts = [question] + [str(m.get("content", "")) for m in (history or []) if m.get("role") == "customer"]
    return any(_ORDER_NUMBER.search(text) for text in texts)


@lru_cache(maxsize=1)
def _default_order_book() -> OrderBook:
    """The fictional demo orders, loaded once."""
    return load_orders()


API_KEY_ENV = "COHERE_API_KEY"


def api_key_from_env(environ: Mapping[str, str] | None = None) -> str:
    """Return the Cohere API key, or raise ConfigurationError if it is missing
    or blank. The key itself never appears in the error."""
    environ = os.environ if environ is None else environ
    api_key = (environ.get(API_KEY_ENV) or "").strip()
    if not api_key:
        raise ConfigurationError(f"{API_KEY_ENV} is not set")
    return api_key


@lru_cache(maxsize=1)
def _default_client():
    """Create the Cohere client on first use and reuse it afterwards."""
    load_dotenv()
    api_key = api_key_from_env()

    import cohere

    return cohere.ClientV2(api_key=api_key)


def close_default_client() -> None:
    """Close the cached chat client, if one was created, so the next
    `_default_client()` call makes a new one. Errors are raised to the caller."""
    if _default_client.cache_info().currsize == 0:
        return
    client = _default_client()  # The cached client; nothing new is created.
    _default_client.cache_clear()
    close_client(client)


def answer_question(
    question: str,
    *,
    history: list[dict] | None = None,
    trace: dict | None = None,
    retriever=None,
    client=None,
    generator=generate_answer,
    reranker=rerank_articles,
    rerank_config: RerankConfig | None = None,
    rewriter=None,
    tool_generator=generate_answer_with_tools,
    order_book: OrderBook | None = None,
    today: date | None = None,
    is_cancelled=None,
) -> SupportAnswer:
    """Answer a customer question from the knowledge base, or hand off to a human.

    `history` is the earlier messages of the conversation
    ([{"role": "customer" | "assistant", "content": ...}], oldest first). It
    is used only to rewrite a follow-up into a standalone question for
    retrieval (query_rewrite); it is never given to the answer generator and
    is never evidence. Without history (a first message) no rewrite call is
    made and the pipeline is exactly as before.

    If `trace` is a dict, it receives "rewrite" (the query_rewrite outcome)
    and "retrieval_query" (what was searched for), for storage and logs.

    `retriever`, `client`, `generator`, `reranker` and `rewriter` default to
    the real components and can be replaced with fakes in tests. The chat
    model comes from COHERE_CHAT_MODEL (read inside `generate_answer`).
    Reranking is off unless `rerank_config` (default: COHERE_RERANK_*
    settings) enables it.

    Order tools (tools.py) are offered only when the customer's messages
    contain an order number; then `tool_generator` (default
    generate_answer_with_tools) answers instead of `generator`. `order_book`
    (default: the demo orders) and `today` can be replaced in tests. If the
    orders can't be loaded, the question is answered without tools. If
    `tool_generator` is None, tools are never offered. `is_cancelled`
    (order_id -> bool; the API passes CancellationStore.is_cancelled) lets
    tools see cancellations the customer has confirmed.

    When the checked answer is the server-written "can be cancelled" reply,
    `trace["cancellable_order_id"]` names the order, so the API can save a
    pending cancellation with the reply (cancellations.py). The agent never
    creates, sees or confirms an action itself.
    """
    trace = {} if trace is None else trace
    if not isinstance(question, str) or not question.strip():
        return _handoff("empty_question")

    try:
        client = client or _default_client()  # Also loads .env before settings are read.
        rerank_config = rerank_config or config_from_env()
        # Checked here so a typo is a configuration error, not an API failure.
        chat_timeout()
        embed_timeout()
        query_rewrite.rewrite_timeout()
    except Exception as error:
        logger.error("Support agent is not configured: %s", type(error).__name__)
        return _handoff("configuration_error")

    try:
        rewrite = (rewriter or query_rewrite.rewrite_question)(question, history, client)
        if not isinstance(rewrite, query_rewrite.Rewrite) or not isinstance(rewrite.query, str) or not rewrite.query.strip():
            raise TypeError("rewriter returned an unusable result")
    except Exception as error:  # rewrite_question never raises; this is a safety net.
        logger.warning("Query rewrite failed, using the message as written: %s", type(error).__name__)
        rewrite = query_rewrite.Rewrite(query=question, outcome=query_rewrite.FAILED)
    trace["rewrite"] = rewrite.outcome

    if rewrite.outcome == query_rewrite.NEEDS_CLARIFICATION:
        return _checked_result(_clarification(), [], clarification_allowed=True)

    # Only a checked rewrite replaces the message for retrieval.
    query = rewrite.query if rewrite.outcome == query_rewrite.REWRITTEN else question
    trace["retrieval_query"] = query

    tool_context = None
    if tool_generator is not None and mentions_order_number(question, history):
        try:
            tool_context = tools.ToolContext.for_question(
                order_book or _default_order_book(), question, history, today or date.today(),
                is_cancelled=is_cancelled,
            )
        except Exception as error:  # Fail open: answer from the articles alone.
            logger.warning("Order tools unavailable: %s", type(error).__name__)
    if tool_context is not None:
        trace["tools_offered"] = True  # Recorded only when offered: other questions are unchanged.

    try:
        articles = (retriever or _hybrid_retrieve)(query)
    except Exception as error:
        logger.warning("Retrieval failed: %s", describe_error(error))
        return _handoff("retrieval_error")

    if not articles and tool_context is None:
        return _handoff("no_articles")

    if not _valid_articles(articles):
        # A data or retrieval bug: report it as such, not as an LLM failure.
        logger.warning("Retrieval returned malformed articles")
        return _handoff("retrieval_error")

    if rerank_config.enabled and articles:
        try:
            reranked = reranker(query, articles, client, rerank_config)
            retrieved = {article["id"]: article for article in articles}
            # Reranking may only re-order and remove articles, never change them.
            if not _valid_articles(reranked) or any(
                a["id"] not in retrieved or any(a[name] != retrieved[a["id"]][name] for name in ARTICLE_FIELDS)
                for a in reranked
            ):
                raise ValueError("Reranker output is not an unchanged subset of the retrieved articles")
        except Exception as error:
            # Fail open: an outage is not evidence that the question is unsupported.
            # The citation check in generate_answer still applies to these articles.
            logger.warning("Rerank failed, using hybrid order: %s", type(error).__name__)
        else:
            if reranked:
                articles = reranked
            elif rerank_config.min_score is None:
                logger.warning("Rerank returned no articles without a minimum score; using hybrid order")
            else:
                # Reranking succeeded and judged every article below the minimum score.
                logger.info(
                    "Rerank minimum score %.4f removed all %d articles (%s)",
                    rerank_config.min_score, len(articles), ", ".join(a["id"] for a in articles),
                )
                return _handoff("below_rerank_threshold")

    if tool_context is not None:
        try:
            tool_round = tool_generator(
                question, articles, client, tool_context,
                standalone_query=query if query != question else None,
            )
            if not isinstance(tool_round, ToolRound):
                raise TypeError("tool generator did not return a ToolRound")
        except Exception as error:  # Safety net; the tool generator handles API errors itself.
            logger.warning("Answer generation with tools failed: %s", type(error).__name__)
            return _handoff("agent_error")
        trace["tool_calls"] = len(tool_round.results)
        try:
            checked = _checked_result(tool_round.answer, articles, tool_results=tool_round.results)
            if checked.reason == CANCELLATION_AVAILABLE:
                # The gate accepted it only as the exact text for one proposal.
                (order_id,) = {
                    r.data["order_id"] for r in tool_round.results
                    if r.status == tools.CONFIRMATION_REQUIRED and cancellation_message(r.data["order_id"]) == checked.answer
                }
                trace["cancellable_order_id"] = order_id
            return checked
        except Exception as error:
            logger.error("Final answer check failed: %s", type(error).__name__)
            return _handoff("agent_error")

    try:
        if query != question:
            result = generator(question, articles, client, standalone_query=query)
        else:
            result = generator(question, articles, client)  # Exactly as before conversations.
    except Exception as error:  # Safety net; generate_answer handles API errors itself.
        logger.warning("Answer generation failed: %s", type(error).__name__)
        return _handoff("agent_error")

    try:
        return _checked_result(result, articles)
    except Exception as error:  # The gate must never let a result through unchecked.
        logger.error("Final answer check failed: %s", type(error).__name__)
        return _handoff("agent_error")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    question = " ".join(args).strip()
    if not question:
        print('Usage: python -m support_agent.agent "<customer question>"', file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.WARNING)
    result = answer_question(question)

    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"- {source['title']} ({source['id']}): {source['url']}")
    print(f"\n[escalate={result.escalate}, reason={result.reason}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
