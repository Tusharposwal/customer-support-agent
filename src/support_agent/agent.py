"""Customer support agent: retrieve KB articles, then generate a grounded answer.

Usage:
    python -m support_agent.agent "How do I reset my password?"
"""

import logging
import os
import sys
from collections.abc import Mapping
from functools import lru_cache

from dotenv import load_dotenv

from support_agent.answer import HANDOFF_MESSAGE, SupportAnswer, generate_answer
from support_agent.cohere_calls import chat_timeout, close_client, describe_error, embed_timeout
from support_agent.rerank import RerankConfig, config_from_env, rerank_articles

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Required configuration (such as the API key) is missing."""


ARTICLE_FIELDS = ("id", "title", "text", "url")


def _handoff(reason: str) -> SupportAnswer:
    """Safe customer-facing hand-off. Never contains error details."""
    return SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason=reason)


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


def _checked_result(result, articles: list[dict]) -> SupportAnswer:
    """Last gate before the customer sees anything.

    A hand-off always uses the standard message with no sources. An answer must
    have text and at least one source, and every source must be an article that
    was given to the generator (same ID, title and URL).
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

    if result.reason != "answered":
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
    retriever=None,
    client=None,
    generator=generate_answer,
    reranker=rerank_articles,
    rerank_config: RerankConfig | None = None,
) -> SupportAnswer:
    """Answer a customer question from the knowledge base, or hand off to a human.

    `retriever`, `client`, `generator` and `reranker` default to the real
    components and can be replaced with fakes in tests. The chat model comes
    from COHERE_CHAT_MODEL (read inside `generate_answer`). Reranking is off
    unless `rerank_config` (default: COHERE_RERANK_* settings) enables it.
    """
    if not isinstance(question, str) or not question.strip():
        return _handoff("empty_question")

    try:
        client = client or _default_client()  # Also loads .env before settings are read.
        rerank_config = rerank_config or config_from_env()
        # Checked here so a typo is a configuration error, not an API failure.
        chat_timeout()
        embed_timeout()
    except Exception as error:
        logger.error("Support agent is not configured: %s", type(error).__name__)
        return _handoff("configuration_error")

    try:
        articles = (retriever or _hybrid_retrieve)(question)
    except Exception as error:
        logger.warning("Retrieval failed: %s", describe_error(error))
        return _handoff("retrieval_error")

    if not articles:
        return _handoff("no_articles")

    if not _valid_articles(articles):
        # A data or retrieval bug: report it as such, not as an LLM failure.
        logger.warning("Retrieval returned malformed articles")
        return _handoff("retrieval_error")

    if rerank_config.enabled:
        try:
            reranked = reranker(question, articles, client, rerank_config)
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

    try:
        result = generator(question, articles, client)
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
