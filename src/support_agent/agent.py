"""Customer support agent: retrieve KB articles, then generate a grounded answer.

Usage:
    python -m support_agent.agent "How do I reset my password?"
"""

import logging
import os
import sys
from functools import lru_cache

from dotenv import load_dotenv

from support_agent.answer import HANDOFF_MESSAGE, SupportAnswer, generate_answer

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Required configuration (such as the API key) is missing."""


def _handoff(reason: str) -> SupportAnswer:
    """Safe customer-facing hand-off. Never contains error details."""
    return SupportAnswer(answer=HANDOFF_MESSAGE, sources=[], escalate=True, reason=reason)


def _hybrid_retrieve(question: str) -> list[dict]:
    # Imported here, not at the top: semantic_kb creates a Cohere client
    # when it is imported, and we only want that when a question is asked.
    from support_agent.hybrid_kb import retrieve_hybrid

    return retrieve_hybrid(question)


@lru_cache(maxsize=1)
def _default_client():
    """Create the Cohere client on first use and reuse it afterwards."""
    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ConfigurationError("COHERE_API_KEY is not set")

    import cohere

    return cohere.ClientV2(api_key=api_key)


def answer_question(
    question: str,
    *,
    retriever=None,
    client=None,
    generator=generate_answer,
) -> SupportAnswer:
    """Answer a customer question from the knowledge base, or hand off to a human.

    `retriever`, `client` and `generator` default to the real components and can
    be replaced with fakes in tests. The chat model comes from COHERE_CHAT_MODEL
    (read inside `generate_answer`).
    """
    if not isinstance(question, str) or not question.strip():
        return _handoff("empty_question")

    try:
        client = client or _default_client()
    except Exception as error:
        logger.error("Support agent is not configured: %s", type(error).__name__)
        return _handoff("configuration_error")

    try:
        articles = (retriever or _hybrid_retrieve)(question)
    except Exception as error:
        logger.warning("Retrieval failed: %s", type(error).__name__)
        return _handoff("retrieval_error")

    if not articles:
        return _handoff("no_articles")

    try:
        return generator(question, articles, client)
    except Exception as error:  # Safety net; generate_answer handles API errors itself.
        logger.warning("Answer generation failed: %s", type(error).__name__)
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
