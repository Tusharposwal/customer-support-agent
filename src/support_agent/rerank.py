"""Optional Cohere reranking of retrieved articles (off by default).

When enabled, the hybrid-retrieved articles are re-ordered by Cohere Rerank
before answer generation. If a minimum score is configured, articles below it
are removed. This module never creates a Cohere client: the caller passes one,
so importing it costs nothing and needs no API key.

Settings (environment variables; blank means "use the default"):
    COHERE_RERANK_ENABLED          true/false (default false)
    COHERE_RERANK_MODEL            default rerank-v5.0-pro
    COHERE_RERANK_MIN_SCORE        0..1, or unset for no filtering (default unset)
    COHERE_RERANK_TIMEOUT_SECONDS  > 0 (default 5)

No production minimum score has been chosen; see docs/LEARNING_NOTES.md.
"""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_MODEL = "rerank-v5.0-pro"
DEFAULT_TIMEOUT_SECONDS = 5.0

ENABLED_ENV = "COHERE_RERANK_ENABLED"
MODEL_ENV = "COHERE_RERANK_MODEL"
MIN_SCORE_ENV = "COHERE_RERANK_MIN_SCORE"
TIMEOUT_ENV = "COHERE_RERANK_TIMEOUT_SECONDS"

_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = False
    model: str = DEFAULT_MODEL
    min_score: float | None = None  # None = re-order only, never remove articles
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


class RerankError(RuntimeError):
    """Reranking failed. The message never contains response bodies or secrets."""


# --- Configuration -----------------------------------------------------------


def _raw(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return None if value is None or value.strip() == "" else value.strip()


def _parse_bool(name: str, raw: str | None) -> bool:
    if raw is None:
        return False
    if raw.lower() in _TRUE:
        return True
    if raw.lower() in _FALSE:
        return False
    raise ValueError(f"{name} must be true or false")


def _parse_float(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def config_from_env(environ: Mapping[str, str] | None = None) -> RerankConfig:
    """Read the rerank settings. Raises ValueError for an invalid value, so a
    typo can never silently switch reranking off (or on). Error messages name
    the setting but never repeat its value."""
    environ = os.environ if environ is None else environ

    enabled = _parse_bool(ENABLED_ENV, _raw(environ, ENABLED_ENV))
    model = _raw(environ, MODEL_ENV) or DEFAULT_MODEL

    min_score = None
    raw_min = _raw(environ, MIN_SCORE_ENV)
    if raw_min is not None:
        min_score = _parse_float(MIN_SCORE_ENV, raw_min)
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"{MIN_SCORE_ENV} must be between 0 and 1")

    timeout = DEFAULT_TIMEOUT_SECONDS
    raw_timeout = _raw(environ, TIMEOUT_ENV)
    if raw_timeout is not None:
        timeout = _parse_float(TIMEOUT_ENV, raw_timeout)
        if timeout <= 0:
            raise ValueError(f"{TIMEOUT_ENV} must be greater than 0")

    return RerankConfig(enabled=enabled, model=model, min_score=min_score, timeout_seconds=timeout)


# --- Reranking ---------------------------------------------------------------


def document_text(article: dict) -> str:
    """The text sent to Cohere Rerank. Must stay identical to the experiment's
    "title-text-v1" format so the saved evaluation scores still apply."""
    return f"Title: {article['title']}\nText: {article['text']}"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_results(response, count: int) -> list[tuple[int, float]]:
    """Return (index, score) pairs. Every article must be ranked exactly once."""
    results = getattr(response, "results", None)
    if not isinstance(results, (list, tuple)):
        raise RerankError("Rerank response has no results list")

    pairs = []
    for item in results:
        index = getattr(item, "index", None)
        score = getattr(item, "relevance_score", None)
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < count:
            raise RerankError("Rerank response has an invalid article index")
        if not _is_number(score):
            raise RerankError("Rerank response has an invalid relevance score")
        pairs.append((index, float(score)))

    if sorted(index for index, _ in pairs) != list(range(count)):
        raise RerankError("Rerank response does not rank every article exactly once")
    return pairs


def rerank_articles(question: str, articles: list[dict], client, config: RerankConfig) -> list[dict]:
    """Return copies of `articles` sorted by rerank relevance (highest first),
    each with a `rerank_score` field added and every original field kept.

    If `config.min_score` is set, articles scoring below it are dropped, so the
    result may be empty. Raises RerankError on any API failure or malformed
    response. Exactly one HTTP request is made: SDK retries are disabled.
    """
    if not articles:
        return []

    try:
        response = client.rerank(
            model=config.model,
            query=question,
            documents=[document_text(article) for article in articles],
            request_options={"max_retries": 0, "timeout": config.timeout_seconds},
        )
    except Exception as error:
        # Only the exception type: API error text can echo request details.
        raise RerankError(f"Rerank request failed: {type(error).__name__}") from error

    pairs = _parse_results(response, len(articles))
    pairs.sort(key=lambda pair: pair[1], reverse=True)  # stable: ties keep API order

    return [
        {**articles[index], "rerank_score": score}
        for index, score in pairs
        if config.min_score is None or score >= config.min_score
    ]
