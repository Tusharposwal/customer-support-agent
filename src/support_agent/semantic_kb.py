import cohere
from dotenv import load_dotenv

# agent doesn't import this module at load time (only inside a function), so
# importing from it here creates no import cycle.
from support_agent.agent import api_key_from_env
from support_agent.cohere_calls import CohereCallError, call_with_retries, close_client, embed_timeout
from support_agent.kb import load_articles

load_dotenv()  # Local .env for development; real environment variables (e.g. on Render) win.

# The same key rule as the chat client and startup validation: set, not blank,
# surrounding whitespace removed. Raises ConfigurationError (a RuntimeError)
# naming the setting otherwise, never its value.
API_KEY = api_key_from_env()

client = cohere.ClientV2(api_key=API_KEY)
MODEL = "embed-v4.0"


def close() -> None:
    """Close the embedding client's connection pool (at server shutdown).
    The module can't embed afterwards; a server process shuts down only once."""
    close_client(client)


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    """One vector per text. Bounded time, at most one retry for a temporary
    failure (see cohere_calls); raises CohereCallError on failure."""

    def request(request_options):
        return client.embed(
            texts=texts,
            model=MODEL,
            input_type=input_type,
            embedding_types=["float"],
            request_options=request_options,
        )

    response = call_with_retries(request, operation="embed", timeout_seconds=embed_timeout())

    vectors = getattr(getattr(response, "embeddings", None), "float", None)
    if (
        not isinstance(vectors, list)
        or len(vectors) != len(texts)
        or any(not isinstance(vector, list) or not vector for vector in vectors)
    ):
        # Checked so a short list can't silently drop articles (zip) and be cached.
        raise CohereCallError("embed", "malformed_response")
    return vectors


# (embedded texts, their vectors) for the most recent knowledge base.
_document_cache: tuple[tuple[str, ...], list[list[float]]] | None = None


def _embedded_articles():
    """The current articles with their embeddings.

    Articles are re-read on every call, as keyword retrieval does, so both
    halves of hybrid retrieval see the same knowledge base. The articles are
    embedded again only when their text changes; a failure is not cached, so
    the next question tries again.
    """
    global _document_cache
    articles = load_articles()
    if not articles:
        return []

    texts = tuple(f"{article['title']}\n{article['text']}" for article in articles)

    cached = _document_cache
    if cached is not None and cached[0] == texts:
        embeddings = cached[1]
    else:
        embeddings = _embed(list(texts), "search_document")
        _document_cache = (texts, embeddings)

    return [
        {**article, "embedding": embedding}
        for article, embedding in zip(articles, embeddings)
    ]


def _cosine_similarity(vector_a, vector_b):
    """Measure the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def retrieve_semantic(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve articles by semantic similarity."""
    if not query.strip() or top_k <= 0:
        return []

    query_embedding = _embed([query], "search_query")[0]
    scored_articles = []

    for article in _embedded_articles():
        score = _cosine_similarity(
            query_embedding,
            article["embedding"],
        )

        scored_articles.append(
            {
                **{key: value for key, value in article.items() if key != "embedding"},
                "score": score,
            }
        )

    scored_articles.sort(
        key=lambda article: article["score"],
        reverse=True,
    )

    return scored_articles[:top_k]
