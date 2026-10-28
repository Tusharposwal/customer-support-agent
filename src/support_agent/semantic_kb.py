import os
from functools import lru_cache

import cohere
from dotenv import load_dotenv

from support_agent.kb import load_articles

load_dotenv()

API_KEY = os.getenv("COHERE_API_KEY")
if not API_KEY:
    raise RuntimeError("COHERE_API_KEY is missing from .env")

client = cohere.ClientV2(api_key=API_KEY)
MODEL = "embed-v4.0"


@lru_cache(maxsize=1)
def _embedded_articles():
    """Load articles and embed them once per process."""
    articles = load_articles()

    texts = [f"{article['title']}\n{article['text']}" for article in articles]

    response = client.embed(
        texts=texts,
        model=MODEL,
        input_type="search_document",
        embedding_types=["float"],
    )

    embeddings = response.embeddings.float

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

    response = client.embed(
        texts=[query],
        model=MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )

    query_embedding = response.embeddings.float[0]
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
