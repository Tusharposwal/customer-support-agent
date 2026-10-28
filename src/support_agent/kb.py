import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_PATH = PROJECT_ROOT / "data" / "kb.json"

STOP_WORDS = {
    "what",
    "is",
    "the",
    "on",
    "a",
    "an",
    "how",
    "do",
    "i",
    "my",
    "to",
    "for",
    "in",
    "of",
    "and",
    "can",
    "you",
}


def load_articles() -> list[dict]:
    """Load support articles from the JSON knowledge base."""
    with KB_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text: str) -> set[str]:
    """Convert text into lowercase words and remove stop words."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return words - STOP_WORDS


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Return the most relevant articles using keyword overlap."""
    query_tokens = tokenize(query)

    # Expand common login-problem queries.
    if "log" in query_tokens or "login" in query_tokens:
        query_tokens.update({"reset", "password"})

    if not query_tokens:
        return []

    articles = load_articles()
    scored_articles = []

    for article in articles:
        article_text = f"{article['title']} {article['text']}"
        article_tokens = tokenize(article_text)

        overlap = query_tokens & article_tokens
        score = len(overlap) / len(query_tokens)

        MIN_SCORE = 0.30


        if score >= MIN_SCORE:
            scored_articles.append({**article, "score": score})

    scored_articles.sort(
        key=lambda article: article["score"],
        reverse=True,
    )

    return scored_articles[:top_k]
