import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_PATH = PROJECT_ROOT / "data" / "kb.json"

REQUIRED_FIELDS = ("id", "title", "text", "url")

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


def load_articles(path: Path | None = None) -> list[dict]:
    """Load and validate support articles from the JSON knowledge base."""
    path = KB_PATH if path is None else Path(path)

    with path.open("r", encoding="utf-8") as file:
        try:
            articles = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Knowledge base {path} is not valid JSON: {error.msg} "
                f"(line {error.lineno}, column {error.colno})"
            ) from error

    _validate_articles(articles, path)
    return articles


def _validate_articles(articles, path: Path) -> None:
    """Raise ValueError if the knowledge base content is malformed."""
    if not isinstance(articles, list):
        raise ValueError(
            f"Knowledge base {path} must contain a JSON list of articles, "
            f"got {type(articles).__name__}"
        )

    seen_ids = set()

    for index, article in enumerate(articles):
        location = f"Knowledge base {path}, article #{index}"

        if not isinstance(article, dict):
            raise ValueError(
                f"{location} must be a JSON object, got {type(article).__name__}"
            )

        missing = [field for field in REQUIRED_FIELDS if field not in article]
        if missing:
            raise ValueError(
                f"{location} is missing required field(s): {', '.join(missing)}"
            )

        for field in REQUIRED_FIELDS:
            value = article[field]

            if not isinstance(value, str):
                raise ValueError(
                    f"{location} field '{field}' must be a string, "
                    f"got {type(value).__name__}"
                )

            if not value.strip():
                raise ValueError(f"{location} field '{field}' must not be empty")

        article_id = article["id"]
        if article_id in seen_ids:
            raise ValueError(f"{location} has duplicate id '{article_id}'")
        seen_ids.add(article_id)


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
