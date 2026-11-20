"""Calibrate a semantic-similarity threshold from saved embeddings.

Offline (default, no Cohere calls):
    python evals/calibrate_semantic.py

Create the fixture (exactly 2 Cohere embed calls, refuses to overwrite):
    python evals/calibrate_semantic.py --generate

The fixture stores the article and question vectors plus the hashes of the
texts that were embedded. If the knowledge base or the question list changes,
the fixture is rejected instead of silently giving stale scores.
"""

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from support_agent import kb
from support_agent.kb import load_articles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evals" / "retrieval_queries.json"
FIXTURE_PATH = PROJECT_ROOT / "evals" / "fixtures" / "semantic_embeddings.json"

# The 3-article knowledge base (v1) that every saved fixture in evals/fixtures/
# was made from and both v1 question sets were labelled against. data/kb.json
# has since grown (KB v2, Phase 5); historical experiments read this frozen,
# byte-identical copy so their saved results stay reproducible.
HISTORICAL_KB_PATH = PROJECT_ROOT / "evals" / "fixtures" / "kb_v1.json"

FIXTURE_VERSION = 1
MAX_EMBED_CALLS = 2


class FixtureError(ValueError):
    """The saved fixture is missing, malformed or out of date."""


@contextmanager
def historical_kb():
    """Point support_agent.kb at the frozen v1 knowledge base inside the block.

    Both load_articles() and keyword retrieve() read kb.KB_PATH when called,
    so everything in the block (including the real hybrid retrieval) sees v1.
    """
    saved = kb.KB_PATH
    kb.KB_PATH = HISTORICAL_KB_PATH
    try:
        yield HISTORICAL_KB_PATH
    finally:
        kb.KB_PATH = saved


# --- Inputs and hashes -------------------------------------------------------


def load_cases(path: Path = DATASET_PATH) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def document_text(article: dict) -> str:
    """The exact text semantic_kb._embedded_articles() embeds."""
    return f"{article['title']}\n{article['text']}"


def _sha256_json(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def kb_hash(articles: list[dict]) -> str:
    return _sha256_json([{"id": a["id"], "text": document_text(a)} for a in articles])


def questions_hash(cases: list[dict]) -> str:
    """Hash of the question texts that were embedded (labels are read live)."""
    return _sha256_json([case["query"] for case in cases])


# --- Fixture generation (the only code that talks to Cohere) ----------------


def generate_fixture(articles: list[dict], cases: list[dict], path: Path = FIXTURE_PATH) -> dict:
    path = Path(path)
    if path.exists():
        raise FixtureError(f"{path} already exists; delete it first to regenerate.")

    # Imported here so offline runs never load the API key or create a client.
    from support_agent import semantic_kb

    calls = []
    real_embed = semantic_kb.client.embed

    def counted_embed(**kwargs):
        if len(calls) >= MAX_EMBED_CALLS:
            raise RuntimeError(f"Refusing embed call #{len(calls) + 1}; limit is {MAX_EMBED_CALLS}.")
        calls.append(
            {
                "model": kwargs["model"],
                "input_type": kwargs["input_type"],
                "texts": len(kwargs["texts"]),
            }
        )
        return real_embed(**kwargs)

    semantic_kb.client.embed = counted_embed

    # Call 1: all articles, using the production document configuration.
    embedded_articles = semantic_kb._embedded_articles()

    # Call 2: all questions, using the same settings as retrieve_semantic().
    response = semantic_kb.client.embed(
        texts=[case["query"] for case in cases],
        model=semantic_kb.MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    question_vectors = response.embeddings.float

    if [a["id"] for a in embedded_articles] != [a["id"] for a in articles]:
        raise FixtureError("Embedded articles do not match the current knowledge base.")

    fixture = {
        "version": FIXTURE_VERSION,
        "generated_on": date.today().isoformat(),
        "model": calls[0]["model"],
        "document_input_type": calls[0]["input_type"],
        "query_input_type": calls[1]["input_type"],
        "embedding_type": "float",
        "dimensions": len(question_vectors[0]),
        "kb_sha256": kb_hash(articles),
        "questions_sha256": questions_hash(cases),
        "embed_calls": calls,
        "documents": [
            {"id": a["id"], "embedding": a["embedding"]} for a in embedded_articles
        ],
        "questions": [
            {"query": case["query"], "embedding": vector}
            for case, vector in zip(cases, question_vectors)
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return fixture


# --- Offline loading and validation ------------------------------------------


def load_fixture(articles: list[dict], cases: list[dict], path: Path = FIXTURE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FixtureError(f"{path} not found. Run with --generate (2 Cohere calls).")

    fixture = json.loads(path.read_text(encoding="utf-8"))

    if fixture.get("version") != FIXTURE_VERSION:
        raise FixtureError(f"Unsupported fixture version: {fixture.get('version')!r}")
    if fixture.get("kb_sha256") != kb_hash(articles):
        raise FixtureError("Knowledge base changed since the fixture was made; regenerate it.")
    if fixture.get("questions_sha256") != questions_hash(cases):
        raise FixtureError("Question list changed since the fixture was made; regenerate it.")

    vectors = [d["embedding"] for d in fixture["documents"]]
    vectors += [q["embedding"] for q in fixture["questions"]]
    if any(len(vector) != fixture["dimensions"] for vector in vectors):
        raise FixtureError("Fixture vectors do not all have the recorded dimensions.")

    return fixture


# --- Scoring -----------------------------------------------------------------


def cosine_similarity(vector_a, vector_b) -> float:
    """Same formula as semantic_kb._cosine_similarity (kept here so offline
    runs do not import semantic_kb, which needs an API key at import time)."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def build_row(case: dict, scores: dict[str, float]) -> dict:
    """One question's scores against every article, in the shared row format."""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    expected = case["expected_id"]

    return {
        "query": case["query"],
        "expected_id": expected,
        "answerable": expected is not None,
        "scores": scores,
        "top_id": ranked[0][0],
        "top_score": ranked[0][1],
        "margin": ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1],
        # The number a threshold must keep (answerable) or cut (unsupported).
        "key_score": scores[expected] if expected is not None else ranked[0][1],
    }


def score_questions(fixture: dict, cases: list[dict]) -> list[dict]:
    """Score every question against every article."""
    rows = []

    for case, question in zip(cases, fixture["questions"]):
        scores = {
            doc["id"]: cosine_similarity(question["embedding"], doc["embedding"])
            for doc in fixture["documents"]
        }
        rows.append(build_row(case, scores))

    return rows


def threshold_metrics(rows: list[dict], threshold: float) -> dict:
    """An article is accepted when its score >= threshold."""
    answerable = [r for r in rows if r["answerable"]]
    unsupported = [r for r in rows if not r["answerable"]]

    false_rejections = sum(r["key_score"] < threshold for r in answerable)
    false_acceptances = sum(r["key_score"] >= threshold for r in unsupported)
    wrong_article = sum(
        any(score >= threshold for doc_id, score in r["scores"].items() if doc_id != r["expected_id"])
        for r in answerable
    )

    return {
        "threshold": threshold,
        "false_rejections": false_rejections,
        "false_acceptances": false_acceptances,
        "wrong_article": wrong_article,
        "answerable": len(answerable),
        "unsupported": len(unsupported),
        "frr": false_rejections / len(answerable) if answerable else 0.0,
        "far": false_acceptances / len(unsupported) if unsupported else 0.0,
        "war": wrong_article / len(answerable) if answerable else 0.0,
    }


def separation(rows: list[dict]) -> dict:
    lowest_answerable = min(r["key_score"] for r in rows if r["answerable"])
    highest_unsupported = max(r["key_score"] for r in rows if not r["answerable"])
    clean = lowest_answerable > highest_unsupported

    return {
        "clean": clean,
        "lowest_answerable": lowest_answerable,
        "highest_unsupported": highest_unsupported,
        "gap": lowest_answerable - highest_unsupported,
        "midpoint": (lowest_answerable + highest_unsupported) / 2 if clean else None,
    }


def candidate_thresholds(rows: list[dict]) -> list[float]:
    """Every observed key score is a point where some decision flips."""
    return sorted({r["key_score"] for r in rows})


# --- Report ------------------------------------------------------------------


def print_report(fixture: dict, rows: list[dict]) -> None:
    print("\nSEMANTIC THRESHOLD CALIBRATION")
    print(
        f"model={fixture['model']}  documents={fixture['document_input_type']}  "
        f"queries={fixture['query_input_type']}  dims={fixture['dimensions']}"
    )

    print_score_table(rows, [doc["id"] for doc in fixture["documents"]])
    print_tradeoffs(rows)


def print_score_table(rows: list[dict], doc_ids: list[str]) -> None:
    print("\nPer-question scores (key = expected article score, or max score if unsupported)\n")
    print(f"{'question':<48} {'expect':<7} " + " ".join(f"{d:>7}" for d in doc_ids)
          + f" {'top':>7} {'margin':>7} {'key':>7}")
    for r in rows:
        flag = "" if not r["answerable"] or r["top_id"] == r["expected_id"] else "  <- top-1 wrong"
        print(
            f"{r['query'][:48]:<48} {r['expected_id'] or 'none':<7} "
            + " ".join(f"{r['scores'][d]:>7.4f}" for d in doc_ids)
            + f" {r['top_id']:>7} {r['margin']:>7.4f} {r['key_score']:>7.4f}{flag}"
        )


def print_tradeoffs(rows: list[dict]) -> None:
    print("\nThreshold trade-offs (accept article if score >= threshold)\n")
    print(f"{'threshold':>9}  {'false reject':>13}  {'false accept':>13}  {'wrong article':>13}")
    for threshold in candidate_thresholds(rows):
        m = threshold_metrics(rows, threshold)
        print(
            f"{threshold:>9.4f}  "
            f"{m['false_rejections']:>2}/{m['answerable']:<2} ({m['frr']:>4.0%})  "
            f"{m['false_acceptances']:>2}/{m['unsupported']:<2} ({m['far']:>4.0%})  "
            f"{m['wrong_article']:>2}/{m['answerable']:<2} ({m['war']:>4.0%})"
        )

    s = separation(rows)
    print("\nSeparation")
    print(f"  lowest answerable (expected article) score: {s['lowest_answerable']:.4f}")
    print(f"  highest unsupported (max article) score:    {s['highest_unsupported']:.4f}")
    if s["clean"]:
        m = threshold_metrics(rows, s["midpoint"])
        print(f"  CLEAN separation. Valid interval: ({s['highest_unsupported']:.4f}, "
              f"{s['lowest_answerable']:.4f}]  gap={s['gap']:.4f}")
        print(f"  Midpoint {s['midpoint']:.4f}: false reject {m['frr']:.0%}, false accept "
              f"{m['far']:.0%}, wrong article {m['war']:.0%}")
    else:
        print(f"  NO clean separation (overlap {-s['gap']:.4f}). Use the trade-off table.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generate", action="store_true",
                        help="create the fixture with 2 Cohere embed calls")
    args = parser.parse_args(argv)

    with historical_kb():
        return _run(args)


def _run(args) -> int:
    articles = load_articles()
    cases = load_cases(DATASET_PATH)

    try:
        if args.generate:
            fixture = generate_fixture(articles, cases, FIXTURE_PATH)
            print(f"Fixture written to {FIXTURE_PATH} using {len(fixture['embed_calls'])} embed calls:")
            for call in fixture["embed_calls"]:
                print(f"  - {call}")
        else:
            fixture = load_fixture(articles, cases, FIXTURE_PATH)
    except FixtureError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_report(fixture, score_questions(fixture, cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
