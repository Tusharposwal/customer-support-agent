"""Cohere rerank experiment: can rerank scores separate answerable from
unsupported questions? (Experiment only; production retrieval is unchanged.)

Offline report (default, no Cohere calls):
    python evals/rerank_experiment.py

Show the plan without calling Cohere:
    python evals/rerank_experiment.py --dry-run

Fill in missing results (at most 17 rerank calls; resumes, never overwrites):
    python evals/rerank_experiment.py --run

Every command takes --dataset original (default, 17 questions) or
--dataset holdout (20 questions, separate results file, at most 20 calls).
"""

import argparse
import json
import os
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import calibrate_semantic as cal
from support_agent.kb import load_articles, retrieve

RESULTS_PATH = cal.PROJECT_ROOT / "evals" / "fixtures" / "rerank_results.json"
HOLDOUT_DATASET_PATH = cal.PROJECT_ROOT / "evals" / "rerank_holdout_queries.json"
HOLDOUT_RESULTS_PATH = cal.PROJECT_ROOT / "evals" / "fixtures" / "rerank_holdout_results.json"
RESULTS_VERSION = 1
DOCUMENT_FORMAT = "title-text-v1"  # "Title: <title>\nText: <text>", no URL

PRIMARY_MODEL = "rerank-v5.0-pro"
FALLBACK_MODEL = "rerank-v3.5"
MODEL_ENV = "COHERE_RERANK_MODEL"

EXPECTED_QUESTIONS = 17
EXPECTED_ARTICLES = 3
MAX_SUCCESSFUL_CALLS = 17
# The SDK retries 429/5xx/408/409 twice by default; one call must be one request.
REQUEST_OPTIONS = {"max_retries": 0}

_UNAVAILABLE_WORDS = (
    "not found", "not available", "unavailable", "does not exist",
    "unknown", "invalid", "not supported", "no access",
)


class ResultsError(ValueError):
    """Saved rerank results are malformed, out of date or inconsistent."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    queries_path: Path
    results_path: Path
    expected_questions: int
    max_calls: int


DATASETS = {
    "original": DatasetConfig("original", cal.DATASET_PATH, RESULTS_PATH, 17, 17),
    "holdout": DatasetConfig("holdout", HOLDOUT_DATASET_PATH, HOLDOUT_RESULTS_PATH, 20, 20),
}
DEFAULT_DATASET = "original"


# --- Inputs ------------------------------------------------------------------


def document_string(article: dict) -> str:
    return f"Title: {article['title']}\nText: {article['text']}"


def dataset_hash(cases: list[dict]) -> str:
    """Questions plus labels. Informational: labels are read live."""
    return cal._sha256_json(cases)


def configured_model() -> str:
    return os.getenv(MODEL_ENV) or PRIMARY_MODEL


def is_model_unavailable(error: Exception) -> bool:
    """True only for a 400/404 whose error body is about the model."""
    if getattr(error, "status_code", None) not in (400, 404):
        return False
    body = str(getattr(error, "body", "")).lower()
    return "model" in body and any(word in body for word in _UNAVAILABLE_WORDS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Saved results -----------------------------------------------------------


def new_results(articles: list[dict], cases: list[dict], dataset: str = DEFAULT_DATASET) -> dict:
    return {
        "version": RESULTS_VERSION,
        "dataset": dataset,
        "document_format": DOCUMENT_FORMAT,
        "model": None,
        "kb_sha256": cal.kb_hash(articles),
        "questions_sha256": cal.questions_hash(cases),
        "dataset_sha256": dataset_hash(cases),
        "created_at": _now(),
        "results": {},
    }


def load_results(
    articles: list[dict],
    cases: list[dict],
    path: Path = RESULTS_PATH,
    dataset: str = DEFAULT_DATASET,
) -> dict | None:
    """Return validated saved results, or None if there is no file."""
    path = Path(path)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("version") != RESULTS_VERSION:
        raise ResultsError(f"Unsupported results version: {data.get('version')!r}")
    # Milestone 3's file predates the "dataset" field; it is the original dataset.
    saved_dataset = data.get("dataset", DEFAULT_DATASET)
    if saved_dataset != dataset:
        raise ResultsError(f"{path.name} holds {saved_dataset!r} results, not {dataset!r}.")
    if data.get("document_format") != DOCUMENT_FORMAT:
        raise ResultsError("Document format changed since the results were saved.")
    if data.get("kb_sha256") != cal.kb_hash(articles):
        raise ResultsError("Knowledge base changed since the results were saved.")
    if data.get("questions_sha256") != cal.questions_hash(cases):
        raise ResultsError("Question list changed since the results were saved.")

    queries = {case["query"] for case in cases}
    ids = [article["id"] for article in articles]

    for query, entry in data.get("results", {}).items():
        where = f"Saved result for {query!r}"
        if query not in queries or entry.get("query") != query:
            raise ResultsError(f"{where} does not match the dataset.")
        if entry.get("model") != data.get("model"):
            raise ResultsError(f"{where} used a different model.")

        returned = entry.get("results", [])
        if sorted(item.get("index") for item in returned) != list(range(len(ids))):
            raise ResultsError(f"{where} does not rank every article exactly once.")
        for item in returned:
            if item.get("article_id") != ids[item["index"]]:
                raise ResultsError(f"{where} maps index {item['index']} to the wrong article.")
            if not isinstance(item.get("relevance_score"), (int, float)):
                raise ResultsError(f"{where} has a non-numeric relevance score.")

    return data


def save_results(data: dict, path: Path = RESULTS_PATH) -> None:
    """Write atomically so an interrupted run never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


# --- Live run (the only code that talks to Cohere) ---------------------------


def choose_model(saved: dict | None, configured: str) -> tuple[str, bool]:
    """Return (model to use, whether the approved fallback is still allowed)."""
    locked = saved.get("model") if saved else None

    if locked is None:
        return configured, configured == PRIMARY_MODEL
    if locked == configured or (configured == PRIMARY_MODEL and locked == FALLBACK_MODEL):
        return locked, False
    raise ResultsError(
        f"Saved results use {locked!r} but {MODEL_ENV} resolves to {configured!r}; "
        "mixing models would make scores incomparable."
    )


def preflight(
    articles: list[dict],
    cases: list[dict],
    saved: dict | None,
    configured: str,
    *,
    expected_questions: int | None = None,
    max_calls: int | None = None,
) -> dict:
    """Check sizes and limits before any request. Defaults are the original
    dataset's module-level limits."""
    expected_questions = EXPECTED_QUESTIONS if expected_questions is None else expected_questions
    max_calls = MAX_SUCCESSFUL_CALLS if max_calls is None else max_calls

    if len(articles) != EXPECTED_ARTICLES:
        raise ResultsError(f"Expected {EXPECTED_ARTICLES} articles, found {len(articles)}.")
    if len(cases) != expected_questions:
        raise ResultsError(f"Expected {expected_questions} questions, found {len(cases)}.")

    done = set(saved["results"]) if saved else set()
    missing = [case["query"] for case in cases if case["query"] not in done]
    if len(missing) > max_calls:
        raise ResultsError(f"{len(missing)} calls needed, limit is {max_calls}.")

    model, fallback_allowed = choose_model(saved, configured)
    return {
        "articles": len(articles),
        "questions": len(cases),
        "saved": len(done),
        "missing": missing,
        "model": model,
        "fallback": FALLBACK_MODEL if fallback_allowed else None,
        "max_calls": max_calls,
    }


def _rerank(client, model: str, query: str, documents: list[str], clock):
    start = clock()
    response = client.rerank(
        model=model,
        query=query,
        documents=documents,
        request_options=REQUEST_OPTIONS,
    )
    return response, clock() - start


def _entry(case: dict, model: str, response, latency: float, articles: list[dict]) -> dict:
    returned = [
        {
            "index": item.index,
            "article_id": articles[item.index]["id"],
            "relevance_score": float(item.relevance_score),
        }
        for item in response.results
    ]
    if sorted(item["index"] for item in returned) != list(range(len(articles))):
        raise ResultsError(f"Rerank response for {case['query']!r} did not rank every article once.")

    billed = getattr(getattr(response, "meta", None), "billed_units", None)
    return {
        "query": case["query"],
        "expected_id": case["expected_id"],
        "model": model,
        "results": returned,
        "search_units": getattr(billed, "search_units", None),
        "latency_ms": round(latency * 1000, 1),
        "completed_at": _now(),
    }


def run(
    client,
    articles,
    cases,
    configured,
    path=RESULTS_PATH,
    *,
    dataset: str = DEFAULT_DATASET,
    expected_questions: int | None = None,
    max_calls: int | None = None,
    clock=time.perf_counter,
    log=print,
) -> dict:
    """Rerank every question without a saved result. Returns call statistics.

    Any error stops the run. Results completed before the error are kept.
    """
    saved = load_results(articles, cases, path, dataset)
    plan = preflight(
        articles, cases, saved, configured,
        expected_questions=expected_questions, max_calls=max_calls,
    )
    max_calls = plan["max_calls"]
    data = saved or new_results(articles, cases, dataset)
    model, fallback = plan["model"], plan["fallback"]
    documents = [document_string(article) for article in articles]
    stats = {"successful": 0, "failed": 0, "fallback_used": False}

    for case in cases:
        if case["query"] in data["results"]:
            continue
        if stats["successful"] >= max_calls:
            raise ResultsError("Call limit reached; refusing further requests.")

        try:
            response, latency = _rerank(client, model, case["query"], documents, clock)
        except Exception as error:
            stats["failed"] += 1
            if not (fallback and not data["results"] and is_model_unavailable(error)):
                raise
            log(f"{model} is unavailable (HTTP {error.status_code}); "
                f"switching to approved fallback {fallback}.")
            model, fallback, stats["fallback_used"] = fallback, None, True
            try:
                response, latency = _rerank(client, model, case["query"], documents, clock)
            except Exception:
                stats["failed"] += 1
                raise

        stats["successful"] += 1
        data["model"] = model
        data["results"][case["query"]] = _entry(case, model, response, latency, articles)
        save_results(data, path)
        log(f"[{stats['successful']:>2}] {case['query']}")

    stats["model"] = model
    return stats


# --- Offline retrieval methods -----------------------------------------------


def fixture_semantic_retriever(fixture: dict, articles: list[dict]):
    """Same contract as semantic_kb.retrieve_semantic, using saved vectors."""
    vectors = {q["query"]: q["embedding"] for q in fixture["questions"]}
    doc_vectors = {d["id"]: d["embedding"] for d in fixture["documents"]}

    def retrieve_semantic(query: str, top_k: int = 3) -> list[dict]:
        if not query.strip() or top_k <= 0:
            return []
        scored = [
            {**article, "score": cal.cosine_similarity(vectors[query], doc_vectors[article["id"]])}
            for article in articles
        ]
        scored.sort(key=lambda article: article["score"], reverse=True)
        return scored[:top_k]

    return retrieve_semantic


@contextmanager
def production_hybrid(semantic_retriever):
    """Yield the real hybrid_kb.retrieve_hybrid with semantic search served from
    saved vectors. semantic_kb (API key, Cohere client) is never imported."""
    import support_agent

    names = ("support_agent.semantic_kb", "support_agent.hybrid_kb")
    saved_modules = {name: sys.modules.get(name, _MISSING) for name in names}
    saved_attrs = {name: getattr(support_agent, name, _MISSING) for name in ("semantic_kb", "hybrid_kb")}

    if sys.modules.get("support_agent.semantic_kb") is None:
        stub = types.ModuleType("support_agent.semantic_kb")
        stub.retrieve_semantic = semantic_retriever
        sys.modules["support_agent.semantic_kb"] = stub

    try:
        from support_agent import hybrid_kb

        original = hybrid_kb.retrieve_semantic
        hybrid_kb.retrieve_semantic = semantic_retriever
        try:
            yield hybrid_kb.retrieve_hybrid
        finally:
            hybrid_kb.retrieve_semantic = original
    finally:
        for name, module in saved_modules.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name, value in saved_attrs.items():
            if value is _MISSING:
                if hasattr(support_agent, name):
                    delattr(support_agent, name)
            else:
                setattr(support_agent, name, value)


_MISSING = object()


def rerank_rows(data: dict, cases: list[dict]) -> list[dict]:
    return [
        cal.build_row(
            case,
            {item["article_id"]: item["relevance_score"] for item in data["results"][case["query"]]["results"]},
        )
        for case in cases
    ]


def ranked_ids(rows: list[dict]) -> dict[str, list[str]]:
    return {
        row["query"]: [doc_id for doc_id, _ in sorted(row["scores"].items(), key=lambda i: i[1], reverse=True)]
        for row in rows
    }


def method_metrics(cases: list[dict], returned: dict[str, list[str]]) -> dict:
    """Same definitions as the semantic calibration (an article returned = accepted)."""
    m = {"top1": 0, "recall3": 0, "wrong_article": 0, "false_accept": 0, "abstain": 0,
         "answerable": 0, "unsupported": 0}

    for case in cases:
        ids, expected = returned[case["query"]], case["expected_id"]
        if expected is None:
            m["unsupported"] += 1
            m["false_accept" if ids else "abstain"] += 1
        else:
            m["answerable"] += 1
            m["top1"] += ids[:1] == [expected]
            m["recall3"] += expected in ids[:3]
            m["wrong_article"] += any(doc_id != expected for doc_id in ids)

    return m


def compare_methods(articles, cases, embedding_fixture, rerank_data) -> dict[str, dict]:
    """Semantic and hybrid need saved question vectors; without an embedding
    fixture (holdout) only keyword and rerank are compared."""
    queries = [case["query"] for case in cases]
    returned = {"Keyword": {q: [a["id"] for a in retrieve(q, top_k=3)] for q in queries}}

    if embedding_fixture is not None:
        semantic = fixture_semantic_retriever(embedding_fixture, articles)
        returned["Semantic"] = {q: [a["id"] for a in semantic(q, top_k=3)] for q in queries}
        with production_hybrid(semantic) as retrieve_hybrid:
            returned["Hybrid"] = {q: [a["id"] for a in retrieve_hybrid(q, top_k=3)] for q in queries}

    returned["Rerank"] = ranked_ids(rerank_rows(rerank_data, cases))

    return {name: method_metrics(cases, ids) for name, ids in returned.items()}


# --- Report ------------------------------------------------------------------


def usage_summary(data: dict) -> dict:
    entries = list(data["results"].values())
    latencies = [e["latency_ms"] for e in entries]
    units = [e["search_units"] for e in entries if e["search_units"] is not None]
    return {
        "calls": len(entries),
        "model": data["model"],
        "search_units": sum(units) if units else None,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
    }


def print_report(articles, cases, embedding_fixture, data) -> None:
    usage = usage_summary(data)
    print("\nRERANK EXPERIMENT")
    print(f"model={usage['model']}  document_format={data['document_format']}  "
          f"saved results={usage['calls']}/{len(cases)}")
    print(f"billed search units={usage['search_units']}  latency avg="
          f"{usage['avg_latency_ms']:.1f} ms  max={usage['max_latency_ms']:.1f} ms")

    if embedding_fixture is None:
        print("\nSemantic and hybrid are not shown: no saved embeddings exist for these "
              "questions (they would need embedding calls).")
    print("\nMethod comparison (no thresholds; any returned article counts as accepted)\n")
    print(f"{'method':<9} {'top-1':>7} {'recall@3':>9} {'false accept':>13} {'abstain':>8} {'wrong article':>14}")
    for name, m in compare_methods(articles, cases, embedding_fixture, data).items():
        a, u = m["answerable"], m["unsupported"]
        print(f"{name:<9} {m['top1']:>4}/{a:<2} {m['recall3']:>6}/{a:<2} {m['false_accept']:>10}/{u:<2} "
              f"{m['abstain']:>5}/{u:<2} {m['wrong_article']:>11}/{a:<2}")

    rows = rerank_rows(data, cases)
    print("\nRerank relevance scores")
    cal.print_score_table(rows, [article["id"] for article in articles])
    cal.print_tradeoffs(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="show the plan; no Cohere calls")
    mode.add_argument("--run", action="store_true", help="rerank missing questions (Cohere calls)")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT_DATASET,
                        help="question set to use (default: original)")
    args = parser.parse_args(argv)

    # The saved results were made against the frozen v1 knowledge base.
    with cal.historical_kb():
        return _run(args)


def _run(args) -> int:
    config = DATASETS[args.dataset]
    limits = {"expected_questions": config.expected_questions, "max_calls": config.max_calls}
    articles = load_articles()
    cases = cal.load_cases(config.queries_path)

    try:
        saved = load_results(articles, cases, config.results_path, config.name)

        if args.dry_run or args.run:
            from dotenv import load_dotenv

            load_dotenv()  # Same as the app; only COHERE_RERANK_MODEL is read here.
            plan = preflight(articles, cases, saved, configured_model(), **limits)
            print(f"dataset={config.name}  queries={config.queries_path.name}  "
                  f"results={config.results_path.name}")
            print(f"articles={plan['articles']}  questions={plan['questions']}  "
                  f"saved={plan['saved']}  to request={len(plan['missing'])}")
            print(f"model={plan['model']}  fallback={plan['fallback']}  document_format={DOCUMENT_FORMAT}  "
                  f"max successful calls={plan['max_calls']}  SDK retries=0")
            if args.dry_run:
                return 0

            from support_agent.agent import _default_client  # existing ClientV2 factory

            stats = run(_default_client(), articles, cases, configured_model(), config.results_path,
                        dataset=config.name, **limits)
            print(f"successful calls={stats['successful']}  failed calls={stats['failed']}  "
                  f"model={stats['model']}  fallback used={stats['fallback_used']}")
            saved = load_results(articles, cases, config.results_path, config.name)

        if not saved or len(saved["results"]) != len(cases):
            done = len(saved["results"]) if saved else 0
            print(f"Rerank results incomplete ({done}/{len(cases)}). Run with --run.", file=sys.stderr)
            return 1

        # Saved embeddings exist only for the original questions.
        embedding_fixture = (
            cal.load_fixture(articles, cases, cal.FIXTURE_PATH) if config.name == "original" else None
        )
    except (ResultsError, cal.FixtureError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # API errors: report type and status, never headers.
        status = getattr(error, "status_code", None)
        body = str(getattr(error, "body", ""))[:300]
        print(f"Stopped: {type(error).__name__} (HTTP {status}) {body}", file=sys.stderr)
        return 1

    print_report(articles, cases, embedding_fixture, saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
