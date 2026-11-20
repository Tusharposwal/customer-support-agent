"""Offline evaluation of the rerank step inside the real answer_question().

Offline (default, no Cohere calls; original and holdout reported separately):
    python evals/evaluate_rerank_integration.py
    python evals/evaluate_rerank_integration.py --dataset holdout --thresholds 0.2 0.4

Create the holdout question embeddings (exactly 1 Cohere embed call; refuses
to overwrite; article embeddings are reused from semantic_embeddings.json):
    python evals/evaluate_rerank_integration.py --embed-holdout

How the offline run works: every question goes through the production
answer_question() with
  - retriever: the real hybrid_kb.retrieve_hybrid, semantic scores served from
    saved embeddings;
  - client: a replay client whose rerank() returns the saved Cohere scores
    (its chat() refuses to run);
  - generator: a recorder that notes which articles would reach the LLM.
So the integration code itself is measured, not a copy of it. LLM answers and
citations are NOT evaluated here.
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import calibrate_semantic as cal
import rerank_experiment as rx
from support_agent.agent import answer_question
from support_agent.answer import SupportAnswer
from support_agent.kb import load_articles
from support_agent.rerank import RerankConfig, rerank_articles

HOLDOUT_EMBEDDINGS_PATH = cal.PROJECT_ROOT / "evals" / "fixtures" / "semantic_holdout_embeddings.json"
HOLDOUT_EMBEDDINGS_VERSION = 1
HOLDOUT_QUESTIONS = 20
MAX_EMBED_CALLS = 1

# A fixed, evenly spaced grid for illustration. It was not derived from the
# data and none of these values is a recommendation.
DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)
SENT = "sent_to_generator"


class EvaluationError(RuntimeError):
    """The offline replay could not reproduce a saved input."""


# --- Holdout question embeddings (the only code that talks to Cohere) ---------


def _check_vectors(vectors, count: int, dimensions: int) -> None:
    if len(vectors) != count:
        raise cal.FixtureError(f"Expected {count} question vectors, got {len(vectors)}.")
    if any(len(vector) != dimensions for vector in vectors):
        raise cal.FixtureError(f"Question vectors do not all have {dimensions} dimensions.")


def generate_holdout_embeddings(client, model: str, articles, cases, document_fixture, path=HOLDOUT_EMBEDDINGS_PATH) -> dict:
    """Embed every holdout question in ONE request, using retrieve_semantic's settings."""
    path = Path(path)
    if path.exists():
        raise cal.FixtureError(f"{path} already exists; delete it first to regenerate.")
    if len(cases) != HOLDOUT_QUESTIONS:
        raise cal.FixtureError(f"Expected {HOLDOUT_QUESTIONS} holdout questions, found {len(cases)}.")
    if model != document_fixture["model"]:
        raise cal.FixtureError(
            f"Query model {model!r} differs from the saved article embeddings' {document_fixture['model']!r}."
        )

    request = {"model": model, "input_type": "search_query", "texts": len(cases)}
    response = client.embed(
        texts=[case["query"] for case in cases],
        model=model,
        input_type="search_query",
        embedding_types=["float"],
        request_options={"max_retries": 0},  # one call must be one HTTP request
    )
    vectors = response.embeddings.float
    _check_vectors(vectors, len(cases), document_fixture["dimensions"])

    fixture = {
        "version": HOLDOUT_EMBEDDINGS_VERSION,
        "dataset": "holdout",
        "generated_on": date.today().isoformat(),
        "model": model,
        "query_input_type": "search_query",
        "embedding_type": "float",
        "dimensions": document_fixture["dimensions"],
        "kb_sha256": cal.kb_hash(articles),
        "questions_sha256": cal.questions_hash(cases),
        "document_embeddings": cal.FIXTURE_PATH.name,
        "embed_calls": [request],
        "questions": [
            {"query": case["query"], "embedding": vector} for case, vector in zip(cases, vectors)
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(fixture), encoding="utf-8")
    os.replace(temporary, path)
    return fixture


def load_holdout_embeddings(articles, cases, document_fixture, path=HOLDOUT_EMBEDDINGS_PATH) -> dict:
    """Validate the holdout fixture and combine it with the saved article vectors."""
    path = Path(path)
    if not path.exists():
        raise cal.FixtureError(f"{path.name} not found. Run with --embed-holdout (1 Cohere call).")

    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("version") != HOLDOUT_EMBEDDINGS_VERSION:
        raise cal.FixtureError(f"Unsupported holdout embeddings version: {fixture.get('version')!r}")
    if fixture.get("dataset") != "holdout":
        raise cal.FixtureError(f"{path.name} is not a holdout fixture.")
    if fixture.get("kb_sha256") != cal.kb_hash(articles):
        raise cal.FixtureError("Knowledge base changed since the holdout embeddings were made.")
    if fixture.get("questions_sha256") != cal.questions_hash(cases):
        raise cal.FixtureError("Holdout questions changed since the embeddings were made.")
    if fixture.get("model") != document_fixture["model"] or fixture.get("query_input_type") != "search_query":
        raise cal.FixtureError("Holdout embeddings do not match the saved article embeddings' settings.")
    if [q["query"] for q in fixture["questions"]] != [case["query"] for case in cases]:
        raise cal.FixtureError("Holdout embeddings are not in dataset order.")
    _check_vectors([q["embedding"] for q in fixture["questions"]], len(cases), document_fixture["dimensions"])

    return {"documents": document_fixture["documents"], "questions": fixture["questions"]}


# --- Offline replay ------------------------------------------------------------


class ReplayClient:
    """Returns saved rerank scores. Any mismatch raises instead of guessing."""

    def __init__(self, rerank_data: dict, articles: list[dict]):
        self.model = rerank_data["model"]
        self.scores = {
            query: {item["article_id"]: item["relevance_score"] for item in entry["results"]}
            for query, entry in rerank_data["results"].items()
        }
        self.ids_by_document = {rx.document_string(a): a["id"] for a in articles}
        self.calls = []

    def rerank(self, *, model, query, documents, request_options=None, **unused):
        self.calls.append({"model": model, "query": query, "request_options": request_options})
        if unused:
            raise EvaluationError(f"Unexpected rerank arguments: {sorted(unused)}")
        if model != self.model:
            raise EvaluationError(f"Saved scores are for {self.model!r}, not {model!r}.")
        if query not in self.scores:
            raise EvaluationError(f"No saved rerank result for {query!r}.")
        try:
            ids = [self.ids_by_document[document] for document in documents]
        except KeyError:
            raise EvaluationError("A rerank document differs from the saved document format.") from None

        scores = self.scores[query]
        ranked = sorted(range(len(ids)), key=lambda i: scores[ids[i]], reverse=True)
        return SimpleNamespace(
            results=[SimpleNamespace(index=i, relevance_score=scores[ids[i]]) for i in ranked]
        )

    def chat(self, **kwargs):
        raise EvaluationError("The evaluation must never call chat.")


def configurations(model: str, thresholds) -> list[tuple[str, RerankConfig]]:
    configs = [
        ("Hybrid only", RerankConfig()),
        ("Hybrid + rerank", RerankConfig(enabled=True, model=model)),
    ]
    configs += [
        (f"Rerank, min {t:.2f}", RerankConfig(enabled=True, model=model, min_score=t))
        for t in thresholds
    ]
    return configs


def run_configuration(cases, retriever, client, config: RerankConfig) -> dict:
    """Send every question through answer_question with one configuration."""
    returned, reasons, candidates, failures = {}, Counter(), [], []

    def recording_reranker(question, articles, client_arg, config_arg):
        try:
            return rerank_articles(question, articles, client_arg, config_arg)
        except Exception as error:
            failures.append(error)
            raise

    for case in cases:
        sent = []

        def recording_generator(question, articles, client_arg):
            sent.extend(articles)
            # A well-formed placeholder answer (it cites a supplied article), so
            # the agent's final answer check accepts it.
            first = articles[0]
            source = {"id": first["id"], "title": first["title"], "url": first["url"]}
            return SupportAnswer(answer="(recorded)", sources=[source], escalate=False, reason="answered")

        def counted_retriever(question):
            articles = retriever(question)
            candidates.append(len(articles))
            return articles

        result = answer_question(
            case["query"], retriever=counted_retriever, client=client,
            generator=recording_generator, reranker=recording_reranker, rerank_config=config,
        )
        if failures:
            # The agent would fail open and hide this; an evaluation must not.
            raise EvaluationError(f"Replay failed for {case['query']!r}: {failures[0]}") from failures[0]

        returned[case["query"]] = [article["id"] for article in sent]
        # "answered" here only means the question reached the (recording) generator.
        reasons[SENT if result.reason == "answered" else result.reason] += 1

    metrics = rx.method_metrics(cases, returned)
    metrics["false_reject"] = sum(
        not returned[case["query"]] for case in cases if case["expected_id"] is not None
    )
    return {"metrics": metrics, "reasons": dict(reasons), "candidates": candidates, "returned": returned}


def evaluate(articles, cases, semantic_fixture, rerank_data, thresholds=DEFAULT_THRESHOLDS) -> dict:
    if len(rerank_data["results"]) != len(cases):
        raise EvaluationError(f"Saved rerank results are incomplete ({len(rerank_data['results'])}/{len(cases)}).")

    client = ReplayClient(rerank_data, articles)
    semantic = rx.fixture_semantic_retriever(semantic_fixture, articles)
    with rx.production_hybrid(semantic) as retrieve_hybrid:
        return {
            name: run_configuration(cases, retrieve_hybrid, client, config)
            for name, config in configurations(rerank_data["model"], thresholds)
        }


# --- Report --------------------------------------------------------------------


def print_report(dataset: str, cases, results: dict, model: str) -> None:
    answerable = sum(case["expected_id"] is not None for case in cases)
    a, u = answerable, len(cases) - answerable
    print(f"\n{dataset.upper()} ({len(cases)} questions: {a} answerable, {u} unsupported)  rerank model={model}")

    candidates = results["Hybrid only"]["candidates"]
    print(f"Hybrid retrieval passed {min(candidates)}-{max(candidates)} articles per question to the next step.\n")

    print(f"{'configuration':<18} {'top-1':>6} {'recall@3':>9} {'false reject':>13} "
          f"{'false accept':>13} {'abstain':>8} {'wrong article':>14}  hand-offs")
    for name, result in results.items():
        m = result["metrics"]
        handoffs = ", ".join(f"{r}={n}" for r, n in sorted(result["reasons"].items()) if r != SENT) or "-"
        print(f"{name:<18} {m['top1']:>3}/{a:<2} {m['recall3']:>6}/{a:<2} {m['false_reject']:>10}/{a:<2} "
              f"{m['false_accept']:>10}/{u:<2} {m['abstain']:>5}/{u:<2} {m['wrong_article']:>11}/{a:<2}  {handoffs}")


def print_caveats() -> None:
    print(
        "\nNotes\n"
        "- Metrics describe which articles would reach answer generation; the LLM and its\n"
        "  citation check (a second safety gate) are not run here.\n"
        "- The thresholds are an illustrative grid, not tuned values. All 37 questions were\n"
        "  inspected during development, so neither dataset validates a threshold. No\n"
        "  production threshold is recommended; validation needs a new, unseen question set."
    )


def _threshold(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("thresholds must be between 0 and 1")
    return number


def _dataset_inputs(name: str, articles: list[dict]):
    config = rx.DATASETS[name]
    cases = cal.load_cases(config.queries_path)
    original_cases = cal.load_cases(rx.DATASETS["original"].queries_path)
    document_fixture = cal.load_fixture(articles, original_cases, cal.FIXTURE_PATH)

    if name == "original":
        semantic_fixture = document_fixture
    else:
        semantic_fixture = load_holdout_embeddings(articles, cases, document_fixture)

    rerank_data = rx.load_results(articles, cases, config.results_path, name)
    if rerank_data is None:
        raise EvaluationError(f"{config.results_path.name} not found.")
    return cases, semantic_fixture, rerank_data


def embed_holdout(articles) -> dict:
    cases = cal.load_cases(rx.HOLDOUT_DATASET_PATH)
    original_cases = cal.load_cases(rx.DATASETS["original"].queries_path)
    document_fixture = cal.load_fixture(articles, original_cases, cal.FIXTURE_PATH)
    if HOLDOUT_EMBEDDINGS_PATH.exists():  # checked before any client is created
        raise cal.FixtureError(f"{HOLDOUT_EMBEDDINGS_PATH.name} already exists; delete it first to regenerate.")

    # Imported only here: semantic_kb loads .env and creates a client (no request).
    from support_agent import semantic_kb

    calls = []
    real_embed = semantic_kb.client.embed

    def counted_embed(**kwargs):
        if len(calls) >= MAX_EMBED_CALLS:
            raise RuntimeError(f"Refusing embed call #{len(calls) + 1}; limit is {MAX_EMBED_CALLS}.")
        calls.append(kwargs["model"])
        return real_embed(**kwargs)

    fixture = generate_holdout_embeddings(
        SimpleNamespace(embed=counted_embed), semantic_kb.MODEL, articles, cases, document_fixture,
    )
    print(f"Wrote {HOLDOUT_EMBEDDINGS_PATH.name}: {len(fixture['questions'])} vectors, "
          f"model={fixture['model']}, dims={fixture['dimensions']}, embed calls={len(calls)}")
    return fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["original", "holdout", "both"], default="both")
    parser.add_argument("--thresholds", type=_threshold, nargs="+", default=list(DEFAULT_THRESHOLDS),
                        help="minimum scores to compare (illustration only)")
    parser.add_argument("--embed-holdout", action="store_true",
                        help="create holdout question embeddings with 1 Cohere embed call")
    args = parser.parse_args(argv)

    # Saved fixtures belong to the frozen v1 knowledge base; keyword retrieval
    # inside the real hybrid retriever must read the same articles.
    with cal.historical_kb():
        return _run(args)


def _run(args) -> int:
    articles = load_articles()

    try:
        if args.embed_holdout:
            embed_holdout(articles)
            return 0

        names = ["original", "holdout"] if args.dataset == "both" else [args.dataset]
        print("RERANK INTEGRATION EVALUATION (offline replay through answer_question)")
        for name in names:
            cases, semantic_fixture, rerank_data = _dataset_inputs(name, articles)
            results = evaluate(articles, cases, semantic_fixture, rerank_data, args.thresholds)
            print_report(name, cases, results, rerank_data["model"])
    except (cal.FixtureError, rx.ResultsError, EvaluationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # API errors: type and status only, never headers or keys.
        print(f"Stopped: {type(error).__name__} (HTTP {getattr(error, 'status_code', None)})", file=sys.stderr)
        return 1

    print_caveats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
