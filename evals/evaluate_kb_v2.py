"""Evaluate retrieval and answer quality on the KB v2 question set.

Retrieval and answer quality are measured and reported separately.

Retrieval (offline by default, no Cohere calls):
    python evals/evaluate_kb_v2.py
    python evals/evaluate_kb_v2.py --k 5
    python evals/evaluate_kb_v2.py --retriever hybrid    # LIVE: Cohere embed calls

Answers (offline scoring of a recorded run):
    python evals/evaluate_kb_v2.py --answers evals/results/answers.json

Record answers (LIVE: one answer_question() per question, i.e. Cohere embed
and chat calls; refuses to overwrite an existing file):
    python evals/evaluate_kb_v2.py --record-answers evals/results/answers.json

Dataset: evals/support_eval_v2.json, labelled against data/kb.json (KB v2).
The v1 question sets (retrieval_queries.json, rerank_holdout_queries.json)
are labelled against the frozen 3-article KB and are not used here.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from support_agent.kb import load_articles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evals" / "support_eval_v2.json"

DEFAULT_K = 3  # Production retrieval passes the top 3 articles to the LLM.
ANSWERS_VERSION = 1

CATEGORIES = (
    "account",
    "password_login",
    "orders_shipping",
    "returns_refunds",
    "payments_billing",
    "cancellations",
    "delivery_problems",
    "product_info",
    "technical",
)
LABELS = ("answerable", "unsupported", "ambiguous")
# Edge-case tags describe the wording; scope tags say why a question is unsupported.
EDGE_TAGS = ("typo", "multi_intent", "negation", "boundary", "informal", "short", "paraphrase", "adversarial")
SCOPE_TAGS = ("out_of_domain", "not_covered", "near_miss")
TAGS = EDGE_TAGS + SCOPE_TAGS

ARTICLE_ID = re.compile(r"^KB-\d{3}$")
QUESTION_ID = re.compile(r"^V2-\d{3}$")

_FIELDS = {
    "answerable": {"id", "query", "label", "category", "expected_ids", "also_relevant_ids", "answer_facts", "tags"},
    "unsupported": {"id", "query", "label", "category", "related_ids", "tags", "note"},
    "ambiguous": {"id", "query", "label", "category", "related_ids", "tags", "note"},
}


class DatasetError(ValueError):
    """The KB or the question set breaks a labelling rule."""


# --- Validation ----------------------------------------------------------------


def normalise_query(query: str) -> str:
    """Lowercase, punctuation removed: catches near-identical duplicates."""
    return " ".join(re.findall(r"\w+", query.lower()))


def validate_kb(articles: list[dict]) -> None:
    """Rules for KB v2 on top of kb.load_articles()'s own checks."""
    seen_urls, seen_titles = set(), set()
    for article in articles:
        where = f"Article {article.get('id')!r}"
        if not ARTICLE_ID.match(article["id"]):
            raise DatasetError(f"{where}: id must look like KB-001")
        if article.get("category") not in CATEGORIES:
            raise DatasetError(f"{where}: category must be one of {', '.join(CATEGORIES)}")
        for field, seen in (("url", seen_urls), ("title", seen_titles)):
            key = article[field].strip().lower()
            if key in seen:
                raise DatasetError(f"{where}: duplicate {field}")
            seen.add(key)


def _id_list(case: dict, field: str, kb_ids: set[str], where: str, required: bool) -> list[str]:
    ids = case[field]
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
        raise DatasetError(f"{where}: {field} must be a list of article ids")
    if required and not ids:
        raise DatasetError(f"{where}: {field} must not be empty")
    if len(set(ids)) != len(ids):
        raise DatasetError(f"{where}: {field} has duplicates")
    unknown = [i for i in ids if i not in kb_ids]
    if unknown:
        raise DatasetError(f"{where}: {field} has unknown article id(s): {', '.join(unknown)}")
    return ids


def validate_cases(cases, articles: list[dict]) -> None:
    """Check structure, ids and label rules. Raises DatasetError on the first problem."""
    if not isinstance(cases, list) or not cases:
        raise DatasetError("The dataset must be a non-empty JSON list")

    kb_ids = {article["id"] for article in articles}
    seen_ids, seen_queries = set(), set()

    for index, case in enumerate(cases):
        where = f"Question #{index}"
        if not isinstance(case, dict):
            raise DatasetError(f"{where} must be a JSON object")
        label = case.get("label")
        if label not in LABELS:
            raise DatasetError(f"{where}: label must be one of {', '.join(LABELS)}")
        if set(case) != _FIELDS[label]:
            missing = sorted(_FIELDS[label] - set(case))
            extra = sorted(set(case) - _FIELDS[label])
            raise DatasetError(f"{where} ({label}): missing {missing}, unexpected {extra}")

        if not isinstance(case["id"], str) or not QUESTION_ID.match(case["id"]):
            raise DatasetError(f"{where}: id must look like V2-001")
        where = f"Question {case['id']}"
        if case["id"] in seen_ids:
            raise DatasetError(f"{where}: duplicate id")
        seen_ids.add(case["id"])

        if not isinstance(case["query"], str) or not case["query"].strip():
            raise DatasetError(f"{where}: query must be a non-empty string")
        normalised = normalise_query(case["query"])
        if normalised in seen_queries:
            raise DatasetError(f"{where}: duplicate query")
        seen_queries.add(normalised)

        if case["category"] not in CATEGORIES + ("none",):
            raise DatasetError(f"{where}: unknown category {case['category']!r}")
        if not isinstance(case["tags"], list) or any(tag not in TAGS for tag in case["tags"]):
            raise DatasetError(f"{where}: tags must be a list drawn from {', '.join(TAGS)}")

        if label == "answerable":
            if case["category"] == "none":
                raise DatasetError(f"{where}: an answerable question needs a category")
            expected = _id_list(case, "expected_ids", kb_ids, where, required=True)
            also = _id_list(case, "also_relevant_ids", kb_ids, where, required=False)
            if set(expected) & set(also):
                raise DatasetError(f"{where}: an id is both expected and also relevant")
            if "multi_intent" in case["tags"] and len(expected) < 2:
                raise DatasetError(f"{where}: a multi_intent question needs at least 2 expected ids")
            facts = case["answer_facts"]
            if not isinstance(facts, list) or not facts or any(
                not isinstance(group, list) or not group
                or any(not isinstance(alt, str) or not alt.strip() for alt in group)
                for group in facts
            ):
                raise DatasetError(f"{where}: answer_facts must be a non-empty list of non-empty string lists")
            if set(case["tags"]) & set(SCOPE_TAGS):
                raise DatasetError(f"{where}: scope tags are only for unsupported questions")
        else:
            _id_list(case, "related_ids", kb_ids, where, required=False)
            if not isinstance(case["note"], str) or not case["note"].strip():
                raise DatasetError(f"{where}: {label} questions must explain the label in 'note'")
            if label == "unsupported" and len(set(case["tags"]) & set(SCOPE_TAGS)) != 1:
                raise DatasetError(f"{where}: unsupported questions need exactly one of {', '.join(SCOPE_TAGS)}")
            if label == "ambiguous" and set(case["tags"]) & set(SCOPE_TAGS):
                raise DatasetError(f"{where}: scope tags are only for unsupported questions")


def load_cases(path: Path = DATASET_PATH) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


# --- Retrieval metrics -----------------------------------------------------------


def run_retriever(cases: list[dict], retriever, k: int) -> dict[str, list[str]]:
    """Question id -> ranked article ids returned by `retriever(query, top_k=k)`."""
    return {case["id"]: [a["id"] for a in retriever(case["query"], top_k=k)][:k] for case in cases}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def retrieval_metrics(cases: list[dict], returned: dict[str, list[str]], k: int) -> dict:
    """Ranking metrics for answerable questions; abstention for the others.

    expected_ids are the articles needed to answer; also_relevant_ids also
    contain the answer. Recall counts expected only; precision, top-1 and MRR
    count either as relevant.
    """
    answerable = [c for c in cases if c["label"] == "answerable"]
    rows = []
    for case in answerable:
        ids = returned[case["id"]][:k]
        expected = set(case["expected_ids"])
        relevant = expected | set(case["also_relevant_ids"])
        first = next((rank for rank, doc in enumerate(ids, start=1) if doc in relevant), None)
        rows.append({
            "case": case,
            "top1": bool(ids) and ids[0] in relevant,
            "hit": bool(expected & set(ids)),
            "covered": expected <= set(ids),
            "recall": len(expected & set(ids)) / len(expected),
            "precision": len(relevant & set(ids)) / k,
            "returned_precision": len(relevant & set(ids)) / len(ids) if ids else None,
            "rr": 1 / first if first else 0.0,
            "empty": not ids,
        })

    def summary(selected):
        return {
            "questions": len(selected),
            "top1": sum(r["top1"] for r in selected),
            "hit": sum(r["hit"] for r in selected),
            "covered": sum(r["covered"] for r in selected),
            "recall": _mean(r["recall"] for r in selected),
            "precision": _mean(r["precision"] for r in selected),
            "returned_precision": _mean(r["returned_precision"] for r in selected if r["returned_precision"] is not None),
            "mrr": _mean(r["rr"] for r in selected),
            "no_result": sum(r["empty"] for r in selected),
        }

    by_category = {
        category: summary([r for r in rows if r["case"]["category"] == category])
        for category in CATEGORIES
        if any(r["case"]["category"] == category for r in rows)
    }
    by_tag = {
        tag: summary([r for r in rows if tag in r["case"]["tags"]])
        for tag in EDGE_TAGS
        if any(tag in r["case"]["tags"] for r in rows)
    }

    unsupported = [c for c in cases if c["label"] == "unsupported"]
    scope = {
        tag: {
            "questions": sum(tag in c["tags"] for c in unsupported),
            "abstained": sum(tag in c["tags"] and not returned[c["id"]] for c in unsupported),
        }
        for tag in SCOPE_TAGS
    }
    ambiguous = [c for c in cases if c["label"] == "ambiguous"]

    return {
        "k": k,
        "answerable": summary(rows),
        "by_category": by_category,
        "by_tag": by_tag,
        "misses": [(r["case"]["id"], r["case"]["query"], r["case"]["expected_ids"], returned[r["case"]["id"]])
                   for r in rows if not r["hit"]],
        "unsupported": {
            "questions": len(unsupported),
            "abstained": sum(not returned[c["id"]] for c in unsupported),
            "by_scope": scope,
        },
        "ambiguous": {
            "questions": len(ambiguous),
            "returned_any": sum(bool(returned[c["id"]]) for c in ambiguous),
        },
    }


# --- Answer metrics --------------------------------------------------------------


def _normalise_text(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    return " ".join(text.split())


def fact_present(answer: str, alternatives: list[str]) -> bool:
    """True if any alternative appears in the answer as whole words."""
    text = _normalise_text(answer)
    return any(
        re.search(rf"(?<!\w){re.escape(_normalise_text(alt))}(?!\w)", text) for alt in alternatives
    )


def sha256_json(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_answers(path: Path, cases: list[dict], articles: list[dict]) -> dict:
    """Load a recorded run; reject it if the questions or the KB have changed."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != ANSWERS_VERSION:
        raise DatasetError(f"Unsupported answers file version: {data.get('version')!r}")
    if data.get("dataset_sha256") != sha256_json(cases):
        raise DatasetError("The question set changed since these answers were recorded.")
    if data.get("kb_sha256") != sha256_json(articles):
        raise DatasetError("The knowledge base changed since these answers were recorded.")
    results = data.get("results")
    missing = [c["id"] for c in cases if not isinstance(results, dict) or c["id"] not in results]
    if missing:
        raise DatasetError(f"Answers are missing for {len(missing)} question(s), e.g. {missing[0]}.")
    return data


def answer_metrics(cases: list[dict], results: dict[str, dict], articles: list[dict]) -> dict:
    """Score recorded agent outputs. Fact matching is a proxy for correctness."""
    by_id = {article["id"]: article for article in articles}
    out = {"answerable": Counter(), "unsupported": Counter(), "ambiguous": Counter(),
           "reasons": {label: Counter() for label in LABELS},
           "fact_recall": [], "review": [], "false_answers": []}

    for case in cases:
        result = results[case["id"]]
        label = case["label"]
        answered = result.get("escalate") is False
        sources = result.get("sources") or []
        out["reasons"][label][result.get("reason")] += 1

        valid = answered and bool(sources) and all(
            isinstance(s, dict) and s.get("id") in by_id
            and s.get("title") == by_id[s["id"]]["title"] and s.get("url") == by_id[s["id"]]["url"]
            for s in sources
        )

        if label == "answerable":
            counts = out["answerable"]
            counts["questions"] += 1
            if not answered:
                counts["handed_off"] += 1
                continue
            counts["answered"] += 1
            counts["valid_citations"] += valid
            relevant = set(case["expected_ids"]) | set(case["also_relevant_ids"])
            supported = valid and bool(relevant & {s["id"] for s in sources})
            counts["supporting_cited"] += supported
            found = [fact_present(result.get("answer") or "", group) for group in case["answer_facts"]]
            out["fact_recall"].append(sum(found) / len(found))
            counts["all_facts"] += all(found)
            counts["correct_proxy"] += supported and all(found)
        elif label == "unsupported":
            counts = out["unsupported"]
            counts["questions"] += 1
            counts["abstained" if not answered else "answered"] += 1
            if answered:
                out["false_answers"].append(case["id"])
        else:
            counts = out["ambiguous"]
            counts["questions"] += 1
            counts["handed_off" if not answered else "answered"] += 1
            if answered:
                out["review"].append(case["id"])

    out["mean_fact_recall"] = _mean(out["fact_recall"])
    return out


def record_answers(cases: list[dict], articles: list[dict], path: Path) -> dict:
    """LIVE: run the production agent on every question and save its outputs."""
    path = Path(path)
    if path.exists():
        raise DatasetError(f"{path} already exists; choose a new file name.")

    # Imported here so offline runs never create a Cohere client.
    from support_agent.agent import _default_client, answer_question
    from support_agent.answer import DEFAULT_MODEL
    from support_agent.rerank import config_from_env

    _default_client()  # Loads local settings the same way the app does.
    rerank = config_from_env()
    results = {}
    for number, case in enumerate(cases, start=1):
        result = answer_question(case["query"])
        results[case["id"]] = {
            "answer": result.answer, "sources": result.sources,
            "escalate": result.escalate, "reason": result.reason,
        }
        print(f"[{number}/{len(cases)}] {case['id']} {result.reason}")

    data = {
        "version": ANSWERS_VERSION,
        "recorded_on": date.today().isoformat(),
        "chat_model": os.getenv("COHERE_CHAT_MODEL") or DEFAULT_MODEL,
        "rerank": {"enabled": rerank.enabled, "model": rerank.model, "min_score": rerank.min_score},
        "dataset_sha256": sha256_json(cases),
        "kb_sha256": sha256_json(articles),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return data


# --- Reports ---------------------------------------------------------------------


def _pct(part: float, whole: float) -> str:
    return f"{part / whole:.0%}" if whole else "n/a"


def print_dataset_summary(cases: list[dict], articles: list[dict]) -> None:
    labels = Counter(c["label"] for c in cases)
    categories = Counter(a["category"] for a in articles)
    print(f"KB: {len(articles)} articles  " + ", ".join(f"{c}={categories[c]}" for c in CATEGORIES))
    print(f"Questions: {len(cases)}  " + ", ".join(f"{label}={labels[label]}" for label in LABELS))


def print_retrieval_report(name: str, metrics: dict) -> None:
    k, a = metrics["k"], metrics["answerable"]
    n = a["questions"]
    print(f"\nRETRIEVAL ({name}, K={k})  -- which articles would reach the LLM; no answers are generated")
    print(f"Answerable ({n})")
    print(f"  Top-1 relevant          {a['top1']:>3}/{n}  ({_pct(a['top1'], n)})")
    print(f"  Expected article @{k}     {a['hit']:>3}/{n}  ({_pct(a['hit'], n)})")
    print(f"  All expected @{k}         {a['covered']:>3}/{n}  ({_pct(a['covered'], n)})")
    print(f"  Recall@{k}                {a['recall']:.3f}")
    print(f"  Precision@{k}             {a['precision']:.3f}  (relevant / {k})")
    print(f"  Precision of returned   {a['returned_precision']:.3f}  (relevant / articles returned)")
    print(f"  MRR@{k}                   {a['mrr']:.3f}")
    print(f"  No article returned     {a['no_result']:>3}/{n}")

    print(f"\n  {'category':<18} {'n':>3} {'hit@' + str(k):>7} {'recall':>7} {'top-1':>6}")
    for category, s in metrics["by_category"].items():
        print(f"  {category:<18} {s['questions']:>3} {s['hit']:>7} {s['recall']:>7.2f} {s['top1']:>6}")
    print(f"\n  {'edge-case tag':<18} {'n':>3} {'hit@' + str(k):>7} {'recall':>7}")
    for tag, s in metrics["by_tag"].items():
        print(f"  {tag:<18} {s['questions']:>3} {s['hit']:>7} {s['recall']:>7.2f}")

    u = metrics["unsupported"]
    print(f"\nUnsupported ({u['questions']}): no article returned {u['abstained']}/{u['questions']} "
          f"({_pct(u['abstained'], u['questions'])})")
    for tag, s in u["by_scope"].items():
        print(f"  {tag:<14} {s['abstained']}/{s['questions']}")
    m = metrics["ambiguous"]
    print(f"Ambiguous ({m['questions']}): returned at least one article {m['returned_any']}/{m['questions']} "
          "(not scored: retrieval cannot resolve ambiguity)")

    if metrics["misses"]:
        print(f"\nAnswerable questions whose expected article was not in the top {k}:")
        for case_id, query, expected, got in metrics["misses"]:
            print(f"  {case_id} {query[:60]!r} expected={expected} got={got}")


def print_answer_report(metrics: dict, meta: dict) -> None:
    a, u, m = metrics["answerable"], metrics["unsupported"], metrics["ambiguous"]
    print(f"\nANSWERS (recorded {meta.get('recorded_on')}, chat model={meta.get('chat_model')}, "
          f"rerank={meta.get('rerank')})")
    n = a["questions"]
    print(f"Answerable ({n})")
    print(f"  Answered                {a['answered']:>3}/{n}  ({_pct(a['answered'], n)})")
    print(f"  Handed off (false)      {a['handed_off']:>3}/{n}")
    print(f"  Valid citations         {a['valid_citations']:>3}/{a['answered']} answered")
    print(f"  Supporting article cited{a['supporting_cited']:>4}/{a['answered']} answered")
    print(f"  All key facts present   {a['all_facts']:>3}/{a['answered']} answered "
          f"(mean fact recall {metrics['mean_fact_recall']:.2f})")
    print(f"  Correct (proxy)         {a['correct_proxy']:>3}/{n}  "
          "(answered, supporting article cited, all key facts present)")
    print(f"Unsupported ({u['questions']}): abstained {u['abstained']}/{u['questions']}, "
          f"answered {u['answered']} {metrics['false_answers'] or ''}")
    print(f"Ambiguous ({m['questions']}): handed off {m['handed_off']}, "
          f"answered {m['answered']} -> review manually: {metrics['review'] or '-'}")
    for label, reasons in metrics["reasons"].items():
        print(f"  reasons ({label}): " + ", ".join(f"{r}={c}" for r, c in sorted(reasons.items(), key=str)))
    print("\nKey-fact matching is a string proxy: it can miss correct paraphrases and "
          "cannot detect added wrong claims. Answered ambiguous questions need human review.")


def _retriever(name: str):
    if name == "keyword":
        from support_agent.kb import retrieve
        return retrieve
    if name == "semantic":  # LIVE: imports the Cohere embedding client.
        from support_agent.semantic_kb import retrieve_semantic
        return retrieve_semantic
    from support_agent.hybrid_kb import retrieve_hybrid  # LIVE as well.
    return retrieve_hybrid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", choices=["keyword", "semantic", "hybrid"], default="keyword",
                        help="keyword is offline; semantic and hybrid call Cohere embed")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="cut-off for @K metrics (default 3)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--answers", type=Path, help="score a recorded answers file (offline)")
    action.add_argument("--record-answers", type=Path, help="LIVE: record agent answers to a new file")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least 1")

    try:
        articles = load_articles()
        validate_kb(articles)
        cases = load_cases()
        validate_cases(cases, articles)
        print_dataset_summary(cases, articles)

        if args.record_answers:
            record_answers(cases, articles, args.record_answers)
            print(f"Saved answers to {args.record_answers}")
            return 0
        if args.answers:
            data = load_answers(args.answers, cases, articles)
            print_answer_report(answer_metrics(cases, data["results"], articles), data)
            return 0

        returned = run_retriever(cases, _retriever(args.retriever), args.k)
        print_retrieval_report(args.retriever, retrieval_metrics(cases, returned, args.k))
    except DatasetError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
