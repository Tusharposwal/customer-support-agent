import json
from pathlib import Path

from support_agent.kb import retrieve
from support_agent.semantic_kb import retrieve_semantic
from support_agent.hybrid_kb import retrieve_hybrid
from calibrate_semantic import historical_kb  # same folder (run as a script)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evals" / "retrieval_queries.json"

# These questions are labelled against the frozen 3-article KB (v1). For the
# current knowledge base use evals/evaluate_kb_v2.py.


def evaluate_retriever(name, retriever, cases):
    top1_correct = 0
    hit_at_3 = 0
    answerable_count = 0
    abstention_correct = 0
    unanswerable_count = 0

    print(f"\n{'=' * 55}")
    print(f"{name.upper()} RETRIEVAL")
    print("=" * 55)

    for case in cases:
        query = case["query"]
        expected_id = case["expected_id"]
        results = retriever(query, top_k=3)

        ids = [item["id"] for item in results]
        scores = [(item["id"], round(item["score"], 3)) for item in results]

        if expected_id is None:
            unanswerable_count += 1
            correct_abstention = len(results) == 0

            if correct_abstention:
                abstention_correct += 1

            status = "PASS" if correct_abstention else "FAIL"
            print(f"[{status}] {query}")
            print(f"  Expected: no result")
            print(f"  Retrieved: {scores}")
            continue

        answerable_count += 1

        if ids and ids[0] == expected_id:
            top1_correct += 1

        if expected_id in ids:
            hit_at_3 += 1

        status = "PASS" if ids and ids[0] == expected_id else "FAIL"

        print(f"[{status}] {query}")
        print(f"  Expected: {expected_id}")
        print(f"  Retrieved: {scores}")

    print(f"\nMetrics for {name}:")
    print(
        f"Top-1 accuracy: {top1_correct}/{answerable_count} "
        f"({top1_correct / answerable_count:.0%})"
    )
    print(
        f"Recall@3: {hit_at_3}/{answerable_count} "
        f"({hit_at_3 / answerable_count:.0%})"
    )

    if unanswerable_count:
        print(
            f"Abstention accuracy: "
            f"{abstention_correct}/{unanswerable_count} "
            f"({abstention_correct / unanswerable_count:.0%})"
        )


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    evaluate_retriever("Keyword", retrieve, cases)
    evaluate_retriever("Semantic", retrieve_semantic, cases)

    evaluate_retriever("Hybrid", retrieve_hybrid, cases)


if __name__ == "__main__":
    with historical_kb():
        main()
