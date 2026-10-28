import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from support_agent.kb import retrieve

DATASET_PATH = PROJECT_ROOT / "evals" / "retrieval_queries.json"


def evaluate():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    passed = 0
    answerable_total = 0
    answerable_hits = 0
    unanswerable_total = 0
    correct_abstentions = 0

    print("\nRETRIEVAL EVALUATION\n")

    for case in cases:
        query = case["query"]
        expected_id = case["expected_id"]
        results = retrieve(query, top_k=3)
        print(
            "       Scores:",
            [(item["id"], round(item["score"], 3)) for item in results],
        )
        result_ids = [item["id"] for item in results]

        if expected_id is None:
            unanswerable_total += 1
            correct = len(results) == 0

            if correct:
                correct_abstentions += 1
        else:
            answerable_total += 1
            correct = expected_id in result_ids

            if correct:
                answerable_hits += 1

        if correct:
            passed += 1

        status = "PASS" if correct else "FAIL"
        print(f"[{status}] {query}")
        print(f"       Expected: {expected_id}")
        print(f"       Retrieved: {result_ids}\n")

    total = len(cases)

    print("=" * 40)
    print(f"Overall accuracy: {passed}/{total} ({passed / total:.0%})")

    if answerable_total:
        print(
            f"Recall@3: {answerable_hits}/{answerable_total} "
            f"({answerable_hits / answerable_total:.0%})"
        )

    if unanswerable_total:
        print(
            f"Abstention accuracy: "
            f"{correct_abstentions}/{unanswerable_total} "
            f"({correct_abstentions / unanswerable_total:.0%})"
        )


if __name__ == "__main__":
    evaluate()
