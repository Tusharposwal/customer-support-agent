"""Offline checks for the rerank holdout dataset. No Cohere calls."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from support_agent.kb import load_articles

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
HOLDOUT_PATH = EVALS_DIR / "rerank_holdout_queries.json"
ORIGINAL_PATH = EVALS_DIR / "retrieval_queries.json"


def _load(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalise(query):
    """Lowercase and drop punctuation so near-identical copies are caught."""
    return " ".join(re.findall(r"\w+", query.lower()))


@pytest.fixture(scope="module")
def holdout():
    return _load(HOLDOUT_PATH)


def test_structure_matches_original_schema(holdout):
    schema = {"query", "expected_id"}
    assert all(set(case) == schema for case in _load(ORIGINAL_PATH))
    assert isinstance(holdout, list)

    for case in holdout:
        assert isinstance(case, dict)
        assert set(case) == schema
        assert isinstance(case["query"], str) and case["query"].strip()
        assert case["expected_id"] is None or isinstance(case["expected_id"], str)


def test_exactly_20_questions(holdout):
    assert len(holdout) == 20


def test_ten_answerable_and_ten_unsupported(holdout):
    answerable = [c for c in holdout if c["expected_id"] is not None]
    assert len(answerable) == 10
    assert len(holdout) - len(answerable) == 10


def test_expected_ids_exist_and_cover_every_article(holdout):
    kb_ids = {article["id"] for article in load_articles()}
    used = {c["expected_id"] for c in holdout if c["expected_id"] is not None}

    assert used <= kb_ids
    assert used == kb_ids


def test_no_duplicate_questions(holdout):
    normalised = [_normalise(c["query"]) for c in holdout]
    assert len(set(normalised)) == len(normalised)


def test_no_overlap_with_original_dataset(holdout):
    original = {_normalise(c["query"]) for c in _load(ORIGINAL_PATH)}
    assert not original & {_normalise(c["query"]) for c in holdout}


def test_original_dataset_still_matches_saved_rerank_results():
    """The 17 questions must be exactly the ones the saved experiment used."""
    original = _load(ORIGINAL_PATH)
    saved = _load(EVALS_DIR / "fixtures" / "rerank_results.json")
    encoded = json.dumps(
        [case["query"] for case in original],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )

    assert len(original) == 17
    assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == saved["questions_sha256"]
