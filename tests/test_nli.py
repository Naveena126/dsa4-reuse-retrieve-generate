"""Tests for src/dsa4/nli.py (unit tests + 10 hand-made NLI examples)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsa4.nli import NLISupportChecker, aggregate_scores, build_hypothesis

HAND_MADE_EXAMPLES = [
    # --- 5 SUPPORTED ---
    (
        "What is the capital of France?",
        "Paris",
        ["France is a country in Western Europe. Its capital and largest city is Paris."],
        True,
    ),
    (
        "Who wrote Pride and Prejudice?",
        "Jane Austen",
        [
            "The novel was published in 1813.",
            "Pride and Prejudice is a romantic novel written by Jane Austen.",
        ],
        True,
    ),
    (
        "In what year did the Apollo 11 mission land on the Moon?",
        "1969",
        ["Apollo 11 was the spaceflight that first landed humans on the Moon in July 1969."],
        True,
    ),
    (
        "What chemical element has the symbol Au?",
        "Gold",
        ["Gold is a chemical element with the symbol Au and atomic number 79."],
        True,
    ),
    (
        "Which river flows through Cairo?",
        "The Nile",
        ["Cairo is the capital of Egypt and sits near the Nile Delta along the Nile River."],
        True,
    ),
    # --- 5 UNSUPPORTED / LOOK-ALIKE TRAPS ---
    (
        "What is the capital of Germany?",
        "Paris",  # Entity swap look-alike
        ["Berlin is the capital and largest city of Germany."],
        False,
    ),
    (
        "In what year did World War II end?",
        "1918",  # Date swap look-alike (WWI vs WWII)
        ["World War II ended in 1945 with the surrender of Germany and Japan."],
        False,
    ),
    (
        "How many continents are there on Earth?",
        "5",  # Number swap look-alike
        ["Earth is conventionally divided into seven continents."],
        False,
    ),
    (
        "Who directed Inception?",
        "Steven Spielberg",  # Wrong entity
        ["Inception is a 2010 science fiction film written and directed by Christopher Nolan."],
        False,
    ),
    (
        "What is the boiling point of water at sea level?",
        "100 degrees Celsius",  # Unrelated passage (missing evidence)
        ["Mount Everest is the highest mountain above sea level, located in the Himalayas."],
        False,
    ),
]


def test_build_hypothesis():
    assert build_hypothesis("  Who? ", " Alice ") == "Question: Who? Answer: Alice"


def test_aggregate_scores():
    assert aggregate_scores([0.1, 0.9, 0.4], "max") == pytest.approx(0.9)
    assert aggregate_scores([0.2, 0.8], "mean") == pytest.approx(0.5)
    assert aggregate_scores([], "max") == 0.0
    with pytest.raises(ValueError):
        aggregate_scores([0.5], "median")


def test_empty_inputs_return_zero():
    checker = object.__new__(NLISupportChecker)
    checker._score_pairs = lambda pairs: [0.99 for _ in pairs]
    out_empty_passages = checker.score("Who?", "Alice", [])
    out_empty_answer = checker.score("Who?", "   ", ["Alice won the race."])
    assert out_empty_passages["support_score"] == 0.0
    assert out_empty_passages["best_passage_idx"] == -1
    assert out_empty_answer["support_score"] == 0.0


@pytest.mark.skipif(
    os.environ.get("RUN_NLI_MODEL") != "1",
    reason="Set RUN_NLI_MODEL=1 (or run `python tests/test_nli.py`) to run DeBERTa-v3-base.",
)
def test_ten_hand_made_examples():
    checker = NLISupportChecker()
    items = [(q, a, p) for q, a, p, _ in HAND_MADE_EXAMPLES]
    results = checker.score_batch(items)

    correct = 0
    for (q, a, _, expected), res in zip(HAND_MADE_EXAMPLES, results):
        pred = res["support_score"] >= 0.5
        correct += int(pred == expected)
        print(
            f"[{'OK' if pred == expected else 'FAIL'}] "
            f"score={res['support_score']:.3f} expected={expected} | Q={q} | A={a}"
        )

    assert correct >= 9, f"Expected at least 9/10 correct, got {correct}/10"


if __name__ == "__main__":
    test_ten_hand_made_examples()