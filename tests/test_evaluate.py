"""Unit tests for src/dsa4/evaluate.py (DSA 4 Day 1 deliverable)."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsa4.evaluate import (  # noqa: E402
    aggregate_seed_results,
    exact_match_score,
    format_results_table,
    is_safe_reuse,
    normalize_answer,
    save_results_table,
    score_qa_predictions,
    token_f1_score,
)


def test_normalize_answer():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("  An apple, a day! ") == "apple day"
    assert normalize_answer("A.B.C.") == "abc"
    assert normalize_answer(None) == ""


def test_exact_match_aliases_and_single_string():
    aliases = ["Paris", "City of Light"]
    assert exact_match_score("the Paris.", aliases) == 1
    assert exact_match_score("City of Light", aliases) == 1
    assert exact_match_score("London", aliases) == 0
    assert exact_match_score("paris", "The Paris!") == 1
    assert exact_match_score("", aliases) == 0
    assert exact_match_score("Paris", []) == 0


def test_token_f1_score():
    assert token_f1_score("Paris", ["Paris"]) == 1.0
    assert token_f1_score("Paris France", ["Paris"]) == pytest.approx(2.0 / 3.0)
    assert token_f1_score("the capital of France", ["Paris", "France"]) == pytest.approx(0.5)
    assert token_f1_score("London", ["Paris"]) == 0.0
    assert token_f1_score("", [""]) == 1.0


def test_lookalike_and_safe_reuse_decision_2():
    assert is_safe_reuse("Paris", ["Paris"], is_lookalike=False) is True
    assert is_safe_reuse("Paris", ["Paris"], is_lookalike=True) is False
    assert is_safe_reuse("Paris", [], is_lookalike=False) is False

    batch = score_qa_predictions(
        predictions=["Paris", "Wrong Answer", "1979", "Berlin"],
        gold_answers_list=[["Paris"], ["Paris"], ["1979", "79"], []],
        is_lookalike_list=[False, True, False, False],
    )
    assert batch["n_scored"] == 2
    assert batch["n_skipped_no_gold"] == 2
    assert batch["EM"] == 1.0
    assert batch["F1"] == 1.0


def test_aggregate_seed_results_and_save(tmp_path):
    per_seed = [
        {"dataset": "TriviaQA", "method": "V2_logprob", "seed": 0, "EM": 0.70, "F1": 0.75, "retrieval_rate": 0.40},
        {"dataset": "TriviaQA", "method": "V2_logprob", "seed": 1, "EM": 0.72, "F1": 0.77, "retrieval_rate": 0.42},
        {"dataset": "TriviaQA", "method": "V2_logprob", "seed": 2, "EM": 0.71, "F1": 0.76, "retrieval_rate": 0.41},
        {"dataset": "TriviaQA", "method": "V3_entropy", "seed": 0, "EM": 0.68, "F1": 0.73, "retrieval_rate": 0.35},
        {"dataset": "TriviaQA", "method": "V3_entropy", "seed": 1, "EM": 0.70, "F1": 0.75, "retrieval_rate": 0.37},
        {"dataset": "TriviaQA", "method": "V3_entropy", "seed": 2, "EM": 0.69, "F1": 0.74, "retrieval_rate": 0.36},
    ]
    summary = aggregate_seed_results(per_seed)
    assert list(summary["method"]) == ["V2_logprob", "V3_entropy"]
    assert summary.loc[0, "n_seeds"] == 3
    assert summary.loc[0, "EM"] == "0.710 ± 0.010"
    assert summary.loc[0, "EM_mean"] == pytest.approx(0.71)
    assert summary.loc[1, "retrieval_rate"] == "0.360 ± 0.010"

    clean = format_results_table(summary)
    assert list(clean.columns) == ["dataset", "method", "n_seeds", "EM", "F1", "retrieval_rate"]

    out_csv = save_results_table(summary, tmp_path / "metrics" / "task2_summary.csv", formatted_only=True)
    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8")
    assert "0.710 ± 0.010" in content
