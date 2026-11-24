"""Shared evaluation utilities and results tables for DSA 4 (Tasks 1–4).

Owner: Manimegalai Kumar Periyasamy (Task 2 — Retrieval Trigger)

Builds on `dsa4.text` (`normalize`, `exact_match`) to provide:
  - normalize_answer: SQuAD-style text normalisation (via `dsa4.text.normalize`)
  - exact_match_score: normalised Exact Match (EM) against all accepted aliases
  - token_f1_score: maximum token-level F1 across all accepted aliases
  - is_safe_reuse: safe-reuse check enforcing Locked Decision #2 (look-alikes are unsafe)
  - score_qa_predictions: batch EM & F1 (excludes look-alikes / missing gold answers)
  - aggregate_seed_results: turns per-seed results (seeds 0–2) into a mean ± std table
  - format_results_table: extracts clean presentation-ready mean ± std columns
  - save_results_table: writes results tables to CSV under `results/metrics/`
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    from .text import contains_answer, exact_match, normalize
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dsa4.text import contains_answer, exact_match, normalize


def _coerce_answers(answers: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Accept either a sequence of answer aliases or a single answer string."""
    if answers is None:
        return []
    if isinstance(answers, str):
        return [answers]
    return [a for a in answers if a is not None]


def normalize_answer(text: Optional[str]) -> str:
    """Normalize text for QA comparison (delegates to `dsa4.text.normalize`)."""
    return normalize(text or "")


def exact_match_score(
    prediction: Optional[str],
    answers: Optional[Union[str, Sequence[str]]],
) -> int:
    """Return 1 if normalized prediction matches any accepted answer/alias, else 0."""
    gold_list = _coerce_answers(answers)
    if not gold_list:
        return 0
    return int(exact_match(prediction or "", gold_list))


def token_f1_score(
    prediction: Optional[str],
    answers: Optional[Union[str, Sequence[str]]],
) -> float:
    """Return the maximum token-level F1 across all accepted answers/aliases."""
    gold_list = _coerce_answers(answers)
    if not gold_list:
        return 0.0

    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0

    for ans in gold_list:
        ans_tokens = normalize_answer(ans).split()
        if len(pred_tokens) == 0 and len(ans_tokens) == 0:
            current_f1 = 1.0
        elif len(pred_tokens) == 0 or len(ans_tokens) == 0:
            current_f1 = 0.0
        else:
            common = Counter(pred_tokens) & Counter(ans_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                current_f1 = 0.0
            else:
                precision = num_same / len(pred_tokens)
                recall = num_same / len(ans_tokens)
                current_f1 = 2.0 * precision * recall / (precision + recall)
        if current_f1 > best_f1:
            best_f1 = current_f1

    return best_f1


def is_safe_reuse(
    cached_answer: Optional[str],
    gold_answers: Optional[Union[str, Sequence[str]]],
    is_lookalike: bool = False,
) -> bool:
    """Return True iff reusing `cached_answer` is safe for the incoming question.

    Per DSA 4 shared setup and Locked Decision #2:
      - Any reuse of a look-alike question is always unsafe.
      - Otherwise, reuse is safe only if the cached answer matches the new
        question's gold answers/aliases under normalised exact match.
    """
    if is_lookalike:
        return False
    gold_list = _coerce_answers(gold_answers)
    if not gold_list:
        return False
    return bool(exact_match_score(cached_answer, gold_list))


def score_qa_predictions(
    predictions: Sequence[Optional[str]],
    gold_answers_list: Sequence[Optional[Union[str, Sequence[str]]]],
    is_lookalike_list: Optional[Sequence[bool]] = None,
) -> dict:
    """Compute mean EM and token F1 over questions that have gold answers.

    Per DSA 4 Locked Decision #2, look-alike questions have no gold answer
    (marked via `is_lookalike_list` or empty/None gold answers) and count
    only for unsafe reuse, never for accuracy.
    """
    if len(predictions) != len(gold_answers_list):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(gold_answers_list)} gold_answers_list entries."
        )
    if is_lookalike_list is not None and len(is_lookalike_list) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(is_lookalike_list)} is_lookalike_list entries vs "
            f"{len(predictions)} predictions."
        )

    em_scores: List[int] = []
    f1_scores: List[float] = []
    skipped = 0

    for idx, (pred, answers) in enumerate(zip(predictions, gold_answers_list)):
        lookalike = bool(is_lookalike_list[idx]) if is_lookalike_list is not None else False
        gold_list = _coerce_answers(answers)
        if lookalike or not gold_list:
            skipped += 1
            continue
        em_scores.append(exact_match_score(pred, gold_list))
        f1_scores.append(token_f1_score(pred, gold_list))

    return {
        "EM": float(np.mean(em_scores)) if em_scores else float("nan"),
        "F1": float(np.mean(f1_scores)) if f1_scores else float("nan"),
        "n_scored": len(em_scores),
        "n_skipped_no_gold": skipped,
    }


def aggregate_seed_results(
    seed_results: Union[pd.DataFrame, Iterable[dict]],
    group_cols: Optional[Sequence[str]] = None,
    metric_cols: Optional[Sequence[str]] = None,
    seed_col: str = "seed",
    decimals: int = 3,
) -> pd.DataFrame:
    """Turn per-seed results (e.g. seeds 0–2) into a mean ± std summary table.

    Works for any task's numeric metrics (EM, F1, retrieval_rate, AUROC, ECE,
    calls_saved, unsafe_reuse, weighted_cost, latency_ms, etc.).

    Returns a DataFrame with `<metric>_mean`, `<metric>_std` (sample std, ddof=1),
    and formatted `<metric>` ("mean ± std") columns for each metric.
    """
    df = (
        seed_results.copy()
        if isinstance(seed_results, pd.DataFrame)
        else pd.DataFrame(list(seed_results))
    )
    if df.empty:
        return pd.DataFrame()

    if group_cols is None:
        group_cols = [
            c
            for c in df.columns
            if c != seed_col and not pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        group_cols = list(group_cols)

    if metric_cols is None:
        metric_cols = [
            c
            for c in df.columns
            if c not in group_cols
            and c != seed_col
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        metric_cols = list(metric_cols)

    def _summarize(sub: pd.DataFrame) -> dict:
        row: dict = {"n_seeds": int(len(sub))}
        for m in metric_cols:
            vals = sub[m].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                mean_v, std_v = float("nan"), float("nan")
                fmt = "nan"
            else:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                fmt = f"{mean_v:.{decimals}f} ± {std_v:.{decimals}f}"
            row[f"{m}_mean"] = mean_v
            row[f"{m}_std"] = std_v
            row[m] = fmt
        return row

    if group_cols:
        records = []
        for keys, sub in df.groupby(group_cols, sort=False, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            entry = dict(zip(group_cols, keys))
            entry.update(_summarize(sub))
            records.append(entry)
        return pd.DataFrame(records)

    return pd.DataFrame([_summarize(df)])


def format_results_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return only group columns, `n_seeds`, and formatted `mean ± std` columns."""
    keep_cols = [
        c
        for c in summary_df.columns
        if not (c.endswith("_mean") or c.endswith("_std"))
    ]
    return summary_df[keep_cols].copy()


def save_results_table(
    summary_df: pd.DataFrame,
    csv_path: Union[str, Path],
    formatted_only: bool = False,
) -> Path:
    """Save a results table to CSV (creating parent directories if needed)."""
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_write = format_results_table(summary_df) if formatted_only else summary_df
    to_write.to_csv(out_path, index=False)
    return out_path


def run_tests() -> None:
    """Five self-contained unit tests for quick CLI verification."""
    # Test 1: Answer normalization
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("  An apple, a day! ") == "apple day"
    assert normalize_answer(None) == ""

    # Test 2: Exact match against multiple accepted aliases (and single string)
    assert exact_match_score("the Paris.", ["Paris", "City of Light"]) == 1
    assert exact_match_score("City of Light", ["Paris", "City of Light"]) == 1
    assert exact_match_score("Paris", "the Paris!") == 1
    assert exact_match_score("London", ["Paris", "City of Light"]) == 0
    assert exact_match_score("", ["Paris"]) == 0

    # Test 3: Token-level F1
    assert token_f1_score("Paris", ["Paris"]) == 1.0
    assert abs(token_f1_score("Paris France", ["Paris"]) - (2.0 / 3.0)) < 1e-9
    assert token_f1_score("London", ["Paris"]) == 0.0
    assert token_f1_score("", [""]) == 1.0

    # Test 4: Look-alike & safe-reuse handling (Locked Decision #2)
    assert is_safe_reuse("Paris", ["Paris", "City of Light"], is_lookalike=False) is True
    assert is_safe_reuse("Paris", ["Paris", "City of Light"], is_lookalike=True) is False
    batch = score_qa_predictions(
        predictions=["Paris", "Aliens", "1979"],
        gold_answers_list=[["Paris"], [], ["1979", "79"]],
        is_lookalike_list=[False, True, False],
    )
    assert batch["n_scored"] == 2
    assert batch["n_skipped_no_gold"] == 1
    assert batch["EM"] == 1.0
    assert batch["F1"] == 1.0

    # Test 5: Per-seed results -> mean ± std table
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
    assert summary.loc[0, "EM"] == "0.710 ± 0.010"
    assert summary.loc[1, "retrieval_rate"] == "0.360 ± 0.010"
    clean = format_results_table(summary)
    assert list(clean.columns) == ["dataset", "method", "n_seeds", "EM", "F1", "retrieval_rate"]

    print("All 5 tests passed!")


__all__ = [
    "contains_answer",
    "normalize_answer",
    "exact_match_score",
    "token_f1_score",
    "is_safe_reuse",
    "score_qa_predictions",
    "aggregate_seed_results",
    "format_results_table",
    "save_results_table",
    "run_tests",
]


if __name__ == "__main__":
    run_tests()
