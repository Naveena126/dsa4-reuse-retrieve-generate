"""Unit tests for Task 2 retrieval trigger module (`dsa4.retrieval`).

All tests run on small synthetic datasets (no downloads, no GPU).
Run: python -m pytest -q tests/test_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsa4 import data, retrieval  # noqa: E402
from dsa4.cost import CostWeights  # noqa: E402


def _make_synthetic_data(n_per_dataset: int = 60):
    """Build synthetic questions + cache with well-behaved uncertainty signals.

    On TriviaQA: direct answers are mostly right when logprob is high / entropy is low,
    and RAG fixes wrong direct answers.
    On HotpotQA: direct answers are harder and slightly overconfident (simulating shift).
    """
    rng = np.random.default_rng(42)
    q_rows = []
    c_rows = []

    for ds in ("triviaqa", "hotpotqa"):
        for i in range(n_per_dataset):
            qid = f"{ds[:3]}_{i}"
            gold = f"Answer {i}"
            # Make TriviaQA easier directly (65% direct correct), HotpotQA harder (30% direct correct)
            if ds == "triviaqa":
                direct_ok = (i % 20) < 13
                rag_ok = (i % 20) < 18  # RAG fixes 5/20, hurts 0
            else:
                direct_ok = (i % 20) < 6
                rag_ok = (i % 20) < 15  # RAG fixes 9/20

            # Log-prob and entropy correlate with direct correctness (shifted on HotpotQA)
            if direct_ok:
                d_lp = float(rng.normal(-0.15, 0.08))
                d_ent = float(np.clip(rng.normal(0.20, 0.08), 0.01, 2.0))
            else:
                # HotpotQA wrong answers are slightly more confident (overconfident shift)
                lp_mean = -0.55 if ds == "hotpotqa" else -1.10
                ent_mean = 0.65 if ds == "hotpotqa" else 1.25
                d_lp = float(rng.normal(lp_mean, 0.15))
                d_ent = float(np.clip(rng.normal(ent_mean, 0.15), 0.05, 3.0))

            # Question phrasing slightly reflects complexity so TF-IDF / classifier can learn signal
            prefix = "Compare two multi-hop facts about" if (not direct_ok and rag_ok) else "What simple fact is"
            question = f"{prefix} entity {i} in {ds}?"

            q_rows.append({
                "qid": qid,
                "dataset": ds,
                "question": question,
                "answer": gold,
                "aliases": [gold, f"the {gold}"],
                "docs": [],
                "gold_titles": [],
            })
            c_rows.append({
                "qid": qid,
                "dataset": ds,
                "block": (i % 4) + 1,
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "direct": {
                    "answer": gold if direct_ok else "Wrong answer",
                    "logprob_mean": min(-0.001, d_lp),
                    "entropy_mean": d_ent,
                    "n_tokens": 4,
                    "latency_s": 0.40,
                    "correct": direct_ok,
                },
                "rag": {
                    "answer": f"the {gold}!" if rag_ok else "Still wrong",
                    "logprob_mean": -0.10 if rag_ok else -0.80,
                    "entropy_mean": 0.15 if rag_ok else 0.90,
                    "n_tokens": 5,
                    "latency_s": 0.90,
                    "retrieval_s": 0.05,
                    "correct": rag_ok,
                },
            })

    q_df = data.assign_splits(pd.DataFrame(q_rows))
    c_df = pd.DataFrame(c_rows)
    return q_df, c_df


def test_prepare_retrieval_frame_and_silver_labels():
    q_df, c_df = _make_synthetic_data(20)
    df = retrieval.prepare_retrieval_frame(q_df, c_df)
    assert len(df) == 40
    # Normalized alias matching ("the Answer i!" == "Answer i")
    assert set(df["direct_em"].unique()).issubset({0, 1})
    assert set(df["rag_em"].unique()).issubset({0, 1})
    assert (df["retrieval_helps"] == ((df["rag_em"] == 1) & (df["direct_em"] == 0)).astype(int)).all()
    assert (df["direct_conf"] >= 0.0).all() and (df["direct_conf"] <= 1.0).all()
    assert np.allclose(df["neg_logprob"], -df["direct_logprob"])


def test_pre_vs_post_generation_accounting():
    q_df, c_df = _make_synthetic_data(20)
    df = retrieval.prepare_retrieval_frame(q_df, c_df)
    mask = np.array([i % 2 == 0 for i in range(len(df))], dtype=bool)
    w = CostWeights(lookup=0.0, retrieval=1.0, call_3b=3.0, call_7b=10.0)

    pre = retrieval.evaluate_policy(df, mask, is_pre_generation=True, classifier_latency_ms=5.0, weights=w)
    post = retrieval.evaluate_policy(df, mask, is_pre_generation=False, weights=w)

    # Identical EM, F1, retrieval_rate
    assert pre["EM"] == pytest.approx(post["EM"])
    assert pre["F1"] == pytest.approx(post["F1"])
    assert pre["retrieval_rate"] == pytest.approx(0.5)

    # Pre-generation pays 1 LLM call/query (1000/1k); post-generation pays 1 + 0.5 = 1.5 calls/query (1500/1k)
    assert pre["llm_calls_per_1k"] == pytest.approx(1000.0)
    assert post["llm_calls_per_1k"] == pytest.approx(1500.0)
    # Pre-gen tokens: 0.5 * 5 + 0.5 * 4 = 4.5 -> 4500/1k; Post-gen tokens: 4 + 0.5 * 5 = 6.5 -> 6500/1k
    assert pre["tokens_per_1k"] == pytest.approx(4500.0)
    assert post["tokens_per_1k"] == pytest.approx(6500.0)
    # Pre-gen cost: 0.5*11 + 0.5*10 = 10.5 -> 10500/1k; Post-gen cost: 0.5*21 + 0.5*10 = 15.5 -> 15500/1k
    assert pre["cost_per_1k"] == pytest.approx(10500.0)
    assert post["cost_per_1k"] == pytest.approx(15500.0)
    # Latency: pre = 5 + 0.5*900 + 0.5*400 = 655 ms; post = 400 + 0.5*900 = 850 ms
    assert pre["mean_latency_ms"] == pytest.approx(655.0)
    assert post["mean_latency_ms"] == pytest.approx(850.0)


def test_oracle_matches_or_beats_always_rag():
    q_df, c_df = _make_synthetic_data(20)
    df = retrieval.prepare_retrieval_frame(q_df, c_df)
    always = retrieval.evaluate_policy(df, np.ones(len(df), dtype=bool), is_pre_generation=True)
    oracle = retrieval.evaluate_oracle(df)
    assert oracle["EM"] >= always["EM"]
    assert oracle["retrieval_rate"] == pytest.approx(df["retrieval_helps"].mean())
    assert oracle["cost_per_1k"] <= always["cost_per_1k"]


def test_sweep_pick_threshold_and_auroc():
    q_df, c_df = _make_synthetic_data(40)
    df = retrieval.prepare_retrieval_frame(q_df, c_df)
    tq = df[df["dataset"] == "triviaqa"].reset_index(drop=True)

    curve_lp = retrieval.sweep_thresholds(tq, tq["direct_logprob"], direction="below", is_pre_generation=False)
    assert curve_lp["retrieval_rate"].iloc[0] == pytest.approx(0.0)
    assert curve_lp["retrieval_rate"].iloc[-1] == pytest.approx(1.0)

    t_lp = retrieval.pick_threshold(curve_lp, always_rag_em=float(tq["rag_em"].mean()), em_tolerance=0.01)
    chosen_mask = retrieval.apply_threshold(tq["direct_logprob"], t_lp, direction="below")
    chosen = retrieval.evaluate_policy(tq, chosen_mask, is_pre_generation=False)
    assert chosen["EM"] >= float(tq["rag_em"].mean()) - 0.01
    assert chosen["retrieval_rate"] < 0.7  # well below always-retrieve on synthetic TriviaQA

    auroc_lp = retrieval.compute_trigger_auroc(tq, tq["neg_logprob"], target_col="retrieval_helps")
    auroc_ent = retrieval.compute_trigger_auroc(tq, tq["direct_entropy"], target_col="retrieval_helps")
    assert auroc_lp > 0.85
    assert auroc_ent > 0.85


def test_calibration_ece_and_platt_scaler():
    # Perfect calibration on two groups
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    cal = retrieval.expected_calibration_error(y, p, n_bins=10)
    assert cal["ece"] == pytest.approx(0.0)
    assert cal["brier"] == pytest.approx(0.0)
    assert len(cal["bins_df"]) == 10

    # Platt scaling maps entropy to valid probabilities
    q_df, c_df = _make_synthetic_data(40)
    df = retrieval.prepare_retrieval_frame(q_df, c_df)
    tq = df[df["dataset"] == "triviaqa"]
    hq = df[df["dataset"] == "hotpotqa"]
    scaler = retrieval.fit_platt_scaler(tq["direct_entropy"], tq["retrieval_helps"])
    assert scaler is not None
    probs_hq = retrieval.apply_platt_scaler(scaler, hq["direct_entropy"])
    assert probs_hq.shape == (len(hq),)
    assert (probs_hq >= 0.0).all() and (probs_hq <= 1.0).all()


def test_confident_wrong_and_tfidf_and_multiseed_transfer():
    q_df, c_df = _make_synthetic_data(60)
    df = retrieval.prepare_retrieval_frame(q_df, c_df)

    cw = retrieval.confident_wrong_examples(df, top_k=5, dataset="hotpotqa")
    assert len(cw) == 5
    assert (cw["dataset"] == "hotpotqa").all()
    assert cw["direct_conf"].is_monotonic_decreasing

    # Train TF-IDF proxy per seed to exercise the v4_probs_by_seed interface end-to-end
    v4_probs = {}
    v4_lat = {}
    for seed in (0, 1, 2):
        tr = df[(df["dataset"] == "triviaqa") & (df[f"split_s{seed}"] == "train")]
        clf = retrieval.train_tfidf_trigger(tr, target_col="retrieval_helps", seed=seed)
        probs, lat_ms = retrieval.predict_tfidf_proba(clf, df["question"].tolist())
        v4_probs[seed] = dict(zip(df["qid"], probs))
        v4_lat[seed] = lat_ms

    per_seed, summary = retrieval.evaluate_triggers_across_seeds(
        df,
        seeds=(0, 1, 2),
        em_tolerance=0.01,
        v4_probs_by_seed=v4_probs,
        v4_latency_ms_by_seed=v4_lat,
    )
    assert set(per_seed["seed"].unique()) == {0, 1, 2}
    assert set(per_seed["eval_set"].unique()) == {
        "TriviaQA (in-domain)",
        "HotpotQA (transfer)",
        "HotpotQA (in-domain)",
    }
    methods = set(summary["method"].unique())
    assert {
        "V1a Never retrieve",
        "V1b Always retrieve",
        "Oracle (cheapest correct)",
        "V2 Log-probability",
        "V3 Predictive entropy",
        "V4 DistilBERT classifier",
    }.issubset(methods)
    assert (summary["n_seeds"] == 3).all()
