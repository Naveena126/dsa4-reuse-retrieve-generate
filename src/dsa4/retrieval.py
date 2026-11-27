"""Task 2: Retrieval Trigger (When to Use RAG) — policies, transfer, and calibration.

Evaluates retrieval-trigger policies offline on the shared DSA 4 answer cache:
  - V1a Never retrieve (direct LLM answer only; 0% retrieval)
  - V1b Always retrieve (top-5 RAG on every question; 100% retrieval)
  - V2  Token log-probability trigger (retrieve if direct mean log-prob < threshold)
  - V3  Predictive entropy trigger (retrieve if direct mean token entropy > threshold)
  - V4  DistilBERT-base (66M) complexity classifier trained on question text -> silver labels

Fair-comparison & efficiency accounting:
  - Post-generation triggers (V2, V3) always pay the initial direct LLM pass first;
    triggered queries pay retrieval + a second RAG LLM pass (tokens, latency, cost).
  - Pre-generation routing (V1 fixed policies, V4 DistilBERT) decides before generation
    and pays a single LLM pass per query (+ classifier latency for V4).
  - Thresholds are tuned on TriviaQA validation and applied unchanged to TriviaQA test
    and HotpotQA test (zero-retuning transfer), with in-domain HotpotQA tuning reported
    as an upper reference.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from .cost import CostWeights, call_cost, cheapest_correct_action
from .evaluate import aggregate_seed_results, exact_match_score, token_f1_score

# Default cost weights per the Day 1 plan (lookup=0, retrieval=1, 3B=3, 7B=10)
DEFAULT_COST_WEIGHTS = CostWeights(lookup=0.0, retrieval=1.0, call_3b=3.0, call_7b=10.0)


def _extract_dict_field(series: pd.Series, key: str, default: Any = np.nan) -> list[Any]:
    return [d.get(key, default) if isinstance(d, Mapping) else default for d in series]


def prepare_retrieval_frame(
    questions_df: pd.DataFrame,
    cache_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join questions metadata with the shared answer cache and compute Task 2 columns.

    Computes normalised Exact Match (EM) and token F1 for both direct and RAG answers
    against all accepted aliases (via `dsa4.evaluate`), plus silver labels:
      - `retrieval_helps`: 1 if RAG is exact-match correct while direct is wrong, else 0
      - `retrieval_hurts`: 1 if direct is exact-match correct while RAG is wrong, else 0
      - `direct_wrong`:    1 if direct is wrong, else 0
      - `neg_logprob`:     -direct_logprob (uncertainty score; higher = more uncertain)
      - `direct_conf`:     exp(direct_logprob) clipped to [0, 1] (for calibration / ECE)
    """
    q_cols = [
        c
        for c in ("qid", "dataset", "question", "answer", "aliases", "block", "split_s0", "split_s1", "split_s2")
        if c in questions_df.columns
    ]
    q_sub = questions_df[q_cols].drop_duplicates("qid").copy()
    c_cols = [c for c in ("qid", "direct", "rag", "model", "aliases", "question", "dataset") if c in cache_df.columns]
    c_sub = cache_df[c_cols].drop_duplicates("qid").copy()

    # Prefer question metadata from questions_df; keep cache columns where missing
    overlap = [c for c in ("dataset", "question", "aliases") if c in q_sub.columns and c in c_sub.columns]
    c_sub = c_sub.drop(columns=overlap)
    merged = q_sub.merge(c_sub, on="qid", how="inner").reset_index(drop=True)

    direct_ans = _extract_dict_field(merged["direct"], "answer", "")
    rag_ans = _extract_dict_field(merged["rag"], "answer", "")

    if "aliases" in merged.columns:
        aliases_list = merged["aliases"].tolist()
        direct_em = [exact_match_score(a, g) for a, g in zip(direct_ans, aliases_list)]
        direct_f1 = [token_f1_score(a, g) for a, g in zip(direct_ans, aliases_list)]
        rag_em = [exact_match_score(a, g) for a, g in zip(rag_ans, aliases_list)]
        rag_f1 = [token_f1_score(a, g) for a, g in zip(rag_ans, aliases_list)]
    else:
        direct_em = [int(bool(v)) for v in _extract_dict_field(merged["direct"], "correct", False)]
        direct_f1 = [float(v) for v in direct_em]
        rag_em = [int(bool(v)) for v in _extract_dict_field(merged["rag"], "correct", False)]
        rag_f1 = [float(v) for v in rag_em]

    out = merged.drop(columns=["direct", "rag"]).copy()
    out["direct_answer"] = direct_ans
    out["rag_answer"] = rag_ans
    out["direct_em"] = np.asarray(direct_em, dtype=int)
    out["direct_f1"] = np.asarray(direct_f1, dtype=float)
    out["rag_em"] = np.asarray(rag_em, dtype=int)
    out["rag_f1"] = np.asarray(rag_f1, dtype=float)

    out["direct_logprob"] = pd.to_numeric(pd.Series(_extract_dict_field(merged["direct"], "logprob_mean")), errors="coerce").to_numpy(dtype=float)
    out["direct_entropy"] = pd.to_numeric(pd.Series(_extract_dict_field(merged["direct"], "entropy_mean")), errors="coerce").to_numpy(dtype=float)
    out["direct_n_tokens"] = pd.to_numeric(pd.Series(_extract_dict_field(merged["direct"], "n_tokens", 0)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["direct_latency_s"] = pd.to_numeric(pd.Series(_extract_dict_field(merged["direct"], "latency_s", 0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    out["rag_logprob"] = pd.to_numeric(pd.Series(_extract_dict_field(merged["rag"], "logprob_mean")), errors="coerce").to_numpy(dtype=float)
    out["rag_entropy"] = pd.to_numeric(pd.Series(_extract_dict_field(merged["rag"], "entropy_mean")), errors="coerce").to_numpy(dtype=float)
    out["rag_n_tokens"] = pd.to_numeric(pd.Series(_extract_dict_field(merged["rag"], "n_tokens", 0)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["rag_latency_s"] = pd.to_numeric(pd.Series(_extract_dict_field(merged["rag"], "latency_s", 0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["rag_retrieval_s"] = pd.to_numeric(pd.Series(_extract_dict_field(merged["rag"], "retrieval_s", 0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Silver labels & derived scores
    out["retrieval_helps"] = ((out["rag_em"] == 1) & (out["direct_em"] == 0)).astype(int)
    out["retrieval_hurts"] = ((out["rag_em"] == 0) & (out["direct_em"] == 1)).astype(int)
    out["direct_wrong"] = (out["direct_em"] == 0).astype(int)
    out["rag_f1_gain"] = out["rag_f1"] - out["direct_f1"]

    out["neg_logprob"] = -out["direct_logprob"]
    with np.errstate(over="ignore", invalid="ignore"):
        conf = np.exp(out["direct_logprob"].to_numpy(dtype=float))
    out["direct_conf"] = np.clip(conf, 0.0, 1.0)
    return out


def apply_threshold(
    scores: Sequence[float] | np.ndarray | pd.Series,
    threshold: float,
    direction: str = "above",
) -> np.ndarray:
    """Convert scalar scores into a boolean retrieval decision mask.

    direction:
      - "below": retrieve if score < threshold (e.g. V2 direct_logprob)
      - "above": retrieve if score > threshold (e.g. V3 direct_entropy, V4 P(retrieve))
    NaN scores default to True (retrieve conservatively).
    """
    arr = np.asarray(scores, dtype=float)
    if direction == "below":
        mask = arr < threshold
    elif direction == "above":
        mask = arr > threshold
    else:
        raise ValueError(f"Unknown direction {direction!r}; expected 'above' or 'below'.")
    return np.where(np.isnan(arr), True, mask).astype(bool)


def evaluate_policy(
    df: pd.DataFrame,
    retrieve_mask: Sequence[bool] | np.ndarray,
    *,
    is_pre_generation: bool = False,
    classifier_latency_ms: float = 0.0,
    weights: Optional[CostWeights] = None,
) -> dict[str, float | int]:
    """Evaluate a binary retrieval-trigger policy on a prepared retrieval DataFrame.

    Accounting:
      - Pre-generation policies (V1 fixed, V4 DistilBERT): single 7B LLM call per query;
        if retrieved, pay retrieval + rag_latency + rag_tokens; else direct_latency + direct_tokens.
      - Post-generation policies (V2 log-prob, V3 entropy): always run the direct 7B pass first;
        if triggered, also run retrieval + a second RAG 7B pass.
    """
    if len(df) == 0:
        return {
            "n_questions": 0,
            "EM": float("nan"),
            "F1": float("nan"),
            "retrieval_rate": float("nan"),
            "retrievals_saved": float("nan"),
            "retrievals_per_1k": float("nan"),
            "tokens_per_1k": float("nan"),
            "llm_calls_per_1k": float("nan"),
            "cost_per_1k": float("nan"),
            "mean_latency_ms": float("nan"),
            "median_latency_ms": float("nan"),
            "p95_latency_ms": float("nan"),
            "em_vs_always_rag": float("nan"),
        }

    w = weights or DEFAULT_COST_WEIGHTS
    mask = np.asarray(retrieve_mask, dtype=bool)
    if len(mask) != len(df):
        raise ValueError(f"retrieve_mask length {len(mask)} != df length {len(df)}")

    direct_em = df["direct_em"].to_numpy(dtype=float)
    rag_em = df["rag_em"].to_numpy(dtype=float)
    direct_f1 = df["direct_f1"].to_numpy(dtype=float)
    rag_f1 = df["rag_f1"].to_numpy(dtype=float)
    direct_tok = df["direct_n_tokens"].to_numpy(dtype=float)
    rag_tok = df["rag_n_tokens"].to_numpy(dtype=float)
    direct_lat_ms = df["direct_latency_s"].to_numpy(dtype=float) * 1000.0
    rag_lat_ms = df["rag_latency_s"].to_numpy(dtype=float) * 1000.0

    em_arr = np.where(mask, rag_em, direct_em)
    f1_arr = np.where(mask, rag_f1, direct_f1)
    retr_rate = float(mask.mean())

    c_7b = call_cost("call_7b", w)
    c_ret = call_cost("retrieval", w)

    if is_pre_generation:
        tok_arr = np.where(mask, rag_tok, direct_tok)
        lat_ms = classifier_latency_ms + np.where(mask, rag_lat_ms, direct_lat_ms)
        cost_arr = np.where(mask, c_ret + c_7b, c_7b)
        llm_calls_per_1k = 1000.0
    else:
        tok_arr = direct_tok + np.where(mask, rag_tok, 0.0)
        lat_ms = direct_lat_ms + np.where(mask, rag_lat_ms, 0.0)
        cost_arr = np.where(mask, c_7b + c_ret + c_7b, c_7b)
        llm_calls_per_1k = (1.0 + retr_rate) * 1000.0

    em_val = float(em_arr.mean())
    always_rag_em = float(rag_em.mean())

    return {
        "n_questions": int(len(df)),
        "EM": em_val,
        "F1": float(f1_arr.mean()),
        "retrieval_rate": retr_rate,
        "retrievals_saved": 1.0 - retr_rate,
        "retrievals_per_1k": retr_rate * 1000.0,
        "tokens_per_1k": float(tok_arr.mean() * 1000.0),
        "llm_calls_per_1k": float(llm_calls_per_1k),
        "cost_per_1k": float(cost_arr.mean() * 1000.0),
        "mean_latency_ms": float(lat_ms.mean()),
        "median_latency_ms": float(np.median(lat_ms)),
        "p95_latency_ms": float(np.percentile(lat_ms, 95)),
        "em_vs_always_rag": em_val - always_rag_em,
    }


def evaluate_oracle(
    df: pd.DataFrame,
    weights: Optional[CostWeights] = None,
) -> dict[str, float | int]:
    """Per-question oracle: pick the cheapest correct action (direct vs RAG) via `dsa4.cost`.

    Delegates directly to `dsa4.cost.cheapest_correct_action` with `"direct"` and `"rag"`
    candidates. If neither is correct, defaults to `"direct"` (lowest cost).
    """
    w = weights or DEFAULT_COST_WEIGHTS
    mask = np.zeros(len(df), dtype=bool)
    for i, (qid, d_ok, r_ok) in enumerate(
        zip(df["qid"].to_numpy(), df["direct_em"].to_numpy(), df["rag_em"].to_numpy())
    ):
        candidates = [
            {"question_id": qid, "action": "direct", "correct": bool(d_ok)},
            {"question_id": qid, "action": "rag", "correct": bool(r_ok)},
        ]
        chosen = cheapest_correct_action(candidates, w, question_id=qid)
        mask[i] = bool(chosen is not None and chosen["action"] == "rag")
    return evaluate_policy(df, mask, is_pre_generation=True, weights=w)


def _default_threshold_grid(scores: np.ndarray, direction: str, n_grid: int = 101) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([0.0])
    lo, hi = float(finite.min()), float(finite.max())
    eps = max(1e-6, 1e-4 * (hi - lo))
    qs = np.quantile(finite, np.linspace(0.0, 1.0, max(3, n_grid)))
    grid = np.unique(np.concatenate([[lo - eps], qs, [hi + eps]]))
    # Sort so retrieval_rate increases monotonically along the sweep
    if direction == "above":
        grid = grid[::-1]
    return grid


def sweep_thresholds(
    df: pd.DataFrame,
    scores: Sequence[float] | np.ndarray | pd.Series,
    direction: str = "above",
    thresholds: Optional[Sequence[float]] = None,
    *,
    n_grid: int = 101,
    is_pre_generation: bool = False,
    classifier_latency_ms: float = 0.0,
    weights: Optional[CostWeights] = None,
) -> pd.DataFrame:
    """Sweep thresholds over `scores` and return the full EM/F1 vs. retrieval-rate curve."""
    arr = np.asarray(scores, dtype=float)
    grid = (
        np.asarray(list(thresholds), dtype=float)
        if thresholds is not None
        else _default_threshold_grid(arr, direction=direction, n_grid=n_grid)
    )
    rows = []
    for t in grid:
        mask = apply_threshold(arr, float(t), direction=direction)
        metrics = evaluate_policy(
            df,
            mask,
            is_pre_generation=is_pre_generation,
            classifier_latency_ms=classifier_latency_ms,
            weights=weights,
        )
        rows.append({"threshold": float(t), **metrics})
    curve = pd.DataFrame(rows)
    return curve.sort_values(["retrieval_rate", "EM", "F1"], ascending=[True, False, False]).reset_index(drop=True)


def pick_threshold(
    curve: pd.DataFrame,
    always_rag_em: Optional[float] = None,
    em_tolerance: float = 0.01,
) -> float:
    """Select the operating threshold on validation matching always-RAG EM with minimum retrievals.

    Finds thresholds whose EM is at least `always_rag_em - em_tolerance` and returns the one
    with the lowest `retrieval_rate` (tie-breaking by highest EM, then highest F1).
    If no threshold reaches the target EM, selects the lowest-retrieval threshold within
    0.005 EM of the best achievable EM on the curve.
    """
    if curve.empty:
        raise ValueError("Cannot pick threshold from an empty curve.")
    target = float(always_rag_em) if always_rag_em is not None else float(curve.iloc[-1]["EM"])
    ok = curve[curve["EM"] >= target - em_tolerance]
    if ok.empty:
        best_em = float(curve["EM"].max())
        ok = curve[curve["EM"] >= best_em - 0.005]
    chosen = ok.sort_values(["retrieval_rate", "EM", "F1"], ascending=[True, False, False]).iloc[0]
    return float(chosen["threshold"])


def compute_trigger_auroc(
    df: pd.DataFrame,
    uncertainty_scores: Sequence[float] | np.ndarray | pd.Series,
    target_col: str = "retrieval_helps",
) -> float:
    """Compute ROC-AUC of `uncertainty_scores` (higher = more likely to retrieve) for `target_col`."""
    y = df[target_col].to_numpy(dtype=int)
    s = np.asarray(uncertainty_scores, dtype=float)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE), MCE, Brier score, and reliability-diagram bins.

    Uses equal-width bins on [0, 1]. Non-finite entries are dropped before computation.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], np.clip(p[valid], 0.0, 1.0)

    if len(y) == 0:
        return {
            "ece": float("nan"),
            "mce": float("nan"),
            "brier": float("nan"),
            "n_samples": 0,
            "bins_df": pd.DataFrame(
                columns=["bin_lower", "bin_upper", "bin_center", "count", "avg_confidence", "empirical_accuracy", "abs_gap"]
            ),
        }

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    n = len(y)

    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == n_bins - 1:
            in_bin = (p >= lo) & (p <= hi)
        else:
            in_bin = (p >= lo) & (p < hi)
        cnt = int(in_bin.sum())
        if cnt > 0:
            avg_conf = float(p[in_bin].mean())
            emp_acc = float(y[in_bin].mean())
            gap = abs(emp_acc - avg_conf)
            ece += (cnt / n) * gap
            mce = max(mce, gap)
        else:
            avg_conf, emp_acc, gap = float("nan"), float("nan"), float("nan")
        rows.append({
            "bin_lower": lo,
            "bin_upper": hi,
            "bin_center": 0.5 * (lo + hi),
            "count": cnt,
            "avg_confidence": avg_conf,
            "empirical_accuracy": emp_acc,
            "abs_gap": gap,
        })

    brier = float(brier_score_loss(y.astype(int), p))
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "n_samples": int(n),
        "bins_df": pd.DataFrame(rows),
    }


def fit_platt_scaler(
    scores_val: Sequence[float] | np.ndarray,
    y_val: Sequence[int] | np.ndarray,
) -> Optional[LogisticRegression]:
    """Fit a 1D logistic (Platt) calibrator mapping scalar score -> P(y=1 | score) on validation."""
    s = np.asarray(scores_val, dtype=float)
    y = np.asarray(y_val, dtype=int)
    valid = np.isfinite(s)
    s, y = s[valid], y[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(C=1.0, solver="lbfgs")
    clf.fit(s.reshape(-1, 1), y)
    return clf


def apply_platt_scaler(
    scaler: Optional[LogisticRegression],
    scores: Sequence[float] | np.ndarray,
    fallback_prob: float = 0.5,
) -> np.ndarray:
    """Apply a fitted Platt scaler to new scores (handling NaNs safely)."""
    s = np.asarray(scores, dtype=float)
    out = np.full(len(s), float(fallback_prob), dtype=float)
    if scaler is None or len(s) == 0:
        return out
    valid = np.isfinite(s)
    if valid.any():
        out[valid] = scaler.predict_proba(s[valid].reshape(-1, 1))[:, 1]
    return out


def confident_wrong_examples(
    df: pd.DataFrame,
    top_k: int = 15,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """Return the most confident direct answers that were wrong and fixed by RAG."""
    sub = df if dataset is None else df[df["dataset"] == dataset]
    bad = sub[sub["retrieval_helps"] == 1].copy()
    bad = bad.sort_values(["direct_conf", "direct_entropy"], ascending=[False, True])
    cols = [
        c
        for c in (
            "qid",
            "dataset",
            "question",
            "direct_answer",
            "rag_answer",
            "answer",
            "direct_logprob",
            "direct_conf",
            "direct_entropy",
        )
        if c in bad.columns
    ]
    return bad[cols].head(top_k).reset_index(drop=True)


def train_tfidf_trigger(
    train_df: pd.DataFrame,
    target_col: str = "retrieval_helps",
    seed: int = 0,
):
    """Fast CPU-only lexical baseline (TF-IDF + balanced LogisticRegression) for sanity checks."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    y = train_df[target_col].to_numpy(dtype=int)
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    pipe.fit(train_df["question"].astype(str).tolist(), y)
    return pipe


def predict_tfidf_proba(model, questions: Sequence[str]) -> Tuple[np.ndarray, float]:
    """Predict P(retrieve | question) and average inference latency in ms/query."""
    q_list = [str(q) for q in questions]
    t0 = time.perf_counter()
    probs = model.predict_proba(q_list)[:, 1]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(probs, dtype=float), elapsed_ms / max(1, len(q_list))


def train_distilbert_trigger(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    target_col: str = "retrieval_helps",
    model_name: str = "distilbert-base-uncased",
    seed: int = 0,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    use_class_weight: bool = True,
    device: Optional[str] = None,
) -> Tuple[Any, Any, pd.DataFrame]:
    """Fine-tune DistilBERT-base (66M) on question text to predict whether retrieval helps.

    Uses class-weighted CrossEntropyLoss when `use_class_weight=True` (addressing the
    imbalance risk documented in the DSA 4 Modeling Tasks) and keeps the checkpoint with
    the best validation AUROC (falling back to lowest validation loss if val has 1 class).

    Returns (best_model, tokenizer, history_df).
    """
    import copy
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _make_loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        enc = tokenizer(
            frame["question"].astype(str).tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(frame[target_col].to_numpy(dtype=int), dtype=torch.long)
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(train_df, shuffle=True)
    val_loader = _make_loader(val_df, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(dev)

    y_train = train_df[target_col].to_numpy(dtype=int)
    if use_class_weight and len(np.unique(y_train)) == 2:
        cw = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_score = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(dev)
            attention_mask = attention_mask.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses, val_probs, val_labels = [], [], []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(dev)
                attention_mask = attention_mask.to(dev)
                labels = labels.to(dev)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                val_losses.append(float(loss_fn(logits, labels).item()))
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.cpu().numpy().tolist())

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_auroc = (
            float(roc_auc_score(val_labels, val_probs))
            if len(np.unique(val_labels)) == 2
            else float("nan")
        )
        score = val_auroc if np.isfinite(val_auroc) else -val_loss
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val_loss": val_loss,
            "val_auroc": val_auroc,
        })

    model.load_state_dict(best_state)
    model.eval()
    return model, tokenizer, pd.DataFrame(history)


def predict_distilbert_proba(
    model: Any,
    tokenizer: Any,
    questions: Sequence[str],
    *,
    batch_size: int = 64,
    max_length: int = 128,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Predict P(retrieve | question) with a trained DistilBERT model.

    Returns (probabilities_array, mean_latency_ms_per_query).
    """
    import torch

    q_list = [str(q) for q in questions]
    if not q_list:
        return np.array([], dtype=float), 0.0

    dev = torch.device(device or str(next(model.parameters()).device))
    model.eval()
    all_probs: list[float] = []

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(q_list), batch_size):
            batch = q_list[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(dev)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(all_probs, dtype=float), elapsed_ms / len(q_list)


def evaluate_triggers_across_seeds(
    df: pd.DataFrame,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    em_tolerance: float = 0.01,
    v4_probs_by_seed: Optional[Mapping[int, Mapping[str, float]]] = None,
    v4_latency_ms_by_seed: Optional[Mapping[int, float]] = None,
    weights: Optional[CostWeights] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run V1 (never/always), V2 (log-prob), V3 (entropy), Oracle, and optional V4 across seeds.

    Tunes thresholds strictly on TriviaQA validation (`split_s{seed} == 'val'`) and evaluates:
      1. `triviaqa_test` (in-domain test)
      2. `hotpotqa_transfer` (HotpotQA test using TriviaQA-tuned threshold unchanged!)
      3. `hotpotqa_indomain` (HotpotQA test using HotpotQA-val-tuned threshold as upper reference)

    Parameters
    ----------
    v4_probs_by_seed : optional dict mapping `seed -> {qid: p_retrieve}` for V4 DistilBERT.
    v4_latency_ms_by_seed : optional dict mapping `seed -> mean_classifier_latency_ms`.

    Returns
    -------
    (per_seed_df, summary_df) where `summary_df` is aggregated via `dsa4.evaluate.aggregate_seed_results`.
    """
    w = weights or DEFAULT_COST_WEIGHTS
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        scol = f"split_s{seed}"
        tq_val = df[(df["dataset"] == "triviaqa") & (df[scol] == "val")].reset_index(drop=True)
        tq_test = df[(df["dataset"] == "triviaqa") & (df[scol] == "test")].reset_index(drop=True)
        hq_val = df[(df["dataset"] == "hotpotqa") & (df[scol] == "val")].reset_index(drop=True)
        hq_test = df[(df["dataset"] == "hotpotqa") & (df[scol] == "test")].reset_index(drop=True)

        eval_sets = [
            ("TriviaQA (in-domain)", tq_test, False),
            ("HotpotQA (transfer)", hq_test, False),
            ("HotpotQA (in-domain)", hq_test, True),
        ]

        # Fixed policies (V1) and Oracle on each evaluation set
        for eval_name, test_df, is_hq_indomain in eval_sets:
            if is_hq_indomain:
                # Fixed policies and oracle have no threshold to retune; avoid duplicate rows
                continue
            rows.append({
                "eval_set": eval_name,
                "method": "V1a Never retrieve",
                "seed": seed,
                "threshold": float("nan"),
                "auroc_helps": float("nan"),
                "auroc_wrong": float("nan"),
                **evaluate_policy(test_df, np.zeros(len(test_df), dtype=bool), is_pre_generation=True, weights=w),
            })
            rows.append({
                "eval_set": eval_name,
                "method": "V1b Always retrieve",
                "seed": seed,
                "threshold": float("nan"),
                "auroc_helps": float("nan"),
                "auroc_wrong": float("nan"),
                **evaluate_policy(test_df, np.ones(len(test_df), dtype=bool), is_pre_generation=True, weights=w),
            })
            rows.append({
                "eval_set": eval_name,
                "method": "Oracle (cheapest correct)",
                "seed": seed,
                "threshold": float("nan"),
                "auroc_helps": float("nan"),
                "auroc_wrong": float("nan"),
                **evaluate_oracle(test_df, weights=w),
            })

        # Scalar threshold policies: V2 log-prob, V3 entropy, and optional V4 DistilBERT
        trigger_specs: list[dict[str, Any]] = [
            {
                "method": "V2 Log-probability",
                "direction": "below",
                "score_fn": lambda f: f["direct_logprob"].to_numpy(dtype=float),
                "uncert_fn": lambda f: f["neg_logprob"].to_numpy(dtype=float),
                "is_pre_gen": False,
                "clf_ms": 0.0,
            },
            {
                "method": "V3 Predictive entropy",
                "direction": "above",
                "score_fn": lambda f: f["direct_entropy"].to_numpy(dtype=float),
                "uncert_fn": lambda f: f["direct_entropy"].to_numpy(dtype=float),
                "is_pre_gen": False,
                "clf_ms": 0.0,
            },
        ]

        if v4_probs_by_seed is not None and seed in v4_probs_by_seed:
            pmap = v4_probs_by_seed[seed]
            clf_ms = float(v4_latency_ms_by_seed.get(seed, 0.0)) if v4_latency_ms_by_seed else 0.0
            trigger_specs.append({
                "method": "V4 DistilBERT classifier",
                "direction": "above",
                "score_fn": lambda f, m=pmap: np.array([m.get(qid, np.nan) for qid in f["qid"]], dtype=float),
                "uncert_fn": lambda f, m=pmap: np.array([m.get(qid, np.nan) for qid in f["qid"]], dtype=float),
                "is_pre_gen": True,
                "clf_ms": clf_ms,
            })

        for spec in trigger_specs:
            tq_val_scores = spec["score_fn"](tq_val)
            hq_val_scores = spec["score_fn"](hq_val)

            tq_curve = sweep_thresholds(
                tq_val,
                tq_val_scores,
                direction=spec["direction"],
                is_pre_generation=spec["is_pre_gen"],
                classifier_latency_ms=spec["clf_ms"],
                weights=w,
            )
            t_tq = pick_threshold(tq_curve, always_rag_em=float(tq_val["rag_em"].mean()), em_tolerance=em_tolerance)

            hq_curve = sweep_thresholds(
                hq_val,
                hq_val_scores,
                direction=spec["direction"],
                is_pre_generation=spec["is_pre_gen"],
                classifier_latency_ms=spec["clf_ms"],
                weights=w,
            )
            t_hq = pick_threshold(hq_curve, always_rag_em=float(hq_val["rag_em"].mean()), em_tolerance=em_tolerance)

            for eval_name, test_df, is_hq_indomain in eval_sets:
                thresh = t_hq if is_hq_indomain else t_tq
                test_scores = spec["score_fn"](test_df)
                test_uncert = spec["uncert_fn"](test_df)
                mask = apply_threshold(test_scores, thresh, direction=spec["direction"])
                metrics = evaluate_policy(
                    test_df,
                    mask,
                    is_pre_generation=spec["is_pre_gen"],
                    classifier_latency_ms=spec["clf_ms"],
                    weights=w,
                )
                rows.append({
                    "eval_set": eval_name,
                    "method": spec["method"],
                    "seed": seed,
                    "threshold": thresh,
                    "auroc_helps": compute_trigger_auroc(test_df, test_uncert, target_col="retrieval_helps"),
                    "auroc_wrong": compute_trigger_auroc(test_df, test_uncert, target_col="direct_wrong"),
                    **metrics,
                })

    per_seed = pd.DataFrame(rows)
    metric_cols = [
        "threshold",
        "EM",
        "F1",
        "retrieval_rate",
        "retrievals_saved",
        "auroc_helps",
        "auroc_wrong",
        "retrievals_per_1k",
        "tokens_per_1k",
        "cost_per_1k",
        "mean_latency_ms",
        "p95_latency_ms",
    ]
    summary = aggregate_seed_results(
        per_seed,
        group_cols=["eval_set", "method"],
        metric_cols=metric_cols,
        seed_col="seed",
        decimals=3,
    )
    return per_seed, summary


__all__ = [
    "DEFAULT_COST_WEIGHTS",
    "prepare_retrieval_frame",
    "apply_threshold",
    "evaluate_policy",
    "evaluate_oracle",
    "sweep_thresholds",
    "pick_threshold",
    "compute_trigger_auroc",
    "expected_calibration_error",
    "fit_platt_scaler",
    "apply_platt_scaler",
    "confident_wrong_examples",
    "train_tfidf_trigger",
    "predict_tfidf_proba",
    "train_distilbert_trigger",
    "predict_distilbert_proba",
    "evaluate_triggers_across_seeds",
]
