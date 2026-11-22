"""Question sampling, splits and block assignment (owner: Naveena).

Everything is seeded, so running it twice gives the same questions,
the same splits and the same blocks.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_PER_DATASET = 1000
SPLIT_SEEDS = (0, 1, 2)
N_BLOCKS = 4
MAX_TQA_CHARS = 20000  # keep the start of each Wikipedia page; enough for top-5 retrieval


def load_triviaqa(n: int = N_PER_DATASET, seed: int = 0) -> pd.DataFrame:
    """Sample n TriviaQA rc.wikipedia validation questions that have evidence pages."""
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
    keep = [i for i, ex in enumerate(ds["entity_pages"]) if any(c.strip() for c in ex["wiki_context"])]
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(keep, size=n, replace=False).tolist())
    rows = []
    for i in idx:
        ex = ds[int(i)]
        ans = ex["answer"]
        aliases = sorted({ans["value"], *ans["aliases"]})
        docs = [
            {"title": t, "text": c[:MAX_TQA_CHARS]}
            for t, c in zip(ex["entity_pages"]["title"], ex["entity_pages"]["wiki_context"])
            if c.strip()
        ]
        rows.append({
            "qid": f"tqa_{ex['question_id']}",
            "dataset": "triviaqa",
            "question": ex["question"],
            "answer": ans["value"],
            "aliases": aliases,
            "docs": docs,
            "gold_titles": [],
        })
    return pd.DataFrame(rows)


def load_hotpotqa(n: int = N_PER_DATASET, seed: int = 0) -> pd.DataFrame:
    """Sample n HotpotQA distractor validation questions (2 gold + 8 distractor paragraphs each)."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
    rows = []
    for i in idx:
        ex = ds[int(i)]
        docs = [
            {"title": t, "text": " ".join(s).strip()}
            for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])
        ]
        rows.append({
            "qid": f"hqa_{ex['id']}",
            "dataset": "hotpotqa",
            "question": ex["question"],
            "answer": ex["answer"],
            "aliases": [ex["answer"]],
            "docs": docs,
            "gold_titles": sorted(set(ex["supporting_facts"]["title"])),
        })
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame, seeds=SPLIT_SEEDS, frac=(0.6, 0.2, 0.2)) -> pd.DataFrame:
    """Add split_s<seed> columns: 60/20/20 by source question, stratified by dataset.

    Splitting by source question means a question and all its variants
    always land in the same split.
    """
    out = df.copy()
    for seed in seeds:
        col = f"split_s{seed}"
        out[col] = ""
        for _, sub in out.groupby("dataset"):
            ids = sub.index.to_numpy().copy()
            np.random.default_rng(seed).shuffle(ids)
            n_tr = int(round(frac[0] * len(ids)))
            n_va = int(round(frac[1] * len(ids)))
            out.loc[ids[:n_tr], col] = "train"
            out.loc[ids[n_tr:n_tr + n_va], col] = "val"
            out.loc[ids[n_tr + n_va:], col] = "test"
    return out


def assign_blocks(df: pd.DataFrame, n_blocks: int = N_BLOCKS, seed: int = 0) -> pd.DataFrame:
    """Add a 'block' column (1..n_blocks), balanced per dataset (250 + 250 each for 1,000 + 1,000)."""
    out = df.copy()
    out["block"] = 0
    for _, sub in out.groupby("dataset"):
        ids = sub.index.to_numpy().copy()
        np.random.default_rng(seed).shuffle(ids)
        for k, i in enumerate(ids):
            out.loc[i, "block"] = k % n_blocks + 1
    return out


def save_questions(df: pd.DataFrame, data_dir) -> Path:
    """Write questions.jsonl (one question per line, including its documents)."""
    path = Path(data_dir) / "questions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_questions(data_dir) -> pd.DataFrame:
    return pd.read_json(Path(data_dir) / "questions.jsonl", lines=True)
