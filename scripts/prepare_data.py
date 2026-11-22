"""Day 1: sample questions, make splits and blocks, build the passage index.

Run once (on Kaggle, GPU on, internet on):
    python scripts/prepare_data.py --out data
Then publish the output folder as the Kaggle dataset `dsa4-data`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from dsa4 import data, index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=data.N_PER_DATASET)
    ap.add_argument("--recall-sample", type=int, default=200)
    args = ap.parse_args()

    q = pd.concat([data.load_triviaqa(args.n), data.load_hotpotqa(args.n)], ignore_index=True)
    q = data.assign_blocks(data.assign_splits(q))
    data.save_questions(q, args.out)
    print("questions:", q.groupby(["dataset", "block"]).size().unstack().to_string())
    print("split s0:", q.groupby(["dataset", "split_s0"]).size().unstack().to_string())

    passages = index.build_passages(q)
    print("passages:", len(passages))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(index.EMBED_MODEL)
    vecs = index.embed([index.passage_text(p) for p in passages.to_dict("records")], model=model)
    idx = index.build_index(vecs)
    index.save_index(idx, passages, args.out)

    sample = q.sample(min(args.recall_sample, len(q)), random_state=0)
    qv = index.embed(sample["question"].tolist(), model=model, is_query=True)
    _, pids = index.retrieve(idx, qv, k=5)
    rec = index.recall_at_k(sample, passages, pids)
    print("Recall@5:", rec.groupby(sample["dataset"]).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
