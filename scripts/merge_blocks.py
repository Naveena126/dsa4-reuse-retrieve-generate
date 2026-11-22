"""Day 2: merge the four block files into the shared answer cache.

    python scripts/merge_blocks.py --data /kaggle/input/dsa4-data \
        --blocks /kaggle/input/dsa4-block-1 /kaggle/input/dsa4-block-2 \
                 /kaggle/input/dsa4-block-3 /kaggle/input/dsa4-block-4 \
        --out /kaggle/working/cache

Writes cache.jsonl (one line per question) with correctness labels added:
direct.correct and rag.correct = normalised exact match against the aliases.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsa4 import data  # noqa: E402
from dsa4.text import exact_match  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--blocks", nargs="+", required=True, help="folders (or files) holding block<N>.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    q = data.load_questions(args.data).set_index("qid")
    recs = {}
    for b in args.blocks:
        files = [Path(b)] if Path(b).is_file() else sorted(Path(b).rglob("block*.jsonl"))
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        recs[r["qid"]] = r  # later files win if a question appears twice

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_bad = 0
    with open(out / "cache.jsonl", "w", encoding="utf-8") as f:
        for qid, r in recs.items():
            aliases = q.loc[qid, "aliases"]
            for mode in ("direct", "rag"):
                r[mode]["correct"] = bool(exact_match(r[mode]["answer"], aliases))
            r["question"] = q.loc[qid, "question"]
            r["aliases"] = list(aliases)
            n_bad += r.get("variants") is None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    missing = sorted(set(q.index) - set(recs))
    print(f"merged {len(recs)} questions; {len(missing)} missing; {n_bad} without usable variants")
    if missing:
        print("first missing:", missing[:5])


if __name__ == "__main__":
    main()
