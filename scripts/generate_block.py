"""Generate one block of the shared answer cache (each member runs their own block).

For every question in the block:
  1. two paraphrases + one look-alike (JSON from Qwen),
  2. a direct answer (no retrieval),
  3. a RAG answer with the top-5 passages,
saving the text, mean token log-probability, mean entropy, passage IDs and latency.

The output file is appended line by line, so a stopped run resumes where it left off.

    python scripts/generate_block.py --block 1 --data /kaggle/input/dsa4-data --out /kaggle/working/block1
    python scripts/generate_block.py --block 1 ... --limit 20      # smoke test
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsa4 import data, index  # noqa: E402
from dsa4.generation import (QwenGenerator, answer_messages, clean_answer,  # noqa: E402
                             parse_variants, variant_messages)


def done_ids(path: Path):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["qid"] for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--data", required=True, help="folder with questions.jsonl, passages.jsonl, passages.faiss")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--resume-from", default="", help="earlier block file to copy in before resuming")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"block{args.block}.jsonl"
    if args.resume_from and not out_path.exists():
        out_path.write_bytes(Path(args.resume_from).read_bytes())

    q = data.load_questions(args.data)
    q = q[q["block"] == args.block].reset_index(drop=True)
    if args.limit:
        q = q.head(args.limit)
    skip = done_ids(out_path)
    todo = q[~q["qid"].isin(skip)]
    print(f"block {args.block}: {len(q)} questions, {len(skip)} done, {len(todo)} to go")
    if todo.empty:
        return

    idx, passages = index.load_index(args.data)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(index.EMBED_MODEL)
    gen = QwenGenerator(args.model, load_4bit=not args.no_4bit)

    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo.itertuples(), 1):
            rec = {"qid": r.qid, "dataset": r.dataset, "block": args.block, "model": args.model}

            # 1. variants (retry once if the JSON is unusable)
            variants = None
            for _ in range(2):
                v = gen.generate(variant_messages(r.question, r.answer), max_new_tokens=160, with_scores=False)
                variants = parse_variants(v["text"])
                if variants:
                    break
            rec["variants"] = variants  # None = failed; excluded from the stream later

            # 2. direct answer
            t0 = time.time()
            d = gen.generate(answer_messages(r.question))
            rec["direct"] = {"answer": clean_answer(d["text"]), "raw": d["text"],
                             "logprob_mean": d["logprob_mean"], "entropy_mean": d["entropy_mean"],
                             "n_tokens": d["n_tokens"], "latency_s": round(time.time() - t0, 3)}

            # 3. RAG answer with the top-5 passages
            t0 = time.time()
            qv = index.embed([r.question], model=embedder, is_query=True)
            scores, pids = index.retrieve(idx, qv, k=5)
            pids = [int(p) for p in pids[0] if p >= 0]
            texts = [index.passage_text(passages.iloc[p]) for p in pids]
            t_retr = time.time() - t0
            g = gen.generate(answer_messages(r.question, texts))
            rec["rag"] = {"answer": clean_answer(g["text"]), "raw": g["text"],
                          "logprob_mean": g["logprob_mean"], "entropy_mean": g["entropy_mean"],
                          "n_tokens": g["n_tokens"], "passage_ids": pids,
                          "passage_scores": [round(float(s), 4) for s in scores[0][:len(pids)]],
                          "retrieval_s": round(t_retr, 3), "latency_s": round(time.time() - t0, 3)}

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                per_q = (time.time() - t_start) / n
                left_h = per_q * (len(todo) - n) / 3600
                print(f"{n}/{len(todo)}  {per_q:.1f} s/question  ~{left_h:.1f} h left", flush=True)


if __name__ == "__main__":
    main()
