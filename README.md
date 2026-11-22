# DSA 4 · Reuse, Retrieve or Generate?

Shared code for the DSA 4 Action Learning project (Modeling Defense 1).

## What is here

| Path | What it does | Owner |
| --- | --- | --- |
| `src/dsa4/data.py` | Samples 1,000 TriviaQA + 1,000 HotpotQA questions, 60/20/20 splits by source question (seeds 0–2), blocks 1–4 | Naveena |
| `src/dsa4/index.py` | Passages, bge-base-en-v1.5 embeddings, FAISS index, top-5 retrieval, Recall@5 | Naveena |
| `src/dsa4/stream.py` | Question stream (sources, paraphrases, look-alikes, 0/20/50% repeats) and the safe-reuse rule | Naveena |
| `src/dsa4/reuse.py` | Task 1 cache simulation and metrics | Naveena |
| `src/dsa4/generation.py` | Qwen prompts, generation, token log-probability and entropy | Naveena |
| `src/dsa4/text.py` | Answer normalisation and exact match | shared |
| `src/dsa4/evaluate.py` | EM / F1 tables over seeds | Manimegalai (to add) |
| `src/dsa4/nli.py` | NLI support checker | Meet (to add) |
| `src/dsa4/cost.py` | Cost and latency model, oracle | Vijaya (to add) |
| `scripts/prepare_data.py` | Day 1: questions, splits, blocks, index → Kaggle dataset `dsa4-data` | Naveena |
| `scripts/generate_block.py` | Day 1 evening: one block of the answer cache (everyone runs their own) | Naveena |
| `scripts/merge_blocks.py` | Day 2: merge the four blocks → Kaggle dataset `dsa4-cache` | Naveena |
| `notebooks/00–02_*.ipynb` | Kaggle runners for the three scripts | — |
| `notebooks/NaveenaSivapuram_Cache_Reuse.ipynb` | Task 1 notebook (exact match, V1, V2, transfer, error analysis) | Naveena |

## Set up the repository (Naveena, once)

```bash
unzip dsa4-reuse-retrieve-generate.zip && cd dsa4-reuse-retrieve-generate
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas faiss-cpu pytest nbformat
python -m pytest -q tests                       # expect 13 passed
git init && git add . && git commit -m "Shared code: data, index, stream, generator, Task 1 starter"
git branch -M main
git remote add origin https://github.com/<your-username>/dsa4-reuse-retrieve-generate.git
git push -u origin main
```

Make the repository **public** (Kaggle clones it) and add the other three as collaborators.
In every Kaggle notebook, set `REPO_URL` in the first code cell to your repository.

## Kaggle order

1. `00_prepare_data` (Naveena, GPU) → publish output as **dsa4-data**.
2. `01_generate_block` (everyone, GPU, own account, own `BLOCK`) → smoke test first, then Save & Run All → publish **dsa4-block-N**.
3. `02_merge_blocks` (Naveena, CPU) → publish **dsa4-cache**.
4. Task notebooks: add **dsa4-data** and **dsa4-cache** as inputs.

Never commit `data/`, the cache, embeddings or `kaggle.json`; `.gitignore` blocks them.

## The reuse rule every task uses

Reusing a cached answer is **safe** only if the cached question and the new question have the same correct answer: same source question, and not a look-alike (`stream.answer_key`). A paraphrase that misses the cache uses its source question's stored answer; look-alikes count only for unsafe reuse, not for accuracy.
