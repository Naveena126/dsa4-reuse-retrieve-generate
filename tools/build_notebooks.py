"""Builds the Kaggle runner notebooks in notebooks/ from the cells below.

    python tools/build_notebooks.py
Change REPO_URL first to your GitHub repository.
"""
from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/YOUR-USERNAME/dsa4-reuse-retrieve-generate.git"
ROOT = Path(__file__).resolve().parents[1]

SETUP = f"""# Setup: clone the team repository and install what Kaggle lacks
import os, subprocess, sys
REPO_URL = "{REPO_URL}"
if not os.path.exists("/kaggle/working/repo"):
    subprocess.run(["git", "clone", "-q", REPO_URL, "/kaggle/working/repo"], check=True)
else:
    subprocess.run(["git", "-C", "/kaggle/working/repo", "pull", "-q"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu", "bitsandbytes>=0.43"], check=True)
os.chdir("/kaggle/working/repo")
sys.path.insert(0, "src")
print("ready")"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def save(cells, name):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, ROOT / "notebooks" / name)
    print("wrote", name)


prepare = [
    md("# DSA 4 · 00 Prepare data (Naveena, Day 1)\n\nSettings: **GPU T4**, **Internet on**. "
       "Samples 1,000 TriviaQA + 1,000 HotpotQA questions, makes the splits and blocks, builds the "
       "FAISS index. Publish `/kaggle/working/data` as the dataset **dsa4-data**."),
    code(SETUP),
    code("!python scripts/prepare_data.py --out /kaggle/working/data"),
    code("!ls -lh /kaggle/working/data"),
]

generate = [
    md("# DSA 4 · 01 Generate my block (everyone, Day 1 evening)\n\n"
       "Settings: **GPU T4**, **Internet on**. Add Input → **dsa4-data**.\n\n"
       "1. Set `BLOCK` below (Naveena 1, Manimegalai 2, Meet 3, Vijaya 4).\n"
       "2. Run the setup cell and the **smoke test** (20 questions) interactively. Post the printed "
       "seconds per question in the team chat.\n"
       "3. Comment out the smoke test, then **Save Version → Save & Run All** for the full block.\n"
       "4. Next morning: Output → **New Dataset** `dsa4-block-<BLOCK>` (later runs: New Version)."),
    code(SETUP),
    code('BLOCK = 1      # <- change to your block\n'
         'DATA = "/kaggle/input/dsa4-data"\n'
         'OUT = f"/kaggle/working/block{BLOCK}"\n'
         'MODEL = "Qwen/Qwen2.5-7B-Instruct"   # fallback: "Qwen/Qwen2.5-3B-Instruct"\n'
         '# If you are resuming, add your own dsa4-block-<BLOCK> dataset as input and set:\n'
         'RESUME = ""  # e.g. f"/kaggle/input/dsa4-block-{BLOCK}/block{BLOCK}.jsonl"'),
    code('# Smoke test: 20 questions. Check the answers are short and the variants look right.\n'
         '!python scripts/generate_block.py --block {BLOCK} --data {DATA} --out /kaggle/working/smoke '
         '--model {MODEL} --limit 20\n'
         'import pandas as pd\n'
         'sm = pd.read_json(f"/kaggle/working/smoke/block{BLOCK}.jsonl", lines=True)\n'
         'pd.set_option("display.max_colwidth", 80)\n'
         'sm[["qid"]].assign(direct=sm["direct"].str["answer"], rag=sm["rag"].str["answer"],\n'
         '                   lookalike=sm["variants"].str["lookalike"])'),
    code('# Full run for the whole block (resumable)\n'
         'resume = f"--resume-from {RESUME}" if RESUME else ""\n'
         '!python scripts/generate_block.py --block {BLOCK} --data {DATA} --out {OUT} --model {MODEL} {resume}'),
]

merge = [
    md("# DSA 4 · 02 Merge blocks (Naveena, Day 2)\n\nSettings: CPU is enough. Add Input → **dsa4-data** "
       "and all four **dsa4-block-N** datasets. Publish `/kaggle/working/cache` as **dsa4-cache**."),
    code(SETUP),
    code("!python scripts/merge_blocks.py --data /kaggle/input/dsa4-data "
         "--blocks /kaggle/input/dsa4-block-1 /kaggle/input/dsa4-block-2 "
         "/kaggle/input/dsa4-block-3 /kaggle/input/dsa4-block-4 --out /kaggle/working/cache"),
    code('import pandas as pd\n'
         'c = pd.read_json("/kaggle/working/cache/cache.jsonl", lines=True)\n'
         'acc = pd.DataFrame({"direct": c["direct"].str["correct"], "rag": c["rag"].str["correct"], '
         '"dataset": c["dataset"]})\n'
         'acc.groupby("dataset").mean().round(3)'),
]

if __name__ == "__main__":
    save(prepare, "00_prepare_data.ipynb")
    save(generate, "01_generate_block.ipynb")
    save(merge, "02_merge_blocks.ipynb")
