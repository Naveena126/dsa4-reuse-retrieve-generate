"""Builds notebooks/NaveenaSivapuram_Cache_Reuse.ipynb (Task 1 starter: exact match, V1, V2).

    python tools/build_task1_notebook.py
"""
from pathlib import Path

import nbformat as nbf

from build_notebooks import SETUP, code, md

ROOT = Path(__file__).resolve().parents[1]

cells = [
    md("# Task 1 · Cache Reuse Rule\n**Owner:** Naveena Sivapuram · DSA 4\n\n"
       "Which matching rule saves the most LLM calls without reusing answers for look-alike questions?\n\n"
       "Kaggle settings: **GPU T4** (only for the embeddings), **Internet on**. "
       "Add Input → **dsa4-data** and **dsa4-cache**."),
    md("## 1. Introduction\n\n*Write this in your own words from the Task 1 page of the Modeling Tasks "
       "document: the problem, the research question, the hypothesis and the four papers.*"),
    md("## 2. Setup and data"),
    code(SETUP),
    code('import numpy as np, pandas as pd, matplotlib.pyplot as plt\n'
         'from pathlib import Path\n'
         'from dsa4 import data, stream, reuse\n\n'
         'DATA = "/kaggle/input/dsa4-data"\n'
         'CACHE = "/kaggle/input/dsa4-cache/cache.jsonl"\n'
         'MET, FIG = Path("results/metrics"), Path("results/figures")\n'
         'MET.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)\n\n'
         'q = data.load_questions(DATA)\n'
         'cache = pd.read_json(CACHE, lines=True)\n'
         'cache = cache[cache["variants"].notna()]          # questions whose variants were generated\n'
         'var = stream.variants_table(cache)\n'
         'q = q[q["qid"].isin(cache["qid"])].reset_index(drop=True)\n'
         'print(len(q), "source questions;", var["kind"].value_counts().to_dict())'),
    md("## 3. Data exploration"),
    code('acc = pd.DataFrame({"dataset": cache["dataset"], "direct": cache["direct"].str["correct"],\n'
         '                    "rag": cache["rag"].str["correct"]})\n'
         'display(acc.groupby("dataset")[["direct", "rag"]].mean().round(3).rename(columns=lambda c: c + " accuracy"))\n'
         'display(var[var["kind"] == "lookalike"]["changed"].value_counts().rename("look-alikes by type of change"))\n'
         'pd.set_option("display.max_colwidth", 90)\n'
         'ex = var[var["qid"] == var["qid"].iloc[0]]\n'
         'print("Source:", q.set_index("qid").loc[ex["qid"].iloc[0], "question"])\n'
         'display(ex[["kind", "text"]])'),
    md("**Hand-check.** *Add the agreement rate from the 100 hand-checked variants here once the shared "
       "sheet is complete (share of paraphrases that really keep the meaning, and of look-alikes that "
       "really change the answer).*"),
    md("## 4. Embeddings\n\nEvery source question and variant is embedded once with each model "
       "(normalised vectors, so the dot product is the cosine similarity)."),
    code('from sentence_transformers import SentenceTransformer\n\n'
         'texts = pd.concat([q[["qid", "question"]].rename(columns={"question": "text"}).assign(vid=q["qid"]),\n'
         '                   var[["vid", "text"]]], ignore_index=True).drop_duplicates("vid")\n'
         'MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-base-en-v1.5"}\n'
         'EMB = {}\n'
         'for name, mid in MODELS.items():\n'
         '    m = SentenceTransformer(mid)\n'
         '    v = m.encode(texts["text"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=True)\n'
         '    EMB[name] = dict(zip(texts["vid"], v.astype("float32")))\n'
         'print({k: len(v) for k, v in EMB.items()})'),
    md("## 5. Method\n\nThe stream replays each split's source questions, then their paraphrases and "
       "look-alikes at random later positions. A query reuses the closest cached entry if the rule says so; "
       "otherwise it is answered and stored. A reuse is **safe** only if both questions have the same "
       "correct answer.\n\n"
       "- **calls saved** = reuses ÷ all queries\n"
       "- **unsafe reuse** = unsafe reuses ÷ all reuses (our 2% target)\n"
       "- **safe recall** = safe reuses ÷ queries that had a safe entry available\n\n"
       "Thresholds are chosen on the validation split (most calls saved with unsafe reuse ≤ 2%) and "
       "reported on the test split, over seeds 0, 1 and 2."),
    code('THRESH = np.round(np.arange(0.60, 1.0001, 0.01), 2)\n'
         'TARGET = 0.02\n'
         'RULES = {"V1 MiniLM": "minilm", "V1 bge": "bge", "V2 ensemble": "ensemble"}\n\n'
         'def make_stream(seed, split, datasets=("triviaqa", "hotpotqa"), repeat=0.0):\n'
         '    src = q[(q[f"split_s{seed}"] == split) & q["dataset"].isin(datasets)]\n'
         '    return stream.build_stream(src[["qid", "question"]], var[var["qid"].isin(src["qid"])],\n'
         '                               repeat_rate=repeat, seed=seed)\n\n'
         'def sim_for(s, rule):\n'
         '    if rule == "ensemble":\n'
         '        return (sim_for(s, "minilm") + sim_for(s, "bge")) / 2\n'
         '    return reuse.similarity_matrix(np.stack([EMB[rule][v] for v in s["vid"]]))\n\n'
         's = make_stream(0, "val")\n'
         'print(len(s), "events in the seed-0 validation stream:", s["kind"].value_counts().to_dict())'),
    md("## 6. Results: exact match, V1 and V2"),
    code('rows, curves = [], {}\n'
         'for seed in (0, 1, 2):\n'
         '    sv, st = make_stream(seed, "val"), make_stream(seed, "test")\n'
         '    rows.append({"rule": "V0 exact match", "seed": seed, "threshold": np.nan,\n'
         '                 **reuse.summarise(reuse.simulate_exact(st))})\n'
         '    for label, rule in RULES.items():\n'
         '        cv = reuse.sweep(sv, sim_for(sv, rule), THRESH)\n'
         '        curves[(label, seed)] = cv\n'
         '        t = reuse.pick_threshold(cv, TARGET)\n'
         '        rows.append({"rule": label, "seed": seed, "threshold": t,\n'
         '                     **reuse.summarise(reuse.simulate(st, sim_for(st, rule), t))})\n'
         'res = pd.DataFrame(rows)\n'
         'res.to_csv(MET / "task1_results_by_seed.csv", index=False)\n'
         'summary = res.groupby("rule")[["threshold", "calls_saved", "unsafe_reuse", "safe_recall"]].agg(["mean", "std"]).round(3)\n'
         'summary.to_csv(MET / "task1_summary.csv")\n'
         'summary'),
    code('fig, ax = plt.subplots(figsize=(6, 4))\n'
         'for label in RULES:\n'
         '    cv = curves[(label, 0)]\n'
         '    ax.plot(cv["unsafe_reuse"] * 100, cv["calls_saved"] * 100, marker=".", label=label)\n'
         'ax.axvline(TARGET * 100, color="grey", ls="--", lw=1)\n'
         'ax.set_xlabel("Unsafe reuse (% of reuses)"); ax.set_ylabel("Calls saved (% of queries)")\n'
         'ax.set_title("Savings vs. risk on validation (seed 0)"); ax.legend(); ax.grid(alpha=.3)\n'
         'fig.tight_layout(); fig.savefig(FIG / "task1_savings_vs_risk.png", dpi=150)'),
    code('# Same rules on a stream where 20% of queries are exact repeats\n'
         'rows20 = []\n'
         'for seed in (0, 1, 2):\n'
         '    st = make_stream(seed, "test", repeat=0.2)\n'
         '    for label, rule in RULES.items():\n'
         '        t = res[(res["rule"] == label) & (res["seed"] == seed)]["threshold"].iloc[0]\n'
         '        rows20.append({"rule": label, "seed": seed, **reuse.summarise(reuse.simulate(st, sim_for(st, rule), t))})\n'
         'pd.DataFrame(rows20).groupby("rule")[["calls_saved", "unsafe_reuse"]].agg(["mean", "std"]).round(3)'),
    md("## 7. Transfer: TriviaQA → HotpotQA\n\nThreshold tuned on TriviaQA validation, applied without "
       "retuning to HotpotQA test, compared with tuning on HotpotQA itself."),
    code('rows_t = []\n'
         'for seed in (0, 1, 2):\n'
         '    tq_val = make_stream(seed, "val", ("triviaqa",))\n'
         '    hq_val, hq_test = make_stream(seed, "val", ("hotpotqa",)), make_stream(seed, "test", ("hotpotqa",))\n'
         '    for label, rule in RULES.items():\n'
         '        t_tq = reuse.pick_threshold(reuse.sweep(tq_val, sim_for(tq_val, rule), THRESH), TARGET)\n'
         '        t_hq = reuse.pick_threshold(reuse.sweep(hq_val, sim_for(hq_val, rule), THRESH), TARGET)\n'
         '        for tuned, t in (("TriviaQA (transfer)", t_tq), ("HotpotQA (in-domain)", t_hq)):\n'
         '            rows_t.append({"rule": label, "tuned_on": tuned, "seed": seed, "threshold": t,\n'
         '                           **reuse.summarise(reuse.simulate(hq_test, sim_for(hq_test, rule), t))})\n'
         'transfer = pd.DataFrame(rows_t)\n'
         'transfer.to_csv(MET / "task1_transfer.csv", index=False)\n'
         'transfer.groupby(["rule", "tuned_on"])[["threshold", "calls_saved", "unsafe_reuse"]].mean().round(3)'),
    md("## 8. Error analysis\n\nWhich unsafe reuses does each rule make? Look-alikes are grouped by what "
       "was changed; \"different question\" means a source or paraphrase matched an unrelated cached question."),
    code('vmeta = var.set_index("vid")\n'
         'def failures(seed, label):\n'
         '    st = make_stream(seed, "test")\n'
         '    t = res[(res["rule"] == label) & (res["seed"] == seed)]["threshold"].iloc[0]\n'
         '    r = reuse.simulate(st, sim_for(st, RULES[label]), t)\n'
         '    bad = st[(r["reused"]) & (r["safe"] == False)].copy()\n'
         '    bad["matched_text"] = st["text"].to_numpy()[r.loc[bad.index, "match"].to_numpy()]\n'
         '    bad["similarity"] = r.loc[bad.index, "best_sim"].round(3)\n'
         '    bad["type"] = [vmeta.loc[v, "changed"] if k == "lookalike" else "different question"\n'
         '                   for k, v in zip(bad["kind"], bad["vid"])]\n'
         '    return bad\n\n'
         'err = pd.concat([failures(0, label).assign(rule=label) for label in RULES], ignore_index=True)\n'
         'err.to_csv(MET / "task1_unsafe_reuses_seed0.csv", index=False)\n'
         'display(pd.crosstab(err["rule"], err["type"]))\n'
         'err[["rule", "type", "similarity", "text", "matched_text"]].head(10)'),
    md("## 9. Next variations (Days 3–4)\n\n- **V4** fine-tuned MiniLM (paraphrases as positives, look-alikes "
       "and hard negatives as negatives; training split only)\n- **V5** entity guard: veto reuse when named "
       "entities, dates or numbers differ\n- **V6** gradient-boosting reuse classifier\n- **V3** vCache, or the "
       "backup per-entry threshold update\n\nEach new rule only needs a similarity matrix (or a reuse decision "
       "per event) for the same streams, so it slots into sections 6–8 unchanged."),
    md("## 10. Discussion and references\n\n*Write in your own words once all variations are run.*"),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, ROOT / "notebooks" / "NaveenaSivapuram_Cache_Reuse.ipynb")
print("wrote NaveenaSivapuram_Cache_Reuse.ipynb")
