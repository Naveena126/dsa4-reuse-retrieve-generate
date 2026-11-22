"""Task 1: simulate a semantic cache over a question stream.

Every query is compared with the entries already in the cache. If the rule
says "reuse", the cached answer is served; otherwise the query is answered
and stored (store-everything, the Day-1 admission baseline).
"""
import numpy as np
import pandas as pd

from .stream import answer_key
from .text import normalize

NEVER_REUSE = 1.01  # above any cosine similarity


def similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between all stream events (vectors already normalised)."""
    v = np.asarray(vecs, dtype=np.float32)
    return v @ v.T


def simulate(stream: pd.DataFrame, sim: np.ndarray, threshold: float) -> pd.DataFrame:
    """Replay the stream once. sim[i, j] = similarity of event i and event j.

    Returns one row per event: reused (bool), matched event, best similarity, safe (bool or NaN).
    """
    keys = [answer_key(q, k, v) for q, k, v in zip(stream["qid"], stream["kind"], stream["vid"])]
    stored, stored_keys = [], set()
    rows = []
    for i in range(len(stream)):
        if stored:
            s = sim[i, stored]
            j = int(np.argmax(s))
            best, match = float(s[j]), stored[j]
        else:
            best, match = -1.0, -1
        reused = best >= threshold
        rows.append({"reused": reused, "match": match if reused else -1, "best_sim": best,
                     "safe": (keys[i] == keys[match]) if reused else np.nan,
                     "could_reuse": keys[i] in stored_keys})
        if not reused:
            stored.append(i)
            stored_keys.add(keys[i])
    return pd.DataFrame(rows, index=stream.index)


def simulate_exact(stream: pd.DataFrame) -> pd.DataFrame:
    """Exact-match cache: reuse only if the normalised text was seen before (lower bound)."""
    keys = [answer_key(q, k, v) for q, k, v in zip(stream["qid"], stream["kind"], stream["vid"])]
    seen, seen_keys = {}, set()
    rows = []
    for i, t in enumerate(stream["text"]):
        n = normalize(t)
        match = seen.get(n, -1)
        reused = match >= 0
        rows.append({"reused": reused, "match": match, "best_sim": 1.0 if reused else 0.0,
                     "safe": (keys[i] == keys[match]) if reused else np.nan,
                     "could_reuse": keys[i] in seen_keys})
        if not reused:
            seen[n] = i
            seen_keys.add(keys[i])
    return pd.DataFrame(rows, index=stream.index)


def summarise(res: pd.DataFrame) -> dict:
    """calls_saved = reuses / events; unsafe_reuse = unsafe / reuses;
    safe_recall = safe reuses / events that had a safe entry available."""
    n = len(res)
    reused = res["reused"].sum()
    unsafe = int((res["reused"] & (res["safe"] == False)).sum())  # noqa: E712
    safe = int((res["reused"] & (res["safe"] == True)).sum())  # noqa: E712
    opp = int(res["could_reuse"].sum())
    return {"calls_saved": reused / n if n else 0.0,
            "unsafe_reuse": unsafe / reused if reused else 0.0,
            "safe_recall": safe / opp if opp else 0.0,
            "reuses": int(reused), "unsafe": unsafe, "events": n}


def sweep(stream: pd.DataFrame, sim: np.ndarray, thresholds) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        rows.append({"threshold": float(t), **summarise(simulate(stream, sim, t))})
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, max_unsafe: float = 0.02) -> float:
    """Most calls saved while unsafe reuse stays at or below the target (on validation)."""
    ok = curve[curve["unsafe_reuse"] <= max_unsafe]
    if ok.empty:
        return NEVER_REUSE  # no threshold meets the target: the rule should not reuse at all
    return float(ok.sort_values(["calls_saved", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
