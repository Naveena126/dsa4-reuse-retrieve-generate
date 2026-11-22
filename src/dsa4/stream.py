"""Question stream builder.

A stream is the order in which questions reach the system. Each source
question is asked first; its two paraphrases and its look-alike arrive later;
exact repeats are added so that a chosen share of the stream is repeats.

Reuse rule used by every task: reusing the answer of cached entry E for
query Q is SAFE only if E and Q have the same correct answer, i.e. the same
answer_key (same source question, and neither or the very same look-alike).
Everything else is UNSAFE.
"""
import numpy as np
import pandas as pd

KINDS = ("source", "paraphrase", "lookalike", "repeat")


def variants_table(cache: pd.DataFrame) -> pd.DataFrame:
    """Long table of variants from the merged answer cache: qid, kind, text."""
    rows = []
    for r in cache.itertuples():
        v = r.variants or {}
        for j, p in enumerate((v.get("paraphrases") or [])[:2]):
            rows.append({"qid": r.qid, "kind": "paraphrase", "vid": f"{r.qid}#p{j}", "text": p})
        if v.get("lookalike"):
            rows.append({"qid": r.qid, "kind": "lookalike", "vid": f"{r.qid}#l",
                         "text": v["lookalike"], "changed": v.get("changed", "")})
    return pd.DataFrame(rows)


def build_stream(sources: pd.DataFrame, variants: pd.DataFrame, repeat_rate: float = 0.0,
                 seed: int = 0) -> pd.DataFrame:
    """Return the stream as a DataFrame with columns pos, qid, kind, vid, text.

    sources:  one row per source question (qid, question) in ONE split.
    variants: rows from variants_table() for those qids.
    repeat_rate: share of the final stream that are exact repeats (0, 0.2, 0.5).
    """
    rng = np.random.default_rng(seed)
    order = sources["qid"].to_numpy().copy()
    rng.shuffle(order)
    text_of = dict(zip(sources["qid"], sources["question"]))

    events = [{"qid": q, "kind": "source", "vid": q, "text": text_of[q]} for q in order]
    first_pos = {q: i for i, q in enumerate(order)}
    # place each variant at a random position after its source
    var = variants[variants["qid"].isin(first_pos)]
    keyed = []
    for r in var.itertuples():
        lo = first_pos[r.qid] + 1
        keyed.append((rng.uniform(lo, len(order) + 1), {"qid": r.qid, "kind": r.kind, "vid": r.vid, "text": r.text}))
    merged = [(float(i), e) for i, e in enumerate(events)] + keyed
    merged.sort(key=lambda t: t[0])
    stream = [e for _, e in merged]

    if repeat_rate > 0:
        n_rep = int(round(repeat_rate * len(stream) / (1 - repeat_rate)))
        for _ in range(n_rep):
            pos = int(rng.integers(1, len(stream) + 1))
            earlier = [e for e in stream[:pos] if e["kind"] == "source"]
            if not earlier:
                continue
            e = earlier[int(rng.integers(len(earlier)))]
            stream.insert(pos, {"qid": e["qid"], "kind": "repeat", "vid": e["vid"], "text": e["text"]})

    out = pd.DataFrame(stream)
    out.insert(0, "pos", range(len(out)))
    return out


def answer_key(qid: str, kind: str, vid: str) -> str:
    """Questions with the same key share the same correct answer.

    A source, its paraphrases and its repeats share the source's answer;
    a look-alike has its own (different) answer.
    """
    return vid if kind == "lookalike" else qid


def reuse_is_safe(query_key: str, cached_key: str) -> bool:
    return query_key == cached_key
