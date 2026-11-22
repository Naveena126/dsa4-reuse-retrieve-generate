"""Tests on small synthetic data (no downloads, no GPU). Run: python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsa4 import data, index, reuse, stream  # noqa: E402
from dsa4.generation import parse_variants, token_scores  # noqa: E402
from dsa4.text import contains_answer, exact_match, normalize  # noqa: E402


def toy_questions(n=40):
    rows = []
    for i in range(n):
        ds = "triviaqa" if i % 2 == 0 else "hotpotqa"
        rows.append({"qid": f"{ds[:3]}_{i}", "dataset": ds, "question": f"Who wrote book {i}?",
                     "answer": f"Author {i}", "aliases": [f"Author {i}"],
                     "docs": [{"title": f"Book {i}", "text": f"Book {i} was written by Author {i}. " * 30}],
                     "gold_titles": [f"Book {i}"] if ds == "hotpotqa" else []})
    return pd.DataFrame(rows)


def test_normalize_and_match():
    assert normalize("The  Beatles!") == "beatles"
    assert exact_match("the beatles", ["Beatles"])
    assert not exact_match("", ["Beatles"])
    assert contains_answer("It was the Beatles in 1963.", ["beatles"])


def test_splits_by_source_and_ratio():
    q = data.assign_splits(toy_questions(100))
    for s in data.SPLIT_SEEDS:
        counts = q[f"split_s{s}"].value_counts()
        assert counts["train"] == 60 and counts["val"] == 20 and counts["test"] == 20
    assert (q["split_s0"] != q["split_s1"]).any()  # seeds give different splits


def test_blocks_balanced():
    q = data.assign_blocks(toy_questions(80))
    tab = q.groupby(["dataset", "block"]).size()
    assert set(tab.values) == {10}


def test_passages_and_retrieval():
    q = toy_questions(6)
    p = index.build_passages(q)
    assert (p.groupby("qid").size() >= 1).all()
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(p), 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = index.build_index(vecs)
    _, pids = index.retrieve(idx, vecs[:3], k=2)
    assert list(pids[:, 0]) == [0, 1, 2]  # each vector is its own nearest neighbour
    rec = index.recall_at_k(q.iloc[:3], p, pids)
    assert rec.dtype == bool


def test_chunking():
    chunks = index.chunk_text("w " * 250, words=100)
    assert len(chunks) == 3


def toy_stream(repeat_rate=0.0):
    src = toy_questions(10)
    var_rows = []
    for q in src["qid"]:
        var_rows += [{"qid": q, "kind": "paraphrase", "vid": f"{q}#p0", "text": "p0 " + q},
                     {"qid": q, "kind": "paraphrase", "vid": f"{q}#p1", "text": "p1 " + q},
                     {"qid": q, "kind": "lookalike", "vid": f"{q}#l", "text": "l " + q, "changed": "entity"}]
    return stream.build_stream(src, pd.DataFrame(var_rows), repeat_rate=repeat_rate, seed=1)


def test_stream_order_and_repeats():
    s = toy_stream()
    assert len(s) == 40
    first = s[s["kind"] == "source"].set_index("qid")["pos"]
    for r in s[s["kind"] != "source"].itertuples():
        assert r.pos > first[r.qid]  # every variant comes after its source
    s2 = toy_stream(repeat_rate=0.2)
    share = (s2["kind"] == "repeat").mean()
    assert 0.15 <= share <= 0.25


def test_safety_keys():
    assert stream.reuse_is_safe(stream.answer_key("a", "paraphrase", "a#p0"), stream.answer_key("a", "source", "a"))
    assert not stream.reuse_is_safe(stream.answer_key("a", "lookalike", "a#l"), stream.answer_key("a", "source", "a"))
    assert not stream.reuse_is_safe(stream.answer_key("b", "source", "b"), stream.answer_key("a", "source", "a"))


def test_simulation_extremes():
    s = toy_stream()
    n = len(s)
    sim = np.eye(n, dtype=np.float32)
    none = reuse.summarise(reuse.simulate(s, sim, threshold=1.01))
    assert none["reuses"] == 0
    allr = reuse.summarise(reuse.simulate(s, np.ones((n, n), np.float32), threshold=0.5))
    assert allr["reuses"] == n - 1  # everything after the first event is reused
    assert allr["unsafe_reuse"] > 0.5


def test_simulation_perfect_embedder():
    s = toy_stream(0.2)
    keys = [stream.answer_key(q, k, v) for q, k, v in zip(s["qid"], s["kind"], s["vid"])]
    sim = np.array([[1.0 if a == b else 0.0 for b in keys] for a in keys], np.float32)
    res = reuse.summarise(reuse.simulate(s, sim, 0.9))
    assert res["unsafe_reuse"] == 0 and res["safe_recall"] == 1.0


def test_exact_match_cache():
    s = toy_stream(0.2)
    res = reuse.summarise(reuse.simulate_exact(s))
    assert res["unsafe_reuse"] == 0 and res["reuses"] == (s["kind"] == "repeat").sum()


def test_pick_threshold():
    curve = pd.DataFrame({"threshold": [0.8, 0.9, 0.95], "calls_saved": [0.5, 0.3, 0.1],
                          "unsafe_reuse": [0.10, 0.02, 0.0]})
    assert reuse.pick_threshold(curve, 0.02) == 0.9
    assert reuse.pick_threshold(curve, 0.0) == 0.95
    assert reuse.pick_threshold(curve.iloc[:1], 0.02) == reuse.NEVER_REUSE


def test_token_scores():
    logits = np.log(np.array([[0.5, 0.25, 0.25], [0.9, 0.05, 0.05]]))
    lp, ent = token_scores(logits, np.array([0, 0]))
    assert lp == pytest.approx((np.log(0.5) + np.log(0.9)) / 2)
    h1 = -(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25))
    h2 = -(0.9 * np.log(0.9) + 2 * 0.05 * np.log(0.05))
    assert ent == pytest.approx((h1 + h2) / 2)


def test_parse_variants():
    ok = parse_variants('Sure: {"paraphrases": ["a?", "b?"], "lookalike": "c?", "changed": "date"}')
    assert ok == {"paraphrases": ["a?", "b?"], "lookalike": "c?", "changed": "date"}
    assert parse_variants("no json here") is None
    assert parse_variants('{"paraphrases": ["a?"], "lookalike": "c?"}') is None
