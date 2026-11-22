"""Passages, embeddings and the FAISS index used for RAG (owner: Naveena).

One index pools the passages of all 2,000 sampled questions:
HotpotQA paragraphs as they are, TriviaQA Wikipedia pages cut into ~100-word chunks.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .text import contains_answer

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
CHUNK_WORDS = 100
MAX_CHUNKS_PER_PAGE = 20


def chunk_text(text: str, words: int = CHUNK_WORDS, max_chunks: int = MAX_CHUNKS_PER_PAGE):
    toks = text.split()
    return [" ".join(toks[i:i + words]) for i in range(0, len(toks), words)][:max_chunks]


def build_passages(questions: pd.DataFrame) -> pd.DataFrame:
    """One row per passage: pid, source qid, title, text."""
    rows, seen = [], set()
    for q in questions.itertuples():
        for d in q.docs:
            pieces = [d["text"]] if q.dataset == "hotpotqa" else chunk_text(d["text"])
            for j, piece in enumerate(pieces):
                key = (d["title"], piece)
                if not piece.strip() or key in seen:  # same paragraph can appear under several questions
                    continue
                seen.add(key)
                rows.append({"pid": len(rows), "qid": q.qid, "title": d["title"], "chunk": j, "text": piece})
    return pd.DataFrame(rows)


def passage_text(p) -> str:
    return f"{p['title']}: {p['text']}"


def embed(texts, model=None, batch_size: int = 64, is_query: bool = False) -> np.ndarray:
    """Normalised bge embeddings (float32). Pass a loaded SentenceTransformer to reuse it."""
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    vecs = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=len(texts) > 1000, convert_to_numpy=True)
    return vecs.astype("float32")


def build_index(vectors: np.ndarray):
    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine on normalised vectors
    index.add(vectors)
    return index


def save_index(index, passages: pd.DataFrame, data_dir):
    import faiss
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(data_dir / "passages.faiss"))
    passages.to_json(data_dir / "passages.jsonl", orient="records", lines=True, force_ascii=False)


def load_index(data_dir):
    import faiss
    data_dir = Path(data_dir)
    index = faiss.read_index(str(data_dir / "passages.faiss"))
    passages = pd.read_json(data_dir / "passages.jsonl", lines=True)
    return index, passages


def retrieve(index, query_vecs: np.ndarray, k: int = 5):
    """Return (scores, pids) arrays of shape (n_queries, k)."""
    scores, pids = index.search(np.ascontiguousarray(query_vecs, dtype="float32"), k)
    return scores, pids


def recall_at_k(questions: pd.DataFrame, passages: pd.DataFrame, pids: np.ndarray) -> pd.Series:
    """Per question: HotpotQA = both gold titles retrieved; TriviaQA = an answer alias appears in a passage."""
    hits = []
    for q, row_pids in zip(questions.itertuples(), pids):
        got = passages.iloc[[p for p in row_pids if p >= 0]]
        if q.dataset == "hotpotqa":
            hits.append(set(q.gold_titles) <= set(got["title"]))
        else:
            hits.append(any(contains_answer(t, q.aliases) for t in got["text"]))
    return pd.Series(hits, index=questions.index, name="recall_at_k")
