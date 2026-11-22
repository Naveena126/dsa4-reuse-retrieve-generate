"""Answer normalisation shared by every task.

Manimegalai's evaluate.py builds the exact match / F1 tables on top of this.
"""
import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse spaces (SQuAD style)."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, aliases) -> bool:
    """True if the normalised prediction equals any normalised alias."""
    pred = normalize(prediction)
    return bool(pred) and any(pred == normalize(a) for a in aliases)


def contains_answer(text: str, aliases) -> bool:
    """True if any normalised alias appears inside the normalised text."""
    t = f" {normalize(text)} "
    return any(f" {normalize(a)} " in t for a in aliases if normalize(a))
