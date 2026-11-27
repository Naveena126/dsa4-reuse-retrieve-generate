"""DeBERTa-v3-base NLI support checker shared by Task 3 and Task 4."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def build_hypothesis(question: str, answer: str) -> str:
    """Format a (question, answer) pair as an NLI hypothesis."""
    return f"Question: {question.strip()} Answer: {answer.strip()}"


def aggregate_scores(scores: Sequence[float], method: str = "max") -> float:
    """Aggregate per-passage entailment probabilities into a single support score."""
    if not scores:
        return 0.0
    if method == "max":
        return float(max(scores))
    if method == "mean":
        return float(sum(scores) / len(scores))
    raise ValueError(f"Unsupported aggregation method: {method!r}. Use 'max' or 'mean'.")


class NLISupportChecker:
    """Checks whether retrieved passages logically support a candidate answer."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_NLI_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.entailment_idx = self._resolve_entailment_idx()

    def _resolve_entailment_idx(self) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            name = str(label).lower()
            if "entail" in name or "support" in name:
                return int(idx)
        # Fallback: binary classifier uses class 1 as positive; 3-class MNLI uses 0
        num_labels = getattr(self.model.config, "num_labels", 3)
        return 1 if num_labels == 2 else 0

    @torch.inference_mode()
    def _score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Return entailment probabilities for (premise, hypothesis) pairs."""
        if not pairs:
            return []

        probs_out: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            encoded = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoded).logits
            probs = torch.softmax(logits.float(), dim=-1)
            probs_out.extend(probs[:, self.entailment_idx].cpu().tolist())

        return probs_out

    def score(
        self,
        question: str,
        answer: str,
        passages: Sequence[str],
        aggregation: str = "max",
    ) -> Dict[str, object]:
        """Score how strongly `passages` support `answer` for `question`."""
        return self.score_batch([(question, answer, passages)], aggregation=aggregation)[0]

    def is_supported(
        self,
        question: str,
        answer: str,
        passages: Sequence[str],
        threshold: float = 0.5,
        aggregation: str = "max",
    ) -> bool:
        """Return True if the aggregated support score meets or exceeds `threshold`."""
        result = self.score(question, answer, passages, aggregation=aggregation)
        return float(result["support_score"]) >= threshold

    def score_batch(
        self,
        items: Sequence[Tuple[str, str, Sequence[str]]],
        aggregation: str = "max",
    ) -> List[Dict[str, object]]:
        """Batched scoring over (question, answer, passages) tuples for fast GPU runs."""
        flat_pairs: List[Tuple[str, str]] = []
        spans: List[Tuple[int, int]] = []

        for question, answer, passages in items:
            clean_passages = [p.strip() for p in passages if p and p.strip()]
            if not answer or not answer.strip() or not clean_passages:
                spans.append((len(flat_pairs), len(flat_pairs)))
                continue

            hypothesis = build_hypothesis(question, answer)
            start = len(flat_pairs)
            for passage in clean_passages:
                flat_pairs.append((passage, hypothesis))
            spans.append((start, len(flat_pairs)))

        flat_scores = self._score_pairs(flat_pairs)

        results: List[Dict[str, object]] = []
        for start, end in spans:
            p_scores = flat_scores[start:end]
            if not p_scores:
                results.append(
                    {
                        "support_score": 0.0,
                        "passage_scores": [],
                        "best_passage_idx": -1,
                    }
                )
            else:
                best_idx = max(range(len(p_scores)), key=lambda i: p_scores[i])
                results.append(
                    {
                        "support_score": aggregate_scores(p_scores, method=aggregation),
                        "passage_scores": p_scores,
                        "best_passage_idx": best_idx,
                    }
                )
        return results