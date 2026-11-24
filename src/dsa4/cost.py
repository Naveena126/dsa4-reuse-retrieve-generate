"""Cost model and latency tracking for DSA4 cache admission."""

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any


@dataclass
class CostWeights:
    """Relative costs of each action, supplied by the experiment configuration."""

    lookup: float
    retrieval: float
    call_3b: float
    call_7b: float

    def as_dict(self) -> dict[str, float]:
        return {
            "lookup": self.lookup,
            "retrieval": self.retrieval,
            "call_3b": self.call_3b,
            "call_7b": self.call_7b,
        }

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if value < 0:
                raise ValueError(f"{name} cost cannot be negative")


@dataclass
class LatencyLog:
    """Store latency measurements in milliseconds by action type."""

    samples: dict[str, list[float]] = field(
        default_factory=lambda: {
            "lookup": [],
            "retrieval": [],
            "call_3b": [],
            "call_7b": [],
        }
    )

    def record(self, action: str, latency_ms: float) -> None:
        if action not in self.samples:
            raise ValueError(f"Unknown action type: {action}")
        if latency_ms < 0:
            raise ValueError("Latency cannot be negative")

        self.samples[action].append(float(latency_ms))

    def summary(self, action: str) -> dict[str, float | int]:
        values = self.samples.get(action, [])
        if not values:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }

        return {
            "count": len(values),
            "mean_ms": mean(values),
            "median_ms": median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }


def call_cost(action: str, weights: CostWeights) -> float:
    """Return the cost of a single action."""
    costs = weights.as_dict()
    if action not in costs:
        raise ValueError(f"Unknown action type: {action}")
    return costs[action]


def total_cost(actions: list[str], weights: CostWeights) -> float:
    """Calculate the total cost of a sequence of actions."""
    return sum(call_cost(action, weights) for action in actions)


def cheapest_correct_action(
    candidates: list[dict[str, Any]],
    weights: CostWeights,
    question_id: Any = None,
) -> dict[str, Any] | None:
    """Oracle: select the cheapest correct direct or RAG answer."""

    action_costs = {
        "direct": total_cost(["call_7b"], weights),
        "rag": total_cost(["retrieval", "call_7b"], weights),
    }

    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("correct", False)
        and (
            question_id is None
            or candidate.get("question_id") == question_id
        )
    ]

    if not eligible:
        return None

    def candidate_cost(candidate: dict[str, Any]) -> float:
        action = candidate["action"]
        if action not in action_costs:
            raise ValueError(f"Unknown answer action: {action}")
        return action_costs[action]

    best = min(eligible, key=candidate_cost)

    result = dict(best)
    result["oracle_cost"] = candidate_cost(best)
    return result
