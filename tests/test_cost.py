import pytest

from src.dsa4.cost import (
    CostWeights,
    LatencyLog,
    call_cost,
    total_cost,
    cheapest_correct_action,
)


@pytest.fixture
def weights():
    return CostWeights(
        lookup=0,
        retrieval=1,
        call_3b=3,
        call_7b=10,
    )


def test_call_cost(weights):
    assert call_cost("lookup", weights) == 0
    assert call_cost("retrieval", weights) == 1
    assert call_cost("call_3b", weights) == 3
    assert call_cost("call_7b", weights) == 10


def test_total_cost(weights):
    actions = ["lookup", "retrieval", "call_3b"]
    assert total_cost(actions, weights) == 4


def test_cheapest_correct_action(weights):
    candidates = [
        {"action": "direct", "correct": True},
        {"action": "rag", "correct": True},
    ]

    best = cheapest_correct_action(candidates, weights)
    assert best["action"] == "direct"
    assert best["oracle_cost"] == 10


def test_rag_oracle_cost_includes_retrieval(weights):
    candidates = [{"action": "rag", "correct": True}]

    best = cheapest_correct_action(candidates, weights)
    assert best["action"] == "rag"
    assert best["oracle_cost"] == 11


def test_oracle_selects_rag_when_direct_is_incorrect(weights):
    candidates = [
        {"action": "direct", "correct": False},
        {"action": "rag", "correct": True},
    ]

    best = cheapest_correct_action(candidates, weights)
    assert best["action"] == "rag"
    assert best["oracle_cost"] == 11


def test_no_correct_candidate(weights):
    candidates = [
        {"action": "call_3b", "correct": False},
        {"action": "call_7b", "correct": False},
    ]

    assert cheapest_correct_action(candidates, weights) is None


def test_oracle_question_filter(weights):
    candidates = [
        {"question_id": 1, "action": "direct", "correct": True},
        {"question_id": 2, "action": "rag", "correct": True},
    ]

    best = cheapest_correct_action(
        candidates,
        weights,
        question_id=1,
    )

    assert best["action"] == "direct"


def test_latency_log():
    log = LatencyLog()
    log.record("retrieval", 100)
    log.record("retrieval", 200)

    summary = log.summary("retrieval")

    assert summary["count"] == 2
    assert summary["mean_ms"] == 150
    assert summary["median_ms"] == 150
    assert summary["min_ms"] == 100
    assert summary["max_ms"] == 200


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        CostWeights(
            lookup=-1,
            retrieval=1,
            call_3b=3,
            call_7b=10,
        )


def test_negative_latency_rejected():
    log = LatencyLog()

    with pytest.raises(ValueError):
        log.record("retrieval", -5)