"""Tests for evaluation metrics."""

import pytest

from rag_assistant.rag.evaluation.metrics import (
    EvalResult,
    evaluate_retrieval,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_perfect_recall(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b"]

        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_zero_recall(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = ["x", "y"]

        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_recall(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "x"]  # Only "a" is retrieved

        assert recall_at_k(retrieved, relevant, k=5) == 0.5

    def test_respects_k_limit(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["e"]  # Only in position 5

        assert recall_at_k(retrieved, relevant, k=3) == 0.0  # Not in top 3
        assert recall_at_k(retrieved, relevant, k=5) == 1.0  # In top 5

    def test_empty_relevant(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant: list[str] = []

        # No relevant docs = trivially complete
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_empty_retrieved(self) -> None:
        retrieved: list[str] = []
        relevant = ["a"]

        assert recall_at_k(retrieved, relevant, k=5) == 0.0


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_perfect_precision(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c", "d", "e"]

        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_precision(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = ["x", "y", "z"]

        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_precision(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "c"]  # 2 of top 5

        assert precision_at_k(retrieved, relevant, k=5) == 0.4

    def test_k_zero(self) -> None:
        retrieved = ["a", "b"]
        relevant = ["a"]

        assert precision_at_k(retrieved, relevant, k=0) == 0.0


class TestReciprocalRank:
    """Tests for reciprocal_rank function."""

    def test_first_position(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = ["a"]

        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_position(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = ["b"]

        assert reciprocal_rank(retrieved, relevant) == 0.5

    def test_third_position(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = ["c"]

        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_not_found(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = ["x"]

        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_multiple_relevant(self) -> None:
        # Only first relevant matters for RR
        retrieved = ["a", "b", "c"]
        relevant = ["b", "c"]  # "b" is first relevant at rank 2

        assert reciprocal_rank(retrieved, relevant) == 0.5

    def test_empty_lists(self) -> None:
        assert reciprocal_rank([], ["a"]) == 0.0
        assert reciprocal_rank(["a"], []) == 0.0


class TestMeanReciprocalRank:
    """Tests for mean_reciprocal_rank function."""

    def test_basic_mrr(self) -> None:
        results = [
            (["a", "b", "c"], ["a"]),  # RR = 1.0
            (["a", "b", "c"], ["b"]),  # RR = 0.5
            (["a", "b", "c"], ["c"]),  # RR = 0.333
        ]

        mrr = mean_reciprocal_rank(results)
        expected = (1.0 + 0.5 + 1 / 3) / 3
        assert mrr == pytest.approx(expected)

    def test_empty_results(self) -> None:
        assert mean_reciprocal_rank([]) == 0.0

    def test_all_miss(self) -> None:
        results = [
            (["a", "b"], ["x"]),
            (["c", "d"], ["y"]),
        ]
        assert mean_reciprocal_rank(results) == 0.0

    def test_all_hit_first(self) -> None:
        results = [
            (["a", "b"], ["a"]),
            (["c", "d"], ["c"]),
        ]
        assert mean_reciprocal_rank(results) == 1.0


class TestEvaluateRetrieval:
    """Tests for evaluate_retrieval function."""

    def test_returns_eval_result(self) -> None:
        results = [
            (["a", "b", "c", "d", "e"], ["a"]),
            (["a", "b", "c", "d", "e"], ["b"]),
        ]

        eval_result = evaluate_retrieval(results)

        assert isinstance(eval_result, EvalResult)
        assert eval_result.num_queries == 2

    def test_perfect_retrieval(self) -> None:
        # All queries have relevant doc at position 1
        results = [
            (["gold_1", "x", "y"], ["gold_1"]),
            (["gold_2", "x", "y"], ["gold_2"]),
            (["gold_3", "x", "y"], ["gold_3"]),
        ]

        eval_result = evaluate_retrieval(results)

        assert eval_result.recall_at_1 == 1.0
        assert eval_result.recall_at_5 == 1.0
        assert eval_result.mrr == 1.0

    def test_empty_results(self) -> None:
        eval_result = evaluate_retrieval([])

        assert eval_result.num_queries == 0
        assert eval_result.recall_at_1 == 0.0
        assert eval_result.mrr == 0.0

    def test_to_dict(self) -> None:
        results = [(["a"], ["a"])]
        eval_result = evaluate_retrieval(results)

        d = eval_result.to_dict()
        assert "recall@1" in d
        assert "recall@5" in d
        assert "mrr" in d
        assert d["num_queries"] == 1

    def test_str_representation(self) -> None:
        results = [(["a"], ["a"])]
        eval_result = evaluate_retrieval(results)

        s = str(eval_result)
        assert "Recall@1" in s
        assert "MRR" in s


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_creation(self) -> None:
        result = EvalResult(
            recall_at_1=0.8,
            recall_at_5=0.9,
            recall_at_10=0.95,
            mrr=0.85,
            precision_at_5=0.2,
            num_queries=100,
        )

        assert result.recall_at_1 == 0.8
        assert result.num_queries == 100

    def test_to_dict_keys(self) -> None:
        result = EvalResult(
            recall_at_1=0.8,
            recall_at_5=0.9,
            recall_at_10=0.95,
            mrr=0.85,
            precision_at_5=0.2,
            num_queries=100,
        )

        d = result.to_dict()
        assert set(d.keys()) == {
            "recall@1",
            "recall@5",
            "recall@10",
            "mrr",
            "precision@5",
            "num_queries",
        }
