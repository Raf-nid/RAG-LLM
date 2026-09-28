"""
Retrieval evaluation metrics.

This module implements standard information retrieval metrics:
- Recall@K: What fraction of relevant documents are in the top K?
- Precision@K: What fraction of the top K are relevant?
- MRR: Mean Reciprocal Rank - average of 1/rank for first relevant doc

These metrics assume a single "gold" document per query (common in RAG).
For multiple gold documents, recall makes more sense than precision.
"""

from __future__ import annotations

from dataclasses import dataclass


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """
    Calculate Recall@K.

    Recall@K = |retrieved[:k] ∩ relevant| / |relevant|

    Measures what fraction of relevant documents appear in the top K results.

    Parameters
    ----------
    retrieved_ids
        IDs of retrieved chunks, ordered by score descending.
    relevant_ids
        IDs of chunks marked as relevant (gold contexts).
    k
        Number of top results to consider.

    Returns
    -------
    float
        Recall score between 0 and 1.

    Examples
    --------
    >>> recall_at_k(["a", "b", "c", "d", "e"], ["c"], k=5)
    1.0
    >>> recall_at_k(["a", "b", "c", "d", "e"], ["c"], k=2)
    0.0
    """
    if not relevant_ids:
        return 1.0  # No relevant docs = trivially complete

    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """
    Calculate Precision@K.

    Precision@K = |retrieved[:k] ∩ relevant| / k

    Measures what fraction of the top K results are relevant.

    Parameters
    ----------
    retrieved_ids
        IDs of retrieved chunks, ordered by score descending.
    relevant_ids
        IDs of chunks marked as relevant.
    k
        Number of top results to consider.

    Returns
    -------
    float
        Precision score between 0 and 1.

    Examples
    --------
    >>> precision_at_k(["a", "b", "c"], ["a", "c"], k=3)
    0.6666666666666666
    """
    if k == 0:
        return 0.0

    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return len(top_k & relevant) / k


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    RR = 1 / rank_of_first_relevant

    Returns 0 if no relevant document is retrieved.

    Parameters
    ----------
    retrieved_ids
        IDs of retrieved chunks, ordered by score descending.
    relevant_ids
        IDs of chunks marked as relevant.

    Returns
    -------
    float
        Reciprocal rank between 0 and 1.

    Examples
    --------
    >>> reciprocal_rank(["a", "b", "c"], ["b"])
    0.5
    >>> reciprocal_rank(["a", "b", "c"], ["x"])
    0.0
    """
    relevant = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    results: list[tuple[list[str], list[str]]],
) -> float:
    """
    Calculate Mean Reciprocal Rank over multiple queries.

    MRR = (1/|Q|) * Σ (1 / rank_i)

    Parameters
    ----------
    results
        List of (retrieved_ids, relevant_ids) tuples.

    Returns
    -------
    float
        MRR between 0 and 1.
    """
    if not results:
        return 0.0

    return sum(reciprocal_rank(r, rel) for r, rel in results) / len(results)


@dataclass
class EvalResult:
    """
    Evaluation results for a retrieval run.

    Attributes
    ----------
    recall_at_1
        Recall@1 (hit rate).
    recall_at_5
        Recall@5.
    recall_at_10
        Recall@10.
    mrr
        Mean Reciprocal Rank.
    precision_at_5
        Precision@5.
    num_queries
        Number of queries evaluated.
    """

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision_at_5: float
    num_queries: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary for serialization."""
        return {
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
            "precision@5": self.precision_at_5,
            "num_queries": self.num_queries,
        }

    def __str__(self) -> str:
        return (
            f"EvalResult(Recall@1={self.recall_at_1:.3f}, "
            f"Recall@5={self.recall_at_5:.3f}, "
            f"Recall@10={self.recall_at_10:.3f}, "
            f"MRR={self.mrr:.3f}, "
            f"P@5={self.precision_at_5:.3f}, "
            f"n={self.num_queries})"
        )


def evaluate_retrieval(
    results: list[tuple[list[str], list[str]]],
) -> EvalResult:
    """
    Compute all retrieval metrics.

    Parameters
    ----------
    results
        List of (retrieved_ids, relevant_ids) tuples.
        Each tuple represents one query's results.

    Returns
    -------
    EvalResult
        Comprehensive evaluation metrics.
    """
    if not results:
        return EvalResult(
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            precision_at_5=0.0,
            num_queries=0,
        )

    # Compute individual metrics
    r1 = sum(recall_at_k(r, rel, k=1) for r, rel in results) / len(results)
    r5 = sum(recall_at_k(r, rel, k=5) for r, rel in results) / len(results)
    r10 = sum(recall_at_k(r, rel, k=10) for r, rel in results) / len(results)
    mrr = mean_reciprocal_rank(results)
    p5 = sum(precision_at_k(r, rel, k=5) for r, rel in results) / len(results)

    return EvalResult(
        recall_at_1=r1,
        recall_at_5=r5,
        recall_at_10=r10,
        mrr=mrr,
        precision_at_5=p5,
        num_queries=len(results),
    )
