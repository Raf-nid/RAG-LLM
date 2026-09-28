#!/usr/bin/env python3
"""
Evaluate retrieval quality on T²-RAGBench.

This script measures Recall@K, MRR, and Precision@K for:
- Dense retrieval (embeddings)
- BM25 retrieval (lexical)
- Hybrid retrieval (dense + BM25 + RRF)

Usage Examples
--------------
# Quick evaluation with 50 samples
uv run python scripts/evaluate_retrieval.py --max-eval 50

# Full evaluation on FinQA dev set
uv run python scripts/evaluate_retrieval.py --subset FinQA --max-samples 1000

# Compare all methods with verbose output
uv run python scripts/evaluate_retrieval.py --max-eval 100 --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


@dataclass
class MethodResult:
    """Results for one retrieval method."""

    name: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision_at_5: float
    latency_ms: float
    num_queries: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality on T²-RAGBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="FinQA",
        choices=["FinQA", "ConvFinQA", "TAT-DQA"],
        help="Which subset to evaluate (default: FinQA)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max documents to index (default: 500)",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=100,
        help="Max queries to evaluate (default: 100)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-query results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/hf_cache",
        help="HuggingFace cache directory",
    )

    args = parser.parse_args()

    from rag_assistant.config import Settings
    from rag_assistant.rag.embeddings.model import create_embedding_model
    from rag_assistant.rag.evaluation import evaluate_retrieval
    from rag_assistant.rag.ingestion import (
        create_eval_set,
        load_ragbench_subset,
        ragbench_to_chunks,
    )
    from rag_assistant.rag.retrieval import (
        BM25Retriever,
        DenseRetriever,
        HybridRetriever,
    )
    from rag_assistant.rag.store import QdrantStore

    logger.info("=" * 70)
    logger.info("Retrieval Evaluation on T²-RAGBench")
    logger.info("=" * 70)
    logger.info("Subset: %s", args.subset)
    logger.info("Max documents: %d", args.max_samples)
    logger.info("Max eval queries: %d", args.max_eval)
    logger.info("Top-K: %d", args.top_k)

    # Load data
    t0 = time.perf_counter()
    logger.info("\n[1/5] Loading T²-RAGBench dataset...")

    data = load_ragbench_subset(
        subset=args.subset,  # type: ignore
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
    )

    logger.info(
        "Loaded %d QA pairs, %d contexts in %.1fs",
        data.num_qa_pairs,
        data.num_contexts,
        time.perf_counter() - t0,
    )

    # Convert to chunks
    t0 = time.perf_counter()
    logger.info("\n[2/5] Converting to DocumentChunks...")
    chunks = ragbench_to_chunks(data)
    logger.info("Created %d chunks in %.1fs", len(chunks), time.perf_counter() - t0)

    # Create eval set
    logger.info("\n[3/5] Creating evaluation set...")
    eval_examples = create_eval_set(data)
    eval_examples = eval_examples[: args.max_eval]
    logger.info("Using %d evaluation queries", len(eval_examples))

    # Setup retrievers
    t0 = time.perf_counter()
    logger.info("\n[4/5] Setting up retrievers...")

    settings = Settings()
    embedding_model = create_embedding_model(settings)
    logger.info("Embedding model: %s (%d dim)", settings.embedding_model, embedding_model.dimensions)

    # Qdrant store (in-memory)
    from qdrant_client import QdrantClient

    qdrant_client = QdrantClient(":memory:")
    store = QdrantStore(
        client=qdrant_client,
        collection_name="ragbench_eval",
        embedding_dimensions=embedding_model.dimensions,
    )
    store.ensure_collection()

    # Index all chunks
    logger.info("Computing embeddings...")
    texts = [c.text for c in chunks]
    embeddings = embedding_model.embed_documents(texts)

    logger.info("Indexing into Qdrant...")
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        store.upsert_chunk(chunk, embedding)

    # BM25 index
    logger.info("Building BM25 index...")
    bm25 = BM25Retriever(default_top_k=args.top_k)
    bm25.index_chunks(chunks)

    # Dense retriever
    dense = DenseRetriever(
        embedding_model=embedding_model,
        store=store,
        default_top_k=args.top_k,
    )

    # Hybrid retriever
    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        default_top_k=args.top_k,
    )

    logger.info("Setup complete in %.1fs", time.perf_counter() - t0)

    # Run evaluations
    logger.info("\n[5/5] Running evaluations...")

    methods = [
        ("Dense", dense),
        ("BM25", bm25),
        ("Hybrid", hybrid),
    ]

    results: list[MethodResult] = []

    for method_name, retriever in methods:
        logger.info("\nEvaluating %s...", method_name)

        eval_results: list[tuple[list[str], list[str]]] = []
        total_time = 0.0

        for i, ex in enumerate(eval_examples):
            t_start = time.perf_counter()
            retrieved = retriever.retrieve(ex.question, top_k=args.top_k)
            total_time += time.perf_counter() - t_start

            retrieved_ids = [r.chunk_metadata.chunk_id for r in retrieved]
            eval_results.append((retrieved_ids, ex.gold_context_ids))

            if args.verbose and i < 5:
                hit = any(rid in ex.gold_context_ids for rid in retrieved_ids[:5])
                logger.info(
                    "  Q%d: %s... | Hit@5: %s",
                    i + 1,
                    ex.question[:50],
                    "YES" if hit else "NO",
                )

        metrics = evaluate_retrieval(eval_results)
        avg_latency = (total_time / len(eval_examples)) * 1000

        results.append(
            MethodResult(
                name=method_name,
                recall_at_1=metrics.recall_at_1,
                recall_at_5=metrics.recall_at_5,
                recall_at_10=metrics.recall_at_10,
                mrr=metrics.mrr,
                precision_at_5=metrics.precision_at_5,
                latency_ms=avg_latency,
                num_queries=metrics.num_queries,
            )
        )

        logger.info(
            "  Recall@1=%.3f, Recall@5=%.3f, Recall@10=%.3f, MRR=%.3f, Latency=%.1fms",
            metrics.recall_at_1,
            metrics.recall_at_5,
            metrics.recall_at_10,
            metrics.mrr,
            avg_latency,
        )

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "%-10s | %8s | %8s | %9s | %6s | %10s",
        "Method",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR",
        "Latency",
    )
    logger.info("-" * 70)

    for r in results:
        logger.info(
            "%-10s | %8.3f | %8.3f | %9.3f | %6.3f | %8.1f ms",
            r.name,
            r.recall_at_1,
            r.recall_at_5,
            r.recall_at_10,
            r.mrr,
            r.latency_ms,
        )

    logger.info("-" * 70)
    logger.info("Evaluated on %d queries with %d documents indexed", args.max_eval, len(chunks))

    # Best method
    best = max(results, key=lambda r: r.recall_at_5)
    logger.info("\nBest by Recall@5: %s (%.3f)", best.name, best.recall_at_5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
