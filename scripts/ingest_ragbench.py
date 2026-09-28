#!/usr/bin/env python3
"""
Ingest T²-RAGBench dataset into the RAG pipeline.

This script:
1. Downloads T²-RAGBench from Hugging Face Hub
2. Converts contexts to DocumentChunks with rich metadata
3. Indexes into Qdrant (dense vectors) and BM25 (lexical)
4. Exports evaluation set (questions + gold contexts) to JSONL

Usage Examples
--------------
# Quick test with 100 samples from FinQA only
uv run python scripts/ingest_ragbench.py --subset FinQA --max-samples 100

# Full FinQA dataset
uv run python scripts/ingest_ragbench.py --subset FinQA

# All subsets (full dataset, ~7000 documents)
uv run python scripts/ingest_ragbench.py --subset all

# With custom paths
uv run python scripts/ingest_ragbench.py \\
    --subset FinQA \\
    --max-samples 500 \\
    --eval-output data/eval/finqa_500.jsonl \\
    --cache-dir data/hf_cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress verbose HF logs
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest T²-RAGBench dataset into the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 100 samples
  uv run python scripts/ingest_ragbench.py --subset FinQA --max-samples 100

  # Full FinQA subset
  uv run python scripts/ingest_ragbench.py --subset FinQA

  # All subsets
  uv run python scripts/ingest_ragbench.py --subset all
        """,
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="FinQA",
        choices=["FinQA", "ConvFinQA", "TAT-DQA", "all"],
        help="Which subset(s) to load (default: FinQA)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of QA pairs per subset (for testing)",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default="data/eval/ragbench_eval.jsonl",
        help="Path to save evaluation set (default: data/eval/ragbench_eval.jsonl)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/hf_cache",
        help="Directory for HuggingFace cache (default: data/hf_cache)",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip vector/BM25 indexing (just export eval set)",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant server URL (default: in-memory)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="ragbench",
        help="Qdrant collection name (default: ragbench)",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from rag_assistant.config import Settings
    from rag_assistant.rag.embeddings.model import create_embedding_model
    from rag_assistant.rag.ingestion import (
        create_eval_set,
        load_ragbench,
        ragbench_to_chunks,
    )
    from rag_assistant.rag.retrieval import BM25Retriever
    from rag_assistant.rag.store import QdrantStore

    # Determine subsets to load
    if args.subset == "all":
        subsets = ["FinQA", "ConvFinQA", "TAT-DQA"]
    else:
        subsets = [args.subset]

    logger.info("=" * 60)
    logger.info("T²-RAGBench Ingestion")
    logger.info("=" * 60)
    logger.info("Subsets: %s", subsets)
    logger.info("Max samples per subset: %s", args.max_samples or "unlimited")
    logger.info("Eval output: %s", args.eval_output)
    logger.info("Cache dir: %s", args.cache_dir)

    # Load data
    t0 = time.perf_counter()
    logger.info("\n[1/4] Loading T²-RAGBench dataset...")

    data = load_ragbench(
        subsets=subsets,  # type: ignore
        max_samples_per_subset=args.max_samples,
        cache_dir=args.cache_dir,
    )

    logger.info(
        "Loaded %d QA pairs, %d unique contexts in %.1fs",
        data.num_qa_pairs,
        data.num_contexts,
        time.perf_counter() - t0,
    )

    # Convert to chunks
    t0 = time.perf_counter()
    logger.info("\n[2/4] Converting to DocumentChunks...")

    chunks = ragbench_to_chunks(data)

    logger.info("Created %d chunks in %.1fs", len(chunks), time.perf_counter() - t0)

    # Sample chunk for verification
    if chunks:
        sample = chunks[0]
        logger.info("Sample chunk:")
        logger.info("  Source: %s", sample.metadata.source)
        logger.info("  Title: %s", sample.metadata.title)
        logger.info("  Section: %s", sample.metadata.section)
        logger.info("  Text: %.100s...", sample.text)

    # Index into vector store and BM25
    if not args.skip_indexing:
        t0 = time.perf_counter()
        logger.info("\n[3/4] Indexing documents...")

        # Create settings (use in-memory Qdrant by default)
        settings = Settings()
        if args.qdrant_url:
            settings.qdrant_url = args.qdrant_url

        # Create embedding model
        logger.info("Loading embedding model...")
        embedding_model = create_embedding_model(settings)

        # Create Qdrant store
        logger.info("Creating Qdrant collection: %s", args.collection_name)
        store = QdrantStore(
            collection_name=args.collection_name,
            url=settings.qdrant_url,
            embedding_dim=embedding_model.dimension,
        )

        # Index chunks
        logger.info("Computing embeddings and indexing...")
        texts = [c.text for c in chunks]
        embeddings = embedding_model.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            store.upsert_chunk(chunk, embedding)

        logger.info("Dense indexing complete: %d vectors", store.count())

        # BM25 index
        logger.info("Building BM25 index...")
        bm25 = BM25Retriever()
        bm25.index_chunks(chunks)
        logger.info("BM25 indexing complete: %d documents", bm25.document_count)

        logger.info("Indexing completed in %.1fs", time.perf_counter() - t0)
    else:
        logger.info("\n[3/4] Skipping indexing (--skip-indexing)")

    # Export evaluation set
    t0 = time.perf_counter()
    logger.info("\n[4/4] Exporting evaluation set...")

    eval_examples = create_eval_set(data, output_path=args.eval_output)

    logger.info(
        "Exported %d evaluation examples to %s in %.1fs",
        len(eval_examples),
        args.eval_output,
        time.perf_counter() - t0,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total contexts (documents): %d", data.num_contexts)
    logger.info("Total QA pairs: %d", data.num_qa_pairs)
    logger.info("Evaluation set: %s", args.eval_output)

    # Subset breakdown
    subset_counts: dict[str, int] = {}
    for qa in data.qa_pairs:
        subset_counts[qa.subset] = subset_counts.get(qa.subset, 0) + 1
    for subset, count in sorted(subset_counts.items()):
        logger.info("  %s: %d QA pairs", subset, count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
