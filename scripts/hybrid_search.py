#!/usr/bin/env python
"""
Demo script for hybrid retrieval (dense + BM25 with RRF fusion).

This script demonstrates the hybrid retrieval workflow:
1. Loads sample documents
2. Indexes them in both vector store (dense) and BM25 index
3. Queries using hybrid search
4. Shows detailed fusion results

Usage
-----
    # Basic query
    uv run python scripts/hybrid_search.py "How do I create an agent?"

    # Compare retrievers
    uv run python scripts/hybrid_search.py "create_react_agent function" --compare

    # Show fusion details
    uv run python scripts/hybrid_search.py "langchain tools" --details
"""

from __future__ import annotations

import argparse
import sys

from rag_assistant.config import settings
from rag_assistant.rag.embeddings.model import create_embedding_model
from rag_assistant.rag.ingestion.pipeline import IngestionPipeline
from rag_assistant.rag.ingestion.sample_docs import get_sample_documents
from rag_assistant.rag.retrieval.bm25 import BM25Retriever
from rag_assistant.rag.retrieval.dense import DenseRetriever
from rag_assistant.rag.retrieval.hybrid import HybridRetriever
from rag_assistant.rag.store.qdrant import create_qdrant_store
from rag_assistant.schemas.chunk import DocumentChunk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo hybrid retrieval with RRF fusion."
    )
    parser.add_argument("query", help="The search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare dense vs BM25 vs hybrid results",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed fusion information",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Hybrid Retrieval Demo - Dense + BM25 with RRF Fusion")
    print("=" * 70)
    print()

    # Initialize components
    print("Initializing components...")

    print(f"  - Embedding model: {settings.embedding_model}")
    embedding_model = create_embedding_model(settings)

    print(f"  - Vector store: {settings.qdrant_url}")
    store = create_qdrant_store(settings, embedding_model.dimensions)
    store.ensure_collection()

    # Ingest sample documents
    print()
    print("Ingesting sample documents...")

    pipeline = IngestionPipeline(
        embedding_model=embedding_model,
        store=store,
    )

    # Collect all chunks for BM25 indexing
    all_chunks: list[DocumentChunk] = []

    docs = get_sample_documents()
    for doc in docs:
        result = pipeline.ingest_document(
            text=doc["text"],
            source=doc["source"],
            url=doc["url"],
            title=doc["title"],
        )
        # Get chunks for BM25
        from rag_assistant.rag.ingestion.chunker import chunk_document

        chunks = chunk_document(
            text=doc["text"],
            source=doc["source"],
            url=doc["url"],
            title=doc["title"],
        )
        all_chunks.extend(chunks)
        print(f"  - {doc['title']}: {result.chunks_created} chunks")

    print(f"Total: {len(all_chunks)} chunks indexed")

    # Build retrievers
    print()
    print("Building retrievers...")

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        store=store,
    )
    print("  - Dense retriever ready")

    bm25_retriever = BM25Retriever()
    bm25_retriever.index_chunks(all_chunks)
    print(f"  - BM25 retriever ready ({bm25_retriever.document_count} documents)")

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )
    print("  - Hybrid retriever ready")

    # Query
    print()
    print("-" * 70)
    print(f"Query: {args.query}")
    print("-" * 70)

    if args.compare:
        # Compare all three retrievers
        print()
        print("DENSE RETRIEVAL (semantic similarity):")
        dense_results = dense_retriever.retrieve(args.query, top_k=args.top_k)
        for i, r in enumerate(dense_results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.chunk_metadata.title}")
            print(f"     {r.text[:80]}...")

        print()
        print("BM25 RETRIEVAL (keyword matching):")
        bm25_results = bm25_retriever.retrieve(args.query, top_k=args.top_k)
        for i, r in enumerate(bm25_results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.chunk_metadata.title}")
            print(f"     {r.text[:80]}...")

        print()
        print("HYBRID RETRIEVAL (RRF fusion):")
        hybrid_results = hybrid_retriever.retrieve(args.query, top_k=args.top_k)
        for i, r in enumerate(hybrid_results, 1):
            method = r.retrieval_method.value
            print(f"  {i}. [{r.score:.4f}] ({method}) {r.chunk_metadata.title}")
            print(f"     {r.text[:80]}...")

    elif args.details:
        # Show detailed fusion
        results, fusion_details = hybrid_retriever.retrieve_with_details(
            args.query, top_k=args.top_k
        )

        print()
        print("FUSION DETAILS:")
        print()
        print(
            f"{'Rank':<5} {'RRF Score':<12} {'Dense Rank':<12} "
            f"{'BM25 Rank':<12} {'Title'}"
        )
        print("-" * 70)

        for i, fr in enumerate(fusion_details[: args.top_k * 2], 1):
            dense_r = str(fr.dense_rank) if fr.dense_rank else "-"
            lex_r = str(fr.lexical_rank) if fr.lexical_rank else "-"
            title = fr.chunk.metadata.title[:30]
            print(f"{i:<5} {fr.fused_score:<12.6f} {dense_r:<12} {lex_r:<12} {title}")

        print()
        print("FINAL RESULTS:")
        for i, r in enumerate(results, 1):
            method = r.retrieval_method.value
            print(f"  {i}. [{r.score:.4f}] ({method}) {r.chunk_metadata.title}")

    else:
        # Standard hybrid search
        print()
        print("HYBRID RESULTS:")
        results = hybrid_retriever.retrieve(args.query, top_k=args.top_k)

        if not results:
            print("  No results found.")
        else:
            for i, r in enumerate(results, 1):
                method = r.retrieval_method.value
                print(f"  {i}. [{r.score:.4f}] ({method}) {r.chunk_metadata.title}")
                print(f"     Source: {r.chunk_metadata.source}")
                print(f"     {r.text[:100]}...")
                print()

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
