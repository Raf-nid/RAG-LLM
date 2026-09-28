#!/usr/bin/env python
"""
Demo script for the complete RAG pipeline.

This script demonstrates the full RAG workflow:
1. Loads sample documents into the vector store
2. Queries the RAG pipeline with a question
3. Displays the grounded answer with citations

Usage
-----
    # Basic query
    uv run python scripts/rag_query.py "How do I create an agent?"

    # With source filter
    uv run python scripts/rag_query.py "What is FastAPI?" --source fastapi

    # Adjust number of chunks
    uv run python scripts/rag_query.py "Explain tool calling" --top-k 3

Requirements
------------
- GROQ_API_KEY set in .env file
- Or LLM_PROVIDER=ollama with Ollama running locally
"""

from __future__ import annotations

import argparse
import sys

from rag_assistant.config import settings
from rag_assistant.rag.embeddings.model import create_embedding_model
from rag_assistant.rag.generation import GroundedGenerator, create_generator
from rag_assistant.rag.ingestion.pipeline import IngestionPipeline
from rag_assistant.rag.ingestion.sample_docs import get_sample_documents
from rag_assistant.rag.pipeline import RAGPipeline
from rag_assistant.rag.retrieval.dense import DenseRetriever
from rag_assistant.rag.store.qdrant import create_qdrant_store
from rag_assistant.schemas.rag_response import InsufficientContextResponse, RAGResponse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query the RAG pipeline with grounded answers and citations."
    )
    parser.add_argument("query", help="The question to ask")
    parser.add_argument(
        "--source",
        help="Filter by source (e.g., langchain, fastapi)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip document ingestion (use existing data)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Pipeline Demo - Grounded Answers with Citations")
    print("=" * 60)
    print()

    # Initialize components
    print("Initializing components...")

    print(f"  - Embedding model: {settings.embedding_model}")
    embedding_model = create_embedding_model(settings)
    print(f"    Dimensions: {embedding_model.dimensions}")

    print(f"  - Vector store: {settings.qdrant_url}")
    store = create_qdrant_store(settings, embedding_model.dimensions)
    store.ensure_collection()

    print(f"  - LLM provider: {settings.llm_provider}")
    print(f"  - LLM model: {settings.llm_model}")

    # Ingest sample documents
    if not args.skip_ingest:
        print()
        print("Ingesting sample documents...")
        pipeline = IngestionPipeline(
            embedding_model=embedding_model,
            store=store,
        )

        docs = get_sample_documents()
        for doc in docs:
            result = pipeline.ingest_document(
                text=doc["text"],
                source=doc["source"],
                url=doc["url"],
                title=doc["title"],
            )
            print(f"  - {doc['title']}: {result.chunks_created} chunks")

        print(f"Total documents indexed: {store.count()}")

    # Build retriever
    retriever = DenseRetriever(
        embedding_model=embedding_model,
        store=store,
    )

    # Build generator
    generator = create_generator(settings)

    # Build pipeline
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        default_top_k=args.top_k,
    )

    # Execute query
    print()
    print("-" * 60)
    print(f"Query: {args.query}")
    if args.source:
        print(f"Source filter: {args.source}")
    print("-" * 60)
    print()

    response = rag_pipeline.query(
        args.query,
        source_filter=args.source,
    )

    # Display response
    if isinstance(response, RAGResponse):
        print("ANSWER:")
        print(response.answer)
        print()

        if response.sources:
            print("SOURCES:")
            for i, source in enumerate(response.sources, 1):
                print(f"  [{i}] {source.title}")
                print(f"      URL: {source.url}")
                if source.relevance_score:
                    print(f"      Score: {source.relevance_score:.3f}")

        if response.retrieval_metadata:
            print()
            print("METADATA:")
            print(f"  - Chunks retrieved: {response.retrieval_metadata.total_candidates}")
            print(f"  - Total latency: {response.retrieval_metadata.latency_ms:.0f}ms")

    elif isinstance(response, InsufficientContextResponse):
        print("INSUFFICIENT CONTEXT")
        print(response.message)
        print()
        print(f"Candidates found: {response.candidates_found}")
        if response.suggestion:
            print(f"Suggestion: {response.suggestion}")

    print()
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
