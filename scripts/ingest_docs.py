#!/usr/bin/env python3
"""
Document ingestion script.

Ingests sample documentation into the Qdrant vector store.

Usage
-----
    # Ingest all sample documents (in-memory Qdrant)
    uv run python scripts/ingest_docs.py

    # Ingest with external Qdrant server
    QDRANT_URL=http://localhost:6333 uv run python scripts/ingest_docs.py

    # Ingest specific source only
    uv run python scripts/ingest_docs.py --source langchain

    # Query after ingestion
    uv run python scripts/ingest_docs.py --query "How do I create an agent?"
"""

import argparse
import logging
import sys

# Override QDRANT_URL to use in-memory by default for the MVP
import os

if "QDRANT_URL" not in os.environ:
    os.environ["QDRANT_URL"] = ":memory:"

from rag_assistant.config import Settings
from rag_assistant.rag.embeddings import create_embedding_model
from rag_assistant.rag.ingestion.pipeline import IngestionPipeline
from rag_assistant.rag.ingestion.sample_docs import get_sample_documents
from rag_assistant.rag.retrieval import create_retriever
from rag_assistant.rag.store import create_qdrant_store

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    parser.add_argument(
        "--source",
        type=str,
        help="Ingest only documents from this source (e.g., langchain)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a test query after ingestion",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results for test query (default: 3)",
    )
    args = parser.parse_args()

    # Load settings
    settings = Settings()

    print("=" * 60)
    print("RAG MVP - Document Ingestion")
    print("=" * 60)
    print()

    # Create components
    print("Initializing components...")
    print(f"  Embedding model: {settings.embedding_model}")
    print(f"  Embedding device: {settings.embedding_device}")
    print(f"  Qdrant URL: {settings.qdrant_url}")
    print(f"  Collection: {settings.qdrant_collection_name}")
    print(f"  Chunk size: {settings.chunk_size}")
    print(f"  Chunk overlap: {settings.chunk_overlap}")
    print()

    # Create embedding model (this downloads the model on first run)
    print("Loading embedding model (may download on first run)...")
    embedding_model = create_embedding_model(settings)
    print(f"  Model loaded: {embedding_model.model_name}")
    print(f"  Dimensions: {embedding_model.dimensions}")
    print()

    # Create store
    store = create_qdrant_store(settings, embedding_model.dimensions)

    # Create pipeline
    pipeline = IngestionPipeline(embedding_model, store, settings)

    # Get documents to ingest
    documents = get_sample_documents()
    if args.source:
        documents = [d for d in documents if d["source"] == args.source]
        if not documents:
            print(f"No documents found for source: {args.source}")
            return 1

    # Group by source
    sources = sorted(set(d["source"] for d in documents))
    print(f"Sources to ingest: {', '.join(sources)}")
    print(f"Total documents: {len(documents)}")
    print()

    # Ingest documents
    print("Ingesting documents...")
    print("-" * 40)

    total_chunks = 0
    for source in sources:
        source_docs = [d for d in documents if d["source"] == source]
        result = pipeline.ingest_documents(source_docs, source)

        status = "OK" if result.success else f"ERRORS: {result.errors}"
        print(
            f"  {source}: {result.documents_processed} docs → "
            f"{result.chunks_indexed} chunks [{status}]"
        )
        total_chunks += result.chunks_indexed

    print("-" * 40)
    print(f"Total chunks indexed: {total_chunks}")
    print()

    # Show stats
    stats = pipeline.get_stats()
    print("Index Statistics:")
    print(f"  Collection: {stats['collection']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Dimensions: {stats['embedding_dimensions']}")
    print()

    # Run test query if requested
    if args.query:
        print("=" * 60)
        print("Test Query")
        print("=" * 60)
        print(f"Query: {args.query}")
        print()

        retriever = create_retriever(
            settings,
            embedding_model=embedding_model,
            store=store,
        )

        results = retriever.retrieve(args.query, top_k=args.top_k)

        print(f"Retrieved {len(results)} chunks:")
        print("-" * 40)
        for i, chunk in enumerate(results, 1):
            print(f"\n[{i}] Score: {chunk.score:.4f}")
            print(f"    Source: {chunk.chunk_metadata.source}")
            print(f"    Title: {chunk.chunk_metadata.title}")
            print(f"    URL: {chunk.chunk_metadata.url}")
            print(f"    Text preview: {chunk.text[:200]}...")

    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
