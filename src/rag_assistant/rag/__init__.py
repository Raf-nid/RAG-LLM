"""
RAG (Retrieval-Augmented Generation) pipeline.

This package contains all components for document ingestion,
embedding, storage, retrieval, and generation.

Subpackages
-----------
embeddings
    Embedding model wrappers for converting text to vectors.
store
    Vector store integration (Qdrant).
ingestion
    Document loading, parsing, cleaning, and chunking.
retrieval
    Dense, lexical (BM25), and hybrid retrieval implementations.
generation
    Grounded answer generation with citations.

Main Classes
------------
RAGPipeline
    End-to-end pipeline combining retrieval and generation.
GroundedGenerator
    LLM-based answer generator with source citations.
DenseRetriever
    Semantic similarity search using embeddings.
BM25Retriever
    Lexical search using BM25 scoring.
HybridRetriever
    Combined dense + lexical search with RRF fusion.
"""

from .generation import GroundedGenerator, create_generator
from .pipeline import RAGPipeline, create_rag_pipeline
from .retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    create_hybrid_retriever,
    create_retriever,
)

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "GroundedGenerator",
    "HybridRetriever",
    "RAGPipeline",
    "create_generator",
    "create_hybrid_retriever",
    "create_rag_pipeline",
    "create_retriever",
]
