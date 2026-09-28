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
    Dense and hybrid retrieval implementations.
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
"""

from .generation import GroundedGenerator, create_generator
from .pipeline import RAGPipeline, create_rag_pipeline
from .retrieval import DenseRetriever, create_retriever

__all__ = [
    "DenseRetriever",
    "GroundedGenerator",
    "RAGPipeline",
    "create_generator",
    "create_rag_pipeline",
    "create_retriever",
]
