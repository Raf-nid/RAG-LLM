"""
Complete RAG pipeline combining retrieval and generation.

This module provides the end-to-end RAG pipeline that:
1. Retrieves relevant chunks for a query
2. Generates a grounded answer with citations
3. Handles insufficient context gracefully

Usage
-----
    from rag_assistant.rag.pipeline import RAGPipeline, create_rag_pipeline
    from rag_assistant.config import settings

    pipeline = create_rag_pipeline(settings)
    response = pipeline.query("How do I create a LangChain agent?")

    if isinstance(response, RAGResponse):
        print(response.answer)
        for source in response.sources:
            print(f"  - {source.title}: {source.url}")
    else:
        print(response.message)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from rag_assistant.schemas.chunk import RetrievedChunk
from rag_assistant.schemas.rag_response import (
    InsufficientContextResponse,
    RAGResponse,
    RetrievalMetadata,
)

from .generation.generator import GroundedGenerator
from .retrieval.dense import DenseRetriever

if TYPE_CHECKING:
    from rag_assistant.config import Settings
    from rag_assistant.llm.interface import LLMProviderProtocol
    from rag_assistant.rag.embeddings.model import EmbeddingModel
    from rag_assistant.rag.store.qdrant import QdrantStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Combines retrieval (dense) and generation (grounded with citations)
    into a single query interface.

    Parameters
    ----------
    retriever
        Dense retriever for finding relevant chunks.
    generator
        Grounded generator for producing answers.
    default_top_k
        Default number of chunks to retrieve.
    score_threshold
        Minimum retrieval score to consider a chunk relevant.
    """

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: GroundedGenerator,
        *,
        default_top_k: int = 5,
        score_threshold: float | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._default_top_k = default_top_k
        self._score_threshold = score_threshold

    @property
    def retriever(self) -> DenseRetriever:
        """Return the retriever."""
        return self._retriever

    @property
    def generator(self) -> GroundedGenerator:
        """Return the generator."""
        return self._generator

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        source_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> RAGResponse | InsufficientContextResponse:
        """
        Execute a complete RAG query.

        Parameters
        ----------
        question
            The user's question.
        top_k
            Number of chunks to retrieve. Defaults to pipeline default.
        source_filter
            Optional: filter by source (e.g., "langchain").
        score_threshold
            Optional: minimum score for retrieved chunks.

        Returns
        -------
        RAGResponse | InsufficientContextResponse
            Either a grounded answer with citations, or an indication
            that the context was insufficient.
        """
        if top_k is None:
            top_k = self._default_top_k
        if score_threshold is None:
            score_threshold = self._score_threshold

        start_time = time.perf_counter()

        # Step 1: Retrieve relevant chunks
        logger.info("RAG query: %r (top_k=%d)", question[:50], top_k)
        chunks = self._retriever.retrieve(
            query=question,
            top_k=top_k,
            source_filter=source_filter,
            score_threshold=score_threshold,
        )

        retrieval_ms = (time.perf_counter() - start_time) * 1000
        logger.debug("Retrieved %d chunks in %.0fms", len(chunks), retrieval_ms)

        # Build retrieval metadata
        retrieval_metadata = RetrievalMetadata(
            total_candidates=len(chunks),
            dense_candidates=len(chunks),
            lexical_candidates=0,
            reranked=False,
            latency_ms=retrieval_ms,
        )

        # Step 2: Generate grounded answer
        response = self._generator.generate(
            query=question,
            chunks=chunks,
            retrieval_metadata=retrieval_metadata,
        )

        total_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "RAG query completed in %.0fms (retrieval=%.0fms)",
            total_ms,
            retrieval_ms,
        )

        return response

    async def aquery(
        self,
        question: str,
        *,
        top_k: int | None = None,
        source_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> RAGResponse | InsufficientContextResponse:
        """
        Async version of query.

        Parameters
        ----------
        question
            The user's question.
        top_k
            Number of chunks to retrieve.
        source_filter
            Optional source filter.
        score_threshold
            Optional minimum score.

        Returns
        -------
        RAGResponse | InsufficientContextResponse
            The RAG response.
        """
        if top_k is None:
            top_k = self._default_top_k
        if score_threshold is None:
            score_threshold = self._score_threshold

        start_time = time.perf_counter()

        # Retrieve (sync for now, async retrieval can be added later)
        chunks = self._retriever.retrieve(
            query=question,
            top_k=top_k,
            source_filter=source_filter,
            score_threshold=score_threshold,
        )

        retrieval_ms = (time.perf_counter() - start_time) * 1000

        retrieval_metadata = RetrievalMetadata(
            total_candidates=len(chunks),
            dense_candidates=len(chunks),
            lexical_candidates=0,
            reranked=False,
            latency_ms=retrieval_ms,
        )

        # Generate (async)
        response = await self._generator.agenerate(
            query=question,
            chunks=chunks,
            retrieval_metadata=retrieval_metadata,
        )

        return response

    def retrieve_only(
        self,
        question: str,
        *,
        top_k: int | None = None,
        source_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks without generating an answer.

        Useful for debugging or when you want to inspect the retrieval
        results before generation.

        Parameters
        ----------
        question
            The query.
        top_k
            Number of chunks.
        source_filter
            Optional source filter.
        score_threshold
            Optional minimum score.

        Returns
        -------
        list[RetrievedChunk]
            Retrieved chunks.
        """
        if top_k is None:
            top_k = self._default_top_k
        if score_threshold is None:
            score_threshold = self._score_threshold

        return self._retriever.retrieve(
            query=question,
            top_k=top_k,
            source_filter=source_filter,
            score_threshold=score_threshold,
        )

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(retriever={self._retriever!r}, "
            f"generator={self._generator!r})"
        )


def create_rag_pipeline(
    settings: Settings,
    *,
    retriever: DenseRetriever | None = None,
    generator: GroundedGenerator | None = None,
    embedding_model: EmbeddingModel | None = None,
    store: QdrantStore | None = None,
    provider: LLMProviderProtocol | None = None,
) -> RAGPipeline:
    """
    Create a complete RAG pipeline from settings.

    Parameters
    ----------
    settings
        Application settings.
    retriever
        Optional: pre-built retriever.
    generator
        Optional: pre-built generator.
    embedding_model
        Optional: pre-built embedding model (for retriever).
    store
        Optional: pre-built vector store (for retriever).
    provider
        Optional: pre-built LLM provider (for generator).

    Returns
    -------
    RAGPipeline
        Configured pipeline.
    """
    # Build retriever if not provided
    if retriever is None:
        from .embeddings.model import create_embedding_model
        from .retrieval.dense import create_retriever
        from .store.qdrant import create_qdrant_store

        if embedding_model is None:
            embedding_model = create_embedding_model(settings)

        if store is None:
            store = create_qdrant_store(settings, embedding_model.dimensions)
            store.ensure_collection()

        retriever = create_retriever(
            settings,
            embedding_model=embedding_model,
            store=store,
        )

    # Build generator if not provided
    if generator is None:
        from .generation.generator import create_generator

        generator = create_generator(settings, provider=provider)

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        default_top_k=5,
        score_threshold=None,
    )
