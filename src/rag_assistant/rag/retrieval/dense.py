"""
Dense retrieval using embeddings.

Dense retrieval works by:
1. Converting the query to an embedding vector
2. Finding the most similar document vectors
3. Returning the top-k matches

This is "semantic search" - it finds documents that are semantically
similar even if they don't share exact words with the query.

Limitations
-----------
- May miss exact keyword matches (API names, class names)
- Depends on embedding model quality
- Struggles with out-of-vocabulary terms

These limitations are addressed by hybrid retrieval (Step 10).

Usage
-----
    from rag_assistant.rag.retrieval import create_retriever
    from rag_assistant.config import settings

    retriever = create_retriever(settings)
    results = retriever.retrieve("How do I create a LangChain agent?")
"""

import logging
import time

from rag_assistant.config import Settings
from rag_assistant.rag.embeddings.model import EmbeddingModel, create_embedding_model
from rag_assistant.rag.store.qdrant import QdrantStore, create_qdrant_store
from rag_assistant.schemas.chunk import RetrievedChunk

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense semantic retriever.

    Uses embeddings for similarity-based retrieval.

    Parameters
    ----------
    embedding_model
        Model for embedding queries.
    store
        Vector store to search.
    default_top_k
        Default number of results to return.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        store: QdrantStore,
        default_top_k: int = 5,
    ) -> None:
        self._embedding_model = embedding_model
        self._store = store
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Parameters
        ----------
        query
            The search query.
        top_k
            Number of results (default: 5).
        source_filter
            Optional: filter by source (e.g., "langchain").
        score_threshold
            Optional: minimum similarity score.

        Returns
        -------
        list[RetrievedChunk]
            Retrieved chunks ordered by relevance (highest first).
        """
        if top_k is None:
            top_k = self._default_top_k

        start_time = time.perf_counter()

        # Embed the query
        query_vector = self._embedding_model.embed_query(query)
        embed_time = time.perf_counter() - start_time

        # Search the store
        results = self._store.search(
            query_vector=query_vector,
            top_k=top_k,
            source_filter=source_filter,
            score_threshold=score_threshold,
        )
        total_time = time.perf_counter() - start_time

        logger.debug(
            "Retrieved %d chunks for query (embed: %.1fms, total: %.1fms)",
            len(results),
            embed_time * 1000,
            total_time * 1000,
        )

        return results

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get the embedding model."""
        return self._embedding_model

    @property
    def store(self) -> QdrantStore:
        """Get the vector store."""
        return self._store

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(model={self._embedding_model.model_name!r}, "
            f"top_k={self._default_top_k})"
        )


def create_retriever(
    settings: Settings,
    *,
    embedding_model: EmbeddingModel | None = None,
    store: QdrantStore | None = None,
) -> DenseRetriever:
    """
    Create a dense retriever from settings.

    Parameters
    ----------
    settings
        Application settings.
    embedding_model
        Optional: pre-built embedding model.
    store
        Optional: pre-built store.

    Returns
    -------
    DenseRetriever
        Configured retriever ready for use.
    """
    if embedding_model is None:
        embedding_model = create_embedding_model(settings)

    if store is None:
        store = create_qdrant_store(settings, embedding_model.dimensions)
        store.ensure_collection()

    return DenseRetriever(
        embedding_model=embedding_model,
        store=store,
    )
