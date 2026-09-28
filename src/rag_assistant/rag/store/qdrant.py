"""
Qdrant vector store integration.

Qdrant is a vector database designed for similarity search. It stores
vectors (embeddings) along with payloads (metadata) and supports
efficient nearest-neighbor search.

Key Concepts
------------
**Collection**: A named container for vectors, similar to a table in SQL.
Each collection has a fixed vector configuration (dimensions, distance).

**Point**: A single entry in a collection, consisting of:
- id: Unique identifier (we use chunk_id)
- vector: The embedding
- payload: Metadata dict (source, title, url, etc.)

**Distance**: How similarity is measured:
- Cosine: angle between vectors (most common for text)
- Euclidean: straight-line distance
- Dot: dot product (for normalized vectors)

**Payload Index**: Indexes on payload fields for filtering.
Creating an index on "source" allows fast filtering like
`source == "langchain"`.

Usage Modes
-----------
1. **Server mode**: Connect to a running Qdrant instance
   QDRANT_URL=http://localhost:6333

2. **In-memory mode**: No server needed, data lost on restart
   QDRANT_URL=:memory:

3. **Local persistence**: Stores data on disk without a server
   QDRANT_URL=path/to/data  (detected if path exists)

Usage
-----
    from rag_assistant.rag.store import create_qdrant_store
    from rag_assistant.config import settings

    store = create_qdrant_store(settings, embedding_dimensions=384)

    # Index a chunk
    store.upsert_chunk(chunk, embedding)

    # Search
    results = store.search(query_vector, top_k=5)
"""

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag_assistant.config import Settings
from rag_assistant.schemas.chunk import (
    ChunkMetadata,
    DocumentChunk,
    RetrievalMethod,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Qdrant vector store wrapper.

    Handles collection creation, document indexing, and similarity search.
    Supports both server mode and in-memory mode for testing.

    Parameters
    ----------
    client
        Qdrant client instance.
    collection_name
        Name of the collection to use.
    embedding_dimensions
        Dimensionality of the embedding vectors.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_dimensions: int,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedding_dimensions = embedding_dimensions

    def ensure_collection(self) -> None:
        """
        Create the collection if it doesn't exist.

        Uses cosine distance which is standard for text embeddings.
        Creates payload indexes for efficient filtering.
        """
        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)

        if not exists:
            logger.info(
                "Creating Qdrant collection %r with %d dimensions",
                self._collection_name,
                self._embedding_dimensions,
            )
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for common filters
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="document_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        else:
            logger.debug("Collection %r already exists", self._collection_name)

    def upsert_chunk(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        """
        Insert or update a single chunk.

        Uses the chunk_id as the point ID. If a point with that ID
        already exists, it will be overwritten (idempotent upsert).

        Parameters
        ----------
        chunk
            The document chunk with metadata.
        embedding
            The embedding vector for this chunk.
        """
        payload = chunk.metadata.to_qdrant_payload()
        payload["text"] = chunk.text  # Store text in payload for retrieval

        point = PointStruct(
            id=chunk.metadata.chunk_id,
            vector=embedding,
            payload=payload,
        )
        self._client.upsert(
            collection_name=self._collection_name,
            points=[point],
        )
        logger.debug("Upserted chunk %s", chunk.metadata.chunk_id)

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Insert or update multiple chunks in batches.

        More efficient than individual upserts due to reduced
        network overhead.

        Parameters
        ----------
        chunks
            List of document chunks.
        embeddings
            Corresponding embedding vectors (same length as chunks).
        batch_size
            Number of points per batch (default 100).

        Returns
        -------
        int
            Number of chunks upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have same length: "
                f"{len(chunks)} vs {len(embeddings)}"
            )

        if not chunks:
            return 0

        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            payload = chunk.metadata.to_qdrant_payload()
            payload["text"] = chunk.text
            points.append(
                PointStruct(
                    id=chunk.metadata.chunk_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert in batches
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._collection_name,
                points=batch,
            )
            total += len(batch)
            logger.debug("Upserted batch of %d chunks (total: %d)", len(batch), total)

        return total

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search for similar chunks.

        Parameters
        ----------
        query_vector
            The query embedding.
        top_k
            Number of results to return.
        source_filter
            Optional: filter by source (e.g., "langchain").
        score_threshold
            Optional: minimum similarity score (0-1 for cosine).

        Returns
        -------
        list[RetrievedChunk]
            Retrieved chunks with scores, ordered by relevance.
        """
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter),
                    )
                ]
            )

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        retrieved = []
        for result in response.points:
            payload: dict[str, Any] = result.payload or {}

            metadata = ChunkMetadata(
                chunk_id=str(result.id),
                source=payload.get("source", "unknown"),
                title=payload.get("title", "Unknown"),
                url=payload.get("url", ""),
                document_type=payload.get("document_type", "official_docs"),
                section=payload.get("section"),
                version=payload.get("version"),
                chunk_index=payload.get("chunk_index", 0),
                total_chunks=payload.get("total_chunks", 1),
            )

            chunk = DocumentChunk(
                text=payload.get("text", ""),
                metadata=metadata,
            )

            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=result.score,
                    retrieval_method=RetrievalMethod.dense,
                )
            )

        return retrieved

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source.

        Useful for re-indexing a documentation source.

        Parameters
        ----------
        source
            Source identifier (e.g., "langchain").

        Returns
        -------
        int
            Number of points deleted (approximate).
        """
        # Count before delete
        count_before = self.count(source_filter=source)

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source),
                    )
                ]
            ),
        )

        logger.info("Deleted %d chunks from source %r", count_before, source)
        return count_before

    def count(self, source_filter: str | None = None) -> int:
        """
        Count chunks in the collection.

        Parameters
        ----------
        source_filter
            Optional: count only chunks from this source.

        Returns
        -------
        int
            Number of chunks.
        """
        if source_filter:
            result = self._client.count(
                collection_name=self._collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_filter),
                        )
                    ]
                ),
            )
        else:
            result = self._client.count(collection_name=self._collection_name)

        return result.count

    def collection_info(self) -> dict[str, Any]:
        """
        Get collection information for debugging.

        Returns
        -------
        dict
            Collection stats including point count, config, etc.
        """
        info = self._client.get_collection(self._collection_name)
        return {
            "name": self._collection_name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.name if info.status else "unknown",
        }

    def __repr__(self) -> str:
        return (
            f"QdrantStore(collection={self._collection_name!r}, "
            f"dimensions={self._embedding_dimensions})"
        )


def create_qdrant_store(
    settings: Settings,
    embedding_dimensions: int,
    *,
    client: QdrantClient | None = None,
) -> QdrantStore:
    """
    Create a Qdrant store from application settings.

    Parameters
    ----------
    settings
        Application settings.
    embedding_dimensions
        Dimensionality of embedding vectors.
    client
        Optional: pre-built client for testing.

    Returns
    -------
    QdrantStore
        Configured store ready for use.
    """
    if client is None:
        url = settings.qdrant_url

        # In-memory mode for testing
        if url == ":memory:":
            logger.info("Using in-memory Qdrant")
            client = QdrantClient(":memory:")
        else:
            api_key = (
                settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None
            )
            logger.info("Connecting to Qdrant at %s", url)
            client = QdrantClient(url=url, api_key=api_key)

    store = QdrantStore(
        client=client,
        collection_name=settings.qdrant_collection_name,
        embedding_dimensions=embedding_dimensions,
    )

    return store
