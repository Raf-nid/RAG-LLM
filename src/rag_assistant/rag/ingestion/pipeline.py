"""
Document ingestion pipeline.

Orchestrates the full ingestion flow:
1. Accept documents (text + metadata)
2. Chunk documents
3. Generate embeddings
4. Store in Qdrant

This pipeline is designed to be:
- **Idempotent**: Re-running with the same documents won't create duplicates
- **Incremental**: Can add documents without re-indexing everything
- **Observable**: Reports statistics and progress

Usage
-----
    from rag_assistant.rag.ingestion import IngestionPipeline
    from rag_assistant.rag.embeddings import create_embedding_model
    from rag_assistant.rag.store import create_qdrant_store

    # Create components
    embeddings = create_embedding_model(settings)
    store = create_qdrant_store(settings, embeddings.dimensions)
    pipeline = IngestionPipeline(embeddings, store, settings)

    # Ingest a document
    result = pipeline.ingest_document(
        text="Document content...",
        source="langchain",
        url="https://...",
        title="Introduction",
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from rag_assistant.config import Settings
from rag_assistant.rag.embeddings.model import EmbeddingModel
from rag_assistant.rag.ingestion.chunker import chunk_document
from rag_assistant.rag.store.qdrant import QdrantStore
from rag_assistant.schemas.chunk import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """
    Result of an ingestion operation.

    Tracks statistics for monitoring and debugging.
    """

    source: str
    documents_processed: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def success(self) -> bool:
        """True if no errors occurred."""
        return len(self.errors) == 0

    @property
    def duration_seconds(self) -> float | None:
        """Duration in seconds, or None if not completed."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


class IngestionPipeline:
    """
    Document ingestion pipeline.

    Coordinates chunking, embedding, and storage of documents.

    Parameters
    ----------
    embedding_model
        Model for generating embeddings.
    store
        Vector store for indexing.
    settings
        Application settings (for chunk_size, chunk_overlap).
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        store: QdrantStore,
        settings: Settings,
    ) -> None:
        self._embedding_model = embedding_model
        self._store = store
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

        # Ensure collection exists
        self._store.ensure_collection()

    def ingest_document(
        self,
        text: str,
        source: str,
        url: str,
        title: str,
        *,
        section: str | None = None,
        version: str | None = None,
        document_type: str = "official_docs",
    ) -> IngestionResult:
        """
        Ingest a single document.

        Parameters
        ----------
        text
            Document text content.
        source
            Source identifier (e.g., "langchain").
        url
            Original source URL.
        title
            Document title.
        section
            Section heading (optional).
        version
            Documentation version (optional).
        document_type
            Type of content.

        Returns
        -------
        IngestionResult
            Statistics about the ingestion.
        """
        result = IngestionResult(source=source)

        try:
            # Chunk the document
            chunks = chunk_document(
                text=text,
                source=source,
                url=url,
                title=title,
                section=section,
                version=version,
                document_type=document_type,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            result.documents_processed = 1
            result.chunks_created = len(chunks)

            if not chunks:
                logger.warning("No chunks created from document: %s", title)
                result.completed_at = datetime.now()
                return result

            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            embeddings = self._embedding_model.embed_documents(texts)

            # Index in store
            indexed = self._store.upsert_chunks(chunks, embeddings)
            result.chunks_indexed = indexed

            logger.info(
                "Ingested document '%s' (%s): %d chunks",
                title,
                source,
                indexed,
            )

        except Exception as exc:
            logger.exception("Error ingesting document '%s': %s", title, exc)
            result.errors.append(str(exc))

        result.completed_at = datetime.now()
        return result

    def ingest_documents(
        self,
        documents: list[dict[str, str]],
        source: str,
    ) -> IngestionResult:
        """
        Ingest multiple documents from the same source.

        Parameters
        ----------
        documents
            List of dicts with keys: text, url, title, and optionally
            section, version, document_type.
        source
            Common source identifier.

        Returns
        -------
        IngestionResult
            Aggregated statistics.
        """
        result = IngestionResult(source=source)
        all_chunks: list[DocumentChunk] = []

        for doc in documents:
            try:
                chunks = chunk_document(
                    text=doc["text"],
                    source=source,
                    url=doc["url"],
                    title=doc["title"],
                    section=doc.get("section"),
                    version=doc.get("version"),
                    document_type=doc.get("document_type", "official_docs"),
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                )
                all_chunks.extend(chunks)
                result.documents_processed += 1
            except Exception as exc:
                logger.exception("Error chunking document '%s': %s", doc.get("title"), exc)
                result.errors.append(f"{doc.get('title')}: {exc}")

        result.chunks_created = len(all_chunks)

        if all_chunks:
            try:
                # Batch embed all chunks
                texts = [chunk.text for chunk in all_chunks]
                logger.info("Generating embeddings for %d chunks...", len(texts))
                embeddings = self._embedding_model.embed_documents(texts)

                # Batch index
                logger.info("Indexing %d chunks...", len(all_chunks))
                indexed = self._store.upsert_chunks(all_chunks, embeddings)
                result.chunks_indexed = indexed

            except Exception as exc:
                logger.exception("Error during embedding/indexing: %s", exc)
                result.errors.append(str(exc))

        result.completed_at = datetime.now()
        return result

    def get_stats(self) -> dict[str, int | str]:
        """
        Get current index statistics.

        Returns
        -------
        dict
            Statistics including chunk counts.
        """
        info = self._store.collection_info()
        return {
            "collection": info["name"],
            "total_chunks": info["points_count"],
            "status": info["status"],
            "embedding_model": self._embedding_model.model_name,
            "embedding_dimensions": self._embedding_model.dimensions,
        }
