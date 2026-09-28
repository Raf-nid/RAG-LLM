"""
Chunk and metadata schemas for the RAG system.

These models define the data structures used throughout the ingestion
and retrieval pipelines. They are pure Pydantic models with no external
dependencies.

Design Principles
-----------------
- All metadata needed for citations must be present
- Chunk IDs must be stable and content-based
- Fields are typed strictly to catch errors early
- Models are frozen (immutable) where appropriate

Usage
-----
During ingestion:
    chunk = DocumentChunk(text="...", metadata=ChunkMetadata(...))

During retrieval:
    retrieved = RetrievedChunk(chunk=chunk, score=0.85, retrieval_method="dense")
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field


class DocumentSource(StrEnum):
    """Known documentation sources."""

    langchain = "langchain"
    langgraph = "langgraph"
    fastapi = "fastapi"
    pydantic = "pydantic"
    qdrant = "qdrant"
    groq = "groq"
    ollama = "ollama"
    docker = "docker"


class DocumentType(StrEnum):
    """Types of documentation content."""

    official_docs = "official_docs"
    api_reference = "api_reference"
    tutorial = "tutorial"
    example = "example"
    changelog = "changelog"


class ChunkMetadata(BaseModel):
    """
    Metadata for a document chunk.

    All fields required for proper citation display and filtering.
    The chunk_id is auto-generated if not provided.

    Attributes
    ----------
    chunk_id
        Unique, stable identifier. Generated from content hash if not provided.
    source
        Documentation source (e.g., "langchain", "fastapi").
    title
        Document or page title.
    url
        Original source URL (for citation links).
    document_type
        Classification of the content type.
    indexed_at
        When this chunk was indexed (UTC).
    section
        Section heading within the document (if available).
    version
        Documentation version (if known).
    chunk_index
        Position of this chunk within the source document.
    total_chunks
        Total number of chunks from the source document.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique, stable chunk identifier (content hash).")
    source: str = Field(description="Documentation source identifier (e.g., 'langchain').")
    title: str = Field(
        description="Document or page title.",
        min_length=1,
    )
    url: str = Field(
        description="Original source URL.",
        min_length=1,
    )
    document_type: str = Field(
        default=DocumentType.official_docs,
        description="Type of documentation content.",
    )
    indexed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when this chunk was indexed (UTC).",
    )
    section: str | None = Field(
        default=None,
        description="Section heading within the document.",
    )
    version: str | None = Field(
        default=None,
        description="Documentation version.",
    )
    chunk_index: int = Field(
        default=0,
        ge=0,
        description="Position of this chunk within the document.",
    )
    total_chunks: int = Field(
        default=1,
        ge=1,
        description="Total chunks from this document.",
    )

    def to_qdrant_payload(self) -> dict[str, str | int | None]:
        """
        Convert to Qdrant payload format.

        Qdrant stores payloads as JSON-serializable dicts.
        Datetime is converted to ISO format string.
        """
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "document_type": self.document_type,
            "indexed_at": self.indexed_at.isoformat(),
            "section": self.section,
            "version": self.version,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


class DocumentChunk(BaseModel):
    """
    A chunk of text with its metadata.

    This is the unit of storage in the vector database.
    Each chunk is independently retrievable and citable.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        description="The chunk text content.",
        min_length=1,
    )
    metadata: ChunkMetadata = Field(
        description="Metadata for citation and filtering.",
    )

    @property
    def chunk_id(self) -> str:
        """Convenience accessor for metadata.chunk_id."""
        return self.metadata.chunk_id


class RetrievalMethod(StrEnum):
    """Method used to retrieve a chunk."""

    dense = "dense"
    lexical = "lexical"
    hybrid = "hybrid"
    reranked = "reranked"


class RetrievedChunk(BaseModel):
    """
    A chunk returned from retrieval with score.

    Includes the retrieval score and method for debugging
    and evaluation purposes.
    """

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk = Field(
        description="The retrieved document chunk.",
    )
    score: float = Field(
        description="Retrieval score (higher is more relevant).",
    )
    retrieval_method: RetrievalMethod = Field(
        description="Method used to retrieve this chunk.",
    )

    @property
    def text(self) -> str:
        """Convenience accessor for chunk text."""
        return self.chunk.text

    @property
    def chunk_metadata(self) -> ChunkMetadata:
        """Convenience accessor for chunk metadata."""
        return self.chunk.metadata


def generate_chunk_id(source: str, url: str, text: str) -> str:
    """
    Generate a stable, content-based chunk ID as a valid UUID.

    Uses UUID5 (SHA-1 based, deterministic) with a custom namespace
    derived from the content. This ensures:
    - Same content -> same UUID (idempotent indexing)
    - Different content -> different UUID (no collisions)
    - Valid UUID format (required by Qdrant 1.19+)

    Parameters
    ----------
    source
        Documentation source identifier.
    url
        Source URL.
    text
        Chunk text content.

    Returns
    -------
    str
        UUID string in standard format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).

    Example
    -------
    >>> generate_chunk_id("langchain", "https://...", "Some text")
    '550e8400-e29b-41d4-a716-446655440000'
    """
    content = f"{source}:{url}:{text}"
    content_hash = sha256(content.encode()).hexdigest()
    namespace = uuid.UUID(content_hash[:32])
    return str(uuid.uuid5(namespace, content))
