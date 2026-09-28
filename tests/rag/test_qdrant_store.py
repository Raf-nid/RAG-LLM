"""Tests for Qdrant store."""

import uuid

import pytest
from qdrant_client import QdrantClient

from rag_assistant.config import Settings
from rag_assistant.rag.store.qdrant import QdrantStore, create_qdrant_store
from rag_assistant.schemas.chunk import (
    ChunkMetadata,
    DocumentChunk,
    RetrievalMethod,
)


def make_uuid(name: str) -> str:
    """Generate a deterministic UUID for testing."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


@pytest.fixture
def in_memory_client() -> QdrantClient:
    """Create an in-memory Qdrant client for testing."""
    return QdrantClient(":memory:")


@pytest.fixture
def store(in_memory_client: QdrantClient) -> QdrantStore:
    """Create a store with in-memory client."""
    store = QdrantStore(
        client=in_memory_client,
        collection_name="test_collection",
        embedding_dimensions=384,
    )
    store.ensure_collection()
    return store


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """Create a sample chunk for testing."""
    metadata = ChunkMetadata(
        chunk_id=make_uuid("test_chunk_001"),
        source="langchain",
        title="Test Document",
        url="https://example.com/test",
        document_type="official_docs",
        section="Introduction",
        version="1.0",
        chunk_index=0,
        total_chunks=1,
    )
    return DocumentChunk(
        text="This is test content about LangChain agents.",
        metadata=metadata,
    )


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector."""
    return [0.1] * 384


class TestQdrantStoreInit:
    """Tests for store initialization."""

    def test_ensure_collection_creates_collection(
        self, in_memory_client: QdrantClient
    ) -> None:
        store = QdrantStore(
            client=in_memory_client,
            collection_name="new_collection",
            embedding_dimensions=384,
        )
        store.ensure_collection()

        collections = in_memory_client.get_collections().collections
        names = [c.name for c in collections]
        assert "new_collection" in names

    def test_ensure_collection_idempotent(self, store: QdrantStore) -> None:
        # Calling twice should not raise
        store.ensure_collection()
        store.ensure_collection()


class TestUpsertChunk:
    """Tests for upserting chunks."""

    def test_upsert_single_chunk(
        self,
        store: QdrantStore,
        sample_chunk: DocumentChunk,
        sample_embedding: list[float],
    ) -> None:
        store.upsert_chunk(sample_chunk, sample_embedding)

        count = store.count()
        assert count == 1

    def test_upsert_same_id_updates(
        self,
        store: QdrantStore,
        sample_chunk: DocumentChunk,
        sample_embedding: list[float],
    ) -> None:
        # Insert twice with same ID
        store.upsert_chunk(sample_chunk, sample_embedding)
        store.upsert_chunk(sample_chunk, sample_embedding)

        # Should still be only 1 chunk (upsert, not insert)
        count = store.count()
        assert count == 1


class TestUpsertChunks:
    """Tests for batch upserting."""

    def test_upsert_multiple_chunks(
        self,
        store: QdrantStore,
    ) -> None:
        chunks = []
        embeddings = []
        for i in range(5):
            metadata = ChunkMetadata(
                chunk_id=make_uuid(f"chunk_{i}"),
                source="test",
                title=f"Doc {i}",
                url=f"https://example.com/{i}",
            )
            chunks.append(DocumentChunk(text=f"Content {i}", metadata=metadata))
            embeddings.append([float(i) / 10] * 384)

        count = store.upsert_chunks(chunks, embeddings)
        assert count == 5
        assert store.count() == 5

    def test_upsert_chunks_mismatched_lengths_raises(
        self,
        store: QdrantStore,
    ) -> None:
        chunks = [
            DocumentChunk(
                text="Test",
                metadata=ChunkMetadata(
                    chunk_id=make_uuid("mismatch_test"),
                    source="test",
                    title="Test",
                    url="https://example.com",
                ),
            )
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            store.upsert_chunks(chunks, embeddings)

    def test_upsert_empty_list(self, store: QdrantStore) -> None:
        count = store.upsert_chunks([], [])
        assert count == 0


class TestSearch:
    """Tests for similarity search."""

    def test_search_returns_results(
        self,
        store: QdrantStore,
        sample_chunk: DocumentChunk,
        sample_embedding: list[float],
    ) -> None:
        store.upsert_chunk(sample_chunk, sample_embedding)

        results = store.search(sample_embedding, top_k=5)

        assert len(results) == 1
        assert results[0].chunk.text == sample_chunk.text
        assert results[0].retrieval_method == RetrievalMethod.dense

    def test_search_returns_scores(
        self,
        store: QdrantStore,
        sample_chunk: DocumentChunk,
        sample_embedding: list[float],
    ) -> None:
        store.upsert_chunk(sample_chunk, sample_embedding)

        results = store.search(sample_embedding, top_k=5)

        # Searching with exact embedding should give high score
        assert results[0].score > 0.99

    def test_search_respects_top_k(self, store: QdrantStore) -> None:
        # Insert 10 chunks
        for i in range(10):
            metadata = ChunkMetadata(
                chunk_id=make_uuid(f"chunk_{i}"),
                source="test",
                title=f"Doc {i}",
                url=f"https://example.com/{i}",
            )
            chunk = DocumentChunk(text=f"Content {i}", metadata=metadata)
            store.upsert_chunk(chunk, [float(i) / 10] * 384)

        results = store.search([0.5] * 384, top_k=3)
        assert len(results) == 3

    def test_search_with_source_filter(self, store: QdrantStore) -> None:
        # Insert chunks from different sources
        for source in ["langchain", "fastapi", "pydantic"]:
            metadata = ChunkMetadata(
                chunk_id=make_uuid(f"chunk_{source}"),
                source=source,
                title=f"Doc {source}",
                url=f"https://example.com/{source}",
            )
            chunk = DocumentChunk(text=f"Content about {source}", metadata=metadata)
            store.upsert_chunk(chunk, [0.5] * 384)

        # Search with filter
        results = store.search([0.5] * 384, source_filter="langchain")

        assert len(results) == 1
        assert results[0].chunk_metadata.source == "langchain"

    def test_search_empty_collection(self, store: QdrantStore) -> None:
        results = store.search([0.5] * 384)
        assert results == []


class TestCount:
    """Tests for counting chunks."""

    def test_count_empty(self, store: QdrantStore) -> None:
        assert store.count() == 0

    def test_count_all(self, store: QdrantStore) -> None:
        for i in range(3):
            metadata = ChunkMetadata(
                chunk_id=make_uuid(f"chunk_{i}"),
                source="test",
                title=f"Doc {i}",
                url=f"https://example.com/{i}",
            )
            chunk = DocumentChunk(text=f"Content {i}", metadata=metadata)
            store.upsert_chunk(chunk, [0.5] * 384)

        assert store.count() == 3

    def test_count_with_source_filter(self, store: QdrantStore) -> None:
        for i, source in enumerate(["langchain", "langchain", "fastapi"]):
            metadata = ChunkMetadata(
                chunk_id=make_uuid(f"chunk_{source}_{i}"),
                source=source,
                title="Doc",
                url="https://example.com",
            )
            chunk = DocumentChunk(text="Content", metadata=metadata)
            store.upsert_chunk(chunk, [0.5] * 384)

        assert store.count(source_filter="langchain") == 2
        assert store.count(source_filter="fastapi") == 1


class TestDeleteBySource:
    """Tests for deleting chunks by source."""

    def test_delete_removes_chunks(self, store: QdrantStore) -> None:
        for i, source in enumerate(["langchain", "langchain", "fastapi"]):
            metadata = ChunkMetadata(
                chunk_id=make_uuid(f"delete_chunk_{source}_{i}"),
                source=source,
                title="Doc",
                url="https://example.com",
            )
            chunk = DocumentChunk(text="Content", metadata=metadata)
            store.upsert_chunk(chunk, [0.5] * 384)

        deleted = store.delete_by_source("langchain")

        assert deleted == 2
        assert store.count() == 1
        assert store.count(source_filter="fastapi") == 1


class TestCreateQdrantStore:
    """Tests for the factory function."""

    def test_creates_store_with_settings(self) -> None:
        settings = Settings(
            qdrant_url=":memory:",
            qdrant_collection_name="test_collection",
            groq_api_key="test-key",
        )
        store = create_qdrant_store(settings, embedding_dimensions=384)
        assert store._collection_name == "test_collection"

    def test_uses_provided_client(self, in_memory_client: QdrantClient) -> None:
        settings = Settings(
            qdrant_collection_name="test_collection",
            groq_api_key="test-key",
        )
        store = create_qdrant_store(
            settings,
            embedding_dimensions=384,
            client=in_memory_client,
        )
        assert store._client is in_memory_client
