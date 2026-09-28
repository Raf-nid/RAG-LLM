"""Tests for the ingestion pipeline."""


import pytest
from qdrant_client import QdrantClient

from rag_assistant.config import Settings
from rag_assistant.rag.ingestion.pipeline import IngestionPipeline, IngestionResult
from rag_assistant.rag.store.qdrant import QdrantStore


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self) -> None:
        self._dimensions = 384

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return "mock-model"

    def embed_query(self, text: str) -> list[float]:
        return [0.5] * self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * self._dimensions for _ in texts]


@pytest.fixture
def mock_embedding_model() -> MockEmbeddingModel:
    """Create a mock embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
def in_memory_store() -> QdrantStore:
    """Create an in-memory store."""
    client = QdrantClient(":memory:")
    store = QdrantStore(
        client=client,
        collection_name="test_pipeline",
        embedding_dimensions=384,
    )
    return store


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        chunk_size=200,
        chunk_overlap=20,
        groq_api_key="test-key",
    )


@pytest.fixture
def pipeline(
    mock_embedding_model: MockEmbeddingModel,
    in_memory_store: QdrantStore,
    settings: Settings,
) -> IngestionPipeline:
    """Create a pipeline with mock components."""
    return IngestionPipeline(
        embedding_model=mock_embedding_model,
        store=in_memory_store,
        settings=settings,
    )


class TestIngestionResult:
    """Tests for IngestionResult."""

    def test_success_when_no_errors(self) -> None:
        result = IngestionResult(source="test")
        assert result.success is True

    def test_not_success_when_errors(self) -> None:
        result = IngestionResult(source="test", errors=["Something failed"])
        assert result.success is False

    def test_duration_none_when_not_completed(self) -> None:
        result = IngestionResult(source="test")
        assert result.duration_seconds is None


class TestIngestDocument:
    """Tests for single document ingestion."""

    def test_ingest_creates_chunks(self, pipeline: IngestionPipeline) -> None:
        result = pipeline.ingest_document(
            text="This is a test document with some content.",
            source="test",
            url="https://example.com",
            title="Test Document",
        )

        assert result.success
        assert result.documents_processed == 1
        assert result.chunks_created >= 1
        assert result.chunks_indexed >= 1

    def test_ingest_preserves_metadata(
        self,
        pipeline: IngestionPipeline,
        in_memory_store: QdrantStore,
    ) -> None:
        pipeline.ingest_document(
            text="Test content",
            source="langchain",
            url="https://example.com/langchain",
            title="LangChain Guide",
            section="Introduction",
            version="0.1",
        )

        # Search for the chunk and verify metadata
        results = in_memory_store.search([0.5] * 384)
        assert len(results) == 1
        assert results[0].chunk_metadata.source == "langchain"
        assert results[0].chunk_metadata.title == "LangChain Guide"
        assert results[0].chunk_metadata.section == "Introduction"
        assert results[0].chunk_metadata.version == "0.1"

    def test_ingest_empty_document(self, pipeline: IngestionPipeline) -> None:
        result = pipeline.ingest_document(
            text="",
            source="test",
            url="https://example.com",
            title="Empty",
        )

        assert result.success
        assert result.chunks_created == 0
        assert result.chunks_indexed == 0

    def test_ingest_idempotent(
        self,
        pipeline: IngestionPipeline,
        in_memory_store: QdrantStore,
    ) -> None:
        # Ingest the same document twice
        pipeline.ingest_document(
            text="Same content",
            source="test",
            url="https://example.com",
            title="Test",
        )
        pipeline.ingest_document(
            text="Same content",
            source="test",
            url="https://example.com",
            title="Test",
        )

        # Should only have 1 chunk (upsert, not insert)
        count = in_memory_store.count()
        assert count == 1


class TestIngestDocuments:
    """Tests for batch document ingestion."""

    def test_ingest_multiple_documents(self, pipeline: IngestionPipeline) -> None:
        documents = [
            {
                "text": "First document content",
                "url": "https://example.com/1",
                "title": "Doc 1",
            },
            {
                "text": "Second document content",
                "url": "https://example.com/2",
                "title": "Doc 2",
            },
            {
                "text": "Third document content",
                "url": "https://example.com/3",
                "title": "Doc 3",
            },
        ]

        result = pipeline.ingest_documents(documents, source="test")

        assert result.success
        assert result.documents_processed == 3
        assert result.chunks_indexed >= 3

    def test_ingest_with_optional_fields(self, pipeline: IngestionPipeline) -> None:
        documents = [
            {
                "text": "Content with all fields",
                "url": "https://example.com",
                "title": "Full Doc",
                "section": "Intro",
                "version": "1.0",
                "document_type": "tutorial",
            }
        ]

        result = pipeline.ingest_documents(documents, source="test")
        assert result.success

    def test_ingest_empty_list(self, pipeline: IngestionPipeline) -> None:
        result = pipeline.ingest_documents([], source="test")
        assert result.success
        assert result.documents_processed == 0
        assert result.chunks_indexed == 0


class TestGetStats:
    """Tests for pipeline statistics."""

    def test_get_stats_empty_collection(self, pipeline: IngestionPipeline) -> None:
        stats = pipeline.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["embedding_model"] == "mock-model"
        assert stats["embedding_dimensions"] == 384

    def test_get_stats_after_ingestion(self, pipeline: IngestionPipeline) -> None:
        pipeline.ingest_document(
            text="Some content",
            source="test",
            url="https://example.com",
            title="Test",
        )

        stats = pipeline.get_stats()
        assert stats["total_chunks"] >= 1
