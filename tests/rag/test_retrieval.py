"""Tests for dense retrieval."""

import uuid

import pytest
from qdrant_client import QdrantClient

from rag_assistant.config import Settings
from rag_assistant.rag.retrieval.dense import DenseRetriever, create_retriever
from rag_assistant.rag.store.qdrant import QdrantStore
from rag_assistant.schemas.chunk import (
    ChunkMetadata,
    DocumentChunk,
    RetrievalMethod,
    RetrievedChunk,
)


def make_uuid(name: str) -> str:
    """Generate a deterministic UUID for testing."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


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
        # Return a deterministic vector based on text
        return [float(ord(c) % 10) / 10 for c in text[:self._dimensions].ljust(self._dimensions)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


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
        collection_name="test_retrieval",
        embedding_dimensions=384,
    )
    store.ensure_collection()
    return store


@pytest.fixture
def populated_store(
    in_memory_store: QdrantStore,
    mock_embedding_model: MockEmbeddingModel,
) -> QdrantStore:
    """Create a store with some test data."""
    documents = [
        ("langchain", "Agents", "LangChain agents use LLMs to decide which tools to call."),
        ("langchain", "Tools", "Tools are functions that agents can call."),
        ("fastapi", "First Steps", "FastAPI is a modern web framework."),
        ("pydantic", "Models", "Pydantic uses type hints for data validation."),
    ]

    for source, title, text in documents:
        metadata = ChunkMetadata(
            chunk_id=make_uuid(f"{source}_{title.lower().replace(' ', '_')}"),
            source=source,
            title=title,
            url=f"https://example.com/{source}",
        )
        chunk = DocumentChunk(text=text, metadata=metadata)
        embedding = mock_embedding_model.embed_query(text)
        in_memory_store.upsert_chunk(chunk, embedding)

    return in_memory_store


@pytest.fixture
def retriever(
    mock_embedding_model: MockEmbeddingModel,
    populated_store: QdrantStore,
) -> DenseRetriever:
    """Create a retriever with mock components."""
    return DenseRetriever(
        embedding_model=mock_embedding_model,
        store=populated_store,
    )


class TestDenseRetriever:
    """Tests for the dense retriever."""

    def test_retrieve_returns_results(self, retriever: DenseRetriever) -> None:
        results = retriever.retrieve("How do agents work?")
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_respects_top_k(self, retriever: DenseRetriever) -> None:
        results = retriever.retrieve("How do agents work?", top_k=2)
        assert len(results) <= 2

    def test_retrieve_default_top_k(self) -> None:
        mock_model = MockEmbeddingModel()
        client = QdrantClient(":memory:")
        store = QdrantStore(client, "test", 384)
        store.ensure_collection()

        retriever = DenseRetriever(mock_model, store, default_top_k=3)

        # Insert some chunks
        for i in range(10):
            metadata = ChunkMetadata(
                chunk_id=make_uuid(f"chunk_{i}"),
                source="test",
                title=f"Doc {i}",
                url=f"https://example.com/{i}",
            )
            chunk = DocumentChunk(text=f"Content {i}", metadata=metadata)
            store.upsert_chunk(chunk, mock_model.embed_query(f"Content {i}"))

        results = retriever.retrieve("Content")
        assert len(results) == 3  # Default top_k

    def test_retrieve_with_source_filter(self, retriever: DenseRetriever) -> None:
        results = retriever.retrieve("agents", source_filter="langchain")
        assert all(r.chunk_metadata.source == "langchain" for r in results)

    def test_retrieve_results_have_scores(self, retriever: DenseRetriever) -> None:
        results = retriever.retrieve("How do agents work?")
        for result in results:
            assert isinstance(result.score, float)
            assert 0 <= result.score <= 1  # Cosine similarity range

    def test_retrieve_results_have_retrieval_method(
        self, retriever: DenseRetriever
    ) -> None:
        results = retriever.retrieve("agents")
        for result in results:
            assert result.retrieval_method == RetrievalMethod.dense

    def test_retrieve_empty_query(self, retriever: DenseRetriever) -> None:
        # Should not raise, just return results
        results = retriever.retrieve("")
        assert isinstance(results, list)

    def test_retrieve_from_empty_store(
        self, mock_embedding_model: MockEmbeddingModel
    ) -> None:
        client = QdrantClient(":memory:")
        store = QdrantStore(client, "empty_collection", 384)
        store.ensure_collection()

        retriever = DenseRetriever(mock_embedding_model, store)
        results = retriever.retrieve("anything")

        assert results == []


class TestRetrieverProperties:
    """Tests for retriever properties."""

    def test_embedding_model_property(
        self,
        retriever: DenseRetriever,
        mock_embedding_model: MockEmbeddingModel,
    ) -> None:
        assert retriever.embedding_model is mock_embedding_model

    def test_store_property(
        self,
        retriever: DenseRetriever,
        populated_store: QdrantStore,
    ) -> None:
        assert retriever.store is populated_store

    def test_repr(self, retriever: DenseRetriever) -> None:
        repr_str = repr(retriever)
        assert "DenseRetriever" in repr_str
        assert "mock-model" in repr_str


class TestCreateRetriever:
    """Tests for the factory function."""

    def test_creates_retriever_with_provided_components(
        self,
        mock_embedding_model: MockEmbeddingModel,
        in_memory_store: QdrantStore,
    ) -> None:
        settings = Settings(groq_api_key="test-key")
        retriever = create_retriever(
            settings,
            embedding_model=mock_embedding_model,
            store=in_memory_store,
        )

        assert retriever.embedding_model is mock_embedding_model
        assert retriever.store is in_memory_store
