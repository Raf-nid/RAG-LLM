"""Tests for embedding models."""

from unittest.mock import MagicMock, patch

from rag_assistant.config import Settings
from rag_assistant.rag.embeddings.model import (
    LocalEmbeddingModel,
    create_embedding_model,
)


class MockSentenceTransformer:
    """Mock for SentenceTransformer to avoid loading real models in tests."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._embedding_dim = 384  # all-MiniLM-L6-v2 dimensions

    def get_sentence_embedding_dimension(self) -> int:
        return self._embedding_dim

    def encode(
        self,
        sentences: str | list[str],
        convert_to_numpy: bool = False,
    ) -> MagicMock:
        """Return mock embeddings of correct shape."""
        import numpy as np

        if isinstance(sentences, str):
            result = np.random.randn(self._embedding_dim).astype(np.float32)
        else:
            result = np.random.randn(len(sentences), self._embedding_dim).astype(np.float32)
        return result


class TestLocalEmbeddingModel:
    """Tests for LocalEmbeddingModel."""

    @patch("rag_assistant.rag.embeddings.model.SentenceTransformer", MockSentenceTransformer)
    def test_init_stores_model_name(self) -> None:
        model = LocalEmbeddingModel("test-model", device="cpu")
        assert model.model_name == "test-model"

    @patch("rag_assistant.rag.embeddings.model.SentenceTransformer", MockSentenceTransformer)
    def test_dimensions_from_model(self) -> None:
        model = LocalEmbeddingModel("test-model")
        assert model.dimensions == 384

    @patch("rag_assistant.rag.embeddings.model.SentenceTransformer", MockSentenceTransformer)
    def test_embed_query_returns_list_of_floats(self) -> None:
        model = LocalEmbeddingModel("test-model")
        embedding = model.embed_query("test query")
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    @patch("rag_assistant.rag.embeddings.model.SentenceTransformer", MockSentenceTransformer)
    def test_embed_documents_returns_list_of_lists(self) -> None:
        model = LocalEmbeddingModel("test-model")
        embeddings = model.embed_documents(["doc1", "doc2", "doc3"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    @patch("rag_assistant.rag.embeddings.model.SentenceTransformer", MockSentenceTransformer)
    def test_embed_documents_empty_list(self) -> None:
        model = LocalEmbeddingModel("test-model")
        embeddings = model.embed_documents([])
        assert embeddings == []

    @patch("rag_assistant.rag.embeddings.model.SentenceTransformer", MockSentenceTransformer)
    def test_repr(self) -> None:
        model = LocalEmbeddingModel("test-model", device="cuda")
        repr_str = repr(model)
        assert "test-model" in repr_str
        assert "cuda" in repr_str
        assert "384" in repr_str


class TestCreateEmbeddingModel:
    """Tests for the factory function."""

    @patch("rag_assistant.rag.embeddings.model.SentenceTransformer", MockSentenceTransformer)
    def test_creates_model_from_settings(self) -> None:
        settings = Settings(
            embedding_model="test-model",
            embedding_device="cpu",
            groq_api_key="test-key",
        )
        model = create_embedding_model(settings)
        assert model.model_name == "test-model"

    @patch("rag_assistant.rag.embeddings.model.SentenceTransformer", MockSentenceTransformer)
    def test_uses_default_settings(self) -> None:
        settings = Settings(groq_api_key="test-key")
        model = create_embedding_model(settings)
        assert model.model_name == "all-MiniLM-L6-v2"


class TestEmbeddingModelProtocol:
    """Tests that LocalEmbeddingModel satisfies the Protocol."""

    @patch("rag_assistant.rag.embeddings.model.SentenceTransformer", MockSentenceTransformer)
    def test_satisfies_protocol(self) -> None:
        model = LocalEmbeddingModel("test-model")
        # These should not raise TypeErrors if the protocol is satisfied
        _ = model.dimensions
        _ = model.model_name
        _ = model.embed_query("test")
        _ = model.embed_documents(["test"])
