"""
Embedding model wrapper.

Provides a unified interface for embedding models, currently backed by
sentence-transformers. The interface is kept simple and provider-agnostic
so that alternative implementations (e.g., Ollama embeddings) can be
added later.

Key Concepts
------------
**Embedding**: A fixed-size vector representation of text. Similar texts
produce similar vectors, enabling semantic search via vector similarity.

**Sentence Transformers**: A library that provides pre-trained models
optimized for producing sentence/paragraph embeddings. Unlike token-level
embeddings from BERT, these are designed for comparing full texts.

**Model Selection**:
- `all-MiniLM-L6-v2`: 384 dims, fast, good for development
- `all-mpnet-base-v2`: 768 dims, better quality
- `BAAI/bge-base-en-v1.5`: 768 dims, state-of-the-art retrieval

Usage
-----
    from rag_assistant.rag.embeddings import create_embedding_model
    from rag_assistant.config import settings

    model = create_embedding_model(settings)

    # Single text
    vector = model.embed_query("What is RAG?")

    # Batch of documents
    vectors = model.embed_documents(["Doc 1...", "Doc 2..."])
"""

from typing import Protocol

from sentence_transformers import SentenceTransformer

from rag_assistant.config import Settings


class EmbeddingModel(Protocol):
    """
    Protocol for embedding models.

    Any class implementing these methods can be used for embeddings.
    This allows for easy testing with mocks and future provider additions.
    """

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Parameters
        ----------
        text
            The query text to embed.

        Returns
        -------
        list[float]
            Embedding vector as a list of floats.
        """
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents in a batch.

        Parameters
        ----------
        texts
            List of document texts to embed.

        Returns
        -------
        list[list[float]]
            List of embedding vectors.
        """
        ...


class LocalEmbeddingModel:
    """
    Local embedding model backed by sentence-transformers.

    This model runs entirely on the local machine (CPU or GPU),
    with no external API calls. The model is downloaded from
    HuggingFace on first use and cached locally.

    Parameters
    ----------
    model_name
        Name of the sentence-transformers model.
        Examples: "all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"
    device
        Device to run inference on: "cpu", "cuda", or "mps".
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model = SentenceTransformer(model_name, device=device)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        if self._dimensions is None:
            raise RuntimeError("Model dimensions not available")
        return int(self._dimensions)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        For models like BGE that require a query prefix, this method
        would add it. For standard models, it's equivalent to embedding
        a single document.
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        result: list[float] = embedding.tolist()
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents in a batch.

        Batching is more efficient than embedding one at a time
        due to GPU parallelism and reduced overhead.
        """
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        result: list[list[float]] = embeddings.tolist()
        return result

    def __repr__(self) -> str:
        return (
            f"LocalEmbeddingModel(model_name={self._model_name!r}, "
            f"device={self._device!r}, dimensions={self._dimensions})"
        )


def create_embedding_model(settings: Settings) -> LocalEmbeddingModel:
    """
    Create an embedding model from application settings.

    Parameters
    ----------
    settings
        Application settings containing EMBEDDING_MODEL and EMBEDDING_DEVICE.

    Returns
    -------
    LocalEmbeddingModel
        Configured embedding model ready for use.
    """
    return LocalEmbeddingModel(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
