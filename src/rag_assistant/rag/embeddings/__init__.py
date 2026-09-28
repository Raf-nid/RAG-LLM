"""Embedding model implementations."""

from rag_assistant.rag.embeddings.model import (
    EmbeddingModel,
    LocalEmbeddingModel,
    create_embedding_model,
)

__all__ = [
    "EmbeddingModel",
    "LocalEmbeddingModel",
    "create_embedding_model",
]
