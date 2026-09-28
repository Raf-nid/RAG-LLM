# ADR 002 — Qdrant as Vector Store

## Context

The RAG system requires a vector database to store document embeddings and enable semantic search. We need a solution that supports:

- Dense vector similarity search (cosine, dot product, euclidean)
- Metadata storage and filtering
- Hybrid search (dense + sparse vectors)
- Reasonable performance for a learning project
- Free/open-source deployment option

## Problem

Which vector database should we use for the RAG system?

## Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **Qdrant** | Native hybrid search, rich filtering, gRPC + REST, Docker-friendly, good Python SDK, open source | Newer than Pinecone/Weaviate |
| Pinecone | Managed, scalable, popular | Paid, no self-hosted option |
| Weaviate | GraphQL API, modules for embeddings | More complex setup, heavier |
| Chroma | Simple, in-process | Less mature, limited filtering |
| pgvector | PostgreSQL extension, no new infra | Slower, less feature-rich |
| Milvus | Scalable, feature-rich | Complex setup, heavy for learning project |

## Decision

Use **Qdrant** as the vector store.

Reasons:

1. **Hybrid search support**: Native sparse vectors for BM25-style retrieval alongside dense vectors.
2. **Rich filtering**: Payload filtering with complex conditions (essential for source/document_type filters).
3. **Good Python SDK**: `qdrant-client` is well-maintained with sync/async support.
4. **LangChain integration**: `langchain-qdrant` provides a ready-to-use retriever.
5. **Docker-friendly**: Single container deployment for local development.
6. **Free**: Open source with generous cloud tier if needed.
7. **Learning value**: Understanding Qdrant is directly applicable to production AI systems.

## Consequences

### Positive

- Clean separation between vector storage (Qdrant) and relational data (PostgreSQL).
- Built-in support for the hybrid retrieval strategy.
- Straightforward local development with Docker.
- Payload filtering enables efficient source-based queries.

### Negative

- Another service to run locally (Docker container).
- Learning curve for Qdrant-specific concepts (collections, points, payloads).
- Must design idempotent ingestion to avoid duplicate points.

### Configuration

```python
# In config.py (already present)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=           # Optional for local
QDRANT_COLLECTION_NAME=rag_assistant
```

### Collection Design

```python
# Collection with dense and sparse vectors
{
    "name": "rag_assistant",
    "vectors": {
        "dense": {
            "size": 768,      # Depends on embedding model
            "distance": "Cosine"
        }
    },
    "sparse_vectors": {
        "sparse": {}          # For BM25/SPLADE
    }
}
```

## Related Decisions

- ADR 003: Local embedding model selection
- ADR 004: Hybrid retrieval strategy
