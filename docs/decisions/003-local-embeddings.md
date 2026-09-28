# ADR 003 — Local Embedding Models (No OpenAI)

## Context

The RAG system requires an embedding model to convert text into vectors for semantic search. Per project requirements:

- **No OpenAI dependencies** — the project uses Groq and Ollama only
- **Local and free** — embeddings must run locally without API costs
- **Good quality** — must work well on technical documentation

## Problem

Which embedding model should we use for the RAG system?

## Alternatives

| Model | Dimensions | Size | Performance | Notes |
|-------|-----------|------|-------------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 80MB | Good baseline | Very fast |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 420MB | Better quality | Slower |
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | Excellent for retrieval | Optimized for search |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Better quality | State-of-the-art |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 550MB | Good for code/docs | Longer context |
| `text-embedding-ada-002` (OpenAI) | 1536 | - | Excellent | **Not allowed** |

## Decision

Use **`sentence-transformers/all-MiniLM-L6-v2`** for initial development, with a plan to evaluate **`BAAI/bge-base-en-v1.5`** for production.

### Rationale

1. **Speed during development**: MiniLM is ~5x faster than larger models, accelerating iteration.
2. **Good enough baseline**: 384-dimensional embeddings are sufficient to validate the RAG pipeline.
3. **Easy upgrade path**: Switching models requires only re-indexing documents.
4. **No external dependencies**: Runs locally via `sentence-transformers` library.
5. **BGE as upgrade**: BAAI's BGE models consistently rank highest on retrieval benchmarks (MTEB).

### Why not start with BGE?

BGE is better, but:

- Slower inference (2-3x)
- Larger index size (768 vs 384 dimensions = 2x storage)
- For a learning project, fast iteration is more valuable than marginal quality gains

We will measure retrieval quality with evaluation datasets and upgrade if needed.

## Consequences

### Positive

- Zero API costs for embeddings
- Fast local inference
- Full control over embedding model
- Can experiment with different models
- Works offline

### Negative

- Lower quality than proprietary embeddings (OpenAI, Cohere)
- Requires local compute (CPU is fine for development)
- Need to download models (~80-500MB)
- Must re-index when changing models

### Implementation

```python
from sentence_transformers import SentenceTransformer

class LocalEmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
```

### Configuration

```python
# In config.py (to be added)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu  # or "cuda" if available
```

### Dependencies

```toml
# In pyproject.toml
dependencies = [
    # ... existing ...
    "sentence-transformers>=3.0",
]
```

## Evaluation Plan

Compare models using the retrieval evaluation dataset:

| Model | Recall@5 | MRR | Latency (ms) |
|-------|----------|-----|--------------|
| MiniLM | TBD | TBD | TBD |
| BGE-base | TBD | TBD | TBD |

Upgrade to BGE if it shows >10% improvement on Recall@5.

## Related Decisions

- ADR 002: Qdrant as vector store
- ADR 004: Hybrid retrieval strategy
