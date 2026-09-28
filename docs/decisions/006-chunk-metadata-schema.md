# ADR 006 — Chunk Metadata Schema

## Context

Each document chunk in the RAG system needs metadata for:

- **Citation**: Displaying source attribution in answers
- **Filtering**: Limiting search to specific sources/types
- **Debugging**: Understanding retrieval results
- **Idempotency**: Avoiding duplicate chunks during re-indexing

## Problem

What metadata should we store with each chunk, and how should we structure it?

## Decision

Define a structured metadata schema with required and optional fields.

### Required Fields

| Field | Type | Purpose |
|-------|------|---------|
| `chunk_id` | `str` | Unique, stable identifier (content hash) |
| `source` | `str` | Documentation source (e.g., "langchain", "fastapi") |
| `title` | `str` | Document or page title |
| `url` | `str` | Original source URL |
| `document_type` | `str` | Type classification (e.g., "official_docs", "api_reference") |
| `indexed_at` | `datetime` | Timestamp when chunk was indexed |

### Optional Fields

| Field | Type | Purpose |
|-------|------|---------|
| `section` | `str \| None` | Section heading (if available) |
| `version` | `str \| None` | Documentation version |
| `chunk_index` | `int` | Position within document |
| `total_chunks` | `int` | Total chunks from document |
| `parent_url` | `str \| None` | Parent page URL (for nested docs) |
| `language` | `str` | Content language (default: "en") |

### Stable Chunk IDs

Chunk IDs must be **deterministic** and **content-based** to support idempotent re-indexing:

```python
import hashlib

def generate_chunk_id(source: str, url: str, text: str) -> str:
    """Generate a stable, content-based chunk ID."""
    content = f"{source}:{url}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

This ensures:

- Same content → same ID (idempotent updates)
- Different content → different ID (no collisions)
- ID is short enough to be readable in logs

### Source Values

Standardized source identifiers:

| Source | Value |
|--------|-------|
| LangChain | `langchain` |
| LangGraph | `langgraph` |
| FastAPI | `fastapi` |
| Pydantic | `pydantic` |
| Qdrant | `qdrant` |
| Groq | `groq` |
| Ollama | `ollama` |
| Docker | `docker` |

### Document Types

| Type | Description |
|------|-------------|
| `official_docs` | Official documentation pages |
| `api_reference` | API reference pages |
| `tutorial` | Tutorial or guide content |
| `example` | Code examples |
| `changelog` | Version changelogs |

## Consequences

### Positive

- Clear contract for citation display
- Efficient filtering by source and type
- Idempotent ingestion (no duplicates)
- Good debugging information

### Negative

- Must extract all metadata during ingestion
- Schema changes require re-indexing

### Implementation

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata for a document chunk."""
    
    # Required
    chunk_id: str
    source: str
    title: str
    url: str
    document_type: str
    indexed_at: datetime
    
    # Optional
    section: str | None = None
    version: str | None = None
    chunk_index: int = 0
    total_chunks: int = 1
    parent_url: str | None = None
    language: str = "en"
    
    def to_qdrant_payload(self) -> dict:
        """Convert to Qdrant payload format."""
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
            "parent_url": self.parent_url,
            "language": self.language,
        }
```

### Qdrant Payload Indexing

```python
# Fields to index for filtering
indexed_fields = {
    "source": "keyword",        # Exact match filter
    "document_type": "keyword", # Exact match filter
    "indexed_at": "datetime",   # Range filter (for freshness)
}
```

## Related Decisions

- ADR 002: Qdrant as vector store
- ADR 004: Hybrid retrieval strategy
