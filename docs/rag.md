# RAG Architecture Design

This document describes the Retrieval-Augmented Generation (RAG) architecture for the AI Engineering Learning Assistant.

## Overview

The RAG system enables the assistant to answer technical questions grounded in official documentation. It retrieves relevant documentation chunks, ranks them by relevance, and uses them as context for generating accurate, cited answers.

## Knowledge Base

The initial knowledge base contains official documentation for:

- LangChain
- LangGraph
- FastAPI
- Pydantic
- Qdrant
- Groq
- Ollama
- Docker

## Architecture Diagrams

### Offline Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Document   │    │   Document   │    │   Document   │                  │
│  │   Sources    │    │    Loader    │    │    Parser    │                  │
│  │              │───►│              │───►│              │                  │
│  │ (URLs, files)│    │ (fetch HTML) │    │  (extract    │                  │
│  └──────────────┘    └──────────────┘    │   content)   │                  │
│                                          └──────┬───────┘                  │
│                                                 │                          │
│                                                 ▼                          │
│                                          ┌──────────────┐                  │
│                                          │   Cleaner    │                  │
│                                          │              │                  │
│                                          │ (normalize,  │                  │
│                                          │  strip HTML) │                  │
│                                          └──────┬───────┘                  │
│                                                 │                          │
│                                                 ▼                          │
│                                          ┌──────────────┐                  │
│                                          │   Chunker    │                  │
│                                          │              │                  │
│                                          │ (split by    │                  │
│                                          │  sections)   │                  │
│                                          └──────┬───────┘                  │
│                                                 │                          │
│                                                 ▼                          │
│                                          ┌──────────────┐                  │
│                                          │  Metadata    │                  │
│                                          │  Enricher    │                  │
│                                          │              │                  │
│                                          │ (source, URL │                  │
│                                          │  title, etc) │                  │
│                                          └──────┬───────┘                  │
│                                                 │                          │
│                                                 ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Embedding  │    │   Qdrant     │    │  Indexed     │                  │
│  │    Model     │───►│   Indexer    │───►│  Chunks      │                  │
│  │              │    │              │    │              │                  │
│  │ (local, free)│    │ (vectors +   │    │ (searchable) │                  │
│  └──────────────┘    │  metadata)   │    └──────────────┘                  │
│                      └──────────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Online Query Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             QUERY PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                          │
│  │  User Query  │                                                          │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                          │
│  │    Query     │  (optional: rewrite for better retrieval)                │
│  │   Analyzer   │                                                          │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────┐                           │
│  │           HYBRID RETRIEVAL                   │                           │
│  │  ┌────────────┐         ┌────────────┐      │                           │
│  │  │   Dense    │         │  Lexical   │      │                           │
│  │  │ Retrieval  │         │ Retrieval  │      │                           │
│  │  │            │         │   (BM25)   │      │                           │
│  │  │ (semantic) │         │ (keyword)  │      │                           │
│  │  └─────┬──────┘         └─────┬──────┘      │                           │
│  │        │                      │             │                           │
│  │        └──────────┬───────────┘             │                           │
│  │                   ▼                         │                           │
│  │            ┌────────────┐                   │                           │
│  │            │   Fusion   │                   │                           │
│  │            │ (RRF/merge)│                   │                           │
│  │            └────────────┘                   │                           │
│  └─────────────────────┬───────────────────────┘                           │
│                        │                                                    │
│                        ▼                                                    │
│                 ┌────────────┐                                              │
│                 │  Reranker  │  (cross-encoder for precision)              │
│                 └──────┬─────┘                                              │
│                        │                                                    │
│                        ▼                                                    │
│                 ┌────────────┐                                              │
│                 │  Context   │  (select top-k for LLM context)             │
│                 │  Selector  │                                              │
│                 └──────┬─────┘                                              │
│                        │                                                    │
│                        ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                    GENERATION                                 │          │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────┐          │          │
│  │  │  Prompt    │───►│    LLM     │───►│  Response  │          │          │
│  │  │ Constructor│    │   (Groq/   │    │   Parser   │          │          │
│  │  │            │    │   Ollama)  │    │            │          │          │
│  │  │ (context + │    │            │    │ (extract   │          │          │
│  │  │  question) │    │            │    │  answer +  │          │          │
│  │  └────────────┘    └────────────┘    │  citations)│          │          │
│  │                                      └────────────┘          │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                        │                                                    │
│                        ▼                                                    │
│                 ┌────────────┐                                              │
│                 │  Grounded  │                                              │
│                 │   Answer   │                                              │
│                 │     +      │                                              │
│                 │ Citations  │                                              │
│                 └────────────┘                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Document Ingestion

**Purpose**: Transform raw documentation into searchable chunks with rich metadata.

**Components**:

| Component | Responsibility |
|-----------|---------------|
| `DocumentLoader` | Fetch documents from URLs or local files |
| `DocumentParser` | Extract text content from HTML/Markdown |
| `DocumentCleaner` | Normalize whitespace, remove boilerplate |
| `Chunker` | Split documents into semantic chunks |
| `MetadataEnricher` | Add source attribution and structural metadata |

**Chunking Strategy**:

- **Primary**: Section-based chunking (headers as natural boundaries)
- **Fallback**: Recursive character splitting with overlap
- **Target chunk size**: 500-1000 tokens (configurable)
- **Overlap**: 50-100 tokens for context continuity

### 2. Metadata Schema

Every indexed chunk must contain:

```python
@dataclass
class ChunkMetadata:
    # Source identification
    chunk_id: str          # Stable, content-based hash
    source: str            # e.g., "langchain", "fastapi"
    document_type: str     # e.g., "official_docs", "api_reference"
    
    # Citation data
    title: str             # Document/page title
    url: str               # Original source URL
    section: str | None    # Section heading (if available)
    
    # Versioning
    version: str | None    # Documentation version
    indexed_at: datetime   # When this chunk was indexed
    
    # Structural
    chunk_index: int       # Position within the document
    total_chunks: int      # Total chunks from this document
```

### 3. Vector Storage (Qdrant)

**What goes in Qdrant**:

- Document chunk text
- Dense embeddings (768-1536 dimensions depending on model)
- Sparse embeddings (for hybrid search, if supported)
- All chunk metadata as payload fields

**Collection Configuration**:

```python
{
    "vectors": {
        "dense": {
            "size": 768,  # sentence-transformers default
            "distance": "Cosine"
        }
    },
    "sparse_vectors": {  # For hybrid search
        "bm25": {}
    },
    "payload_schema": {
        "source": "keyword",      # Filterable
        "document_type": "keyword",
        "indexed_at": "datetime"
    }
}
```

### 4. Embedding Model

**Requirements**:

- Local (no API costs)
- Free to use
- Good performance on technical documentation
- Reasonable inference speed

**Candidate Models**:

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, good baseline |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Better quality, slower |
| `BAAI/bge-small-en-v1.5` | 384 | Optimized for retrieval |
| `BAAI/bge-base-en-v1.5` | 768 | Better quality |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Good for code/docs |

**Decision**: Start with `sentence-transformers/all-MiniLM-L6-v2` for speed during development, evaluate `BAAI/bge-base-en-v1.5` for production.

### 5. Retrieval Strategy

**Dense Retrieval**:

- Semantic similarity using cosine distance
- Good for conceptual questions
- May miss exact matches (API names, class names)

**Lexical Retrieval (BM25)**:

- Keyword-based matching
- Excellent for exact terms
- No semantic understanding

**Hybrid Retrieval**:

- Combine dense + lexical results
- Use Reciprocal Rank Fusion (RRF) for score normalization
- Configurable weights

```python
# RRF formula
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

# Combined score
final_score = alpha * dense_rrf + (1 - alpha) * lexical_rrf
```

### 6. Reranking

**Purpose**: Improve precision by re-scoring candidates with a cross-encoder.

**When to use**:

- After hybrid retrieval produces candidates
- Before final context selection
- Cross-encoders are slower but more accurate

**Candidate Models**:

| Model | Notes |
|-------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast, good baseline |
| `BAAI/bge-reranker-base` | Better quality |
| `mixedbread-ai/mxbai-rerank-base-v1` | Good for code/docs |

**Trade-off**: Reranking adds latency (~50-200ms per query). Measure impact before enabling.

### 7. Context Construction

**Prompt Template Structure**:

```
System: You are a technical assistant that answers questions based on documentation.
        Answer ONLY from the provided context. If the context doesn't contain
        the answer, say so clearly.

Context:
[Source 1: {title} - {url}]
{chunk_text}

[Source 2: {title} - {url}]
{chunk_text}

...

User: {question}
```

### 8. Response Format

**Structured Output Schema**:

```python
class Source(BaseModel):
    title: str
    url: str
    chunk_id: str
    relevance_score: float | None = None

class RAGResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_metadata: RetrievalMetadata | None = None

class RetrievalMetadata(BaseModel):
    total_candidates: int
    dense_candidates: int
    lexical_candidates: int
    reranked: bool
    latency_ms: float
```

## Data Flow Summary

### What Goes Where

| Storage | Data | Purpose |
|---------|------|---------|
| **Qdrant** | Vectors, chunk text, metadata | Semantic search |
| **PostgreSQL** | Conversations, user sessions, feedback | Application state |
| **Redis** | Query cache, rate limits | Performance (future) |
| **Filesystem** | Raw documents (optional) | Backup/re-indexing |

### PostgreSQL Schema (Future)

For conversation tracking and learning features:

```sql
-- Conversation history
conversations (id, user_id, created_at, updated_at)
messages (id, conversation_id, role, content, created_at)

-- RAG feedback
rag_feedback (id, message_id, helpful, feedback_text)

-- Learning progress
quiz_sessions (id, user_id, topic, score, created_at)
```

### Redis Usage (Future)

- **Query caching**: Cache retrieval results for repeated queries
- **Rate limiting**: Per-user query limits
- **Session state**: Temporary agent state during multi-turn interactions

**Note**: Redis is not needed for the initial implementation. Add it when caching provides measurable benefit.

## LangChain Integration Points

| Component | LangChain Usage | Custom Implementation |
|-----------|----------------|----------------------|
| Embedding | `langchain-community` embeddings | - |
| Vector store | `langchain-qdrant` | - |
| Document loaders | `langchain-community` loaders | Custom for specific sources |
| Text splitters | `langchain-text-splitters` | Custom section-aware splitter |
| Retriever | `langchain-core` retriever interface | Custom hybrid retriever |
| Reranker | - | Custom (cross-encoder) |
| Prompt templates | `langchain-core` prompts | - |
| Output parsing | `langchain-core` parsers | Pydantic models |

**Principle**: Use LangChain where it provides value (embeddings, vector store integration, prompts). Implement custom logic for domain-specific behavior (hybrid fusion, citation extraction).

## Testability

### Unit Tests

- Chunking produces expected output for known inputs
- Metadata is correctly extracted and preserved
- Stable chunk IDs are deterministic
- Score fusion math is correct

### Integration Tests

- End-to-end retrieval returns relevant chunks
- Citation URLs are valid and preserved
- Metadata filtering works correctly

### Evaluation Tests

- Retrieval metrics on evaluation dataset (Recall@K, MRR)
- Answer quality metrics (faithfulness, relevance)
- Latency benchmarks

## Folder Structure

```
src/rag_assistant/
├── config.py                 # Existing settings
├── llm/                      # Existing LLM providers
│   └── ...
├── tools/                    # Existing tools
│   └── ...
├── schemas/                  # Pydantic models
│   ├── __init__.py
│   ├── question_analysis.py  # Existing
│   ├── chunk.py              # NEW: Chunk and metadata models
│   └── rag_response.py       # NEW: RAG response models
├── rag/                      # NEW: RAG pipeline
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py         # Document loading
│   │   ├── parser.py         # Content extraction
│   │   ├── cleaner.py        # Text normalization
│   │   ├── chunker.py        # Text chunking
│   │   └── metadata.py       # Metadata enrichment
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── model.py          # Embedding model wrapper
│   ├── store/
│   │   ├── __init__.py
│   │   └── qdrant.py         # Qdrant client wrapper
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py          # Dense retrieval
│   │   ├── lexical.py        # BM25 retrieval
│   │   ├── hybrid.py         # Hybrid fusion
│   │   └── reranker.py       # Cross-encoder reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── context.py        # Context construction
│   │   ├── prompt.py         # RAG prompts
│   │   └── response.py       # Response parsing + citations
│   └── pipeline.py           # End-to-end RAG orchestration
├── evaluation/               # NEW: Evaluation framework
│   ├── __init__.py
│   ├── datasets/
│   │   └── retrieval_eval.json
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── retrieval.py      # Recall@K, MRR, etc.
│   │   └── generation.py     # Faithfulness, relevance
│   └── runner.py             # Evaluation orchestration
scripts/
├── ingest_docs.py            # NEW: Run ingestion pipeline
├── query_rag.py              # NEW: Test RAG queries
└── evaluate_rag.py           # NEW: Run evaluation suite
tests/
├── rag/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_metadata.py
│   ├── test_retrieval.py
│   └── test_hybrid.py
└── evaluation/
    └── test_metrics.py
experiments/
├── embeddings/               # Compare embedding models
├── chunking/                 # Chunk size experiments
├── retrieval/                # Dense vs hybrid vs reranking
└── evaluation/               # Evaluation methodology
```

## Experimental Comparison Plan

To compare dense retrieval, hybrid retrieval, and reranking:

### 1. Create Evaluation Dataset

```json
{
  "version": "1.0",
  "questions": [
    {
      "id": "q001",
      "question": "How do I create a LangChain agent with tools?",
      "expected_sources": ["langchain"],
      "expected_concepts": ["agent", "tools", "bind_tools"],
      "difficulty": "intermediate"
    }
  ]
}
```

### 2. Metrics to Track

| Metric | Definition |
|--------|-----------|
| Recall@K | % of relevant docs in top K results |
| MRR | Mean Reciprocal Rank of first relevant doc |
| Latency (p50, p95) | Time to retrieve candidates |
| Answer Faithfulness | Is answer supported by context? |

### 3. Configurations to Compare

| Config | Dense | Lexical | Reranker |
|--------|-------|---------|----------|
| A (baseline) | yes | no | no |
| B (hybrid) | yes | yes | no |
| C (hybrid+rerank) | yes | yes | yes |

### 4. Reproducibility

- Pin random seeds where applicable
- Document exact model versions
- Store results in `experiments/retrieval/`
- Use git tags for evaluation checkpoints

## Interface Definitions

### Core Protocols

```python
from typing import Protocol

class EmbeddingModel(Protocol):
    """Interface for embedding models."""
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        ...
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        ...

class Retriever(Protocol):
    """Interface for document retrievers."""
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        ...

class Reranker(Protocol):
    """Interface for reranking models."""
    
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank chunks by relevance to query."""
        ...
```

### Data Models

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    source: str
    title: str
    url: str
    section: str | None
    document_type: str
    version: str | None
    indexed_at: datetime
    chunk_index: int
    total_chunks: int

@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of text with metadata."""
    text: str
    metadata: ChunkMetadata

@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from retrieval with score."""
    chunk: DocumentChunk
    score: float
    retrieval_method: str  # "dense", "lexical", "hybrid"
```

## Security Considerations

- **No arbitrary code execution** from retrieved content
- **URL validation** before storing in metadata
- **Input sanitization** for user queries
- **Rate limiting** for retrieval endpoints (future)
- **No sensitive data** in indexed documents

## Next Steps

1. **Step 8**: Implement basic dense retrieval (documents → chunks → embeddings → Qdrant → retrieval)
2. **Step 9**: Add grounded generation with citations
3. **Step 10**: Implement hybrid retrieval
4. **Step 11**: Add reranking
5. **Step 12**: Build evaluation framework with LangSmith

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Sentence Transformers](https://www.sbert.net/)
- [Hybrid Search Explained](https://qdrant.tech/articles/hybrid-search/)