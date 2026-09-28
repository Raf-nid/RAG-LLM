# Architecture

This document describes the high-level architecture of the AI Engineering Learning Assistant.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐                         │
│  │   Streamlit UI      │    │    API Clients      │                         │
│  │   (future)          │    │    (REST/HTTP)      │                         │
│  └──────────┬──────────┘    └──────────┬──────────┘                         │
│             └──────────────┬───────────┘                                    │
└────────────────────────────┼────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │   FastAPI                                                            │    │
│  │   • /api/v1/chat    — conversational endpoint                       │    │
│  │   • /api/v1/search  — documentation search                          │    │
│  │   • /api/v1/health  — health check                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                    │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
│  │   RAG Service  │    │  Agent Service │    │  Quiz Service  │            │
│  │                │    │   (LangGraph)  │    │    (future)    │            │
│  └───────┬────────┘    └───────┬────────┘    └────────────────┘            │
│          └─────────────────────┴──────────────────┐                         │
└────────────────────────────────────────────────────┼────────────────────────┘
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOMAIN LAYER                                       │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │   RAG Pipeline                                                      │     │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │     │
│  │   │  Retrieval  │→ │  Reranking  │→ │   Context   │→ │Generation│  │     │
│  │   │   (hybrid)  │  │ (optional)  │  │  Selection  │  │ + Citations│ │     │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘  │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │   LLM Abstraction                                                   │     │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │     │
│  │   │   Chat      │  │  Structured │  │    Tool     │                │     │
│  │   │             │  │   Output    │  │   Calling   │                │     │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │   Tools                                                             │     │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │     │
│  │   │ Calculator  │  │  Doc Search │  │   Future    │                │     │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                │     │
│  └────────────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE LAYER                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │     Qdrant     │  │   PostgreSQL   │  │     Redis      │                 │
│  │  (vectors +    │  │ (conversations │  │   (caching,    │                 │
│  │   metadata)    │  │  + app state)  │  │  rate limits)  │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │     Groq       │  │    Ollama      │  │   LangSmith    │                 │
│  │  (cloud LLM)   │  │  (local LLM)   │  │ (observability)│                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐                                     │
│  │  Embedding     │  │   Reranker     │                                     │
│  │   (local)      │  │   (local)      │                                     │
│  └────────────────┘  └────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Layers

### API Layer (FastAPI)

- HTTP endpoints with versioning (`/api/v1/`)
- Request/response validation with Pydantic
- No business logic — delegates to application layer
- Handles authentication, rate limiting (future)

### Application Layer (Services)

- Orchestrates domain components
- Manages transactions and error handling
- Converts between API models and domain models
- No direct infrastructure access

### Domain Layer

- RAG pipeline: retrieval, reranking, generation
- LLM abstraction: provider-agnostic interface
- Tools: calculator, doc search, future tools
- Evaluation: metrics, datasets, runners

### Infrastructure Layer

- Vector store (Qdrant): embeddings and semantic search
- Relational database (PostgreSQL): application state
- Cache (Redis): performance optimization (future)
- LLM providers: Groq (cloud), Ollama (local)
- Observability: LangSmith tracing

## Key Design Principles

1. **Framework as implementation detail**: LangChain/LangGraph do not leak into API or application layers
2. **Provider abstraction**: Application code is agnostic to Groq vs Ollama
3. **Typed boundaries**: All interfaces use Pydantic models
4. **Testability**: Each layer can be tested in isolation
5. **Observable**: LangSmith tracing at critical points

## Component Documentation

- **RAG System**: See [`rag.md`](rag.md)
- **Evaluation Framework**: See [`evaluation.md`](evaluation.md)
- **Agent Workflow**: See [`agents.md`](agents.md) (future)
- **API Reference**: See [`api.md`](api.md) (future)

## Decisions

See [`decisions/`](decisions/) for Architecture Decision Records:

- [ADR 001: Project structure and tooling](decisions/001-project-structure.md)
- [ADR 002: Qdrant as vector store](decisions/002-qdrant-vector-store.md)
- [ADR 003: Local embedding models](decisions/003-local-embeddings.md)
- [ADR 004: Hybrid retrieval strategy](decisions/004-hybrid-retrieval.md)
- [ADR 005: Reranking strategy](decisions/005-reranking-strategy.md)
- [ADR 006: Chunk metadata schema](decisions/006-chunk-metadata-schema.md)
