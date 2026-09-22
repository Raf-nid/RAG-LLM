# Architecture

This document describes the high-level architecture of the AI Engineering Learning Assistant.

## Layers

```
API (FastAPI)
    ↓
Application / orchestration (services)
    ↓
Domain logic (RAG pipeline, agent, evaluation)
    ↓
Infrastructure (Qdrant, PostgreSQL, Redis, LLM providers)
```

Framework objects (LangChain, LangGraph) are implementation details of the infrastructure and domain layers. They do not leak into the API or application layers.

## Components

*To be expanded as features are implemented.*

## Decisions

See [`decisions/`](decisions/) for Architecture Decision Records.
