# AI Engineering Learning Assistant

A production-oriented RAG application built to demonstrate modern AI Engineering practices while serving as a learning tool for AI Engineering concepts.

## Purpose

This repository has two goals:

1. Build a serious, production-oriented AI Engineering application.
2. Develop genuine proficiency in the technologies and concepts used to build it.

The final system is intended to be a credible AI Engineer portfolio project.

## Architecture overview

```
User query
    ↓
FastAPI
    ↓
LLM provider abstraction (Groq / Ollama)
    ↓
RAG pipeline
    ├── Hybrid retrieval (dense + lexical)
    ├── Reranking
    └── Context selection
    ↓
LLM (grounded answer + citations)
    ↓
PostgreSQL (conversation state)
    Qdrant (vector store)
    LangSmith (observability)
```

## Stack

| Layer              | Technology                          |
|--------------------|-------------------------------------|
| Language           | Python 3.12+                        |
| Package manager    | uv                                  |
| API                | FastAPI                             |
| LLM framework      | LangChain, LangGraph                |
| LLM providers      | Groq, Ollama                        |
| Vector store       | Qdrant                              |
| Relational DB      | PostgreSQL                          |
| Cache              | Redis                               |
| Observability      | LangSmith                           |
| Frontend           | Streamlit                           |
| Testing            | pytest                              |
| Linting/Formatting | Ruff                                |
| Containers         | Docker + Docker Compose             |
| CI                 | GitHub Actions                      |

## Getting started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd rag-assistant

# Create a virtual environment and install all dependencies
uv sync --extra dev

# Configure your environment
cp .env.example .env
# Edit .env and set at minimum:
#   LLM_PROVIDER=groq
#   GROQ_API_KEY=<your-key>
```

### Running the tests

```bash
uv run pytest
```

### Linting and formatting

```bash
# Check
uv run ruff check .
uv run ruff format --check .

# Fix
uv run ruff check --fix .
uv run ruff format .
```

### Type checking

```bash
uv run mypy src/
```

## Project structure

```
.
├── src/
│   └── rag_assistant/
│       ├── __init__.py
│       └── config.py          # Typed settings (pydantic-settings)
├── tests/
│   └── test_config.py
├── docs/
│   ├── architecture.md
│   └── decisions/             # Architecture Decision Records
├── experiments/               # Reproducible experiments
├── .env.example               # Environment variable template
├── pyproject.toml             # Project config, dependencies, tool settings
└── README.md
```

## Environment variables

See [`.env.example`](.env.example) for the full list of supported variables with descriptions.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system architecture
- [`docs/decisions/`](docs/decisions/) — Architecture Decision Records (ADRs)

## License

MIT
