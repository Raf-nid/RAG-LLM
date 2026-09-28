"""
Application configuration.

All settings are loaded from environment variables (or a .env file).
Never use os.getenv() elsewhere in the codebase — import `settings` here.

Usage:
    from rag_assistant.config import settings

    print(settings.environment)
    print(settings.llm_provider)
"""

from enum import StrEnum

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment."""

    development = "development"
    staging = "staging"
    production = "production"


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    groq = "groq"
    ollama = "ollama"


class Settings(BaseSettings):
    """
    Typed application settings backed by environment variables.

    Every field maps directly to an environment variable of the same name
    (uppercase). Missing required variables raise a ValidationError at startup,
    failing fast rather than silently using wrong defaults at runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars — do not crash on them
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    environment: Environment = Field(
        default=Environment.development,
        description="Deployment environment (development | staging | production).",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG | INFO | WARNING | ERROR).",
    )

    # ------------------------------------------------------------------
    # LLM providers
    # ------------------------------------------------------------------
    llm_provider: LLMProvider = Field(
        default=LLMProvider.groq,
        description="Active LLM provider (groq | ollama).",
    )
    llm_model: str = Field(
        default="qwen/qwen3.8-27b",
        description="Model name for the active provider.",
    )

    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Groq API key. Required when LLM_PROVIDER=groq.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL. Used when LLM_PROVIDER=ollama.",
    )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name for embeddings.",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embedding inference (cpu | cuda | mps).",
    )

    # ------------------------------------------------------------------
    # Vector store (Qdrant)
    # ------------------------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant instance URL. Use ':memory:' for in-memory mode.",
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API key. Optional for local instances.",
    )
    qdrant_collection_name: str = Field(
        default="rag_assistant",
        description="Qdrant collection used for document storage.",
    )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=4000,
        description="Target chunk size in characters.",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between chunks in characters.",
    )

    # ------------------------------------------------------------------
    # Relational database
    # ------------------------------------------------------------------
    postgres_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/rag_assistant",
        description="PostgreSQL connection URL (asyncpg driver).",
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith API key for tracing and evaluation.",
    )
    langsmith_project: str = Field(
        default="rag-assistant",
        description="LangSmith project name.",
    )
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing. Requires LANGSMITH_API_KEY.",
    )

    @field_validator(
        "groq_api_key",
        "qdrant_api_key",
        "langsmith_api_key",
        mode="before",
    )
    @classmethod
    def empty_secret_to_none(cls, value: object) -> object:
        """Treat blank env vars as unset, not as an empty secret."""
        if value == "":
            return None
        return value


# ---------------------------------------------------------------------------
# Module-level singleton.
# Loaded once at import time. Tests can override via monkeypatch or
# by setting environment variables before importing.
# ---------------------------------------------------------------------------
settings = Settings()
