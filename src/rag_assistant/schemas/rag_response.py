"""
RAG response schemas.

These models define the structured outputs from the RAG pipeline,
including the grounded answer and source citations.

Design Principles
-----------------
- Answers must always include source citations
- Response includes retrieval metadata for observability
- Models are suitable for both API responses and internal use

Usage
-----
The RAG pipeline returns:

    response = RAGResponse(
        answer="LangChain is a framework for building LLM applications...",
        sources=[
            Source(title="Introduction", url="https://...", chunk_id="abc123"),
        ],
        retrieval_metadata=RetrievalMetadata(
            total_candidates=20,
            dense_candidates=10,
            lexical_candidates=10,
            reranked=True,
            latency_ms=150.5,
        ),
    )
"""

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    """
    A source citation for a RAG response.

    Each source represents a document chunk that contributed
    to the answer. The URL enables users to verify the source.
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(
        description="Document or section title.",
        min_length=1,
    )
    url: str = Field(
        description="URL to the source document.",
        min_length=1,
    )
    chunk_id: str = Field(
        description="Unique identifier of the source chunk.",
        min_length=1,
    )
    section: str | None = Field(
        default=None,
        description="Section within the document (if available).",
    )
    relevance_score: float | None = Field(
        default=None,
        description="Retrieval/reranking score (higher is more relevant).",
    )


class RetrievalMetadata(BaseModel):
    """
    Metadata about the retrieval process.

    Useful for debugging, evaluation, and observability.
    Tracks the number of candidates from each retrieval method
    and the total latency.
    """

    model_config = ConfigDict(frozen=True)

    total_candidates: int = Field(
        description="Total number of candidate chunks retrieved.",
        ge=0,
    )
    dense_candidates: int = Field(
        default=0,
        description="Number of candidates from dense retrieval.",
        ge=0,
    )
    lexical_candidates: int = Field(
        default=0,
        description="Number of candidates from lexical retrieval.",
        ge=0,
    )
    reranked: bool = Field(
        default=False,
        description="Whether reranking was applied.",
    )
    latency_ms: float = Field(
        description="Total retrieval latency in milliseconds.",
        ge=0,
    )


class RAGResponse(BaseModel):
    """
    Complete response from the RAG pipeline.

    Contains the generated answer, source citations, and
    optional metadata about the retrieval process.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(
        description="The generated answer grounded in retrieved sources.",
        min_length=1,
    )
    sources: list[Source] = Field(
        description="Sources that support the answer.",
        min_length=0,
    )
    retrieval_metadata: RetrievalMetadata | None = Field(
        default=None,
        description="Metadata about the retrieval process.",
    )

    @property
    def has_sources(self) -> bool:
        """Check if the response has any sources."""
        return len(self.sources) > 0

    @property
    def source_count(self) -> int:
        """Number of sources cited."""
        return len(self.sources)


class InsufficientContextResponse(BaseModel):
    """
    Response when retrieved context is insufficient.

    Returned when the RAG system cannot find relevant information
    to answer the question, instead of hallucinating.
    """

    model_config = ConfigDict(frozen=True)

    message: str = Field(
        default=(
            "I could not find relevant information in the documentation to answer this question."
        ),
        description="User-friendly message explaining the limitation.",
    )
    query: str = Field(
        description="The original user query.",
    )
    candidates_found: int = Field(
        default=0,
        description="Number of candidates that were retrieved (but deemed irrelevant).",
        ge=0,
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggestion for how to rephrase the query or where to look.",
    )
