"""
Documentation search tool (placeholder).

This module defines the tool schema and interface that the agent will use
to search the knowledge base.  The actual Qdrant retrieval logic has not
been implemented yet — that happens when we build the RAG pipeline.

Why implement a placeholder now?
---------------------------------
1. The tool schema is part of the LLM's context.  Defining it now lets us
   test the tool calling mechanism end-to-end without Qdrant.
2. It documents the contract the real implementation must satisfy.
3. The LLM can already "see" this tool and generate tool call requests for it.

When the RAG pipeline is implemented, replace the body of ``execute()``
with a call to the Qdrant retriever.  The schema, name, and description
should remain stable so that tool call generation from the model is
consistent.
"""

from pydantic import BaseModel, Field

from .base import BaseTool

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

SUPPORTED_SOURCES = [
    "langchain",
    "langgraph",
    "fastapi",
    "pydantic",
    "qdrant",
    "groq",
    "docker",
]


class DocSearchInput(BaseModel):
    """Input arguments for the documentation search tool."""

    query: str = Field(
        description=(
            "The search query. Should be a clear, specific question or "
            "keyword phrase about the technology you want to learn about."
        ),
        min_length=1,
    )
    source: str | None = Field(
        default=None,
        description=(
            "Optional: filter results to a specific documentation source. "
            f"Supported values: {', '.join(SUPPORTED_SOURCES)}. "
            "Leave empty to search all sources."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve (1-20).",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class DocSearchTool(BaseTool):
    """
    Search the technical documentation knowledge base.

    Retrieves relevant documentation chunks from the vector store.
    Currently returns a placeholder response — Qdrant integration pending.
    """

    name = "doc_search"
    description = (
        "Search the technical documentation knowledge base for information about "
        "LangChain, LangGraph, FastAPI, Pydantic, Qdrant, Groq, or Docker. "
        "Use this tool to answer questions grounded in official documentation. "
        "Returns relevant text excerpts with source attribution."
    )
    input_schema = DocSearchInput

    def execute(self, args: BaseModel) -> str:
        """
        Search the knowledge base and return relevant excerpts.

        Currently returns a placeholder response.
        This method will be replaced when Qdrant is integrated.
        """
        assert isinstance(args, DocSearchInput)

        # Validate source filter against known sources.
        if args.source is not None and args.source.lower() not in SUPPORTED_SOURCES:
            sources_list = ", ".join(SUPPORTED_SOURCES)
            return f"Unknown source '{args.source}'. Supported sources: {sources_list}."

        # Placeholder: return a clearly marked stub response.
        source_note = f" (filtered to source: {args.source})" if args.source else ""
        return (
            f"[PLACEHOLDER] Documentation search not yet implemented{source_note}. "
            f"Query: '{args.query}', top_k: {args.top_k}. "
            "This tool will return real documentation chunks once Qdrant is integrated."
        )
