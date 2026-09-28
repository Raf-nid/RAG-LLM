"""
Grounded RAG generator with citations.

This module implements the generation step of the RAG pipeline:
1. Takes retrieved chunks as context
2. Calls the LLM with grounding instructions
3. Parses the structured response
4. Maps citations to source chunks
5. Returns a RAGResponse or InsufficientContextResponse

Design Principles
-----------------
**No Hallucination**: The generator explicitly handles cases where the
context is insufficient, returning an InsufficientContextResponse rather
than making up information.

**Explicit Citations**: The LLM is instructed to cite sources using
numbered references [1], [2], etc. These are mapped back to the original
chunk IDs for traceability.

**Structured Output**: We use JSON-mode prompting to get reliable
structured responses that can be parsed into Pydantic models.

**Provider Agnostic**: Works with any LLM provider (Groq, Ollama)
through the LLMProviderProtocol.
"""

from __future__ import annotations

import json
import logging
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from rag_assistant.llm.interface import ChatMessage, LLMProviderProtocol, MessageRole
from rag_assistant.llm.structured import (
    JSONParseError,
    SchemaValidationError,
    extract_json,
)
from rag_assistant.schemas.chunk import RetrievedChunk
from rag_assistant.schemas.rag_response import (
    InsufficientContextResponse,
    RAGResponse,
    RetrievalMetadata,
    Source,
)

from .prompts import RAGPromptBuilder

if TYPE_CHECKING:
    from rag_assistant.config import Settings

logger = logging.getLogger(__name__)


class ConfidenceLevel(StrEnum):
    """Confidence level for a generated answer."""

    high = "high"
    medium = "medium"
    low = "low"


class LLMGenerationOutput(BaseModel):
    """
    Raw structured output from the LLM.

    This is the schema we instruct the LLM to produce.
    We then transform this into RAGResponse or InsufficientContextResponse.
    """

    answer: str = Field(
        description="The generated answer with citation markers like [1], [2].",
    )
    citation_indices: list[int] = Field(
        default_factory=list,
        description="List of citation indices used in the answer.",
    )
    insufficient_context: bool = Field(
        default=False,
        description="True if the context doesn't contain relevant information.",
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.medium,
        description="Confidence level of the answer.",
    )


class GenerationError(Exception):
    """Error during RAG generation."""

    pass


class GroundedGenerator:
    """
    Generates grounded answers with citations from retrieved context.

    The generator:
    1. Formats retrieved chunks into a numbered context
    2. Builds prompts that enforce grounding and citations
    3. Calls the LLM with structured output instructions
    4. Parses the response and maps citations to sources
    5. Returns appropriate response type based on context sufficiency

    Parameters
    ----------
    provider
        LLM provider (Groq, Ollama, etc.).
    prompt_builder
        Builder for RAG prompts. Defaults to RAGPromptBuilder().
    min_confidence_threshold
        Minimum confidence to not mark as insufficient context.
        Answers with lower confidence may be flagged.
    """

    def __init__(
        self,
        provider: LLMProviderProtocol,
        *,
        prompt_builder: RAGPromptBuilder | None = None,
        min_confidence_threshold: ConfidenceLevel = ConfidenceLevel.low,
    ) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder or RAGPromptBuilder()
        self._min_confidence = min_confidence_threshold

    @property
    def provider(self) -> LLMProviderProtocol:
        """Return the LLM provider."""
        return self._provider

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        retrieval_metadata: RetrievalMetadata | None = None,
    ) -> RAGResponse | InsufficientContextResponse:
        """
        Generate a grounded answer from retrieved chunks.

        Parameters
        ----------
        query
            The user's question.
        chunks
            Retrieved chunks to use as context.
        retrieval_metadata
            Optional metadata about the retrieval process.

        Returns
        -------
        RAGResponse | InsufficientContextResponse
            Either a grounded answer with sources, or an indication
            that the context was insufficient.

        Raises
        ------
        GenerationError
            If the LLM response cannot be parsed or is invalid.
        """
        start_time = time.perf_counter()

        # Handle empty context
        if not chunks:
            return InsufficientContextResponse(
                query=query,
                candidates_found=0,
                suggestion="Try rephrasing your question or searching for related topics.",
            )

        # Build messages and get chunk mapping
        messages, chunk_mapping = self._prompt_builder.build_messages(query, chunks)

        # Convert to ChatMessage objects
        chat_messages = [
            ChatMessage(role=MessageRole(m["role"]), content=m["content"])
            for m in messages
        ]

        # Call LLM
        logger.debug("Calling LLM for grounded generation (query=%r)", query[:50])
        response = self._provider.chat(chat_messages)
        logger.debug("LLM response received (latency=%.0fms)", response.latency_ms)

        # Parse structured output
        try:
            output = self._parse_llm_output(response.content)
        except (JSONParseError, SchemaValidationError) as e:
            logger.warning("Failed to parse LLM output: %s", e)
            raise GenerationError(f"Failed to parse LLM response: {e}") from e

        # Handle insufficient context
        if output.insufficient_context or not output.answer.strip():
            return InsufficientContextResponse(
                query=query,
                candidates_found=len(chunks),
                suggestion="The retrieved documents don't seem to contain "
                "relevant information for this question.",
            )

        # Map citations to sources
        sources = self._map_citations_to_sources(
            output.citation_indices,
            chunk_mapping,
            chunks,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update retrieval metadata with generation time
        if retrieval_metadata:
            # Create new metadata with updated latency
            retrieval_metadata = RetrievalMetadata(
                total_candidates=retrieval_metadata.total_candidates,
                dense_candidates=retrieval_metadata.dense_candidates,
                lexical_candidates=retrieval_metadata.lexical_candidates,
                reranked=retrieval_metadata.reranked,
                latency_ms=retrieval_metadata.latency_ms + elapsed_ms,
            )

        return RAGResponse(
            answer=output.answer,
            sources=sources,
            retrieval_metadata=retrieval_metadata,
        )

    async def agenerate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        retrieval_metadata: RetrievalMetadata | None = None,
    ) -> RAGResponse | InsufficientContextResponse:
        """
        Async version of generate.

        Parameters
        ----------
        query
            The user's question.
        chunks
            Retrieved chunks to use as context.
        retrieval_metadata
            Optional metadata about the retrieval process.

        Returns
        -------
        RAGResponse | InsufficientContextResponse
            Either a grounded answer with sources, or an indication
            that the context was insufficient.
        """
        start_time = time.perf_counter()

        # Handle empty context
        if not chunks:
            return InsufficientContextResponse(
                query=query,
                candidates_found=0,
                suggestion="Try rephrasing your question or searching for related topics.",
            )

        # Build messages and get chunk mapping
        messages, chunk_mapping = self._prompt_builder.build_messages(query, chunks)

        # Convert to ChatMessage objects
        chat_messages = [
            ChatMessage(role=MessageRole(m["role"]), content=m["content"])
            for m in messages
        ]

        # Call LLM (async)
        logger.debug("Calling LLM (async) for grounded generation (query=%r)", query[:50])
        response = await self._provider.achat(chat_messages)
        logger.debug("LLM response received (latency=%.0fms)", response.latency_ms)

        # Parse structured output
        try:
            output = self._parse_llm_output(response.content)
        except (JSONParseError, SchemaValidationError) as e:
            logger.warning("Failed to parse LLM output: %s", e)
            raise GenerationError(f"Failed to parse LLM response: {e}") from e

        # Handle insufficient context
        if output.insufficient_context or not output.answer.strip():
            return InsufficientContextResponse(
                query=query,
                candidates_found=len(chunks),
                suggestion="The retrieved documents don't seem to contain "
                "relevant information for this question.",
            )

        # Map citations to sources
        sources = self._map_citations_to_sources(
            output.citation_indices,
            chunk_mapping,
            chunks,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update retrieval metadata with generation time
        if retrieval_metadata:
            retrieval_metadata = RetrievalMetadata(
                total_candidates=retrieval_metadata.total_candidates,
                dense_candidates=retrieval_metadata.dense_candidates,
                lexical_candidates=retrieval_metadata.lexical_candidates,
                reranked=retrieval_metadata.reranked,
                latency_ms=retrieval_metadata.latency_ms + elapsed_ms,
            )

        return RAGResponse(
            answer=output.answer,
            sources=sources,
            retrieval_metadata=retrieval_metadata,
        )

    def _parse_llm_output(self, content: str) -> LLMGenerationOutput:
        """
        Parse LLM output into structured format.

        Parameters
        ----------
        content
            Raw LLM response content.

        Returns
        -------
        LLMGenerationOutput
            Parsed and validated output.

        Raises
        ------
        JSONParseError
            If JSON cannot be extracted.
        SchemaValidationError
            If JSON doesn't match schema.
        """
        json_str = extract_json(content)
        data = json.loads(json_str)
        return LLMGenerationOutput.model_validate(data)

    def _map_citations_to_sources(
        self,
        citation_indices: list[int],
        chunk_mapping: dict[int, str],
        chunks: list[RetrievedChunk],
    ) -> list[Source]:
        """
        Map citation indices to Source objects.

        Parameters
        ----------
        citation_indices
            Indices cited by the model (1-based).
        chunk_mapping
            Mapping from index to chunk_id.
        chunks
            Original retrieved chunks.

        Returns
        -------
        list[Source]
            Source objects for each valid citation.
        """
        # Build chunk lookup by ID
        chunk_by_id: dict[str, RetrievedChunk] = {
            c.chunk_metadata.chunk_id: c for c in chunks
        }

        sources: list[Source] = []
        seen_ids: set[str] = set()

        for idx in citation_indices:
            chunk_id = chunk_mapping.get(idx)
            if not chunk_id or chunk_id in seen_ids:
                continue

            chunk = chunk_by_id.get(chunk_id)
            if not chunk:
                continue

            meta = chunk.chunk_metadata
            sources.append(
                Source(
                    title=meta.title,
                    url=meta.url,
                    chunk_id=chunk_id,
                    section=meta.section,
                    relevance_score=chunk.score,
                )
            )
            seen_ids.add(chunk_id)

        return sources

    def __repr__(self) -> str:
        return f"GroundedGenerator(provider={self._provider!r})"


def create_generator(
    settings: Settings,
    *,
    provider: LLMProviderProtocol | None = None,
) -> GroundedGenerator:
    """
    Create a grounded generator from application settings.

    Parameters
    ----------
    settings
        Application settings.
    provider
        Optional: pre-built provider for testing.

    Returns
    -------
    GroundedGenerator
        Configured generator.
    """
    if provider is None:
        from rag_assistant.llm.factory import create_llm_provider

        provider = create_llm_provider(settings)

    return GroundedGenerator(provider=provider)
