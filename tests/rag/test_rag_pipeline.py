"""Tests for the complete RAG pipeline."""

import uuid

import pytest

from rag_assistant.rag.pipeline import RAGPipeline
from rag_assistant.schemas.chunk import (
    ChunkMetadata,
    DocumentChunk,
    RetrievalMethod,
    RetrievedChunk,
)
from rag_assistant.schemas.rag_response import (
    InsufficientContextResponse,
    RAGResponse,
)


def make_uuid(name: str) -> str:
    """Generate a deterministic UUID for testing."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def make_chunk(
    text: str,
    source: str = "langchain",
    title: str = "Test Doc",
    score: float = 0.9,
) -> RetrievedChunk:
    """Create a test chunk."""
    metadata = ChunkMetadata(
        chunk_id=make_uuid(f"{source}_{title}_{text[:20]}"),
        source=source,
        title=title,
        url="https://example.com",
    )
    return RetrievedChunk(
        chunk=DocumentChunk(text=text, metadata=metadata),
        score=score,
        retrieval_method=RetrievalMethod.dense,
    )


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self._calls: list[dict] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        self._calls.append({
            "query": query,
            "top_k": top_k,
            "source_filter": source_filter,
            "score_threshold": score_threshold,
        })
        chunks = self._chunks
        if source_filter:
            chunks = [c for c in chunks if c.chunk_metadata.source == source_filter]
        if top_k:
            chunks = chunks[:top_k]
        return chunks


class MockGenerator:
    """Mock generator for testing."""

    def __init__(
        self,
        response: RAGResponse | InsufficientContextResponse,
    ) -> None:
        self._response = response
        self._calls: list[dict] = []

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        **kwargs: object,
    ) -> RAGResponse | InsufficientContextResponse:
        self._calls.append({
            "query": query,
            "chunks": chunks,
            "kwargs": kwargs,
        })
        return self._response

    async def agenerate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        **kwargs: object,
    ) -> RAGResponse | InsufficientContextResponse:
        self._calls.append({
            "query": query,
            "chunks": chunks,
            "kwargs": kwargs,
        })
        return self._response


class TestRAGPipeline:
    """Tests for the RAG pipeline."""

    def test_query_basic(self) -> None:
        chunks = [
            make_chunk("LangChain agents use LLMs.", source="langchain"),
        ]
        retriever = MockRetriever(chunks)

        from rag_assistant.schemas.rag_response import Source

        response = RAGResponse(
            answer="LangChain agents use LLMs [1].",
            sources=[
                Source(
                    title="Test Doc",
                    url="https://example.com",
                    chunk_id=chunks[0].chunk_metadata.chunk_id,
                )
            ],
        )
        generator = MockGenerator(response)

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
        )

        result = pipeline.query("How do agents work?")

        assert isinstance(result, RAGResponse)
        assert "LangChain agents" in result.answer
        assert len(retriever._calls) == 1
        assert len(generator._calls) == 1

    def test_query_with_source_filter(self) -> None:
        chunks = [
            make_chunk("LangChain content", source="langchain"),
            make_chunk("FastAPI content", source="fastapi"),
        ]
        retriever = MockRetriever(chunks)
        generator = MockGenerator(
            RAGResponse(answer="Filtered answer", sources=[])
        )

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
        )

        pipeline.query("Question?", source_filter="langchain")

        assert retriever._calls[0]["source_filter"] == "langchain"

    def test_query_with_top_k(self) -> None:
        chunks = [make_chunk(f"Content {i}") for i in range(10)]
        retriever = MockRetriever(chunks)
        generator = MockGenerator(
            RAGResponse(answer="Answer", sources=[])
        )

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
            default_top_k=5,
        )

        pipeline.query("Question?")

        assert retriever._calls[0]["top_k"] == 5

    def test_query_override_top_k(self) -> None:
        retriever = MockRetriever([])
        generator = MockGenerator(
            InsufficientContextResponse(query="Q", candidates_found=0)
        )

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
            default_top_k=5,
        )

        pipeline.query("Question?", top_k=3)

        assert retriever._calls[0]["top_k"] == 3

    def test_query_insufficient_context(self) -> None:
        retriever = MockRetriever([])
        generator = MockGenerator(
            InsufficientContextResponse(
                query="Question?",
                candidates_found=0,
                suggestion="Try a different question.",
            )
        )

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
        )

        result = pipeline.query("Question?")

        assert isinstance(result, InsufficientContextResponse)
        assert result.candidates_found == 0

    def test_retrieve_only(self) -> None:
        chunks = [make_chunk("Content")]
        retriever = MockRetriever(chunks)
        generator = MockGenerator(
            RAGResponse(answer="Placeholder", sources=[])
        )

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
        )

        result = pipeline.retrieve_only("Question?")

        assert len(result) == 1
        assert result[0].text == "Content"
        # Generator should not be called
        assert len(generator._calls) == 0


class TestRAGPipelineAsync:
    """Tests for async pipeline methods."""

    @pytest.mark.asyncio
    async def test_aquery_basic(self) -> None:
        chunks = [make_chunk("Content")]
        retriever = MockRetriever(chunks)
        generator = MockGenerator(
            RAGResponse(answer="Async answer", sources=[])
        )

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
        )

        result = await pipeline.aquery("Async question?")

        assert isinstance(result, RAGResponse)
        assert "Async answer" in result.answer


class TestRAGPipelineProperties:
    """Tests for pipeline properties."""

    def test_retriever_property(self) -> None:
        retriever = MockRetriever([])
        generator = MockGenerator(
            RAGResponse(answer="Test", sources=[])
        )

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
        )

        assert pipeline.retriever is retriever

    def test_generator_property(self) -> None:
        retriever = MockRetriever([])
        generator = MockGenerator(
            RAGResponse(answer="Test", sources=[])
        )

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
        )

        assert pipeline.generator is generator

    def test_repr(self) -> None:
        retriever = MockRetriever([])
        generator = MockGenerator(
            RAGResponse(answer="Test", sources=[])
        )

        pipeline = RAGPipeline(
            retriever=retriever,  # type: ignore
            generator=generator,  # type: ignore
        )

        repr_str = repr(pipeline)
        assert "RAGPipeline" in repr_str
