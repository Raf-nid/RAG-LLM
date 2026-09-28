"""Tests for the grounded generation module."""

import uuid
from unittest.mock import MagicMock

import pytest

from rag_assistant.rag.generation.generator import (
    ConfidenceLevel,
    GenerationError,
    GroundedGenerator,
    LLMGenerationOutput,
)
from rag_assistant.rag.generation.prompts import RAGPromptBuilder
from rag_assistant.schemas.chunk import (
    ChunkMetadata,
    DocumentChunk,
    RetrievalMethod,
    RetrievedChunk,
)
from rag_assistant.schemas.rag_response import (
    InsufficientContextResponse,
    RAGResponse,
    RetrievalMetadata,
)


def make_uuid(name: str) -> str:
    """Generate a deterministic UUID for testing."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def make_chunk(
    text: str,
    source: str = "langchain",
    title: str = "Test Doc",
    url: str = "https://example.com",
    score: float = 0.9,
) -> RetrievedChunk:
    """Create a test chunk."""
    metadata = ChunkMetadata(
        chunk_id=make_uuid(f"{source}_{title}_{text[:20]}"),
        source=source,
        title=title,
        url=url,
        section="Test Section",
    )
    return RetrievedChunk(
        chunk=DocumentChunk(text=text, metadata=metadata),
        score=score,
        retrieval_method=RetrievalMethod.dense,
    )


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response_content: str) -> None:
        self._response_content = response_content
        self._calls: list[tuple[str, list]] = []

    def chat(self, messages: list) -> MagicMock:
        self._calls.append(("chat", messages))
        return MagicMock(
            content=self._response_content,
            model="mock-model",
            latency_ms=100.0,
            usage=MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )

    async def achat(self, messages: list) -> MagicMock:
        self._calls.append(("achat", messages))
        return MagicMock(
            content=self._response_content,
            model="mock-model",
            latency_ms=100.0,
            usage=MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )


class TestRAGPromptBuilder:
    """Tests for the prompt builder."""

    def test_format_context_empty(self) -> None:
        builder = RAGPromptBuilder()
        result = builder.format_context([])

        assert result.text == "No relevant documents found."
        assert result.chunk_mapping == {}

    def test_format_context_single_chunk(self) -> None:
        builder = RAGPromptBuilder()
        chunk = make_chunk("This is test content.", source="langchain", title="Agents")

        result = builder.format_context([chunk])

        assert "[1]" in result.text
        assert "langchain" in result.text
        assert "Agents" in result.text
        assert "This is test content." in result.text
        assert 1 in result.chunk_mapping
        assert result.chunk_mapping[1] == chunk.chunk_metadata.chunk_id

    def test_format_context_multiple_chunks(self) -> None:
        builder = RAGPromptBuilder()
        chunks = [
            make_chunk("First content", source="langchain", title="Doc 1"),
            make_chunk("Second content", source="fastapi", title="Doc 2"),
            make_chunk("Third content", source="pydantic", title="Doc 3"),
        ]

        result = builder.format_context(chunks)

        assert "[1]" in result.text
        assert "[2]" in result.text
        assert "[3]" in result.text
        assert len(result.chunk_mapping) == 3

    def test_build_system_prompt(self) -> None:
        builder = RAGPromptBuilder()
        prompt = builder.build_system_prompt()

        assert "ONLY use information from the CONTEXT" in prompt
        assert "citation" in prompt.lower()
        assert "JSON" in prompt

    def test_build_user_prompt(self) -> None:
        builder = RAGPromptBuilder()
        prompt = builder.build_user_prompt(
            question="How do agents work?",
            context="[1] LangChain | Agents\nAgents use LLMs...",
        )

        assert "How do agents work?" in prompt
        assert "Agents use LLMs" in prompt
        assert "CONTEXT" in prompt

    def test_build_messages(self) -> None:
        builder = RAGPromptBuilder()
        chunks = [make_chunk("Test content")]

        messages, mapping = builder.build_messages("What is this?", chunks)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "What is this?" in messages[1]["content"]
        assert len(mapping) == 1


class TestLLMGenerationOutput:
    """Tests for the LLM output schema."""

    def test_valid_output(self) -> None:
        output = LLMGenerationOutput(
            answer="This is the answer [1].",
            citation_indices=[1],
            insufficient_context=False,
            confidence=ConfidenceLevel.high,
        )

        assert output.answer == "This is the answer [1]."
        assert output.citation_indices == [1]
        assert not output.insufficient_context
        assert output.confidence == ConfidenceLevel.high

    def test_defaults(self) -> None:
        output = LLMGenerationOutput(answer="Test")

        assert output.citation_indices == []
        assert not output.insufficient_context
        assert output.confidence == ConfidenceLevel.medium


class TestGroundedGenerator:
    """Tests for the grounded generator."""

    def test_generate_with_valid_response(self) -> None:
        response_json = """
        {
            "answer": "LangChain agents use LLMs to reason [1].",
            "citation_indices": [1],
            "insufficient_context": false,
            "confidence": "high"
        }
        """
        provider = MockLLMProvider(response_json)
        generator = GroundedGenerator(provider=provider)

        chunks = [
            make_chunk(
                "LangChain agents use LLMs to decide which actions to take.",
                source="langchain",
                title="Agents",
            )
        ]

        response = generator.generate("How do agents work?", chunks)

        assert isinstance(response, RAGResponse)
        assert "[1]" in response.answer
        assert len(response.sources) == 1
        assert response.sources[0].title == "Agents"

    def test_generate_insufficient_context(self) -> None:
        response_json = """
        {
            "answer": "",
            "citation_indices": [],
            "insufficient_context": true,
            "confidence": "low"
        }
        """
        provider = MockLLMProvider(response_json)
        generator = GroundedGenerator(provider=provider)

        chunks = [make_chunk("Unrelated content about cooking recipes.")]

        response = generator.generate("How do quantum computers work?", chunks)

        assert isinstance(response, InsufficientContextResponse)
        assert response.candidates_found == 1

    def test_generate_empty_chunks(self) -> None:
        provider = MockLLMProvider("")
        generator = GroundedGenerator(provider=provider)

        response = generator.generate("Any question?", [])

        assert isinstance(response, InsufficientContextResponse)
        assert response.candidates_found == 0

    def test_generate_multiple_citations(self) -> None:
        response_json = """
        {
            "answer": "Agents [1] use tools [2] to perform actions.",
            "citation_indices": [1, 2],
            "insufficient_context": false,
            "confidence": "high"
        }
        """
        provider = MockLLMProvider(response_json)
        generator = GroundedGenerator(provider=provider)

        chunks = [
            make_chunk("Agents are reasoning systems.", title="Agents"),
            make_chunk("Tools are functions that agents call.", title="Tools"),
        ]

        response = generator.generate("What are agents and tools?", chunks)

        assert isinstance(response, RAGResponse)
        assert len(response.sources) == 2

    def test_generate_with_retrieval_metadata(self) -> None:
        response_json = """
        {
            "answer": "The answer is here [1].",
            "citation_indices": [1],
            "insufficient_context": false,
            "confidence": "medium"
        }
        """
        provider = MockLLMProvider(response_json)
        generator = GroundedGenerator(provider=provider)

        chunks = [make_chunk("Some relevant content.")]
        metadata = RetrievalMetadata(
            total_candidates=10,
            dense_candidates=10,
            lexical_candidates=0,
            reranked=False,
            latency_ms=50.0,
        )

        response = generator.generate(
            "What is this?",
            chunks,
            retrieval_metadata=metadata,
        )

        assert isinstance(response, RAGResponse)
        assert response.retrieval_metadata is not None
        assert response.retrieval_metadata.total_candidates == 10

    def test_generate_invalid_json_raises(self) -> None:
        provider = MockLLMProvider("This is not JSON at all")
        generator = GroundedGenerator(provider=provider)

        chunks = [make_chunk("Some content.")]

        with pytest.raises(GenerationError):
            generator.generate("Question?", chunks)

    def test_generate_with_fenced_json(self) -> None:
        response_json = """
        Here is my response:
        ```json
        {
            "answer": "The answer from fenced JSON [1].",
            "citation_indices": [1],
            "insufficient_context": false,
            "confidence": "high"
        }
        ```
        """
        provider = MockLLMProvider(response_json)
        generator = GroundedGenerator(provider=provider)

        chunks = [make_chunk("Content here.")]

        response = generator.generate("Question?", chunks)

        assert isinstance(response, RAGResponse)
        assert "fenced JSON" in response.answer

    def test_citation_deduplication(self) -> None:
        response_json = """
        {
            "answer": "Referenced multiple times [1] [1] [1].",
            "citation_indices": [1, 1, 1],
            "insufficient_context": false,
            "confidence": "high"
        }
        """
        provider = MockLLMProvider(response_json)
        generator = GroundedGenerator(provider=provider)

        chunks = [make_chunk("Single source.")]

        response = generator.generate("Question?", chunks)

        assert isinstance(response, RAGResponse)
        # Should deduplicate to single source
        assert len(response.sources) == 1

    def test_invalid_citation_index_ignored(self) -> None:
        response_json = """
        {
            "answer": "Reference to nonexistent [99].",
            "citation_indices": [99],
            "insufficient_context": false,
            "confidence": "medium"
        }
        """
        provider = MockLLMProvider(response_json)
        generator = GroundedGenerator(provider=provider)

        chunks = [make_chunk("Only one chunk.")]

        response = generator.generate("Question?", chunks)

        assert isinstance(response, RAGResponse)
        # Invalid index should be ignored
        assert len(response.sources) == 0


class TestGroundedGeneratorAsync:
    """Tests for async generation."""

    @pytest.mark.asyncio
    async def test_agenerate_basic(self) -> None:
        response_json = """
        {
            "answer": "Async answer [1].",
            "citation_indices": [1],
            "insufficient_context": false,
            "confidence": "high"
        }
        """
        provider = MockLLMProvider(response_json)
        generator = GroundedGenerator(provider=provider)

        chunks = [make_chunk("Async content.")]

        response = await generator.agenerate("Async question?", chunks)

        assert isinstance(response, RAGResponse)
        assert "Async answer" in response.answer

    @pytest.mark.asyncio
    async def test_agenerate_empty_chunks(self) -> None:
        provider = MockLLMProvider("")
        generator = GroundedGenerator(provider=provider)

        response = await generator.agenerate("Question?", [])

        assert isinstance(response, InsufficientContextResponse)
