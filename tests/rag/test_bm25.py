"""Tests for BM25 lexical retrieval."""

import uuid

import pytest

from rag_assistant.rag.retrieval.bm25 import (
    BM25Retriever,
    build_bm25_index,
    tokenize,
)
from rag_assistant.schemas.chunk import (
    ChunkMetadata,
    DocumentChunk,
    RetrievalMethod,
)


def make_uuid(name: str) -> str:
    """Generate a deterministic UUID for testing."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def make_chunk(text: str, source: str = "test", title: str = "Test") -> DocumentChunk:
    """Create a test chunk."""
    metadata = ChunkMetadata(
        chunk_id=make_uuid(f"{source}_{title}_{text[:20]}"),
        source=source,
        title=title,
        url="https://example.com",
    )
    return DocumentChunk(text=text, metadata=metadata)


class TestTokenize:
    """Tests for tokenization."""

    def test_basic_tokenization(self) -> None:
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_removes_punctuation(self) -> None:
        tokens = tokenize("Hello, World!")
        assert tokens == ["hello", "world"]

    def test_preserves_underscores_in_words(self) -> None:
        tokens = tokenize("create_react_agent() function")
        assert "create_react_agent" in tokens
        assert "function" in tokens

    def test_removes_special_chars_outside_words(self) -> None:
        tokens = tokenize("hello! world? foo-bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens

    def test_filters_single_chars(self) -> None:
        tokens = tokenize("a b c hello world")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string(self) -> None:
        tokens = tokenize("")
        assert tokens == []

    def test_only_punctuation(self) -> None:
        tokens = tokenize("!@#$%")
        assert tokens == []


class TestBM25Index:
    """Tests for BM25 index building."""

    def test_build_empty_index(self) -> None:
        index = build_bm25_index([])
        assert index.is_empty()
        assert len(index) == 0

    def test_build_index_with_chunks(self) -> None:
        chunks = [
            make_chunk("LangChain agents use LLMs"),
            make_chunk("FastAPI is a web framework"),
        ]
        index = build_bm25_index(chunks)

        assert not index.is_empty()
        assert len(index) == 2
        assert index.bm25 is not None
        assert len(index.tokenized_corpus) == 2

    def test_index_stores_original_chunks(self) -> None:
        chunks = [make_chunk("Test content")]
        index = build_bm25_index(chunks)

        assert index.chunks[0].text == "Test content"


class TestBM25Retriever:
    """Tests for BM25 retriever."""

    @pytest.fixture
    def sample_chunks(self) -> list[DocumentChunk]:
        """Create sample chunks for testing."""
        return [
            make_chunk(
                "LangChain agents use LLMs to decide which tools to call",
                source="langchain",
                title="Agents",
            ),
            make_chunk(
                "FastAPI is a modern web framework for building APIs",
                source="fastapi",
                title="Introduction",
            ),
            make_chunk(
                "Pydantic provides data validation using Python type hints",
                source="pydantic",
                title="Models",
            ),
            make_chunk(
                "Qdrant is a vector database for similarity search",
                source="qdrant",
                title="Overview",
            ),
            make_chunk(
                "Tools are functions that agents can call to perform actions",
                source="langchain",
                title="Tools",
            ),
        ]

    @pytest.fixture
    def retriever(self, sample_chunks: list[DocumentChunk]) -> BM25Retriever:
        """Create a retriever with indexed chunks."""
        retriever = BM25Retriever()
        retriever.index_chunks(sample_chunks)
        return retriever

    def test_index_chunks(self, sample_chunks: list[DocumentChunk]) -> None:
        retriever = BM25Retriever()
        count = retriever.index_chunks(sample_chunks)

        assert count == 5
        assert retriever.is_indexed
        assert retriever.document_count == 5

    def test_retrieve_basic(self, retriever: BM25Retriever) -> None:
        results = retriever.retrieve("langchain agents", top_k=3)

        assert len(results) > 0
        assert all(r.retrieval_method == RetrievalMethod.lexical for r in results)

    def test_retrieve_exact_match_scores_high(
        self, retriever: BM25Retriever
    ) -> None:
        results = retriever.retrieve("langchain agents LLMs tools", top_k=5)

        # Document about agents should score highest
        assert results[0].chunk_metadata.title == "Agents"

    def test_retrieve_respects_top_k(self, retriever: BM25Retriever) -> None:
        results = retriever.retrieve("agents tools", top_k=2)
        assert len(results) <= 2

    def test_retrieve_with_source_filter(self, retriever: BM25Retriever) -> None:
        results = retriever.retrieve("agents", source_filter="langchain")

        assert all(r.chunk_metadata.source == "langchain" for r in results)

    def test_retrieve_no_match_returns_empty(self, retriever: BM25Retriever) -> None:
        results = retriever.retrieve("xyznonexistent12345")
        assert results == []

    def test_retrieve_empty_query(self, retriever: BM25Retriever) -> None:
        results = retriever.retrieve("")
        assert results == []

    def test_retrieve_empty_index(self) -> None:
        retriever = BM25Retriever()
        results = retriever.retrieve("anything")
        assert results == []

    def test_clear_index(self, retriever: BM25Retriever) -> None:
        assert retriever.is_indexed
        retriever.clear_index()
        assert not retriever.is_indexed
        assert retriever.document_count == 0

    def test_default_top_k(self, sample_chunks: list[DocumentChunk]) -> None:
        retriever = BM25Retriever(default_top_k=2)
        retriever.index_chunks(sample_chunks)

        results = retriever.retrieve("agents tools langchain")
        assert len(results) <= 2

    def test_scores_are_positive(self, retriever: BM25Retriever) -> None:
        results = retriever.retrieve("langchain agents")

        for result in results:
            assert result.score > 0

    def test_repr(self, retriever: BM25Retriever) -> None:
        repr_str = repr(retriever)
        assert "BM25Retriever" in repr_str
        assert "documents=5" in repr_str


class TestBM25Scoring:
    """Tests for BM25 scoring behavior."""

    def test_term_frequency_matters(self) -> None:
        chunks = [
            make_chunk(
                "LangChain agent builds agent workflows and agent tools",
                title="Many agents",
            ),
            make_chunk(
                "FastAPI web framework for building APIs quickly",
                title="No agent",
            ),
            make_chunk(
                "Pydantic models validate data with Python types",
                title="Also no agent",
            ),
        ]
        retriever = BM25Retriever()
        retriever.index_chunks(chunks)

        results = retriever.retrieve("agent")

        # Document with "agent" should be first
        assert len(results) >= 1
        assert results[0].chunk_metadata.title == "Many agents"

    def test_idf_affects_scoring(self) -> None:
        chunks = [
            make_chunk(
                "Python is used for data science and machine learning",
                title="Python ML",
            ),
            make_chunk(
                "Python Django framework for web development projects",
                title="Python web",
            ),
            make_chunk(
                "Python specialized unique tooling for rare niche tasks",
                title="Has rare terms",
            ),
        ]
        retriever = BM25Retriever()
        retriever.index_chunks(chunks)

        results = retriever.retrieve("specialized unique niche")

        assert len(results) >= 1
        assert results[0].chunk_metadata.title == "Has rare terms"

    def test_query_with_multiple_matching_docs(self) -> None:
        chunks = [
            make_chunk("LangChain agents use LLMs", title="Agents"),
            make_chunk("FastAPI web framework", title="FastAPI"),
            make_chunk("Pydantic data validation", title="Pydantic"),
        ]
        retriever = BM25Retriever()
        retriever.index_chunks(chunks)

        results = retriever.retrieve("LangChain agents LLMs")

        # Should find the agents doc
        assert len(results) >= 1
        assert results[0].chunk_metadata.title == "Agents"
