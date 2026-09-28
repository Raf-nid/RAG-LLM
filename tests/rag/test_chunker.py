"""Tests for the text chunker."""


from rag_assistant.rag.ingestion.chunker import (
    RecursiveChunker,
    chunk_document,
    clean_text,
)
from rag_assistant.schemas.chunk import generate_chunk_id


class TestCleanText:
    """Tests for text cleaning."""

    def test_normalizes_line_endings(self) -> None:
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        result = clean_text(text)
        assert "\r" not in result
        assert "Line 1\nLine 2\nLine 3\nLine 4" == result

    def test_collapses_multiple_blank_lines(self) -> None:
        text = "Para 1\n\n\n\n\nPara 2"
        result = clean_text(text)
        assert result == "Para 1\n\nPara 2"

    def test_strips_whitespace(self) -> None:
        text = "  Hello   World  "
        result = clean_text(text)
        assert result == "Hello World"

    def test_empty_string(self) -> None:
        assert clean_text("") == ""
        assert clean_text("   ") == ""


class TestRecursiveChunker:
    """Tests for the recursive chunker."""

    def test_small_text_not_split(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        text = "Short text"
        chunks = chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_on_paragraph(self) -> None:
        # chunk_size must be smaller than total text but larger than each paragraph
        # "Paragraph one." = 14 chars, "Paragraph two." = 14 chars, total = 30 chars
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        text = "Paragraph one.\n\nParagraph two."
        chunks = chunker.split(text)
        assert len(chunks) == 2
        assert "Paragraph one." in chunks[0]
        assert "Paragraph two." in chunks[1]

    def test_splits_on_newline_when_paragraph_too_large(self) -> None:
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        text = "Line one.\nLine two.\nLine three."
        chunks = chunker.split(text)
        assert len(chunks) >= 2

    def test_respects_chunk_size(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        text = "A" * 300
        chunks = chunker.split(text)
        # All chunks except possibly the last should be close to chunk_size
        for chunk in chunks[:-1]:
            assert len(chunk) <= 110  # Allow some flexibility

    def test_overlap_adds_context(self) -> None:
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "First part of text.\n\nSecond part of text."
        chunks = chunker.split(text)
        # Second chunk should contain overlap from first
        if len(chunks) > 1:
            # Overlap creates shared content
            assert len(chunks[1]) > len("Second part of text.")


class TestChunkDocument:
    """Tests for the document chunking function."""

    def test_creates_chunks_with_metadata(self) -> None:
        chunks = chunk_document(
            text="Hello world. This is a test document.",
            source="test",
            url="https://example.com",
            title="Test Document",
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata.source == "test"
        assert chunks[0].metadata.url == "https://example.com"
        assert chunks[0].metadata.title == "Test Document"

    def test_generates_stable_chunk_ids(self) -> None:
        chunks1 = chunk_document(
            text="Same content",
            source="test",
            url="https://example.com",
            title="Test",
        )
        chunks2 = chunk_document(
            text="Same content",
            source="test",
            url="https://example.com",
            title="Test",
        )
        assert chunks1[0].metadata.chunk_id == chunks2[0].metadata.chunk_id

    def test_different_content_different_ids(self) -> None:
        chunks1 = chunk_document(
            text="Content A",
            source="test",
            url="https://example.com",
            title="Test",
        )
        chunks2 = chunk_document(
            text="Content B",
            source="test",
            url="https://example.com",
            title="Test",
        )
        assert chunks1[0].metadata.chunk_id != chunks2[0].metadata.chunk_id

    def test_chunk_index_and_total(self) -> None:
        # Create text that will be split into multiple chunks
        text = "Paragraph one.\n\n" * 10 + "Paragraph two.\n\n" * 10
        chunks = chunk_document(
            text=text,
            source="test",
            url="https://example.com",
            title="Test",
            chunk_size=100,
            chunk_overlap=0,
        )
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.total_chunks == len(chunks)

    def test_preserves_section(self) -> None:
        chunks = chunk_document(
            text="Test content",
            source="test",
            url="https://example.com",
            title="Test",
            section="Introduction",
        )
        assert chunks[0].metadata.section == "Introduction"

    def test_preserves_version(self) -> None:
        chunks = chunk_document(
            text="Test content",
            source="test",
            url="https://example.com",
            title="Test",
            version="1.0",
        )
        assert chunks[0].metadata.version == "1.0"

    def test_empty_text_returns_empty_list(self) -> None:
        chunks = chunk_document(
            text="",
            source="test",
            url="https://example.com",
            title="Test",
        )
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        chunks = chunk_document(
            text="   \n\n   ",
            source="test",
            url="https://example.com",
            title="Test",
        )
        assert chunks == []


class TestGenerateChunkId:
    """Tests for chunk ID generation."""

    def test_deterministic(self) -> None:
        id1 = generate_chunk_id("source", "url", "text")
        id2 = generate_chunk_id("source", "url", "text")
        assert id1 == id2

    def test_different_text_different_id(self) -> None:
        id1 = generate_chunk_id("source", "url", "text1")
        id2 = generate_chunk_id("source", "url", "text2")
        assert id1 != id2

    def test_different_source_different_id(self) -> None:
        id1 = generate_chunk_id("source1", "url", "text")
        id2 = generate_chunk_id("source2", "url", "text")
        assert id1 != id2

    def test_id_is_uuid_format(self) -> None:
        import uuid

        chunk_id = generate_chunk_id("source", "url", "text")
        # Should be valid UUID format (36 chars with dashes)
        assert len(chunk_id) == 36
        # Should parse as a valid UUID
        uuid.UUID(chunk_id)
