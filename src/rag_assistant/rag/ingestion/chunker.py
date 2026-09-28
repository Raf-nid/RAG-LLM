"""
Text chunking for document ingestion.

Chunking splits documents into smaller pieces suitable for:
1. Embedding (models have token limits)
2. Retrieval (smaller chunks = more precise retrieval)
3. LLM context (fitting relevant info in context window)

Chunking Strategies
-------------------
**Fixed-size**: Split every N characters. Simple but can cut sentences.

**Recursive**: Try to split on larger boundaries first (paragraphs),
then smaller ones (sentences, words). Preserves semantic units.

**Semantic**: Use embeddings to find natural topic boundaries.
More expensive but highest quality.

This module implements **recursive character splitting** which is
a good balance of quality and simplicity.

Key Parameters
--------------
**chunk_size**: Target size in characters. Too small = context loss,
too large = retrieval precision loss. 500-1000 is typical.

**chunk_overlap**: Characters shared between consecutive chunks.
Preserves context across boundaries. 10-20% of chunk_size is typical.

Usage
-----
    from rag_assistant.rag.ingestion.chunker import chunk_document

    chunks = chunk_document(
        text="Long document text...",
        source="langchain",
        url="https://...",
        title="Getting Started",
        chunk_size=800,
        chunk_overlap=100,
    )
"""

import re
from dataclasses import dataclass

from rag_assistant.schemas.chunk import (
    ChunkMetadata,
    DocumentChunk,
    generate_chunk_id,
)


@dataclass
class RecursiveChunker:
    """
    Recursive text chunker.

    Splits text on progressively smaller separators until
    chunks are within the target size.

    Parameters
    ----------
    chunk_size
        Target chunk size in characters.
    chunk_overlap
        Overlap between consecutive chunks.
    separators
        List of separators to try, from largest to smallest.
    """

    chunk_size: int = 800
    chunk_overlap: int = 100
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    def split(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Parameters
        ----------
        text
            The text to split.

        Returns
        -------
        list[str]
            List of text chunks.
        """
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Recursively split text using the first viable separator."""
        if not separators:
            # No more separators, just slice by character
            return self._split_by_char(text)

        separator = separators[0]
        remaining_seps = separators[1:]

        if not separator:
            # Empty separator = split by character
            return self._split_by_char(text)

        # Split on this separator
        parts = text.split(separator)

        chunks: list[str] = []
        current_chunk = ""

        for part in parts:
            # Would adding this part exceed chunk size?
            test_chunk = current_chunk + separator + part if current_chunk else part

            if len(test_chunk) <= self.chunk_size:
                # Still fits, keep accumulating
                current_chunk = test_chunk
            else:
                # Doesn't fit
                if current_chunk:
                    chunks.append(current_chunk)

                # Is the part itself too large?
                if len(part) > self.chunk_size:
                    # Recursively split with smaller separators
                    sub_chunks = self._split_recursive(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap
        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_by_char(self, text: str) -> list[str]:
        """Split text by character count as last resort."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between chunks by including end of previous chunk."""
        if len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            # Take last chunk_overlap chars from previous
            overlap = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev

            # Prepend to current (avoid duplicating if already present)
            if not curr.startswith(overlap):
                curr = overlap + curr

            result.append(curr)

        return result


def clean_text(text: str) -> str:
    """
    Clean and normalize text for chunking.

    - Normalizes whitespace
    - Removes excessive blank lines
    - Strips leading/trailing whitespace
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple blank lines to two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize spaces (but preserve newlines)
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(lines)

    return text.strip()


def chunk_document(
    text: str,
    source: str,
    url: str,
    title: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    section: str | None = None,
    version: str | None = None,
    document_type: str = "official_docs",
) -> list[DocumentChunk]:
    """
    Chunk a document and create DocumentChunk objects with metadata.

    This is the main entry point for the chunking pipeline.
    It cleans the text, splits it, and creates properly
    attributed chunks with stable IDs.

    Parameters
    ----------
    text
        The document text to chunk.
    source
        Source identifier (e.g., "langchain").
    url
        Original source URL.
    title
        Document or page title.
    chunk_size
        Target chunk size in characters.
    chunk_overlap
        Overlap between chunks.
    section
        Section heading (if available).
    version
        Documentation version (if known).
    document_type
        Type of content (official_docs, api_reference, etc.).

    Returns
    -------
    list[DocumentChunk]
        List of chunks with metadata and stable IDs.
    """
    # Clean the text first
    cleaned = clean_text(text)

    if not cleaned:
        return []

    # Split into chunks
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = chunker.split(cleaned)

    # Create DocumentChunk objects with metadata
    chunks: list[DocumentChunk] = []
    total_chunks = len(text_chunks)

    for i, chunk_text in enumerate(text_chunks):
        # Skip empty chunks
        if not chunk_text.strip():
            continue

        # Generate stable ID based on content
        chunk_id = generate_chunk_id(source, url, chunk_text)

        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            source=source,
            title=title,
            url=url,
            document_type=document_type,
            section=section,
            version=version,
            chunk_index=i,
            total_chunks=total_chunks,
        )

        chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))

    return chunks
