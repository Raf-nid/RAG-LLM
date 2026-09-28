"""
BM25 lexical retrieval.

BM25 (Best Matching 25) is a ranking function used for lexical search.
It ranks documents based on term frequency and inverse document frequency,
making it effective for keyword matching.

Key Concepts
------------
**TF (Term Frequency)**: How often a term appears in a document.
More occurrences = higher relevance.

**IDF (Inverse Document Frequency)**: How rare a term is across all documents.
Rare terms are more discriminative.

**BM25 Formula**:
    score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))

Where:
- f(qi, D) = term frequency of qi in document D
- |D| = document length
- avgdl = average document length
- k1, b = tuning parameters (typically k1=1.5, b=0.75)

Why BM25 for RAG?
-----------------
Dense retrieval (embeddings) captures semantic similarity but can miss
exact keyword matches. BM25 excels at finding documents with specific
terms, making it complementary to dense retrieval.

Example:
- Query: "create_react_agent function"
- Dense might find "how to build agents" (semantically similar)
- BM25 finds "use create_react_agent()" (exact match)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from rag_assistant.schemas.chunk import RetrievalMethod, RetrievedChunk

if TYPE_CHECKING:
    from rag_assistant.schemas.chunk import DocumentChunk


@dataclass
class BM25Index:
    """
    BM25 index for a corpus of document chunks.

    This class manages the BM25 index and provides search functionality.
    It stores the original chunks alongside the tokenized corpus.

    Attributes
    ----------
    chunks
        The indexed document chunks.
    tokenized_corpus
        Tokenized version of each chunk's text.
    bm25
        The BM25Okapi index.
    """

    chunks: list[DocumentChunk] = field(default_factory=list)
    tokenized_corpus: list[list[str]] = field(default_factory=list)
    bm25: BM25Okapi | None = field(default=None)

    def __len__(self) -> int:
        return len(self.chunks)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.

    Simple whitespace tokenization with lowercasing and basic cleaning.
    For production, consider using a proper tokenizer (spaCy, nltk).

    Parameters
    ----------
    text
        The text to tokenize.

    Returns
    -------
    list[str]
        List of tokens.
    """
    # Lowercase and remove special characters except alphanumeric and spaces
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    # Split on whitespace and filter empty tokens
    tokens = [t for t in text.split() if t and len(t) > 1]
    return tokens


def build_bm25_index(chunks: list[DocumentChunk]) -> BM25Index:
    """
    Build a BM25 index from document chunks.

    Parameters
    ----------
    chunks
        Document chunks to index.

    Returns
    -------
    BM25Index
        The built index ready for searching.
    """
    if not chunks:
        return BM25Index()

    tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return BM25Index(
        chunks=list(chunks),
        tokenized_corpus=tokenized_corpus,
        bm25=bm25,
    )


class BM25Retriever:
    """
    BM25-based lexical retriever.

    This retriever uses BM25 scoring to find documents based on
    keyword matching. It's designed to work alongside dense retrieval
    in a hybrid setup.

    Parameters
    ----------
    default_top_k
        Default number of results to return.

    Example
    -------
        retriever = BM25Retriever()
        retriever.index(chunks)
        results = retriever.retrieve("langchain agents", top_k=5)
    """

    def __init__(self, *, default_top_k: int = 5) -> None:
        self._default_top_k = default_top_k
        self._index: BM25Index = BM25Index()

    @property
    def index(self) -> BM25Index:
        """Return the current index."""
        return self._index

    @property
    def is_indexed(self) -> bool:
        """Check if documents have been indexed."""
        return not self._index.is_empty()

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._index)

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Index document chunks for BM25 search.

        Parameters
        ----------
        chunks
            Document chunks to index.

        Returns
        -------
        int
            Number of chunks indexed.
        """
        self._index = build_bm25_index(chunks)
        return len(self._index)

    def clear_index(self) -> None:
        """Clear the current index."""
        self._index = BM25Index()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks using BM25 scoring.

        Parameters
        ----------
        query
            The search query.
        top_k
            Number of results to return.
        source_filter
            Optional: filter by source.

        Returns
        -------
        list[RetrievedChunk]
            Retrieved chunks with BM25 scores.
        """
        if top_k is None:
            top_k = self._default_top_k

        if self._index.is_empty() or self._index.bm25 is None:
            return []

        # Tokenize query
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = self._index.bm25.get_scores(query_tokens)

        # Create (score, index) pairs and sort by score descending
        scored_indices = [(score, i) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[0], reverse=True)

        # Build results
        results: list[RetrievedChunk] = []
        for score, idx in scored_indices:
            if score <= 0:
                continue  # Skip zero-score documents

            chunk = self._index.chunks[idx]

            # Apply source filter if specified
            if source_filter and chunk.metadata.source != source_filter:
                continue

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    retrieval_method=RetrievalMethod.lexical,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def __repr__(self) -> str:
        return f"BM25Retriever(documents={len(self._index)}, top_k={self._default_top_k})"
