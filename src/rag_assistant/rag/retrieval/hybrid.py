"""
Hybrid retrieval combining dense and lexical search.

Hybrid retrieval merges results from multiple retrieval methods to get
the best of both worlds:

- **Dense retrieval**: Semantic similarity via embeddings
- **Lexical retrieval**: Exact keyword matching via BM25

Fusion Algorithms
-----------------
**Reciprocal Rank Fusion (RRF)**:
The most common fusion method. For each document, calculate:

    RRF_score = Σ 1 / (k + rank_i)

Where:
- k is a constant (typically 60)
- rank_i is the document's rank in retrieval method i

RRF is robust because it:
- Uses ranks, not scores (different methods have different scales)
- Doesn't require score normalization
- Handles missing documents gracefully

**Example RRF Calculation**:
Query: "langchain agents"

Dense results:
1. Doc A (rank 1)
2. Doc B (rank 2)
3. Doc C (rank 3)

BM25 results:
1. Doc B (rank 1)
2. Doc A (rank 2)
3. Doc D (rank 3)

RRF scores (k=60):
- Doc A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
- Doc B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
- Doc C: 1/(60+3) + 0 = 0.0159
- Doc D: 0 + 1/(60+3) = 0.0159

Final ranking: [A/B (tie), C/D (tie)]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_assistant.schemas.chunk import DocumentChunk, RetrievalMethod, RetrievedChunk

from .bm25 import BM25Retriever
from .dense import DenseRetriever

if TYPE_CHECKING:
    from rag_assistant.config import Settings
    from rag_assistant.rag.embeddings.model import EmbeddingModel
    from rag_assistant.rag.store.qdrant import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """
    Result from hybrid fusion.

    Attributes
    ----------
    chunk
        The document chunk.
    fused_score
        The combined score after fusion.
    dense_rank
        Rank in dense retrieval (None if not retrieved).
    lexical_rank
        Rank in lexical retrieval (None if not retrieved).
    """

    chunk: DocumentChunk
    fused_score: float
    dense_rank: int | None
    lexical_rank: int | None


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    lexical_results: list[RetrievedChunk],
    k: int = 60,
) -> list[FusionResult]:
    """
    Combine results using Reciprocal Rank Fusion (RRF).

    RRF is a simple but effective fusion method that uses ranks
    rather than scores, making it robust to different score scales.

    Parameters
    ----------
    dense_results
        Results from dense retrieval, ordered by score.
    lexical_results
        Results from lexical retrieval, ordered by score.
    k
        RRF constant (default 60, as per original paper).

    Returns
    -------
    list[FusionResult]
        Fused results ordered by RRF score descending.
    """
    # Track RRF scores and metadata by chunk_id
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, DocumentChunk] = {}
    dense_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}

    # Process dense results
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk_metadata.chunk_id
        scores[chunk_id] += 1.0 / (k + rank)
        chunks[chunk_id] = result.chunk
        dense_ranks[chunk_id] = rank

    # Process lexical results
    for rank, result in enumerate(lexical_results, start=1):
        chunk_id = result.chunk_metadata.chunk_id
        scores[chunk_id] += 1.0 / (k + rank)
        if chunk_id not in chunks:
            chunks[chunk_id] = result.chunk
        lexical_ranks[chunk_id] = rank

    # Build fusion results
    fusion_results = [
        FusionResult(
            chunk=chunks[chunk_id],
            fused_score=score,
            dense_rank=dense_ranks.get(chunk_id),
            lexical_rank=lexical_ranks.get(chunk_id),
        )
        for chunk_id, score in scores.items()
    ]

    # Sort by fused score descending
    fusion_results.sort(key=lambda x: x.fused_score, reverse=True)

    return fusion_results


class HybridRetriever:
    """
    Hybrid retriever combining dense and lexical search.

    This retriever runs both dense (embedding-based) and lexical (BM25)
    retrieval, then fuses the results using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    dense_retriever
        Dense retriever for semantic search.
    bm25_retriever
        BM25 retriever for lexical search.
    default_top_k
        Default number of results to return.
    rrf_k
        RRF constant (default 60).
    dense_weight
        Number of candidates to fetch from dense retrieval.
        Higher values increase recall but slow down search.
    lexical_weight
        Number of candidates to fetch from lexical retrieval.

    Example
    -------
        hybrid = HybridRetriever(dense_retriever, bm25_retriever)
        results = hybrid.retrieve("langchain agents", top_k=5)
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        *,
        default_top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: int = 20,
        lexical_weight: int = 20,
    ) -> None:
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._default_top_k = default_top_k
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._lexical_weight = lexical_weight

    @property
    def dense_retriever(self) -> DenseRetriever:
        """Return the dense retriever."""
        return self._dense

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """Return the BM25 retriever."""
        return self._bm25

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks using hybrid search.

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
            Retrieved chunks with fused scores.
        """
        if top_k is None:
            top_k = self._default_top_k

        # Run dense retrieval
        logger.debug("Running dense retrieval (top_k=%d)", self._dense_weight)
        dense_results = self._dense.retrieve(
            query=query,
            top_k=self._dense_weight,
            source_filter=source_filter,
        )
        logger.debug("Dense retrieval returned %d results", len(dense_results))

        # Run lexical retrieval
        logger.debug("Running lexical retrieval (top_k=%d)", self._lexical_weight)
        lexical_results = self._bm25.retrieve(
            query=query,
            top_k=self._lexical_weight,
            source_filter=source_filter,
        )
        logger.debug("Lexical retrieval returned %d results", len(lexical_results))

        # If only one method returned results, use that
        if not dense_results and not lexical_results:
            return []

        if not dense_results:
            return lexical_results[:top_k]

        if not lexical_results:
            return dense_results[:top_k]

        # Fuse results with RRF
        fusion_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            lexical_results=lexical_results,
            k=self._rrf_k,
        )

        # Convert to RetrievedChunk
        results: list[RetrievedChunk] = []
        for fr in fusion_results[:top_k]:
            # Determine retrieval method based on which retrievers found it
            if fr.dense_rank is not None and fr.lexical_rank is not None:
                method = RetrievalMethod.hybrid
            elif fr.dense_rank is not None:
                method = RetrievalMethod.dense
            else:
                method = RetrievalMethod.lexical

            results.append(
                RetrievedChunk(
                    chunk=fr.chunk,
                    score=fr.fused_score,
                    retrieval_method=method,
                )
            )

        logger.debug(
            "Hybrid retrieval returned %d results (dense=%d, lexical=%d, hybrid=%d)",
            len(results),
            sum(1 for r in results if r.retrieval_method == RetrievalMethod.dense),
            sum(1 for r in results if r.retrieval_method == RetrievalMethod.lexical),
            sum(1 for r in results if r.retrieval_method == RetrievalMethod.hybrid),
        )

        return results

    def retrieve_with_details(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> tuple[list[RetrievedChunk], list[FusionResult]]:
        """
        Retrieve with detailed fusion information.

        Returns both the final results and the raw fusion results
        for debugging and analysis.

        Parameters
        ----------
        query
            The search query.
        top_k
            Number of results.
        source_filter
            Optional source filter.

        Returns
        -------
        tuple[list[RetrievedChunk], list[FusionResult]]
            (final results, fusion details)
        """
        if top_k is None:
            top_k = self._default_top_k

        dense_results = self._dense.retrieve(
            query=query,
            top_k=self._dense_weight,
            source_filter=source_filter,
        )

        lexical_results = self._bm25.retrieve(
            query=query,
            top_k=self._lexical_weight,
            source_filter=source_filter,
        )

        if not dense_results and not lexical_results:
            return [], []

        fusion_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            lexical_results=lexical_results,
            k=self._rrf_k,
        )

        # Convert to RetrievedChunk
        results: list[RetrievedChunk] = []
        for fr in fusion_results[:top_k]:
            if fr.dense_rank is not None and fr.lexical_rank is not None:
                method = RetrievalMethod.hybrid
            elif fr.dense_rank is not None:
                method = RetrievalMethod.dense
            else:
                method = RetrievalMethod.lexical

            results.append(
                RetrievedChunk(
                    chunk=fr.chunk,
                    score=fr.fused_score,
                    retrieval_method=method,
                )
            )

        return results, fusion_results

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(dense={self._dense!r}, "
            f"bm25={self._bm25!r}, rrf_k={self._rrf_k})"
        )


def create_hybrid_retriever(
    settings: Settings,
    *,
    dense_retriever: DenseRetriever | None = None,
    bm25_retriever: BM25Retriever | None = None,
    embedding_model: EmbeddingModel | None = None,
    store: QdrantStore | None = None,
    chunks_for_bm25: list[DocumentChunk] | None = None,
) -> HybridRetriever:
    """
    Create a hybrid retriever from settings.

    Parameters
    ----------
    settings
        Application settings.
    dense_retriever
        Optional pre-built dense retriever.
    bm25_retriever
        Optional pre-built BM25 retriever.
    embedding_model
        Optional embedding model for dense retriever.
    store
        Optional vector store for dense retriever.
    chunks_for_bm25
        Optional chunks to index in BM25.

    Returns
    -------
    HybridRetriever
        Configured hybrid retriever.
    """
    # Build dense retriever if not provided
    if dense_retriever is None:
        from .dense import create_retriever

        dense_retriever = create_retriever(
            settings,
            embedding_model=embedding_model,
            store=store,
        )

    # Build BM25 retriever if not provided
    if bm25_retriever is None:
        bm25_retriever = BM25Retriever()
        if chunks_for_bm25:
            bm25_retriever.index_chunks(chunks_for_bm25)

    return HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )
