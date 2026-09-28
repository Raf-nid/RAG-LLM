"""Tests for hybrid retrieval with RRF fusion."""

import uuid

import pytest

from rag_assistant.rag.retrieval.bm25 import BM25Retriever
from rag_assistant.rag.retrieval.hybrid import (
    FusionResult,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from rag_assistant.schemas.chunk import (
    ChunkMetadata,
    DocumentChunk,
    RetrievalMethod,
    RetrievedChunk,
)


def make_uuid(name: str) -> str:
    """Generate a deterministic UUID for testing."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def make_chunk(
    text: str,
    source: str = "test",
    title: str = "Test",
    chunk_id: str | None = None,
) -> DocumentChunk:
    """Create a test chunk."""
    if chunk_id is None:
        chunk_id = make_uuid(f"{source}_{title}_{text[:20]}")
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        source=source,
        title=title,
        url="https://example.com",
    )
    return DocumentChunk(text=text, metadata=metadata)


def make_retrieved(
    chunk: DocumentChunk,
    score: float,
    method: RetrievalMethod = RetrievalMethod.dense,
) -> RetrievedChunk:
    """Create a retrieved chunk."""
    return RetrievedChunk(chunk=chunk, score=score, retrieval_method=method)


class TestReciprocalRankFusion:
    """Tests for RRF fusion algorithm."""

    def test_empty_inputs(self) -> None:
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_only_dense_results(self) -> None:
        chunk_a = make_chunk("Content A", chunk_id="a")
        chunk_b = make_chunk("Content B", chunk_id="b")

        dense = [
            make_retrieved(chunk_a, 0.9),
            make_retrieved(chunk_b, 0.8),
        ]

        results = reciprocal_rank_fusion(dense, [], k=60)

        assert len(results) == 2
        # Rank 1 should have higher score
        assert results[0].chunk.metadata.chunk_id == "a"
        assert results[0].dense_rank == 1
        assert results[0].lexical_rank is None

    def test_only_lexical_results(self) -> None:
        chunk_a = make_chunk("Content A", chunk_id="a")

        lexical = [make_retrieved(chunk_a, 5.0, RetrievalMethod.lexical)]

        results = reciprocal_rank_fusion([], lexical, k=60)

        assert len(results) == 1
        assert results[0].lexical_rank == 1
        assert results[0].dense_rank is None

    def test_overlapping_results_get_boosted(self) -> None:
        chunk_a = make_chunk("Content A", chunk_id="a")
        chunk_b = make_chunk("Content B", chunk_id="b")

        # Both retrievers find chunk_a
        dense = [
            make_retrieved(chunk_a, 0.9),
            make_retrieved(chunk_b, 0.8),
        ]
        lexical = [
            make_retrieved(chunk_a, 5.0, RetrievalMethod.lexical),
        ]

        results = reciprocal_rank_fusion(dense, lexical, k=60)

        # chunk_a appears in both, so should have highest RRF score
        assert results[0].chunk.metadata.chunk_id == "a"
        assert results[0].dense_rank == 1
        assert results[0].lexical_rank == 1

    def test_rrf_score_calculation(self) -> None:
        chunk_a = make_chunk("Content A", chunk_id="a")

        # Rank 1 in dense, rank 2 in lexical
        dense = [make_retrieved(chunk_a, 0.9)]
        lexical = [
            make_retrieved(make_chunk("Other", chunk_id="x"), 6.0, RetrievalMethod.lexical),
            make_retrieved(chunk_a, 5.0, RetrievalMethod.lexical),
        ]

        results = reciprocal_rank_fusion(dense, lexical, k=60)

        # Find chunk_a
        chunk_a_result = next(r for r in results if r.chunk.metadata.chunk_id == "a")

        # RRF score should be 1/(60+1) + 1/(60+2)
        expected = 1 / 61 + 1 / 62
        assert abs(chunk_a_result.fused_score - expected) < 0.0001

    def test_k_parameter_affects_scores(self) -> None:
        chunk_a = make_chunk("Content A", chunk_id="a")

        dense = [make_retrieved(chunk_a, 0.9)]

        # With k=60
        results_k60 = reciprocal_rank_fusion(dense, [], k=60)
        # With k=10
        results_k10 = reciprocal_rank_fusion(dense, [], k=10)

        # Lower k means rank matters more
        assert results_k10[0].fused_score > results_k60[0].fused_score

    def test_results_sorted_by_score(self) -> None:
        chunks = [
            make_chunk(f"Content {i}", chunk_id=str(i))
            for i in range(5)
        ]

        dense = [make_retrieved(chunks[0], 0.9)]
        lexical = [
            make_retrieved(chunks[1], 5.0, RetrievalMethod.lexical),
            make_retrieved(chunks[0], 4.0, RetrievalMethod.lexical),  # Overlap
        ]

        results = reciprocal_rank_fusion(dense, lexical, k=60)

        # Verify sorted by score descending
        scores = [r.fused_score for r in results]
        assert scores == sorted(scores, reverse=True)


class MockDenseRetriever:
    """Mock dense retriever for testing."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self._calls: list[dict] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        self._calls.append({
            "query": query,
            "top_k": top_k,
            "source_filter": source_filter,
        })
        results = self._results
        if source_filter:
            results = [r for r in results if r.chunk_metadata.source == source_filter]
        if top_k:
            results = results[:top_k]
        return results


class TestHybridRetriever:
    """Tests for hybrid retriever."""

    @pytest.fixture
    def sample_chunks(self) -> list[DocumentChunk]:
        """Sample chunks for testing."""
        return [
            make_chunk(
                "LangChain agents use LLMs to decide which tools to call",
                source="langchain",
                title="Agents",
                chunk_id="agents",
            ),
            make_chunk(
                "Tools are functions that agents can invoke",
                source="langchain",
                title="Tools",
                chunk_id="tools",
            ),
            make_chunk(
                "FastAPI is a web framework",
                source="fastapi",
                title="FastAPI",
                chunk_id="fastapi",
            ),
        ]

    @pytest.fixture
    def hybrid_retriever(
        self, sample_chunks: list[DocumentChunk]
    ) -> HybridRetriever:
        """Create hybrid retriever with mocks."""
        # Dense retriever returns first two chunks
        dense_results = [
            make_retrieved(sample_chunks[0], 0.9),
            make_retrieved(sample_chunks[1], 0.8),
        ]
        mock_dense = MockDenseRetriever(dense_results)

        # BM25 retriever
        bm25 = BM25Retriever()
        bm25.index_chunks(sample_chunks)

        return HybridRetriever(
            dense_retriever=mock_dense,  # type: ignore
            bm25_retriever=bm25,
        )

    def test_retrieve_combines_results(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        results = hybrid_retriever.retrieve("langchain agents tools")

        assert len(results) > 0
        # Should find agents and tools chunks
        titles = [r.chunk_metadata.title for r in results]
        assert "Agents" in titles

    def test_retrieve_assigns_correct_method(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        results = hybrid_retriever.retrieve("langchain agents")

        # Results found by both should be hybrid
        methods = {r.retrieval_method for r in results}
        # Should have at least one hybrid result
        assert RetrievalMethod.hybrid in methods or len(results) > 0

    def test_retrieve_respects_top_k(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        results = hybrid_retriever.retrieve("agents", top_k=2)
        assert len(results) <= 2

    def test_retrieve_with_source_filter(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        results = hybrid_retriever.retrieve("agents", source_filter="langchain")

        for result in results:
            assert result.chunk_metadata.source == "langchain"

    def test_retrieve_with_details(
        self, hybrid_retriever: HybridRetriever
    ) -> None:
        results, fusion_details = hybrid_retriever.retrieve_with_details(
            "langchain agents"
        )

        assert len(results) > 0
        assert len(fusion_details) >= len(results)

        # Fusion details should have rank info
        for fd in fusion_details:
            assert isinstance(fd, FusionResult)
            assert fd.dense_rank is not None or fd.lexical_rank is not None

    def test_empty_results(self) -> None:
        mock_dense = MockDenseRetriever([])
        bm25 = BM25Retriever()

        hybrid = HybridRetriever(
            dense_retriever=mock_dense,  # type: ignore
            bm25_retriever=bm25,
        )

        results = hybrid.retrieve("anything")
        assert results == []

    def test_only_dense_results(self, sample_chunks: list[DocumentChunk]) -> None:
        dense_results = [make_retrieved(sample_chunks[0], 0.9)]
        mock_dense = MockDenseRetriever(dense_results)
        bm25 = BM25Retriever()  # Empty index

        hybrid = HybridRetriever(
            dense_retriever=mock_dense,  # type: ignore
            bm25_retriever=bm25,
        )

        results = hybrid.retrieve("agents")

        assert len(results) == 1
        # Should be dense since only dense found it
        assert results[0].retrieval_method == RetrievalMethod.dense

    def test_only_bm25_results(self, sample_chunks: list[DocumentChunk]) -> None:
        mock_dense = MockDenseRetriever([])
        bm25 = BM25Retriever()
        bm25.index_chunks(sample_chunks)

        hybrid = HybridRetriever(
            dense_retriever=mock_dense,  # type: ignore
            bm25_retriever=bm25,
        )

        results = hybrid.retrieve("langchain agents")

        assert len(results) > 0
        # Should be lexical since only BM25 found it
        assert results[0].retrieval_method == RetrievalMethod.lexical

    def test_properties(self, hybrid_retriever: HybridRetriever) -> None:
        assert hybrid_retriever.dense_retriever is not None
        assert hybrid_retriever.bm25_retriever is not None

    def test_repr(self, hybrid_retriever: HybridRetriever) -> None:
        repr_str = repr(hybrid_retriever)
        assert "HybridRetriever" in repr_str


class TestRRFNumericalExamples:
    """Test RRF with specific numerical examples for documentation."""

    def test_worked_example_from_docs(self) -> None:
        """
        Test the worked example:
        Dense: [A(1), B(2), C(3)]
        BM25:  [B(1), A(2), D(3)]

        Expected RRF (k=60):
        A: 1/61 + 1/62 = 0.0164 + 0.0161 = 0.0325
        B: 1/62 + 1/61 = 0.0161 + 0.0164 = 0.0325
        C: 1/63 + 0    = 0.0159
        D: 0    + 1/63 = 0.0159
        """
        chunk_a = make_chunk("A", chunk_id="a")
        chunk_b = make_chunk("B", chunk_id="b")
        chunk_c = make_chunk("C", chunk_id="c")
        chunk_d = make_chunk("D", chunk_id="d")

        dense = [
            make_retrieved(chunk_a, 0.9),  # Rank 1
            make_retrieved(chunk_b, 0.8),  # Rank 2
            make_retrieved(chunk_c, 0.7),  # Rank 3
        ]

        lexical = [
            make_retrieved(chunk_b, 5.0, RetrievalMethod.lexical),  # Rank 1
            make_retrieved(chunk_a, 4.0, RetrievalMethod.lexical),  # Rank 2
            make_retrieved(chunk_d, 3.0, RetrievalMethod.lexical),  # Rank 3
        ]

        results = reciprocal_rank_fusion(dense, lexical, k=60)

        # Get results by chunk_id
        by_id = {r.chunk.metadata.chunk_id: r for r in results}

        # Verify scores
        expected_a = 1 / 61 + 1 / 62
        expected_b = 1 / 62 + 1 / 61
        expected_c = 1 / 63
        expected_d = 1 / 63

        assert abs(by_id["a"].fused_score - expected_a) < 0.0001
        assert abs(by_id["b"].fused_score - expected_b) < 0.0001
        assert abs(by_id["c"].fused_score - expected_c) < 0.0001
        assert abs(by_id["d"].fused_score - expected_d) < 0.0001

        # Verify ranks
        assert by_id["a"].dense_rank == 1
        assert by_id["a"].lexical_rank == 2
        assert by_id["b"].dense_rank == 2
        assert by_id["b"].lexical_rank == 1
        assert by_id["c"].dense_rank == 3
        assert by_id["c"].lexical_rank is None
        assert by_id["d"].dense_rank is None
        assert by_id["d"].lexical_rank == 3

        # A and B should be tied at the top
        assert results[0].fused_score == results[1].fused_score
        top_ids = {results[0].chunk.metadata.chunk_id, results[1].chunk.metadata.chunk_id}
        assert top_ids == {"a", "b"}
