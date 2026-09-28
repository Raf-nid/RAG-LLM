"""Tests for RAGBench loader (offline fixtures, no network)."""

import json
import tempfile
from pathlib import Path

import pytest

from rag_assistant.rag.ingestion.ragbench_loader import (
    EvalExample,
    RAGBenchContext,
    RAGBenchData,
    RAGBenchQA,
    context_to_chunk,
    create_eval_set,
    load_eval_set,
    ragbench_to_chunks,
)
from rag_assistant.schemas.chunk import DocumentChunk


@pytest.fixture
def sample_context() -> RAGBenchContext:
    """Create a sample context for testing."""
    return RAGBenchContext(
        context_id="finqa_train_ctx_123",
        text="Apple Inc. reported revenue of $394 billion in fiscal 2022.",
        table="| Year | Revenue |\n|------|------|\n| 2022 | $394B |",
        pre_text="Financial highlights from the annual report.",
        post_text="See notes for details.",
        file_name="pdf/AAPL/2022/page_45.pdf",
        company_name="Apple Inc.",
        company_symbol="AAPL",
        report_year="2022",
        page_number="45",
        company_sector="Information Technology",
    )


@pytest.fixture
def sample_qa() -> RAGBenchQA:
    """Create a sample QA pair."""
    return RAGBenchQA(
        id="finqa_train_0",
        question="What was Apple's revenue in fiscal 2022?",
        answer="394",
        context_id="finqa_train_ctx_123",
        subset="FinQA",
        split="train",
    )


@pytest.fixture
def sample_data(sample_context: RAGBenchContext, sample_qa: RAGBenchQA) -> RAGBenchData:
    """Create sample RAGBench data."""
    return RAGBenchData(
        contexts={sample_context.context_id: sample_context},
        qa_pairs=[sample_qa],
    )


class TestRAGBenchContext:
    """Tests for RAGBenchContext dataclass."""

    def test_basic_creation(self, sample_context: RAGBenchContext) -> None:
        assert sample_context.context_id == "finqa_train_ctx_123"
        assert "Apple Inc." in sample_context.text
        assert sample_context.company_name == "Apple Inc."

    def test_optional_fields(self) -> None:
        ctx = RAGBenchContext(
            context_id="test_ctx",
            text="Some text content",
        )
        assert ctx.table is None
        assert ctx.company_name is None
        assert ctx.company_sector is None


class TestRAGBenchData:
    """Tests for RAGBenchData dataclass."""

    def test_num_contexts(self, sample_data: RAGBenchData) -> None:
        assert sample_data.num_contexts == 1

    def test_num_qa_pairs(self, sample_data: RAGBenchData) -> None:
        assert sample_data.num_qa_pairs == 1

    def test_empty_data(self) -> None:
        data = RAGBenchData()
        assert data.num_contexts == 0
        assert data.num_qa_pairs == 0


class TestContextToChunk:
    """Tests for context_to_chunk conversion."""

    def test_creates_document_chunk(self, sample_context: RAGBenchContext) -> None:
        chunk = context_to_chunk(sample_context, subset="finqa")

        assert isinstance(chunk, DocumentChunk)
        assert chunk.text == sample_context.text

    def test_sets_source_with_subset(self, sample_context: RAGBenchContext) -> None:
        chunk = context_to_chunk(sample_context, subset="finqa")
        assert chunk.metadata.source == "ragbench-finqa"

    def test_creates_title_from_company_and_year(
        self, sample_context: RAGBenchContext
    ) -> None:
        chunk = context_to_chunk(sample_context, subset="finqa")
        assert chunk.metadata.title == "Apple Inc. (2022)"

    def test_creates_url_from_filename(self, sample_context: RAGBenchContext) -> None:
        chunk = context_to_chunk(sample_context, subset="finqa")
        assert "t2-ragbench://" in chunk.metadata.url
        assert "AAPL" in chunk.metadata.url

    def test_includes_page_and_sector_in_section(
        self, sample_context: RAGBenchContext
    ) -> None:
        chunk = context_to_chunk(sample_context, subset="finqa")
        assert chunk.metadata.section is not None
        assert "Page 45" in chunk.metadata.section
        assert "Information Technology" in chunk.metadata.section

    def test_generates_stable_chunk_id(self, sample_context: RAGBenchContext) -> None:
        chunk1 = context_to_chunk(sample_context, subset="finqa")
        chunk2 = context_to_chunk(sample_context, subset="finqa")

        # Same content should produce same ID
        assert chunk1.chunk_id == chunk2.chunk_id

    def test_handles_missing_company(self) -> None:
        ctx = RAGBenchContext(
            context_id="test_ctx",
            text="Some financial data",
        )
        chunk = context_to_chunk(ctx, subset="test")

        assert chunk.metadata.title == "Unknown (Unknown)"


class TestRagbenchToChunks:
    """Tests for ragbench_to_chunks function."""

    def test_converts_all_contexts(self, sample_data: RAGBenchData) -> None:
        chunks = ragbench_to_chunks(sample_data)
        assert len(chunks) == 1

    def test_infers_subset_from_context_id(self) -> None:
        data = RAGBenchData(
            contexts={
                "finqa_ctx_1": RAGBenchContext(
                    context_id="finqa_ctx_1",
                    text="FinQA content",
                ),
                "convfinqa_ctx_2": RAGBenchContext(
                    context_id="convfinqa_ctx_2",
                    text="ConvFinQA content",
                ),
                "tat_ctx_3": RAGBenchContext(
                    context_id="tat_ctx_3",
                    text="TAT-DQA content",
                ),
            },
            qa_pairs=[],
        )

        chunks = ragbench_to_chunks(data)

        sources = {c.metadata.source for c in chunks}
        assert "ragbench-finqa" in sources
        assert "ragbench-convfinqa" in sources
        assert "ragbench-tatdqa" in sources


class TestEvalSetCreation:
    """Tests for evaluation set creation."""

    def test_create_eval_set(self, sample_data: RAGBenchData) -> None:
        examples = create_eval_set(sample_data)

        assert len(examples) == 1
        assert examples[0].question == "What was Apple's revenue in fiscal 2022?"
        assert examples[0].answer == "394"
        assert examples[0].subset == "FinQA"

    def test_maps_context_to_chunk_id(self, sample_data: RAGBenchData) -> None:
        examples = create_eval_set(sample_data)

        # Gold context ID should be a valid chunk ID (UUID format)
        assert len(examples[0].gold_context_ids) == 1
        chunk_id = examples[0].gold_context_ids[0]
        assert "-" in chunk_id  # UUID has dashes

    def test_saves_to_file(self, sample_data: RAGBenchData) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            examples = create_eval_set(sample_data, output_path=f.name)

            # Check file was created
            path = Path(f.name)
            assert path.exists()

            # Check content
            with open(path) as fp:
                line = fp.readline()
                data = json.loads(line)
                assert data["question"] == examples[0].question


class TestLoadEvalSet:
    """Tests for loading evaluation set from file."""

    def test_load_eval_set(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            # Write test data
            f.write(
                json.dumps(
                    {
                        "id": "test_1",
                        "question": "What is 2+2?",
                        "answer": "4",
                        "gold_context_ids": ["chunk_abc"],
                        "subset": "test",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "id": "test_2",
                        "question": "What is 3+3?",
                        "answer": "6",
                        "gold_context_ids": ["chunk_def"],
                        "subset": "test",
                    }
                )
                + "\n"
            )
            f.flush()

            examples = load_eval_set(f.name)

        assert len(examples) == 2
        assert examples[0].id == "test_1"
        assert examples[0].question == "What is 2+2?"
        assert examples[1].gold_context_ids == ["chunk_def"]


class TestEvalExample:
    """Tests for EvalExample dataclass."""

    def test_basic_creation(self) -> None:
        ex = EvalExample(
            id="ex_1",
            question="Sample question?",
            answer="42",
            gold_context_ids=["chunk_1", "chunk_2"],
            subset="FinQA",
        )

        assert ex.id == "ex_1"
        assert len(ex.gold_context_ids) == 2
