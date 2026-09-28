"""
T²-RAGBench dataset loader.

This module loads the T²-RAGBench dataset from Hugging Face Hub and converts
it to our internal DocumentChunk schema with rich provenance metadata.

Dataset Overview
----------------
T²-RAGBench is a benchmark for RAG on financial documents with text and tables:
- 3 subsets: FinQA, ConvFinQA, TAT-DQA
- ~23,000 QA pairs across ~7,300 unique document contexts
- Pre-extracted text and tables from SEC financial filings
- Context-independent reformulated questions for fair retrieval evaluation
- License: CC-BY-4.0

Reference: https://huggingface.co/datasets/G4KMU/t2-ragbench
Paper: "T²-RAGBench: Text-and-Table Benchmark for Evaluating RAG" (EACL 2026)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from datasets import load_dataset  # type: ignore[import-untyped]

from rag_assistant.schemas.chunk import ChunkMetadata, DocumentChunk, generate_chunk_id

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

RAGBenchSubset = Literal["FinQA", "ConvFinQA", "TAT-DQA"]


@dataclass
class RAGBenchQA:
    """
    A question-answer pair from T²-RAGBench.

    This represents a single evaluation example with the question,
    ground truth answer, and reference to the relevant context.

    Attributes
    ----------
    id
        Unique identifier for this QA pair.
    question
        The context-independent reformulated question.
    answer
        Ground truth answer (numeric for this dataset).
    context_id
        ID of the relevant document context.
    subset
        Which subset this came from (FinQA, ConvFinQA, TAT-DQA).
    split
        Original split (train, dev, test, turn_0).
    """

    id: str
    question: str
    answer: str
    context_id: str
    subset: str
    split: str


@dataclass
class RAGBenchContext:
    """
    A document context from T²-RAGBench.

    Each context represents a page or section from a financial document,
    potentially containing both text and tabular data.

    Attributes
    ----------
    context_id
        Unique identifier for this context.
    text
        Full extracted text content (includes text and table).
    table
        Table content in markdown format (if present).
    pre_text
        Text before the table (FinQA/ConvFinQA only).
    post_text
        Text after the table (FinQA/ConvFinQA only).
    file_name
        Source PDF reference (e.g., "pdf/AAPL/2018/page_45.pdf").
    company_name
        Company name from the financial report.
    company_symbol
        Stock ticker symbol (FinQA/ConvFinQA only).
    report_year
        Year of the financial report.
    page_number
        Page number in the source PDF (FinQA/ConvFinQA only).
    company_sector
        Industry sector classification.
    """

    context_id: str
    text: str
    table: str | None = None
    pre_text: str | None = None
    post_text: str | None = None
    file_name: str | None = None
    company_name: str | None = None
    company_symbol: str | None = None
    report_year: str | None = None
    page_number: str | None = None
    company_sector: str | None = None


@dataclass
class RAGBenchData:
    """
    Loaded T²-RAGBench data.

    Contains both the document contexts (for indexing) and
    QA pairs (for evaluation).
    """

    contexts: dict[str, RAGBenchContext] = field(default_factory=dict)
    qa_pairs: list[RAGBenchQA] = field(default_factory=list)

    @property
    def num_contexts(self) -> int:
        return len(self.contexts)

    @property
    def num_qa_pairs(self) -> int:
        return len(self.qa_pairs)


def load_ragbench_subset(
    subset: RAGBenchSubset,
    split: str | None = None,
    max_samples: int | None = None,
    cache_dir: str | None = None,
) -> RAGBenchData:
    """
    Load a single subset of T²-RAGBench.

    Parameters
    ----------
    subset
        Which subset to load: "FinQA", "ConvFinQA", or "TAT-DQA".
    split
        Optional: specific split to load (train, dev, test, turn_0).
        If None, loads all available splits.
    max_samples
        Optional: limit number of QA pairs loaded (for testing).
    cache_dir
        Optional: directory for caching downloaded data.

    Returns
    -------
    RAGBenchData
        Loaded contexts and QA pairs.
    """
    logger.info("Loading T2-RAGBench subset: %s", subset)

    ds = load_dataset(
        "G4KMU/t2-ragbench",
        subset,
        cache_dir=cache_dir,
    )

    data = RAGBenchData()
    total_loaded = 0

    splits_to_load = [split] if split else list(ds.keys())

    for split_name in splits_to_load:
        if split_name not in ds:
            logger.warning("Split '%s' not found in subset '%s'", split_name, subset)
            continue

        split_ds: Dataset = ds[split_name]

        for row in split_ds:
            if max_samples and total_loaded >= max_samples:
                break

            # Extract context
            context_id = row["context_id"]
            if context_id not in data.contexts:
                data.contexts[context_id] = RAGBenchContext(
                    context_id=context_id,
                    text=row["context"],
                    table=row.get("table"),
                    pre_text=row.get("pre_text"),
                    post_text=row.get("post_text"),
                    file_name=row.get("file_name"),
                    company_name=row.get("company_name"),
                    company_symbol=row.get("company_symbol"),
                    report_year=str(row.get("report_year")) if row.get("report_year") else None,
                    page_number=str(row.get("page_number")) if row.get("page_number") else None,
                    company_sector=row.get("company_sector"),
                )

            # Extract QA pair
            data.qa_pairs.append(
                RAGBenchQA(
                    id=row["id"],
                    question=row["question"],
                    answer=str(row["program_answer"]),
                    context_id=context_id,
                    subset=subset,
                    split=split_name,
                )
            )

            total_loaded += 1

        if max_samples and total_loaded >= max_samples:
            break

    logger.info(
        "Loaded %d QA pairs and %d unique contexts from %s",
        len(data.qa_pairs),
        len(data.contexts),
        subset,
    )

    return data


def load_ragbench(
    subsets: list[RAGBenchSubset] | None = None,
    max_samples_per_subset: int | None = None,
    cache_dir: str | None = None,
) -> RAGBenchData:
    """
    Load T²-RAGBench dataset (one or more subsets).

    Parameters
    ----------
    subsets
        List of subsets to load. Default: all three.
    max_samples_per_subset
        Optional: limit samples per subset (for testing).
    cache_dir
        Optional: directory for caching downloaded data.

    Returns
    -------
    RAGBenchData
        Combined contexts and QA pairs from all requested subsets.
    """
    if subsets is None:
        subsets = ["FinQA", "ConvFinQA", "TAT-DQA"]

    combined = RAGBenchData()

    for subset in subsets:
        data = load_ragbench_subset(
            subset=subset,
            max_samples=max_samples_per_subset,
            cache_dir=cache_dir,
        )

        # Merge contexts (avoid duplicates by ID)
        for ctx_id, ctx in data.contexts.items():
            if ctx_id not in combined.contexts:
                combined.contexts[ctx_id] = ctx

        # Add all QA pairs
        combined.qa_pairs.extend(data.qa_pairs)

    logger.info(
        "Total loaded: %d QA pairs, %d unique contexts",
        len(combined.qa_pairs),
        len(combined.contexts),
    )

    return combined


def context_to_chunk(context: RAGBenchContext, subset: str = "ragbench") -> DocumentChunk:
    """
    Convert a RAGBenchContext to a DocumentChunk.

    Creates rich metadata for citations including company info,
    report year, page number, and sector classification.

    Parameters
    ----------
    context
        The RAGBench context to convert.
    subset
        Subset name for the source field.

    Returns
    -------
    DocumentChunk
        Chunk ready for indexing.
    """
    # Build title from company and year
    company = context.company_name or "Unknown"
    year = context.report_year or "Unknown"
    title = f"{company} ({year})"

    # Build URL from file_name (these are references, not real URLs)
    url = f"t2-ragbench://{context.file_name}" if context.file_name else f"t2-ragbench://{context.context_id}"

    # Build section info from page and sector
    section_parts = []
    if context.page_number:
        section_parts.append(f"Page {context.page_number}")
    if context.company_sector:
        section_parts.append(context.company_sector)
    section = " | ".join(section_parts) if section_parts else None

    # Generate stable chunk ID
    chunk_id = generate_chunk_id(
        source=f"ragbench-{subset.lower()}",
        url=url,
        text=context.text,
    )

    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        source=f"ragbench-{subset.lower()}",
        title=title,
        url=url,
        document_type="financial_report",
        section=section,
        version=context.report_year,
    )

    return DocumentChunk(text=context.text, metadata=metadata)


def ragbench_to_chunks(data: RAGBenchData) -> list[DocumentChunk]:
    """
    Convert all RAGBench contexts to DocumentChunks.

    Parameters
    ----------
    data
        Loaded RAGBench data.

    Returns
    -------
    list[DocumentChunk]
        Chunks ready for indexing.
    """
    chunks: list[DocumentChunk] = []

    # Group contexts by subset for proper source tagging
    for ctx in data.contexts.values():
        # Infer subset from context_id prefix
        if ctx.context_id.startswith("finqa"):
            subset = "finqa"
        elif ctx.context_id.startswith("convfinqa"):
            subset = "convfinqa"
        elif ctx.context_id.startswith("tat"):
            subset = "tatdqa"
        else:
            subset = "ragbench"

        chunk = context_to_chunk(ctx, subset=subset)
        chunks.append(chunk)

    return chunks


@dataclass
class EvalExample:
    """
    Single evaluation example with question and gold context.

    Attributes
    ----------
    id
        Example identifier.
    question
        The question text.
    answer
        Ground truth answer.
    gold_context_ids
        IDs of relevant contexts (chunk_ids after conversion).
    subset
        Source subset name.
    """

    id: str
    question: str
    answer: str
    gold_context_ids: list[str]
    subset: str


def create_eval_set(
    data: RAGBenchData,
    output_path: str | Path | None = None,
) -> list[EvalExample]:
    """
    Create an evaluation set from RAGBench data.

    Maps QA pairs to their gold context chunk_ids for retrieval evaluation.

    Parameters
    ----------
    data
        Loaded RAGBench data.
    output_path
        Optional: path to save as JSONL.

    Returns
    -------
    list[EvalExample]
        Evaluation examples ready for Recall@K, MRR computation.
    """
    # Pre-compute context_id -> chunk_id mapping
    context_to_chunk_id: dict[str, str] = {}
    for ctx in data.contexts.values():
        if ctx.context_id.startswith("finqa"):
            subset = "finqa"
        elif ctx.context_id.startswith("convfinqa"):
            subset = "convfinqa"
        elif ctx.context_id.startswith("tat"):
            subset = "tatdqa"
        else:
            subset = "ragbench"

        chunk = context_to_chunk(ctx, subset=subset)
        context_to_chunk_id[ctx.context_id] = chunk.chunk_id

    # Create eval examples
    examples: list[EvalExample] = []
    for qa in data.qa_pairs:
        chunk_id = context_to_chunk_id.get(qa.context_id)
        if chunk_id:
            examples.append(
                EvalExample(
                    id=qa.id,
                    question=qa.question,
                    answer=qa.answer,
                    gold_context_ids=[chunk_id],
                    subset=qa.subset,
                )
            )

    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for ex in examples:
                f.write(
                    json.dumps(
                        {
                            "id": ex.id,
                            "question": ex.question,
                            "answer": ex.answer,
                            "gold_context_ids": ex.gold_context_ids,
                            "subset": ex.subset,
                        }
                    )
                    + "\n"
                )

        logger.info("Saved %d eval examples to %s", len(examples), output_path)

    return examples


def load_eval_set(path: str | Path) -> list[EvalExample]:
    """
    Load an evaluation set from JSONL file.

    Parameters
    ----------
    path
        Path to the JSONL file.

    Returns
    -------
    list[EvalExample]
        Loaded evaluation examples.
    """
    examples: list[EvalExample] = []

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(
                EvalExample(
                    id=data["id"],
                    question=data["question"],
                    answer=data["answer"],
                    gold_context_ids=data["gold_context_ids"],
                    subset=data["subset"],
                )
            )

    return examples
