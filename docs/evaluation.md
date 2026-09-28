# RAG Evaluation Framework

This document describes the evaluation strategy for the RAG system.

## Overview

Evaluation is a first-class feature, not an afterthought. Every change to the retrieval pipeline must be measured against a baseline to ensure it actually improves the system.

## T²-RAGBench Baseline

We use the [T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench) dataset for evaluation. This benchmark provides:

- **23,088 QA pairs** across financial documents with text and tables
- **7,300+ unique document contexts** from SEC filings
- **3 subsets**: FinQA, ConvFinQA, TAT-DQA
- **Context-independent questions** (reformulated for fair retrieval evaluation)
- **License**: CC-BY-4.0

### Baseline Measurements (FinQA subset, n=50 queries, 192 documents)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Latency (ms) |
|--------|----------|----------|-----------|-----|--------------|
| **Dense** | 0.480 | 0.600 | 0.660 | 0.536 | 6.3 |
| **BM25** | 0.840 | 0.960 | 0.960 | 0.893 | 0.7 |
| **Hybrid** | 0.520 | 0.840 | 0.980 | 0.656 | 7.2 |

**Key observations**:
- BM25 excels on this dataset due to exact keyword matching (financial terms, numbers)
- Hybrid achieves the highest Recall@10 (0.98), combining the best of both methods
- Dense retrieval struggles with exact financial terminology

### Metric Targets (based on baseline)

| Metric | Current Best | Target | Notes |
|--------|-------------|--------|-------|
| **Recall@5** | 0.96 (BM25) | > 0.90 | Maintain BM25 performance |
| **Recall@10** | 0.98 (Hybrid) | > 0.95 | Hybrid for maximum recall |
| **MRR** | 0.89 (BM25) | > 0.80 | First result should be relevant |

## Evaluation Dimensions

### 1. Retrieval Quality

How well does the system retrieve relevant documents?

| Metric | Definition | Target |
|--------|-----------|--------|
| **Recall@K** | % of relevant docs in top K results | > 90% (K=5) |
| **MRR** | Mean Reciprocal Rank of first relevant doc | > 0.8 |
| **Precision@K** | % of top K results that are relevant | > 60% |

### 2. Generation Quality

How good are the generated answers?

| Metric | Definition | Target |
|--------|-----------|--------|
| **Faithfulness** | Is the answer supported by the context? | > 90% |
| **Relevance** | Does the answer address the question? | > 85% |
| **Citation Accuracy** | Are citations correct and complete? | > 95% |

### 3. System Performance

| Metric | Definition | Target |
|--------|-----------|--------|
| **Retrieval Latency (p50)** | Median retrieval time | < 100ms |
| **Retrieval Latency (p95)** | 95th percentile | < 300ms |
| **End-to-end Latency** | Query to answer | < 3s |
| **Token Usage** | Tokens per query (context + generation) | < 4000 |

## Evaluation Dataset

### Structure

```json
{
  "version": "1.0",
  "created_at": "2024-01-01",
  "questions": [
    {
      "id": "q001",
      "question": "How do I create a LangChain agent with tools?",
      "category": "practical",
      "difficulty": "intermediate",
      "expected_sources": ["langchain"],
      "expected_concepts": ["agent", "tools", "bind_tools"],
      "reference_answer": "To create a LangChain agent with tools...",
      "relevant_chunk_ids": ["abc123", "def456"]
    }
  ]
}
```

### Question Categories

| Category | Description | Example |
|----------|-------------|---------|
| `conceptual` | "What is X?" questions | "What is RAG?" |
| `practical` | "How do I X?" questions | "How do I add a tool?" |
| `specific` | Exact API/class questions | "What parameters does ChatGroq accept?" |
| `comparison` | "X vs Y" questions | "Dense vs hybrid retrieval?" |
| `debug` | "Why does X fail?" questions | "Why is my retriever returning empty?" |

### Dataset Requirements

- Minimum 50 questions for reliable metrics
- Balanced across categories and sources
- Ground truth manually verified
- Versioned and tracked in git

### Sample Questions

```json
[
  {
    "id": "q001",
    "question": "What is the difference between LangChain and LangGraph?",
    "category": "comparison",
    "expected_sources": ["langchain", "langgraph"],
    "expected_concepts": ["chain", "graph", "state", "workflow"]
  },
  {
    "id": "q002",
    "question": "How do I configure Qdrant for hybrid search?",
    "category": "practical",
    "expected_sources": ["qdrant"],
    "expected_concepts": ["sparse vectors", "dense vectors", "collection"]
  },
  {
    "id": "q003",
    "question": "What is the BaseModel.model_validate method?",
    "category": "specific",
    "expected_sources": ["pydantic"],
    "expected_concepts": ["validation", "model_validate", "dict"]
  }
]
```

## Metric Implementations

### Recall@K

```python
def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = |retrieved ∩ relevant| / |relevant|
    
    Parameters
    ----------
    retrieved_ids
        IDs of retrieved chunks, ordered by score.
    relevant_ids
        IDs of chunks marked as relevant in ground truth.
    k
        Number of top results to consider.
    
    Returns
    -------
    float
        Recall score between 0 and 1.
    """
    if not relevant_ids:
        return 1.0  # No relevant docs = trivially complete
    
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    
    return len(top_k & relevant) / len(relevant)
```

### Mean Reciprocal Rank (MRR)

```python
def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """
    Calculate reciprocal rank of first relevant result.
    
    RR = 1 / rank_of_first_relevant
    
    Returns 0 if no relevant document is retrieved.
    """
    relevant = set(relevant_ids)
    
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    
    return 0.0

def mean_reciprocal_rank(results: list[tuple[list[str], list[str]]]) -> float:
    """Calculate MRR over multiple queries."""
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in results) / len(results)
```

### Faithfulness (LLM-as-Judge)

```python
FAITHFULNESS_PROMPT = """
You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Is the answer fully supported by the context? Consider:
1. Does every claim in the answer have support in the context?
2. Does the answer add information not present in the context?
3. Does the answer contradict the context?

Respond with ONLY a JSON object:
{
    "faithful": true/false,
    "explanation": "brief explanation",
    "unsupported_claims": ["list of claims not supported by context"]
}
"""
```

## Experiment Workflow

### 1. Define Configurations

```python
configurations = [
    {
        "name": "baseline_dense",
        "retrieval": "dense",
        "reranking": False,
    },
    {
        "name": "hybrid",
        "retrieval": "hybrid",
        "reranking": False,
    },
    {
        "name": "hybrid_reranked",
        "retrieval": "hybrid",
        "reranking": True,
    },
]
```

### 2. Run Evaluation

```bash
# Run evaluation for all configurations
uv run python scripts/evaluate_rag.py \
    --dataset experiments/datasets/eval_v1.json \
    --configs baseline_dense,hybrid,hybrid_reranked \
    --output experiments/results/
```

### 3. Compare Results

| Config | Recall@5 | MRR | Latency (p50) |
|--------|----------|-----|---------------|
| baseline_dense | 0.72 | 0.65 | 45ms |
| hybrid | 0.81 | 0.73 | 85ms |
| hybrid_reranked | 0.84 | 0.79 | 180ms |

### 4. Document Findings

```markdown
## Experiment: Hybrid vs Dense Retrieval

**Date**: 2024-01-15
**Dataset**: eval_v1.json (50 questions)

### Results

Hybrid retrieval improved Recall@5 by 12.5% over dense-only,
with an acceptable latency increase of ~40ms.

Reranking provided an additional 3.7% improvement but doubled latency.
Recommend enabling hybrid, deferring reranking until latency budget allows.
```

## LangSmith Integration

### Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-assistant-eval"
```

### Evaluation Datasets

LangSmith can store evaluation datasets and track results over time:

```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset("rag-eval-v1")

# Add examples
for q in questions:
    client.create_example(
        inputs={"question": q["question"]},
        outputs={"expected_sources": q["expected_sources"]},
        dataset_id=dataset.id,
    )
```

### Running Evaluations

```python
from langsmith.evaluation import evaluate

results = evaluate(
    rag_pipeline,
    data="rag-eval-v1",
    evaluators=[
        faithfulness_evaluator,
        relevance_evaluator,
    ],
    experiment_prefix="hybrid-retrieval",
)
```

## File Structure

```
experiments/
├── datasets/
│   ├── eval_v1.json           # Main evaluation dataset
│   └── eval_specific.json     # API-specific questions
├── results/
│   ├── 2024-01-15_baseline/
│   │   ├── metrics.json
│   │   └── details.csv
│   └── 2024-01-15_hybrid/
│       ├── metrics.json
│       └── details.csv
└── reports/
    └── 2024-01_retrieval_comparison.md
```

## Best Practices

1. **Version datasets**: Track changes to evaluation questions
2. **Document methodology**: Every experiment gets a markdown report
3. **No cherry-picking**: Report all results, even negative ones
4. **Reproducibility**: Pin random seeds, model versions, and dependencies
5. **Statistical significance**: Use multiple runs for noisy metrics
6. **Baseline comparison**: Always compare against a known baseline
7. **Incremental changes**: Change one variable at a time

## References

- [RAGAS: RAG Assessment](https://docs.ragas.io/)
- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
