# ADR 004 — Hybrid Retrieval Strategy

## Context

The RAG system retrieves documentation chunks to answer technical questions. Two main retrieval approaches exist:

1. **Dense retrieval**: Semantic similarity using embeddings
2. **Lexical retrieval**: Keyword matching (BM25, TF-IDF)

Technical documentation contains both conceptual content and exact terms (API names, class names, parameters). A single retrieval method may not serve both needs well.

## Problem

How should we retrieve relevant documentation chunks?

## Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Dense only** | Semantic understanding, handles paraphrasing | Misses exact matches, API names |
| **Lexical only** | Exact term matching, fast | No semantic understanding |
| **Hybrid** | Best of both worlds | More complex, needs fusion |
| **Learned sparse** (SPLADE) | Learned term importance | Requires training/fine-tuning |

## Decision

Implement **hybrid retrieval** combining dense and lexical search, with **Reciprocal Rank Fusion (RRF)** for score combination.

### Rationale

1. **Technical documentation specificity**: Users often search for exact terms like `ChatGroq`, `BaseModel`, or `bind_tools`. Lexical retrieval excels here.

2. **Conceptual questions**: Users also ask "how does RAG reduce hallucination?" — dense retrieval handles semantic similarity.

3. **RRF simplicity**: Reciprocal Rank Fusion is simple, doesn't require score normalization, and works well in practice.

4. **Qdrant support**: Qdrant natively supports sparse vectors, making hybrid search efficient.

### RRF Formula

```python
def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)

# Combined score for a document appearing in both result sets
final_score = rrf_score(dense_rank) + rrf_score(lexical_rank)
```

### Why RRF?

- **No score normalization needed**: Dense and lexical scores have different scales
- **Rank-based**: Only uses relative position, not absolute scores
- **Simple**: No hyperparameters to tune (k=60 is standard)
- **Proven**: Widely used in production systems

## Consequences

### Positive

- Better retrieval for both conceptual and specific queries
- Measurable comparison vs. dense-only baseline
- Industry-standard approach for documentation search
- Qdrant's sparse vector support makes implementation clean

### Negative

- Two retrieval calls (can be parallelized)
- Slightly higher latency
- BM25 index must be built and maintained
- More complex testing (need to verify fusion logic)

### Implementation Phases

**Step 8 (MVP)**: Dense retrieval only
**Step 10**: Add lexical (BM25) and hybrid fusion
**Step 11**: Add reranking on top

### Configuration

```python
# In config.py (to be added)
RETRIEVAL_MODE=hybrid     # dense | lexical | hybrid
HYBRID_DENSE_WEIGHT=0.5   # Weight for dense in hybrid (0-1)
RETRIEVAL_TOP_K=20        # Candidates before fusion
FINAL_TOP_K=5             # Results after fusion
```

### BM25 Options

| Option | Pros | Cons |
|--------|------|------|
| **Qdrant sparse vectors** | Native, efficient | Requires sparse encoding |
| **rank-bm25 library** | Simple Python | In-memory, no persistence |
| **Elasticsearch** | Mature, scalable | Another service to run |

Decision: Start with **`rank-bm25`** for simplicity, migrate to Qdrant sparse vectors if needed.

### Evaluation Plan

Compare configurations using retrieval evaluation dataset:

| Config | Dense | Lexical | Fusion | Recall@5 | MRR |
|--------|-------|---------|--------|----------|-----|
| A | yes | no | - | TBD | TBD |
| B | no | yes | - | TBD | TBD |
| C | yes | yes | RRF | TBD | TBD |

Hybrid should outperform single methods on a mixed query set (conceptual + specific).

## Related Decisions

- ADR 002: Qdrant as vector store
- ADR 003: Local embedding models
- ADR 005: Reranking strategy
