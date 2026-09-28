# ADR 005 — Reranking Strategy

## Context

After initial retrieval (dense, lexical, or hybrid), the candidate set may contain documents that are loosely related but not the most relevant. Reranking uses a more powerful model to re-score candidates.

## Problem

Should we add a reranking step, and if so, which model should we use?

## Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **No reranking** | Simpler, faster | Lower precision |
| **Cross-encoder reranker** | Much better precision | Slower (O(n) inference) |
| **LLM-based reranking** | Very accurate | Expensive, slow |
| **ColBERT-style** | Efficient late interaction | Complex setup |

## Decision

Add **cross-encoder reranking** as an optional step, disabled by default. Enable only if experiments show measurable improvement.

### Rationale

1. **Not always needed**: For simple queries, hybrid retrieval may be sufficient.
2. **Latency cost**: Reranking adds 50-200ms per query.
3. **Measurable benefit**: We should enable reranking only if it improves Recall@5 or answer quality.
4. **Learning value**: Understanding when reranking helps is a key AI engineering skill.

### Cross-Encoder Candidates

| Model | Size | Latency | Quality |
|-------|------|---------|---------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 80MB | ~50ms | Good |
| `BAAI/bge-reranker-base` | 440MB | ~100ms | Better |
| `BAAI/bge-reranker-v2-m3` | 560MB | ~150ms | Best |
| `mixedbread-ai/mxbai-rerank-base-v1` | 440MB | ~100ms | Good for code |

Decision: Start with **`cross-encoder/ms-marco-MiniLM-L-6-v2`** for speed, evaluate larger models if needed.

## Consequences

### Positive

- Improved precision when enabled
- Can be toggled per-query based on importance
- Measurable impact through evaluation
- Industry-standard approach

### Negative

- Additional model to load and run
- Increases query latency
- Requires more memory
- May not provide enough benefit to justify cost

### Implementation

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        pairs = [(query, chunk.chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # Sort by reranker score
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        if top_k:
            ranked = ranked[:top_k]
        
        return [
            RetrievedChunk(
                chunk=chunk.chunk,
                score=float(score),
                retrieval_method="reranked"
            )
            for chunk, score in ranked
        ]
```

### Configuration

```python
# In config.py (to be added)
RERANKING_ENABLED=false
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKING_TOP_K=5      # Number of results after reranking
RERANKING_CANDIDATES=20 # Number of candidates to rerank
```

### Evaluation Plan

Compare with and without reranking:

| Config | Retrieval | Reranking | Recall@5 | MRR | Latency |
|--------|-----------|-----------|----------|-----|---------|
| A | hybrid | no | TBD | TBD | TBD |
| B | hybrid | yes | TBD | TBD | TBD |

Enable reranking only if:
- Recall@5 improves by >5%
- Latency increase is acceptable (<200ms)
- Answer quality improves (manual evaluation)

## Related Decisions

- ADR 004: Hybrid retrieval strategy
- ADR 003: Local embedding models
