# Day 20 — NotebookLM Notes: Advanced RAG

## Why Naive RAG Fails

1. **Keyword mismatch** → BM25 misses semantic; vector misses exact → hybrid search fixes
2. **Low recall** → top-3 misses relevant docs → retrieve top-20, rerank to top-3
3. **Ambiguous queries** → context missing → query rewriting
4. **Multi-aspect queries** → single retrieval misses aspects → multi-query retrieval
5. **Too small chunks** → precise but no context → parent-child retrieval

## Hybrid Search (BM25 + Vector)

```
BM25: exact keyword match → catches "heart attack" when query says "cardiovascular"
Vector: semantic similarity → catches synonyms and paraphrases
Hybrid score = (1-α) × BM25_norm + α × vector_norm
α = 0.5: balanced; α = 0.7: semantic-weighted; α = 0.3: keyword-weighted
Alternative: Reciprocal Rank Fusion (RRF) — rank-based combination
```

## Two-Stage Retrieval (Retrieve + Rerank)

```
Stage 1: Bi-encoder (fast)
  embed query → embed docs → cosine similarity
  Retrieve top-20 candidates in <100ms

Stage 2: Cross-encoder (slow, accurate)
  read (query, doc) together → relevance score
  Rerank top-20 → keep top-3 in ~500ms

Why two stages? Cross-encoder on all 10M docs would take hours.
```

## Query Rewriting

Converts conversational references to self-contained queries:
- "What about evaluation?" → "What are the evaluation metrics for RAG pipelines?"
- Use last N turns of conversation history as context for rewriting

## Interview Facts

1. Hybrid search improves recall significantly over pure vector search
2. RRF (Reciprocal Rank Fusion): combine rankings, not scores (robust to scale differences)
3. Cross-encoder is 10-50x slower than bi-encoder but significantly more accurate
4. Multi-query retrieval: 3 sub-queries → deduplicate → rerank → top-K
5. Semantic chunking: split when cosine similarity between consecutive sentences drops below threshold
6. Parent-child retrieval: index small (precise), return parent chunk (full context)

## Common Mistakes

- Not deduplicating across multi-query results
- Using cross-encoder on the full corpus (too slow — use it only for reranking candidates)
- Hybrid search with un-normalized scores (different scales → meaningless combination)
- Query rewriting every query (adds latency — do it only for follow-up questions)
- Reranking before BM25/vector (waste — rerank is for narrowing down candidates)
