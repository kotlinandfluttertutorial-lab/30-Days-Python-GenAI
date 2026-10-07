# Day 21 — NotebookLM Notes: Production RAG

## Production Concerns

**Async ingestion**: Documents arrive at different times. Process asynchronously, don't block HTTP requests. Use `asyncio.to_thread()` for CPU-bound operations in async context.

**Deduplication**: Hash document content. Don't re-embed unchanged documents. Use `sha256(text)[:16]` as deduplication key.

**Caching**: Same question, don't pay for LLM again. Hash the query string as cache key. Set TTL (1 hour typical). Invalidate cache when documents are updated.

**Retry + backoff**: LLM APIs fail. Retry 3 times with 1s, 2s, 4s delays (exponential). Only retry on transient errors (rate limit, timeout). Don't retry on auth errors.

**Timeout**: Set max time for LLM calls (~30s). Never wait forever. Return cached fallback or error if timeout.

**Rate limiting**: Token bucket per user. Prevent API cost abuse. Return 429 with retry-after header.

**Observability**: Log every request with: trace_id, cache_hit, retrieval_ms, llm_ms, total_ms, tokens, cost_usd.

## Production Checklist

```
Ingestion:
  [ ] Async processing (don't block HTTP)
  [ ] Content hash deduplication
  [ ] Batch embedding (not one-by-one)
  [ ] Retry on embedding failures
  [ ] Progress tracking

Query:
  [ ] Cache check before everything
  [ ] Timeout on LLM call (30s)
  [ ] Retry with backoff (3 attempts)
  [ ] Rate limiting per user
  [ ] Full trace logging
  [ ] Cost tracking per request
```

## Interview Facts

1. `asyncio.to_thread()`: run sync (blocking) code in async context without blocking event loop
2. Redis cache TTL: invalidate when knowledge base changes, not just time-based
3. Exponential backoff: 1s, 2s, 4s — prevents thundering herd on API recovery
4. Rate limiting token bucket: smoother than fixed window counter
5. Trace IDs: correlate all spans in a request (retrieval, LLM, cache) for debugging
6. Content hash: detect duplicate documents across ingestion runs

## Cost Optimization

1. Cache frequent queries (30% cache hit = 30% cost reduction)
2. Smaller chunks = fewer tokens in context
3. Reduce top_k (3 chunks vs 5 chunks = 40% fewer context tokens)
4. Route simple queries to cheaper model (gpt-4o-mini vs gpt-4o)
5. Compress context: extract key sentences instead of full chunks
