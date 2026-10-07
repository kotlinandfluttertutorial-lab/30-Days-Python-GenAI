# Day 21 — Day Summary: Production RAG

## What You Covered
Async ingestion pipeline with deduplication, Redis caching, retry with exponential backoff, timeout handling, rate limiting, full observability traces.

## Key Takeaways
1. Async ingestion: `asyncio.to_thread()` runs blocking embed/store code without blocking the event loop
2. Content hash deduplication: don't re-embed unchanged documents
3. Redis cache: same query = cache hit = instant response + zero LLM cost
4. Exponential backoff: 1s → 2s → 4s for transient API failures
5. Trace logging: every request must log trace_id, latency, tokens, cost for debugging

## Tomorrow: Day 22 — RAG Evaluation
Measuring faithfulness, relevance, groundedness — RAGAS framework — building an evaluation dataset.
