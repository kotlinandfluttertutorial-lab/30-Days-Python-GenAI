# Day 20 — Day Summary: Advanced RAG

## What You Covered
Hybrid search (BM25 + vector), two-stage retrieve-then-rerank, query rewriting for conversational context, multi-query retrieval for complex questions, semantic chunking.

## Key Takeaways
1. Hybrid search: BM25 handles keywords, vector handles semantics — combine both
2. Two-stage: fast retrieval (top-20) → slow accurate reranking (top-3)
3. Query rewriting converts ambiguous follow-ups to self-contained queries
4. Multi-query: decompose complex questions, retrieve all aspects, deduplicate
5. These techniques together close the gap between prototype RAG and production RAG

## Tomorrow: Day 21 — Production RAG
Async ingestion pipelines, caching, retries, timeouts, observability, cost optimization.
