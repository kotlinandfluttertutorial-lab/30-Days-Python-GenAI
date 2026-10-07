# Day 30 — NotebookLM Notes: Job Readiness

## 30-Day Achievement Summary

You can now:
- Build production ML prediction services (Day 9)
- Implement transformers and attention from scratch (Day 14)
- Build complete RAG pipelines with evaluation (Days 19-22)
- Build AI agents with tool calling (Days 23-25)
- Deploy production FastAPI AI backends (Day 26)
- Containerize and deploy with Docker (Day 27)
- Implement LLMOps, security, guardrails (Day 28)
- Design enterprise AI systems (Day 29)

## Interview Cheat Sheet

### One-Line Answers (use in phone screens)

- **RAG**: "Retrieval-Augmented Generation — chunks docs, embeds, stores in vector DB, retrieves relevant chunks at query time to ground LLM responses in facts."
- **Hallucination**: "LLM generates statistically likely tokens without truth verification — no ground truth mechanism exists."
- **Embedding**: "Dense vector mapping text to semantic space — similar meanings produce similar vectors measured by cosine similarity."
- **Agent**: "LLM + tools + memory in a ReAct loop: think → act → observe → repeat until final answer."
- **Why RAG over fine-tuning**: "RAG: instant updates, source citations, no GPU needed, private docs stay local."
- **Transformer attention**: "Attention(Q,K,V) = softmax(QK^T / √d_k) × V — each token attends to all others."
- **Why async**: "LLM calls are I/O-bound; async lets one thread handle thousands concurrent without blocking."

### Numbers to Memorize

```
GPT-4o context:         128K tokens
Claude 3.5 context:     200K tokens
Vocab size GPT:         ~50K tokens
all-MiniLM-L6-v2:       384 dimensions
text-embedding-3-small: 1536 dimensions
GPT-4o input price:     $5/1M tokens
GPT-4o-mini price:      $0.15/1M tokens
Default chunk size:     512 tokens
Default overlap:        50 tokens
RAG P95 target:         <3 seconds
```

## Questions That Stump Candidates

1. "When would you NOT use agents?" → Use RAG or workflows; agents are complex, slow, expensive
2. "Is temperature=0 truly deterministic?" → Near-deterministic; quantization effects exist
3. "Why does RAG still hallucinate?" → LLM can ignore context or context can be incomplete
4. "What's p95 latency?" → 95% of requests complete within this time
5. "Hybrid search vs pure vector?" → BM25 handles keywords, vector handles semantics — both needed

## Interview Facts

1. RAG ingestion = offline, query = online per request
2. Same embedding model for indexing AND querying (critical rule)
3. Cross-encoder reranker: slow but accurate — use for top-20→top-3 reranking only
4. `asyncio.gather()` = concurrent; sequential awaits = no concurrency benefit
5. Never expose raw exceptions to API clients (security + UX)
6. p95 more important than average for SLA monitoring
7. Content hash dedup: same document uploaded twice → skip re-ingestion
8. Prompt is code: version it, test it, rollback-capable
