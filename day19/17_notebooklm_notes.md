# Day 19 — NotebookLM Notes: RAG Fundamentals

## Why RAG Exists
1. LLMs have knowledge cutoffs → RAG provides current information
2. LLMs hallucinate → RAG grounds responses in retrieved facts
3. LLMs don't know private docs → RAG injects them at query time
4. Users need sources → RAG enables document citations

## Two Pipelines

**INGESTION (offline):**
`Documents → Load → Clean → Chunk → Embed → Store`

**QUERY (online, per request):**
`Query → Embed → Search → Retrieve → Context → Prompt → LLM → Answer`

## Chunking Rules

- Default: 512 tokens, 50-100 token overlap
- Always overlap: prevents information loss at boundaries
- Sentence-aware: don't split mid-sentence
- Smaller chunks = precise retrieval; Larger = more context per match

## Context Construction

```python
context = "\n\n---\n\n".join([
    f"[Source: {chunk['source']}]\n{chunk['text']}"
    for chunk in retrieved_chunks
])
prompt = f"Context:\n{context}\n\nQuestion: {query}"
```

## RAG vs Fine-Tuning Decision

| Use RAG | Use Fine-Tuning |
|---------|----------------|
| Knowledge updates frequently | Style/behavior change |
| Need source citations | Bake in specific format |
| Privacy: keep docs local | Domain-specific vocabulary |
| Limited compute | High query volume |

## Interview Facts

1. Same embedding model MUST be used for indexing AND querying
2. Overlap prevents information loss at chunk boundaries
3. Context window = max tokens including system + context + response
4. Retrieved chunks should be ordered by relevance (highest score first)
5. RAG reduces but doesn't eliminate hallucination — LLM can ignore context
6. Ingestion is offline (amortized cost); query is online (latency-critical)

## Common Mistakes

- Using different embedding model for indexing vs querying (wrong results)
- No overlap between chunks (splits key information across boundaries)
- Including all retrieved chunks even if low score (adds noise)
- Not citing sources in prompt → LLM makes up citations
- Re-ingesting unchanged documents on every startup (use checksums)
