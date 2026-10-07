# Final AI Engineer Interview Readiness Checklist

Check off each item only when you can answer it confidently without notes.

---

## Python ✓

- [ ] Explain async/await and when it helps AI backends
- [ ] Write a retry decorator with exponential backoff from memory
- [ ] Explain dataclass vs Pydantic and when to use each
- [ ] Handle LLM API errors (rate limit, timeout, auth) properly
- [ ] Parse JSON from LLM output safely (strip code fences)
- [ ] Implement a simple token bucket rate limiter
- [ ] Use `asyncio.gather()` for concurrent LLM calls
- [ ] Load environment variables securely with dotenv

## Machine Learning ✓

- [ ] Explain bias-variance tradeoff with examples
- [ ] Choose the right evaluation metric for a given problem
- [ ] Prevent data leakage in train/val/test splits
- [ ] Use sklearn Pipeline to prevent leakage in CV
- [ ] Explain Random Forest vs Logistic Regression tradeoffs
- [ ] Implement cross-validation and interpret results
- [ ] Serialize and serve an sklearn model via FastAPI

## Deep Learning ✓

- [ ] Write the PyTorch training loop from memory (5 steps)
- [ ] Explain `model.train()` vs `model.eval()`
- [ ] Explain He initialization and why zeros fail
- [ ] Describe backpropagation and the chain rule
- [ ] Implement early stopping with checkpointing

## Transformers ✓

- [ ] Write the attention formula from memory
- [ ] Explain Q, K, V intuitively
- [ ] Explain why we divide by √d_k
- [ ] Implement a causal mask
- [ ] Explain encoder-only vs decoder-only models

## LLMs ✓

- [ ] Explain LLM hallucination at a technical level
- [ ] Explain temperature, top-k, top-p
- [ ] Calculate token costs for a given use case
- [ ] Implement streaming with OpenAI SDK
- [ ] Handle `finish_reason="length"` properly

## RAG ✓

- [ ] Draw the complete RAG pipeline from memory
- [ ] Explain hybrid search and implement BM25 + vector combination
- [ ] Explain two-stage retrieval (retrieve + rerank)
- [ ] Explain what metadata to store per chunk
- [ ] Implement content hash deduplication
- [ ] Explain context precision, recall, faithfulness
- [ ] Design a RAG evaluation pipeline

## Agents ✓

- [ ] Implement ReAct loop from scratch
- [ ] Use OpenAI native tool calling API
- [ ] Handle tool errors gracefully (feed error back to LLM)
- [ ] Implement MAX_ITERATIONS guard
- [ ] Explain indirect prompt injection

## FastAPI ✓

- [ ] Build a production AI endpoint with auth + rate limiting
- [ ] Implement Server-Sent Events streaming
- [ ] Use dependency injection for auth and rate limiting
- [ ] Write a global exception handler

## Docker ✓

- [ ] Write a production Dockerfile from memory
- [ ] Configure Docker Compose with health checks
- [ ] Pass environment variables securely
- [ ] Debug a container that won't start

## System Design ✓

- [ ] Design an enterprise RAG system (draw the architecture)
- [ ] Explain the complete query request flow (every step)
- [ ] Design the database schema (5 tables)
- [ ] Discuss LLM routing by complexity
- [ ] Explain trade-offs between pgvector and Qdrant

## LLMOps + Security ✓

- [ ] Describe what every LLM request log must contain
- [ ] Implement prompt injection detection
- [ ] Implement PII detection and redaction
- [ ] Design output guardrails
- [ ] Set up daily cost monitoring with alerts

---

## Interview Readiness Score

Count your checkmarks:
- **90-100%**: Excellent — apply immediately
- **75-89%**: Good — apply now, review weak areas
- **60-74%**: Close — 1-2 more weeks of focused review
- **Below 60%**: Revisit the relevant days in the program
