# Day 02 — Prerequisites and Context

---

## What You Need from Day 1

- [ ] Virtual environment created and activated
- [ ] `.env` file created with at least one API key (or Ollama running)
- [ ] Mental model of the AI engineering stack
- [ ] Understanding that LLM calls are I/O-bound (they wait for a network response)

## Python Background Assumed

You need basic programming experience:
- Functions, loops, conditionals
- What a variable is
- What a class is (conceptually)

You do NOT need advanced Python experience — this day covers everything.

---

## Context: Why These Patterns Specifically?

Every Python pattern taught today maps directly to AI engineering use cases.

| Pattern | AI Use Case | First Appears in Program |
|---------|------------|--------------------------|
| Type hints | FastAPI, Pydantic validation | Day 26 (FastAPI) |
| Dataclasses | LLM message objects | Day 2+ (used everywhere) |
| async/await | Concurrent LLM calls | Day 26 (FastAPI) |
| Generators | Streaming responses | Day 26 (SSE streaming) |
| Exception hierarchy | LLM retry logic | Day 21 (production RAG) |
| JSON handling | LLM structured output | Day 16 (prompt engineering) |
| Comprehensions | Batch processing | Day 17 (embeddings) |
| Decorators | Retry, caching | Day 21 (production RAG) |
| Context managers | DB connections | Day 26 (FastAPI + DB) |

None of these are academic. Every one appears in production AI code.
