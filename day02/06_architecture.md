# Day 02 — Architecture
## Python Code Architecture for AI Systems

---

## Architecture 1: Python Module Structure for an AI Project

```
ai-project/
├── src/
│   ├── __init__.py
│   │
│   ├── llm/                    ← LLM interaction layer
│   │   ├── __init__.py
│   │   ├── client.py           ← LLMClient class
│   │   ├── prompts.py          ← Prompt templates
│   │   └── streaming.py        ← Streaming response handling
│   │
│   ├── rag/                    ← RAG pipeline
│   │   ├── __init__.py
│   │   ├── chunker.py          ← Text chunking strategies
│   │   ├── embedder.py         ← Embedding generation
│   │   ├── retriever.py        ← Vector search
│   │   └── pipeline.py         ← Full RAG pipeline
│   │
│   ├── agents/                 ← Agent system
│   │   ├── __init__.py
│   │   ├── base.py             ← Base agent class
│   │   ├── tools.py            ← Tool definitions
│   │   └── memory.py           ← Agent memory
│   │
│   ├── api/                    ← FastAPI application
│   │   ├── __init__.py
│   │   ├── routes.py           ← API endpoints
│   │   ├── models.py           ← Pydantic request/response models
│   │   └── middleware.py       ← Auth, rate limiting
│   │
│   └── utils/
│       ├── __init__.py
│       ├── tokens.py           ← Token counting
│       ├── logging.py          ← Logging setup
│       └── config.py           ← Configuration management
│
├── tests/
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_rag_pipeline.py
│
├── .env.example
├── requirements.txt
└── README.md

DATA FLOW:
User Request
    ↓
src/api/routes.py          (receive HTTP request)
    ↓
src/rag/pipeline.py        (orchestrate RAG)
    ├── src/rag/embedder.py    (embed query)
    ├── src/rag/retriever.py   (find chunks)
    └── src/llm/client.py      (generate answer)
    ↓
src/api/routes.py          (return HTTP response)
```

---

## Architecture 2: Async Request Handling

```
FastAPI Async Architecture:

HTTP Request
    ↓
[Uvicorn ASGI Server]
    ↓
[Event Loop]
    ↓
async def chat_endpoint(request):
    │
    ├── await validate_auth(request)     # 5ms (DB lookup)
    │         ↑ event loop runs others
    │
    ├── await embed_query(request.text)  # 200ms (API call)
    │         ↑ event loop runs others
    │
    ├── await search_vectors(embedding)  # 50ms (DB query)
    │         ↑ event loop runs others
    │
    └── await generate_response(...)     # 2000ms (LLM call)
              ↑ event loop runs others
    
Total: ~2255ms (all waits overlap with other requests)
100 concurrent requests: still ~2255ms (not 225 seconds)
```

---

## Architecture 3: Error Handling Flow

```
Request
    ↓
try:
    response = await call_llm(prompt)
except LLMRateLimitError as e:
    ─────────────────────────────────
    │ Wait e.retry_after seconds    │
    │ Retry up to 3 times          │
    │ If still failing → 429 error │
    ─────────────────────────────────
except LLMAuthError:
    ─────────────────────────────────
    │ Alert ops team                │
    │ Return 500 error              │
    │ Do NOT retry                  │
    ─────────────────────────────────
except LLMContextLengthError as e:
    ─────────────────────────────────
    │ Reduce context size           │
    │ Retry with shorter prompt     │
    ─────────────────────────────────
except Exception as e:
    ─────────────────────────────────
    │ Log full traceback            │
    │ Return 500 generic error      │
    │ Never expose internal error   │
    ─────────────────────────────────
```
