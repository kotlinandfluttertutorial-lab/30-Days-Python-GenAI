# Day 01 — Architecture
## AI System Architectures You Must Know

---

## Architecture 1: The Minimal AI Application

The simplest possible AI application. Understand this before building anything complex.

```
┌──────────────────────────────────────────────────────────────┐
│                    MINIMAL AI APPLICATION                     │
└──────────────────────────────────────────────────────────────┘

User
 │
 │ HTTP POST /chat { "message": "Hello" }
 ▼
┌──────────────┐
│   FastAPI    │  ← Your application code
│   Endpoint   │
└──────┬───────┘
       │
       │ API Call: "Hello"
       ▼
┌──────────────┐
│   LLM API    │  ← OpenAI / Anthropic / Gemini
│  (External)  │
└──────┬───────┘
       │
       │ Response: "Hello! How can I help you?"
       ▼
┌──────────────┐
│   FastAPI    │
│   Returns    │
└──────┬───────┘
       │
       │ HTTP 200 { "response": "Hello! How can I help you?" }
       ▼
     User

COMPONENTS:
1. FastAPI — handles HTTP, validates input, returns output
2. LLM API — the intelligence (not yours, you call it)
3. Business logic — sits between the two

WHAT'S MISSING (that you'll add over 30 days):
- Context/history (the model forgets each call)
- Knowledge retrieval (the model doesn't know your data)
- Tool use (the model can't take actions)
- Authentication (anyone can call it)
- Rate limiting (someone could cost you thousands)
- Evaluation (you don't know if responses are good)
- Caching (same question costs money each time)
```

---

## Architecture 2: RAG System Architecture

The most important architecture for AI Engineers.

```
┌──────────────────────────────────────────────────────────────────┐
│                      RAG SYSTEM ARCHITECTURE                      │
└──────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════╗
║                  INGESTION PIPELINE                            ║
║                  (Offline / Background)                        ║
║                                                                ║
║  Documents                                                     ║
║  (PDF/Word/Web)                                                ║
║       │                                                        ║
║       ▼                                                        ║
║  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   ║
║  │  LOAD   │──→ │  CLEAN  │──→ │  CHUNK  │──→ │  EMBED  │   ║
║  │         │    │         │    │         │    │         │   ║
║  │ PyPDF   │    │ Remove  │    │ 512 tok │    │ Embed   │   ║
║  │ Docx    │    │ noise   │    │ overlap │    │ model   │   ║
║  │ HTML    │    │ Normalize│   │ sliding │    │ call    │   ║
║  └─────────┘    └─────────┘    └─────────┘    └────┬────┘   ║
║                                                      │        ║
║                                                      ▼        ║
║                                               ┌─────────┐    ║
║                                               │ VECTOR  │    ║
║                                               │  DB     │    ║
║                                               │ChromaDB │    ║
║                                               └─────────┘    ║
╚════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════╗
║                  RETRIEVAL + GENERATION                        ║
║                  (Online / Per Request)                        ║
║                                                                ║
║  User Query                                                    ║
║      │                                                         ║
║      ▼                                                         ║
║  ┌─────────┐    ┌─────────────┐    ┌─────────────────────┐   ║
║  │  EMBED  │──→ │  SIMILARITY │──→ │    TOP-K CHUNKS     │   ║
║  │  QUERY  │    │   SEARCH    │    │                     │   ║
║  │         │    │  Vector DB  │    │  Chunk 1: "Policy..." │  ║
║  └─────────┘    └─────────────┘    │  Chunk 2: "Terms..." │  ║
║                                    │  Chunk 3: "FAQ..."   │  ║
║                                    └──────────┬──────────┘   ║
║                                               │               ║
║                                    ┌──────────▼──────────┐   ║
║                                    │   BUILD PROMPT      │   ║
║                                    │                     │   ║
║                                    │ System: You are...  │   ║
║                                    │ Context: [chunks]   │   ║
║                                    │ User: [query]       │   ║
║                                    └──────────┬──────────┘   ║
║                                               │               ║
║                                    ┌──────────▼──────────┐   ║
║                                    │     LLM CALL        │   ║
║                                    │                     │   ║
║                                    │  OpenAI / Claude    │   ║
║                                    └──────────┬──────────┘   ║
║                                               │               ║
║                                    ┌──────────▼──────────┐   ║
║                                    │   RESPONSE +        │   ║
║                                    │   CITATIONS         │   ║
║                                    └─────────────────────┘   ║
╚════════════════════════════════════════════════════════════════╝
```

---

## Architecture 3: AI Agent Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      AGENT ARCHITECTURE                           │
└──────────────────────────────────────────────────────────────────┘

User Goal: "Find the cheapest flight to Paris next month"
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                        AGENT LOOP                               │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    LLM (Brain)                           │  │
│  │                                                          │  │
│  │  Input: User goal + Tool results + Memory               │  │
│  │  Output: Next action (tool call or final answer)        │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                        │                                         │
│           ┌────────────▼────────────┐                           │
│           │     DECIDE ACTION       │                           │
│           └────────────┬────────────┘                           │
│                        │                                         │
│        ┌───────────────┼───────────────┐                        │
│        ▼               ▼               ▼                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐            │
│  │  SEARCH  │   │  FETCH   │   │  FINAL ANSWER     │           │
│  │   TOOL   │   │  PRICES  │   │  (goal achieved)  │           │
│  └────┬─────┘   └────┬─────┘   └──────────────────┘            │
│       │              │                                           │
│       ▼              ▼                                           │
│  [Result 1]      [Result 2]                                     │
│       │              │                                           │
│       └──────┬────────┘                                         │
│              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    MEMORY                                 │  │
│  │                                                          │  │
│  │  Short-term: Current conversation                       │  │
│  │  Long-term: User preferences, past searches             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

TOOL TYPES:
┌────────────────┬────────────────────────────────────────┐
│  Tool          │  What it does                          │
├────────────────┼────────────────────────────────────────┤
│  Web Search    │  Retrieves current information          │
│  Calculator    │  Performs math                          │
│  Code Executor │  Runs Python code                       │
│  Database      │  Queries structured data               │
│  File System   │  Reads/writes files                     │
│  REST API      │  Calls external services               │
│  RAG           │  Retrieves from knowledge base         │
└────────────────┴────────────────────────────────────────┘
```

---

## Architecture 4: Production AI Backend

```
┌──────────────────────────────────────────────────────────────────┐
│                   PRODUCTION AI BACKEND                           │
└──────────────────────────────────────────────────────────────────┘

                        INTERNET
                            │
                            ▼
                   ┌────────────────┐
                   │  Load Balancer │
                   │  (Nginx/ALB)   │
                   └───────┬────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  FastAPI     │ │  FastAPI     │ │  FastAPI     │
    │  Instance 1  │ │  Instance 2  │ │  Instance 3  │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           └────────────────┼────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  PostgreSQL  │ │    Redis     │ │  Vector DB   │
    │  (Primary)   │ │   (Cache)    │ │  (ChromaDB)  │
    └──────────────┘ └──────────────┘ └──────────────┘
              │             │             │
              └─────────────┼─────────────┘
                            │
                   ┌────────────────┐
                   │  LLM Gateway   │
                   │  (Rate limit,  │
                   │   fallback)    │
                   └───────┬────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   OpenAI     │ │  Anthropic   │ │  Local LLM   │
    │   GPT-4o     │ │   Claude     │ │   (Fallback) │
    └──────────────┘ └──────────────┘ └──────────────┘
```

---

## Architecture 5: Data Flow in Embeddings + Vector Search

```
CREATING EMBEDDINGS:

Text: "Machine learning is a subset of AI"
         │
         ▼
┌──────────────────────┐
│   EMBEDDING MODEL    │
│                      │
│  Input: 7 words      │
│  Hidden layers: 12   │
│  Output: 1536 dims   │
└──────────────────────┘
         │
         ▼
Vector: [0.12, -0.34, 0.89, ..., 0.45]  ← 1536 numbers
         │
         ▼
┌──────────────────────┐
│    VECTOR DATABASE   │
│                      │
│  id: doc_001         │
│  vector: [...]       │
│  metadata: {         │
│    source: "ml.pdf", │
│    page: 12          │
│  }                   │
└──────────────────────┘


SEARCHING:

Query: "What is ML?"
         │
         ▼
┌──────────────────────┐
│   EMBEDDING MODEL    │
│   (same model!)      │
└──────────────────────┘
         │
         ▼
Query Vector: [0.10, -0.31, 0.87, ..., 0.42]
         │
         ▼
┌──────────────────────────────────────────────┐
│          VECTOR DATABASE                      │
│                                              │
│  Compare query vector to all stored vectors  │
│  Using cosine similarity or inner product    │
│                                              │
│  Rankings:                                   │
│  1. doc_001: similarity=0.95 ✓               │
│  2. doc_047: similarity=0.89 ✓               │
│  3. doc_023: similarity=0.87 ✓               │
│  ...                                         │
│  4000. doc_999: similarity=0.12 ✗            │
└──────────────────────────────────────────────┘
         │
         ▼
Return top-3 documents to RAG pipeline
```

---

## Architecture 6: LLMOps Monitoring Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     LLMOPS MONITORING                             │
└──────────────────────────────────────────────────────────────────┘

Every LLM Request:
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  Request In → ┌────────────────────────────────────────────┐    │
│               │         TRACE (spans)                       │    │
│               │                                             │    │
│               │  ┌──────────┐  ┌──────────┐  ┌─────────┐  │    │
│               │  │ Retrieve │→ │  Prompt  │→ │   LLM   │  │    │
│               │  │   span   │  │  build   │  │  call   │  │    │
│               │  │  120ms   │  │   5ms    │  │  890ms  │  │    │
│               │  └──────────┘  └──────────┘  └─────────┘  │    │
│               └────────────────────────────────────────────┘    │
│                                                                   │
│  Logged automatically:                                           │
│  - request_id: uuid                                              │
│  - timestamp: ISO 8601                                           │
│  - user_id: anonymized                                           │
│  - prompt_tokens: 1150                                           │
│  - completion_tokens: 150                                        │
│  - total_cost_usd: 0.008                                         │
│  - latency_ms: 1015                                              │
│  - model: gpt-4o                                                 │
│  - eval_score: 0.89                                              │
│  - hallucination_detected: false                                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

Stored in:
┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐
│  Structured    │  │  Metrics DB    │  │   Alert System         │
│  Logs          │  │  (Prometheus)  │  │                        │
│  (JSON)        │  │                │  │  if cost > $100/hour   │
│                │  │  p99 latency   │  │    → alert             │
│  ELK Stack     │  │  error rate    │  │  if eval_score < 0.7   │
│  CloudWatch    │  │  token usage   │  │    → alert             │
└────────────────┘  └────────────────┘  └────────────────────────┘
```

---

## Component Explanation: Every Piece of the Stack

### Why FastAPI?
- Async by design — handles concurrent LLM calls without blocking
- Pydantic integration — automatic request/response validation
- Auto-generated docs (OpenAPI/Swagger)
- Python — same language as ML ecosystem
- Fast — close to Node.js performance

### Why PostgreSQL?
- Store users, conversations, document metadata
- ACID transactions — never lose data
- pgvector extension — vector search in the same DB
- JSON support — flexible document storage
- Battle-tested at scale

### Why Redis?
- Cache LLM responses (same question, don't pay twice)
- Session storage for conversation history
- Rate limiting counters
- Task queue (background ingestion)
- Pub/Sub for streaming

### Why Vector Database?
- Relational DBs cannot efficiently do "find top-K most similar vectors"
- Built-in ANN algorithms (HNSW, IVF)
- Metadata filtering alongside vector search
- Scaled to billions of vectors

### Why Docker?
- Reproducible environment
- Dependency isolation
- Easy deployment to any cloud
- Docker Compose for local multi-service setup
- Kubernetes for production orchestration

### Why Multiple LLM Providers?
- Fallback: if OpenAI is down, use Anthropic
- Cost optimization: use cheaper model for simple tasks
- Capability routing: use best model for complex tasks
- Avoid vendor lock-in
