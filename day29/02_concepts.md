# Day 29 — Concepts
## AI System Design + Enterprise AI Knowledge Assistant

---

## The Enterprise AI Knowledge Assistant — Full System Design

---

### Functional Requirements

```
1. Document Management
   - Upload documents (PDF, Word, TXT, HTML)
   - Asynchronous processing (background ingestion)
   - Track document status (pending/processing/complete/error)
   - Support up to 1 million documents

2. Retrieval & Q&A
   - Natural language queries over all documents
   - Hybrid search (BM25 + vector)
   - Reranking for precision
   - Source citations in every answer
   - Conversation history (multi-turn)

3. Agent Capabilities
   - Tool use: search, calculator, web (optional)
   - Multi-step reasoning
   - Human-in-the-loop for sensitive actions

4. Authentication & Authorization
   - User accounts with JWT tokens
   - Role-based access (admin, user)
   - Document-level permissions

5. Observability
   - Every LLM call logged
   - Cost per user per day
   - Eval scores tracked
   - Alerts on anomalies
```

---

### Non-Functional Requirements

```
Performance:
  - Query response: P95 < 3 seconds
  - Document ingestion: < 60 seconds per document
  - Uptime: 99.9%

Scale:
  - 1,000 concurrent users
  - 1 million documents
  - 10,000 queries per day

Security:
  - API key / JWT auth
  - PII detection and redaction
  - Prompt injection defense
  - Rate limiting: 100 queries/hour per user
  - Data encryption at rest and in transit

Cost:
  - Monthly LLM budget: configurable per team
  - Cache hit rate target: > 30%
```

---

### Complete Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE AI KNOWLEDGE ASSISTANT                 │
└─────────────────────────────────────────────────────────────────────┘

CLIENT (Web / Mobile / API)
         │
         │ HTTPS
         ▼
┌────────────────────┐
│   API GATEWAY      │  Rate limiting, SSL termination, routing
│   (Nginx/CloudFront)│
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   FastAPI Backend   │  Business logic, auth, orchestration
│   (Multiple replicas)│
│                    │
│  POST /documents   │  ← Upload document
│  GET  /documents   │  ← List documents
│  POST /chat        │  ← RAG + Agent query
│  GET  /chat/stream │  ← Streaming response
│  GET  /health      │
└────────┬───────────┘
         │
         │ Orchestrates:
         ├────────────────────────────────────────┐
         │                                        │
         ▼                                        ▼
┌─────────────────────┐             ┌─────────────────────┐
│   RAG PIPELINE       │             │   AGENT SYSTEM      │
│                     │             │                     │
│  1. Embed query     │             │  1. Plan             │
│  2. Hybrid search   │             │  2. Select tool      │
│  3. Rerank top-K    │             │  3. Execute tool     │
│  4. Build context   │             │  4. Observe result   │
│  5. Generate answer │             │  5. Repeat/conclude │
└──────┬──────────────┘             └──────┬──────────────┘
       │                                   │
       └──────────────┬────────────────────┘
                      │
                      ▼ LLM CALL
         ┌────────────────────────┐
         │   LLM GATEWAY          │
         │   (Provider routing)   │
         │                       │
         │   Primary: GPT-4o     │
         │   Fallback: Claude    │
         │   Budget model: Mini  │
         └────────────────────────┘

STORAGE LAYER:
┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  ┌──────────────┐
│  PostgreSQL  │  │    Redis     │  │   Vector DB       │  │  File Store  │
│              │  │              │  │   (pgvector or    │  │  (S3/local)  │
│  Users       │  │  Cache       │  │    Qdrant)        │  │  Raw docs    │
│  Documents   │  │  Rate limits │  │                   │  │              │
│  Conversations│  │  Sessions   │  │  Embeddings       │  │              │
│  Messages    │  │  Job queues  │  │  Metadata         │  │              │
└──────────────┘  └──────────────┘  └───────────────────┘  └──────────────┘

BACKGROUND WORKERS:
┌────────────────────────────────────────────────────────┐
│  Document Ingestion Worker (Celery / asyncio tasks)     │
│  Load → Clean → Chunk → Embed → Store → Update status │
└────────────────────────────────────────────────────────┘

OBSERVABILITY:
┌────────────────────────────────────────────────────────┐
│  Structured Logs → ELK Stack / CloudWatch              │
│  Metrics → Prometheus + Grafana                        │
│  Traces → OpenTelemetry                                │
│  Evals → Custom eval pipeline                          │
│  Alerts → PagerDuty / Slack                            │
└────────────────────────────────────────────────────────┘
```

---

### Database Schema

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW(),
    daily_token_budget INTEGER DEFAULT 100000
);

-- Documents
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    filename VARCHAR(255) NOT NULL,
    content_hash VARCHAR(64) UNIQUE NOT NULL,  -- Deduplication
    status VARCHAR(50) DEFAULT 'pending',      -- pending/processing/complete/error
    chunk_count INTEGER DEFAULT 0,
    file_size_bytes INTEGER,
    ingested_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Messages
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(50) NOT NULL,  -- user/assistant
    content TEXT NOT NULL,
    sources JSONB DEFAULT '[]',
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 8) DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    eval_score DECIMAL(4, 3),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector embeddings (pgvector)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    chunk_index INTEGER,
    metadata JSONB DEFAULT '{}'
);
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

---

### API Design

```
POST /api/v1/auth/login        → JWT token
POST /api/v1/auth/register     → Create account

POST /api/v1/documents         → Upload document (multipart/form-data)
GET  /api/v1/documents         → List user's documents
GET  /api/v1/documents/{id}    → Document + status
DELETE /api/v1/documents/{id}  → Delete document

POST /api/v1/chat              → RAG query (returns answer + sources)
GET  /api/v1/chat/stream       → Streaming query (SSE)

GET  /api/v1/conversations     → List conversations
GET  /api/v1/conversations/{id}/messages  → Conversation history

GET  /api/v1/admin/metrics     → Cost, tokens, requests (admin only)
GET  /api/v1/health            → Service health
```

---

### Request Flow (Query)

```
1. POST /api/v1/chat {"query": "What is our refund policy?"}
2. Authenticate JWT → get user_id
3. Rate limit check → 100 queries/hour per user
4. PII scan on query
5. Check Redis cache for this query (by user)
6. Cache miss → embed query (async, 100ms)
7. Hybrid search (BM25 + vector, 50ms)
8. Rerank top-20 → top-5 (200ms)
9. Build context from chunks
10. Check token budget
11. LLM call with streaming (1-3s)
12. Output guardrails scan
13. Cache result in Redis (TTL: 1 hour)
14. Save message to PostgreSQL
15. Log structured request record
16. Return response + sources + latency
```

---

### Trade-offs

```
1. LLM PROVIDER:
   GPT-4o vs Claude 3.5 vs Llama 3.1
   → Route by complexity: simple=mini, complex=4o, private=local llama

2. VECTOR DB:
   pgvector (simple infra) vs Qdrant (better scale/hybrid search)
   → Start with pgvector. Migrate if >5M chunks or need better hybrid.

3. CACHE STRATEGY:
   Cache by query hash (fast) vs semantic dedup (better hits)
   → Start with hash cache. Add semantic dedup when cache hit rate < 20%.

4. CHUNKING:
   Fixed (simple) vs recursive (better quality)
   → Recursive by default. Semantic chunking for very long docs.

5. SCALE:
   Single FastAPI → Docker Compose → Kubernetes
   → Start simple. Containerize from day 1. Scale when needed.
```
