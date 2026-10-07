# Day 29 — Architecture Diagrams

---

## Main System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE AI KNOWLEDGE ASSISTANT                          │
│                           System Architecture                                 │
└──────────────────────────────────────────────────────────────────────────────┘

 CLIENT                  API LAYER               AI ORCHESTRATION
 ──────                  ─────────               ────────────────

 Web App                 Nginx/CloudFront         FastAPI Backend
 Mobile App   ─HTTP──►  (SSL, routing,  ────────► (Auth, validate,
 API Client              rate limit)               orchestrate)
                                                       │
                                         ┌─────────────┴────────────┐
                                         │                          │
                                    RAG Pipeline              Agent System
                                    ───────────              ────────────
                                    Embed query              Plan task
                                    Hybrid search            Select tool
                                    Rerank                   Execute tool
                                    Build context            Observe
                                    Generate answer          Loop / Answer
                                         │                          │
                                         └─────────────┬────────────┘
                                                       │
                                               LLM GATEWAY
                                               ──────────
                                            Route by complexity
                                            ┌───────────────────┐
                                            │ Simple → Mini     │
                                            │ Complex → GPT-4o  │
                                            │ Private → Local   │
                                            └───────────────────┘

 STORAGE LAYER
 ─────────────
 ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  ┌──────────────┐
 │ PostgreSQL  │  │   Redis     │  │  Vector DB      │  │  File Store  │
 │             │  │             │  │  (pgvector)     │  │              │
 │ users       │  │ Query cache │  │                 │  │ Raw PDFs     │
 │ documents   │  │ Rate limits │  │ chunk embeddings│  │ Word docs    │
 │ convos      │  │ Sessions    │  │ + metadata      │  │ Web pages    │
 │ messages    │  │ Job queue   │  │                 │  │              │
 └─────────────┘  └─────────────┘  └─────────────────┘  └──────────────┘

 BACKGROUND WORKERS
 ──────────────────
 ┌──────────────────────────────────────────────────────┐
 │  Celery Worker                                        │
 │  document_ingestion_task():                          │
 │    Download → Clean → Chunk → Embed → Store → Update │
 └──────────────────────────────────────────────────────┘

 OBSERVABILITY
 ─────────────
 ┌──────────────────────────────────────────────────────┐
 │  Structured Logs → JSON → ELK / CloudWatch           │
 │  Metrics → Prometheus → Grafana dashboards           │
 │  Traces → OpenTelemetry → Jaeger                     │
 │  Eval Pipeline → Custom → Grafana eval dashboard     │
 │  Alerts → Cost budget, eval regression, error rate   │
 └──────────────────────────────────────────────────────┘
```

---

## Document Ingestion Flow

```
User uploads PDF
      │
      ▼
POST /api/v1/documents
  - Validate file type + size
  - Compute content_hash (SHA256)
  - Check for duplicate (same hash → skip)
  - Save metadata to PostgreSQL (status: "pending")
  - Enqueue ingestion task in Redis
  - Return document_id immediately (202 Accepted)
      │
      ▼ (Background)
Celery Worker picks up task
      │
      ├── Extract text (PyPDF2/docx2txt/beautifulsoup)
      ├── Clean text (normalize whitespace, remove noise)
      ├── Chunk (RecursiveCharacterTextSplitter 512 tokens, 50 overlap)
      ├── Embed batch (text-embedding-3-small, batches of 100)
      ├── Store chunks + embeddings in pgvector
      ├── Update document status → "complete"
      └── Send notification (webhook / websocket)
```

---

## Query Flow (RAG + Agent)

```
POST /api/v1/chat {"query": "What is our refund policy?"}
      │
      ├── 1. Authenticate JWT → user_id
      ├── 2. Rate limit (100/hour per user)
      ├── 3. PII scan input
      ├── 4. Cache lookup (Redis, key=hash(user_id + query))
      │
      │   ┌── CACHE HIT → return cached response ──────────────────────► Response
      │   │
      │   └── CACHE MISS ──────────────────────────────────────────────┐
      │                                                                 │
      │   5. Embed query (100ms)                                        │
      │   6. Hybrid search (BM25 + vector cosine, 50ms)                │
      │   7. Rerank top-20 → top-5 (200ms, cross-encoder)              │
      │   8. Build context from chunks                                  │
      │   9. Check user daily token budget                              │
      │  10. Build prompt (system + context + history + query)         │
      │  11. LLM streaming call (1000-3000ms)                          │
      │  12. Output guardrails (PII, toxicity, prompt leak)            │
      │  13. Cache result (TTL: 1 hour)                                │
      │  14. Save message to PostgreSQL                                │
      │  15. Structured log (trace_id, tokens, cost, latency)         │
      └────────────────────────────────────────────────────────────► Response + Sources
```

---

## Security Architecture

```
PERIMETER:
  HTTPS everywhere (TLS 1.3)
  API Gateway: DDoS protection, IP allowlist (optional)

AUTH:
  JWT tokens (short-lived, 1 hour)
  Refresh tokens (7 days)
  API keys for service-to-service

AUTHORIZATION:
  RBAC: admin, user roles
  Document-level: user can only access their own docs

INPUT SECURITY:
  PII detection → redact before external LLM
  Injection scan → block known patterns
  Input validation → Pydantic (type, length, format)

LLM SECURITY:
  Structural separation of system/user content
  Tool permission gates (sensitive tools → human approval)
  Tool result scanning (indirect injection)

OUTPUT SECURITY:
  Guardrails: PII, toxicity, length
  Never expose internal error messages
  Response logged (audit trail)

INFRASTRUCTURE:
  Non-root Docker containers
  Secrets in environment (not code)
  Encrypted volumes at rest
  Dependency pinning (no open ranges)
```
