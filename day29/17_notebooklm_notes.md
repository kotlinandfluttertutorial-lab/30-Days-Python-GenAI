# Day 29 — NotebookLM Notes: AI System Design

## System Design Framework

```
1. FUNCTIONAL REQUIREMENTS (what it does)
   - Document upload, ingestion, search, Q&A, conversations

2. NON-FUNCTIONAL REQUIREMENTS (how well it does it)
   - P95 latency < 3s, 99.9% uptime, 1M documents, 1K concurrent users

3. API DESIGN
   - RESTful, versioned (/api/v1/), JWT auth, pagination

4. DATA MODEL
   - users, documents, conversations, messages, document_chunks (pgvector)

5. COMPONENTS
   - FastAPI, PostgreSQL+pgvector, Redis, Background workers, LLM Gateway

6. TRADE-OFFS
   - Every choice has pros/cons; show you understand the trade-offs
```

## Key Architecture Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| LLM routing | By complexity | Save cost on simple queries |
| Vector DB | pgvector (start) | Minimal infra; migrate to Qdrant at scale |
| Cache | Redis by query hash | Fast, simple; add semantic dedup later |
| Chunking | Recursive char split | Better quality than fixed-size |
| Auth | JWT | Stateless, scalable |
| Background jobs | Celery/asyncio | Don't block HTTP on document ingestion |

## Request Flow (memorize this)

```
Request → Auth → Rate Limit → PII Scan → Cache Check →
[Cache Hit: return] 
[Cache Miss: Embed → Search → Rerank → Build Context → Token Check → LLM → Guardrails → Cache → Save → Log → Return]
```

## Database Schema (know these tables)

```sql
users(id, email, role, daily_token_budget)
documents(id, user_id, status, content_hash, chunk_count)
conversations(id, user_id, title)
messages(id, conversation_id, role, content, sources, cost_usd, eval_score)
document_chunks(id, document_id, content, embedding vector(1536), metadata)
```

## Interview Facts

1. Content hash deduplication: same document uploaded twice → skip re-ingestion
2. Async ingestion: document processing in background, don't block HTTP
3. LLM gateway: route cheap queries to mini models, expensive to full models
4. pgvector IVF index: must CREATE INDEX with ivfflat for fast search
5. Conversations are separate from messages — one conversation has many messages
6. daily_token_budget on user → enforce per-user cost limits
7. sources in messages → enables citation UI, audit trail

## Scaling Progression

```
MVP:          Single docker-compose (FastAPI + Postgres + Redis)
Growth:       Multiple FastAPI replicas behind load balancer
Scale:        Kubernetes, separate vector DB (Qdrant), CDN
Enterprise:   Multi-region, dedicated LLM gateway, custom eval pipeline
```
