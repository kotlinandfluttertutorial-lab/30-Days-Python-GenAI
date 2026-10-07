# Resume Project Descriptions

## Usage Instructions

Pick the projects most relevant to the role. Use bullet points with action verbs and metrics. Customize technology stack to match job requirements.

---

## Project 1: ML Prediction Service

**Title:** Machine Learning Prediction Service  
**Tech:** Python, scikit-learn, FastAPI, Docker, Pydantic

- Built end-to-end ML pipeline: data loading → preprocessing → model training → evaluation → serialization → REST API → Docker deployment
- Compared 3 algorithms (Logistic Regression, Random Forest, Gradient Boosting) via 5-fold cross-validation; selected model achieved F1=0.89
- Implemented FastAPI prediction service with Pydantic input validation, batch prediction endpoint, and /health + /model/info endpoints
- Containerized with Docker and docker-compose; model loaded at startup (not per-request) for optimal latency

---

## Project 2: Semantic Search Engine

**Title:** Semantic Search Engine with Embeddings  
**Tech:** Python, sentence-transformers, NumPy, FastAPI

- Built semantic search system using free local embedding model (all-MiniLM-L6-v2, 384 dims)
- Implemented batch embedding with rate limiting, normalized vector storage, cosine similarity search in <10ms for 10K documents
- Added metadata filtering to combine semantic search with structured attribute filtering
- Demonstrated 3× improvement in recall over keyword search on semantically similar queries

---

## Project 3: Enterprise RAG System

**Title:** Production RAG Q&A Assistant  
**Tech:** Python, ChromaDB, pgvector, sentence-transformers, FastAPI, Redis

- Built complete RAG pipeline: PDF/Word/HTML ingestion, recursive chunking (512 tokens, 50 overlap), batch embedding, ChromaDB/pgvector storage
- Implemented hybrid retrieval (BM25 + vector cosine similarity, α=0.5) with cross-encoder reranking for top-K precision
- Achieved faithfulness score >0.87 and answer relevance >0.82 via RAGAS evaluation framework on 100-question test set
- Production features: Redis caching (35% cache hit rate), async ingestion with deduplication, exponential backoff retry, structured logging with trace IDs

---

## Project 4: AI Research Agent

**Title:** Tool-Using AI Research Agent  
**Tech:** Python, OpenAI API, Pydantic, FastAPI

- Built production AI agent using OpenAI native function calling with tools: web search, calculator, knowledge base, database query
- Implemented ReAct loop with MAX_ITERATIONS guard (prevents infinite loops), tool error handling, and graceful degradation
- Deployed prompt injection defense: regex-based input scanning + structural separation of system/user content
- Reduced tool selection errors by 40% through detailed tool schema descriptions and example outputs

---

## Project 5: Enterprise AI Knowledge Assistant (Architecture)

**Title:** Enterprise AI Knowledge Assistant — System Design  
**Tech:** FastAPI, PostgreSQL + pgvector, Redis, Docker Compose, OpenAI API

- Designed complete multi-tenant AI Q&A system for 1M+ documents with document-level access control
- Architecture: FastAPI backend → hybrid RAG pipeline → LLM routing (complexity-based) → PostgreSQL + pgvector + Redis
- Database schema: 5 tables (users, documents, conversations, messages, document_chunks with vector(1536))
- Implemented: JWT auth, token bucket rate limiting (100/hour), PII detection, output guardrails, daily cost budget alerts
- Full Docker Compose deployment with health checks; all services connected via bridge network

---

## Shorter Descriptions (for limited resume space)

### One-liner versions:
- "Built production RAG system with hybrid search, reranking, evaluation framework, and Redis caching — achieves P95 query latency <3s on 100K documents"
- "Implemented AI agent with OpenAI function calling, prompt injection defense, tool error handling, and audit logging"
- "Designed enterprise AI knowledge assistant: FastAPI + PostgreSQL + pgvector + Redis, serving 1M documents with document-level access control"
