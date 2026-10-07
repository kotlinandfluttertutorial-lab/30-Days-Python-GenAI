# LinkedIn Project Descriptions

## Post Template: Project Announcement

---

### Post 1: RAG System

**I built a production RAG system from scratch. Here's what I learned:**

After 30 days of intensive AI engineering study, I built a complete Retrieval-Augmented Generation system.

What it does:
🔹 Ingests PDF, Word, and HTML documents
🔹 Chunks text with 512-token chunks and 50-token overlap
🔹 Embeds with sentence-transformers (free, local)
🔹 Stores in ChromaDB/pgvector
🔹 Retrieves with hybrid search (BM25 + vector)
🔹 Reranks with cross-encoder for precision
🔹 Generates grounded answers with source citations

The 3 things I wish I knew before building it:
1. Hybrid search (BM25 + vector) consistently beats pure vector search. BM25 catches exact keywords; vectors catch semantics. You need both.
2. The reranker doubles precision but adds 200ms. Worth it.
3. Never change your embedding model mid-project. Changing it means re-embedding everything.

Technologies: Python, ChromaDB, sentence-transformers, FastAPI, Redis

---

### Post 2: AI Agent

**Most people overcomplicate AI agents. Here's the simple truth:**

An agent is just:
LLM + Tools + MAX_ITERATIONS + Error handling

The loop:
1. LLM thinks about what to do
2. Calls a tool
3. Observes the result
4. Repeats until it has the answer

The critical parts nobody mentions:
❌ No MAX_ITERATIONS = infinite loop = $500 bill
❌ No tool error handling = agent breaks silently
❌ No injection defense = tool results can hijack your agent
❌ No tool call budget = expensive recursive tool calls

I built this using OpenAI's native function calling API — much more reliable than parsing JSON from free text.

---

### Post 3: Completed the 30-Day Program

**30 days. ~360 hours. Here's what I built:**

📊 ML Prediction Service — sklearn + FastAPI + Docker
🔍 Semantic Search Engine — embeddings + vector search
🤖 Enterprise RAG Assistant — full pipeline + RAGAS evaluation
🛠️ AI Research Agent — tool calling + MCP + security
🏗️ Enterprise AI System Design — 1M docs, full architecture

Skills acquired:
✅ RAG system design and implementation
✅ AI agents with tool calling
✅ Production FastAPI AI backends
✅ Docker + docker-compose deployment
✅ LLMOps: logging, metrics, cost monitoring
✅ AI security: PII, injection defense, guardrails
✅ Enterprise system design

If you're a software engineer wanting to transition into AI Engineering — this path works. Build things. Deploy them. Learn by doing.

#AIEngineering #GenAI #LLM #RAG #Python #MachineLearning

---

## LinkedIn Featured Section Descriptions

**Enterprise RAG Assistant**
A production-ready RAG system supporting 100K+ documents. Features: hybrid BM25+vector retrieval, cross-encoder reranking, faithfulness evaluation with RAGAS (score: 0.87), Redis caching (35% hit rate), async ingestion with deduplication, and full observability stack. Built with Python, FastAPI, ChromaDB, and PostgreSQL+pgvector.

**AI Research Agent**
Production AI agent using OpenAI function calling API. Implements the ReAct pattern with calculator, web search, and database tools. Production features: MAX_ITERATIONS guard, tool error handling, prompt injection defense, rate limiting, and comprehensive audit logging. Built with Python and FastAPI.

**Enterprise AI Knowledge Assistant**
System design for a complete multi-tenant AI Q&A platform supporting 1M+ documents. Architecture includes document-level access control, LLM routing by query complexity, hybrid RAG pipeline, and full observability. Built with Python, FastAPI, PostgreSQL+pgvector, Redis, and Docker Compose.
