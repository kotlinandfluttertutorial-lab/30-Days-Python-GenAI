# AI Engineer Introduction Scripts

---

## 30-Second Introduction (elevator pitch)

"I'm a software engineer with [X] years of experience, transitioning into AI engineering. I've spent the last month building production AI systems — a RAG-based knowledge assistant, an AI agent with tool calling, and a full FastAPI AI backend — all containerized with Docker. I'm excited about roles where I can build things like RAG pipelines, LLM-powered APIs, and AI agents that actually ship to users."

---

## 60-Second Introduction (phone screen opener)

"I'm a software engineer with [X] years of experience in [your background]. I recently completed an intensive AI engineering program where I built production-quality AI systems from scratch. Specifically, I built a complete RAG pipeline handling 100K+ documents, an AI agent using OpenAI's function calling API, a FastAPI backend with streaming responses and JWT authentication, and containerized everything with Docker Compose. I understand the full stack: from the mathematics of embeddings and attention, to building evaluation frameworks, to production concerns like LLMOps, prompt injection defense, and cost optimization. I'm looking to bring that hands-on production experience to a team building serious AI products."

---

## 2-Minute Introduction (technical interview)

"I'm [name], a software engineer with [X] years of experience [briefly: what you built before].

I transitioned into AI engineering through an intensive self-directed program focused on building, not just learning. I'll walk you through the key things I built:

First, I built a **complete RAG system** — ingestion pipeline that handles PDF/Word/HTML, chunks with overlap, embeds with sentence-transformers, stores in ChromaDB and pgvector. The query pipeline uses hybrid search (BM25 plus vector), cross-encoder reranking, and generates cited answers. I built a custom evaluation framework measuring faithfulness and relevance.

Second, I built **AI agents** using OpenAI's native tool calling API — tools for calculator, web search, and database queries. I implemented the full production concerns: MAX_ITERATIONS guard, tool error handling, prompt injection defense, and audit logging.

Third, I built a **production FastAPI AI backend** with JWT authentication, token bucket rate limiting, streaming SSE responses, Pydantic validation, and structured logging of every LLM call with trace IDs, token counts, and cost.

Everything is containerized with Docker and docker-compose with PostgreSQL, Redis, and the AI API running as services.

I can design systems, implement them in Python, and explain every architectural decision from the math through the production concerns. I'm ready to contribute from day one."

---

## 5-Minute Introduction (detailed technical discussion)

[Use the 2-minute version as the opener, then continue:]

"Let me go deeper on the technical decisions I made and why.

**On RAG architecture:** I chose hybrid search because pure vector search misses exact keyword matches — if someone asks about 'GPT-4' and the document says 'OpenAI's model', vector search scores it lower than expected. BM25 catches exact matches; vector search catches semantic similarity. Combining them with normalized scores and a 0.5 weight consistently outperforms either alone. I added a cross-encoder reranker for the top-20 candidates, which improved precision significantly — the cost is ~200ms additional latency, which is acceptable when total query time is 2-3 seconds.

**On embeddings:** I used `all-MiniLM-L6-v2` for development (free, 384 dimensions, fast) and `text-embedding-3-small` for production (1536 dimensions, better quality). Critical lesson: you must lock the embedding model — changing it requires re-embedding all documents. I implemented content hash deduplication so the same document uploaded twice doesn't get re-ingested.

**On agents:** I found that agents are significantly overused. For 80% of Q&A use cases, a well-designed RAG pipeline is faster, cheaper, and more reliable. I use agents only when the path to the answer is genuinely unknown at design time — like researching a topic that requires combining web search, calculation, and database lookups dynamically.

**On production concerns:** I log every LLM call with trace_id, latency, token counts, and cost_usd. I track p95 latency rather than average — tail latency is what affects user experience. I implemented daily cost budget alerts at 80% of limit, not 100%, to have time to investigate before overspending.

**What I want to build next:** [mention specific company projects you've researched]."

---

## Resume Project Descriptions

### Project 1: ML Prediction Service
> Built an end-to-end machine learning prediction service using scikit-learn Random Forest, served via FastAPI with Pydantic validation. Implemented model serialization with joblib, batch prediction endpoint, health checks, and Docker deployment. Evaluated multiple algorithms using 5-fold cross-validation and selected based on F1 score.

### Project 2: Semantic Search Engine
> Built a production semantic search engine using sentence-transformers (all-MiniLM-L6-v2) for free local embeddings. Implemented batch embedding with rate limiting, cosine similarity search, metadata filtering, and similar-document discovery. Achieved sub-10ms search latency across 10,000 documents.

### Project 3: Enterprise RAG Assistant
> Built a complete Retrieval-Augmented Generation system: document ingestion pipeline (load, clean, chunk, embed), hybrid BM25 + vector search, cross-encoder reranking, context-grounded generation with citations, RAGAS-based evaluation (faithfulness, relevance, precision). Implemented Redis caching, async processing, retry with exponential backoff, and structured logging.

### Project 4: AI Research Agent
> Built a production AI agent using OpenAI's native function calling API with tools: web search, calculator, knowledge base, file reader. Implemented ReAct loop with MAX_ITERATIONS guard, tool error handling, prompt injection defense (input scanning + structural separation), audit logging, and rate limiting per agent run.

### Project 5: Enterprise AI Knowledge Assistant (System Design)
> Designed a complete enterprise AI Q&A system: FastAPI backend with JWT auth and rate limiting, PostgreSQL + pgvector database, Redis cache, async document ingestion workers, LLM routing by query complexity, output guardrails (PII, toxicity), structured observability, and Docker Compose deployment. Full system design with functional requirements, database schema, API design, and trade-off analysis.
