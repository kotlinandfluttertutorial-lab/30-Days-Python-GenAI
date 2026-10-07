# Day 18 — Day Summary: Vector Databases

## What You Built
Document Search Engine using ChromaDB (with metadata filtering + CRUD) and FAISS (high-performance in-process search). Side-by-side comparison of all major vector databases.

## Key Takeaways
1. Vector DBs use ANN algorithms (HNSW, IVF) for sub-linear similarity search
2. ChromaDB: easiest for RAG prototyping; FAISS: fastest for production performance
3. pgvector: best choice if you already run PostgreSQL
4. Always store metadata at indexing time — you can't add it later easily
5. HNSW: fast search, high memory; IVF: less memory, needs training

## Phase 6 Half-Complete (Days 16-18)
GenAI engineering foundation: prompt engineering, embeddings, vector databases.

## Tomorrow: Day 19 — RAG Fundamentals (Project 3 begins)
The complete RAG pipeline. The most important skill for AI Engineering roles.
