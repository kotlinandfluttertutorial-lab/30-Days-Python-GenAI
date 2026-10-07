# Day 17 — Day Summary: Embeddings

## What You Built
PROJECT 2: Semantic Search Engine — free local embeddings with `all-MiniLM-L6-v2`, batch indexing, semantic search, filtered search, and similar document finding.

## Key Takeaways
1. Embeddings encode meaning as geometry — similar meaning = close vectors
2. Always batch embed, never one-at-a-time for production systems
3. Lock your embedding model — changing it means re-embedding everything
4. Normalized vectors + inner product = cosine similarity, faster
5. Metadata filtering + semantic search = hybrid retrieval foundation

## Tomorrow: Day 18 — Vector Databases
ChromaDB, FAISS, pgvector — persistent vector storage for production RAG.
